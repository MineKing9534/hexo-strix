"""Fast graph packing for KLENT inference and fitting."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from hexo_a0.graph import (
    axis_states_to_batch,
    graph_batch_fn_from_model_config,
)
from hexo_a0.model import legacy_edges_to_lean, legacy_lean_columns
from hexo_axis_models import decode_hxr1
from hexo_klent.model import is_dense_axis_config

_RASTER_BUCKETS = (9, 17, 25, 33, 41, 49, 65, 81, 97, 129)


def raster_shape(state: object) -> tuple[int, int]:
    """Return the exact default Rust crop bucket for one non-terminal state."""

    coords = [
        coord for coord, _player in state.placed_stones()
    ]
    coords.extend(state.legal_moves())
    if not coords:
        raise ValueError("cannot determine raster shape for an empty state")
    q_values, r_values = zip(*coords, strict=True)
    span_q = max(q_values) - min(q_values) + 1
    span_r = max(r_values) - min(r_values) + 1
    for bucket in _RASTER_BUCKETS:
        if bucket >= span_q and bucket >= span_r:
            return bucket, bucket

    def round_bucket(span: int) -> int:
        return 1 if span <= 1 else 1 + 8 * ((span - 1 + 7) // 8)

    return round_bucket(span_q), round_bucket(span_r)


def order_states_for_batching(
    states: list[object],
    model_config,
) -> tuple[list[object], list[int]]:
    """Group dense crop buckets and retain the original state indices."""

    if not is_dense_axis_config(model_config):
        return states, list(range(len(states)))
    source_indices = sorted(
        range(len(states)),
        key=lambda index: raster_shape(states[index]),
    )
    return [states[index] for index in source_indices], source_indices


def restore_state_order(
    ordered_items: list[object],
    source_indices: list[int],
) -> list[object]:
    """Undo ``order_states_for_batching`` for per-state model outputs."""

    if len(ordered_items) != len(source_indices):
        raise ValueError("ordered items and source indices must have equal length")
    restored: list[object | None] = [None] * len(source_indices)
    for source_index, item in zip(
        source_indices, ordered_items, strict=True
    ):
        restored[source_index] = item
    if any(item is None for item in restored):
        raise ValueError("source indices are not a complete permutation")
    return list(restored)


def _packed_ranges(
    edge_counts: list[int],
    edge_budget: int,
) -> list[tuple[int, int]]:
    """Return contiguous ranges with the same greedy edge-budget semantics."""

    if not edge_counts:
        return []
    if edge_budget <= 0:
        return [(0, len(edge_counts))]

    ranges: list[tuple[int, int]] = []
    start = 0
    packed_edges = 0
    for index, edges in enumerate(edge_counts):
        if index > start and packed_edges + edges > edge_budget:
            ranges.append((start, index))
            start = index
            packed_edges = 0
        packed_edges += edges
    ranges.append((start, len(edge_counts)))
    return ranges


def _supports_collated_axis(model_config) -> bool:
    """Whether the legacy Rust batch can reproduce this model's node schema."""

    if getattr(model_config, "axis_relational", False):
        return True
    return (
        not getattr(model_config, "compact_stone_onehot", False)
        and getattr(model_config, "node_coords", True)
        and getattr(model_config, "moves_scope", "node") == "node"
    )


def _native_axis_batch(
    *,
    x,
    edge_index,
    edge_attr,
    legal_mask,
    batch_index,
    num_graphs: int,
    model_config,
):
    """Convert one collated legacy axis slice to the native model schema."""

    # Resolve dynamic legal-node indices while the tensors are still on CPU.
    # Calling ``nonzero`` after the batch reaches ROCm synchronizes the device
    # so PyTorch can discover the data-dependent output size.  Legal ordering
    # is already fixed by the Rust graph builder, so these indices and counts
    # are durable batch metadata rather than model work.
    legal_idx = legal_mask.nonzero(as_tuple=False).squeeze(1)
    legal_counts = torch.bincount(
        batch_index.index_select(0, legal_idx),
        minlength=num_graphs,
    )
    common = dict(
        legal_mask=legal_mask,
        legal_idx=legal_idx,
        legal_counts=legal_counts,
        batch=batch_index,
    )
    if getattr(model_config, "axis_relational", False):
        columns = legacy_lean_columns(model_config)
        lean_x = x if columns is None else x[:, columns]
        edge_index, edge_type, edge_dist, global_edge_index = (
            legacy_edges_to_lean(edge_index, edge_attr)
        )
        result = Batch(
            x=lean_x,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_dist=edge_dist,
            global_edge_index=global_edge_index,
            **common,
        )
    else:
        result = Batch(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            **common,
        )
    # Match Batch.from_data_list's static graph count. Without this attribute,
    # PyG derives it through Tensor.item(), breaking the compiled core.
    result._num_graphs = num_graphs
    return result


def _axis_batches(
    states: list[object],
    *,
    model_config,
    edge_budget: int,
) -> list[tuple[Batch, slice]]:
    """Build once in Rust, then slice at exact per-graph edge boundaries."""

    batch, _aux = axis_states_to_batch(
        states,
        prune_empty_edges=model_config.prune_empty_edges,
        threat_features=getattr(model_config, "threat_features", False),
        relative_stones=getattr(
            model_config, "relative_stone_encoding", False
        ),
    )
    if edge_budget <= 0:
        return [
            (
                _native_axis_batch(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    legal_mask=batch.legal_mask,
                    batch_index=batch.batch,
                    num_graphs=batch.num_graphs,
                    model_config=model_config,
                ),
                slice(0, len(states)),
            )
        ]

    node_counts = torch.bincount(
        batch.batch, minlength=batch.num_graphs
    )
    edge_graphs = batch.batch.index_select(0, batch.edge_index[0])
    edge_counts = torch.bincount(
        edge_graphs, minlength=batch.num_graphs
    )
    node_offsets = [0, *node_counts.cumsum(0).tolist()]
    edge_offsets = [0, *edge_counts.cumsum(0).tolist()]

    prepared = []
    for start, end in _packed_ranges(edge_counts.tolist(), edge_budget):
        node_start, node_end = node_offsets[start], node_offsets[end]
        edge_start, edge_end = edge_offsets[start], edge_offsets[end]
        prepared.append(
            (
                _native_axis_batch(
                    x=batch.x[node_start:node_end],
                    edge_index=(
                        batch.edge_index[:, edge_start:edge_end] - node_start
                    ),
                    edge_attr=batch.edge_attr[edge_start:edge_end],
                    legal_mask=batch.legal_mask[node_start:node_end],
                    batch_index=batch.batch[node_start:node_end] - start,
                    num_graphs=end - start,
                    model_config=model_config,
                ),
                slice(start, end),
            )
        )
    return prepared


def _raster_batches(
    states: list[object],
    *,
    model_config,
    cell_budget: int,
) -> list[tuple[object, slice]]:
    """Build contiguous same-shape dense batches within a compute-cell budget."""

    import hexo_rs

    encoded_batches = hexo_rs.game_states_to_raster_batches_hxr1(
        states,
        int(getattr(model_config, "dense_ray_radius", 5)),
        bool(model_config.prune_empty_edges),
        max(0, cell_budget),
    )
    return [
        (
            decode_hxr1(encoded),
            slice(start, end),
        )
        for encoded, start, end in encoded_batches
    ]


def prepare_graph_batches(
    states: list[object],
    *,
    model_config,
    edge_budget: int,
) -> list[tuple[object, slice]]:
    """Build packed graph batches and their contiguous source-state slices.

    Axis graphs use the Rust pre-collated byte path, converted once to the
    native lean tensors expected by the representation. This avoids constructing
    and tensorising one Python/PyG object per position. When an edge budget is
    active, the collated edge ownership recovers the exact old packing
    boundaries, and lightweight tensor slices form the selected microbatches
    without rebuilding any graph.
    """

    if not states:
        return []

    if is_dense_axis_config(model_config):
        # ``edge_budget`` is the established KLENT microbatch budget knob. For
        # a dense backend its equivalent unit is padded raster cells.
        return _raster_batches(
            states,
            model_config=model_config,
            cell_budget=edge_budget,
        )

    if (
        model_config.graph_type == "axis"
        and _supports_collated_axis(model_config)
    ):
        return _axis_batches(
            states,
            model_config=model_config,
            edge_budget=edge_budget,
        )

    graph_batch_fn = graph_batch_fn_from_model_config(model_config)
    graphs = graph_batch_fn(states)
    edge_counts = []
    for graph in graphs:
        edges = int(graph.edge_index.shape[1])
        global_edges = getattr(graph, "global_edge_index", None)
        if global_edges is not None:
            edges += int(global_edges.shape[1])
        edge_counts.append(edges)
    return [
        (
            Batch.from_data_list(graphs[start:end]),
            slice(start, end),
        )
        for start, end in _packed_ranges(edge_counts, edge_budget)
    ]


def move_batch_to_device(batch, device: torch.device):
    """Move a prepared graph or raster batch to a device."""

    return batch.to(device)
