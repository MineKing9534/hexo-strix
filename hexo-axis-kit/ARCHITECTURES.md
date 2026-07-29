# Recommended model architectures

## 1. `AxisGineCompatNet`

This is the first model to test. It compiles the current axis-relational graph algebra into fixed-shape raster operations:

1. Normalize each active cell with per-cell channel LayerNorm.
2. For each of the three undirected winning axes, gather up to five sources in both directions using the fused line kernel (or the portable shift/mask reference on CPU).
3. Apply the exact packed ray mask before aggregation.
4. Add learned unsigned-distance embeddings before the message ReLU.
5. Apply one shared GINE MLP to all three axis relations at active cells only.
6. Sum the three axis outputs.
7. Maintain an explicit global state which reproduces the graph dummy node's bidirectional star relation.
8. Apply the same node-update MLP and outer residual/ReLU as the current representation network.
9. Preserve JK-cat, policy, Q, distributional value, and horizon heads.

The compact API projects active input cells once and keeps the complete block
stack and JK representation compact. KLENT gathers legal embeddings directly
from that representation. The separate dense-map API remains the readable
reference and direct-inference path; there is no compact-to-dense compatibility
wrapper.

The included checkpoint converter maps the current modules into their dense equivalents. Floating-point reduction order changes, so use close numerical parity rather than bit parity as the acceptance test.

Recommended first acceptance gates:

- converted checkpoint policy top-1 agreement above 99.9% on a fixed corpus;
- mean policy KL below 1e-4 in FP32, then characterize BF16 drift;
- identical solver tactical-suite outcomes;
- materially higher states/second at the same batch cells.

If parity is not sufficiently close, compare each block's real-cell and global-token outputs in FP32. The representation was deliberately divided at the same module boundaries to make this straightforward.

## 2. `PersistentRayAxisNet`

This keeps the full compatibility backbone and adds a narrow six-direction stream after selected layers.

For each directed ray it retains persistent latent state and updates it from:

- the previous state for that ray;
- the new blocker-aware directional line message;
- the scalar trunk at the same cell;
- a D6-equivariant orientation-ring mixture.

The orientation-ring update has four tied relative-direction classes:

```text
same
adjacent directions (+/- 60 degrees)
next-nearest directions (+/- 120 degrees)
opposite direction (180 degrees)
```

Opposite rays are then paired into three undirected axes using:

```text
sum
elementwise product
absolute difference
```

The three axes are folded invariantly with:

```text
sum across axes
pairwise product across axes
max across axes
```

The pairwise-product statistic provides a direct soft fork feature. All ray and axis slots use shared weights, so rotations permute slots and reflections reverse them without changing the scalar output.

The final fold projection is zero-initialized by default. Loading a converted compatibility checkpoint with `strict=False` therefore starts with exactly the base function in the supplied tests. The new stream begins affecting the trunk after its final projection takes its first optimizer updates.

Recommended training sequence:

1. Convert and validate `AxisGineCompatNet`.
2. Load that state into `PersistentRayAxisNet` with `strict=False`.
3. Apply a short learning-rate warmup.
4. Initially train on the current search corpus and completed-Q labels.
5. Move to on-policy KLENT after parity and stability are established.

## Inference outputs

Both models return dense maps:

```text
policy_logits [B,H,W]
q_values      [B,H,W]
value         [B]
value_logits  [B,bins]
```

Illegal policy cells are set to the dtype minimum and illegal Q cells to zero. `legal_flat_indices` gathers legal results in the engine's sorted legal-move order.

For KLENT, compute the improved legal policy in FP32:

```python
u = (q + beta * logits) / (alpha + beta)
pi_k = softmax(u)
v_k = (pi_k * q).sum()
```

## Performance path

CUDA/ROCm execution uses a custom Triton active-cell gather which performs:

```text
gather + distance projection + ReLU + masked axis accumulation
```

without materializing every shifted or padded-axis tensor. It reads Rust's
packed ray words directly, returns `[active,3,channels]`, and uses a
source-centric custom backward. Normalization, the GINE and global MLPs,
real-to-global reduction, and the node update remain in active-cell order
throughout the compact API.
CPU execution reconstructs the readable dense reference for portability and
parity testing.

The persistent-ray side stream has its own fused forward/backward gather. It
consumes compact projected sources and emits six directed messages directly
in active-cell order instead of scattering a source raster or materializing
thirty shifted `[B,C,H,W]` tensors. Its normalization, pointwise projections,
gated update, orientation-ring mixture, and invariant fold also run only on
active cells in the compact API. The padded implementation remains available
as a readable CPU/reference path.

## KLENT helper

`hexo_axis_models.klent` includes:

- segmented policy improvement over variable legal-action counts;
- vectorized one-action-per-state Gumbel sampling;
- `V_K = sum(pi_K * Q)` computation;
- actor-aware lambda returns which preserve the sign between the first and second placement of a turn and negate only when control passes.
