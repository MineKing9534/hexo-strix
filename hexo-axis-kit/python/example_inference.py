#!/usr/bin/env python3
"""Minimal HXR1 -> dense Axis-GINE inference example."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hexo_axis_models import AxisGineCompatNet, AxisGineConfig, decode_hxr1
from hexo_axis_models.checkpoint import load_strix_axis_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", type=Path, help="HXR1 request emitted by RasterBatch::encode_hxr1")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    batch = decode_hxr1(args.request.read_bytes()).to(device)
    model = AxisGineCompatNet(AxisGineConfig())
    if args.checkpoint:
        report = load_strix_axis_checkpoint(model, args.checkpoint)
        print(
            f"converted {len(report.copied)} tensors; "
            f"shape mismatches={len(report.shape_mismatches)}"
        )
    model = model.to(device=device, dtype=dtype).eval()
    if args.compile:
        model = torch.compile(model, fullgraph=True)

    planes = batch.planes.to(dtype=dtype)
    scalars = batch.scalars.to(dtype=dtype)
    with torch.inference_mode():
        output = model.forward_packed(planes, scalars, batch.active_mask, batch.ray_bits)

    policy = output.policy_logits.reshape(-1).index_select(0, batch.legal_flat_indices)
    q_values = output.q_values.reshape(-1).index_select(0, batch.legal_flat_indices)
    print("legal logits:", tuple(policy.shape))
    print("legal Q:", tuple(q_values.shape))
    print("values:", output.value.float().cpu().tolist())
    print("legal offsets:", batch.legal_offsets.cpu().tolist())


if __name__ == "__main__":
    main()
