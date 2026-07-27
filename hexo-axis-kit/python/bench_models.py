#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import time

import torch

from hexo_axis_models.model import (
    AxisGineCompatNet,
    AxisGineConfig,
    PersistentRayAxisNet,
    PersistentRayConfig,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(batch: int, size: int, radius: int, device: torch.device, dtype: torch.dtype):
    planes = torch.zeros(batch, 8, size, size, device=device, dtype=dtype)
    center = size // 2
    planes[:, 0, center, center] = 1
    planes[:, 1, center, center + 1] = 1
    planes[:, 2, :, :] = (torch.rand(batch, size, size, device=device) < 0.35).to(dtype)
    planes[:, 2, center, center] = 0
    planes[:, 2, center, center + 1] = 0
    planes[:, 3:] = torch.rand(batch, 5, size, size, device=device, dtype=dtype)
    active = (planes[:, 0:1] + planes[:, 1:2] + planes[:, 2:3]) > 0
    scalars = torch.rand(batch, 5, device=device, dtype=dtype)
    ray_mask = torch.rand(batch, 6, radius, size, size, device=device) < 0.12
    ray_mask &= active.unsqueeze(1)
    return planes, scalars, active, ray_mask


def benchmark(model, inputs, warmup: int, iterations: int, training: bool) -> float:
    model.train(training)
    if training:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(warmup):
        if training:
            optimizer.zero_grad(set_to_none=True)
            out = model(*inputs)
            legal = inputs[0][:, 2].bool()
            loss = out.policy_logits[legal].float().mean() + out.q_values[legal].float().mean() + out.value.float().mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                model(*inputs)
    synchronize(inputs[0].device)
    start = time.perf_counter()
    for _ in range(iterations):
        if training:
            optimizer.zero_grad(set_to_none=True)
            out = model(*inputs)
            legal = inputs[0][:, 2].bool()
            loss = out.policy_logits[legal].float().mean() + out.q_values[legal].float().mean() + out.value.float().mean()
            loss.backward()
            optimizer.step()
        else:
            with torch.inference_mode():
                model(*inputs)
    synchronize(inputs[0].device)
    return (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--sizes", default="17,25,33,49")
    parser.add_argument("--batches", default="16,32,64,128")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--csv", default="axis_models_bench.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    sizes = [int(x) for x in args.sizes.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    configs = [
        ("axis_gine_4x128", AxisGineCompatNet, AxisGineConfig()),
        ("persistent_ray_4x128", PersistentRayAxisNet, PersistentRayConfig()),
    ]
    rows = []
    for name, model_type, config in configs:
        for size in sizes:
            for batch in batches:
                model = model_type(config).to(device=device, dtype=dtype)
                if args.compile:
                    model = torch.compile(model, fullgraph=True)
                inputs = make_inputs(batch, size, config.line_radius, device, dtype)
                seconds = benchmark(model, inputs, args.warmup, args.iterations, args.training)
                row = {
                    "model": name,
                    "size": size,
                    "batch": batch,
                    "mode": "train" if args.training else "infer",
                    "seconds_per_step": seconds,
                    "states_per_second": batch / seconds,
                    "dtype": args.dtype,
                    "compiled": args.compile,
                    **{f"cfg_{k}": v for k, v in asdict(config).items() if not isinstance(v, tuple)},
                }
                print(row)
                rows.append(row)
                del model, inputs
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    fieldnames = sorted({key for row in rows for key in row})
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
