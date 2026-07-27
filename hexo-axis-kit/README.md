# HeXO dense axis-line kit

This package contains a concrete board representation and two model architectures designed to retain Strix's axis-line strength while removing runtime graph traversal from neural inference.

## Contents

```text
rust/hexo-raster/
  Cargo.toml
  src/lib.rs                 Rust crop, planes, blocker masks, batching, HXR1 wire format
  tests/raster.rs            Engine-facing representation tests

python/hexo_axis_models/
  model.py                   AxisGineCompatNet and PersistentRayAxisNet
  fused_line.py              Triton line gather/backward plus CPU reference
  ops.py                     packed-mask unpacking and dense line primitives
  checkpoint.py              current Strix checkpoint converter
  wire.py                    HXR1 decoder
  klent.py                   segmented KLENT improvement, sampling, lambda returns
  __init__.py

python/tests/                CPU unit tests
python/bench_models.py       ROCm throughput harness
python/example_inference.py  minimal Rust-request to model-output example
configs/                     architecture manifests
integration/raster_axis_parity.rs  graph-versus-ray-mask parity test for hexo-mcts
REPRESENTATION.md            exact tensor and bit contracts
ARCHITECTURES.md             model rationale and rollout plan
```

## Rust integration

Copy the crate into the workspace:

```bash
cp -r rust/hexo-raster /path/to/hexo-strix/hexo-rs/hexo-raster
```

Add it to `hexo-rs/Cargo.toml`:

```toml
[workspace]
members = [
  "hexo-engine",
  "hexo-solver",
  "hexo-mcts",
  "hexo-infer",
  "hexo-wasm",
  "hexo-py",
  "hexo-raster",
]
resolver = "3"
```

Add this dependency to the consumer, probably `hexo-mcts` first:

```toml
hexo-raster = { path = "../hexo-raster" }
```

Build one position:

```rust
use hexo_raster::{build_raster, RasterSpec};

let raster = build_raster(&game, &RasterSpec::default())?;
```

Batch only positions in the same bucket:

```rust
use hexo_raster::RasterBatch;

let batch = RasterBatch::from_positions(&positions)?;
let request_bytes = batch.encode_hxr1();
```

The legal index array is aligned with `GameState::legal_moves()` and is already globalized across `[batch, height, width]` in `RasterBatch`.

## Python integration

From `python/`:

```bash
pip install -e .
pytest -q
```

Decode and run:

```python
import torch
from hexo_axis_models import AxisGineCompatNet, AxisGineConfig, decode_hxr1

batch = decode_hxr1(request_bytes).to("cuda")
model = AxisGineCompatNet(AxisGineConfig()).to("cuda", torch.bfloat16).eval()

with torch.inference_mode():
    out = model.forward_packed(
        batch.planes.to(torch.bfloat16),
        batch.scalars.to(torch.bfloat16),
        batch.active_mask,
        batch.ray_bits,
    )

legal_logits = out.policy_logits.reshape(-1)[batch.legal_flat_indices]
legal_q = out.q_values.reshape(-1)[batch.legal_flat_indices]
```

Convert a current lean axis-relational checkpoint:

```python
from hexo_axis_models.checkpoint import load_strix_axis_checkpoint

report = load_strix_axis_checkpoint(model, "model_selfplay.pt")
print(len(report.copied), report.shape_mismatches)
```

The converter expects the current 4-layer/128-wide lean schema by default. Instantiate a matching `AxisGineConfig` for another checkpoint.

## Recommended rollout

1. Add the Rust representation and write a corpus of HXR1 batches from existing states.
2. Convert the strongest compatible checkpoint into `AxisGineCompatNet`.
3. Validate FP32 block-by-block parity and whole-model policy/Q/value agreement.
4. Benchmark eager and `torch.compile(fullgraph=True)` in BF16 on the Framework Desktop.
5. Fine-tune the compatibility model against current MCTS targets if numerical drift matters.
6. Load the compatible weights into `PersistentRayAxisNet` with `strict=False` and train the zero-output ray graft.
7. Only then switch the actor from Gumbel-MCTS targets to on-policy KLENT.

## Validation

The Python suite covers packed-bit layout, model shapes and gradients, padded
versus active-compacted parity (including parameter gradients),
function-preserving ray graft initialization, D6 orientation-ring
equivariance, HXR1 decoding, segmented KLENT improvement/sampling, and
actor-aware lambda-return signs. The Triton test compares FP32 forward and
backward tensors directly with the portable reference and skips cleanly on
CPU-only hosts.

The integrated workspace also runs:

```bash
cargo test -p hexo-raster
cargo test -p hexo-mcts --test raster_axis_parity
```

The current results are five raster tests and one graph/ray semantic parity
test passing, alongside the KLENT/Python suite.

## First benchmark command

```bash
cd python
python bench_models.py \
  --device cuda \
  --dtype bfloat16 \
  --compile \
  --sizes 17,25,33,49,65 \
  --batches 16,32,64,128 \
  --csv axis_models_rocm.csv
```

Run inference and training separately. The persistent-ray model should be judged on strength per wall-clock, not raw forward speed alone.
