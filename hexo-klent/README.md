# HeXO KLENT

This package is an isolated reference implementation of
[KLENT](https://arxiv.org/abs/2602.10894) for HeXO. It deliberately does not
share the AlphaZero trainer or MCTS actor:

1. Freeze the current policy/Q network.
2. Collect a fixed-size batch of fresh search-free transitions in persistent
   CPU/Rust actor processes. Live game lanes are replaced as games finish,
   while the parent learner owns the only GPU model and dynamically batches
   every actor's inference requests.
3. Form player-aware TD(lambda) action-value targets. A still-live position at
   either the configured rollout horizon or collection-chunk boundary is
   bootstrapped from the frozen network; it is not treated as a draw because
   HeXO has no draw outcome.
4. Train for exactly one shuffled epoch, then discard the batch.

It does reuse the Rust `GameState` and batch graph builders, plus the existing
D6-invariant representation network. This keeps the comparison focused on the
self-play/training algorithm rather than on a second engine implementation.

Run the small-board reference configuration with:

```console
uv run hexo-klent train --config configs/klent/reference-w1.toml
```

Use `--iterations 1 --device cpu` for a short integration run. Checkpoints,
JSONL metrics, and TensorBoard events are written beneath the configured
`run.output_dir`. `collection.positions_per_iteration` fixes the amount of
fresh training data even as game lengths change, while
`collection.parallel_games` controls the number of live game lanes.
`collection.workers` controls independent actor processes,
`collection.inference_batch_size` caps shared GPU batches, and
`collection.inference_edge_budget` automatically splits batches as expanding
graphs become more expensive. `collection.batch_timeout_ms` is the short
request-coalescing window. Actors own no GPU state; the learner's frozen model
serves collection before the same model switches back to fitting.
Axis positions are built once into Rust-collated outer batches, then sliced at
the configured edge-budget boundaries; this preserves the same optimizer
microbatches without tensorising one PyG object per position. `run.compile =
true` compiles the shared GNN core. The experimental dense backend specializes
its expensive relational blocks by fixed raster size while keeping batch size
dynamic. `training.prefetch_batches = true` prepares the next CPU graph or
raster chunk while the GPU fits the current one. Use `--workers N` for a
command-line override. JSONL metrics include collection and fitting elapsed
times plus their position/example throughput. They report horizon, dense
spatial, and collection-chunk truncations separately; all three use
frozen-network bootstrap values rather than draw targets.

`training.batch_size` is the effective optimizer batch. The edge budget may
split it into smaller GPU microbatches, but `training.grad_accumulation = true`
weights each microbatch by its share of examples and performs clipping plus one
AdamW update only after the complete outer batch. This keeps optimizer batch
size and update count stable as self-play positions grow. Set it to `false`
only for an explicit per-microbatch optimization ablation.

## Dense axis compatibility backend

`model.architecture = "dense_axis"` selects the HXR1 raster representation and
the checkpoint-compatible dense transcription of the production
axis-relational GINE model. Its fixed ray radius must cover
`game.win_length - 1`. For this backend the existing `inference_edge_budget`
and `training.edge_budget` knobs count padded raster cells, not graph edges.
States are grouped by raster size before inference and fitting. Rust plans
those groups from dimensions first and materializes only one budgeted group at
a time, so the CPU never constructs a complete 512-position raster chunk at
once.

HeXO's placement radius is local to each existing stone, not a fixed board
boundary. A wandering game can therefore have a very large, mostly empty
bounding box. `collection.dense_position_cell_limit` ends a lane before its
next stored position would exceed that footprint. This is an ordinary KLENT
truncation: the successor is evaluated once with the frozen policy to
bootstrap returns, and no draw is introduced. Set the limit no higher than
`training.edge_budget`; the metrics
`collection/spatial_truncations` and
`collection/max_dense_position_cells` expose how often the guard matters and
how close retained positions come to it. Zero disables the guard.

Initialize a new S3 run from the trained production D6 Q-head checkpoint:

```console
uv run hexo-klent train \
  --config configs/klent/dense-axis-s3-from-d6-qhead.toml \
  --init-from runs/gine-mini/4l-128p32v-lean-d6-qhead/checkpoints/checkpoint_00215547.pt
```

`--init-from` is valid only for a new dense run and is mutually exclusive with
`--resume`. It converts the relational representation plus policy and Q heads,
requires every target tensor to match, records the source path/SHA256/training
step in KLENT checkpoints, and starts a fresh AdamW optimizer. Resume uses the
ordinary dense KLENT checkpoint and does not repeat conversion. Dense
checkpoints work with periodic evaluation, Gumbel MCTS, and the standalone
head-to-head/SPRT command.

The dense compatibility backend uses a custom Triton destination-gather with a
custom fused backward on CUDA/ROCm; CPU execution retains the readable
shift/mask reference. Its per-cell GINE MLPs are evaluated only at active
stone/legal cells and scattered back into the raster, so bucket padding does
not waste matrix-multiply work. Checkpoint parameter names, shapes, and
optimizer state remain unchanged.

On matched 33x33 S3 positions on the Framework Desktop APU, warmed BF16 fit
improved from about 118 positions/s before fusion to about 387 positions/s at
batch 512 (3.28x). Batch 1024 was effectively tied at about 388/s, so the
supplied config retains an effective optimizer batch of 512 but bounds each
forward/backward microbatch at 200,000 padded cells. The compiled
sparse graph model remains faster on this matched workload at about 581/s:
the fused line kernel must still visit padded cells even though the MLPs no
longer do. Steady search-free collection measured about 1,477 positions/s
with one in-process actor and about 1,417/s in the production-shaped
four-actor/64-lane setup. These are warmed rates; the first encounter with a
new raster bucket includes one-time compilation.

FP16 is not materially beneficial on this gfx1151 APU. A matched
pre-compaction batch-512 fit measured about 292 positions/s versus 290/s for
BF16, a gain below 1%, while FP16's first compiler specialization was much
slower and its exponent range is narrower. KLENT therefore keeps BF16 as the
default.

Ctrl-C is handled by the parent process: it stops and reaps the persistent
actors, closes TensorBoard, and exits with status 130 without child-process
tracebacks. An interrupted iteration is not added to `metrics.jsonl` and does
not produce a partial checkpoint; resume from the latest completed checkpoint.
Resuming is always explicit. Pass a checkpoint path, or use `--resume latest`
to select the most recently written `checkpoint_*.pt` or `final.pt` in the
configured `run.output_dir`:

```console
uv run hexo-klent train \
  --config configs/klent/reference-s1.toml \
  --resume latest \
  --iterations 10
```

Selecting by write time preserves the newest branch when an earlier generation
was resumed while stale, higher-numbered checkpoints remain in the directory.

## Training cockpit

Interactive runs open a dedicated full-screen KLENT cockpit. It is independent
of the AlphaZero training display and presents:

- the live `COLLECT -> FIT -> EVAL -> COMMIT` phase and run progress;
- invocation-relative progress plus a live ETA, landing time, and timing-lock
  confidence. Regular and scheduled-evaluation generations are estimated
  separately from the resumed run's deduplicated metric history;
- self-play/fitting throughput, truncation composition, losses, Q scale,
  entropy, reverse KL, gradients, and target concentration;
- factual health signals for finite numerics, exact position/example counts,
  measured gradient-clipping frequency/severity, and Q saturation;
- the latest result for every configured evaluation opponent, including its
  generation and MCTS budget;
- 96-generation sparklines, seeded from an existing `metrics.jsonl` when
  resuming in the same output directory. Duplicate generation records use the
  newest appended value, while generations beyond the resumed checkpoint are
  hidden as stale history from the abandoned branch.

The cockpit is enabled automatically when stdout is an interactive terminal.
Use `--no-tui` for plain logs or `--tui` to request it explicitly:

```console
uv run hexo-klent train \
  --config configs/klent/reference-s1.toml \
  --tui
```

After a successful run, the completed cockpit remains visible until you press
Enter. The final checkpoint and metrics have already been saved, and training
workers plus cached accelerator memory are released before this pause.

The display is observational: JSONL, TensorBoard, and checkpoints remain the
authoritative artifacts. Evaluation coverage is shown separately from
decided-game win rate. Gradient clipping becomes an amber health signal only
when it is both frequent and consequential: at least half of optimizer steps
are clipped and their all-step mean scale is below `0.8`, or the p95 raw norm
is at least four times the configured limit. Lighter clipping remains visible
without putting the whole run into `WATCH`.

## Metrics

`metrics.jsonl` contains one object per completed iteration. TensorBoard uses
the same slash-separated names. Evaluation fields appear only on iterations
divisible by `evaluation.interval`; the console line is a compact subset of
the JSONL record.

### Iteration

| Metric | Meaning |
| --- | --- |
| `iteration` | Completed frozen-policy collect-then-fit generation. |
| `iteration_seconds` | Wall time for collection, fitting, and any scheduled evaluation. It excludes the final metrics/checkpoint writes. |

### Collection

| Metric | Meaning |
| --- | --- |
| `collection/positions` | Fresh transitions collected. This should equal `positions_per_iteration` exactly. |
| `collection/games` | Number of trajectory fragments represented in the chunk. This includes terminal games and bootstrapped horizon/chunk fragments. |
| `collection/p1_wins`, `p2_wins` | Terminal fragments won by each side. |
| `collection/truncations` | Non-terminal fragments closed with a bootstrap value. Equal to horizon plus chunk truncations. |
| `collection/horizon_truncations` | Games that reached `game.rollout_horizon`. |
| `collection/chunk_truncations` | Live lanes closed when the position budget was exhausted. |
| `collection/mean_game_length` | Mean number of stored transitions per trajectory fragment, not necessarily per terminal game. |
| `collection/mean_entropy` | Mean entropy, in nats, of the improved policy used to sample actions. Lower values mean more concentrated action selection. |
| `collection/mean_normalized_entropy` | Per-position improved-policy entropy divided by `log(legal_actions)`, then averaged. This is zero for a forced single-action position and otherwise lies in `[0, 1]`; values near one are close to uniform after accounting for action-set size. |
| `collection/mean_target_top1_probability` | Mean probability mass assigned to the improved policy's most likely legal action. Unlike `training/played_action_target_top1`, this is calculated exactly rather than estimated from sampled actions. |
| `collection/mean_legal_actions` | Mean number of legal actions presented to the policy at each collected position. Read this with raw and normalized entropy to distinguish a growing action set from a genuinely flatter policy. |
| `collection/mean_reverse_kl` | Mean `KL(improved policy || raw network policy)` in nats; see below. |
| `collection/mean_abs_q` | Per-position mean absolute predicted Q across all legal actions, then averaged equally across positions. |
| `collection/mean_q_span` | Per-position `max(Q) - min(Q)` across legal actions, then averaged. A small span means the Q head sees little separation between available moves, irrespective of its absolute value scale. |
| `collection/mean_abs_return` | Mean absolute TD(lambda) return for the sampled actions. |
| `collection/mean_abs_bootstrap_value` | Mean absolute improved-policy/Q state value at truncated successor states; zero when there were no truncations. |
| `collection/worker_processes` | CPU actor processes participating in the chunk. |
| `collection/elapsed_seconds` | Collection wall time. |
| `collection/positions_per_second` | `positions / elapsed_seconds`. |

For each position, let:

```text
pi_raw = softmax(policy_logits)
pi_improved = softmax((beta * policy_logits + Q) / (alpha + beta))
```

The reported reverse KL is:

```text
KL(pi_improved || pi_raw)
  = sum_a pi_improved(a) * log(pi_improved(a) / pi_raw(a))
```

This is the unweighted KL term in KLENT's per-position policy-improvement
objective:

```text
E_pi[Q] + alpha * entropy(pi) - beta * KL(pi || pi_raw)
```

It measures how far the resulting regularized improvement moves away from the
raw policy head. Zero means the two distributions agree. A larger value means
a more substantial policy correction. “Reverse” identifies the direction: the
improved/new distribution supplies the expectation, rather than the raw/old
policy. The logged value is not multiplied by `beta` and is a collection
diagnostic, not a separately optimized training loss. Its useful scale depends
on the number of legal actions and the configured `alpha` and `beta`.

### Training

| Metric | Meaning |
| --- | --- |
| `training/examples` | Fresh positions fitted once during the epoch. Normally equal to `collection/positions`. |
| `training/microbatches` | Edge-budgeted graph batches sent through the model. |
| `training/optimizer_steps` | AdamW updates. With accumulation enabled, normally `ceil(examples / training.batch_size)`. |
| `training/mean_microbatch_size` | Mean positions in one edge-budgeted model batch. |
| `training/mean_optimizer_batch_size` | Mean positions contributing to one AdamW update. |
| `training/mean_microbatches_per_step` | Mean edge-budgeted forward/backward passes accumulated into each update. |
| `training/elapsed_seconds` | Fitting wall time. |
| `training/examples_per_second` | `examples / elapsed_seconds`. |
| `training/policy_loss` | Cross-entropy from the stored improved-policy targets to the policy head. Soft targets have non-zero entropy, so this is not expected to approach zero. |
| `training/policy_excess_kl` | `max(0, policy_loss - collection/mean_entropy)`: the reducible part of policy cross-entropy, equal to the mean `KL(stored improved policy || model policy)` as each sample is encountered during the evolving one-pass fit. Unlike raw policy loss, this can be compared when target entropy changes. Older metrics files are derived automatically for TUI history. |
| `training/q_loss` | MSE between the played action's Q prediction and its TD(lambda) return. Unplayed actions receive no direct Q regression target. |
| `training/total_loss` | `policy_loss + q_loss_weight * q_loss`. |
| `training/mean_grad_norm` | Optimizer-step mean of the total gradient norm before clipping. |
| `training/grad_norm_p50`, `grad_norm_p95`, `grad_norm_max` | Median, 95th percentile, and maximum pre-clipping total gradient norm across optimizer steps. |
| `training/clipped_optimizer_steps` | Number of optimizer steps whose raw gradient norm exceeded `training.max_grad_norm`; zero when clipping is disabled. |
| `training/clip_fraction` | `clipped_optimizer_steps / optimizer_steps`. |
| `training/mean_clip_scale` | Mean multiplier implied by clipping across every optimizer step: `min(1, max_grad_norm / raw_norm)`. Unclipped steps contribute `1`; lower values mean stronger limiting. |
| `training/played_action_target_top1` | Fraction of sampled self-play actions that happened to equal the stored improved policy's argmax. This measures target-policy concentration/sampling, not model classification accuracy. |

### Accelerator memory

On CUDA and ROCm devices, each synchronous phase boundary releases unused
caching-allocator blocks and records `memory/<phase>_...` metrics, where
`<phase>` is `collection`, `training`, or a scheduled `evaluation`. The
`allocated_gib` value is memory still held by live tensors.
`reserved_before_gib` and `reserved_after_gib` show the allocator reserve
immediately before and after `empty_cache()`, while `cache_released_gib` is
their difference. `peak_allocated_gib` and `peak_reserved_gib` are the
high-water marks since the preceding phase boundary.

Long fits also check the allocator reserve every 128 microbatches. When it
exceeds 8192 MB, KLENT releases inactive blocks immediately instead of waiting
for the end of the fit. This avoids unified-memory host OOMs while retaining
the normal fast path below the pressure threshold. The cadence and threshold
share production's `HEXO_CACHE_CLEAR_CHECK_EVERY` and
`HEXO_CACHE_CLEAR_RESERVED_MB` environment overrides; a non-positive threshold
disables in-fit clearing. `training/allocator_pressure_checks`,
`allocator_pressure_clears`, `allocator_pressure_released_gib`, and
`allocator_pressure_max_reserved_gib` report its activity.

A large `reserved_before_gib` is harmless when `reserved_after_gib` falls back
to the small live working set. A rising `reserved_after_gib` indicates
genuinely persistent device allocations. The TUI's `GPU CACHE` signal shows
post-fit and peak reserve together; it enters `WATCH` above 32 GiB peak or
8 GiB post-fit, and escalates above 64 GiB peak or 16 GiB post-fit.

### Evaluation

Each named opponent emits `evaluation/<name>/...`:

| Metric suffix | Meaning |
| --- | --- |
| `games`, `wins`, `losses` | Results from the model's perspective, with model sides alternated. |
| `truncations` | Games still undecided at the evaluation rollout horizon. These are not HeXO draws. |
| `decided_rate` | `(wins + losses) / games`. |
| `win_rate_decided` | `wins / (wins + losses)`; truncations are excluded. Read it together with `decided_rate`. |
| `mean_game_length` | Mean Rust move count, including the forced opening placement. |
| `mean_opponent_depth` | Mean depth actually completed by SealBot; zero for non-search opponents. It can be below the configured maximum when a result is resolved early. |
| `configured_depth` | SealBot's configured fixed maximum depth, when applicable. |
| `placement_radius` | Radius actually used for this opponent, including any opponent-level override. |
| `mcts_simulations`, `mcts_actions` | Model-side Gumbel MCTS budget. Zero simulations means greedy raw-policy play. |
| `opponent_mcts_simulations`, `opponent_mcts_actions` | Checkpoint-side Gumbel MCTS budget. Emitted for fixed and lagged checkpoint opponents. |
| `configured_lag_iterations` | Requested generation distance for a lagged opponent. |
| `opponent_iteration` | Exact historical generation selected for a lagged opponent. |

Periodic evaluators are an explicit opponent list:

```toml
[evaluation]
interval = 10

[[evaluation.opponents]]
name = "random"
kind = "random"
games = 64

[[evaluation.opponents]]
name = "sealbot_raw"
kind = "sealbot"
games = 16
depth = 2
placement_radius = 8
mcts_simulations = 0
mcts_actions = 16

[[evaluation.opponents]]
name = "sealbot_mcts24/8"
kind = "sealbot"
games = 16
depth = 2
placement_radius = 8
mcts_simulations = 24
mcts_actions = 8

[[evaluation.opponents]]
name = "s1_iteration_0200"
kind = "checkpoint"
checkpoint = "runs/klent/reference-s1/checkpoints/checkpoint_000200.pt"
games = 64
placement_radius = 2
mcts_simulations = 24
mcts_actions = 8
opponent_mcts_simulations = 24
opponent_mcts_actions = 8

[[evaluation.opponents]]
name = "lag_100"
kind = "lagged"
lag_iterations = 100
games = 16
placement_radius = 2
mcts_simulations = 24
mcts_actions = 8
```

Each opponent faces the KLENT model with sides alternated between games.
SealBot uses `max_depth = depth` with a non-binding time limit, making its
configured search strength independent of machine load. Its S1 entries
override the evaluation radius to 8—the full HeXO setting SealBot is tuned
for—without changing radius-2 KLENT self-play. `mcts_simulations = 0` evaluates
greedy raw-policy actions; a positive value routes model turns through the
existing Rust Gumbel MCTS using the KLENT policy/Q adapter and `mcts_actions`
root candidates. A `checkpoint` opponent also accepts its own independent
`opponent_mcts_simulations` and `opponent_mcts_actions`. This allows raw/raw,
searched/raw, raw/searched, or matched-search comparisons; a zero simulation
budget selects greedy raw-policy play on that side.

Checkpoint paths are resolved from the trainer's working directory. Both
`hexo-klent-v1` files and HeXO-A0/Strix checkpoints carrying an embedded
`model_config` are supported, including checkpoints whose graph architecture
differs from the current KLENT model. The anchor is loaded once per trainer
process, kept on CPU between evaluation rounds, and moved to the accelerator
only for its games.

A `lagged` opponent selects the exact
`checkpoint_(current_iteration - lag_iterations).pt` and applies
`mcts_simulations` plus `mcts_actions` to both sides. It first searches the
current run's checkpoint directory, then the checkpoint directory from which
the run resumed. This lets a continuation in a new output directory use its
original history until its own checkpoints cover the requested lag. Missing
exact generations are skipped with a warning, so choose lags aligned with the
checkpoint interval. Checkpoint-history directories are themselves persisted
in new checkpoints, so interrupting and resuming a continuation does not lose
access to its original ancestry. Moving lag targets are retained in a bounded
CPU cache; they do not accumulate on the accelerator or grow memory without
limit.

`name` supplies the stable metrics path, so multiple variants of one opponent
kind can run in the same evaluation round. Evaluator implementations are
registered by `kind` in `hexo_klent.evaluation`, so a new opponent does not
require another training-loop branch. If the optional SealBot checkout/build
or a configured checkpoint is unavailable, that in-loop evaluation is skipped
with a warning while the remaining opponents still run. The W1 reference
keeps only the random evaluator: SealBot is tuned for full 6-in-a-row HeXO and
is therefore enabled only in the S1 reference configuration.

The fixed full-rule/radius-2 S1 experiment is:

```console
uv run hexo-klent train --config configs/klent/reference-s1.toml
```

Use the KLENT-native SPRT command to compare two immutable KLENT checkpoints
with the same MCTS budget and inference precision:

```console
uv run hexo-klent sprt \
  --candidate path/to/candidate.pt \
  --opponent path/to/reference.pt \
  --radius 2 \
  --max-moves 1000 \
  --mcts-simulations 24 \
  --mcts-actions 8 \
  --max-games 1000 \
  --state-file runs/klent/sprt/example/state.json
```

Games are played as consecutive candidate-as-P1/candidate-as-P2 pentanomial
pairs. The default test compares parity (`s0 = 0.50`) with a five-point score
advantage (`s1 = 0.55`) at 5% type-I and type-II error bounds. Evidence is
unbounded within the fixed-checkpoint match, and the command stops early on an
SPRT decision or at `max_games`. The state file is atomically replaced after
each complete pair and contains W-D-L, score and confidence interval, Elo
estimate, LLR and decision bounds, throughput, game length, truncations, and
the exact test settings. MCTS inference follows KLENT's BF16 autocast path by
default; use `--precision float32` or `--no-compile` for explicit ablations.

KLENT checkpoints can be passed directly to the existing head-to-head command.
The loader recognizes `hexo-klent-v1` and derives each MCTS leaf value from the
checkpoint's policy/Q outputs:

```console
uv run hexo-a0 head-to-head \
  --checkpoint-a runs/klent/reference-s1/checkpoints/final.pt \
  --checkpoint-b path/to/alphazero-checkpoint.pt \
  --win-length 6 --radius 2 --max-moves 300
```
