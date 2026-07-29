# HeXO KLENT

This package is an isolated reference implementation of
[KLENT](https://arxiv.org/abs/2602.10894) for HeXO. It deliberately does not
share the AlphaZero trainer or MCTS actor:

1. Freeze the current policy/Q network.
2. Collect fresh search-free transitions in persistent CPU/Rust actor
   processes. The configured position count is the point where actors stop
   starting replacement games; every live lane then drains to a genuine HeXO
   win under the same frozen model.
3. Form player-aware TD(lambda) action-value targets only for naturally
   terminal games. A game stopped by a rollout-horizon or dense-spatial safety
   cap is discarded in its entirety and contributes no FIT samples.
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
`run.output_dir`. `collection.positions_per_iteration` is a soft collection
budget: terminal draining normally makes the resulting FIT batch slightly
larger, while discarded safety-capped games can make it smaller. The recorded
`collection.positions` always equals the number of genuinely terminal FIT
examples. `collection.parallel_games` is the total number of live game lanes
across all actor processes.
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
times plus their position/example throughput. They report horizon and dense
spatial caps separately, along with how many positions were discarded. A
collection budget never truncates a trajectory and no non-terminal position
is trained as either a draw or a bootstrapped target.

`training.batch_size` is the effective optimizer batch. The edge budget may
split it into smaller GPU microbatches, but `training.grad_accumulation = true`
weights each microbatch by its share of examples and performs clipping plus one
AdamW update only after the complete outer batch. This keeps optimizer batch
size and update count stable as self-play positions grow. Set it to `false`
only for an explicit per-microbatch optimization ablation.
`training.policy_diagnostic_samples` selects an evenly spaced, deterministic
slice of each fresh collection for a forward-only policy check after fitting.
The default is 2,048 positions; zero disables the check.

## Axis-GINE graph backend

`model.architecture = "graph"` uses the production-compatible relational GNN.
For FIT, `training.fit_max_autotune = true` enables compile-time-only GEMM
tuning and `training.fit_compile_seed_nodes` chooses one representative graph
fragment for that initial compile. Runtime tuning stays disabled: changing
ragged row counts would otherwise repeatedly launch compiler workers and grow
host memory. The seed runs one forward/backward without an optimizer step,
then releases its gradients before the exactly-once training epoch begins.
The trainer also raises its own soft open-file limit to 65,536 (bounded by the
process hard limit) before compilation. Triton's compile-time autotuning can
otherwise exhaust a tmux server's common 1,024-descriptor soft limit and leave
an incomplete compiler cache.

Initialize the S4 graph run directly from the production D6 Q-head checkpoint:

```console
uv run hexo-klent train \
  --config configs/klent/axis-gine/s4-from-d6-qhead.toml \
  --init-from runs/gine-mini/4l-128p32v-lean-d6-qhead/checkpoints/checkpoint_00215547.pt \
  --tui
```

The graph path strictly copies every representation, policy, and Q tensor and
ignores production-only value/auxiliary heads. It begins at KLENT iteration 0
with fresh AdamW state. Subsequent launches use `--resume latest`, not
`--init-from`.

For a pretrained graft, `training.learning_rate_warmup_iterations` linearly
ramps a generation-level learning rate from
`training.learning_rate_warmup_start_factor * training.learning_rate` to the
configured rate. The applied value is recorded as `training/learning_rate` in
JSONL and TensorBoard. Because it is derived from the committed iteration,
ordinary checkpoint resume requires no separate scheduler state.

KLENT's test-time MCTS adapter follows Appendix M of the paper and reconstructs
the state value as the raw learned-policy expectation
`sum_a pi_theta(a|s) Q_theta(s,a)`. Self-play lambda-returns use the
improved-policy state values at intermediate timesteps; terminal outcomes end
the recursion.

## Dense raster backends

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
next stored position would exceed that footprint. The whole non-terminal game
is then discarded so none of its positions enter FIT. Set the limit no higher
than `training.edge_budget`; the metrics
`collection/spatial_truncations` and
`collection/max_dense_position_cells` expose how often the guard matters and
how close retained positions come to it. Zero disables the guard.

Initialize a new S3 run from the trained production D6 Q-head checkpoint:

```console
uv run hexo-klent train \
  --config configs/klent/dense-axis-s3-from-d6-qhead.toml \
  --init-from runs/gine-mini/4l-128p32v-lean-d6-qhead/checkpoints/checkpoint_00215547.pt
```

`--init-from` is valid only for a new run and is mutually exclusive with
`--resume`. For `graph`, it performs the strict direct copy described above.
For `dense_axis`, it converts the relational representation plus policy and Q
heads and requires every target tensor to match. For
`persistent_ray_axis`, it may read either that production checkpoint or a
KLENT `dense_axis` checkpoint; the compatibility trunk and policy/Q heads are
copied while newly introduced ray parameters retain their initialization.
All paths record source provenance and start a fresh AdamW optimizer. Resume
uses the ordinary architecture-matching KLENT checkpoint and does not repeat
conversion. Raster checkpoints work with periodic evaluation, Gumbel MCTS,
the policy viewer, and the standalone head-to-head/SPRT command.

The dense compatibility backend uses a custom Triton active-cell gather with a
custom source-centric backward on CUDA/ROCm; CPU execution retains the
readable shift/mask reference. The fused op reads Rust's packed ray words
directly and returns three-axis features only for active stone/legal cells.
Normalization, GINE/global MLPs, node updates, inter-layer state, and JK
features remain compact throughout the KLENT forward. Legal heads gather
directly through Rust's dense-to-active lookup. The compact and dense reference
APIs are separate, so no compact-to-dense compatibility scatter remains.
Checkpoint parameter names, shapes, and optimizer state remain unchanged.

Before the packed compact gather, matched 33x33 S3 positions on the Framework
Desktop APU showed warmed BF16 fit
improved from about 118 positions/s before fusion to about 387 positions/s at
batch 512 (3.28x). Batch 1024 was effectively tied at about 388/s, so the
supplied config retains an effective optimizer batch of 512 but bounds each
forward/backward microbatch at 200,000 padded cells. The compiled
sparse graph model remains faster on this matched workload at about 581/s:
the fused line kernel still visited padded cells even though the MLPs no
longer did. Steady search-free collection measured about 1,477 positions/s
with one in-process actor and about 1,417/s in the production-shaped
four-actor/64-lane setup. These are warmed rates; the first encounter with a
new raster bucket includes one-time compilation.

With the packed compact path, a matched grafted S4 benchmark from dense
checkpoint 230 used 8,192 positions, one actor, batch 512, and the configured
500,000-cell fit budget. Warmed `dense_axis` collection measured about 1,174
positions/s and fit measured about 425 positions/s. Fit peaked at 4.01 GiB
allocated and 6.37 GiB reserved. This benchmark exercises compiled
forward/backward and the same Rust-collated batches as the trainer.

FP16 is not materially beneficial on this gfx1151 APU. A matched
pre-compaction batch-512 fit measured about 292 positions/s versus 290/s for
BF16, a gain below 1%, while FP16's first compiler specialization was much
slower and its exponent range is narrower. KLENT therefore keeps BF16 as the
default.

### Persistent-ray backend

`model.architecture = "persistent_ray_axis"` retains the complete
checkpoint-compatible `dense_axis` trunk and inserts a narrow six-direction
latent stream after the configured `model.ray_after_layers`. The ray state is
persistent across network layers within one position evaluation; it is not
carried between game moves or inference calls. Opposite rays are folded into
axes, then the three axes are combined with sum, pair-product, and maximum
statistics before a residual update enters the scalar trunk. The pair-product
provides an explicit soft fork feature.

The KLENT class is a compile-transparent subclass, not a module wrapper around
a completed dense forward. Like `dense_axis`, it deletes unused value heads
and applies policy/Q MLPs only to legal cells. Its Python dispatch therefore
does not add tensor work relative to using the model directly.

Persistent pointwise updates are compacted to active cells. CUDA/ROCm uses a
custom fused forward/backward gather that consumes compact projected sources
and emits six directed messages only for active destinations, avoiding both a
zero-filled source raster and the reference implementation's thirty shifted
raster tensors. It reads packed ray words directly. Its source-centric
backward gathers six destination gradients per source locally rather than
issuing one global atomic write per ray contribution. The compatibility GINE
blocks use the same compact packed three-axis gather described above.

Before compact sources and the packed three-axis path, a matched warmed BF16
S1 benchmark using the generation-200 dense
checkpoint, 8,192 identical positions, batch 512, and a 200,000-cell fit
budget, `persistent_ray_axis` collected about 1,730 positions/s versus 2,206/s
for `dense_axis` (78%). Fit measured about 571 positions/s versus 1,100/s
(52%), with 2.69 GiB versus 1.92 GiB peak allocated memory. Six repeated
compiled fit passes held live allocation flat at 75 MiB after each pass and
returned reserve to roughly 0.11 GiB at the phase boundary. The persistent
backend performs materially more learned work, so compare strength per
wall-clock as well as raw positions/second.

Before eliminating cross-layer raster materialization, a matched grafted S4
benchmark after compact-source and packed-gather fusion
from dense checkpoint 230 used the same 8,192 positions and seed for both
architectures. Warmed `persistent_ray_axis` collection measured about 885
positions/s versus 1,174/s for `dense_axis` (75%). Fit measured about 182
positions/s versus 425/s (43%), with 6.16 GiB versus 4.01 GiB peak allocated
and 8.83 GiB versus 6.37 GiB peak reserved. On a separate 2,048-position
allocator stress check, five consecutive measured persistent fit passes held
at 216 positions/s, 6.955 GiB allocated, and 8.273--8.275 GiB reserved; no
pass-to-pass GPU growth was observed. A ten-generation S4 canary alternating
fresh 2,048-position collection and fit held warmed collection at
996--1,064 positions/s and fit at 204--219 positions/s. Process RSS moved from
2.322 GiB after the first compiled generation to 2.389 GiB after generation
ten (69 MiB), with no accelerating or GiB-scale host-memory growth.

Keeping the trunk and JK features compact across all layer boundaries raised
the 8,192-position PersistentRay collection benchmark from 885 to 940
positions/s even though the new deterministic trajectory contained 2.1% more
legal actions per position. In a separately instrumented, production-shaped
32,768-position pass, throughput rose from 866 to 957 positions/s (10.5%) and
model-forward time fell from 28.90 to 24.16 seconds (16.4%). Fit rose from 182
to 200 positions/s despite using 112 rather than 107 raster microbatches. Peak
fit allocation fell from 6.16 to 4.37 GiB and peak reserve from 8.83 to
6.43 GiB.

The default branch adds 152,592 parameters to the 4-layer, 128-wide KLENT
model. `exact_graft_init = true` zero-initializes each fold output, so a graft
starts with exactly the source model's function. For example:

```console
uv run hexo-klent train \
  --config configs/klent/persistent-ray/s1-graft.toml \
  --init-from runs/klent/dense-from-scratch/s1/checkpoints/final.pt
```

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
By default, resume also restores the checkpoint's optimizer learning rate,
even if the selected config contains a different value. To branch from a
checkpoint with a new configured learning rate while retaining Adam moments
and all other optimizer state, pass `--resume-configured-lr`:

```console
uv run hexo-klent train \
  --config path/to/lower-lr-branch.toml \
  --resume path/to/checkpoint_000060.pt \
  --resume-configured-lr
```

## Training cockpit

Interactive runs open a dedicated full-screen KLENT cockpit. It is independent
of the AlphaZero training display and presents:

- the live `COLLECT -> FIT -> EVAL -> COMMIT` phase and run progress;
- invocation-relative progress plus a live ETA, landing time, and timing-lock
  confidence. Regular and scheduled-evaluation generations are estimated
  separately from the resumed run's deduplicated metric history;
- self-play/fitting throughput, discarded safety-cap composition, losses, Q scale,
  entropy, reverse KL, gradients, and target concentration;
- factual health signals for finite numerics, terminal-only position/example counts,
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

Press `P` while training to arm a cooperative pause at the end of the current
generation. Collection, fitting, evaluation, metric persistence, and any
scheduled checkpoint therefore finish atomically before the process becomes
idle. Press `P` again before the boundary to cancel, or while paused to resume
the next generation in the same process. The TUI freezes its active timer and
ETA while paused and releases unused accelerator cache; model parameters and
idle actor workers remain resident. `Ctrl-C` remains the durable shutdown path.

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
| `collection/positions` | Fresh transitions from genuinely terminal games and therefore the exact number of examples entering FIT. It may differ from the soft `positions_per_iteration` budget after terminal draining or discarded caps. |
| `collection/discarded_positions` | Generated positions discarded because their whole game hit a horizon or dense-spatial safety cap before producing a winner. |
| `collection/games` | Number of finished games observed, including terminal wins and discarded safety-capped games. |
| `collection/p1_wins`, `p2_wins` | Naturally terminal games won by each side. |
| `collection/truncations` | Non-terminal games discarded from FIT. Equal to horizon plus spatial truncations; collection-budget truncations are never created. |
| `collection/horizon_truncations` | Games discarded after reaching `game.rollout_horizon`. |
| `collection/spatial_truncations` | Dense games discarded after exceeding `collection.dense_position_cell_limit`. |
| `collection/chunk_truncations` | Legacy compatibility metric. New collections always record zero because live lanes drain at the position budget. |
| `collection/mean_game_length` | Mean generated game length across terminal and discarded games. |
| `collection/mean_entropy` | Mean entropy, in nats, of the improved policy used to sample actions. Lower values mean more concentrated action selection. |
| `collection/mean_normalized_entropy` | Per-position improved-policy entropy divided by `log(legal_actions)`, then averaged. This is zero for a forced single-action position and otherwise lies in `[0, 1]`; values near one are close to uniform after accounting for action-set size. |
| `collection/mean_target_top1_probability` | Mean probability mass assigned to the improved policy's most likely legal action. Unlike `training/played_action_target_top1`, this is calculated exactly rather than estimated from sampled actions. |
| `collection/mean_prior_normalized_entropy` | The same normalized-entropy statistic for the raw network policy before KLENT improvement. Compare it with target normalized entropy to distinguish a diffuse learned prior from flattening introduced by the improvement operator. |
| `collection/mean_prior_top1_probability` | Mean top-1 mass of the raw network policy. The TUI displays target/prior top-1 together, so an alpha ablation shows whether the target sharpens first and whether fitting transfers that concentration into the next generation's prior. |
| `collection/mean_legal_actions` | Mean number of legal actions presented to the policy at each collected position. Read this with raw and normalized entropy to distinguish a growing action set from a genuinely flatter policy. |
| `collection/mean_reverse_kl` | Mean `KL(improved policy || raw network policy)` in nats; see below. |
| `collection/mean_abs_q` | Per-position mean absolute predicted Q across all legal actions, then averaged equally across positions. |
| `collection/mean_q_span` | Per-position `max(Q) - min(Q)` across legal actions, then averaged. A small span means the Q head sees little separation between available moves, irrespective of its absolute value scale. |
| `collection/mean_abs_return` | Mean absolute TD(lambda) return for terminal-game sampled actions. |
| `collection/mean_abs_bootstrap_value` | Legacy compatibility metric. Terminal-only collection records zero. |
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
| `training/policy_diagnostic_examples` | Positions in the deterministic collection-wide slice used for the matched target-fit diagnostic. |
| `training/policy_diagnostic_seconds` | Wall time of the extra forward-only diagnostics before and after fitting. |
| `training/policy_target_kl_collection` | Collection actor's `KL(stored improved policy || raw policy)` on the diagnostic slice. |
| `training/policy_target_kl_before` | The same pre-fit KL recomputed through the trainer model. This is the proposed policy-improvement distance before fitting. |
| `training/policy_target_kl_sync_gap` | `policy_target_kl_before - policy_target_kl_collection`. It should stay near zero; a material gap exposes a collection/trainer inference mismatch rather than a FIT effect. |
| `training/policy_target_kl_after` | On the same positions, `KL(stored improved policy || model policy after fitting)`. The stored policy is frozen while the joint fit also changes Q and the shared representation, so a higher value means the old target was not retained—not necessarily that the resulting policy is weaker or unstable. |
| `training/policy_target_progress` | `1 - policy_target_kl_after / policy_target_kl_before`, displayed as **target retention**. Positive values mean the frozen target gap shrank; negative values mean the joint update rewrote the policy away from that old target. This is a diagnostic of alternating policy/Q optimization and does not independently affect TUI health. It is reported as zero when the pre-fit gap is numerically zero. |
| `training/policy_target_top1_agreement_before`, `policy_target_top1_agreement_after` | Fraction of diagnostic positions where the raw policy and stored improved-policy target choose the same top action, before and after FIT. This distinguishes argmax rewrites from KL movement confined to lower-ranked probability mass. |
| `training/policy_target_top1_agreement_delta` | Post-fit minus pre-fit top-action agreement. This is still a target-fit signal, not a direct strength measurement. |
| `training/q_loss` | MSE between the played action's Q prediction and its TD(lambda) return. Unplayed actions receive no direct Q regression target. |
| `training/total_loss` | `policy_loss + q_loss_weight * q_loss`. |
| `training/trunk_gradient_diagnostic_examples` | Positions in the deterministic post-fit diagnostic slice used to compare the two objectives on the shared representation. Gradients are accumulated as an example-weighted mean across every edge-budgeted microbatch; this diagnostic does not update parameters or optimizer state. |
| `training/trunk_gradient_diagnostic_seconds` | Wall time of the extra policy/Q trunk-gradient diagnostic. It performs two backwards per edge-budgeted microbatch in the diagnostic slice. |
| `training/policy_trunk_grad_norm` | L2 norm of the collection-wide mean policy-loss gradient over shared-trunk parameters; policy-head parameters are excluded. |
| `training/q_trunk_grad_norm` | L2 norm of the configured `q_loss_weight * q_loss` collection-wide mean gradient over the same shared-trunk parameters. This makes its magnitude directly comparable with `policy_trunk_grad_norm`. |
| `training/policy_q_trunk_grad_cosine` | Cosine similarity between the policy and weighted-Q shared-trunk gradients after averaging across the diagnostic slice. Positive values mean the objectives agree in aggregate, values near zero mean mostly independent directions, and negative values expose net trunk-gradient conflict. This is a sampled post-fit diagnostic rather than an optimizer-step average. |
| `training/mean_grad_norm` | Optimizer-step mean of the total gradient norm before clipping. |
| `training/grad_norm_p50`, `grad_norm_p95`, `grad_norm_max` | Median, 95th percentile, and maximum pre-clipping total gradient norm across optimizer steps. |
| `training/clipped_optimizer_steps` | Number of optimizer steps whose raw gradient norm exceeded `training.max_grad_norm`; zero when clipping is disabled. |
| `training/clip_fraction` | `clipped_optimizer_steps / optimizer_steps`. |
| `training/mean_clip_scale` | Mean multiplier implied by clipping across every optimizer step: `min(1, max_grad_norm / raw_norm)`. Unclipped steps contribute `1`; lower values mean stronger limiting. |
| `training/mean_parameter_update_norm`, `parameter_update_norm_p95` | Mean and 95th-percentile global L2 norm of the exact parameter delta produced by each AdamW step, including weight decay. Unlike the raw gradient or clip scale, this measures how far the optimizer actually moved the model. |
| `training/mean_update_to_weight_ratio`, `update_to_weight_ratio_p95` | Mean and 95th-percentile parameter-update L2 norm divided by the global pre-step parameter L2 norm. This dimensionless update-to-weight ratio is useful for comparing optimizer movement over time; the TUI displays its mean in parts per million. |
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
| `mean_game_length` | Mean Rust move count, including sampled or rule-forced opening placements. |
| `mean_opponent_depth` | Mean depth actually completed by SealBot; zero for non-search opponents. It can be below the configured maximum when a result is resolved early. |
| `opening_pairs` | Number of sampled openings replayed as candidate-P1/candidate-P2 pairs. Zero for random, SealBot, or legacy no-opening evaluation. |
| `frac_unique_opening` | Fraction of sampled pair openings that are distinct. Read this with `opening_pairs` to verify that the opening sampler is producing useful coverage. |
| `configured_depth` | SealBot's configured fixed maximum depth, when applicable. |
| `placement_radius` | Radius actually used for this opponent, including any opponent-level override. |
| `mcts_simulations`, `mcts_actions` | Model-side Gumbel MCTS budget. Zero simulations means greedy raw-policy play. |
| `opponent_mcts_simulations`, `opponent_mcts_actions` | Checkpoint-side Gumbel MCTS budget. Emitted for fixed, lagged, and best-so-far checkpoint opponents. |
| `opening_plies`, `opening_temperature` | Paired-opening settings used by fixed, lagged, and best-so-far checkpoint opponents. Eight plies are four complete two-placement HeXO turns. |
| `configured_lag_iterations` | Requested generation distance for a lagged opponent. |
| `opponent_iteration` | Exact historical generation selected for a lagged or best-so-far opponent. |
| `promotion_win_rate`, `promoted` | Best-so-far promotion threshold and whether the just-evaluated candidate replaced the incumbent. |

Periodic evaluators are an explicit opponent list:

```toml
[evaluation]
interval = 10
# Defaults shown explicitly: fixed and lagged checkpoint opponents sample one
# opening per two-game pair, replay it with sides swapped, and turn Gumbel
# noise off after the opening.
opening_plies = 8
opening_temperature = 0.5
opening_generator = "alternate"

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

[[evaluation.opponents]]
name = "best_so_far"
kind = "best_so_far"
checkpoint = "runs/klent/reference-s1/checkpoints/checkpoint_000200.pt"
best_promotion_win_rate = 0.55
games = 32
placement_radius = 2
mcts_simulations = 24
mcts_actions = 8
opponent_mcts_simulations = 24
opponent_mcts_actions = 8
```

Each opponent faces the KLENT model with sides alternated between games.
Fixed and lagged checkpoint matches additionally use the same paired-opening
protocol as `hexo-a0 head-to-head`: one raw-policy opening is sampled for each
two-game pair, then replayed exactly with the candidate as P1 and P2. By
default, the opponent generates the first pair's opening, the candidate the
second, and so on (`opening_generator = "alternate"`). `"a"` always uses the
candidate; `"b"` or `"champion"` always uses the checkpoint opponent. Once an
opening is applied, in-tree Gumbel noise is disabled on both sides, so the
paired result conditions on the same position rather than two unrelated search
seeds. `opening_plies = 0` restores the legacy empty-board, noise-on protocol.
Paired evaluation requires an even game count. Opening sampling preserves the
trainer's torch RNG state and therefore does not perturb the next generation's
shuffle or self-play randomness.

`best_so_far` starts from its configured `checkpoint`, then persists the
incumbent under `<output_dir>/best_so_far/<name>.json`. At each evaluation the
current in-memory candidate plays that stable incumbent. If its decided-game
win rate is at least `best_promotion_win_rate`, the trainer saves the current
generation even when it is not otherwise a checkpoint interval and atomically
promotes that checkpoint. The state file is reused by ordinary `--resume`, so
the incumbent cannot silently reset to the configured bootstrap checkpoint.

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

Standalone head-to-head uses a fixed pentanomial pair-score variance of `0.5`.
It reports the empirical variance as telemetry but does not adapt the LLR model
online from a handful of early pairs. This conservative default matches the
roughly `0.44`-`0.51` empirical variance seen in substantial HeXO matches. Use
`--pair-variance` to choose a
different fixed value deliberately.
