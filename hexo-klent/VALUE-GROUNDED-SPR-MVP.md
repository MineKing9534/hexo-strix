# Value-grounded SPR for KLENT

Status: implementation-ready MVP plan for HeXO KLENT. This document commits the first
training-only experiment and records the intended path toward hierarchical
strategic goals. Later stages are motivation, not part of the MVP.

## Summary

Add a self-taught, candidate-conditioned auxiliary loss to the KLENT training
regime. From the current position and only the placement selected at that
position, a small predictor must produce a latent whose learned Q readout
matches the Q-vector of the actual self-play position four placements later.

```text
current position s_t + candidate placement a_t
                    |
                    v
        predicted strategic latent H_hat_t+4
                    |
                    v
          predicted future Q-field

target: Q-field of the actual self-play state s_t+4
```

The intervening actions are deliberately hidden from the predictor. They are
used by the real game trajectory to determine the target position, but they are
not model inputs. The target is produced by the same frozen network that
collected the generation. There is no MCTS teacher, expert checkpoint, future
latent target, or learned board-transition target.

The MVP is an anticipatory representation objective. It is not yet an internal
planner because its predicted latent does not affect inference-time action
selection.

## Motivation

### Predict strategic consequence, not future state

A raw future position is determined by our later choices and the opponent's
choices. Asking a candidate-only predictor to reconstruct that exact position
would force it to infer arbitrary details of one sampled continuation. Those
details are not the desired abstraction.

Instead, value-grounded SPR asks the predictor to preserve only the future
position's strategic action values. Different future positions may be treated
as equivalent whenever they have the same relevant Q readout. The predicted
latent is otherwise unconstrained.

```text
different continuations and board arrangements
                    |
                    v
       equivalent future strategic Q-field
```

This is inspired by value equivalence but does not claim the stronger formal
condition of preserving every Bellman update for specified policy and value
function classes. “Value-grounded SPR” or “Q-readout-equivalent prediction” is
the precise name for the implemented objective.

### Candidate-only conditioning is intentional

The predictor sees `s_t` and `a_t`, not the realized action sequence from
`t + 1` through `t + 3`. Under squared error, the ideal deterministic predictor
therefore estimates the conditional future Q-field induced by self-play:

```text
E[Q_theta(s_t+4, .) | s_t, a_t, current self-play continuation]
```

This is the anticipatory behavior being tested. Given a candidate placement,
the network should learn the strategic consequences that tend to follow from
its own continuation and the opponent's response, without explicitly rolling
either out at inference.

Conditioning on the complete realized sequence would answer the easier and less
useful question “what followed these already-known moves?” It would not test
whether the current representation can anticipate the consequence of choosing
the candidate.

### Why self-play targets are sufficient

SPR is self-taught. The target need not come from a fixed expert or MCTS. At
generation `n`, the frozen KLENT network supplies both the self-play response
distribution and the Q estimate at the reached future position. As training
improves, the intended limiting behavior is:

```text
self-play policies -> equilibrium/minimax play
learned Q_theta     -> minimax Q*
future target       -> Q*(s_t+4, .) on equilibrium-reachable positions
```

The predictor then learns the conditional minimax consequences of a candidate
on the support of strong play. If optimal replies are effectively unique, or
different optimal replies reach Q-equivalent positions, future-value prediction
can become highly accurate.

If several equally strong replies lead to genuinely different Q-fields, a
deterministic predictor retains irreducible error and approaches their
conditional mean. That is useful evidence for a later distributional or
multi-hypothesis extension; it is not a reason to add an expert to the MVP.

### Why the Q-vector rather than scalar V

A scalar `V(s)` states how favourable a position is but discards which moves
make it favourable. The legal-action Q-vector carries a richer strategic
signature:

```text
V target:  how good is the future position?
Q target:  how good is each available continuation from it?
```

The Q-field therefore supplies more pressure for the shared trunk to encode
threats, replies, forks, and strategically interchangeable routes. The scalar
value remains available as an aggregation of policy and Q; it does not need a
second auxiliary target in the MVP.

### This is not a learned world model

The Rust engine remains the exact world model for legality, transitions, turn
semantics, terminal detection, and any concrete search. Value-grounded SPR does
not learn to reproduce those mechanics. It learns an amortized strategic
forecast above them:

```text
exact engine:       what positions and moves are legal?
value-grounded SPR: what strategic value structure tends to follow this choice?
```

## Relationship to the RPS Chess implementation

The RPS Chess value mode establishes the reference semantics:

```text
z_t       = pooled_trunk(s_t)
z_hat     = predictor(z_t, a_t)
q_hat     = auxiliary_value_head(z_hat)
q_target  = stop_gradient(RL_Q_head(s_t+horizon))

L_SPR = Q_distance(q_hat, q_target)
      + VICReg(z_t)
      + VICReg(z_hat)
```

Although RPS Chess collection stores every action over the configured horizon,
value mode passes only the first action to the predictor. It uses the actual
self-play future position and masks the target to future legal actions. It does
not compare the predicted latent with the future trunk latent.

The Strix MVP preserves those decisions. Necessary adaptations are:

- RPS Chess has a fixed 648-action vector; bounded HeXO uses a spatial,
  coordinate-indexed Q-field.
- The exact-D6 CNN trunk should retain a spatial predicted latent rather than
  destroy orientation through global pooling.
- Four atomic placements return HeXO to the same player and the same
  within-turn phase, making `t + 4` the first semantic horizon.
- Ragged Q error is normalized within each future position before positions
  are averaged.
- Strix materializes the target Q-vectors before FIT so one generation has a
  stationary target. RPS Chess currently recomputes its stop-gradient target
  from the changing online network during FIT.

## MVP contract

### In scope

- Training-only value-grounded SPR for `hex_d6_dilated_cnn`.
- Bounded-board KLENT runs with `collection.board_radius > 0`.
- One candidate placement as the only action input.
- One fixed horizon of four atomic placements.
- Actual on-policy self-play continuations.
- Full legal Q targets from the generation's frozen collection model.
- A spatial predictor, separate per-cell auxiliary Q readout, and VICReg.
- Checkpointing, metrics, tests, and a matched control experiment.

### Explicitly out of scope

- Expert checkpoints or MCTS labels.
- Conditioning on opponent moves or later moves by our policy.
- Future-state or future-latent reconstruction.
- Inference-time use of the predicted latent.
- Multiple horizons, response distributions, or uncertainty heads.
- Goal-conditioned policies or learned subgoal selection.
- Graph, dense-axis, persistent-ray, and non-D6 CNN backends.
- Offline expert-game datasets.

## Data flow

### Preserve trajectory order until pairs are formed

KLENT already retains ordered `Trajectory.steps` and flattens them only after
collection. Before flattening, form eligible source/target pairs from every
genuinely terminal training trajectory:

```text
source: step[t]
input:  step[t].chosen placement a_t
target: step[t + 4].state
hidden: actions at t + 1, t + 2, t + 3
```

Eligibility requirements:

1. `t + 4` exists; do not clamp a short future to the final sample.
2. Source and target have the same `player`.
3. Source and target have the same `moves_remaining_this_turn()`.
4. Both states are non-terminal stored decisions.
5. The source action index resolves to the exact coordinate returned by the
   source state's legal-move ordering.

The player/phase checks are invariants, not a recovery mechanism. A violation
indicates a misunderstanding of engine turn semantics and must fail loudly.

### Deterministic sampling

The initial production-shaped generation may contain more than 131,000
positions. Select a bounded auxiliary subset after constructing all eligible
pairs:

- 2,048 training pairs per generation.
- 256 additional held-out diagnostic pairs.
- Sample deterministically from `(run seed, generation, trajectory index,
  step index)`.
- Assign entire trajectories to training or diagnostic selection before
  choosing steps, avoiding near-duplicate positions across the two sets.

All ordinary KLENT examples remain in policy/Q FIT. SPR sampling controls only
the auxiliary work.

### Materialize stationary self-taught targets

After collection and before the first optimizer update:

1. Put the collection model in evaluation/inference mode.
2. Batch-evaluate the selected `s_t+4` states with `forward_batch`.
3. Split the flat Q output by future legal-action count.
4. Store each future legal coordinate and its FP32 Q target.
5. Restore the model's prior mode.

The model has not changed since it generated the trajectories, so these are
generation-local self-play targets. Materialization prevents earlier FIT
updates from moving the targets for later batches and avoids retaining a second
network.

Store only the 2,304 selected ragged Q-vectors, not Q-vectors for the complete
generation. Target storage should remain small and bounded.

## Architecture

### Fixed board frame

The first implementation supports bounded play because it provides a durable
coordinate vocabulary. For board radius `R`, map axial coordinate `(q, r)` to
the fixed `(2R + 1) x (2R + 1)` frame with origin `(-R, -R)`. Mask cells outside
the valid hexagon:

```text
max(abs(q), abs(r), abs(q + r)) <= R
```

For the current radius-20 run this is a 41x41 tensor. Re-pad the selected
source states' ordinary HXR1 raster planes and masks into this frame before the
auxiliary trunk forward. The ordinary policy/Q FIT path remains cropped and
unchanged.

### Candidate encoding

Create one spatial candidate plane with value one at the selected placement
coordinate and zero elsewhere. The predictor input is:

```text
concat(current trunk field H_t, candidate plane A_t)
```

No tensor accepted by the predictor may contain the later self-play actions,
future stones, future legal mask, or future Q values. Keep the predictor API
narrow enough for tests to enforce this structurally.

### Spatial predictor

Add a `ValueGroundedSPR` training module with:

1. A 1x1 input projection from `channels + 1` back to `channels`.
2. Two exact-D6 residual blocks, initially using dilations 1 and 2.
3. A residual connection from `H_t`.
4. A final normalization matching the trunk's feature convention.
5. The fixed valid-board mask after every spatial block.

Initialize the predictor's final residual projection to zero. Its initial
forecast is therefore close to the current representation rather than an
arbitrary field, while candidate-conditioned corrections can grow during
training.

### Auxiliary Q readout

Use a separate shared linear per-cell readout followed by `tanh`:

```text
q_hat(q, r) = tanh(linear(H_hat[:, q, r]))
```

Gather `q_hat` only at the actual future state's legal coordinates. This is the
spatial analogue of RPS Chess's separate `SprValueHead`. Do not feed the future
legal mask into the predictor itself.

The auxiliary readout and predictor are trained by SPR. The ordinary KLENT
Q-head is trained only by its existing played-action return targets and any
separately configured search-Q ablation.

### VICReg populations

Compute a masked mean over the fixed valid-board field for each source and
prediction:

```text
z_t   = masked_mean(H_t)
z_hat = masked_mean(H_hat)
```

Apply VICReg variance and covariance penalties independently to `z_t` and
`z_hat`. There is intentionally no invariance/MSE term between either latent
and no use of `trunk(s_t+4)` as a latent target.

The future-Q loss supplies local, action-specific pressure. VICReg prevents the
global strategic representation from collapsing to the few directions needed
by a scalar per-cell readout.

## Loss and gradient flow

For sample `i` with future legal coordinates `C_i`:

```text
L_value_i = mean over c in C_i of (q_hat_i(c) - q_target_i(c))^2
L_value   = mean over samples of L_value_i

L_spr = L_value
      + variance_weight  * (VarReg(z_t)   + VarReg(z_hat))
      + covariance_weight * (CovReg(z_t) + CovReg(z_hat))

L_total = L_policy
        + q_loss_weight * L_Q
        + effective_spr_weight * L_spr
```

Normalize legal actions inside each state first. Otherwise late positions with
large legal sets would dominate merely because their ragged vectors are longer.

Gradient routes:

```text
L_policy -> policy head + shared trunk
L_Q      -> normal Q head + shared trunk
L_spr    -> auxiliary Q readout + predictor + shared trunk
```

No gradient enters the materialized target. The combined gradient reaches the
shared trunk in the same optimizer step as policy and Q whenever the optimizer
group contains SPR examples.

With gradient accumulation, weight each SPR microbatch by its share of the
optimizer group's SPR examples, independently of the ordinary example and
played-Q populations.

## Configuration

Add a top-level configuration section with disabled defaults:

```toml
[value_spr]
enabled = false
horizon_placements = 4
samples_per_iteration = 2048
diagnostic_samples = 256
loss_weight = 0.05
warmup_iterations = 5
predictor_blocks = 2
predictor_dilations = [1, 2]
variance_weight = 0.5
covariance_weight = 0.1
```

Validation when enabled:

- Architecture must be `hex_d6_dilated_cnn`.
- `collection.board_radius` must be positive.
- Horizon must be exactly four for the MVP.
- Sample counts, block counts, and dilations must be positive.
- Loss and regularizer weights must be finite and non-negative.
- `critic_head_only` and `heads_only` are incompatible with SPR because both
  promise that the shared representation remains frozen.

Ramp only the outer SPR weight from zero to `loss_weight` over the first five
SPR-enabled generations. If SPR is enabled while resuming an older checkpoint,
the ramp begins at that checkpoint's iteration rather than at global iteration
one.

## Training and checkpoint integration

Keep `ValueGroundedSPR` trainer-owned and checkpoint it separately from the
inference model. Use a second AdamW optimizer containing only predictor and
auxiliary-readout parameters. The existing KLENT optimizer continues to own the
shared trunk and normal heads.

For an optimizer group containing SPR examples:

1. Zero both optimizers.
2. Backpropagate policy, Q, and weighted SPR losses.
3. Compute/clip one global norm across base and auxiliary trainable parameters.
4. Step the base optimizer.
5. Step the auxiliary optimizer.

For groups without SPR examples, do not step the auxiliary optimizer, avoiding
weight-decay-only movement. Synchronize its learning rate with the scheduled
base learning rate and use the same configured weight decay.

Add optional checkpoint fields:

```text
value_spr_state_dict
value_spr_optimizer_state_dict
value_spr_start_iteration
value_spr_version
```

Compatibility behavior:

- Disabled runs retain the historical execution path and checkpoint contents.
- Enabling SPR while resuming a pre-SPR checkpoint restores the base model and
  optimizer exactly, then initializes only the auxiliary module and optimizer.
- Resuming an SPR checkpoint strictly restores both auxiliary states.
- Resuming an SPR checkpoint with SPR disabled restores the base state and logs
  that the optional auxiliary state was intentionally ignored.
- Inference exports continue to exclude every SPR tensor.

## Telemetry and falsification checks

Record JSONL and TensorBoard metrics for:

- Eligible, selected, trained, and diagnostic pair counts.
- Short-horizon exclusions and any player/phase invariant failures.
- Target materialization time, legal-label count, and mean labels per pair.
- Target mean absolute Q, Q span, and standard deviation.
- Training and held-out future-Q MSE.
- Current and predicted VICReg variance/covariance components.
- Minimum latent-channel standard deviation and effective rank.
- SPR parameter gradient norm and shared-trunk SPR gradient norm.
- SPR-to-policy, SPR-to-Q, and policy-to-Q trunk-gradient cosine.
- Extra FIT time, peak allocated/reserved accelerator memory, and positions/s.

On held-out pairs, compute the same loss with:

1. The actual candidate plane.
2. A zero candidate plane.
3. Candidate planes shuffled between source positions.

The central conditioning check is:

```text
loss(actual candidate) < loss(zero candidate)
loss(actual candidate) < loss(shuffled candidate)
```

If those gaps do not emerge, the predictor is copying current-state value
structure rather than learning consequences of the candidate.

Do not interpret a falling auxiliary loss, healthy VICReg statistics, or good
frozen-target fit as a strength result. Playing strength requires matched
evaluation.

## Tests

### Data semantics

- Pair construction uses `t` and exactly `t + 4`.
- Only `step[t].action_index` becomes predictor input.
- Changing actions `t + 1` through `t + 3` without changing the stored target
  metadata cannot change the constructed predictor input.
- Short futures are excluded rather than clamped.
- Source and target player/turn-phase invariants are enforced.
- Candidate and future legal coordinates preserve engine ordering.
- Trajectory-level training/diagnostic selection is deterministic and disjoint.

### Model and loss

- Predictor and auxiliary-head shapes match the configured fixed board.
- The candidate plane changes the forecast.
- D6-transforming source state, candidate coordinate, and target coordinates
  transforms the predicted Q-field consistently.
- Q error is averaged per position, not globally over ragged actions.
- SPR-only backward produces gradients in the predictor, auxiliary head, and
  shared trunk but not the ordinary policy/Q heads.
- VICReg remains finite for small populations and has no latent invariance term.
- Future state tensors are absent from the predictor signature.

### Compatibility and integration

- `enabled = false` preserves existing forward results and optimizer behavior.
- A pre-SPR checkpoint can begin an SPR continuation without changing restored
  base tensors or AdamW moments.
- An SPR checkpoint resumes its auxiliary parameters and optimizer exactly.
- A tiny CPU generation completes with finite losses and metrics.
- A bounded ROCm canary validates memory, target materialization, and the
  compiled ordinary FIT path before a production experiment.

## Initial experiment and acceptance criteria

Create matched, separate continuations from the same D6 KLENT checkpoint and
optimizer state:

```text
control:   ordinary KLENT
treatment: ordinary KLENT + value-grounded SPR
```

Keep self-play positions, batch size, learning rate, policy/Q losses, seed
schedule, and evaluation opponents identical. Compare both equal-generation
and wall-clock cost because the treatment performs additional target and trunk
forwards.

Run order:

1. CPU integration smoke.
2. Small quiet-GPU canary with reduced lanes and positions.
3. Five-generation conditioning/collapse check.
4. A matched continuation long enough to reach periodic fixed-opponent
   evaluations.
5. Paired colour-swapped checkpoint evaluation against the same fixed
   opponents and openings.
6. Replicate a positive result from another seed before attributing strength.

MVP engineering acceptance requires:

- Disabled-path parity.
- No target leakage or opponent-action conditioning.
- Finite training with bounded memory.
- Noncollapsed latents.
- A positive held-out candidate-conditioning gap.
- Recorded shared-trunk gradient interaction and compute overhead.

Research success additionally requires matched playing strength that is at
least non-inferior to the control, with a reproducible positive signal. The
auxiliary metrics alone cannot satisfy this gate.

## Future motivation: from anticipation to hierarchical goals

The MVP is deliberately only the representation-learning foundation. Its
long-term purpose is to make strategic consequences available cheaply enough
to guide multi-timescale decisions.

### Stage 1: multi-horizon value-grounded prediction

Once the four-placement predictor demonstrably uses its candidate input,
extend it to several same-player/same-phase horizons:

```text
a_t -> Q at t+4
a_t -> Q at t+8
a_t -> Q at t+16
a_t -> eventual outcome/value
```

Keep every target self-taught from actual trajectories. Use separate horizon
embeddings or recurrent prediction steps, and retain the candidate-only input
contract. A multi-horizon value profile supplies the missing timescale axis:

```text
near consequence -> medium consequence -> long strategic consequence
```

This should be attempted before a learned goal hierarchy. If the trunk cannot
predict its own future value structure at useful horizons, a goal selector
would have no reliable strategic substrate.

### Stage 2: uncertainty over self-play continuations

Candidate-only prediction deliberately hides future actions. Residual error can
therefore separate two cases:

- The value target is poorly learned.
- Several plausible strong continuations have different value structures.

After the deterministic MVP, test a distributional Q readout, a small ensemble,
or discrete latent hypotheses. Train these from self-play diversity rather than
introducing a privileged expert. Checkpoint mixtures or stronger historical
self-play opponents may broaden the response distribution while keeping the
targets self-generated.

The output should describe strategic alternatives or uncertainty, not attempt
to reconstruct exact future boards.

### Stage 3: inference-time candidate evaluation

Training-only SPR teaches anticipatory features but is not planning. The first
inference use should be an explicit ablation:

```text
baseline score = ordinary policy/Q score for candidate a

fused score = baseline score
            + learned function(predicted strategic latent after a)
```

Stage the gradient boundary:

1. `aux`: predictor discarded at inference, as in the MVP.
2. `fused_detached`: inference consumes the prediction, but policy/Q training
   cannot reshape it through the fusion path.
3. `fused`: end-to-end inference-conditioned training after the detached form
   proves useful and stable.

The exact Rust engine continues to enforce legality and execute candidates. The
strategic predictor contributes ranking information, not fictitious state
transitions.

### Stage 4: value targets as hierarchical goals

A strategic goal should specify an equivalence class of acceptable futures,
not a target board. Let `g` be a desired value level:

```text
policy(a | s, g): choose an action leading toward any state with V(s') >= g
```

Many positions and routes intentionally alias to the same `g`. That is the
point: the goal says what strategic quality to achieve, while the current state
determines which route is feasible. State-target reconstruction would destroy
this abstraction.

The requested hierarchy becomes:

```text
WIN
  -> long strategic value subgoal
      -> medium/near value subgoal
          -> next-turn value target
              -> legal placements from policy/Q and the exact engine
```

Value-grounded SPR supports this hierarchy in three ways:

1. The trunk is trained to represent future value structure, so `g` refers to
   information the representation already preserves.
2. Candidate-conditioned forecasts estimate which current choices move toward
   a desired value class without naming a target board.
3. Multi-horizon forecasts provide evidence for decomposing a distant goal into
   nearer achievable value targets.

An initial goal-conditioned controller can learn by hindsight from self-play:

1. Select a future state from the same trajectory.
2. Derive its scalar value target `g` from the frozen Q/policy readout.
3. Train `policy(a | s, g)` on the earlier action.
4. Contrast with unreachable or lower-value goals from other trajectory
   segments.

No learned goal generator should be introduced until an oracle-goal test shows
that supplying the true future value goal improves tactical action selection.

### Stage 5: recursive goal selection

If oracle and learned goal conditioning work, add a high-level selector that
chooses a reachable next value class from the current long-term goal:

```text
selector(s, g_long, horizon)
    -> g_near

controller(s, g_near)
    -> candidate placements
```

Selection must consider predicted attainability and uncertainty, not merely
choose the numerically highest `g`. The engine validates concrete moves; the
predictor estimates their strategic consequences; the hierarchy chooses which
value class should be achieved next.

A scalar goal should remain the first experiment because its broad aliasing is
the desired abstraction. Add a horizon or urgency component `(g, h)` only if
evidence shows that equal-valued goals with different timescales cannot be
controlled reliably. As learned Q approaches sharp minimax outcomes, horizon,
uncertainty, or proof depth may provide useful intermediate rungs without
turning goals back into state descriptions.

### Long-term success condition

The eventual system should be able to answer progressively richer questions:

```text
MVP:      If I choose this placement, what future Q structure tends to result?
multi-h:  How does that strategic structure evolve across timescales?
goals:    Which value class should I reach next to make WIN achievable?
control:  Which legal placements best realize that next value class?
```

That is a learned hierarchical strategic model above an exact deterministic
game engine—not a replacement for the engine, and not a reconstruction model.
