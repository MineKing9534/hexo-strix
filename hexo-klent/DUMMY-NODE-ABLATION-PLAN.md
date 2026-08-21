# Dummy-node diagnosis and ablation plan

## Purpose

This note summarizes the discussion around Strix/KLENT's global dummy node and
defines the most useful experiments under a limited full-training budget. The
main question is not simply whether global communication is useful. It is
whether the current single, raw-summed global channel is the best way to expose
global context without weakening the spatial representation.

The practical goals are:

1. preserve the strong long-range communication provided by the dummy node;
2. prevent global context from becoming an unnecessarily attractive shortcut;
3. avoid compressing every strategically relevant global fact into one shared
   128-dimensional vector if that limits strength;
4. keep the first experiments cheap enough to run as controlled full-training
   comparisons; and
5. support checkpoint-grafted screening where exact functional equivalence is
   possible.

## Current KLENT architecture

The checkpoint inspected in this discussion was:

```text
runs/klent/axis-gine/
  s4-from-mc25-tau16-alpha001-full-batch2048/
  checkpoints/checkpoint_000246.pt
```

Its relevant architecture is:

```text
4 edge-aware residual layers
hidden width 128
JK-cat across the four layers
pre-LayerNorm

three tied spatial axis relations
one separately parameterized global relation

one dummy node
all real nodes <-> dummy

node update input:
    [local residual, axis output + global output]

policy and per-action Q heads:
    legal real-node embeddings only
```

The global relation has its own GINE branch, so it is already cleaner than the
older Strix checkpoint in which global and spatial edges were mixed inside one
GINE aggregation. However, the axis and global branch outputs are added before
the learned node update. Once added, the update cannot tell which branch
produced a feature.

The dummy performs three conceptually different jobs:

1. carrying genuinely global state;
2. aggregating a summary of all real nodes; and
3. broadcasting that summary into every real node.

Good ablations should separate those jobs rather than treating "dummy node" as
one indivisible feature.

## Checkpoint diagnostic

A read-only diagnostic generated eight on-policy games using 24 Gumbel
simulations and eight candidate actions. It produced 750 positions, of which
256 evenly distributed positions were inspected. Graphs ranged from 218 to
1,107 nodes and games ranged from 38 to 170 placements.

The independently reconstructed forward pass matched the checkpoint exactly.

### Global branch versus spatial branch

| Layer | Median global-output / axis-output norm | Median fraction of real nodes where global output dominates |
| ---: | ---: | ---: |
| 0 | 0.36 | approximately 0% |
| 1 | 0.31 | approximately 0% |
| 2 | 0.35 | approximately 0.1% |
| 3 | 0.92 | 32% |

This is broadly healthy. Spatial processing dominates the first three layers;
the global branch approaches parity only in the final layer. The checkpoint is
therefore not simply replacing spatial reasoning with a global vector
everywhere.

### Raw-sum scaling

| Layer | Median dummy/ordinary global-input norm | Correlation with real-node count |
| ---: | ---: | ---: |
| 0 | 179x | 0.999 |
| 1 | 122x | 0.994 |
| 2 | 99x | 0.998 |
| 3 | 415x | 0.996 |

The dummy's raw incoming sum is almost perfectly coupled to graph size.
Pre-LayerNorm substantially limits the consequences of pure magnitude growth
before the dummy is broadcast on the following layer, but it does not make the
aggregation independent of cardinality or nonlinear operating regime.

The final layer's real-to-dummy aggregation is dead computation. Its result
cannot be broadcast because there is no fifth layer, and the policy/Q heads do
not read the dummy. The useful final-layer dummy-to-real broadcast carries the
dummy state produced by the preceding layer.

### Incoming bandwidth

The entropy effective rank of the messages entering the dummy was approximately:

| Layer | Effective rank |
| ---: | ---: |
| 0 | 4.7 |
| 1 | 13.0 |
| 2 | 17.1 |
| 3 | 10.8 |

This is not complete collapse, but hundreds of node messages occupy a fairly
small effective subspace before being reduced to one vector. The dummy is
behaving as a coarse global summary.

### Runtime-only interventions

These interventions were deliberately out of distribution. They establish
checkpoint reliance, not the playing strength of a retrained alternative.

| Intervention across all layers | Top policy move changed | Median policy KL | Median mean absolute Q change | Median state-value change |
| --- | ---: | ---: | ---: | ---: |
| Mean-normalize incoming dummy aggregation | 72% | 12.08 | 0.169 | 0.334 |
| Remove the dummy star while retaining the branch's per-node transform | 74% | 12.78 | 0.158 | 0.261 |
| Remove the complete global branch | 81% | 14.21 | 0.216 | 0.333 |

The model is strongly co-adapted to the global pathway. That makes removal or
mean normalization useful scientific controls, but not the most promising
strength-seeking experiments.

## Relevant designs from the GNN literature

The closest early precedent is the latent "master node" in Gilmer et al.,
*Neural Message Passing for Quantum Chemistry* (2017). It is connected to all
input nodes with a special edge type and acts as global scratch space read and
written by every node. The paper explicitly allows a different master-node
width, separate update weights, and a recurrent update.

Other common patterns include:

- a bidirectionally connected virtual node whose final state is also the graph
  readout;
- OGB-style virtual state outside the ordinary edge set: add the virtual
  embedding to nodes, locally update nodes, sum-pool nodes back into the
  virtual state through a separate normalized residual MLP, and stop virtual
  updates before the final unusable collection;
- an explicit graph-level state updated separately from nodes and edges, as in
  the Graph Network formulation;
- multiple learned virtual nodes with learned or probabilistic node-to-virtual
  routing; and
- local message passing paired with a separate global-attention path, as in
  GraphGPS and related sparse graph transformers.

A virtual node reduces graph communication distance to two message-passing
steps. A single virtual node can nevertheless replace a topological bottleneck
with a representational bottleneck: all globally routed information must pass
through one fixed-width state, and every receiver initially gets the same
summary.

Raw sum pooling is common, particularly around GIN, because it preserves
cardinality information that mean pooling discards. Strix's graph-size range is
wide enough that aggregation scale deserves testing, but mean pooling must be
given explicit size information if the comparison is intended to isolate
scale rather than remove cardinality.

## Prioritized strength-seeking experiments

### Experiment 1: preserve axis/global branch identity

This is the highest expected-value experiment under a limited budget.

Current fusion:

```text
node_update([h_i, axis_i + global_i])
```

Proposed fusion:

```text
node_update([h_i, axis_i, global_i])
```

An optional residual gate can be added without forcing the two branches back
through a shared sum:

```text
global_effective_i = (1 + delta_i) * global_i
node_update([h_i, axis_i, global_effective_i])
```

`delta_i` may be a per-layer scalar for the cheapest experiment or a small
per-node function of `[h_i, global_i]`. A residual gate is preferable to a
sigmoid gate for checkpoint grafting because `delta = 0` represents exactly
the existing global scale; a finite sigmoid parameter cannot equal exactly
one.

Hypothesis:

- global information is useful, but adding the branches destroys provenance;
- concatenation allows the update to retain spatial features and selectively
  interpret global features; and
- the model can learn different axis/global mixtures by layer and position.

Cost:

- one additional 128-wide input block in each `node_update` first linear;
- a small parameter and compute increase relative to message construction;
- no additional graph edges or message-passing depth.

Expected outcome:

- highest probability of a strength improvement;
- likely a moderate rather than revolutionary gain;
- lowest architectural and optimization risk of the proposed experiments.

### Experiment 2: four learned global tokens

This is the higher-risk, higher-upside experiment.

Replace the single pooled summary with four learned latent slots:

```text
real-node embeddings
        |
        | learned attention pooling
        v
    [g1, g2, g3, g4]
        |
        | receiver-conditioned attention
        v
each real node receives its own mixture
```

The spatial axis branch should remain separate and should be fused by
concatenation or a learned gate rather than unconditional addition.

Use four tokens initially. Eight or sixteen would confound the basic bandwidth
test with a much larger architecture.

Hypothesis:

- one global vector is insufficient to simultaneously represent races,
  threats, tempo, formation and other strategic summaries;
- multiple tokens can specialize without hand-assigning semantic roles; and
- receiver-conditioned mixtures allow different candidate actions to retrieve
  different global context.

Cost:

- approximately `O(4N)` attention/pooling work rather than full `O(N^2)`
  global attention;
- more implementation risk and more new parameters than Experiment 1;
- potentially greater sample requirements.

Expected outcome:

- best chance of a marked gain if the single-vector bottleneck is genuinely
  limiting strength;
- lower probability of success than Experiment 1, but higher upside.

### Experiment 3: global-path stochastic depth

Apply whole-path dropout during training after adopting branch-preserving
fusion:

```text
75-80% of training examples: global path enabled
20-25% of training examples: global contribution disabled
inference: global path always enabled
```

Drop the whole global contribution per position or batch. Do not randomly drop
individual star edges, which would change the meaning of the pooled board in a
hard-to-interpret way.

Hypothesis:

- the global route is useful but may be the easiest path for the optimizer;
- occasional removal forces the spatial representation to remain independently
  useful; and
- normal inference can benefit from both robust spatial features and global
  context.

Expected outcome:

- more likely to improve robustness or provide a modest strength gain than a
  dramatic improvement;
- cheap enough to test after Experiment 1.

## Useful controls, but lower expected strength upside

### Spatial-only

Remove the dummy and complete global branch. This measures the total value of
global communication. Given the intervention sensitivity and the problem's
long-range dependencies, the prior is that this will weaken play.

### Metadata-only

Retain genuinely global rule state, such as move/turn counters, but prevent the
dummy from aggregating board-node embeddings. This distinguishes required game
state from learned global-board shortcuts.

### Mean plus explicit size

Use:

```text
content = mean_i(message(h_i))
size = normalized node count or log(1 + node count)
dummy_update = MLP(dummy, content, size)
```

Do not test mean alone as the primary comparison: that removes both scale and
cardinality information. This experiment cleanly asks whether the model needs
content, size, or the raw magnitude coupling between them.

This is scientifically valuable but has lower expected playing-strength upside
than improved fusion.

### Post-pool normalization

Apply LayerNorm or RMSNorm to the pooled dummy input before its global MLP. This
keeps the direction of the raw sum while limiting its size-dependent magnitude.
It is a more conservative scale intervention than mean aggregation, but the
current pre-normalized residual stack already mitigates much pure scale growth.

### Head-only global context

Run a fully spatial trunk, pool a global summary after local processing, and
score each legal action with:

```text
policy_or_q_head([local_action_embedding_i, global_embedding])
```

This supplies global context without broadcasting it through the spatial
trunk. It is conceptually clean, but removing global communication during
representation learning is a stronger and riskier change than branch-preserving
fusion.

### Reduced global schedule

Try one gather/broadcast exchange followed by local refinement, or global
communication every second layer. This removes the dead final collection and
reduces shortcut capacity, but may also weaken genuinely useful long-range
reasoning.

## Checkpoint grafting

Checkpoint grafting is useful for cheap screening, but it does not replace a
matched from-scratch experiment. The inherited representation is co-adapted to
the original raw-sum dummy, so a grafted continuation measures whether the new
path can be adopted from that solution, not whether its inductive bias learns a
better solution from the beginning.

### Exact graft: concatenated axis/global fusion

The existing first `node_update` linear consumes 256 features:

```text
[h, axis + global]
```

Split its weight matrix into:

```text
W_old = [W_h, W_aggregate]
```

The new first linear consumes 384 features:

```text
[h, axis, global]
```

Initialize it as:

```text
W_new = [W_h, W_aggregate, W_aggregate]
b_new = b_old
```

Then:

```text
W_new [h, axis, global]
    = W_h h + W_aggregate axis + W_aggregate global
    = W_h h + W_aggregate (axis + global)
```

The graft is exactly function-preserving at initialization, subject only to
normal floating-point execution differences. All later `node_update` weights
and the remainder of the checkpoint load unchanged.

If a residual gate is added, use:

```text
global_effective = (1 + delta) * global
delta = 0
```

This is also exactly preserving.

Optimizer state is not uniquely preservable after splitting the aggregate
weights: the old moment combined gradients from both branches, while the new
parameters receive separate gradients. Recommended screening choices are:

1. preserve optimizer state everywhere except the expanded first linear and
   zero the new linear's moments; or
2. reset the optimizer for the entire continuation if matching the optimizer
   transition across candidates is more important than immediate continuity.

Blindly duplicating the old Adam moments into both branch slices is possible,
but it incorrectly treats the historical combined gradient as the history of
each separated branch.

### Exact residual graft: additional global tokens

Replacing the existing dummy outright with attention-pooled tokens cannot be
made exactly equivalent in a natural way. Softmax attention cannot express an
exact finite-parameter one-hot route, and the old raw-sum recurrent update is
not identical to token attention.

For checkpoint screening, keep the old global path and introduce the new token
path as a zero-initialized residual:

```text
global_effective = global_old + P(token_context)
P.weight = 0
P.bias = 0
```

Alternatively, extend concatenated fusion:

```text
node_update([h, axis, global_old, token_context])
```

and initialize the token-context input weights to zero. Both approaches are
exactly output-preserving while allowing gradients to train the new path.

This tests whether extra global bandwidth can improve an existing solution.
The clean from-scratch ablation should replace the single dummy with four
tokens rather than retaining both indefinitely.

### Exact removal of dead final collection

The final layer's real-to-dummy reduction can be skipped without changing
policy or Q outputs, provided that:

- final-layer dummy-to-real broadcasts are retained;
- the global branch's ordinary-node transformations are retained; and
- only the final new dummy state, which no consumer reads, is omitted.

This is an inference/training efficiency cleanup, not a strength experiment.

### Changes that are not checkpoint-equivalent

The following alter the checkpoint's function immediately and should be treated
as fine-tuning probes or trained from scratch:

- mean aggregation;
- post-pool normalization;
- removal of the dummy/global branch;
- metadata-only operation;
- head-only global context;
- reduced global schedules; and
- stochastic global-path dropout during training.

## Recommended allocation under limited compute

If only one full from-scratch run is affordable:

1. run branch-preserving concatenated fusion.

If two are affordable:

1. run branch-preserving concatenated fusion; and
2. run four attention-pooled global tokens with separate fusion.

If a third is affordable:

3. add 20-25% whole-global-path stochastic depth to the concatenated-fusion
   model.

Before committing full compute, use short checkpoint-grafted continuations to
screen optimization stability and whether the new parameters are adopted. Do
not promote a graft solely because its training losses look healthy.

## Evaluation contract

Every candidate should use the same:

- starting distribution and random seeds;
- collected positions or equivalent positions per iteration;
- policy/Q objectives and loss weighting;
- search configuration;
- training position and optimizer-step budget; and
- evaluation openings and colour/side pairing.

Primary evidence should be paired playing-strength evaluation against fixed
checkpoints. Stop or quarantine a candidate if repeated fixed-opponent paired
evaluations regress.

Secondary diagnostics should include:

- strength and calibration bucketed by graph size and game phase;
- global-output/axis-output ratios per layer;
- learned gate values or token-attention usage;
- dummy/token effective rank and variance;
- policy/Q divergence with the global path disabled at inference; and
- tactical or human-corpus measures where available.

Policy and Q losses alone are not evidence that the representation has become
stronger.

## Bottom line

The checkpoint does not show a collapsed or universally dominant dummy node.
It shows a useful, heavily adopted global pathway whose single-vector,
raw-summed representation may still limit or distort learning.

The safest likely improvement is to stop adding spatial and global branch
outputs before the node update. Preserve them separately and let the update
learn how to combine them. The highest-upside extension is four learned global
tokens with receiver-conditioned retrieval. Mean normalization, dummy removal,
and reduced communication remain valuable controls, but are not the best uses
of a very small full-training budget when the immediate goal is marked playing
strength.
