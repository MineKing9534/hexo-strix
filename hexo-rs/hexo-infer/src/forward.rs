//! The HeXONet forward pass over `AxisGraphData`, ported from the eager PyTorch
//! oracle (`hexo-a0/src/hexo_a0/model.py`, parity-pinned by tests/parity.rs).
//!
//! Spec (all f32, LayerNorm eps 1e-5):
//!   x = input_proj(X);  e = edge_proj(A)   (edge projection ONCE, shared by layers)
//!   per layer (pre-norm residual):
//!     h    = LN(x)
//!     m_k  = relu(h[src_k] + lin_i(e_k))         GINE message
//!     agg  = Σ_{k: dst_k=v} m_k                  sum at edge DST
//!     z    = (1+eps_i)*h + agg                   (eps ≡ 0 in practice: no train_eps)
//!     c    = nn2_i(relu(nn0_i(z)))
//!     x    = relu(c + x)                         residual add THEN relu
//!   rep = JK-cat: concat_i LN_final(h_i)  |  no-JK: LN_final(x)
//!   policy: per-legal-node MLP -> logits (legal graph order == legal_moves() order)
//!   value: mean-pool rep over stone nodes (zero stones -> zeros), MLP, tanh

use crate::ops;
use crate::weights::{InferConfig, InferError, ModelWeights};
use hexo_rs::axis_graph::AxisGraphData;

pub struct InferModel {
    pub weights: ModelWeights,
}

impl InferModel {
    pub fn from_safetensors(bytes: &[u8]) -> Result<InferModel, InferError> {
        Ok(InferModel { weights: ModelWeights::from_safetensors(bytes)? })
    }

    pub fn config(&self) -> &InferConfig {
        &self.weights.config
    }

    pub fn source_checkpoint(&self) -> &str {
        self.weights.source_checkpoint()
    }

    /// The representation-network math: input_proj → per-layer
    /// norm/edge_proj/lin/aggregation/nn0/nn2 → JK-cat (or final_norm) → `rep`.
    /// Identical on a single graph or a collated disjoint batch (message
    /// passing never crosses node-index boundaries as long as `edge_src`/
    /// `edge_dst` only reference nodes within their own graph, which holds
    /// for both a single `AxisGraphData` and an offset-collated `AxisBatch`).
    /// Shared verbatim by `forward` and `forward_batch` so there is exactly
    /// one copy of the layer math; the `forward_bits_fingerprint_pinned` test
    /// guards that this extraction stays bit-identical for the single-graph
    /// path.
    fn representation(
        &self,
        n: usize,
        e: usize,
        features: &[f32],
        edge_attr: &[f32],
        edge_src: impl Fn(usize) -> usize,
        edge_dst: impl Fn(usize) -> usize,
    ) -> (Vec<f32>, usize) {
        let w = &self.weights;
        let cfg = &w.config;
        let h = cfg.hidden_dim;
        debug_assert_eq!(features.len(), n * cfg.node_dim, "node feature stride");
        debug_assert_eq!(edge_attr.len(), e * 5, "edge attr stride");

        // Projections
        let mut x = Vec::new();
        ops::linear(features, n, &w.input_proj_w, &w.input_proj_b, cfg.node_dim, h, &mut x);

        // Edge-attr dedupe: edge features are structural (axis one-hot ×
        // signed distance × window feature), so the E edges carry only
        // O(100) distinct rows (91 on a real r=8 midgame vs ~13k edges),
        // while the per-layer edge transform below is ~85% of forward FLOPs.
        // Run edge_proj and each layer's lin over UNIQUE rows and gather by
        // index per edge — bitwise identical to the per-edge computation
        // (`linear` is row-pure; pinned by
        // unique_row_gather_matches_per_row_linear_bitwise and the forward
        // fingerprint test).
        let mut row_index: rustc_hash::FxHashMap<[u32; 5], u32> =
            rustc_hash::FxHashMap::default();
        let mut unique_attr: Vec<f32> = Vec::new();
        let mut edge_row: Vec<u32> = Vec::with_capacity(e);
        for k in 0..e {
            let a = &edge_attr[k * 5..(k + 1) * 5];
            let key = [a[0].to_bits(), a[1].to_bits(), a[2].to_bits(), a[3].to_bits(), a[4].to_bits()];
            let next = (unique_attr.len() / 5) as u32;
            let idx = *row_index.entry(key).or_insert_with(|| {
                unique_attr.extend_from_slice(a);
                next
            });
            edge_row.push(idx);
        }
        let u = unique_attr.len() / 5;
        let mut eproj = Vec::new();
        ops::linear(&unique_attr, u, &w.edge_proj_w, &w.edge_proj_b, 5, h, &mut eproj);

        // Layers
        let mut hs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_layers);
        let mut normed = Vec::new();
        let mut lin_e = Vec::new();
        let mut z = vec![0.0f32; n * h];
        let mut c1 = Vec::new();
        let mut c2 = Vec::new();
        for layer in &w.layers {
            ops::layer_norm(&x, n, h, &layer.norm_w, &layer.norm_b, &mut normed);
            // Over unique edge-attr rows (u ≪ e), gathered via edge_row below.
            ops::linear(&eproj, u, &layer.lin_w, &layer.lin_b, h, h, &mut lin_e);
            // agg[dst] += relu(normed[src] + lin_e[k])  — accumulate straight into z
            // after seeding z with (1+eps)*normed.
            let scale = 1.0 + layer.eps;
            for i in 0..n * h {
                z[i] = scale * normed[i];
            }
            for k in 0..e {
                let s = edge_src(k);
                let d = edge_dst(k);
                let nk = &normed[s * h..(s + 1) * h];
                let er = edge_row[k] as usize;
                let ek = &lin_e[er * h..(er + 1) * h];
                let zd = &mut z[d * h..(d + 1) * h];
                for j in 0..h {
                    let m = nk[j] + ek[j];
                    if m > 0.0 {
                        zd[j] += m;
                    }
                }
            }
            ops::linear(&z, n, &layer.nn0_w, &layer.nn0_b, h, h, &mut c1);
            ops::relu_inplace(&mut c1);
            ops::linear(&c1, n, &layer.nn2_w, &layer.nn2_b, h, h, &mut c2);
            for i in 0..n * h {
                x[i] = (c2[i] + x[i]).max(0.0); // residual add THEN relu (pre-norm branch)
            }
            if cfg.use_jk_cat {
                hs.push(x.clone());
            }
        }

        // Representation: JK-cat of final_norm(h_i) in layer order, or final_norm(x).
        let d = if cfg.use_jk_cat { cfg.num_layers * h } else { h };
        let mut rep = vec![0.0f32; n * d];
        if cfg.use_jk_cat {
            for (li, hi) in hs.iter().enumerate() {
                ops::layer_norm(hi, n, h, &w.final_norm_w, &w.final_norm_b, &mut normed);
                for node in 0..n {
                    rep[node * d + li * h..node * d + (li + 1) * h]
                        .copy_from_slice(&normed[node * h..(node + 1) * h]);
                }
            }
        } else {
            ops::layer_norm(&x, n, h, &w.final_norm_w, &w.final_norm_b, &mut rep);
        }

        (rep, d)
    }

    /// The axis-relational representation (KLENT lean schema): per-layer
    /// `AxisRelationalConv` with 3 tied axis branches (scatter-add into
    /// (N,3,H) buckets, symmetric SUM) + an untied global "star" branch
    /// (dummy node ↔ all real nodes), then `node_update` + residual.
    ///
    /// Mirrors `hexo-a0/src/hexo_a0/axis_conv.py` (`separate_global_star=True`):
    ///   dist_table = axis_lin(dist_embed)                       (window, H)
    ///   messages   = relu(normed[src] + dist_table[edge_dist-1])
    ///   buckets[dst, edge_type] += messages                     (N, 3, H)
    ///   axis_inputs = (1+axis_eps)*normed + buckets[:, :3]
    ///   axis_hidden = relu(axis_nn0(axis_inputs.reshape(N*3, H)))
    ///   agg = axis_nn2(axis_hidden.view(N,3,H).sum(1)) + 3*axis_nn2_bias
    ///   global_edge = global_lin(global_edge_embed)
    ///   global_bucket[real] = relu(normed[dummy] + global_edge)   (index_copy_)
    ///   global_bucket[dummy] += relu(normed[real] + global_edge)  (index_add_)
    ///   agg += global_nn2(relu(global_nn0((1+global_eps)*normed + global_bucket)))
    ///   x = relu(node_update2(relu(node_update0(cat([normed, agg])))) + x)
    fn representation_relational(&self, g: &AxisGraphData) -> (Vec<f32>, usize) {
        let w = &self.weights;
        let cfg = &w.config;
        let h = cfg.hidden_dim;
        let n = g.num_nodes;
        let e = g.edge_src.len();
        const NUM_AXES: usize = 3;
        let window = cfg.axis_window;
        debug_assert_eq!(g.features.len(), n * cfg.node_dim, "node feature stride");
        debug_assert_eq!(g.edge_type.len(), e, "edge_type stride");
        debug_assert_eq!(g.edge_dist.len(), e, "edge_dist stride");

        let mut x = Vec::new();
        ops::linear(&g.features, n, &w.input_proj_w, &w.input_proj_b, cfg.node_dim, h, &mut x);

        let mut hs: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_layers);
        let mut normed = Vec::new();
        for layer in &w.rel_layers {
            ops::layer_norm(&x, n, h, &layer.norm_w, &layer.norm_b, &mut normed);

            // dist_table = axis_lin(dist_embed)  (window, H)
            let mut dist_table = Vec::new();
            ops::linear(&layer.dist_embed, window, &layer.axis_lin_w, &layer.axis_lin_b, h, h, &mut dist_table);

            // messages = relu(normed[src] + dist_table[edge_dist-1]); scatter-add
            // into buckets[dst*3 + edge_type].
            let mut buckets = vec![0.0f32; n * NUM_AXES * h];
            for k in 0..e {
                let s = g.edge_src[k] as usize;
                let d = g.edge_dst[k] as usize;
                let t = g.edge_type[k] as usize;
                let row = g.edge_dist[k] as usize - 1;
                let nb = &normed[s * h..(s + 1) * h];
                let dt = &dist_table[row * h..(row + 1) * h];
                let bucket = &mut buckets[(d * NUM_AXES + t) * h..(d * NUM_AXES + t + 1) * h];
                for j in 0..h {
                    let m = nb[j] + dt[j];
                    if m > 0.0 {
                        bucket[j] += m;
                    }
                }
            }

            // axis_inputs = (1+eps)*normed + buckets[:, :3]
            // axis_hidden = relu(axis_nn0(axis_inputs.reshape(N*3, H)))
            // axis_hidden_sum = axis_hidden.view(N,3,H).sum(dim=1)
            // agg = axis_nn2(axis_hidden_sum) + 3*axis_nn2_bias
            let scale = 1.0 + layer.axis_eps;
            let mut axis_inputs = vec![0.0f32; n * NUM_AXES * h];
            for i in 0..n {
                for a in 0..NUM_AXES {
                    for j in 0..h {
                        axis_inputs[(i * NUM_AXES + a) * h + j] =
                            scale * normed[i * h + j] + buckets[(i * NUM_AXES + a) * h + j];
                    }
                }
            }
            let mut axis_hidden = Vec::new();
            ops::linear(&axis_inputs, n * NUM_AXES, &layer.axis_nn0_w, &layer.axis_nn0_b, h, h, &mut axis_hidden);
            ops::relu_inplace(&mut axis_hidden);
            let mut axis_hidden_sum = vec![0.0f32; n * h];
            for i in 0..n {
                for a in 0..NUM_AXES {
                    for j in 0..h {
                        axis_hidden_sum[i * h + j] += axis_hidden[(i * NUM_AXES + a) * h + j];
                    }
                }
            }
            let mut agg = Vec::new();
            ops::linear(&axis_hidden_sum, n, &layer.axis_nn2_w, &layer.axis_nn2_b, h, h, &mut agg);
            // F.linear(..., bias * num_axes): `linear` already added 1×bias, so add
            // the remaining (num_axes-1)×.
            for i in 0..n {
                for j in 0..h {
                    agg[i * h + j] += (NUM_AXES as f32 - 1.0) * layer.axis_nn2_b[j];
                }
            }

            // Global star branch: dummy node ↔ all real nodes.
            // global_edge = global_lin(global_edge_embed)  (H,)
            let mut global_edge = Vec::new();
            ops::linear(&layer.global_edge_embed, 1, &layer.global_lin_w, &layer.global_lin_b, h, h, &mut global_edge);

            let dummy = n - 1;
            let ng = g.global_edge_src.len();
            let mut global_bucket = vec![0.0f32; n * h];
            // dummy -> real (even indices): assign relu(normed[dummy] + global_edge)
            for k in (0..ng).step_by(2) {
                let real = g.global_edge_dst[k] as usize;
                let nb = &normed[dummy * h..(dummy + 1) * h];
                let gb = &mut global_bucket[real * h..(real + 1) * h];
                for j in 0..h {
                    let m = nb[j] + global_edge[j];
                    gb[j] = if m > 0.0 { m } else { 0.0 };
                }
            }
            // real -> dummy (odd indices): accumulate relu(normed[real] + global_edge)
            for k in (1..ng).step_by(2) {
                let real = g.global_edge_src[k] as usize;
                let nb = &normed[real * h..(real + 1) * h];
                let gb = &mut global_bucket[dummy * h..(dummy + 1) * h];
                for j in 0..h {
                    let m = nb[j] + global_edge[j];
                    if m > 0.0 {
                        gb[j] += m;
                    }
                }
            }

            // global_inputs = (1+global_eps)*normed + global_bucket
            // agg += global_nn2(relu(global_nn0(global_inputs)))
            let gscale = 1.0 + layer.global_eps;
            let mut global_inputs = vec![0.0f32; n * h];
            for i in 0..n {
                for j in 0..h {
                    global_inputs[i * h + j] = gscale * normed[i * h + j] + global_bucket[i * h + j];
                }
            }
            let mut g1 = Vec::new();
            ops::linear(&global_inputs, n, &layer.global_nn0_w, &layer.global_nn0_b, h, h, &mut g1);
            ops::relu_inplace(&mut g1);
            let mut g2 = Vec::new();
            ops::linear(&g1, n, &layer.global_nn2_w, &layer.global_nn2_b, h, h, &mut g2);
            for i in 0..n * h {
                agg[i] += g2[i];
            }

            // conv_out = node_update2(relu(node_update0(cat([normed, agg]))))
            let mut cat = vec![0.0f32; n * 2 * h];
            for i in 0..n {
                cat[i * 2 * h..i * 2 * h + h].copy_from_slice(&normed[i * h..(i + 1) * h]);
                cat[i * 2 * h + h..(i + 1) * 2 * h].copy_from_slice(&agg[i * h..(i + 1) * h]);
            }
            let mut nu1 = Vec::new();
            ops::linear(&cat, n, &layer.node_update0_w, &layer.node_update0_b, 2 * h, h, &mut nu1);
            ops::relu_inplace(&mut nu1);
            let mut nu2 = Vec::new();
            ops::linear(&nu1, n, &layer.node_update2_w, &layer.node_update2_b, h, h, &mut nu2);

            // x = relu(nu2 + x)
            for i in 0..n * h {
                x[i] = (nu2[i] + x[i]).max(0.0);
            }
            if cfg.use_jk_cat {
                hs.push(x.clone());
            }
        }

        // Representation: JK-cat of final_norm(h_i) in layer order, or final_norm(x).
        let d = if cfg.use_jk_cat { cfg.num_layers * h } else { h };
        let mut rep = vec![0.0f32; n * d];
        if cfg.use_jk_cat {
            for (li, hi) in hs.iter().enumerate() {
                ops::layer_norm(hi, n, h, &w.final_norm_w, &w.final_norm_b, &mut normed);
                for node in 0..n {
                    rep[node * d + li * h..node * d + (li + 1) * h]
                        .copy_from_slice(&normed[node * h..(node + 1) * h]);
                }
            }
        } else {
            ops::layer_norm(&x, n, h, &w.final_norm_w, &w.final_norm_b, &mut rep);
        }

        (rep, d)
    }

    /// Forward pass: (logits in legal_moves() order, value in [-1,1] from to_move's view).
    pub fn forward(&self, g: &AxisGraphData) -> (Vec<f32>, f32) {
        let w = &self.weights;
        let cfg = &w.config;
        let n = g.num_nodes;
        let (rep, d) = if cfg.axis_relational {
            self.representation_relational(g)
        } else {
            let e = g.edge_src.len();
            self.representation(
                n,
                e,
                &g.features,
                &g.edge_attr,
                |k| g.edge_src[k] as usize,
                |k| g.edge_dst[k] as usize,
            )
        };

        // Policy head: gather legal rows (graph order == legal_moves() order), MLP -> 1.
        let legal_rows: Vec<usize> = (0..n).filter(|&i| g.legal_mask[i]).collect();
        let nl = legal_rows.len();
        let mut legal_rep = vec![0.0f32; nl * d];
        for (r, &node) in legal_rows.iter().enumerate() {
            legal_rep[r * d..(r + 1) * d].copy_from_slice(&rep[node * d..(node + 1) * d]);
        }
        let mut p1 = Vec::new();
        ops::linear(&legal_rep, nl, &w.policy0_w, &w.policy0_b, d, cfg.policy_hidden, &mut p1);
        ops::relu_inplace(&mut p1);
        let mut logits = Vec::new();
        ops::linear(&p1, nl, &w.policy2_w, &w.policy2_b, cfg.policy_hidden, 1, &mut logits);

        // Value: KLENT Q-head (per-legal-node MLP -> Q, then
        // V = Σ_a softmax(logits)_a · Q_a) or the standard mean-pooled value head.
        // Discriminator is axis_relational (KLENT), not q_hidden (defaults to 64).
        let value = if cfg.axis_relational {
            let mut q1 = Vec::new();
            ops::linear(&legal_rep, nl, &w.q0_w, &w.q0_b, d, cfg.q_hidden, &mut q1);
            ops::relu_inplace(&mut q1);
            let mut q2 = Vec::new();
            ops::linear(&q1, nl, &w.q2_w, &w.q2_b, cfg.q_hidden, 1, &mut q2);
            // softmax(logits) over legal actions, then V = Σ softmax·tanh(Q).
            let mut max_logit = f32::NEG_INFINITY;
            for &l in &logits {
                if l > max_logit {
                    max_logit = l;
                }
            }
            let mut sum_exp = 0.0f32;
            let mut exps = vec![0.0f32; nl];
            for (i, &l) in logits.iter().enumerate() {
                let e = (l - max_logit).exp();
                exps[i] = e;
                sum_exp += e;
            }
            let mut v = 0.0f32;
            for i in 0..nl {
                v += (exps[i] / sum_exp) * q2[i].tanh();
            }
            v
        } else {
            // Value head: mean-pool rep over STONE nodes; zero stones -> zeros (matches
            // _forward_batch_core's scatter_add + clamp(min=1), NOT mean-over-all).
            let mut pooled = vec![0.0f32; d];
            let n_stones = g.stone_mask.iter().filter(|&&b| b).count();
            if n_stones > 0 {
                for node in 0..n {
                    if g.stone_mask[node] {
                        for j in 0..d {
                            pooled[j] += rep[node * d + j];
                        }
                    }
                }
                let inv = 1.0 / n_stones as f32;
                for v in pooled.iter_mut() {
                    *v *= inv;
                }
            }
            let mut v1 = Vec::new();
            ops::linear(&pooled, 1, &w.value0_w, &w.value0_b, d, cfg.value_hidden, &mut v1);
            ops::relu_inplace(&mut v1);
            let mut v2 = Vec::new();
            ops::linear(&v1, 1, &w.value2_w, &w.value2_b, cfg.value_hidden, 1, &mut v2);
            v2[0].tanh()
        };

        (logits, value)
    }

    /// Serial forward over a collated disjoint-graph batch. Bit-identical to
    /// running `forward` on each constituent graph and concatenating the
    /// results (see `forward_batch_matches_per_graph_loop_bitwise`).
    ///
    /// NOT USED IN PRODUCTION — kept as a reference implementation.
    /// The HX04 server deliberately runs `split_batch` + per-graph `forward`
    /// (cross-graph OS threading) instead of this collated batch forward. An
    /// A/B (2026-07-07) showed the collated approach is 3.5–3.8× SLOWER on CPU
    /// at real self-play batch sizes (~16–64 graphs): collating materializes
    /// one large intermediate buffer set that blows L2 (bandwidth-bound) and,
    /// when parallelized, pays a spawn/join barrier per op — whereas the split
    /// keeps each small graph cache-resident and threads coarsely across
    /// graphs. Collation is the right layout for a GPU (big contiguous SIMD
    /// buffers, high bandwidth), not a bandwidth-limited CPU. This method is
    /// retained as the batched-layout reference for a possible future
    /// Rust-on-GPU forward (e.g. CubeCL: ROCm + wgpu/WebGPU) — the shape here
    /// is what such kernels would consume. Full write-up:
    /// docs/research/2026-07-06-hexo-infer-selfplay-ab.md.
    ///
    /// The rep-net portion is shared verbatim with `forward` via
    /// `representation`. Only the heads differ:
    /// - policy: legal rows gathered over the WHOLE batch (node order, which
    ///   is per-graph-contiguous because collation lays nodes out
    ///   graph-by-graph) and run through the MLP in one call — `linear` is
    ///   row-pure, so this is bitwise equal to per-graph calls.
    /// - value: mean-pooled over stone nodes PER GRAPH (via `node_graph`),
    ///   then the value MLP is run per graph, one row at a time, exactly as
    ///   `forward` does, to guarantee identical `ops::linear` numerics.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn forward_batch(&self, b: &AxisBatch) -> BatchOut {
        let w = &self.weights;
        let cfg = &w.config;
        let n = b.num_nodes;
        let e = b.edge_src.len();
        let (rep, d) = self.representation(
            n,
            e,
            b.features,
            b.edge_attr,
            |k| b.edge_src[k] as usize,
            |k| b.edge_dst[k] as usize,
        );

        // Policy head: legal rows over the whole batch, node order (== each
        // graph's legal_moves() order, concatenated).
        let legal_rows: Vec<usize> = (0..n).filter(|&i| b.legal_mask[i]).collect();
        let nl = legal_rows.len();
        let mut legal_rep = vec![0.0f32; nl * d];
        for (r, &node) in legal_rows.iter().enumerate() {
            legal_rep[r * d..(r + 1) * d].copy_from_slice(&rep[node * d..(node + 1) * d]);
        }
        let mut p1 = Vec::new();
        ops::linear(&legal_rep, nl, &w.policy0_w, &w.policy0_b, d, cfg.policy_hidden, &mut p1);
        ops::relu_inplace(&mut p1);
        let mut logits = Vec::new();
        ops::linear(&p1, nl, &w.policy2_w, &w.policy2_b, cfg.policy_hidden, 1, &mut logits);

        let mut legal_counts = vec![0u32; b.num_graphs];
        for &node in &legal_rows {
            legal_counts[b.node_graph[node] as usize] += 1;
        }

        // Value head: per graph, mean-pool rep over stone nodes (ascending
        // node-index order, matching `forward`'s single-graph accumulation),
        // then run the value MLP one row at a time — NOT batched across
        // graphs — to keep bitwise parity with `forward`.
        let mut values = vec![0.0f32; b.num_graphs];
        for gi in 0..b.num_graphs {
            let mut pooled = vec![0.0f32; d];
            let mut n_stones = 0usize;
            for node in 0..n {
                if b.node_graph[node] as usize == gi && b.stone_mask[node] {
                    n_stones += 1;
                    for j in 0..d {
                        pooled[j] += rep[node * d + j];
                    }
                }
            }
            if n_stones > 0 {
                let inv = 1.0 / n_stones as f32;
                for v in pooled.iter_mut() {
                    *v *= inv;
                }
            }
            let mut v1 = Vec::new();
            ops::linear(&pooled, 1, &w.value0_w, &w.value0_b, d, cfg.value_hidden, &mut v1);
            ops::relu_inplace(&mut v1);
            let mut v2 = Vec::new();
            ops::linear(&v1, 1, &w.value2_w, &w.value2_b, cfg.value_hidden, 1, &mut v2);
            values[gi] = v2[0].tanh();
        }

        BatchOut { logits, legal_counts, values }
    }
}

/// A collated batch of disjoint `AxisGraphData` graphs: node feature/edge
/// arrays concatenated, edge indices offset into the shared node space, plus
/// a `node_graph` array mapping each node back to its originating graph.
#[cfg(not(target_arch = "wasm32"))]
pub struct AxisBatch<'a> {
    pub num_nodes: usize,
    pub num_graphs: usize,
    pub node_dim: usize,
    pub features: &'a [f32],    // num_nodes * node_dim
    pub edge_src: &'a [u32],    // E, collated (node offsets already applied)
    pub edge_dst: &'a [u32],    // E
    pub edge_attr: &'a [f32],   // E * 5
    pub legal_mask: &'a [bool], // num_nodes
    pub stone_mask: &'a [bool], // num_nodes
    pub node_graph: &'a [u32],  // num_nodes -> graph id in 0..num_graphs
}

/// `forward_batch` output: logits/legal_counts/values, all per-graph
/// concatenated (logits) or indexed (legal_counts, values) in batch order.
#[cfg(not(target_arch = "wasm32"))]
pub struct BatchOut {
    pub logits: Vec<f32>,       // concatenated per-graph, each graph's legal nodes in node order
    pub legal_counts: Vec<u32>, // len == num_graphs
    pub values: Vec<f32>,       // len == num_graphs
}

#[cfg(test)]
mod tests {
    use super::*;
    use hexo_rs::axis_graph::game_to_axis_graph_raw_opts;
    use hexo_rs::hexo_engine::{GameConfig, GameState};

    fn tiny_model() -> InferModel {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
        InferModel::from_safetensors(&std::fs::read(p).unwrap()).unwrap()
    }

    /// Bit-level pin of the full forward, captured from the pre-dedupe
    /// per-edge implementation. The edge-attr dedupe must be EXACTLY
    /// behavior-preserving (same input row through the same linear ⇒ same
    /// output row, gathered), so any changed bit here is a bug, not drift.
    #[test]
    fn forward_bits_fingerprint_pinned() {
        let model = tiny_model();
        let mut game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
        for m in [(1, 2), (0, -2), (2, 0), (-1, 1), (3, -1), (0, 3), (-2, 0), (1, -3)] {
            game.apply_move(m).unwrap();
        }
        let cfg = model.config();
        let g = game_to_axis_graph_raw_opts(&game, cfg.prune_empty_edges, cfg.threat_features, cfg.relative_stones);
        let (logits, value) = model.forward(&g);
        let mut fp: u64 = logits.len() as u64;
        for v in logits.iter().chain(std::iter::once(&value)) {
            fp = fp.wrapping_mul(0x100000001b3).wrapping_add(v.to_bits() as u64);
        }
        assert_eq!(fp, 2037059312983329231, "forward output bits changed");
    }

    /// The dedupe's core soundness property, pinned against kernel changes:
    /// `linear` must be row-pure (each output row a function of its input row
    /// only), so unique-rows-then-gather is bitwise equal to per-edge rows.
    #[test]
    fn unique_row_gather_matches_per_row_linear_bitwise() {
        let model = tiny_model();
        let w = &model.weights;
        let mut game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
        for m in [(1, 2), (0, -2), (2, 0), (-1, 1)] {
            game.apply_move(m).unwrap();
        }
        let cfg = model.config();
        let g = game_to_axis_graph_raw_opts(&game, cfg.prune_empty_edges, cfg.threat_features, cfg.relative_stones);
        let e = g.edge_src.len();
        let h = cfg.hidden_dim;
        assert!(e > 0);

        // Per-edge reference.
        let mut full = Vec::new();
        ops::linear(&g.edge_attr, e, &w.edge_proj_w, &w.edge_proj_b, 5, h, &mut full);

        // Unique rows + gather.
        let mut index: rustc_hash::FxHashMap<[u32; 5], u32> = rustc_hash::FxHashMap::default();
        let mut unique: Vec<f32> = Vec::new();
        let mut row_of: Vec<u32> = Vec::with_capacity(e);
        for k in 0..e {
            let a = &g.edge_attr[k * 5..(k + 1) * 5];
            let key = [a[0].to_bits(), a[1].to_bits(), a[2].to_bits(), a[3].to_bits(), a[4].to_bits()];
            let next = (unique.len() / 5) as u32;
            let idx = *index.entry(key).or_insert_with(|| {
                unique.extend_from_slice(a);
                next
            });
            row_of.push(idx);
        }
        let u = unique.len() / 5;
        assert!(u < e, "fixture must contain duplicate edge_attr rows");
        let mut uproj = Vec::new();
        ops::linear(&unique, u, &w.edge_proj_w, &w.edge_proj_b, 5, h, &mut uproj);

        for k in 0..e {
            let a = &full[k * h..(k + 1) * h];
            let b = &uproj[row_of[k] as usize * h..(row_of[k] as usize + 1) * h];
            for j in 0..h {
                assert_eq!(a[j].to_bits(), b[j].to_bits(), "edge {k} dim {j}");
            }
        }
    }

    #[test]
    fn forward_shapes_sane() {
        let model = tiny_model();
        let game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
        let cfg = model.config();
        let g = game_to_axis_graph_raw_opts(&game, cfg.prune_empty_edges, cfg.threat_features, cfg.relative_stones);
        let (logits, value) = model.forward(&g);
        assert_eq!(logits.len(), game.legal_moves().len());
        assert!((-1.0..=1.0).contains(&value));
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    /// >=3 tiny graphs of varying node/edge/legal/stone counts (the first has
    /// zero stones — an empty board) for the `forward_batch` parity test.
    #[cfg(not(target_arch = "wasm32"))]
    fn make_test_graphs(model: &InferModel) -> Vec<AxisGraphData> {
        let cfg = model.config();
        let mk = |moves: &[(i32, i32)]| {
            let mut game = GameState::with_config(GameConfig { win_length: 6, placement_radius: 4, max_moves: 300 });
            for &m in moves {
                game.apply_move(m).unwrap();
            }
            game_to_axis_graph_raw_opts(&game, cfg.prune_empty_edges, cfg.threat_features, cfg.relative_stones)
        };
        vec![
            mk(&[]),                                                    // 0 stones
            mk(&[(1, 2), (0, -2)]),                                     // small
            mk(&[(1, 2), (0, -2), (2, 0), (-1, 1), (3, -1), (0, 3), (-2, 0), (1, -3)]), // larger
        ]
    }

    /// Owned backing storage for a collated `AxisBatch` (the struct itself
    /// only borrows). `as_batch` hands out the borrowing view.
    #[cfg(not(target_arch = "wasm32"))]
    struct CollatedBatch {
        num_nodes: usize,
        num_graphs: usize,
        node_dim: usize,
        features: Vec<f32>,
        edge_src: Vec<u32>,
        edge_dst: Vec<u32>,
        edge_attr: Vec<f32>,
        legal_mask: Vec<bool>,
        stone_mask: Vec<bool>,
        node_graph: Vec<u32>,
    }

    #[cfg(not(target_arch = "wasm32"))]
    impl CollatedBatch {
        fn as_batch(&self) -> AxisBatch<'_> {
            AxisBatch {
                num_nodes: self.num_nodes,
                num_graphs: self.num_graphs,
                node_dim: self.node_dim,
                features: &self.features,
                edge_src: &self.edge_src,
                edge_dst: &self.edge_dst,
                edge_attr: &self.edge_attr,
                legal_mask: &self.legal_mask,
                stone_mask: &self.stone_mask,
                node_graph: &self.node_graph,
            }
        }
    }

    /// Concatenate several `AxisGraphData` into one collated disjoint-graph
    /// batch: node offsets applied to edge indices, `node_graph` filled.
    #[cfg(not(target_arch = "wasm32"))]
    fn collate(graphs: &[AxisGraphData]) -> CollatedBatch {
        let node_dim = graphs[0].features.len() / graphs[0].num_nodes;
        let mut features = Vec::new();
        let mut edge_src = Vec::new();
        let mut edge_dst = Vec::new();
        let mut edge_attr = Vec::new();
        let mut legal_mask = Vec::new();
        let mut stone_mask = Vec::new();
        let mut node_graph = Vec::new();
        let mut offset: u32 = 0;
        for (gi, g) in graphs.iter().enumerate() {
            debug_assert_eq!(g.features.len() / g.num_nodes, node_dim, "node_dim mismatch across graphs");
            features.extend_from_slice(&g.features);
            edge_attr.extend_from_slice(&g.edge_attr);
            legal_mask.extend_from_slice(&g.legal_mask);
            stone_mask.extend_from_slice(&g.stone_mask);
            node_graph.extend(std::iter::repeat(gi as u32).take(g.num_nodes));
            edge_src.extend(g.edge_src.iter().map(|&s| s as u32 + offset));
            edge_dst.extend(g.edge_dst.iter().map(|&d| d as u32 + offset));
            offset += g.num_nodes as u32;
        }
        CollatedBatch {
            num_nodes: offset as usize,
            num_graphs: graphs.len(),
            node_dim,
            features,
            edge_src,
            edge_dst,
            edge_attr,
            legal_mask,
            stone_mask,
            node_graph,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn forward_batch_matches_per_graph_loop_bitwise() {
        let model = tiny_model();
        let graphs: Vec<AxisGraphData> = make_test_graphs(&model);
        // Per-graph oracle:
        let oracle: Vec<(Vec<f32>, f32)> = graphs.iter().map(|g| model.forward(g)).collect();
        // Collate into one AxisBatch:
        let collated = collate(&graphs);
        let batch = collated.as_batch();
        let out = model.forward_batch(&batch);
        assert_eq!(out.legal_counts.len(), graphs.len());
        assert_eq!(out.values.len(), graphs.len());
        // Compare per graph, bitwise.
        let mut off = 0usize;
        for (gi, (want_logits, want_value)) in oracle.iter().enumerate() {
            let lc = out.legal_counts[gi] as usize;
            assert_eq!(lc, want_logits.len(), "graph {gi} legal count");
            let got = &out.logits[off..off + lc];
            for (a, b) in got.iter().zip(want_logits) {
                assert_eq!(a.to_bits(), b.to_bits(), "graph {gi} logit bits");
            }
            assert_eq!(out.values[gi].to_bits(), want_value.to_bits(), "graph {gi} value bits");
            off += lc;
        }
        assert_eq!(off, out.logits.len());
    }
}