//! Compact all-defense certificates for PDS-PN wins.
//!
//! The search contributes only the set of positions it proved (`pn == 0`).
//! Reconstruction walks the shared forcing kernel from a fresh root: an OR node
//! records every attacker action already proved and retained by the normal search,
//! ranked by worst-case attacker turns, while an AND node records every exact
//! non-losing cover. The verifier is search-independent: it replays the board and
//! recomputes terminal status, attacker moves, depths, ranks, and the complete
//! cover set. It never trusts proof numbers, TT entries, or branch counts from the
//! search.

use super::io::Position;
use super::kernel::{AndEval, KernelCtx, Node, OrEval};
use super::pn::node_key;
use crate::forcing::CellSet2;
use hexo_engine::types::{Coord, Player};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

pub const CERTIFICATE_VERSION: u32 = 1;
const DEFAULT_VERIFY_NODE_LIMIT: usize = 5_000_000;

/// One action edge in the proof DAG.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofResponse {
    pub action: Vec<Coord>,
    pub child: u32,
}

/// A compact proof node. Node IDs are indices into [`ProofCertificate::nodes`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProofNode {
    /// The attacker completes a six within the placements remaining this turn.
    ImmediateWin { action: Vec<Coord> },
    /// Proved forcing attacker actions (OR choices), shortest verified bound
    /// first. `action`/`child` remain the primary choice for compatibility with
    /// version-1 certificates emitted before alternatives were retained.
    AttackerMove {
        action: Vec<Coord>,
        child: u32,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        alternatives: Vec<ProofResponse>,
    },
    /// Every exact non-losing two-cell defender cover (AND branches).
    DefenderReplies { responses: Vec<ProofResponse> },
    /// The attacker has a threat family with transversal number at least three;
    /// no two defender placements can stop the completion next attacker turn.
    Unstoppable,
}

/// Self-describing proof graph. The root position is supplied separately so a
/// report and a browser download can share the same certificate representation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofCertificate {
    pub version: u32,
    pub width: String,
    pub root: u32,
    pub nodes: Vec<ProofNode>,
}

impl ProofCertificate {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("proof certificate serializes")
    }
}

/// Measurements computed by the independent replay verifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofSummary {
    pub dag_nodes: u64,
    pub proof_edges: u64,
    /// Worst-case attacker turns after choosing the shortest ranked alternative
    /// retained at every attacker node. This is a bound for the proof DAG, not a
    /// claim that the mate is globally shortest.
    pub max_attacker_turns: u32,
}

/// Reconstruct an all-defense DAG from PDS-PN's monotone set of proved node keys.
pub(crate) fn reconstruct(
    pos: &Position,
    wide: bool,
    proven: &FxHashSet<u64>,
) -> Result<ProofCertificate, String> {
    let k = KernelCtx::new_wide(
        &pos.stones,
        pos.attacker,
        pos.config.win_length,
        pos.config.placement_radius,
        wide,
    )
    .ok_or("could not rebuild the proof kernel")?;
    let root_node = Node::Or {
        placements: pos.placements_remaining,
    };
    let root_key = node_key(k.hash(), root_node);
    if !proven.contains(&root_key) {
        return Err("PDS-PN proved the root but did not retain its proof key".to_string());
    }
    let mut builder = ProofBuilder {
        k,
        proven,
        memo: FxHashMap::default(),
        nodes: Vec::new(),
        depths: Vec::new(),
    };
    let root = builder.emit(root_node)?;
    Ok(ProofCertificate {
        version: CERTIFICATE_VERSION,
        width: if wide { "wide" } else { "tight" }.to_string(),
        root,
        nodes: builder.nodes,
    })
}

struct ProofBuilder<'a> {
    k: KernelCtx,
    proven: &'a FxHashSet<u64>,
    memo: FxHashMap<u64, u32>,
    nodes: Vec<ProofNode>,
    depths: Vec<u32>,
}

impl ProofBuilder<'_> {
    fn emit(&mut self, node: Node) -> Result<u32, String> {
        let key = node_key(self.k.hash(), node);
        if let Some(&id) = self.memo.get(&key) {
            return Ok(id);
        }
        if !self.proven.contains(&key) {
            return Err(format!("missing proved node {key:016x}"));
        }

        let (proof_node, depth) = match node {
            Node::Or { placements } => match self.k.or_eval(placements) {
                OrEval::WinNow => {
                    let action = self
                        .k
                        .win_now_cells(placements)
                        .ok_or("win-now node had no completion")?;
                    (
                        ProofNode::ImmediateWin {
                            action: action.cells().to_vec(),
                        },
                        1,
                    )
                }
                OrEval::Loss => return Err("proved OR node re-evaluated as a loss".to_string()),
                OrEval::Moves(moves) => {
                    let mut choices = Vec::new();
                    for (move_index, action) in moves.iter().enumerate() {
                        self.k.place_attacker(action);
                        let child_node = Node::And;
                        let child_key = node_key(self.k.hash(), child_node);
                        let built = self
                            .proven
                            .contains(&child_key)
                            .then(|| self.emit(child_node));
                        self.k.unplace(action);
                        if let Some(built) = built {
                            let child = built?;
                            choices.push((
                                move_index,
                                ProofResponse {
                                    action: action.cells().to_vec(),
                                    child,
                                },
                                1u32.saturating_add(self.depths[child as usize]),
                            ));
                        }
                    }
                    choices.sort_by_key(|(move_index, _, depth)| (*depth, *move_index));
                    let (_, primary, depth) = choices
                        .first()
                        .cloned()
                        .ok_or("proved OR node has no reconstructable proved child")?;
                    let alternatives = choices
                        .into_iter()
                        .skip(1)
                        .map(|(_, response, _)| response)
                        .collect();
                    (
                        ProofNode::AttackerMove {
                            action: primary.action,
                            child: primary.child,
                            alternatives,
                        },
                        depth,
                    )
                }
            },
            Node::And => match self.k.and_eval() {
                AndEval::AttackerWin => (ProofNode::Unstoppable, 1),
                AndEval::Loss => return Err("proved AND node re-evaluated as a loss".to_string()),
                AndEval::Covers(covers) => {
                    let mut responses = Vec::with_capacity(covers.len());
                    let mut depth = 0;
                    for cover in covers {
                        self.k.place_defender(&cover);
                        let child_node = Node::Or { placements: 2 };
                        let child_key = node_key(self.k.hash(), child_node);
                        let built = if self.proven.contains(&child_key) {
                            self.emit(child_node)
                        } else {
                            Err(format!("missing proved defense child {child_key:016x}"))
                        };
                        self.k.unplace(&cover);
                        let child = built?;
                        depth = depth.max(self.depths[child as usize]);
                        responses.push(ProofResponse {
                            action: cover.cells().to_vec(),
                            child,
                        });
                    }
                    (ProofNode::DefenderReplies { responses }, depth)
                }
            },
        };

        let id = u32::try_from(self.nodes.len())
            .map_err(|_| "proof DAG contains more than u32::MAX nodes".to_string())?;
        self.nodes.push(proof_node);
        self.depths.push(depth);
        self.memo.insert(key, id);
        Ok(id)
    }
}

/// Verify a certificate from scratch against the supplied root position.
pub fn verify(pos: &Position, certificate: &ProofCertificate) -> Result<ProofSummary, String> {
    verify_with_limit(pos, certificate, DEFAULT_VERIFY_NODE_LIMIT)
}

pub fn verify_with_limit(
    pos: &Position,
    certificate: &ProofCertificate,
    node_limit: usize,
) -> Result<ProofSummary, String> {
    if certificate.version != CERTIFICATE_VERSION {
        return Err(format!(
            "unsupported proof certificate version {}",
            certificate.version
        ));
    }
    if certificate.nodes.is_empty() {
        return Err("proof certificate has no nodes".to_string());
    }
    if certificate.nodes.len() > node_limit {
        return Err(format!(
            "proof certificate has {} nodes, exceeding verifier limit {node_limit}",
            certificate.nodes.len()
        ));
    }
    let wide = match certificate.width.as_str() {
        "tight" => false,
        "wide" => true,
        other => return Err(format!("invalid proof width {other:?}")),
    };
    let k = KernelCtx::new_wide(
        &pos.stones,
        pos.attacker,
        pos.config.win_length,
        pos.config.placement_radius,
        wide,
    )
    .ok_or("could not build the proof-verification kernel")?;
    let mut verifier = ProofVerifier {
        certificate,
        k,
        seen_states: FxHashMap::default(),
        verified_depths: FxHashMap::default(),
        visiting: FxHashSet::default(),
        reached: FxHashSet::default(),
        edges: 0,
    };
    let root_node = Node::Or {
        placements: pos.placements_remaining,
    };
    let depth = verifier.walk(certificate.root, root_node)?;
    if verifier.reached.len() != certificate.nodes.len() {
        return Err(format!(
            "proof certificate contains {} unreachable node(s)",
            certificate.nodes.len() - verifier.reached.len()
        ));
    }
    Ok(ProofSummary {
        dag_nodes: certificate.nodes.len() as u64,
        proof_edges: verifier.edges,
        max_attacker_turns: depth,
    })
}

struct ProofVerifier<'a> {
    certificate: &'a ProofCertificate,
    k: KernelCtx,
    seen_states: FxHashMap<u32, (Node, Vec<(Coord, Player)>)>,
    verified_depths: FxHashMap<u32, u32>,
    visiting: FxHashSet<u32>,
    reached: FxHashSet<u32>,
    edges: u64,
}

impl ProofVerifier<'_> {
    fn walk(&mut self, id: u32, expected: Node) -> Result<u32, String> {
        let proof = self
            .certificate
            .nodes
            .get(id as usize)
            .ok_or_else(|| format!("proof node {id} is out of range"))?
            .clone();
        let state = self.k.canonical_stones();
        if let Some((prior_node, prior_state)) = self.seen_states.get(&id) {
            if *prior_node != expected || *prior_state != state {
                return Err(format!(
                    "proof node {id} is reused for a different position"
                ));
            }
            return self
                .verified_depths
                .get(&id)
                .copied()
                .ok_or_else(|| format!("proof DAG contains a cycle through node {id}"));
        }
        if !self.visiting.insert(id) {
            return Err(format!("proof DAG contains a cycle through node {id}"));
        }
        self.seen_states.insert(id, (expected, state));
        self.reached.insert(id);

        let depth = match (expected, proof) {
            (Node::Or { placements }, ProofNode::ImmediateWin { action }) => {
                let claimed = parse_action(&action)?;
                let actual = self
                    .k
                    .win_now_cells(placements)
                    .ok_or_else(|| format!("node {id}: immediate-win claim is false"))?;
                if claimed != actual {
                    return Err(format!(
                        "node {id}: immediate-win action is not the recomputed completion"
                    ));
                }
                1
            }
            (
                Node::Or { placements },
                ProofNode::AttackerMove {
                    action,
                    child,
                    alternatives,
                },
            ) => {
                let moves = match self.k.or_eval(placements) {
                    OrEval::Moves(moves) => moves,
                    OrEval::WinNow => {
                        return Err(format!(
                            "node {id}: used a forcing move where an immediate win exists"
                        ));
                    }
                    OrEval::Loss => {
                        return Err(format!("node {id}: attacker node has no forcing move"));
                    }
                };
                let mut choices = Vec::with_capacity(1 + alternatives.len());
                choices.push(ProofResponse { action, child });
                choices.extend(alternatives);
                let mut supplied = FxHashSet::default();
                let mut previous_depth = None;
                let mut primary_depth = None;
                for response in choices {
                    let action = parse_action(&response.action)?;
                    if !supplied.insert(action) {
                        return Err(format!("node {id}: duplicate attacker alternative"));
                    }
                    if !moves.contains(&action) {
                        return Err(format!(
                            "node {id}: attacker action is not generated by the certified width"
                        ));
                    }
                    self.k.place_attacker(&action);
                    self.edges += 1;
                    let child_depth = self.walk(response.child, Node::And);
                    self.k.unplace(&action);
                    let choice_depth = 1u32.saturating_add(child_depth?);
                    if previous_depth.is_some_and(|prior| prior > choice_depth) {
                        return Err(format!(
                            "node {id}: attacker alternatives are not ranked by worst-case depth"
                        ));
                    }
                    previous_depth = Some(choice_depth);
                    primary_depth.get_or_insert(choice_depth);
                }
                primary_depth.ok_or_else(|| format!("node {id}: attacker node has no choice"))?
            }
            (Node::And, ProofNode::DefenderReplies { responses }) => {
                let covers = match self.k.and_eval() {
                    AndEval::Covers(covers) => covers,
                    AndEval::AttackerWin => {
                        return Err(format!(
                            "node {id}: enumerated replies to an unstoppable fork"
                        ));
                    }
                    AndEval::Loss => {
                        return Err(format!("node {id}: defender refutes the claimed proof"));
                    }
                };
                let mut supplied = FxHashMap::default();
                for response in responses {
                    let action = parse_action(&response.action)?;
                    if supplied.insert(action, response.child).is_some() {
                        return Err(format!("node {id}: duplicate defender response"));
                    }
                }
                if supplied.len() != covers.len()
                    || covers.iter().any(|cover| !supplied.contains_key(cover))
                {
                    return Err(format!(
                        "node {id}: defender response set is incomplete or has extras"
                    ));
                }
                let mut max_depth = 0;
                for cover in covers {
                    let child = supplied[&cover];
                    self.k.place_defender(&cover);
                    self.edges += 1;
                    let child_depth = self.walk(child, Node::Or { placements: 2 });
                    self.k.unplace(&cover);
                    max_depth = max_depth.max(child_depth?);
                }
                max_depth
            }
            (Node::And, ProofNode::Unstoppable) => match self.k.and_eval() {
                AndEval::AttackerWin => 1,
                _ => return Err(format!("node {id}: unstoppable-fork claim is false")),
            },
            (Node::Or { .. }, _) => {
                return Err(format!("node {id}: defender-shaped proof at attacker node"));
            }
            (Node::And, _) => {
                return Err(format!("node {id}: attacker-shaped proof at defender node"));
            }
        };

        self.visiting.remove(&id);
        self.verified_depths.insert(id, depth);
        Ok(depth)
    }
}

fn parse_action(cells: &[Coord]) -> Result<CellSet2, String> {
    if cells.is_empty() || cells.len() > 2 {
        return Err("proof action must contain one or two cells".to_string());
    }
    if cells.len() == 2 && cells[0] == cells[1] {
        return Err("proof action repeats a cell".to_string());
    }
    Ok(CellSet2::from_cells(cells))
}

#[cfg(test)]
mod tests {
    use super::ProofNode;

    #[test]
    fn legacy_attacker_node_defaults_to_no_alternatives() {
        let node: ProofNode =
            serde_json::from_str(r#"{"kind":"attacker_move","action":[[1,2],[3,4]],"child":7}"#)
                .expect("legacy version-1 attacker node must deserialize");
        match node {
            ProofNode::AttackerMove { alternatives, .. } => assert!(alternatives.is_empty()),
            _ => panic!("wrong proof-node shape"),
        }
    }
}
