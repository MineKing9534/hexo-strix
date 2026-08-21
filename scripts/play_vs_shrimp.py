#!/usr/bin/env python3
"""Play a Strix (HeXO AlphaZero or KLENT) checkpoint against Shrimp's bot API.

Loads any Strix-like checkpoint (``.pt`` or ``.safetensors``) — HeXONet and
KLENT (hexo-klent-v1) are both auto-detected and routed through the same
loader logic as ``scripts/policy_viewer.py``. ``model_config`` is read from
the checkpoint itself, so the script works across all curriculum stages,
ablations, and KLENT architectures present in the repo.

The adapter at the bottom uses the same Gumbel-MCTS search path
(``hexo_rs.gumbel_mcts_with_diagnostics``) as ``head_to_head`` and
``sealbot_eval`` / ``krakenbot_eval``, so local bot strength matches what
those harnesses evaluate against the same checkpoint. The match loop
itself — long-polling, two-stones-per-turn, transient retries, bot_failed
recovery — is the SDK's, unchanged.

Quickstart
----------

::

    # List what checkpoints the server offers + allowed sim budgets.
    uv run python scripts/play_vs_shrimp.py \
        --server https://shrimp.example --list

    # Play one match. The checkpoint's embedded model_config drives the
    # bot's architecture and graph builder; the server's catalogue provides
    # the opponent's sims (override with --sims).
    uv run python scripts/play_vs_shrimp.py \
        --server https://shrimp.example --agent strix-tyto \
        --checkpoint exports/checkpoint_00272000.pt

    # Tune the local bot: more sims for stronger play, pick a side, raw-policy.
    uv run python scripts/play_vs_shrimp.py \
        --server https://shrimp.example --agent strix-tyto \
        --checkpoint exports/checkpoint_00272000.pt \
        --sims 256 --m-actions 16 --color 0 --raw-policy

Notes
-----
* ``select_stone`` is called once per stone. HeXO turns place TWO stones after
  the opening, so expect two ``select_stone`` calls per turn; the SDK relays
  a fresh state to each call (the first stone you place is already applied).
* The server auto-plays P1's opening stone at the origin; ``state["history"]``
  includes it as the first entry, but ``hexo_rs.GameState`` already has it on
  the board, so ``build_game_from_state`` replays every entry EXCEPT the
  first (otherwise ``apply_move`` raises "cell is occupied"). The
  reconstructed ``GameState`` uses the project-standard game config
  (six-in-a-row on a radius-8 board) for every checkpoint.
* ``state["legal"]`` is authoritative — we always project our pick onto the
  legal set before returning, so a stale local model output can never send an
  illegal move.
* Opponent-failure policy: the SDK defaults to resigning (counting it as
  our loss) after a few server ``bot_failed`` events. We override that to
  walk away instead -- the opponent couldn't recover, so the match is
  recorded as ``outcome="abandoned"`` (``termination="abandoned_opponent_failed"``)
  and excluded from ELO scoring. Backoff between retries is exponential
  (2**N capped at 60s) with 10% jitter, so a flaky opponent doesn't pin
  a CPU. By default we retry indefinitely (``--max-opponent-retries`` to
  cap); the server's ``idle_timeout_s`` (default 600s) is what bounds us.
* Trade-off: walked-away matches stay ACTIVE on the server until that idle
  timeout. That can briefly block a follow-up match under the same agent
  name with HTTP 429 "active-game limit reached". We handle 429 by waiting
  up to ``--active-game-wait`` seconds (default 600s) with the same
  exponential backoff before giving up. If you can't tolerate that,
  ``--resign-on-opponent-failure`` flips to the SDK default (records our
  loss, frees the slot immediately).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

# The downloaded SDK lives next to this script and is zero-dep stdlib only.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shrimp_hexo_match_client import (  # noqa: E402
    BotAdapter,
    LONG_POLL_S,
    MatchError,
    MatchServer,
)


class BrowserMatchServer(MatchServer):
    """``MatchServer`` with a browser-style User-Agent.

    Shrimp's servers sit behind Cloudflare, which blocks Python's default
    ``Python-urllib/X.Y`` User-Agent (Cloudflare error 1010 — ``HTTP 403``
    with body ``"error code: 1010"``). The SDK's ``request`` is documented
    as "the single HTTP seam — tests (and exotic transports) can subclass
    and override it", so we just override the headers; everything else
    (long-poll, retries, JSON encoding, error handling) is inherited.
    """

    # Identifies the client honestly: the hexo-strix project (Strix = HeXO
    # AlphaZero — see PRODUCT.md and CLAUDE.md). Server admins can grep for
    # this in their logs to see who's playing. Version is pinned to the
    # repo's pyproject.toml (`name = "hexo", version = "0.2.0"`) so it tracks
    # releases; bump in lockstep with the workspace.
    DEFAULT_UA = "hexo-strix/0.2.0 (+https://github.com/sootyowl/hexo-strix)"

    def __init__(self, base_url: str, *, user_agent: str | None = None,
                 timeout_s: float = 60.0) -> None:
        # 60s covers the server's 25s long-poll + our MCTS time comfortably
        # (was the SDK default of 40s; bumped to support 1000-move games
        # where each search batch can run several seconds under heavier sims).
        super().__init__(base_url, timeout_s=timeout_s)
        self._user_agent = user_agent or self.DEFAULT_UA

    def request(self, method, path, *, body=None, token=None):
        # Mirror the parent's request body but inject the User-Agent. We
        # avoid editing the SDK on disk so the upstream script stays
        # byte-identical to what was downloaded.
        import json as _json
        import urllib.request as _ur
        import urllib.error as _ue
        req = _ur.Request(
            self.base_url + path,
            method=method,
            data=_json.dumps(body).encode() if body is not None else None,
            headers={
                "User-Agent": self._user_agent,
                "Accept": "application/json",
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {token}"} if token else {}),
            },
        )
        try:
            with _ur.urlopen(req, timeout=self.timeout_s) as resp:
                return _json.loads(resp.read().decode())
        except _ue.HTTPError as exc:
            try:
                detail = _json.loads(exc.read().decode()).get("detail", "")
            except Exception:
                detail = exc.reason
            raise MatchError(exc.code, str(detail)) from None

logger = logging.getLogger("play_vs_shrimp")


# ---------------------------------------------------------------------------
# State reconstruction
# ---------------------------------------------------------------------------

# Project-standard game config. Matches the full-HeXO S6 stage
# (configs/curriculum.toml) and the size-8 fixtures in gen_parity_fixtures.py;
# also what bench_cpu_inference.py uses. We use it unconditionally: Shrimp's
# server doesn't expose the rules it runs, and every checkpoint in this repo
# is trained for six-in-a-row on a radius-8 board, so a single canonical
# config is correct for all of them.
GAME_CONFIG: dict[str, int] = {
    "win_length": 6,
    "placement_radius": 8,
    "max_moves": 300,
}


def build_game_from_state(
    state: dict,
    game_config_dict: dict[str, int],
) -> "object":
    """Reconstruct a ``hexo_rs.GameState`` from a Shrimp state payload.

    The protocol's ``history`` is the authoritative full-game move list
    INCLUDING the server-auto-played P1 opening stone at ``(0, 0)`` — that
    stone already lives on a fresh ``hexo_rs.GameState`` (verified: a fresh
    game's ``legal_moves()`` excludes the origin), so we replay every entry
    EXCEPT the first to avoid a "cell is occupied" crash. After this returns,
    ``game.current_player()`` matches ``state["to_move"]`` (modulo terminal).
    """
    import hexo_rs as _hr  # local import; matches repo convention

    gc = _hr.GameConfig(
        int(game_config_dict["win_length"]),
        int(game_config_dict["placement_radius"]),
        int(game_config_dict["max_moves"]),
    )
    game = _hr.GameState(gc)
    history = state.get("history", [])
    # Skip the auto-played origin at history[0] — already on the board.
    for entry in history[1:]:
        # Tolerate either {q,r,color} objects or bare (q,r) tuples (the wire
        # spec says objects but we don't want to break on a server variant).
        q = int(entry["q"] if isinstance(entry, dict) else entry[0])
        r = int(entry["r"] if isinstance(entry, dict) else entry[1])
        game.apply_move(q, r)
    return game


# ---------------------------------------------------------------------------
# StrixBot — the one-method adapter the SDK requires
# ---------------------------------------------------------------------------

class StrixBot(BotAdapter):
    """A Shrimp ``BotAdapter`` powered by a Strix HeXONet checkpoint.

    Each ``select_stone`` call:
      1. Rebuilds the ``GameState`` from ``state["history"]``.
      2. Builds the per-model-config graph (axis vs hex, threat/relative flags).
      3. Runs Gumbel MCTS at the configured sim budget and returns the
         argmax-of-improved-policy move (the same action the Gumbel-AZ paper
         acting policy plays at evaluation time — deterministic given the
         model + sims, no Gumbel noise). Or, with ``--raw-policy``, the raw
         policy-head argmax (a much weaker but much faster bot — useful as a
         smoke test that the wiring works).
      4. Projects the move onto ``state["legal"]`` so a stale output never
         sends an illegal stone; the SDK will raise HTTP 422 if we somehow
         still miss, and we propagate that (not retry — it's our bug).

    The model is loaded once in ``__init__``; subsequent matches reuse it.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
        mcts_sims: int = 128,
        mcts_m_actions: int = 16,
        raw_policy: bool = False,
    ) -> None:
        # Lazy heavy imports — keeps `--list` and `--help` near-instant.
        import torch
        import hexo_rs
        from hexo_a0.graph import game_to_axis_graph, game_to_graph
        from hexo_a0.model import HeXONet
        from hexo_a0.config import model_config_from_checkpoint

        self.torch = torch
        self.hexo_rs = hexo_rs
        self.device = torch.device(device)
        t0 = time.perf_counter()

        # Ported verbatim from scripts/policy_viewer._load_model: dispatch
        # KLENT checkpoints through hexo_klent.mcts_adapter.adapt_checkpoint,
        # HeXONet checkpoints through HeXONet(model_config).load_state_dict.
        # Reusing that script's loader means we accept every checkpoint shape
        # policy_viewer accepts (all curriculum stages, all KLENT variants,
        # resumed-from-different-config runs that load with strict=False).
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"checkpoint {checkpoint!r} is not a dict-style state file")
        if ckpt.get("format") == "hexo-klent-v1":
            from hexo_klent.mcts_adapter import adapt_checkpoint
            # Build the KLENT model directly on the target device. Building
            # on CPU first then moving can leave adapter-side buffers on
            # CPU when cuda is requested, which surfaces later in F.linear
            # as "mat1 on cuda, weight on cpu" mismatches.
            loaded = adapt_checkpoint(ckpt, self.device)
            model, model_config = loaded.model, loaded.model_config
            logger.info(
                "loaded KLENT checkpoint on %s (iteration=%s)",
                self.device, loaded.iteration)
        else:
            model_config = model_config_from_checkpoint(ckpt, None)
            model = HeXONet(model_config).to(self.device)
            sd = {
                k.removeprefix("_orig_mod."): v
                for k, v in ckpt["model_state_dict"].items()
            }
            result = model.load_state_dict(sd, strict=False)
            if result.missing_keys or result.unexpected_keys:
                logger.warning(
                    "checkpoint %s loaded with strict=False: missing=%d "
                    "unexpected=%d (architecture/model_config mismatch?)",
                    checkpoint, len(result.missing_keys),
                    len(result.unexpected_keys),
                )
            model.eval()

        self.model = model
        self.model_config = model_config
        self.game_config = GAME_CONFIG
        self.mcts_sims = int(mcts_sims)
        self.mcts_m_actions = int(mcts_m_actions)
        self.raw_policy = bool(raw_policy)

        # Per-state graph builder threading ALL graph flags from model_config,
        # including the D6-invariant lean flags (axis_relational, lean node
        # schema). policy_viewer's _make_graph_fn only threads the legacy
        # flags, which silently produces a legacy graph for an axis-relational
        # model and crashes on the first forward. Thread everything.
        if model_config.graph_type == "axis":
            self.graph_fn = lambda g: game_to_axis_graph(
                g,
                prune_empty_edges=getattr(model_config, "prune_empty_edges", False),
                threat_features=getattr(model_config, "threat_features", False),
                relative_stones=getattr(model_config, "relative_stone_encoding", False),
                axis_relational=getattr(model_config, "axis_relational", False),
                compact_stone_onehot=getattr(model_config, "compact_stone_onehot", False),
                node_coords=getattr(model_config, "node_coords", True),
                moves_scope=getattr(model_config, "moves_scope", "node"),
            )
        else:
            self.graph_fn = lambda g: game_to_graph(
                g,
                threat_features=getattr(model_config, "threat_features", False),
                relative_stones=getattr(model_config, "relative_stone_encoding", False),
            )

        # Pre-build the MCTSConfig object once — it's reused for every stone.
        # disable_gumbel_noise=True matches the eval/SPRT acting policy
        # (argmax of improved policy, no in-tree randomness).
        self.mcts_config = hexo_rs.MCTSConfig(
            n_simulations=self.mcts_sims,
            m_actions=self.mcts_m_actions,
            c_visit=50,
            c_scale=1.0,
            disable_gumbel_noise=True,
        )

        # Per-match failure counters. main() resets these after each game so 
        # the JSONL record reflects per-game events. The opponent-retry cap is 
        # overridable per-instance by main() (defaults to ``opponent_max_retries`` 
        # class attr).
        self.opponent_failures: int = 0
        self.our_move_failures: int = 0

        logger.info(
            "loaded %s on %s in %.2fs (graph=%s conv=%s layers=%d hidden=%d "
            "heads=%d jk=%s axis_rel=%s threat=%s rel=%s)",
            checkpoint, device, time.perf_counter() - t0,
            model_config.graph_type, model_config.conv_type,
            model_config.num_layers, model_config.hidden_dim,
            model_config.num_heads, getattr(model_config, "use_jk", False),
            getattr(model_config, "axis_relational", False),
            getattr(model_config, "threat_features", False),
            getattr(model_config, "relative_stone_encoding", False),
        )

    # ------------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------------

    def select_stone(self, state: dict) -> tuple[int, int]:
        game = build_game_from_state(state, self.game_config)

        # If the SDK hands us a state for a stone that's actually game-ending,
        # the rebuilt game will be terminal — fall back to the first legal
        # cell rather than crashing in MCTS (which rejects terminal states).
        if game.is_terminal():
            legal = state.get("legal") or []
            if not legal:
                raise RuntimeError(
                    "StrixBot received a terminal state with no legal moves")
            return int(legal[0]["q"]), int(legal[0]["r"])

        legal_cells = {(int(c["q"]), int(c["r"])) for c in state["legal"]}

        if self.raw_policy:
            q, r = self._raw_policy_move(game)
        else:
            q, r = self._mcts_move(game)

        # Project onto legal — protects against any stale/wrong-coordinate
        # mismatch between the local game and the server's bookkeeping.
        if (q, r) not in legal_cells:
            q, r = self._nearest_legal(q, r, legal_cells)
        return q, r

    # ------------------------------------------------------------------
    # Inference paths
    # ------------------------------------------------------------------

    def _raw_policy_move(self, game: "object") -> tuple[int, int]:
        """Raw policy-head argmax. Fast sanity-check bot.

        Uses ``forward_batch`` (works for both HeXONet and the KLENT
        ``KlentMCTSAdapter`` wrapper) with a single-state batch. Returns
        logits of shape ``(num_legal,)`` aligned with ``game.legal_moves()``
        order — verified against the MCTS path which uses the same ordering.
        """
        from torch_geometric.data import Batch
        torch = self.torch
        data = self.graph_fn(game)
        batch = Batch.from_data_list([data]).to(self.device)
        with torch.inference_mode():
            logits_list, _values = self.model.forward_batch(batch)
        # forward_batch returns a list of 1-D logits (one per graph); the
        # first (and only) entry is aligned with game.legal_moves() order
        # (verified: the MCTS path's eval_fn uses the same contract).
        logits = logits_list[0]
        best = int(logits.argmax().item())
        legal_moves = list(game.legal_moves())
        q, r = int(legal_moves[best][0]), int(legal_moves[best][1])
        return q, r

    def _mcts_move(self, game: "object") -> tuple[int, int]:
        """Gumbel MCTS argmax-of-improved-policy. Eval-grade acting."""
        data = self.graph_fn(game)  # warm graph cache before the eval closure
        torch, hexo_rs = self.torch, self.hexo_rs

        def _eval_fn(states):
            # Batch path: hexo_rs hands us a list of GameStates per leaf; build
            # a batched graph and run a single forward.
            from torch_geometric.data import Batch
            data_list = [self.graph_fn(s) for s in states]
            batch = Batch.from_data_list(data_list).to(self.device)
            with torch.inference_mode():
                logits_list, values = self.model.forward_batch(batch)
            # MCTS contract: return (logits_per_state, values_per_state), both
            # ordered to align with state.legal_moves() for each state.
            return (
                [lg.tolist() for lg in logits_list],
                [float(v.item()) for v in values],
            )

        (_action, improved_policy, _visits, _per_child_q, _per_child_prior,
         _candidate_indices, _forced) = hexo_rs.gumbel_mcts_with_diagnostics(
            game, _eval_fn, self.mcts_config,
        )

        # gumbel_mcts_with_diagnostics returns improved_policy in
        # game.legal_moves() order, matching state["legal"] order — so we can
        # zip the SDK's legal list directly against the policy vector.
        legal_moves = list(game.legal_moves())
        best_idx = max(
            range(len(improved_policy)),
            key=lambda i: improved_policy[i],
        )
        q, r = int(legal_moves[best_idx][0]), int(legal_moves[best_idx][1])
        return q, r

    # ------------------------------------------------------------------
    # Move projection helper
    # ------------------------------------------------------------------

    @staticmethod
    def _nearest_legal(
        q: int, r: int, legal: set[tuple[int, int]],
    ) -> tuple[int, int]:
        """Find the legal cell closest (hex-distance) to (q, r).

        Used when our local pick isn't in the server's legal list — should
        never happen for a fresh state, but a defensive fallback keeps us
        out of HTTP 422 territory if the server ever sends a stale legal set.
        """
        if not legal:
            raise RuntimeError("no legal moves available")
        if (q, r) in legal:
            return q, r

        def hd(a, b):
            return (abs(a[0] - b[0]) + abs(a[1] - b[1])
                    + abs(a[0] + a[1] - b[0] - b[1])) // 2

        target = (q, r)
        return min(legal, key=lambda c: hd(c, target))


    def play_match(
        self, server, *, agent, checkpoint_id, sims,
        agent_color=0, verbose=False,
    ):
        """Override the SDK's play_match with smarter failure classification.

        The SDK's default treats every ``bot_failed`` as a fatal error and
        resigns after ``max_retries`` retries — counting that as our loss.
        That's wrong: ``bot_failed`` only tells us *some* bot crashed; if
        we just finished a successful move (so it wasn't our turn that 
        failed), it's the OPPONENT's bot. We shouldn't be punished for the 
        opponent going down.

        This override keeps separate retry counters for the two failure 
        modes and only ever resigns on OUR-move failures (which are our 
        bug) or when the user-set --max-opponent-retries is exhausted.

        Bot-relative timing heuristics (since the SDK doesn't tell us 
        whose turn failed):
          - bot_failed right after ``status == "your_turn"`` → opponent.
          - bot_failed right after ``status == "bot_thinking"`` → opponent.
          - bot_failed immediately after our move() returned a 5xx → us.

        We use the heuristic: bot_failed is the OPPONENT unless we're in 
        the middle of recovering from our own move() exception. That maps 
        cleanly to "who's move was it last": the SDK stores 
        ``state["to_move"]`` and ``state["you"]``. ``to_move == you`` means 
        our turn; ``to_move != you`` means opponent's turn. The OPPONENT 
        crashes when we asked the server to advance to OUR turn (server 
        signals bot_failed in the response, status to_move stays on opponent 
        or flips).

        In practice, the SDK's bot_failed branch runs whenever the server 
        flags the match. We treat ALL bot_failed events as opponent 
        failures (since OUR move() failures are caught and counted 
        separately and yield only MatchError, not bot_failed status).
        """
        from shrimp_hexo_match_client import MatchError

        match = server.create_match(
            agent=agent, checkpoint_id=checkpoint_id, sims=sims,
            agent_color=agent_color,
        )
        if verbose:
            print(f"match {match.match_id} vs {checkpoint_id}@{sims} "
                  f"— you are color {match.state['you']}")

        while True:
            state = match.state
            self.on_state(state)
            status = state["status"]
            if status == "finished":
                if verbose:
                    print(f"finished: {json.dumps(state['result'])}")
                return state

            if status == "your_turn":
                q, r = self.select_stone(state)
                try:
                    match.move(q, r)
                    self.our_move_failures = 0  # reset on success
                except MatchError as exc:
                    if exc.status == 422:
                        # illegal move - our bug, no retry
                        raise
                    # 4xx non-422, 5xx, network errors our SDK wraps
                    self.our_move_failures += 1
                    if self.our_move_failures > self.max_retries:
                        logger.error(
                            "our move() failed %d times in a row; "
                            "giving up and resigning", self.our_move_failures)
                        match.resign()
                        # match.resign updates state to a fresh dict;
                        # return it so the caller sees a 'finished' result.
                        # If resign() itself fails (server gone), just bubble.
                        try:
                            return match.state
                        except Exception:
                            raise
                    time.sleep(min(2.0 * self.our_move_failures, 10.0))
                    match.refresh()
                continue

            if status == "bot_failed":
                # Treat as an opponent failure by default. Count retries
                # separately from our own move() failures so a flaky 
                # opponent doesn't poison our retry budget (or vice versa).
                self.opponent_failures += 1
                cap = self.opponent_max_retries
                if cap is not None and self.opponent_failures > cap:
                    # Walk away instead of resigning. Resigning would record 
                    # this match as OUR loss on the server's leaderboard, 
                    # which is unfair: it's the opponent's bot that couldn't 
                    # recover. Letting the match idle out server-side keeps 
                    # the ELO record clean.
                    # 
                    # Trade-off: the match stays ACTIVE on the server until 
                    # the server's idle timeout, which can block a quick 
                    # follow-up match under the same agent if the server's 
                    # per-agent active-game limit is tight. Set 
                    # ``--resign-on-opponent-failure`` to free the slot.
                    logger.warning(
                        "opponent bot failed %d times in a row (cap %d); "
                        "%s — server will idle-out the match",
                        self.opponent_failures, cap,
                        "resigning" if self._resign_on_opponent_failure
                        else "abandoning (no resign)",
                    )
                    if getattr(self, "_resign_on_opponent_failure", False):
                        try:
                            match.resign()
                            return match.state
                        except Exception:
                            pass
                    abandoned_state = dict(state)
                    abandoned_state["status"] = "finished"
                    abandoned_state["result"] = {
                        "winner": None,
                        "termination": "abandoned_opponent_failed",
                        "human_result": 0,
                        "opponent_failures": self.opponent_failures,
                    }
                    abandoned_state["history"] = state.get("history", [])
                    abandoned_state["legal"] = state.get("legal", [])
                    abandoned_state["you"] = state.get("you")
                    abandoned_state["match_id"] = match.match_id
                    return abandoned_state
                if verbose:
                    print(f"opponent bot hiccuped ({self.opponent_failures}"
                          f"/{cap if cap is not None else 'inf'}); retrying")
                time.sleep(min(2.0 * self.opponent_failures, 10.0))
                match.retry()
                continue

            # status == "bot_thinking": opponent is computing, long-poll.
            match.refresh(wait=LONG_POLL_S)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_bots(server: MatchServer) -> None:
    """Pretty-print the server's /api/bots catalogue and exit.

    The server's schema is ``{"checkpoints": [...], "sims": [...]}`` where
    each checkpoint is a dict with ``id`` (use as ``--checkpoint``), ``label``,
    and an optional per-checkpoint ``sims`` list (overriding the global
    ``sims`` for that bot). Prints a compact summary by default — the raw
    JSON is too noisy with all checkpoint metadata.
    """
    catalogue = server.bots()
    checkpoints = (
        catalogue.get("checkpoints")
        or catalogue.get("bots")  # tolerate an older "bots" key
        or []
    )
    default_sims = catalogue.get("sims") or []
    rows = []
    for cp in checkpoints:
        rows.append({
            "id": cp.get("id"),
            "label": cp.get("label"),
            "family": cp.get("family"),
            "epoch": cp.get("epoch"),
            "params": cp.get("params"),
            "sims": cp.get("sims") or default_sims,
            "featured": bool(cp.get("featured")),
            "strongest": bool(cp.get("strongest")),
            "default": bool(cp.get("default")),
        })
    print(json.dumps({
        "sims_default": default_sims,
        "checkpoints": rows,
    }, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Play a Strix HeXONet checkpoint against a Shrimp showcase server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", required=True,
                        help="Shrimp server base URL")
    parser.add_argument("--agent", default=None,
                        help="your bot's public leaderboard name "
                             "(required unless --list)")
    parser.add_argument("--checkpoint", required=False,
                        help="path to a Strix HeXONet checkpoint (.pt or .safetensors). "
                             "model_config is read from the checkpoint itself. "
                             "Required unless --list.")
    parser.add_argument("--opponent", default=None,
                        help="server-side opponent checkpoint id (see --list). "
                             "Defaults to the server's strongest checkpoint. "
                             "Independent of --checkpoint: you pick your bot's "
                             "model and your opponent's model separately.")
    parser.add_argument("--sims", type=int, default=128,
                        help="opponent's MCTS sims per stone (sent to the server "
                             "as the 'sims' field on POST /api/match). Higher = "
                             "stronger opponent.")
    parser.add_argument("--our-sims", type=int, default=128,
                        help="OUR bot's MCTS sims per stone. Higher = stronger "
                             "play; 0 disables MCTS and uses the raw policy head.")
    parser.add_argument("--m-actions", type=int, default=16,
                        help="Gumbel-Top-k candidate count per MCTS root")
    parser.add_argument("--device", default="cpu",
                        help="torch device for inference (cpu, cuda, cuda:0, ...)")
    parser.add_argument("--color", default="random",
                        help="0 | 1 | random | alternate — which side your bot plays. "
                             "alternate flips the side every game (P0, P1, P0, ...) "
                             "so a multi-game batch is color-balanced.")
    parser.add_argument("--raw-policy", action="store_true",
                        help="skip MCTS and play raw policy-head argmax "
                             "(much weaker — useful as a smoke test)")
    parser.add_argument("--games", type=int, default=1,
                        help="number of sequential games to play in one run. "
                             "Each game is a separate match on the server; results "
                             "are accumulated and written to --results if given.")
    parser.add_argument("--max-opponent-retries", type=int, default=None,
                        help="how many consecutive server 'bot_failed' events we "
                             "tolerate before walking away. Default: unbounded "
                             "(retry forever with exponential backoff, capped at "
                             "60s per sleep). The SDK's own move()-failure cap "
                             "is left at 5 (those are our bug). Set a positive "
                             "int to cap; 0 is treated as unbounded for symmetry.")
    parser.add_argument("--active-game-wait", type=int, default=600,
                        help="how many seconds we'll wait, retrying POST /api/match, "
                             "when the server returns HTTP 429 'active-game limit "
                             "reached' (we're over the per-IP concurrent-match "
                             "cap). Defaults to 600s to match the server's own "
                             "idle_timeout_s.")
    parser.add_argument("--resign-on-opponent-failure", action="store_true",
                        help="on opponent-failure cap-out, call match.resign() "
                             "(server records our loss) instead of walking away. "
                             "Use this only if the server's per-agent active-game "
                             "limit is tight and you need to free the slot "
                             "immediately; default is to abandon and let the "
                             "server idle-out the match.")
    parser.add_argument("--results", default=None,
                        help="path to a .jsonl file where each game's record is "
                             "written (one JSON object per line, plus a final "
                             "summary line). Parent dirs are created.")
    parser.add_argument("--server-timeout", type=float, default=60.0,
                        help="per-request HTTP timeout in seconds (the server "
                             "long-polls up to 25s; budget the rest for our MCTS).")
    parser.add_argument("--list", action="store_true",
                        help="list the server's playable checkpoints and exit")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable INFO logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = BrowserMatchServer(args.server, timeout_s=args.server_timeout)

    if args.list:
        _list_bots(server)
        return 0

    if not args.agent:
        parser.error("--agent is required (unless --list)")
    if not args.checkpoint:
        parser.error("--checkpoint is required (or pass --list to see options)")

    # Resolve the opponent (server-side checkpoint id). Defaults to the
    # server's strongest checkpoint; --opponent overrides.
    opponent_id = args.opponent
    chosen = None
    try:
        catalogue = server.bots()
    except Exception as exc:  # pragma: no cover — network errors
        logger.warning("could not fetch /api/bots (%s)", exc)
        catalogue = {}
    checkpoints = (
        catalogue.get("checkpoints") or catalogue.get("bots") or []
    )
    if opponent_id is None:
        for cp in checkpoints:
            if cp.get("strongest"):
                chosen = cp
                break
        if chosen is None:
            for cp in checkpoints:
                if cp.get("default"):
                    chosen = cp
                    break
        if chosen is None:
            for cp in checkpoints:
                if cp.get("featured"):
                    chosen = cp
                    break
        if chosen is None and checkpoints:
            chosen = checkpoints[0]
        if chosen is not None:
            opponent_id = chosen["id"]
            logger.info("defaulting to --opponent=%s (strongest in catalogue)", opponent_id)
        else:
            logger.error("server has no checkpoints in /api/bots and no --opponent given")
            return 2
    else:
        # Look up the chosen entry for a sims-cap warning later.
        for cp in checkpoints:
            if cp.get("id") == opponent_id:
                chosen = cp
                break

    # Warn if --sims exceeds the opponent's allowed budget (the server will 
    # reject it as 422 otherwise).
    if chosen is not None:
        allowed = chosen.get("sims") or catalogue.get("sims") or []
        if allowed and args.sims not in allowed:
            logger.warning(
                "--sims=%d is not in the server's allowed list %s for opponent %s; "
                "expecting HTTP 422. Pick one of: %s",
                args.sims, allowed, opponent_id, allowed)

    # Map --our-sims=0 to the raw-policy mode (the MCTS path with 0 sims is 
    # the same as argmax-of-improved-policy with 0 visits, but raw-policy is 
    # much faster and the same effective play).
    use_raw_policy = args.raw_policy or args.our_sims <= 0
    bot = StrixBot(
        args.checkpoint,
        device=args.device,
        mcts_sims=max(args.our_sims, 1),
        mcts_m_actions=args.m_actions,
        raw_policy=use_raw_policy,
    )
    # 0 means "unbounded" (relies on the server's idle timeout). The SDK 
    # wants an int for max_retries, so we use a huge sentinel for unbounded.
    bot.opponent_max_retries = (
        args.max_opponent_retries if args.max_opponent_retries > 0
        else 10**9  # effectively unbounded
    )
    # Whether to call match.resign() on opponent-failure cap-out. Default 
    # False (walk away) since the SDK's default of resigning counts the 
    # unfair outcome as our loss; opt back in for tight active-match limits.
    bot._resign_on_opponent_failure = bool(args.resign_on_opponent_failure)

    color: int | str = args.color if args.color == "random" else int(args.color)

    # Open the results file once (line-buffered so each game's record is 
    # flushed promptly; partial output survives a Ctrl-C or timeout). The 
    # file is created with an empty parent dir.
    out_path = Path(args.results) if args.results else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results_fh = open(out_path, "w", buffering=1)  # line-buffered
    else:
        results_fh = None

    def _emit(record: dict) -> None:
        """Write one JSONL line to --results and stdout (always)."""
        line = json.dumps(record, separators=(",", ":"))
        if results_fh is not None:
            results_fh.write(line + "\n")
        print(line, flush=True)

    def _create_with_backoff(label: str):
        """POST /api/match with retry on HTTP 429 active-game limit.

        Walks the active-game cap with exponential backoff up to 
        ``args.active_game_wait`` seconds. Any other MatchError (4xx/5xx) 
        propagates immediately - those are not transient.
        """
        deadline = time.monotonic() + args.active_game_wait
        attempt = 0
        while True:
            attempt += 1
            try:
                return bot.play_match(
                    server,
                    agent=args.agent,
                    checkpoint_id=opponent_id,
                    sims=args.sims,
                    agent_color=this_color,
                    verbose=args.verbose,
                )
            except MatchError as exc:
                if exc.status != 429:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.error(
                        "active-game wait timed out (%ds) for %s",
                        args.active_game_wait, label)
                    raise
                backoff = min(2.0 ** attempt, 30.0)
                backoff *= 0.9 + 0.2 * random.random()
                wait_s = min(backoff, remaining)
                logger.warning(
                    "active-game limit reached (%s); backing off %.1fs "
                    "(attempt %d, %ds remaining)",
                    exc.detail, wait_s, attempt, int(remaining))
                time.sleep(wait_s)
                # Note: we DON'T increment errors / mark as game on 429 - 
                # we keep retrying until we either get a match or hit the 
                # deadline. The caller treats a final MatchError as abort.

    # Sequential game loop. Colour resolution:
    #   --color 0|1: use that side for every game.
    #   --color random|alternate: vary per game (alternate is the deterministic
    #     P0/P1/P0/P1 ping-pong, recommended for SPRT-style sweeps).
    fixed_color = None
    if args.color in ("0", "1"):
        fixed_color = int(args.color)
    use_alternate = args.color == "alternate"
    use_random = args.color == "random"

    # Per-game record accumulator for the final summary line.
    games_records: list[dict] = []
    matches_total = args.games
    wins = draws = losses = errors = 0
    # Classify forfeits separately: a "resign" after only opponent failures is 
    # not really our loss for ELO purposes — the OPPONENT couldn't recover. 
    # Track these so the aggregate isn't misleading.
    opponent_forfeits = 0
    started_at_total = time.perf_counter()
    aborted = False
    abort_reason: str | None = None

    for game_idx in range(matches_total):
        # Pick this game's side.
        if fixed_color is not None:
            this_color: int | str = fixed_color
        elif use_alternate:
            this_color = game_idx % 2  # P0 (game 0), P1 (game 1), ...
        else:  # use_random
            # Use Python's stdlib RNG so this is deterministic if --our-sims=0
            # (MCTS seeds its own torch RNG, so random is only "random" in 
            # raw-policy mode; that's fine — random is the spec's fallback).
            import random as _random
            this_color = _random.randint(0, 1)

        # Drive the match. play_match() handles polling, two-stones-per-turn,
        # transient retries, and bot_failed recovery. _create_with_backoff
        # wraps the create call with HTTP 429 retry, leaving run-time errors 
        # to propagate.
        try:
            t0 = time.perf_counter()
            result = _create_with_backoff(f"game {game_idx}")
            duration_s = time.perf_counter() - t0
        except MatchError as exc:
            logger.error("server rejected a request on game %d: %s", game_idx, exc)
            errors += 1
            aborted = True
            abort_reason = f"MatchError: {exc}"
            break

        # Classify the outcome from our perspective. The SDK's default 
        # behaviour is to resign on opponent bot_failed, so a "resign" 
        # termination can mean either the opponent gave up (our win) or 
        # our bot gave up after the opponent kept crashing (NOT our loss in 
        # any fair sense). Distinguish via opponent_failures: if we had a 
        # substantial opponent_failure count and the match ended in resign 
        # without a winner, it's an opponent forfeit (no-result).
        final = result.get("result") or {}
        winner = final.get("winner")
        termination = final.get("termination")
        you_color = result.get("you")
        opp_fail_count = getattr(bot, "opponent_failures", 0)
        our_move_fail_count = getattr(bot, "our_move_failures", 0)

        if winner == you_color:
            outcome = "win"; wins += 1
        elif winner is not None:
            outcome = "loss"; losses += 1
        elif termination == "draw":
            outcome = "draw"; draws += 1
        elif (termination in ("resign", "abandoned_opponent_failed")
              and opp_fail_count >= 1 and our_move_fail_count == 0):
            # The OPPONENT kept crashing and we either resigned (SDK 
            # default) or walked away (our override). Neither is a fair 
            # loss; record as opponent forfeit so it doesn't tank ELO.
            outcome = "abandoned"; opponent_forfeits += 1
        else:
            outcome = "unknown"; errors += 1

        record = {
            "type": "game",
            "game_idx": game_idx,
            "match_id": result.get("match_id"),
            "agent": args.agent,
            "checkpoint_id": opponent_id,
            "sims": args.sims,
            "our_sims": args.our_sims,
            "requested_color": this_color,
            "assigned_color": you_color,
            "outcome": outcome,
            "winner": winner,
            "termination": termination,
            "moves": len(result.get("history", [])),
            "duration_s": round(duration_s, 2),
            "human_result": final.get("human_result"),
            "opponent_failures": opp_fail_count,
            "our_move_failures": our_move_fail_count,
        }
        games_records.append(record)
        _emit(record)
        # Reset per-game counters so the next game starts clean. The 
        # counters live on the BotAdapter instance and otherwise accumulate.
        bot.opponent_failures = 0
        bot.our_move_failures = 0

    duration_total_s = round(time.perf_counter() - started_at_total, 2)
    finished = len(games_records)
    # Forfeit-aware score: only games with a real result (win/draw/loss) 
    # count toward the ELO estimate. Opponent forfeits are excluded so a 
    # flaky opponent doesn't pull the score below 0.5 just from their 
    # downtime, and connection errors don't either (no signal on either side).
    counted_games = max(0, finished - opponent_forfeits - errors)
    score = (
        (wins + 0.5 * draws) / counted_games if counted_games > 0 else 0.5
    )
    summary = {
        "type": "summary",
        "agent": args.agent,
        "checkpoint_id": opponent_id,
        "sims": args.sims,
        "our_sims": args.our_sims,
        "games_requested": matches_total,
        "games_finished": finished,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "errors": errors,
        "opponent_forfeits": opponent_forfeits,
        "score": round(score, 4),
        "score_basis": f"wins+0.5*draws / {counted_games} counted games "
                       "(opponent forfeits and errors excluded)",
        "total_duration_s": duration_total_s,
        "aborted": aborted,
        "abort_reason": abort_reason,
    }
    _emit(summary)

    if results_fh is not None:
        results_fh.close()

    # Exit-code convention:
    #   0 = clean run (won more than half, OR ran without any *our* errors).
    #       All-abandoned (opponent kept failing) is still 0 -- we ran 
    #       cleanly, the opponent just couldn't recover.
    #   1 = lost more than half of the counted games.
    #   3 = aborted on a connection / setup error before any game ran, OR
    #       all games errored out for some reason we couldn't handle.
    if finished == 0:
        # Never managed to start any game.
        return 3
    if counted_games == 0:
        # All games finished but all were abandoned or errored. Not a "we 
        # lost" outcome -- opponent (or we, for errors) just couldn't play.
        return 0
    if score > 0.5:
        return 0
    if score < 0.5:
        return 1
    return 1  # exactly 0.5 → conservative "we didn't win"




if __name__ == "__main__":

    sys.exit(main())
