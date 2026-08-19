"""Minimal Observatory HTTP server for Playwright UI tests.

It serves the production template and static assets without loading a model or
opening a database. Browser tests exercise local replay and responsive UI; any
unexpected inference request therefore fails visibly instead of consuming GPU.
"""

from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer

from hexo_a0.serving.app import make_handler_class
from hexo_a0.serving.game import GameManager


def _manager() -> GameManager:
    return GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda _record: None,
        recorder=None,
        mcts_sims=64,
        m_actions=16,
        checkpoint_path="browser-test.safetensors",
        model_label="browser-test",
        difficulty_sims={"quick": 16, "standard": 64},
        default_difficulty="standard",
        idle_ttl_seconds=3600,
        max_games=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    handler = make_handler_class(_manager(), admin_token="", url_prefix="", analyze_ctx=None)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Playwright server listening on http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
