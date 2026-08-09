"""Persistence and HTTP routing for direct saved proof links."""

import http.client
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

from hexo_a0.serving.app import make_handler_class
from hexo_a0.serving.game import GameManager
from hexo_a0.serving.proofs import ProofStore, ProofValidationError


def _bundle():
    return {
        "format": "hexo-pdspn-proof-bundle-v1",
        "position": {
            "stones": [[0, 0, "P1"]],
            "attacker": "P1",
            "placements_remaining": 2,
            "config": {"win_length": 6, "placement_radius": 8, "max_moves": 400},
        },
        "engine": "pdspn",
        "width": "tight",
        "verification": {"dagNodes": 1, "proofEdges": 0, "maxAttackerTurns": 1},
        "certificate": {
            "version": 1,
            "width": "tight",
            "root": 0,
            "nodes": [{"kind": "unstoppable"}],
        },
    }


def _manager():
    return GameManager(
        game_kwargs={"win_length": 6, "placement_radius": 8, "max_moves": 400},
        bot_turn_fn=lambda rec: None,
        recorder=None,
        mcts_sims=64,
        m_actions=16,
        checkpoint_path="x.pt",
        model_label="test",
        difficulty_sims={"standard": 64},
        default_difficulty="standard",
        idle_ttl_seconds=3600,
        max_games=10,
    )


@contextmanager
def _server(prefix=""):
    store = ProofStore(":memory:")
    handler = make_handler_class(
        _manager(), url_prefix=prefix, proof_store=store,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _request(port, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {} if data is None else {
        "Content-Type": "application/json",
        "Content-Length": str(len(data)),
    }
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, path, body=data, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    content_type = response.getheader("Content-Type")
    connection.close()
    return status, content_type, raw


def test_store_is_canonical_content_addressed_and_round_trips():
    store = ProofStore(":memory:")
    bundle = _bundle()
    proof_id = store.put(bundle)
    reordered = dict(reversed(list(bundle.items())))
    assert store.put(reordered) == proof_id
    assert len(proof_id) == 43
    assert json.loads(store.get_json_bytes(proof_id)) == bundle
    assert store.get_json_bytes("not-an-id") is None


def test_store_rejects_structural_summary_mismatch():
    store = ProofStore(":memory:")
    bundle = _bundle()
    bundle["verification"]["proofEdges"] = 1
    try:
        store.put(bundle)
    except ProofValidationError as error:
        assert "edge count" in str(error)
    else:
        raise AssertionError("malformed proof was accepted")


def test_store_accepts_and_counts_ranked_attacker_alternatives():
    store = ProofStore(":memory:")
    bundle = _bundle()
    bundle["certificate"]["nodes"].append({
        "kind": "attacker_move",
        "action": [[1, 0], [2, 0]],
        "child": 0,
        "alternatives": [{"action": [[1, 1], [2, 1]], "child": 0}],
    })
    bundle["certificate"]["root"] = 1
    bundle["verification"] = {
        "dagNodes": 2, "proofEdges": 2, "maxAttackerTurns": 2,
    }
    assert store.put(bundle)

    duplicate = json.loads(json.dumps(bundle))
    duplicate["certificate"]["nodes"][1]["alternatives"][0]["action"] = [[1, 0], [2, 0]]
    try:
        store.put(duplicate)
    except ProofValidationError as error:
        assert "duplicate attacker alternative" in str(error)
    else:
        raise AssertionError("duplicate attacker alternative was accepted")


def test_store_accepts_consistent_shortest_search_metadata():
    store = ProofStore(":memory:")
    bundle = _bundle()
    bundle["certificate"]["nodes"].append({
        "kind": "attacker_move", "action": [[1, 0], [2, 0]], "child": 0,
    })
    bundle["certificate"]["root"] = 1
    bundle["verification"] = {
        "dagNodes": 2, "proofEdges": 1, "maxAttackerTurns": 2,
    }
    bundle["optimization"] = {
        "method": "pdspn-shortest-v1",
        "shortestCertified": True,
        "bestUpperDepth": 2,
        "excludedThroughDepth": 1,
        "thresholdProbes": 5,
        "sampleLine": [
            {"turn": 0, "player": "P1", "cells": [[1, 0], [2, 0]]},
            {"turn": 1, "player": "P2", "cells": [[1, 1], [2, 1]]},
        ],
    }
    assert store.put(bundle)

    malformed = json.loads(json.dumps(bundle))
    malformed["optimization"]["excludedThroughDepth"] = 0
    try:
        store.put(malformed)
    except ProofValidationError as error:
        assert "adjacent" in str(error)
    else:
        raise AssertionError("inconsistent shortest metadata was accepted")


def test_http_save_direct_page_and_bundle_round_trip_with_prefix():
    bundle = _bundle()
    with _server(prefix="/hexo") as port:
        status, content_type, raw = _request(port, "POST", "/hexo/api/proofs", bundle)
        assert status == 200 and content_type == "application/json"
        saved = json.loads(raw)
        assert saved["url"] == f"/hexo/proof/{saved['id']}"

        status, _, raw = _request(port, "GET", f"/hexo/api/proofs/{saved['id']}")
        assert status == 200 and json.loads(raw) == bundle

        status, content_type, raw = _request(port, "GET", saved["url"])
        assert status == 200 and content_type.startswith("text/html")
        assert b'id="proof-explorer"' in raw


def test_http_rejects_malformed_proof_and_unknown_id():
    with _server() as port:
        malformed = _bundle()
        malformed["engine"] = "dfpn"
        status, _, raw = _request(port, "POST", "/api/proofs", malformed)
        assert status == 400 and "PDS-PN" in json.loads(raw)["error"]

        status, _, _ = _request(port, "GET", "/api/proofs/" + "a" * 43)
        assert status == 404
