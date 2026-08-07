"""Content-addressed persistence for shareable PDS-PN proof bundles.

The server deliberately does not claim to prove the bundle: the browser runs
the independent Rust certificate verifier before displaying a saved proof.
This module enforces a tight transport/storage shape, canonicalises JSON for a
stable ID, and stores the highly-compressible DAG in a bounded SQLite cache.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import threading
import zlib


PROOF_BUNDLE_FORMAT = "hexo-pdspn-proof-bundle-v1"
PROOF_MAX_BODY_BYTES = 4 * 1024 * 1024
PROOF_MAX_COMPRESSED_BYTES = 1024 * 1024
PROOF_MAX_NODES = 200_000
PROOF_MAX_STORED_BYTES = 512 * 1024 * 1024
PROOF_MAX_STORED_ITEMS = 10_000
PROOF_ID_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")


class ProofValidationError(ValueError):
    pass


def _plain_int(value, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProofValidationError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ProofValidationError(f"{label} must be in [{minimum},{maximum}]")
    return value


def _action(value, label: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= 2:
        raise ProofValidationError(f"{label} must contain one or two placements")
    cells = []
    for index, cell in enumerate(value):
        if not isinstance(cell, list) or len(cell) != 2:
            raise ProofValidationError(f"{label}[{index}] must be [q,r]")
        cells.append((
            _plain_int(cell[0], f"{label}[{index}].q", -1000, 1000),
            _plain_int(cell[1], f"{label}[{index}].r", -1000, 1000),
        ))
    if len(set(cells)) != len(cells):
        raise ProofValidationError(f"{label} repeats a placement")
    return tuple(cells)


def validate_proof_bundle(bundle: dict) -> None:
    """Reject malformed or resource-abusive bundles at the HTTP boundary.

    This is intentionally structural validation. Kernel-level correctness is
    checked by ``StrixSolver.verify_certificate`` in the viewer's Web Worker.
    """
    if bundle.get("format") != PROOF_BUNDLE_FORMAT:
        raise ProofValidationError("unsupported proof bundle format")
    if bundle.get("engine") != "pdspn":
        raise ProofValidationError("only PDS-PN proof bundles can be saved")
    if bundle.get("width") not in ("tight", "wide"):
        raise ProofValidationError("proof width must be tight or wide")

    position = bundle.get("position")
    if not isinstance(position, dict):
        raise ProofValidationError("proof position must be an object")
    stones = position.get("stones")
    if not isinstance(stones, list) or not 1 <= len(stones) <= 10_000:
        raise ProofValidationError("proof position must contain 1..10000 stones")
    occupied = set()
    for index, stone in enumerate(stones):
        if not isinstance(stone, list) or len(stone) != 3:
            raise ProofValidationError(f"position.stones[{index}] must be [q,r,player]")
        q = _plain_int(stone[0], f"position.stones[{index}].q", -1000, 1000)
        r = _plain_int(stone[1], f"position.stones[{index}].r", -1000, 1000)
        if stone[2] not in ("P1", "P2"):
            raise ProofValidationError(f"position.stones[{index}].player is invalid")
        if (q, r) in occupied:
            raise ProofValidationError("proof position contains a duplicate coordinate")
        occupied.add((q, r))
    if position.get("attacker") not in ("P1", "P2"):
        raise ProofValidationError("proof attacker must be P1 or P2")
    _plain_int(position.get("placements_remaining"), "placements_remaining", 1, 2)
    config = position.get("config")
    if not isinstance(config, dict):
        raise ProofValidationError("proof config must be an object")
    _plain_int(config.get("win_length"), "win_length", 1, 32)
    _plain_int(config.get("placement_radius"), "placement_radius", 1, 64)
    _plain_int(config.get("max_moves"), "max_moves", 1, 100_000)

    certificate = bundle.get("certificate")
    if not isinstance(certificate, dict):
        raise ProofValidationError("certificate must be an object")
    _plain_int(certificate.get("version"), "certificate.version", 1, 1)
    if certificate.get("width") != bundle["width"]:
        raise ProofValidationError("certificate width does not match the bundle")
    nodes = certificate.get("nodes")
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= PROOF_MAX_NODES:
        raise ProofValidationError(f"certificate must contain 1..{PROOF_MAX_NODES} nodes")
    root = _plain_int(certificate.get("root"), "certificate.root", 0, len(nodes) - 1)
    del root

    edges = 0
    for node_id, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ProofValidationError(f"certificate node {node_id} must be an object")
        kind = node.get("kind")
        if kind == "immediate_win":
            _action(node.get("action"), f"certificate node {node_id} action")
        elif kind == "attacker_move":
            supplied = {_action(node.get("action"), f"certificate node {node_id} action")}
            _plain_int(node.get("child"), f"certificate node {node_id} child", 0, len(nodes) - 1)
            edges += 1
            alternatives = node.get("alternatives", [])
            if not isinstance(alternatives, list):
                raise ProofValidationError(
                    f"certificate node {node_id} alternatives must be an array")
            for alternative_id, alternative in enumerate(alternatives):
                if not isinstance(alternative, dict):
                    raise ProofValidationError(
                        f"certificate node {node_id} alternative {alternative_id} must be an object")
                action = _action(
                    alternative.get("action"),
                    f"certificate node {node_id} alternative {alternative_id} action")
                if action in supplied:
                    raise ProofValidationError(
                        f"certificate node {node_id} has a duplicate attacker alternative")
                supplied.add(action)
                _plain_int(
                    alternative.get("child"),
                    f"certificate node {node_id} alternative {alternative_id} child",
                    0, len(nodes) - 1)
                edges += 1
        elif kind == "defender_replies":
            responses = node.get("responses")
            if not isinstance(responses, list) or not responses:
                raise ProofValidationError(f"certificate node {node_id} has no responses")
            for response_id, response in enumerate(responses):
                if not isinstance(response, dict):
                    raise ProofValidationError(
                        f"certificate node {node_id} response {response_id} must be an object")
                _action(response.get("action"),
                        f"certificate node {node_id} response {response_id} action")
                _plain_int(response.get("child"),
                           f"certificate node {node_id} response {response_id} child",
                           0, len(nodes) - 1)
                edges += 1
        elif kind != "unstoppable":
            raise ProofValidationError(f"certificate node {node_id} has an unknown kind")

    verification = bundle.get("verification")
    if not isinstance(verification, dict):
        raise ProofValidationError("verification summary must be an object")
    if _plain_int(verification.get("dagNodes"), "verification.dagNodes", 1,
                  PROOF_MAX_NODES) != len(nodes):
        raise ProofValidationError("verification node count does not match the certificate")
    if _plain_int(verification.get("proofEdges"), "verification.proofEdges", 0,
                  10_000_000) != edges:
        raise ProofValidationError("verification edge count does not match the certificate")
    _plain_int(verification.get("maxAttackerTurns"), "verification.maxAttackerTurns",
               1, 100_000)


class ProofStore:
    """Bounded, thread-safe SQLite store of canonical compressed bundles."""

    def __init__(self, path: str = ":memory:", *,
                 max_items: int = PROOF_MAX_STORED_ITEMS,
                 max_bytes: int = PROOF_MAX_STORED_BYTES):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS proof_bundles("
            " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
            " proof_id TEXT NOT NULL UNIQUE,"
            " payload BLOB NOT NULL,"
            " raw_bytes INTEGER NOT NULL,"
            " compressed_bytes INTEGER NOT NULL)")
        self._conn.commit()

    def put(self, bundle: dict) -> str:
        validate_proof_bundle(bundle)
        canonical = json.dumps(
            bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode("utf-8")
        if len(canonical) > PROOF_MAX_BODY_BYTES:
            raise ProofValidationError("canonical proof bundle is too large")
        compressed = zlib.compress(canonical, level=9)
        if len(compressed) > PROOF_MAX_COMPRESSED_BYTES:
            raise ProofValidationError("compressed proof bundle is too large")
        digest = hashlib.sha256(canonical).digest()
        proof_id = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")

        with self._lock:
            existing = self._conn.execute(
                "SELECT payload FROM proof_bundles WHERE proof_id=?", (proof_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != compressed:
                    raise RuntimeError("proof digest collision")
                return proof_id
            self._conn.execute(
                "INSERT INTO proof_bundles(proof_id,payload,raw_bytes,compressed_bytes)"
                " VALUES(?,?,?,?)",
                (proof_id, compressed, len(canonical), len(compressed)),
            )
            while True:
                count, total = self._conn.execute(
                    "SELECT COUNT(*),COALESCE(SUM(compressed_bytes),0) FROM proof_bundles"
                ).fetchone()
                if count <= self.max_items and total <= self.max_bytes:
                    break
                self._conn.execute(
                    "DELETE FROM proof_bundles WHERE seq=(SELECT MIN(seq) FROM proof_bundles)")
            self._conn.commit()
        return proof_id

    def get_json_bytes(self, proof_id: str) -> bytes | None:
        if not PROOF_ID_RE.fullmatch(proof_id):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT payload,raw_bytes FROM proof_bundles WHERE proof_id=?", (proof_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            raw = zlib.decompress(row[0])
        except zlib.error:
            return None
        if len(raw) != row[1] or len(raw) > PROOF_MAX_BODY_BYTES:
            return None
        return raw
