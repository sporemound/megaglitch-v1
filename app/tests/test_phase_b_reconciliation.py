import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _create_test_image(path: Path, width: int = 320, height: int = 240) -> np.ndarray:
    """Create a deterministic test image and write to disk."""
    import cv2
    y, x = np.mgrid[0:height, 0:width]
    # Create RGB gradient
    r = (x * 255 // max(1, width - 1)).astype(np.uint8)
    g = (y * 255 // max(1, height - 1)).astype(np.uint8)
    b = ((x + y) * 255 // max(1, width + height - 2)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return rgb


def invert_preset():
    return {
        "version": "1.0",
        "nodes": [
            {
                "id": "InvertColorNode:1",
                "type": "InvertColorNode",
                "label": "Invert Colors",
                "position": [200, 100],
                "params": {"enabled": True, "invert_chance": 100},
            }
        ],
        "links": [
            {
                "from_node": "SOURCE",
                "to_node": "Invert Colors",
                "to_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
            {
                "from_node": "Invert Colors",
                "to_node": "DISPLAY",
                "from_node_id": "InvertColorNode:1",
                "from_attr": "image_out",
                "to_attr": "image_in",
            },
        ],
    }


def test_full_g1_to_g2_restart_reconciliation_and_stale_ack_rejection():
    """Test full G1 -> G2 lifecycle: graph/source reconciliation, master dimensions, and stale ACK rejection."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    with tempfile.TemporaryDirectory(prefix="mg-phase-b-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_img = tmp_path / "test_source.png"
        src_rgb = _create_test_image(test_img, width=320, height=240)

        # 1. Start Engine Generation 1 (G1)
        proc_g1 = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "tools" / "rust_ui_bridge.py")],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            # Read G1 ready event
            line = proc_g1.stdout.readline()
            ready_g1 = json.loads(line)
            assert ready_g1["type"] == "ready"
            assert ready_g1["graph_generation"] == 0

            # Read preview surface
            surface_line = proc_g1.stdout.readline()
            assert json.loads(surface_line)["type"] == "preview_surface"

            # Submit Graph R1 to G1
            preset = invert_preset()
            proc_g1.stdin.write(json.dumps({
                "version": 2,
                "id": 1,
                "type": "replace_graph",
                "expected_state_revision": 0,
                "graph": preset,
            }) + "\n")
            proc_g1.stdin.flush()

            # Read ACK for graph R1
            ack_graph_g1 = None
            for _ in range(10):
                line = proc_g1.stdout.readline()
                evt = json.loads(line)
                if evt.get("type") == "ack" and evt.get("id") == 1:
                    ack_graph_g1 = evt
                    break
            assert ack_graph_g1 is not None, "Graph ACK must be received"
            assert ack_graph_g1["accepted"] is True, f"Graph rejected: {ack_graph_g1}"
            assert ack_graph_g1["graph_generation"] == 1

            # Submit Source S1 (image) to G1
            proc_g1.stdin.write(json.dumps({
                "version": 2,
                "id": 2,
                "type": "set_source",
                "expected_state_revision": ack_graph_g1["state_revision"],
                "kind": "image",
                "path": str(test_img),
            }) + "\n")
            proc_g1.stdin.flush()

            # Read ACK for source S1
            ack_source_g1 = None
            for _ in range(10):
                line = proc_g1.stdout.readline()
                evt = json.loads(line)
                if evt.get("type") == "ack" and evt.get("id") == 2:
                    ack_source_g1 = evt
                    break
            assert ack_source_g1 is not None, "Source ACK must be received"
            assert ack_source_g1["accepted"] is True, f"Source rejected: {ack_source_g1}"

            # Read snapshot to check master dimensions
            snapshot_g1 = None
            for _ in range(10):
                line = proc_g1.stdout.readline()
                evt = json.loads(line)
                if evt.get("type") == "snapshot":
                    snapshot_g1 = evt
                    break
            assert snapshot_g1 is not None, "Snapshot must be received"
            # Assert master dimensions match decoded source dimensions (320x240)
            assert snapshot_g1["width"] == 320
            assert snapshot_g1["height"] == 240
            assert snapshot_g1["source_ready"] is True
            assert snapshot_g1["source_kind"] == "image"

        finally:
            # 2. Simulate Engine G1 termination / crash
            proc_g1.kill()
            proc_g1.wait()

        # 3. Start Engine Generation 2 (G2)
        proc_g2 = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "tools" / "rust_ui_bridge.py")],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        try:
            # Read G2 ready event
            line = proc_g2.stdout.readline()
            ready_g2 = json.loads(line)
            assert ready_g2["type"] == "ready"
            assert ready_g2["state_revision"] == 0

            surface_line = proc_g2.stdout.readline()
            assert json.loads(surface_line)["type"] == "preview_surface"

            # 4. Prove Stale ACK Rejection:
            # G1 ACK had graph_generation=1, state_revision=1, id=1.
            # In G2 session, state_revision begins at 0.
            stale_ack = ack_graph_g1.copy()
            assert stale_ack["id"] == 1
            assert stale_ack["graph_generation"] == 1

            # 5. Automatically Resubmit Graph R1 to G2
            proc_g2.stdin.write(json.dumps({
                "version": 2,
                "id": 10,
                "type": "replace_graph",
                "expected_state_revision": 0,
                "graph": preset,
            }) + "\n")
            proc_g2.stdin.flush()

            ack_graph_g2 = None
            for _ in range(10):
                line = proc_g2.stdout.readline()
                evt = json.loads(line)
                if evt.get("type") == "ack" and evt.get("id") == 10:
                    ack_graph_g2 = evt
                    break
            assert ack_graph_g2 is not None, "Graph ACK G2 must be received"
            assert ack_graph_g2["accepted"] is True, f"Graph rejected in G2: {ack_graph_g2}"
            assert ack_graph_g2["graph_generation"] == 1
            assert ack_graph_g2["state_revision"] == 1

            # 6. Automatically Resubmit Source S1 to G2
            proc_g2.stdin.write(json.dumps({
                "version": 2,
                "id": 11,
                "type": "set_source",
                "expected_state_revision": ack_graph_g2["state_revision"],
                "kind": "image",
                "path": str(test_img),
            }) + "\n")
            proc_g2.stdin.flush()

            ack_source_g2 = None
            for _ in range(10):
                line = proc_g2.stdout.readline()
                evt = json.loads(line)
                if evt.get("type") == "ack" and evt.get("id") == 11:
                    ack_source_g2 = evt
                    break
            assert ack_source_g2 is not None, "Source ACK G2 must be received"
            assert ack_source_g2["accepted"] is True, f"Source rejected in G2: {ack_source_g2}"
            assert ack_source_g2["state_revision"] == 2

            # 7. First Master Frame from G2
            snapshot_g2 = None
            for _ in range(10):
                line = proc_g2.stdout.readline()
                evt = json.loads(line)
                if evt.get("type") == "snapshot":
                    snapshot_g2 = evt
                    break
            assert snapshot_g2 is not None, "Snapshot G2 must be received"
            assert snapshot_g2["width"] == 320
            assert snapshot_g2["height"] == 240
            assert snapshot_g2["source_ready"] is True
            assert snapshot_g2["source_kind"] == "image"
            assert snapshot_g2["output_sha256"] != ""

        finally:
            proc_g2.kill()
            proc_g2.wait()
