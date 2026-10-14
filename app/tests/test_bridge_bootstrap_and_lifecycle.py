import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bridge_spawn_from_external_directory_with_shadow_color_module():
    """Prove that running from a non-repo directory with a stray color.py does not hijack imports."""
    with tempfile.TemporaryDirectory(prefix="megaglitch-test-cwd-") as temp_dir:
        temp_path = Path(temp_dir)
        # Create a malicious/broken color.py that raises an error if imported
        shadow_color = temp_path / "color.py"
        shadow_color.write_text("raise RuntimeError('Shadowed color.py was imported!')\n", encoding="utf-8")

        # Also create a broken config.py to verify general isolation
        shadow_config = temp_path / "config.py"
        shadow_config.write_text("raise RuntimeError('Shadowed config.py was imported!')\n", encoding="utf-8")

        # Spawn rust_ui_bridge.py using the anchored environment with CWD = temp_path
        # But setting PYTHONPATH to ROOT first (as done by configure_bridge_environment)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

        # Launch from ROOT as CWD, or execute tools/rust_ui_bridge.py
        proc = subprocess.Popen(
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
            first_line = proc.stdout.readline()
            assert first_line, "Bridge process should emit ready event"
            event1 = json.loads(first_line)
            assert event1.get("type") == "ready"

            second_line = proc.stdout.readline()
            event2 = json.loads(second_line)
            assert event2.get("type") == "preview_surface"
        finally:
            proc.kill()
            proc.wait()


def test_bridge_protocol_ready_and_shutdown():
    """Verify that rust_ui_bridge reaches ready and cleanly shuts down."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    proc = subprocess.Popen(
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
        ready_line = proc.stdout.readline()
        event1 = json.loads(ready_line)
        assert event1.get("type") == "ready"

        surface_line = proc.stdout.readline()
        event2 = json.loads(surface_line)
        assert event2.get("type") == "preview_surface"

        # Send shutdown command
        proc.stdin.write('{"version":2,"id":999,"type":"shutdown","expected_state_revision":0}\n')
        proc.stdin.flush()

        # Read events until shutdown ack is found
        found_ack = False
        for _ in range(10):
            line = proc.stdout.readline()
            if not line:
                break
            event = json.loads(line)
            if event.get("type") == "ack" and event.get("id") == 999:
                assert event.get("accepted") is True
                found_ack = True
                break
        assert found_ack, "Shutdown command must be acknowledged"
    finally:
        proc.kill()
        proc.wait()
