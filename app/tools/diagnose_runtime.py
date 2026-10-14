"""Machine-readable MegaGlitch native/OpenCV/runtime diagnostics."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import struct
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _check(function):
    try:
        details = function() or {}
        return {"status": "ok", **details}
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _python_check():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("native artifacts require CPython 3.12")
    if struct.calcsize("P") != 8:
        raise RuntimeError("native artifacts require 64-bit Python")
    return {
        "executable": sys.executable,
        "version": sys.version,
        "implementation": platform.python_implementation(),
        "pointer_bits": struct.calcsize("P") * 8,
    }


def _packages_check():
    required = [
        "numpy", "PIL", "dearpygui", "imageio", "pyvirtualcam",
        "mss", "moderngl", "lupa",
    ]
    missing = [name for name in required if find_spec(name) is None]
    if missing:
        raise RuntimeError(f"missing required packages: {', '.join(missing)}")
    return {"required_packages": required}


def _temporal_check():
    from megaglitch import _native_temporal
    from megaglitch.native_temporal import native_available

    if not native_available():
        raise RuntimeError("native temporal extension did not load")
    required = ["NativeTemporalRingCore", "amplify_prediction_errors_uniform", "process_pframe_spiral_uniform"]
    missing = [name for name in required if not hasattr(_native_temporal, name)]
    if missing:
        raise RuntimeError(f"native temporal exports missing: {', '.join(missing)}")
    return {"path": str(Path(_native_temporal.__file__).resolve()), "exports": required}


def _rust_check():
    import numpy as np

    from megaglitch.native_runtime import (
        GRAPH_IR_VERSION,
        GraphOpcode,
        GraphOperation,
        GraphSnapshot,
        RuntimeConfig,
        _DLL_PATH,
        create_runtime,
        native_runtime_available,
    )

    if not native_runtime_available():
        raise RuntimeError(f"Rust graph runtime is unavailable at {_DLL_PATH}")
    graph = GraphSnapshot((
        GraphOperation(GraphOpcode.INPUT),
        GraphOperation(GraphOpcode.BRIGHTNESS_CONTRAST, arg0=17.0, arg1=-25.0),
        GraphOperation(GraphOpcode.INVERT),
        GraphOperation(GraphOpcode.OUTPUT),
    ))
    passthrough = GraphSnapshot((
        GraphOperation(GraphOpcode.INPUT),
        GraphOperation(GraphOpcode.OUTPUT),
    ))
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    expected = 255 - np.clip(source.astype(np.float32) * 0.75 + 17, 0, 255).astype(np.uint8)
    with create_runtime(RuntimeConfig(2, 2, history_capacity=2, graph=graph), "native") as runtime:
        runtime.start()
        if not runtime.submit_frame(source):
            raise RuntimeError("Rust graph IR self-test rejected its input frame")
        output, first_sequence = runtime.wait_for_frame(timeout=1.0)
        if not np.array_equal(output, expected):
            raise RuntimeError("Rust graph IR self-test output differs from reference")
        runtime.update_graph(passthrough)
        if not runtime.submit_frame(source):
            raise RuntimeError("Rust graph IR update self-test rejected its input frame")
        updated, second_sequence = runtime.wait_for_frame(timeout=1.0)
        if not np.array_equal(updated, source):
            raise RuntimeError("Rust immutable graph update did not apply to the next frame")
        statistics = runtime.get_statistics()
    return {
        "path": str(_DLL_PATH.resolve()),
        "graph_ir_version": GRAPH_IR_VERSION,
        "graph_ir_opcodes": [opcode.name.lower() for opcode in GraphOpcode],
        "selftest_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "first_sequence": first_sequence,
        "second_sequence": second_sequence,
        "commands_applied": statistics["commands_applied"],
    }


def _opencv_check(run_selftest: bool):
    from megaglitchnode.opencv_cuda_bootstrap import (
        DEFAULT_OPENCV_BIN,
        DEFAULT_OPENCV_PYTHON,
        bootstrap_for_dev_defaults,
        get_cuda_info,
        has_cuda_build,
        run_cuda_selftest,
    )

    opencv_bin = Path(os.getenv("MEGAGLITCH_OPENCV_BIN", DEFAULT_OPENCV_BIN))
    opencv_python = Path(os.getenv("MEGAGLITCH_OPENCV_PYTHON", DEFAULT_OPENCV_PYTHON))
    bootstrap_for_dev_defaults()
    import cv2

    build_ok, build_message = has_cuda_build()
    details = {
        "module_path": str(Path(cv2.__file__).resolve()),
        "version": cv2.__version__,
        "configured_bin": str(opencv_bin),
        "configured_python": str(opencv_python),
        "opencv_dll_count": len(list(opencv_bin.glob("opencv_*.dll"))),
        "cuda_build": bool(build_ok),
        "cuda_build_message": build_message,
        "cuda_info": get_cuda_info(),
    }
    if run_selftest:
        details["cuda_selftest"] = run_cuda_selftest()
    return details


def _shared_memory_check():
    from multiprocessing import shared_memory

    region = shared_memory.SharedMemory(create=True, size=64)
    name = region.name
    region.buf[:4] = b"MGRT"
    attached = shared_memory.SharedMemory(name=name)
    try:
        if bytes(attached.buf[:4]) != b"MGRT":
            raise RuntimeError("shared-memory round trip did not match")
    finally:
        attached.close()
        region.close()
        region.unlink()
    return {"roundtrip": True}


def _windows_runtime_check():
    if os.name != "nt":
        return {"applicable": False}
    libraries = {}
    for name, required in (("vcruntime140.dll", True), ("msvcp140.dll", True), ("nvcuda.dll", False)):
        try:
            ctypes.WinDLL(name)
            libraries[name] = "loadable"
        except OSError as exc:
            libraries[name] = f"missing: {exc}"
            if required:
                raise RuntimeError(f"required Microsoft runtime is not loadable: {name}") from exc
    return {"applicable": True, "libraries": libraries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--cuda-selftest", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-temporal", action="store_true")
    parser.add_argument("--require-rust", action="store_true")
    args = parser.parse_args()

    checks = {
        "python": _check(_python_check),
        "packages": _check(_packages_check),
        "windows_runtime": _check(_windows_runtime_check),
        "shared_memory": _check(_shared_memory_check),
        "native_temporal": _check(_temporal_check),
        "rust_runtime": _check(_rust_check),
        "opencv": _check(lambda: _opencv_check(args.cuda_selftest)),
    }
    result = {
        "schema_version": 1,
        "platform": platform.platform(),
        "project_root": str(ROOT),
        "checks": checks,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    failures = []
    for required in ("python", "packages", "windows_runtime", "shared_memory", "opencv"):
        if checks[required]["status"] != "ok":
            failures.append(required)
    if args.require_temporal and checks["native_temporal"]["status"] != "ok":
        failures.append("native_temporal")
    if args.require_rust and checks["rust_runtime"]["status"] != "ok":
        failures.append("rust_runtime")
    if args.require_cuda:
        opencv = checks["opencv"]
        if opencv["status"] != "ok" or not opencv.get("cuda_build"):
            failures.append("cuda")
        elif args.cuda_selftest and not opencv.get("cuda_selftest", {}).get("success"):
            failures.append("cuda_selftest")
    if failures:
        print(f"Required diagnostics failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
