"""
MegaGlitchNode Main Entry Point

CRITICAL: Bootstrap OpenCV CUDA BEFORE any cv2 imports.
Keep startup side effects inside a function so multiprocessing child processes
do not rerun CUDA verification during module import on Windows.
"""

import hashlib
import multiprocessing as mp
import json
import os
from pathlib import Path
import queue
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SPLASH_IMAGE_PATH = Path(
    os.environ.get("MEGAGLITCH_SPLASH_IMAGE", PROJECT_ROOT / "assets" / "MegaGlitch.png")
)
SPLASH_DURATION_MS = 5_000
SPLASH_TRANSPARENT_COLOR = "#ff00ff"


def make_splash_image_transparent(image, threshold: int = 14):
    """Convert near-black pixels to transparent so the splash has no visible box."""
    rgba = image.convert("RGBA")
    pixels = list(rgba.getdata())
    transparent_pixels = []
    for red, green, blue, alpha in pixels:
        if red <= threshold and green <= threshold and blue <= threshold:
            transparent_pixels.append((red, green, blue, 0))
        else:
            transparent_pixels.append((red, green, blue, alpha))
    rgba.putdata(transparent_pixels)
    return rgba


def show_startup_splash(
    image_path: Path = SPLASH_IMAGE_PATH,
    duration_ms: int = SPLASH_DURATION_MS,
):
    """Display a centered splash image before the Dear PyGui app launches."""
    if not image_path.exists():
        print(f"[WARN] Splash image not found: {image_path}")
        return

    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except Exception as exc:
        print(f"[WARN] Splash screen unavailable: {exc}")
        return

    try:
        root = tk.Tk()
        root.overrideredirect(True)
        root.configure(bg=SPLASH_TRANSPARENT_COLOR)
        root.attributes("-topmost", True)
        try:
            root.wm_attributes("-transparentcolor", SPLASH_TRANSPARENT_COLOR)
        except Exception:
            pass

        image = Image.open(image_path)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        image.thumbnail(
            (
                max(320, int(screen_width * 0.8)),
                max(180, int(screen_height * 0.8)),
            ),
            Image.LANCZOS,
        )
        image = make_splash_image_transparent(image)

        photo = ImageTk.PhotoImage(image)
        label = tk.Label(
            root,
            image=photo,
            bg=SPLASH_TRANSPARENT_COLOR,
            bd=0,
            highlightthickness=0,
        )
        label.image = photo
        label.pack()

        root.update_idletasks()
        window_width = photo.width()
        window_height = photo.height()
        x_pos = max((screen_width - window_width) // 2, 0)
        y_pos = max((screen_height - window_height) // 2, 0)
        root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        root.after(max(0, int(duration_ms)), root.destroy)
        root.mainloop()
    except Exception as exc:
        print(f"[WARN] Failed to show splash screen: {exc}")


def run_diagnostics_only() -> int:
    """Verify the import-sensitive runtime without starting the interactive UI."""
    result = {
        "ok": False,
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
    }
    try:
        import cv2

        from megaglitch.native_runtime import native_runtime_available
        from megaglitch.native_temporal import native_available

        cuda_devices = (
            int(cv2.cuda.getCudaEnabledDeviceCount())
            if hasattr(cv2, "cuda")
            else 0
        )
        result.update(
            {
                "cv2_path": str(getattr(cv2, "__file__", "")),
                "cv2_version": str(cv2.__version__),
                "cuda_devices": cuda_devices,
                "native_temporal": bool(native_available()),
                "native_runtime": bool(native_runtime_available()),
            }
        )
        result["ok"] = bool(
            cuda_devices > 0
            and result["native_temporal"]
            and result["native_runtime"]
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


def run_worker_diagnostics_only() -> int:
    """Exercise and cleanly stop the real shared-transport Python worker."""
    from megaglitch.worker_lifecycle import WorkerProcessLifecycle
    from megaglitch.shared_frame_transport import SharedFrameTransportOwner
    from megaglitchfm_app_new import glitch_process_worker
    import numpy as np

    result = {
        "ok": False,
        "frozen": bool(getattr(sys, "frozen", False)),
        "frame_transport": None,
        "frame_round_trip": {"ok": False},
        "reset_round_trip": False,
    }
    cleanup_errors = []
    raw_queue = None
    output_queue = None
    owner = None
    lifecycle = None
    worker_exercised = False
    try:
        context = mp.get_context()
        raw_queue = context.Queue(maxsize=2)
        output_queue = context.Queue(maxsize=2)
        stop_event = context.Event()
        owner = SharedFrameTransportOwner(
            context,
            16,
            12,
            input_slots=2,
            output_slots=1,
        )
        lifecycle = WorkerProcessLifecycle(
            context,
            stop_event,
            startup_timeout=30.0,
            graceful_timeout=5.0,
            terminate_timeout=2.0,
        )
        lifecycle.start(
            glitch_process_worker,
            (raw_queue, output_queue, stop_event, None, owner.spec()),
        )
        ready_transitions = [
            transition
            for transition in lifecycle.transitions
            if transition.get("state") == "ready"
        ]
        result["frame_transport"] = (
            ready_transitions[-1].get("frame_transport")
            if ready_transitions
            else None
        )
        if result["frame_transport"] != "shared_v1":
            raise RuntimeError(
                "worker did not report required shared_v1 frame transport"
            )

        diagnostic_frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
        diagnostic_payload = owner.encode_payload(
            {
                "image": diagnostic_frame,
                "secondary_image": None,
                "pipeline": [],
                "pipeline_b": [],
                "master_mix": 0.0,
                "extra_params": {},
                "use_cuda": False,
                "fallback_width": 16,
                "fallback_height": 12,
                "timestamp": 0.0,
                "render_mode": "live",
                "frame_role": "master_ingress",
                "master_resolution": {"width": 16, "height": 12},
            },
            block=True,
            timeout=5.0,
        )
        if diagnostic_payload is None:
            raise RuntimeError("could not reserve the diagnostic shared input slot")
        raw_queue.put(diagnostic_payload, timeout=5.0)
        encoded_result = output_queue.get(timeout=10.0)
        decoded_result = owner.decode_result(encoded_result)
        diagnostic_output = np.asarray(decoded_result.get("frame"))
        input_transport = diagnostic_payload.get("frame_transport")
        output_transport = decoded_result.get("frame_transport")
        output_hash = hashlib.sha256(
            memoryview(np.ascontiguousarray(diagnostic_output))
        ).hexdigest()
        expected_hash = "266d05fcd17ddd9f96f5cb7eae3c9f486fd5697a2b0198586a82af2cdc7c0558"
        if (
            input_transport != "shared_v1"
            or output_transport != "shared_v1"
            or diagnostic_output.shape != diagnostic_frame.shape
            or not np.array_equal(diagnostic_output, diagnostic_frame)
            or output_hash != expected_hash
        ):
            raise RuntimeError("shared_v1 deterministic frame round trip did not match")
        result["frame_round_trip"] = {
            "ok": True,
            "input_transport": input_transport,
            "output_transport": output_transport,
            "shape": list(diagnostic_output.shape),
            "sha256": output_hash,
        }

        # This control-message round trip exercises both bounded queues after
        # the shared frame and forms a barrier after input-slot release.
        raw_queue.put("RESET", timeout=5.0)
        reset_response = output_queue.get(timeout=10.0)
        if reset_response != "RESET":
            raise RuntimeError(
                f"worker RESET round trip returned {reset_response!r}"
            )
        result["reset_round_trip"] = True
        worker_exercised = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if lifecycle is not None:
            try:
                lifecycle.stop()
            except Exception as exc:
                cleanup_errors.append(
                    f"worker stop failed: {type(exc).__name__}: {exc}"
                )

        # Release any queued shared descriptors before unlinking their owner.
        if raw_queue is not None:
            while True:
                try:
                    pending_input = raw_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception as exc:
                    cleanup_errors.append(
                        f"input queue drain failed: {type(exc).__name__}: {exc}"
                    )
                    break
                if owner is not None:
                    try:
                        owner.release_input_payload(pending_input)
                    except Exception as exc:
                        cleanup_errors.append(
                            "input descriptor release failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
        if output_queue is not None:
            while True:
                try:
                    pending_output = output_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception as exc:
                    cleanup_errors.append(
                        f"output queue drain failed: {type(exc).__name__}: {exc}"
                    )
                    break
                if owner is not None:
                    try:
                        owner.release_output_result(pending_output)
                    except Exception as exc:
                        cleanup_errors.append(
                            "output descriptor release failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

        if lifecycle is not None:
            result["lifecycle"] = lifecycle.snapshot()
        if owner is not None:
            try:
                result["shared_transport"] = owner.get_statistics()
            except Exception as exc:
                cleanup_errors.append(
                    f"shared statistics failed: {type(exc).__name__}: {exc}"
                )

        for label, managed_queue in (
            ("input", raw_queue),
            ("output", output_queue),
        ):
            if managed_queue is None:
                continue
            try:
                managed_queue.close()
                managed_queue.cancel_join_thread()
            except Exception as exc:
                cleanup_errors.append(
                    f"{label} queue close failed: {type(exc).__name__}: {exc}"
                )
        if owner is not None:
            try:
                owner.close()
            except Exception as exc:
                cleanup_errors.append(
                    f"shared owner close failed: {type(exc).__name__}: {exc}"
                )

    if cleanup_errors:
        result["cleanup_errors"] = cleanup_errors
    lifecycle_result = result.get("lifecycle", {})
    shared_result = result.get("shared_transport", {})
    result["ok"] = bool(
        worker_exercised
        and result["frame_round_trip"].get("ok")
        and result["reset_round_trip"]
        and result["frame_transport"] == "shared_v1"
        and lifecycle_result.get("state") == "stopped"
        and lifecycle_result.get("exitcode") == 0
        and not lifecycle_result.get("forced_termination")
        and lifecycle_result.get("error") is None
        and lifecycle_result.get("status_channel_error") is None
        and shared_result.get("input_slots_in_use") == 0
        and shared_result.get("output_slots_in_use") == 0
        and not cleanup_errors
    )

    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


def run():
    from megaglitchnode.opencv_cuda_bootstrap import (
        bootstrap_for_dev_defaults,
        invalidate_gpu_caches,
        has_cuda_build,
    )

    bootstrap_for_dev_defaults()
    invalidate_gpu_caches()

    if "--diagnose-only" in sys.argv:
        return run_diagnostics_only()
    if "--diagnose-worker" in sys.argv:
        return run_worker_diagnostics_only()

    show_startup_splash()

    from megaglitchfm_app_new import main as app_main

    build_ok, build_message = has_cuda_build()
    if build_ok:
        print(f"[OK] {build_message}; worker process will verify runtime access")
    else:
        print(f"[WARN] {build_message}; using CPU fallback")

    app_main()
    return 0


if __name__ == "__main__":
    # PyInstaller Windows children can start without inherited console stream
    # objects. multiprocessing's bootstrap flushes them before target unpickle.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    mp.freeze_support()
    raise SystemExit(run())
