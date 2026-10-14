import logging
from collections import Counter

import cv2
import numpy as np
from PIL import Image
import gpu_ops as gpo

from effects_registry import apply_node_effect


def record_frame_copy(copy_stats, category, frame):
    """Record an explicit full-frame copy without changing frame ownership."""
    if copy_stats is None or frame is None:
        return
    try:
        nbytes = int(frame.nbytes)
    except (AttributeError, TypeError, ValueError):
        return
    bucket = copy_stats.setdefault(str(category), {"count": 0, "bytes": 0})
    bucket["count"] = int(bucket.get("count", 0)) + 1
    bucket["bytes"] = int(bucket.get("bytes", 0)) + nbytes


def _to_rgb_np(frame):
    if frame is None:
        return None
        
    try:
        import cv2
        if hasattr(cv2, 'cuda_GpuMat') and isinstance(frame, cv2.cuda_GpuMat):
            frame = frame.download()
    except ImportError:
        pass
        
    try:
        from megaglitchnode.gpu import Frame
        if isinstance(frame, Frame):
            frame = frame.numpy()
    except ImportError:
        pass
        
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"), dtype=np.uint8)
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported frame shape: {arr.shape}")
        return arr
    raise ValueError(f"Unsupported frame type: {type(frame)}")


def _resolve_frame(frame):
    if frame is None:
        return None
    if hasattr(cv2, 'cuda_GpuMat') and isinstance(frame, cv2.cuda_GpuMat):
        return frame
    try:
        from megaglitchnode.gpu import Frame
        if isinstance(frame, Frame):
            return frame
    except ImportError:
        pass
    return _to_rgb_np(frame)


def _apply_multimodal_selection(
    input_frame,
    processed_frame,
    params,
    worker_state,
    effect_label,
):
    """Spatially gate destructive output with one upstream typed analysis mask.

    ``multimodal_amount=0`` preserves legacy full-frame behavior.  At ``1`` the
    effect is present only where the master-resolution modulation mask selects
    it.  The operation never resizes the input, result, or analysis map.
    """
    if effect_label in {
        "Multimodal Analysis",
        "Controlled Recursive Feedback",
        "Structural Restoration",
    }:
        return processed_frame
    try:
        amount = float(np.clip(params.get("multimodal_amount", 0.0), 0.0, 1.0))
    except (TypeError, ValueError):
        amount = 0.0
    if amount <= 1e-6 or worker_state is None:
        return processed_frame
    effect_states = getattr(worker_state, "effect_states", None)
    analysis = (
        effect_states.get("multimodal_active")
        if isinstance(effect_states, dict)
        else None
    )
    if analysis is None:
        return processed_frame

    base = _to_rgb_np(gpo.ensure_cpu(input_frame))
    effected = _to_rgb_np(gpo.ensure_cpu(processed_frame))
    if base.shape != effected.shape:
        raise ValueError(
            f"multimodal node output shape {effected.shape} does not match "
            f"master input {base.shape}"
        )
    analysis.assert_master_integrity()
    mask = np.asarray(analysis.modulation_mask, dtype=np.float32)
    if mask.shape != base.shape[:2]:
        raise ValueError(
            f"multimodal mask shape {mask.shape} does not match master "
            f"shape {base.shape[:2]}"
        )
    gate = (1.0 - amount) + amount * np.clip(mask, 0.0, 1.0)
    selected = (
        base.astype(np.float32) * (1.0 - gate[..., None])
        + effected.astype(np.float32) * gate[..., None]
    )
    return np.ascontiguousarray(np.clip(selected, 0, 255).astype(np.uint8))


def _to_rgba_np(frame):
    if frame is None:
        return None
        
    try:
        import cv2
        if hasattr(cv2, 'cuda_GpuMat') and isinstance(frame, cv2.cuda_GpuMat):
            frame = frame.download()
    except ImportError:
        pass
        
    try:
        from megaglitchnode.gpu import Frame
        if isinstance(frame, Frame):
            frame = frame.numpy()
    except ImportError:
        pass
        
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGBA"), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported frame shape: {arr.shape}")
    if arr.shape[2] == 3:
        alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    elif arr.shape[2] != 4:
        raise ValueError(f"Unsupported frame shape: {arr.shape}")
    return arr


def _blank_frame(size):
    width, height = size
    return np.zeros((max(1, int(height)), max(1, int(width)), 3), dtype=np.uint8)


def _bubble_sim2_beauty(outputs, source_frame, width, height, time_value=0.0):
    mask = outputs["bubble_mask"][..., 0].astype(np.float32) / 255.0
    normal = outputs["bubble_normal"].astype(np.float32) / 255.0
    refraction = outputs["bubble_refraction"].astype(np.float32) / 255.0
    highlight = outputs["bubble_highlight"].astype(np.float32) / 255.0
    depth = outputs["bubble_depth"][..., 0].astype(np.float32) / 255.0

    if source_frame is None:
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
        top = np.array([18.0, 54.0, 68.0], dtype=np.float32)[None, None, :]
        bottom = np.array([2.0, 10.0, 18.0], dtype=np.float32)[None, None, :]
        source = (top * (1.0 - y) + bottom * y).repeat(width, axis=1)
        shafts = np.clip(
            0.5
            + 0.5
            * np.sin(
                x[..., 0] * 21.0
                + y[..., 0] * 4.0
                + float(time_value) * 0.015
            ),
            0.0,
            1.0,
        )
        shafts = np.power(shafts, 6.0) * np.square(1.0 - y[..., 0])
        caustics = (
            0.5
            + 0.5
            * np.sin(
                x[..., 0] * 38.0
                + np.sin(y[..., 0] * 14.0 - float(time_value) * 0.025) * 2.2
            )
        )
        atmosphere = shafts * 18.0 + caustics * np.square(1.0 - y[..., 0]) * 4.0
        source += atmosphere[..., None] * np.array([0.35, 0.85, 1.0], dtype=np.float32)
        vignette = 1.0 - 0.20 * np.square((x[..., 0] - 0.5) * 2.0)
        source *= vignette[..., None]
        source = np.clip(source, 0, 255)
    else:
        source = _to_rgb_np(source_frame)
        if source.shape[:2] != (height, width):
            source = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
        source = source.astype(np.float32)

    offset = (refraction[..., :2] - 0.5) * 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = np.clip(xx + offset[..., 0] * width * 0.030 * mask, 0, width - 1)
    map_y = np.clip(yy + offset[..., 1] * height * 0.008 * mask, 0, height - 1)
    refracted = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    atmospheric = (0.04 + depth * 0.10) * mask
    beauty = refracted * (1.0 - atmospheric[..., None])
    rim = refraction[..., 2]
    lens = mask * normal[..., 2]
    beauty *= 1.0 - (mask * 0.07 + rim * 0.05)[..., None]
    beauty += highlight * 255.0
    beauty += lens[..., None] * np.array([8.0, 17.0, 22.0], dtype=np.float32)
    beauty += rim[..., None] * np.array([12.0, 25.0, 31.0], dtype=np.float32)
    return np.clip(np.rint(beauty), 0, 255).astype(np.uint8)


def _generate_bubble_sim2_outputs(width, height, time_val, params, worker_state, node_key, input_frames):
    from megaglitch.bubble_sim2_cpu import generate_bubble_sim2_cpu_outputs

    source_frame = (input_frames or {}).get("image_in")
    state_key = f"bubble_sim2::{node_key if node_key is not None else 'default'}"
    effect_states = getattr(worker_state, "effect_states", None) if worker_state is not None else None
    if effect_states is None and worker_state is not None:
        worker_state.effect_states = {}
        effect_states = worker_state.effect_states

    state = effect_states.get(state_key) if isinstance(effect_states, dict) else None
    gpu_failed = bool(state and state.get("gpu_failed"))
    runtime = state.get("runtime") if state else None

    if not gpu_failed and worker_state is not None:
        try:
            import moderngl
            from megaglitch.nodes import BubbleSimGenerator2Runtime

            context = getattr(worker_state, "lua_gl_ctx", None)
            if context is None:
                context = moderngl.create_standalone_context(require=330)
                worker_state.lua_gl_ctx = context

            particle_size = int(np.clip(params.get("particle_texture_size", 64), 16, 512))
            render_scale = float(np.clip(params.get("render_scale", 0.5), 0.25, 1.0))
            render_width = max(64, int(round(width * render_scale)))
            render_height = max(64, int(round(height * render_scale)))
            if runtime is None:
                runtime = BubbleSimGenerator2Runtime(
                    context, render_width, render_height, particle_size
                )
                state = {"runtime": runtime, "gpu_failed": False, "mode": runtime.runtime_mode}
                effect_states[state_key] = state
                logging.info("Bubble Sim Gen (2) node %s runtime: %s", node_key, runtime.runtime_mode)
            else:
                runtime.resize(render_width, render_height)

            runtime_params = dict(params)
            requested_density = float(np.clip(runtime_params.get("density", 0.12), 0.0, 1.0))
            # Keep the visible population roughly stable when users increase
            # the state texture resolution for smoother simulation.
            runtime_params["density"] = float(
                np.clip(requested_density * (64.0 / particle_size) ** 2, 0.0, 1.0)
            )
            runtime.apply_params(runtime_params)
            last_time = float(state.get("last_time", time_val))
            dt = max(0.0, min(float(time_val) - last_time, 1.0 / 15.0))
            if dt <= 0.0:
                dt = 1.0 / 60.0
            runtime.update(float(time_val), dt, float(time_val))
            gpu_outputs = runtime.render_and_readback()
            state["last_time"] = float(time_val)
            if gpu_outputs is not None:
                # The particle/simulation lattice may run reduced, but all
                # image-domain outputs are promoted at this explicit boundary
                # before source compositing or temporal echo ownership.
                if (render_width, render_height) != (width, height):
                    gpu_outputs = {
                        name: cv2.resize(
                            frame,
                            (width, height),
                            interpolation=(
                                cv2.INTER_LINEAR
                                if name not in {"bubble_mask", "bubble_depth"}
                                else cv2.INTER_AREA
                            ),
                        )
                        for name, frame in gpu_outputs.items()
                    }
                beauty = _bubble_sim2_beauty(
                    gpu_outputs,
                    source_frame,
                    width,
                    height,
                    time_value=time_val,
                )
                echo_mix = float(np.clip(params.get("mutation_strength", 0.0), 0.0, 1.0))
                echo_mix *= float(np.clip(params.get("mutation_echo", 0.35), 0.0, 1.0)) * 0.35
                previous_beauty = state.get("previous_beauty")
                if echo_mix > 0.0 and previous_beauty is not None and previous_beauty.shape == beauty.shape:
                    beauty = np.clip(
                        beauty.astype(np.float32) * (1.0 - echo_mix)
                        + previous_beauty.astype(np.float32) * echo_mix,
                        0,
                        255,
                    ).astype(np.uint8)
                state["previous_beauty"] = beauty.copy()
                gpu_outputs["image_out"] = beauty
                gpu_outputs["final_frame"] = beauty
                debug_key = {
                    "mask": "bubble_mask",
                    "normal": "bubble_normal",
                    "depth": "bubble_depth",
                    "refraction": "bubble_refraction",
                    "highlight": "bubble_highlight",
                }.get(str(params.get("debug_mode", "composite")), "image_out")
                gpu_outputs["bubble_debug"] = gpu_outputs[debug_key]
                return gpu_outputs
        except Exception as exc:
            logging.warning("Bubble Sim Gen (2) GPU unavailable; using CPU fallback: %s", exc)
            if runtime is not None:
                try:
                    runtime.destroy()
                except Exception:
                    pass
            if isinstance(effect_states, dict):
                effect_states[state_key] = {
                    "runtime": None,
                    "gpu_failed": True,
                    "mode": "CPU fallback",
                }

    outputs = generate_bubble_sim2_cpu_outputs(
        width, height, time_val, params, source=source_frame
    )
    if isinstance(effect_states, dict):
        state = effect_states.get(state_key)
        if state is not None:
            echo_mix = float(np.clip(params.get("mutation_strength", 0.0), 0.0, 1.0))
            echo_mix *= float(np.clip(params.get("mutation_echo", 0.35), 0.0, 1.0)) * 0.35
            previous_beauty = state.get("previous_beauty")
            beauty = outputs["image_out"]
            if echo_mix > 0.0 and previous_beauty is not None and previous_beauty.shape == beauty.shape:
                beauty = np.clip(
                    beauty.astype(np.float32) * (1.0 - echo_mix)
                    + previous_beauty.astype(np.float32) * echo_mix,
                    0,
                    255,
                ).astype(np.uint8)
                outputs["image_out"] = beauty
                outputs["final_frame"] = beauty
            state["previous_beauty"] = beauty.copy()
    if isinstance(effect_states, dict):
        if runtime is None or gpu_failed:
            state = effect_states.setdefault(
                state_key,
                {"runtime": None, "gpu_failed": True, "mode": "CPU fallback"},
            )
            state["mode"] = "CPU fallback"
    return outputs


def _resolve_base_frame(primary_frame, secondary_frame, fallback_size):
    primary = _to_rgb_np(primary_frame) if primary_frame is not None else None
    secondary = _to_rgb_np(secondary_frame) if secondary_frame is not None else None
    base = primary if primary is not None else secondary
    if base is None:
        base = _blank_frame(fallback_size)
    return primary, secondary, base


def _generator_output_key(output_name):
    name = str(output_name or "image_out")
    if name == "image_out":
        return "final_frame"
    return name


def _frame_signature(frame):
    if frame is None:
        return None
    try:
        # Avoid downloading to CPU just for signature if possible
        if hasattr(cv2, 'cuda_GpuMat') and isinstance(frame, cv2.cuda_GpuMat):
            return (frame.rows, frame.cols, 0, 0, 0)
    except Exception:
        pass
    arr = _to_rgb_np(gpo.ensure_cpu(frame)) if frame is not None else None
    if arr is None:
        return None
    h, w = arr.shape[:2]
    sample_y = max(1, h // 16)
    sample_x = max(1, w // 16)
    sample = arr[::sample_y, ::sample_x]
    return (
        int(h),
        int(w),
        int(sample[..., 0].sum()),
        int(sample[..., 1].sum()),
        int(sample[..., 2].sum()),
    )


def _params_signature(params):
    return tuple(sorted((str(k), repr(v)) for k, v in (params or {}).items()))


def _normalize_mask_frame(frame, width, height):
    if frame is None:
        return None
    mask = _to_rgb_np(gpo.ensure_cpu(frame))
    if mask.shape[:2] != (height, width):
        raise ValueError(
            f"mask shape {mask.shape[:2]} does not match master shape {(height, width)}"
        )
    gray = mask.astype(np.float32)[..., 0] * 0.2126 + mask.astype(np.float32)[..., 1] * 0.7152 + mask.astype(np.float32)[..., 2] * 0.0722
    return np.clip(gray / 255.0, 0.0, 1.0).astype(np.float32)


def _smooth_gate(mask, threshold, softness):
    softness = max(1e-4, float(softness))
    lo = float(threshold) - softness * 0.5
    hi = float(threshold) + softness * 0.5
    if hi <= lo:
        return (mask >= float(threshold)).astype(np.float32)
    gate = np.clip((mask - lo) / (hi - lo), 0.0, 1.0)
    return gate * gate * (3.0 - 2.0 * gate)


def _coalesce_frame(*frames):
    for frame in frames:
        if frame is not None:
            return _resolve_frame(frame)
    return None


def apply_genesis_priority_composite(input_frames, params, fallback_frame=None):
    base = _coalesce_frame(input_frames.get("base_image"), fallback_frame, input_frames.get("overlay_image"))
    overlay = _coalesce_frame(input_frames.get("overlay_image"), input_frames.get("base_image"), fallback_frame)
    if base is None and overlay is None:
        return None
    if base is None:
        base = np.zeros_like(overlay, dtype=np.uint8)
    if overlay is None:
        overlay = np.zeros_like(base, dtype=np.uint8)
    if overlay.shape[:2] != base.shape[:2]:
        raise ValueError(
            f"overlay shape {overlay.shape[:2]} does not match master shape {base.shape[:2]}"
        )

    h, w = base.shape[:2]
    priority = _normalize_mask_frame(input_frames.get("priority_mask"), w, h)
    if priority is None:
        priority = np.zeros((h, w), dtype=np.float32)
    luma = _normalize_mask_frame(input_frames.get("luma_mask"), w, h)
    corruption = _normalize_mask_frame(input_frames.get("corruption_mask"), w, h)

    if bool(params.get("invert_priority", False)):
        priority = 1.0 - priority

    mode = str(params.get("mode", "strict_priority") or "strict_priority")
    threshold = float(params.get("threshold", 0.55) or 0.55)
    softness = float(params.get("softness", 0.18) or 0.18)
    mix = float(np.clip(params.get("mix", 1.0) or 1.0, 0.0, 1.0))
    luma_influence = float(np.clip(params.get("luma_influence", 0.35) or 0.35, 0.0, 1.0))
    corruption_influence = float(np.clip(params.get("corruption_influence", 0.25) or 0.25, 0.0, 1.0))

    if mode == "soft_priority":
        alpha = _smooth_gate(priority, threshold * 0.75, max(softness, 0.08))
    elif mode == "priority_plus_luma":
        alpha = np.clip(priority + ((luma if luma is not None else priority) * luma_influence), 0.0, 1.0)
        alpha = _smooth_gate(alpha, threshold * 0.85, max(softness, 0.06))
    else:
        alpha = _smooth_gate(priority, threshold, max(softness * 0.35, 0.02))

    if corruption is not None:
        alpha = np.clip(alpha - corruption * corruption_influence * 0.5, 0.0, 1.0)
    alpha = np.clip(alpha * mix, 0.0, 1.0)[..., None]

    out = base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_genesis_mask_key(input_frames, params, fallback_frame=None):
    image = _coalesce_frame(input_frames.get("image_in"), fallback_frame)
    if image is None:
        return None
    h, w = image.shape[:2]
    primary_mask = _normalize_mask_frame(input_frames.get("mask_in"), w, h)
    luma_mask = _normalize_mask_frame(input_frames.get("luma_mask"), w, h)
    priority_mask = _normalize_mask_frame(input_frames.get("priority_mask"), w, h)
    corruption_mask = _normalize_mask_frame(input_frames.get("corruption_mask"), w, h)

    mode = str(params.get("mode", "multiply_mask") or "multiply_mask")
    threshold = float(params.get("threshold", 0.5) or 0.5)
    softness = float(params.get("softness", 0.12) or 0.12)
    strength = float(np.clip(params.get("strength", 1.0) or 1.0, 0.0, 1.0))
    invert = bool(params.get("invert", False))

    if mode == "luma_gate":
        gate = luma_mask if luma_mask is not None else primary_mask
    elif mode == "corruption_gate":
        gate = corruption_mask if corruption_mask is not None else primary_mask
    else:
        gate = primary_mask

    if gate is None:
        gate = np.ones((h, w), dtype=np.float32)

    if mode == "threshold_key":
        gate = _smooth_gate(gate, threshold, softness)
    elif mode == "invert_key":
        gate = 1.0 - _smooth_gate(gate, threshold, softness)
    elif mode == "corruption_gate":
        gate = _smooth_gate(gate, threshold * 0.75, max(softness, 0.05))

    if priority_mask is not None and mode in {"multiply_mask", "threshold_key", "invert_key"}:
        gate = np.clip(np.maximum(gate, priority_mask * 0.5), 0.0, 1.0)
    if corruption_mask is not None and mode != "corruption_gate":
        gate = np.clip(gate * (1.0 - corruption_mask * 0.35), 0.0, 1.0)
    if invert:
        gate = 1.0 - gate

    gate = np.clip(gate, 0.0, 1.0)[..., None]
    gated = image.astype(np.float32) * gate
    out = image.astype(np.float32) * (1.0 - strength) + gated * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def _validate_genesis_outputs(outputs):
    required = {
        "final_frame",
        "plane_a",
        "plane_b",
        "sprite_layer",
        "priority_mask",
        "palette_index_buffer",
        "luma_mask",
        "corruption_mask",
        "scanline_mod_buffer",
        "crt_prepass",
    }
    missing = [key for key in required if key not in outputs]
    if missing:
        raise ValueError(f"Genesis VDP outputs missing required buffers: {missing}")

    baseline_shape = None
    for key, frame in outputs.items():
        arr = _to_rgb_np(frame)
        if arr.dtype != np.uint8:
            raise ValueError(f"Genesis output '{key}' must be uint8")
        if baseline_shape is None:
            baseline_shape = arr.shape
        elif arr.shape != baseline_shape:
            raise ValueError(f"Genesis output '{key}' shape {arr.shape} does not match {baseline_shape}")


def generate_node_outputs(label, width, height, time_val, params, worker_state=None, node_key=None, input_frames=None):
    if params is not None and str(params.get("enabled", True)).lower() == "false":
        passthrough = next(
            (frame for frame in (input_frames or {}).values() if frame is not None),
            None,
        )
        frame = _to_rgb_np(passthrough) if passthrough is not None else _blank_frame((width, height))
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"generator passthrough shape {frame.shape[:2]} does not match "
                f"master shape {(height, width)}"
            )
        return {"image_out": frame, "final_frame": frame}
        
    from megaglitchfm_app_new import (
        generate_colony_reef,
        generate_checkerboard,
        generate_fractal,
        generate_gradient,
        generate_macroblock_echo,
        generate_bubble_sim_preview,
        generate_perlin_noise,
        generate_plasma,
        generate_scan_feedback,
        generate_voronoi,
        generate_xor_lattice,
    )
    from genesis_vdp_renderer import render_genesis_vdp_buffers

    if "Perlin" in label:
        frame = _to_rgb_np(generate_perlin_noise(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if "Gradient" in label:
        frame = _to_rgb_np(generate_gradient(width, height, params))
        return {"image_out": frame, "final_frame": frame}
    if "Plasma" in label:
        frame = _to_rgb_np(generate_plasma(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if "Voronoi" in label:
        frame = _to_rgb_np(generate_voronoi(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if "Fractal" in label:
        frame = _to_rgb_np(generate_fractal(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if "Checkerboard" in label:
        frame = _to_rgb_np(generate_checkerboard(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if "Scan Feedback" in label:
        frame = _to_rgb_np(generate_scan_feedback(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if "XOR Lattice" in label:
        frame = _to_rgb_np(generate_xor_lattice(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if "Macroblock Echo" in label:
        frame = _to_rgb_np(generate_macroblock_echo(width, height, time_val, params))
        return {"image_out": frame, "final_frame": frame}
    if label == "Bubble Sim Gen (2)":
        return _generate_bubble_sim2_outputs(
            width,
            height,
            time_val,
            params or {},
            worker_state,
            node_key,
            input_frames,
        )
    if "Bubble Sim" in label:
        frame = _to_rgb_np(generate_bubble_sim_preview(width, height, time_val, params))
        source_frame = (input_frames or {}).get("image_in")
        composite = frame
        if source_frame is not None:
            source = _to_rgb_np(source_frame)
            if source.shape[:2] != (height, width):
                source = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
            # Bubble composite is rendered on black, so a screen blend preserves
            # the connected image while adding the luminous bubble layer.
            source_f = source.astype(np.float32) / 255.0
            bubble_f = frame.astype(np.float32) / 255.0
            composite = np.rint(np.clip(
                (1.0 - (1.0 - source_f) * (1.0 - bubble_f)) * 255.0,
                0.0,
                255.0,
            )).astype(np.uint8)
        return {
            "image_out": composite,
            "final_frame": composite,
            "bubble_mask": frame,
            "bubble_normal": frame,
            "bubble_depth": frame,
            "bubble_refraction": frame,
            "bubble_highlight": frame,
            "bubble_debug": frame,
        }
    if "Colony Reef" in label:
        frame = _to_rgb_np(generate_colony_reef(width, height, time_val, params, worker_state=worker_state, node_key=node_key))
        return {"image_out": frame, "final_frame": frame}
    if "Genesis VDP" in label:
        outputs = render_genesis_vdp_buffers(
            width,
            height,
            time_val,
            params,
            worker_state=worker_state,
            node_key=node_key,
            input_frames=input_frames,
        )
        _validate_genesis_outputs(outputs)
        outputs["image_out"] = outputs.get("final_frame")
        return outputs
    raise ValueError(f"Unknown generator type: {label}")


def generate_node_frame(label, width, height, time_val, params, worker_state=None, node_key=None, input_frames=None, output_name="image_out"):
    outputs = generate_node_outputs(
        label,
        width,
        height,
        time_val,
        params,
        worker_state=worker_state,
        node_key=node_key,
        input_frames=input_frames,
    )
    target_key = _generator_output_key(output_name)
    frame = outputs.get(target_key)
    if frame is None:
        frame = outputs.get("image_out")
    if frame is None:
        frame = outputs.get("final_frame")
    return _to_rgb_np(frame)


def mix_node_frames(channel_a, channel_b, params):
    if params is not None and str(params.get("enabled", True)).lower() == "false":
        return channel_a

    if channel_a is None:
        return _to_rgb_np(gpo.ensure_cpu(channel_b)) if channel_b is not None else None
    if channel_b is None:
        return _to_rgb_np(gpo.ensure_cpu(channel_a))

    frame_a = _to_rgba_np(gpo.ensure_cpu(channel_a))
    frame_b = _to_rgba_np(gpo.ensure_cpu(channel_b))
    mix = float(max(0.0, min(1.0, params.get("mix", 0.5) or 0.5)))
    if mix <= 0.0:
        return frame_a[..., :3].copy()

    if frame_b.shape[:2] != frame_a.shape[:2]:
        raise ValueError(
            f"mixer inputs must share master shape: {frame_a.shape[:2]} != "
            f"{frame_b.shape[:2]}"
        )

    rgb_a = frame_a[..., :3].astype(np.float32)
    rgb_b = frame_b[..., :3].astype(np.float32)
    alpha_a = frame_a[..., 3:4].astype(np.float32)
    alpha_b = frame_b[..., 3:4].astype(np.float32)

    mode = str(params.get("blend_mode", "crossfade") or "crossfade").lower()
    if mode == "add":
        mixed_rgb = np.clip(rgb_a + rgb_b, 0, 255)
        mixed_alpha = np.maximum(alpha_a, alpha_b)
    elif mode == "multiply":
        mixed_rgb = (rgb_a * rgb_b) / 255.0
        mixed_alpha = np.maximum(alpha_a, alpha_b)
    elif mode == "screen":
        mixed_rgb = 255.0 - ((255.0 - rgb_a) * (255.0 - rgb_b) / 255.0)
        mixed_alpha = np.maximum(alpha_a, alpha_b)
    elif mode == "difference":
        mixed_rgb = np.abs(rgb_a - rgb_b)
        mixed_alpha = np.maximum(alpha_a, alpha_b)
    elif mode == "lighten":
        mixed_rgb = np.maximum(rgb_a, rgb_b)
        mixed_alpha = np.maximum(alpha_a, alpha_b)
    elif mode == "darken":
        mixed_rgb = np.minimum(rgb_a, rgb_b)
        mixed_alpha = np.maximum(alpha_a, alpha_b)
    else:
        mixed_rgb = rgb_b
        mixed_alpha = alpha_b

    out_rgb = rgb_a * (1.0 - mix) + mixed_rgb * mix
    out_alpha = alpha_a * (1.0 - mix) + mixed_alpha * mix
    out = np.concatenate(
        [np.clip(out_rgb, 0, 255), np.clip(out_alpha, 0, 255)],
        axis=2,
    ).astype(np.uint8)
    return out[..., :3]


def execute_flat_node_pipeline(
    pipeline,
    primary_frame,
    secondary_frame=None,
    extra_params=None,
    previous_frame=None,
    feedback_image=None,
    trail_image=None,
    frame_buffer=None,
    worker_state=None,
    fallback_size=(1280, 720),
    copy_stats=None,
    source_copy_policy="copy",
):
    if source_copy_policy not in {"copy", "borrow_unique"}:
        raise ValueError(f"Unsupported source copy policy: {source_copy_policy}")
    extra_params = dict(extra_params or {})
    current_previous_frame = previous_frame
    current_feedback_image = feedback_image
    current_trail_image = trail_image
    current_frame_buffer = frame_buffer if frame_buffer is not None else []

    primary_np, secondary_np, base_frame = _resolve_base_frame(primary_frame, secondary_frame, fallback_size)
    if worker_state is not None:
        worker_state.current_clean_source = base_frame
    frame_stack = []
    generator_cache = {}
    source_counts = Counter(
        effect_config.get("source_key", "primary")
        for effect_config in pipeline or []
        if effect_config.get("node_type") == "source"
    )

    for effect_config in pipeline or []:
        effect_label = effect_config.get("effect_type")
        if effect_label in ("Display", "Global Time Macros"):
            continue

        node_type = effect_config.get("node_type")
        effect_params = dict(effect_config.get("params", {}) or {})
        merged_params = {**effect_params, **extra_params}
        node_id = effect_config.get("node_id")
        if node_id is not None:
            merged_params["_node_id"] = node_id

        if node_type == "source":
            source_key = effect_config.get("source_key", "primary")
            source_frame = secondary_np if source_key == "secondary" else primary_np
            if source_frame is None:
                source_frame = _blank_frame((base_frame.shape[1], base_frame.shape[0]))
            if source_copy_policy == "borrow_unique" and source_counts[source_key] == 1:
                frame_stack.append(source_frame)
            else:
                copied_source = source_frame.copy()
                record_frame_copy(copy_stats, "initial_node", copied_source)
                frame_stack.append(copied_source)
            continue

        if effect_config.get("is_generator"):
            input_order = list(effect_config.get("input_order", []) or [])
            input_frames = {}
            for input_name in reversed(input_order):
                input_frames[input_name] = frame_stack.pop() if frame_stack else None
            input_frames = {name: input_frames.get(name) for name in input_order}

            reference_frame = next(
                (
                    frame for frame in list(input_frames.values()) + ([frame_stack[-1]] if frame_stack else []) + [base_frame]
                    if frame is not None
                ),
                base_frame,
            )
            height, width = reference_frame.shape[:2]
            time_val = float(merged_params.get("time_accum", merged_params.get("noise_warp_time", 0.0)) or 0.0)

            cache_key = (
                effect_label,
                effect_config.get("node_id"),
                int(width),
                int(height),
                round(float(time_val), 6),
                _params_signature(merged_params),
                tuple((name, _frame_signature(frame)) for name, frame in input_frames.items()),
            )
            outputs = generator_cache.get(cache_key)
            if outputs is None:
                outputs = generate_node_outputs(
                    effect_label,
                    width,
                    height,
                    time_val,
                    merged_params,
                    worker_state=worker_state,
                    node_key=effect_config.get("node_id"),
                    input_frames=input_frames,
                )
                generator_cache[cache_key] = outputs
                if effect_label and "Genesis VDP" in effect_label and worker_state is not None:
                    effect_states = getattr(worker_state, "effect_states", None)
                    if effect_states is not None and effect_config.get("node_id") is not None:
                        effect_states[f"genesis_outputs::{effect_config.get('node_id')}"] = {
                            key: _to_rgb_np(value).copy()
                            for key, value in outputs.items()
                            if key != "image_out"
                        }

            selected_output = _generator_output_key(effect_config.get("selected_output", "image_out"))
            selected_frame = outputs.get(selected_output)
            if selected_frame is None:
                logging.warning(
                    "Generator output '%s' missing on node '%s'; falling back to final frame.",
                    selected_output,
                    effect_label,
                )
                selected_frame = outputs.get("image_out")
            if selected_frame is None:
                selected_frame = outputs.get("final_frame")
            frame_stack.append(_resolve_frame(selected_frame))
            continue

        if node_type == "mixer" or effect_label == "Video Mixer":
            frame_b = frame_stack.pop() if frame_stack else None
            frame_a = frame_stack.pop() if frame_stack else base_frame
            frame_stack.append(_resolve_frame(mix_node_frames(frame_a, frame_b, effect_params)))
            continue

        if node_type in {"genesis_priority_composite", "genesis_mask_key"}:
            input_order = list(effect_config.get("input_order", []) or [])
            input_frames = {}
            for input_name in reversed(input_order):
                input_frames[input_name] = frame_stack.pop() if frame_stack else None
            input_frames = {name: input_frames.get(name) for name in input_order}
            fallback_frame = next((frame for frame in input_frames.values() if frame is not None), base_frame)
            if node_type == "genesis_priority_composite":
                composed = apply_genesis_priority_composite(input_frames, effect_params, fallback_frame=fallback_frame)
            else:
                composed = apply_genesis_mask_key(input_frames, effect_params, fallback_frame=fallback_frame)
            frame_stack.append(_resolve_frame(composed if composed is not None else fallback_frame))
            continue

        input_order = list(effect_config.get("input_order", []) or [])
        if node_type == "lua_multi_input" or len(input_order) > 1:
            input_frames = {}
            for input_name in reversed(input_order):
                input_frames[input_name] = frame_stack.pop() if frame_stack else None
            input_frames = {name: input_frames.get(name) for name in input_order}
            input_frame = input_frames.get("image_in")
            if input_frame is None:
                input_frame = base_frame
            lua_params = dict(merged_params)
            lua_params["_aux_input_frames"] = {
                name: frame
                for name, frame in input_frames.items()
                if name != "image_in" and frame is not None
            }
            try:
                (
                    processed_frame,
                    current_previous_frame,
                    current_feedback_image,
                    current_trail_image,
                ) = apply_node_effect(
                    effect_label,
                    input_frame,
                    lua_params,
                    current_previous_frame,
                    current_feedback_image,
                    current_trail_image,
                    current_frame_buffer,
                    worker_state=worker_state,
                )
                processed_frame = _apply_multimodal_selection(
                    input_frame,
                    processed_frame,
                    merged_params,
                    worker_state,
                    effect_label,
                )
                frame_stack.append(_resolve_frame(processed_frame))
            except Exception as exc:
                logging.error("Error applying Lua multi-input shader: %s", exc)
                frame_stack.append(_resolve_frame(input_frame))
            continue

        input_frame = frame_stack.pop() if frame_stack else base_frame
        try:
            (
                processed_frame,
                current_previous_frame,
                current_feedback_image,
                current_trail_image,
            ) = apply_node_effect(
                effect_label,
                input_frame,
                merged_params,
                current_previous_frame,
                current_feedback_image,
                current_trail_image,
                current_frame_buffer,
                worker_state=worker_state,
            )
            processed_frame = _apply_multimodal_selection(
                input_frame,
                processed_frame,
                merged_params,
                worker_state,
                effect_label,
            )
            frame_stack.append(_resolve_frame(processed_frame))
        except Exception as e:
            logging.error(f"Error applying effect {effect_label}: {e}")
            frame_stack.append(_resolve_frame(input_frame))

    output_frame = frame_stack[-1] if frame_stack else base_frame
    
    # Ensure the final output is a CPU numpy array
    final_output_cpu = _to_rgb_np(gpo.ensure_cpu(output_frame))
    
    return final_output_cpu, current_previous_frame, current_feedback_image, current_trail_image


def blend_decks(frame_a, frame_b, mix_ratio=0.5, mode="Normal Mix", fallback_size=(1280, 720)):
    """
    Blend two frames (Deck A and Deck B) using custom transition masks.
    """
    import cv2
    import numpy as np

    mix = float(np.clip(mix_ratio, 0.0, 1.0))

    if frame_a is None and frame_b is None:
        return _blank_frame(fallback_size)
    if frame_a is None:
        b = _to_rgb_np(frame_b).astype(np.float32)
        a = np.zeros_like(b)
    elif frame_b is None:
        a = _to_rgb_np(frame_a).astype(np.float32)
        b = np.zeros_like(a)
    else:
        a = _to_rgb_np(frame_a).astype(np.float32)
        b = _to_rgb_np(frame_b).astype(np.float32)

    # Decks are normalized once at ingress; blending may not repair resolution.
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"deck outputs must share master shape: {a.shape[:2]} != {b.shape[:2]}"
        )

    endpoint_epsilon = 1e-4
    if mix <= endpoint_epsilon:
        return a.astype(np.uint8)
    if mix >= 1.0 - endpoint_epsilon:
        return b.astype(np.uint8)

    mode = str(mode)
    h, w = a.shape[:2]

    def reveal_gate(pattern, softness=0.08):
        """Reveal B as mix moves from 0 to 1 using a normalized pattern."""
        threshold = 1.0 - mix
        half_softness = max(0.0, float(softness)) * 0.5
        lo = threshold - half_softness
        hi = threshold + half_softness
        if hi <= lo:
            return (pattern >= threshold).astype(np.float32)
        gate = np.clip((pattern - lo) / (hi - lo), 0.0, 1.0)
        return gate * gate * (3.0 - 2.0 * gate)

    if mode == "Luma Wipe":
        # Reveal Deck B from the brightest areas of Deck A to the darkest.
        luma = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
        luma = luma / 255.0
        gate = reveal_gate(luma, softness=0.15)[..., None]
        out = a * (1.0 - gate) + b * gate

    elif mode == "Glitch Dissolve":
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        scale = 0.05
        noise = (np.sin(x * scale) * np.cos(y * scale) * 0.5 + 0.5)
        noise = noise + (np.sin(x * scale * 5.3 + y * scale * 2.1) * 0.25)
        noise = np.clip(noise / 1.25, 0.0, 1.0)

        gate = reveal_gate(noise, softness=0.10)[..., None]
        out = a * (1.0 - gate) + b * gate

        edge = gate * (1.0 - gate) * 4.0
        out[..., 0] = np.clip(out[..., 0] + edge[..., 0] * 50.0, 0, 255)
        out[..., 2] = np.clip(out[..., 2] + edge[..., 0] * 50.0, 0, 255)

    elif mode == "Horizontal Wipe":
        pattern = np.broadcast_to(
            np.linspace(1.0, 0.0, w, dtype=np.float32)[None, :],
            (h, w),
        )
        gate = reveal_gate(pattern, softness=0.04)[..., None]
        out = a * (1.0 - gate) + b * gate

    elif mode == "Vertical Wipe":
        pattern = np.broadcast_to(
            np.linspace(1.0, 0.0, h, dtype=np.float32)[:, None],
            (h, w),
        )
        gate = reveal_gate(pattern, softness=0.04)[..., None]
        out = a * (1.0 - gate) + b * gate

    elif mode == "Radial Wipe":
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        x = (x - (w - 1) * 0.5) / max(1.0, w * 0.5)
        y = (y - (h - 1) * 0.5) / max(1.0, h * 0.5)
        radius = np.clip(np.sqrt(x * x + y * y) / np.sqrt(2.0), 0.0, 1.0)
        pattern = 1.0 - radius
        gate = reveal_gate(pattern, softness=0.08)[..., None]
        out = a * (1.0 - gate) + b * gate

    elif mode == "Checker Wipe":
        y, x = np.mgrid[0:h, 0:w]
        cell_size = max(8, min(h, w) // 12)
        checker = ((x // cell_size) + (y // cell_size)) % 2
        local_x = (x % cell_size) / max(1.0, float(cell_size - 1))
        pattern = np.mod(local_x * 0.75 + checker * 0.5, 1.0).astype(np.float32)
        gate = reveal_gate(pattern, softness=0.08)[..., None]
        out = a * (1.0 - gate) + b * gate

    elif mode == "Hard Cut":
        out = a if mix < 0.5 else b

    elif mode == "Additive":
        out = np.clip(a * (1.0 - mix) + b * mix + (a * b / 255.0) * mix * (1.0 - mix) * 4.0, 0, 255)

    elif mode == "Multiply":
        blended = (a * b) / 255.0
        out = a * (1.0 - mix) + blended * mix

    elif mode == "Screen":
        blended = 255.0 - ((255.0 - a) * (255.0 - b) / 255.0)
        out = a * (1.0 - mix) + blended * mix

    elif mode == "Difference":
        blended = np.abs(a - b)
        out = a * (1.0 - mix) + blended * mix

    else:
        # "Normal Mix"
        out = a * (1.0 - mix) + b * mix

    return np.clip(out, 0, 255).astype(np.uint8)
