from typing import Tuple, Optional
from PIL import Image
import cv2
import numpy as np
import logging

import glitch_effects as ge
import cuda_effects as ce
from megaglitch.multimodal_control import (
    ControlledFeedbackConfig,
    ControlledRecursiveFeedback,
    MasterMultimodalAnalyzer,
    MultimodalAnalysisConfig,
    apply_master_optical_flow_remap,
)
from megaglitch.structural_restoration import apply_structural_restoration
from megaglitch.determinism import current_numpy_rng, stable_seed
import os

# Lazy imports for GL/Lua to avoid forcing dependency on startup
_mod_gl = None
_lua_node_cls = None
_TEMPORAL_MACRO_DEFAULTS = {
    "history_offset": 0.0,
    "memory_decay": 0.0,
    "event_hazard": 0.0,
    "clock_quantization": 0.0,
    "sync_desync": 0.0,
    "phase_reset_chance": 0.0,
    "local_time_warp": 0.0,
    "rupture_threshold": 0.0,
    "hysteresis_speed": 0.0,
    "modulator_coupling": 0.0,
}

_CUDA_NODE_BYPASS_LOGGED = set()
_TEMPORAL_MACRO_NAMES = set(_TEMPORAL_MACRO_DEFAULTS.keys())
_TEMPORAL_MACRO_RESERVED = _TEMPORAL_MACRO_NAMES | {
    "_node_id",
    "time_accum",
    "noise_warp_time",
    "frame_num",
}
_TEMPORAL_MACRO_EXCLUDED_LABELS = {
    "Faultline Possession",
    "Lua Shader",
    "P-Frame Glitches",
    "Databending",
    "Feedback / Trail",
    "Frame Buffer",
    "Row Temporal Stutter",
    "FrameDrop / RGB Split",
    "Frame Interpolation Glitch",
    "Temporal Slicing",
    "Time Displacement",
    "JPEG Compression",
    "WebP Corruption",
    "H264 Artifacts",
    "Buffer Underrun",
    "Multimodal Analysis",
    "Optical Flow Remap",
    "Controlled Recursive Feedback",
    "Quantization / Solarization",
    "Edge Reconstruction",
    "Structural Restoration",
}


def _normalize_param_path(base_dir: str, value, fallback: str) -> str:
    candidate = fallback if value in (None, '') else value
    candidate = '' if candidate is None else str(candidate).strip()
    if not candidate:
        return ''
    candidate = candidate.strip().strip('"').strip("'")
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
        candidate = candidate[1:-1].strip()
    candidate = candidate.strip().strip('"').strip("'")
    candidate = candidate.replace('\\', '/')
    if not os.path.isabs(candidate):
        candidate = os.path.join(base_dir, candidate)
    return os.path.normpath(candidate)



def _ensure_lua_gl():
    global _mod_gl, _lua_node_cls
    if _mod_gl is None or _lua_node_cls is None:
        try:
            import moderngl as _mgl
            from lua_glsl_demo import LuaShaderNode as _LSN
            _mod_gl = _mgl
            _lua_node_cls = _LSN
        except Exception as e:
            logging.error(f"Lua Shader deps unavailable: {e}")
            _mod_gl = False
            _lua_node_cls = None


def _pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    # Handle both PIL Images and numpy arrays
    if isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.array(img.convert("RGB"))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB").convert("RGBA")

def _ensure_uint8_np(arr: np.ndarray) -> np.ndarray:
    """Convert array to uint8 numpy array (for pipeline consistency)."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Ensure 3 channels (RGB)
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[..., :3]  # Drop alpha
    return arr


_QUALITY_LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _require_rgb8_master_frame(frame: np.ndarray, effect_name: str) -> np.ndarray:
    """Validate the fixed image-domain contract without normalizing or resizing."""

    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{effect_name} requires a numpy RGB frame")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{effect_name} requires an HxWx3 RGB frame")
    if frame.dtype != np.uint8:
        raise TypeError(f"{effect_name} requires uint8 master frames")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError(f"{effect_name} requires a non-empty frame")
    return frame


def apply_quantization_solarization(
    frame: np.ndarray,
    *,
    quantization_levels: int = 192,
    solarization_threshold: int = 208,
    solarization_blend: float = 0.10,
) -> np.ndarray:
    """Quantize RGB values, then optionally solarize luminance-selected pixels.

    Solarization reverses the already-quantized RGB values only where quantized
    luminance reaches ``solarization_threshold``.  The blend is bounded and the
    function neither resizes nor retains any frame state.
    """

    source = _require_rgb8_master_frame(frame, "quantization / solarization")
    levels = int(np.clip(int(quantization_levels), 2, 256))
    threshold = int(np.clip(int(solarization_threshold), 0, 255))
    solar_blend = float(solarization_blend)
    if not np.isfinite(solar_blend):
        raise ValueError("solarization_blend must be finite")
    solar_blend = float(np.clip(solar_blend, 0.0, 1.0))

    source_f = source.astype(np.float32)
    palette_index = np.rint(source_f * ((levels - 1) / 255.0))
    quantized = np.rint(palette_index * (255.0 / (levels - 1)))

    if solar_blend > 0.0:
        luminance = np.sum(
            quantized * _QUALITY_LUMA_WEIGHTS,
            axis=2,
            dtype=np.float32,
        )
        selected = luminance >= threshold
        if np.any(selected):
            solarized = 255.0 - quantized[selected]
            quantized[selected] = (
                quantized[selected] * (1.0 - solar_blend)
                + solarized * solar_blend
            )

    output = np.clip(np.rint(quantized), 0, 255).astype(np.uint8)
    if output.shape != source.shape:
        raise RuntimeError("quantization / solarization changed master resolution")
    return np.ascontiguousarray(output)


def apply_edge_reconstruction(
    frame: np.ndarray,
    *,
    edge_luma_reinforcement: float = 0.035,
    edge_sensitivity: float = 0.055,
) -> np.ndarray:
    """Apply a restrained Sobel-derived luminance lift at detected edges only.

    This deliberately has no blur, unsharp-mask, local-contrast, resize, or
    temporal stage.  Equal RGB lift preserves channel differences, and the
    available channel headroom bounds the operation without clipping halos.
    """

    source = _require_rgb8_master_frame(frame, "edge reconstruction")
    amount = float(edge_luma_reinforcement)
    sensitivity = float(edge_sensitivity)
    if not np.isfinite(amount):
        raise ValueError("edge_luma_reinforcement must be finite")
    if not np.isfinite(sensitivity):
        raise ValueError("edge_sensitivity must be finite")
    amount = float(np.clip(amount, 0.0, 0.25))
    sensitivity = float(np.clip(sensitivity, 0.001, 0.25))
    if amount <= 0.0:
        return np.ascontiguousarray(source.copy())

    rgb = source.astype(np.float32) * (1.0 / 255.0)
    luminance = np.sum(rgb * _QUALITY_LUMA_WEIGHTS, axis=2, dtype=np.float32)
    gradient_x = cv2.Sobel(
        luminance,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
        borderType=cv2.BORDER_REFLECT101,
    )
    gradient_y = cv2.Sobel(
        luminance,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
        borderType=cv2.BORDER_REFLECT101,
    )
    gradient_x64 = gradient_x.astype(np.float64)
    gradient_y64 = gradient_y.astype(np.float64)
    gradient = (
        np.sqrt(gradient_x64 * gradient_x64 + gradient_y64 * gradient_y64) * 0.25
    ).astype(np.float32)
    edge_weight = np.clip(
        (gradient - sensitivity) / max(4.0 * sensitivity, 1e-6),
        0.0,
        1.0,
    )
    channel_headroom = 1.0 - np.max(rgb, axis=2)
    luma_lift = amount * edge_weight * channel_headroom
    reconstructed = rgb + luma_lift[..., None]

    output = np.clip(np.rint(reconstructed * 255.0), 0, 255).astype(np.uint8)
    if output.shape != source.shape:
        raise RuntimeError("edge reconstruction changed master resolution")
    return np.ascontiguousarray(output)


def _wrap_simple(arr: np.ndarray, fn, *args, **kwargs) -> np.ndarray:
    try:
        out = fn(arr, *args, **kwargs)
        return out
    except Exception as e:
        logging.error(f"Effect apply failed in registry: {e}")
        return arr


def _effect_time(params: dict) -> float:
    return float(params.get("time_accum", params.get("noise_warp_time", 0.0)) or 0.0)


def _cycle01(t: float, speed: float = 1.0, phase: float = 0.0) -> float:
    return 0.5 + 0.5 * np.sin(t * speed + phase)


def _modulate_amount(base: float, evolution: float, t: float, speed: float,
                     phase: float = 0.0, swing: float = 0.5,
                     min_value: float = 0.0, max_value: float = 1.0) -> float:
    value = base * (1.0 + evolution * swing * np.sin(t * speed + phase))
    return float(np.clip(value, min_value, max_value))


def _sample_temporal_history(frame_buffer: list, previous_frame, offset: float, target_shape) -> Optional[np.ndarray]:
    candidates = [frame for frame in (frame_buffer or []) if isinstance(frame, np.ndarray)]
    if previous_frame is not None:
        if isinstance(previous_frame, Image.Image):
            prev_np = np.array(previous_frame.convert("RGB"))
        elif isinstance(previous_frame, np.ndarray):
            prev_np = previous_frame
        else:
            prev_np = None
        if prev_np is not None:
            candidates = candidates + [prev_np]
    frames = []
    for candidate in candidates:
        normalized = _ensure_uint8_np(candidate)
        if normalized.shape[:2] == tuple(target_shape):
            frames.append(normalized)
        else:
            logging.warning(
                "Discarding temporal history at %s; master shape is %s",
                normalized.shape[:2],
                tuple(target_shape),
            )
    if not frames:
        return None

    max_index = len(frames) - 1
    history_index = float(np.clip(offset, 0.0, float(max_index)))
    lower = int(np.floor(history_index))
    upper = min(max_index, lower + 1)
    blend = history_index - lower

    first = frames[-1 - lower]
    second = frames[-1 - upper]

    if blend <= 1e-6:
        return first
    mixed = first.astype(np.float32) * (1.0 - blend) + second.astype(np.float32) * blend
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _apply_temporal_macro_layer(arr: np.ndarray,
                                label: str,
                                params: dict,
                                previous_frame,
                                frame_buffer: list,
                                worker_state: Optional[object]) -> Tuple[np.ndarray, dict]:
    if label in _TEMPORAL_MACRO_EXCLUDED_LABELS:
        return arr, params

    macro_values = {
        name: float(params.get(name, default) or default)
        for name, default in _TEMPORAL_MACRO_DEFAULTS.items()
    }
    macro_delta = {
        name: macro_values[name] - _TEMPORAL_MACRO_DEFAULTS[name]
        for name in _TEMPORAL_MACRO_NAMES
    }

    macro_activity = (
        abs(macro_delta["history_offset"]) * 0.05 +
        abs(macro_delta["memory_decay"]) * 6.0 +
        abs(macro_delta["event_hazard"]) * 1.4 +
        abs(macro_delta["clock_quantization"]) * 6.0 +
        abs(macro_delta["sync_desync"]) * 0.8 +
        abs(macro_delta["phase_reset_chance"]) * 5.0 +
        abs(macro_delta["local_time_warp"]) * 0.45 +
        abs(macro_delta["rupture_threshold"]) * 0.4 +
        abs(macro_delta["hysteresis_speed"]) * 2.5 +
        abs(macro_delta["modulator_coupling"]) * 1.0
    )
    if macro_activity <= 1e-5:
        return arr, params

    h, w = arr.shape[:2]
    node_id = params.get("_node_id", label)
    effect_states = getattr(worker_state, "effect_states", None) if worker_state is not None else None
    state_key = f"temporal_macro::{label}::{node_id}"
    state = effect_states.get(state_key) if isinstance(effect_states, dict) else None

    if state is None or state.get("shape") != (h, w):
        seed = stable_seed(label, node_id, h, w, "temporal_macro")
        rng = np.random.default_rng(seed)
        # Temporal modulation state follows the master grid. The earlier
        # 10-26 px control lattice was silently upscaled every frame and could
        # contaminate master-history claims with reduced-resolution state.
        field_h = h
        field_w = w
        fy, fx = np.mgrid[0:field_h, 0:field_w].astype(np.float32)
        state = {
            "shape": (h, w),
            "rng": rng,
            "field": rng.normal(0.0, 0.15, size=(field_h, field_w)).astype(np.float32),
            "fx": fx / max(1.0, float(field_w - 1)),
            "fy": fy / max(1.0, float(field_h - 1)),
            "phase": float(rng.uniform(0.0, np.pi * 2.0)),
            "clock_offset": float(rng.uniform(-0.2, 0.2)),
            "smooth_time": float(_effect_time(params)),
            "last_base_time": float(_effect_time(params)),
            "local_time": float(_effect_time(params)),
            "stress": 0.0,
            "param_bias": {},
        }
        if isinstance(effect_states, dict):
            effect_states[state_key] = state

    rng = state["rng"]
    coupling = float(np.clip(macro_values["modulator_coupling"], 0.0, 1.0))
    hazard = float(np.clip(macro_values["event_hazard"], 0.0, 1.0))
    warp = float(np.clip(macro_values["local_time_warp"], -2.0, 2.0))
    quant = float(np.clip(macro_values["clock_quantization"], 0.0, 0.25))
    sync = float(np.clip(macro_values["sync_desync"], -1.0, 1.0))
    phase_reset = float(np.clip(macro_values["phase_reset_chance"], 0.0, 0.5))
    threshold = float(max(0.05, macro_values["rupture_threshold"]))
    hysteresis = float(np.clip(macro_values["hysteresis_speed"], 0.0, 1.0))
    decay = float(np.clip(macro_values["memory_decay"], 0.5, 0.999))
    history_offset = float(np.clip(macro_values["history_offset"], 0.0, 32.0))
    reversal = float(np.clip(macro_values.get("time_reversal", 0.0), 0.0, 1.0))

    base_time = _effect_time(params)
    
    # Calculate local time delta for reversal
    last_base = state.get("last_base_time", base_time)
    delta_t = base_time - last_base
    state["last_base_time"] = base_time
    
    flow_rate = 1.0 - 2.0 * reversal
    state["local_time"] = state.get("local_time", base_time) + delta_t * flow_rate
    local_base_time = state["local_time"]
    state["phase"] += 0.031 + 0.17 * hazard + 0.05 * abs(warp) + 0.06 * coupling

    if sync >= 0.0:
        state["clock_offset"] *= max(0.0, 1.0 - (0.02 + 0.16 * sync) * (0.35 + hysteresis))
    else:
        state["clock_offset"] += float(rng.normal(0.0, 0.012 + abs(sync) * 0.06))

    stress_drive = (
        abs(warp) * 0.035 +
        abs(sync) * 0.018 +
        max(0.0, hazard - _TEMPORAL_MACRO_DEFAULTS["event_hazard"]) * 0.10 +
        max(0.0, phase_reset - _TEMPORAL_MACRO_DEFAULTS["phase_reset_chance"]) * 0.18 +
        coupling * 0.04
    )
    state["stress"] = max(0.0, state["stress"] * (0.82 + 0.14 * decay) + stress_drive)

    fault_event = False
    threshold_gate = max(0.12, threshold)
    if state["stress"] > threshold_gate:
        fault_event = True
    elif rng.random() < (hazard * 0.018 + phase_reset * 0.08) * (1.0 + coupling):
        fault_event = True

    field = state["field"]
    drift_x = 0.12 + 0.35 * coupling + 0.10 * max(0.0, -sync)
    drift_y = 0.09 + 0.28 * max(0.0, hazard - _TEMPORAL_MACRO_DEFAULTS["event_hazard"])
    wave = (
        np.sin(state["fx"] * (3.2 + abs(warp) * 1.7) + state["phase"]) +
        np.cos(state["fy"] * (4.1 + quant * 18.0) - state["phase"] * (1.1 + drift_y))
    ) * 0.12
    field = field * (0.78 + 0.18 * decay) + wave.astype(np.float32)
    field = np.roll(field, int(round(drift_x * np.sin(state["phase"]) * field.shape[1] * 0.2)), axis=1)
    field = np.roll(field, int(round((0.08 + abs(warp) * 0.05) * np.cos(state["phase"] * 0.7) * field.shape[0] * 0.2)), axis=0)
    field += rng.normal(0.0, 0.02 + hazard * 0.03 + coupling * 0.02, size=field.shape).astype(np.float32)

    if fault_event:
        burst = np.zeros_like(field, dtype=np.float32)
        cy = int(rng.integers(0, field.shape[0]))
        cx = int(rng.integers(0, field.shape[1]))
        radius = max(2, int(rng.integers(2, max(3, min(field.shape) // 2 + 1))))
        cv2.circle(burst, (cx, cy), radius, 1.0, -1)
        burst = cv2.GaussianBlur(burst, (0, 0), 1.1 + hazard * 3.0)
        field += burst * (0.35 + hazard * 0.8 + coupling * 0.35)
        state["phase"] += float(rng.uniform(-np.pi, np.pi)) * (0.45 + hazard + coupling * 0.5)
        state["clock_offset"] += float(rng.normal(0.0, 0.12 + hazard * 0.35))
        state["stress"] *= max(0.12, 0.45 - coupling * 0.2)

    field = cv2.GaussianBlur(field.astype(np.float32), (0, 0), 0.75 + abs(warp) * 0.35 + coupling * 0.25)
    state["field"] = field

    mask = 0.5 + 0.5 * np.tanh(
        field * (1.8 + hazard * 1.4 + coupling * 0.8)
    )
    mask = mask.astype(np.float32, copy=False)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.0 + max(0.0, history_offset - 4.0) * 0.05)
    mask = np.clip(mask, 0.0, 1.0)

    target_time = (
        local_base_time * (1.0 + warp * (0.38 + 0.18 * coupling) * np.sin(state["phase"] * 0.7 + local_base_time * 0.9)) +
        state["clock_offset"]
    )
    if quant > 1e-6:
        step = max(1.0 / 60.0, 0.02 + quant * 1.8)
        target_time = np.round(target_time / step) * step

    smooth = 0.02 + 0.42 * hysteresis
    state["smooth_time"] += (target_time - state["smooth_time"]) * smooth
    macro_time = float(state["smooth_time"])

    mod_params = dict(params)
    mod_params["time_accum"] = macro_time
    mod_params["noise_warp_time"] = macro_time

    history_strength = np.clip(
        abs(history_offset - _TEMPORAL_MACRO_DEFAULTS["history_offset"]) * 0.055 +
        abs(decay - _TEMPORAL_MACRO_DEFAULTS["memory_decay"]) * 4.5 +
        max(0.0, hazard - _TEMPORAL_MACRO_DEFAULTS["event_hazard"]) * 0.55 +
        coupling * 0.28,
        0.0,
        0.82,
    )
    if history_strength > 1e-4:
        history_frame = _sample_temporal_history(frame_buffer, previous_frame, history_offset, (h, w))
        if history_frame is not None:
            blend_mask = np.clip(mask * history_strength, 0.0, 0.75)[..., None]
            arr = np.clip(
                arr.astype(np.float32) * (1.0 - blend_mask) +
                history_frame.astype(np.float32) * blend_mask,
                0,
                255,
            ).astype(np.uint8)

    rupture_mix = np.clip(
        mask * (
            max(0.0, hazard - _TEMPORAL_MACRO_DEFAULTS["event_hazard"]) * 0.22 +
            abs(warp) * 0.06 +
            quant * 0.55 +
            coupling * 0.10 +
            (0.12 if fault_event else 0.0)
        ),
        0.0,
        0.45,
    )
    if np.max(rupture_mix) > 1e-4:
        shift_x = int(round(np.sin(state["phase"] * 1.3) * (2.0 + 10.0 * max(0.0, hazard - _TEMPORAL_MACRO_DEFAULTS["event_hazard"]) + 4.0 * abs(warp))))
        shift_y = int(round(np.cos(state["phase"] * 0.9) * (1.0 + 7.0 * quant + 3.0 * coupling)))
        shifted = np.roll(np.roll(arr, shift_x, axis=1), shift_y, axis=0)
        arr = np.clip(
            arr.astype(np.float32) * (1.0 - rupture_mix[..., None]) +
            shifted.astype(np.float32) * rupture_mix[..., None],
            0,
            255,
        ).astype(np.uint8)

    perturb_strength = np.clip(
        max(0.0, hazard - _TEMPORAL_MACRO_DEFAULTS["event_hazard"]) * 0.46 +
        abs(warp) * 0.12 +
        quant * 0.70 +
        abs(sync) * 0.16 +
        abs(hysteresis - _TEMPORAL_MACRO_DEFAULTS["hysteresis_speed"]) * 0.42 +
        coupling * 0.26,
        0.0,
        0.9,
    )
    if perturb_strength > 1e-5:
        for key, value in list(mod_params.items()):
            if key in _TEMPORAL_MACRO_RESERVED or isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                continue
            bias = state["param_bias"].get(key)
            if bias is None:
                bias = float(rng.uniform(-1.0, 1.0))
                state["param_bias"][key] = bias

            wave_val = np.sin(macro_time * (0.7 + abs(bias) * 0.9 + coupling * 1.2) + state["phase"] * (0.8 + bias * 0.25))
            base_val = float(value)
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                amplitude = max(2.0 if abs(base_val) < 1.0 else 1.0, min(abs(base_val) * 0.32, 18.0))
            else:
                amplitude = max(0.08, min(abs(base_val) * 0.28, 12.0 if abs(base_val) > 8.0 else 0.45))
            candidate = base_val + amplitude * perturb_strength * bias * wave_val
            if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                mod_params[key] = int(round(candidate))
            else:
                mod_params[key] = float(candidate)

    return arr, mod_params


def _animated_mesh_tear(arr: np.ndarray, intensity: float, grid: int, evolution: float, t: float) -> np.ndarray:
    if intensity <= 0:
        return arr
    if evolution <= 0:
        return _wrap_simple(arr, ge.mesh_tear_np, intensity=intensity, grid=grid)

    h, w, _ = arr.shape
    grid = max(2, int(grid))
    ys = np.linspace(0, h, grid + 1, dtype=np.int32)
    xs = np.linspace(0, w, grid + 1, dtype=np.int32)
    max_jitter = max(1.0, max(h, w) * 0.02 * intensity * (1.0 + 1.5 * evolution))
    src_pts, dst_pts = [], []

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            src_pts.append([x, y])
            phase = t * (0.7 + 2.3 * evolution) + xi * 0.91 + yi * 1.17
            dx = np.sin(phase) * max_jitter * (0.35 + 0.65 * evolution)
            dy = np.cos(phase * 1.23 + 0.7) * max_jitter * (0.25 + 0.75 * evolution)
            dst_pts.append([np.clip(x + dx, 0, max(0, w - 1)), np.clip(y + dy, 0, max(0, h - 1))])

    dst = np.float32(dst_pts).reshape((grid + 1, grid + 1, 2))
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    dst64 = dst.astype(np.float64)

    for yi in range(grid):
        for xi in range(grid):
            tl, tr = dst64[yi, xi], dst64[yi, xi + 1]
            bl, br = dst64[yi + 1, xi], dst64[yi + 1, xi + 1]
            xs0, xs1 = int(xs[xi]), int(xs[xi + 1])
            ys0, ys1 = int(ys[yi]), int(ys[yi + 1])
            ny = ys1 - ys0
            nx = xs1 - xs0
            if ny <= 0 or nx <= 0:
                continue
            t_frac = (np.arange(ny, dtype=np.float64) / max(1.0, float(ny)))[:, None, None]
            s_frac = (np.arange(nx, dtype=np.float64) / max(1.0, float(nx)))[None, :, None]
            left = tl * (1.0 - t_frac) + bl * t_frac
            right = tr * (1.0 - t_frac) + br * t_frac
            p = left * (1.0 - s_frac) + right * s_frac
            map_x[ys0:ys1, xs0:xs1] = p[..., 0].astype(np.float32)
            map_y[ys0:ys1, xs0:xs1] = p[..., 1].astype(np.float32)

    return ge.gpo.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _apply_vhs_tracking_drift(arr: np.ndarray, damage: float, jitter: float, noise: float,
                              evolution: float, t: float) -> np.ndarray:
    if evolution <= 0:
        return arr

    out = arr.astype(np.float32)
    h, w = out.shape[:2]
    band_center = int(_cycle01(t, 0.7 + evolution * 1.8, 0.3) * max(0, h - 1))
    band_height = max(2, int((0.03 + 0.14 * evolution + 0.08 * damage) * h))
    y0 = max(0, band_center - band_height // 2)
    y1 = min(h, y0 + band_height)
    shift_px = int(np.sin(t * (2.1 + evolution * 3.4)) * w * 0.04 * (jitter + 0.25 * evolution))

    if y1 > y0:
        band = out[y0:y1].copy()
        if shift_px:
            band = np.roll(band, shift_px, axis=1)
        chroma_shift = max(1, int(1 + 4 * evolution * _cycle01(t, 1.4, 1.2)))
        band[..., 0] = np.roll(band[..., 0], chroma_shift, axis=1)
        band[..., 2] = np.roll(band[..., 2], -max(1, chroma_shift // 2), axis=1)
        band_noise = np.random.normal(0.0, 12.0 + 72.0 * (noise + evolution * 0.6), band.shape)
        out[y0:y1] = np.clip(band + band_noise, 0, 255)

    ghost_y0 = max(0, y0 - max(1, band_height // 3))
    ghost_y1 = min(h, ghost_y0 + max(1, band_height // 4))
    if ghost_y1 > ghost_y0:
        ghost = np.roll(out[ghost_y0:ghost_y1], max(1, abs(shift_px) // 2 + 1), axis=1)
        out[ghost_y0:ghost_y1] = np.clip(out[ghost_y0:ghost_y1] * 0.55 + ghost * 0.45, 0, 255)

    return out.astype(np.uint8)


def apply_node_effect(label: str,
                      img,  # Can be PIL Image or numpy array
                      params: dict,
                      previous_frame,
                      feedback_image,
                      trail_image,
                      frame_buffer: list,
                      worker_state: Optional[object] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Apply a single node's effect using targeted functions when available.

    Falls back to orchestrator apply_glitch_effect for complex nodes.
    
    IMPORTANT: Returns numpy arrays, not PIL Images, for pipeline consistency.
    """
    label = str(label or "").strip()
    if params is not None and str(params.get("enabled", True)).lower() == "false":
        if worker_state is not None and label in {
            "Multimodal Analysis",
            "Controlled Recursive Feedback",
        }:
            effect_states = getattr(worker_state, "effect_states", None)
            if isinstance(effect_states, dict):
                node_id = params.get("_node_id", label)
                effect_states.pop(f"multimodal_analysis::{node_id}", None)
                effect_states.pop(f"controlled_feedback::{node_id}", None)
                if label == "Multimodal Analysis":
                    effect_states.pop("multimodal_active", None)
        return img, previous_frame, feedback_image, trail_image

    is_cuda = "CUDA" in label
    
    import gpu_ops as gpo
    if is_cuda and not gpo.get_use_cuda():
        # CUDA-labeled nodes can be very expensive on CPU/fallback paths.
        # If CUDA is disabled in-app, treat them as no-op to preserve playback FPS.
        if label not in _CUDA_NODE_BYPASS_LOGGED:
            logging.info(f"Bypassing CUDA node while GPU is disabled: {label}")
            _CUDA_NODE_BYPASS_LOGGED.add(label)
        return img, previous_frame, feedback_image, trail_image
    
    if not is_cuda:
        img = gpo.ensure_cpu(img)
        if isinstance(img, Image.Image):
            img = np.array(img.convert("RGB"))
        if not isinstance(img, np.ndarray):
            logging.error(f"apply_node_effect received unexpected type: {type(img)}")
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        arr = _pil_to_rgb_np(img)
        arr, params = _apply_temporal_macro_layer(arr, label, params, previous_frame, frame_buffer, worker_state)
    else:
        # CUDA path - preserve GpuMat if it exists
        arr = img
        # Pass params through (skip temporal macro for pure GPU nodes to avoid PCIe bottleneck, 
        # or implement a GPU-native temporal macro later)

    if label == "Multimodal Analysis":
        if worker_state is None:
            logging.warning("Multimodal Analysis requires worker-owned state; bypassing")
            return arr, previous_frame, feedback_image, trail_image
        effect_states = getattr(worker_state, "effect_states", None)
        if not isinstance(effect_states, dict):
            logging.warning("Multimodal Analysis worker has no effect-state map; bypassing")
            return arr, previous_frame, feedback_image, trail_image
        entropy_window = max(3, min(31, int(params.get("entropy_window", 9) or 9)))
        if entropy_window % 2 == 0:
            entropy_window = min(31, entropy_window + 1)
        motion_low = float(np.clip(params.get("motion_low", 0.035), 0.0, 0.99))
        motion_high = float(np.clip(params.get("motion_high", 0.42), motion_low + 0.001, 1.0))
        edge_low = float(np.clip(params.get("edge_low", 0.025), 0.0, 0.99))
        edge_high = float(np.clip(params.get("edge_high", 0.32), edge_low + 0.001, 1.0))
        config = MultimodalAnalysisConfig(
            entropy_window=entropy_window,
            flow_scale_pixels=float(params.get("flow_scale_pixels", 12.0) or 12.0),
            motion_low=motion_low,
            motion_high=motion_high,
            edge_low=edge_low,
            edge_high=edge_high,
            scene_change_threshold=float(
                np.clip(params.get("scene_change_threshold", 0.32), 0.0, 1.0)
            ),
            modulation_gamma=float(params.get("modulation_gamma", 1.12) or 1.12),
        )
        node_id = params.get("_node_id", label)
        state_key = f"multimodal_analysis::{node_id}"
        state = effect_states.get(state_key)
        if state is None or state.get("config") != config:
            state = {
                "config": config,
                "analyzer": MasterMultimodalAnalyzer(config),
                "previous_source": None,
            }
            effect_states[state_key] = state
        analysis = state["analyzer"].analyze(
            arr,
            state.get("previous_source"),
            audio_fft_bands=params.get("audio_fft_bands"),
            transient=float(params.get("audio_transient", 0.0) or 0.0),
            beat_state=params.get("beat_state", False),
        )
        state["previous_source"] = arr.copy()
        state["analysis"] = analysis
        effect_states["multimodal_active"] = analysis
        effect_states["multimodal_telemetry"] = dict(analysis.telemetry)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Controlled Recursive Feedback":
        if worker_state is None:
            logging.warning("Controlled Recursive Feedback requires worker-owned state; bypassing")
            return arr, previous_frame, feedback_image, trail_image
        effect_states = getattr(worker_state, "effect_states", None)
        analysis = (
            effect_states.get("multimodal_active")
            if isinstance(effect_states, dict)
            else None
        )
        if analysis is None:
            logging.warning(
                "Controlled Recursive Feedback has no upstream Multimodal Analysis; bypassing"
            )
            return arr, previous_frame, feedback_image, trail_image
        config = ControlledFeedbackConfig(
            opacity=float(params.get("feedback_opacity", 0.42) or 0.0),
            opacity_cap=float(params.get("opacity_cap", 0.68) or 0.0),
            decay=float(params.get("feedback_decay", 0.92) or 0.0),
            clean_source_floor=float(params.get("clean_source_floor", 0.32) or 0.0),
            motion_suppression=float(params.get("motion_suppression", 0.72) or 0.0),
            edge_protection=float(params.get("edge_protection", 0.28) or 0.0),
            scene_change_threshold=float(
                params.get("scene_change_threshold", 0.32) or 0.0
            ),
        )
        node_id = params.get("_node_id", label)
        state_key = f"controlled_feedback::{node_id}"
        state = effect_states.get(state_key)
        if state is None or state.get("config") != config:
            state = {
                "config": config,
                "feedback": ControlledRecursiveFeedback(config),
            }
            effect_states[state_key] = state
        result = state["feedback"].process(arr, analysis, enabled=True)
        state["telemetry"] = result.telemetry
        effect_states["controlled_feedback_telemetry"] = result.telemetry
        return result.frame, previous_frame, feedback_image, trail_image

    if label == "Optical Flow Remap":
        if worker_state is None:
            logging.warning("Optical Flow Remap requires worker-owned analysis; bypassing")
            return arr, previous_frame, feedback_image, trail_image
        effect_states = getattr(worker_state, "effect_states", None)
        analysis = (
            effect_states.get("multimodal_active")
            if isinstance(effect_states, dict)
            else None
        )
        if analysis is None:
            logging.warning(
                "Optical Flow Remap has no upstream Multimodal Analysis; bypassing"
            )
            return arr, previous_frame, feedback_image, trail_image
        result = apply_master_optical_flow_remap(
            arr,
            analysis,
            strength=float(params.get("flow_remap_strength", 0.65) or 0.0),
            displacement_cap_pixels=float(
                params.get("flow_displacement_cap", 24.0) or 0.0
            ),
            direction=str(params.get("flow_direction", "backward")),
            border_mode=str(params.get("flow_border_mode", "reflect")),
        )
        return result, previous_frame, feedback_image, trail_image

    # CUDA GPU-accelerated effects
    if "CUDA" in label:
        try:
            img_array = arr
            
            prev_array = None
            if previous_frame is not None:
                prev_array = gpo.ensure_cpu(previous_frame) # previous_frame might be used for flow and expects numpy in legacy nodes
            
            # Get time value
            time_val = params.get("time_accum", 0.0)
            
            # CUDA temporal resources belong to the current stream/node. They
            # must never leak across duplicate nodes, decks, live/offline, or
            # consecutive deterministic renders.
            cuda_state_store = None
            if worker_state is not None:
                effect_states = getattr(worker_state, "effect_states", None)
                if isinstance(effect_states, dict):
                    cuda_state_store = effect_states.setdefault(
                        "cuda_temporal_states", {}
                    )
            result_array, next_prev_array = ce.apply_cuda_effect(
                label,
                img_array,
                params,
                prev_array,
                time_val,
                state_store=cuda_state_store,
                state_namespace=params.get("_node_id", label),
            )
            
            # Return directly (could be GpuMat or numpy)
            if next_prev_array is None:
                next_prev_array = result_array
            
            return result_array, next_prev_array, feedback_image, trail_image
            
        except Exception as e:
            logging.error(f"CUDA effect {label} failed: {e}", exc_info=True)
            # Return original image on error
            return img, previous_frame, feedback_image, trail_image

    # Lua Shader (GPU) node
    if label in {"Lua Shader", "Abyssal Signal Feedback", "Kaleido Feedback"}:
        try:
            _ensure_lua_gl()
            if not _mod_gl or not _lua_node_cls:
                return img, previous_frame, feedback_image, trail_image

            # Resolve paths
            base_dir = os.path.dirname(__file__)
            shader_path = _normalize_param_path(base_dir, params.get('shader_path'), 'lua_glsl_demo/shaders/effect.frag')
            lua_path    = _normalize_param_path(base_dir, params.get('lua_path'),    'lua_glsl_demo/scripts/effect.lua')
            if not shader_path or not lua_path:
                return img, previous_frame, feedback_image, trail_image

            use_input = bool(params.get('use_input', True))
            freeze    = bool(params.get('freeze_time', False))

            # img is numpy array now (H, W, C)
            h, w = img.shape[:2]

            # Keep one GL context per worker, but one persistent shader node per
            # graph node. A single global node causes adjacent Lua shaders to
            # recreate each other every frame and destroys feedback history.
            ctx = getattr(worker_state, 'lua_gl_ctx', None)
            effect_states = getattr(worker_state, 'effect_states', None)
            if not isinstance(effect_states, dict):
                effect_states = {}
                worker_state.effect_states = effect_states
            node_id = params.get('_node_id')
            state_key = f"lua_shader::{node_id if node_id is not None else (shader_path, lua_path)}"
            shader_state = effect_states.get(state_key, {})
            node = shader_state.get('node')
            cached_paths = shader_state.get('paths')
            cached_size = shader_state.get('size')

            need_new = False
            if ctx is None:
                try:
                    ctx = _mod_gl.create_standalone_context()
                except Exception as e:
                    logging.error(f"Failed to create GL context: {e}")
                    return img, previous_frame, feedback_image, trail_image
                need_new = True

            if node is None:
                need_new = True
            if cached_paths != (shader_path, lua_path):
                need_new = True
            if not need_new and cached_size != (w, h):
                # Rebuild GL targets for new size
                try:
                    node.size = (w, h)  # triggers resize in render()
                except Exception:
                    need_new = True

            if need_new:
                try:
                    # Dispose old if present
                    try:
                        if node: node.dispose()
                    except Exception:
                        pass
                    for texture in (shader_state.get("aux_textures") or {}).values():
                        try:
                            texture.release()
                        except Exception:
                            pass
                    node = _lua_node_cls(ctx, w, h, shader_path, lua_path, auto_reload=True)
                    worker_state.lua_gl_ctx = ctx
                    shader_state = {
                        'node': node,
                        'paths': (shader_path, lua_path),
                        'size': (w, h),
                        'aux_textures': {},
                    }
                    effect_states[state_key] = shader_state
                except Exception as e:
                    logging.error(f"Failed to init Lua Shader node: {e}")
                    return img, previous_frame, feedback_image, trail_image
            else:
                shader_state['size'] = (w, h)

            # Time control
            try:
                node.freeze(freeze)
            except Exception:
                pass

            # Optional input upload
            input_tex = None
            if use_input:
                try:
                    # img is numpy (H, W, 3) - convert to RGBA
                    if img.shape[2] == 3:
                        # Add alpha channel
                        rgba_arr = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
                    else:
                        rgba_arr = img
                    arr = rgba_arr.astype('u1')  # 8-bit RGBA
                    # Allocate or reuse texture
                    tex = getattr(worker_state, 'lua_gl_input_tex', None)
                    if tex is None or getattr(worker_state, 'lua_gl_input_size', None) != (w, h):
                        try:
                            if tex is not None:
                                tex.release()
                        except Exception:
                            pass
                        tex = ctx.texture((w, h), components=4, dtype='f1')  # RGBA8 normalized
                        tex.filter = (_mod_gl.LINEAR, _mod_gl.LINEAR)
                        worker_state.lua_gl_input_tex = tex
                        worker_state.lua_gl_input_size = (w, h)
                    tex.write(arr.tobytes())
                    gpo.record_external_transfer("cpu_to_opengl", int(arr.nbytes))
                    input_tex = tex
                except Exception as e:
                    logging.error(f"Lua Shader input upload failed: {e}")

            # Render one frame
            try:
                named_input_textures = {}
                aux_frames = params.get("_aux_input_frames") or {}
                aux_uniforms = {}
                shader_stem = os.path.splitext(os.path.basename(str(shader_path)))[0].lower()
                if (
                    aux_frames.get("feedback_in") is not None
                    or label == "Kaleido Feedback"
                    or shader_stem == "kaleido_feedback"
                ):
                    aux_uniforms["feedback_in"] = ("uFeedbackInput", (0, 0, 0, 255))
                if label == "Abyssal Signal Feedback":
                    aux_uniforms.update({
                        "bubble_mask": ("u_bubble_mask", (0, 0, 0, 255)),
                        "bubble_normal": ("u_bubble_normal", (128, 128, 255, 255)),
                        "bubble_depth": ("u_bubble_depth", (0, 0, 0, 255)),
                        "bubble_refraction": ("u_bubble_refraction", (128, 128, 0, 255)),
                        "bubble_highlight": ("u_bubble_highlight", (0, 0, 0, 255)),
                    })
                if aux_uniforms:
                    texture_cache = shader_state.setdefault("aux_textures", {})
                    for input_name, (uniform_name, neutral_rgba) in aux_uniforms.items():
                        aux_frame = aux_frames.get(input_name)
                        if aux_frame is None:
                            if input_name == "feedback_in":
                                if img.shape[2] == 3:
                                    rgba = np.dstack([img, np.full((h, w), 255, dtype=np.uint8)])
                                else:
                                    rgba = img[..., :4]
                                rgba = np.clip(rgba, 0, 255).astype(np.uint8)
                            else:
                                rgba = np.empty((h, w, 4), dtype=np.uint8)
                                rgba[...] = neutral_rgba
                        else:
                            aux_arr = np.asarray(aux_frame)
                            if aux_arr.ndim == 2:
                                aux_arr = np.repeat(aux_arr[..., None], 3, axis=2)
                            aux_h, aux_w = aux_arr.shape[:2]
                            if aux_arr.shape[2] == 3:
                                rgba = np.dstack([
                                    aux_arr,
                                    np.full((aux_h, aux_w), 255, dtype=np.uint8),
                                ])
                            else:
                                rgba = aux_arr[..., :4]
                            rgba = np.clip(rgba, 0, 255).astype(np.uint8)

                        texture = texture_cache.get(input_name)
                        texture_size = (rgba.shape[1], rgba.shape[0])
                        if texture is None or texture.size != texture_size:
                            if texture is not None:
                                try:
                                    texture.release()
                                except Exception:
                                    pass
                            texture = ctx.texture(texture_size, components=4, dtype="f1")
                            texture.filter = (_mod_gl.LINEAR, _mod_gl.LINEAR)
                            texture_cache[input_name] = texture
                        texture.write(rgba.tobytes())
                        gpo.record_external_transfer(
                            "cpu_to_opengl", int(rgba.nbytes)
                        )
                        named_input_textures[uniform_name] = texture

                uniform_params = {}
                for key, value in params.items():
                    if key in {"shader_preset", "shader_path", "lua_path", "metadata_path", "use_input", "freeze_time", "flip_y", "_node_id", "_aux_input_frames"}:
                        continue
                    if isinstance(value, bool):
                        uniform_params[key] = 1.0 if value else 0.0
                    elif isinstance(value, (int, float)):
                        uniform_params[key] = float(value)
                node.render(
                    external_params=uniform_params,
                    input_tex=input_tex,
                    named_input_textures=named_input_textures,
                )
                out_tex = node.output_texture()
                raw = out_tex.read()  # bytes
                gpo.record_external_transfer("opengl_to_cpu", len(raw))
                out = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
                if bool(params.get('flip_y', True)):
                    out = np.flipud(out)
                out_img = Image.fromarray(out, mode='RGBA')
                # Return numpy array for pipeline consistency
                return np.array(out_img.convert("RGB")), previous_frame, feedback_image, trail_image
            except Exception as e:
                logging.error(f"Lua Shader render failed: {e}")
                return img, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Lua Shader node error: {e}")
            return img, previous_frame, feedback_image, trail_image

    # Map of node labels to lightweight handlers
    # NOTE: All effects must return numpy arrays (not PIL Images) for consistent pipeline handling
    # Helper to ensure we return numpy array (not PIL Image)
    def _ensure_numpy_return(result_arr, prev, feed, trail):
        """Ensure the returned frame is a numpy array, not PIL Image."""
        if isinstance(result_arr, Image.Image):
            result_arr = np.array(result_arr.convert("RGB"))
        if result_arr.dtype != np.uint8:
            result_arr = np.clip(result_arr, 0, 255).astype(np.uint8)
        return result_arr, prev, feed, trail

    if label == "Brightness / Contrast":
        b = int(params.get("brightness", 0) or 0)
        c = int(params.get("contrast", 0) or 0)
        arr = _wrap_simple(arr, ge.apply_brightness_contrast, brightness=b, contrast=c)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Color Adjust":
        hue = float(params.get("hue_shift", 0) or 0)
        saturation = params.get("saturation", 1.0)
        sat = 1.0 if saturation is None else float(saturation)
        gamma = float(params.get("gamma", 1.0) or 1.0)
        if (
            arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3
            and not gpo.get_use_cuda() and all(np.isfinite(v) for v in (hue, sat, gamma))
        ):
            arr = _wrap_simple(arr, ge.apply_color_adjust_rgb8, hue, sat, gamma)
            return arr, previous_frame, feedback_image, trail_image
        arr = _wrap_simple(arr, ge.apply_hsv_adjust, hue_shift=hue, saturation=sat)
        arr = _wrap_simple(arr, ge.apply_gamma, gamma=gamma)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Vignette":
        intensity = float(params.get("vignette_intensity", 0.0) or 0.0)
        radius = float(params.get("vignette_radius", 0.75) or 0.75)
        arr = _wrap_simple(arr, ge.apply_vignette, intensity=intensity, radius=radius)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Blur / Sharpen":
        blur_r = float(params.get("blur_radius", 0.0) or 0.0)
        sharp_a = float(params.get("sharpen_amount", 0.0) or 0.0)
        if blur_r > 0:
            arr = _wrap_simple(arr, ge.apply_blur, radius=blur_r)
        if sharp_a > 0:
            arr = _wrap_simple(arr, ge.apply_sharpen, amount=sharp_a, radius=2)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Quantization / Solarization":
        arr = _wrap_simple(
            arr,
            apply_quantization_solarization,
            quantization_levels=params.get("quantization_levels", 192),
            solarization_threshold=params.get("solarization_threshold", 208),
            solarization_blend=params.get("solarization_blend", 0.10),
        )
        return arr, previous_frame, feedback_image, trail_image

    if label == "Edge Reconstruction":
        arr = _wrap_simple(
            arr,
            apply_edge_reconstruction,
            edge_luma_reinforcement=params.get("edge_luma_reinforcement", 0.035),
            edge_sensitivity=params.get("edge_sensitivity", 0.055),
        )
        return arr, previous_frame, feedback_image, trail_image

    if label == "Structural Restoration":
        arr = _wrap_simple(
            arr,
            apply_structural_restoration,
            unsharp_amount=params.get("unsharp_amount", 0.20),
            unsharp_radius=params.get("unsharp_radius", 1.10),
            local_contrast=params.get("local_contrast", 0.12),
            local_contrast_radius=params.get("local_contrast_radius", 10.0),
            edge_luma_reinforcement=params.get("edge_luma_reinforcement", 0.05),
            edge_sensitivity=params.get("edge_sensitivity", 0.05),
            grain_normalization=params.get("grain_normalization", 0.0),
            grain_target=params.get("grain_target", 0.012),
            final_clamp_low=params.get("final_clamp_low", 0),
            final_clamp_high=params.get("final_clamp_high", 255),
        )
        return arr, previous_frame, feedback_image, trail_image

    if label == "Bloom":
        strength = float(params.get("bloom_strength", 0.0) or 0.0)
        thresh = int(params.get("bloom_threshold", 200) or 200)
        arr = _wrap_simple(arr, ge.apply_bloom, strength=strength, threshold=thresh)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Edges / Emboss":
        edge = float(params.get("edge_mix", 0.0) or 0.0)
        emb = float(params.get("emboss_strength", 0.0) or 0.0)
        arr = _wrap_simple(arr, ge.apply_edges_emboss, edge_mix=edge, emboss=emb)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Barrel/Pincushion":
        amt = float(params.get("barrel_amount", 0.0) or 0.0)
        arr = _wrap_simple(arr, ge.barrel_pincushion_np, amount=amt)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Swirl/Twirl":
        amt = float(params.get("swirl_amount", 0.0) or 0.0)
        rf = float(params.get("swirl_radius_factor", 0.75) or 0.75)
        arr = _wrap_simple(arr, ge.swirl_np, amount=amt, radius_factor=rf)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Ripple":
        amp = float(params.get("ripple_amp", 0.0) or 0.0)
        freq = float(params.get("ripple_freq", 3.0) or 3.0)
        t = float(params.get("noise_warp_time", 0.0) or 0.0)
        arr = _wrap_simple(arr, ge.ripple_np, amp=amp, freq=freq, t=t)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Rotation / Kaleidoscope":
        # Kaleidoscope first, then simple rotation
        segs = int(params.get("kaleidoscope_segments", 1) or 1)
        if segs > 1:
            speed = float(params.get("kaleidoscope_speed", 0.2) or 0.2)
            angle_offset = float(params.get("noise_warp_time", 0.0) or 0.0) * speed
            arr = _wrap_simple(arr, ge.kaleidoscope_np, segments=segs, angle_offset=angle_offset, radius_factor=1.0)
        rot = float(params.get("rotation_angle", 0.0) or 0.0)
        if rot:
            arr = _wrap_simple(arr, ge.rotate_in_frame_np, angle=rot, overscan=0.18)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Vectorscope XY":
        intensity = float(params.get("vectorscope_intensity", 0.5) or 0.5)
        freq_x = float(params.get("freq_x", 1.0) or 1.0)
        freq_y = float(params.get("freq_y", 2.0) or 2.0)
        phase = float(params.get("phase_offset", 0.0) or 0.0)
        circular = float(params.get("circular_distortion", 0.3) or 0.3)
        t = float(params.get("noise_warp_time", 0.0) or 0.0)
        arr = _wrap_simple(arr, ge.vectorscope_xy_np, 
                          intensity=intensity, freq_x=freq_x, freq_y=freq_y,
                          phase=phase, circular=circular, t=t)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Glitch Amount":
        # Sine-based UV warp similar to orchestrator
        h, w = arr.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        ts = float(params.get("time_scale", 1.0) or 1.0)
        t_off = float(params.get("noise_warp_time", 0.0) or 0.0) * (2.0 ** ts)
        uv = float(params.get("uv_size", 1.0) or 1.0)
        gx = float(params.get("glitch_amount_x", 0.0) or 0.0)
        gy = float(params.get("glitch_amount_y", 0.0) or 0.0)
        fx = float(params.get("glitch_freq_x", 1.0) or 1.0)
        fy = float(params.get("glitch_freq_y", 1.0) or 1.0)
        glitch_x = gx * np.sin(2.0 * np.pi * (map_y / max(1e-3, fy * uv)) + t_off) if fy * uv != 0 else 0
        glitch_y = gy * np.sin(2.0 * np.pi * (map_x / max(1e-3, fx * uv)) + t_off / 2.0) if fx * uv != 0 else 0
        map_x = map_x + glitch_x
        map_y = map_y + glitch_y
        out = ge.gpo.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image

    if label == "CRT Emulation":
        scan = float(params.get("crt_scanline", 0.0) or 0.0)
        mask = str(params.get("crt_mask", "rgb") or "rgb")
        curv = float(params.get("crt_curvature", 0.0) or 0.0)
        bloom = float(params.get("crt_bloom", 0.0) or 0.0)
        arr = _wrap_simple(arr, ge.crt_emulation_np, scanline=scan, mask_type=mask, curvature=curv, bloom=bloom)
        return arr, previous_frame, feedback_image, trail_image

    if label == "VHS Damage":
        dmg = float(params.get("vhs_damage", 0.0) or 0.0)
        band_min = int(params.get("vhs_band_min", 4) or 4)
        band_max = int(params.get("vhs_band_max", 24) or 24)
        jitter = float(params.get("vhs_jitter", 0.2) or 0.2)
        noise = float(params.get("vhs_noise", 0.2) or 0.2)
        evolution = float(params.get("vhs_tracking_evolution", 0.0) or 0.0)
        t = _effect_time(params)
        speed = float(params.get("vhs_tracking_evolution_speed", 0.8) or 0.8)
        if evolution > 0:
            dmg = _modulate_amount(dmg, evolution, t, speed * 1.1, phase=0.2, swing=0.55)
            jitter = _modulate_amount(jitter, evolution, t, speed * 1.7, phase=1.1, swing=0.8)
            noise = _modulate_amount(noise, evolution, t, speed * 1.3, phase=2.0, swing=0.65)
            band_span = max(2, band_max - band_min)
            band_min = max(2, int(round(band_min * (1.0 + evolution * 0.35 * np.sin(t * speed * 0.9 + 0.4)))))
            band_max = max(band_min + 1, int(round(band_min + band_span * (0.8 + 0.7 * _cycle01(t, speed * 0.7, 0.9)))))
        arr = _wrap_simple(arr, ge.vhs_damage_np, damage=dmg, band_min=band_min, band_max=band_max, jitter=jitter, noise=noise)
        arr = _apply_vhs_tracking_drift(arr, dmg, jitter, noise, evolution, t * max(0.0, speed))
        return arr, previous_frame, feedback_image, trail_image

    if label == "VHS Jitter":
        intensity = float(params.get("vhs_jitter", 0.0) or 0.0)
        max_shift = int(params.get("vhs_max_shift", 8) or 8)
        try:
            arr = ge.vhs_jitter_np(arr, intensity=intensity, max_shift=max_shift)
        except Exception as e:
            logging.error(f"VHS jitter failed: {e}")
        return arr, previous_frame, feedback_image, trail_image

    if label == "Rolling Shutter Wobble":
        inten = float(params.get("wobble_intensity", 0.0) or 0.0)
        freq = float(params.get("wobble_freq", 3.0) or 3.0)
        phase = float(params.get("noise_warp_time", 0.0) or 0.0) * float(params.get("wobble_speed", 0.5) or 0.5)
        arr = _wrap_simple(arr, ge.rolling_shutter_wobble_np, intensity=inten, freq=freq, phase=phase)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Scanline Shuffle":
        intensity = float(params.get("scan_shuffle", 0.0) or 0.0)
        max_span = int(params.get("scan_span", 32) or 32)
        prob = float(params.get("scan_prob", 0.12) or 0.12)
        evolution = float(params.get("scan_shuffle_evolution", 0.0) or 0.0)
        speed = float(params.get("scan_shuffle_evolution_speed", 1.3) or 1.3)
        if evolution <= 0:
            arr = _wrap_simple(arr, ge.scanline_shuffle_np, intensity=intensity, max_span=max_span, prob=prob)
        else:
            t = _effect_time(params) * speed
            h = arr.shape[0]
            out = arr.copy()
            rng = current_numpy_rng()
            anim_prob = float(np.clip(prob + evolution * 0.3 * np.sin(t * 1.1 + 0.3), 0.0, 1.0))
            anim_span = max(1, int(round(max_span * (1.0 + evolution * (0.4 + 0.8 * _cycle01(t, 0.8, 1.2))))))
            band_phase = np.linspace(0.0, np.pi * (4.0 + 12.0 * evolution), h, dtype=np.float32)
            band_mask = 0.5 + 0.5 * np.sin(band_phase + t * (1.8 + 3.0 * evolution))
            for y in range(h):
                local_prob = float(np.clip(anim_prob * (0.25 + 1.25 * band_mask[y]), 0.0, 1.0))
                if rng.random() < local_prob:
                    local_span = max(1, int(round(anim_span * (0.35 + 0.65 * band_mask[y]))))
                    phase_offset = int(round(np.sin(t * 1.6 + y * 0.07) * local_span * evolution * 0.4))
                    src_y = int(np.clip(y + int(rng.integers(-local_span, local_span + 1)) + phase_offset, 0, h - 1))
                    local_mix = float(np.clip(intensity * (0.55 + 0.45 * band_mask[y]), 0.0, 1.0))
                    out[y] = cv2.addWeighted(out[y], 1.0 - local_mix, arr[src_y], local_mix, 0)
            arr = _ensure_uint8_np(out)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Chroma Leak":
        intensity = float(params.get("chroma_intensity", 0.0) or 0.0)
        subsample = str(params.get("chroma_mode", "420") or "420")
        drift = int(params.get("chroma_drift", 2) or 2)
        arr = _wrap_simple(arr, ge.chroma_leak_np, intensity=intensity, subsample=subsample, drift_px=drift)
        return arr, previous_frame, feedback_image, trail_image

    if label == "FFT Phase Scramble":
        intensity = float(params.get("fft_phase", 0.0) or 0.0)
        bands = int(params.get("fft_bands", 3) or 3)
        try:
            arr = ge.fft_phase_scramble_np(arr, intensity, bands)
        except Exception as e:
            logging.error(f"FFT phase scramble failed: {e}")
        return arr, previous_frame, feedback_image, trail_image

    if label == "Kaleidoscope":
        segs = int(params.get("kaleido_segments", 6) or 6)
        angle = float(params.get("kaleido_angle", 0.0) or 0.0)
        radius = float(params.get("kaleido_radius", 1.0) or 1.0)
        arr = _wrap_simple(arr, ge.kaleidoscope_np, segments=max(2, segs), angle_offset=angle, radius_factor=radius)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Noise Warp":
        intensity = float(params.get("noise_warp_intensity", 0.0) or 0.0)
        scale = float(params.get("noise_warp_scale", 64.0) or 64.0)
        t = float(params.get("noise_warp_time", 0.0) or 0.0)
        arr = _wrap_simple(arr, ge.noise_field_warp_np, intensity=intensity, scale=scale, t=t)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Watercolor Posterize":
        levels = int(params.get("watercolor_levels", 6) or 6)
        sigma = float(params.get("watercolor_sigma", 12.0) or 12.0)
        passes = int(params.get("watercolor_passes", 2) or 2)
        arr = _wrap_simple(arr, ge.bilateral_posterize_np, levels=levels, bilateral_sigma=sigma, passes=passes)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Halftone":
        cell = int(params.get("halftone_cell", 8) or 8)
        contrast = float(params.get("halftone_contrast", 1.0) or 1.0)
        try:
            arr = ge.halftone_np(arr, cell=cell, contrast=contrast)
        except Exception as e:
            logging.error(f"Halftone failed: {e}")
        return arr, previous_frame, feedback_image, trail_image

    if label == "Palette Crunch":
        levels = int(params.get("palette_levels", 6) or 6)
        dither = bool(params.get("palette_dither", True))
        arr = _wrap_simple(arr, ge.palette_crunch_np, levels=levels, dither=dither)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Channel Flow Offset":
        intensity = float(params.get("channel_flow", 0.0) or 0.0)
        max_px = int(params.get("channel_flow_max", 8) or 8)
        evolution = float(params.get("channel_flow_evolution", 0.0) or 0.0)
        speed = float(params.get("channel_flow_evolution_speed", 1.1) or 1.1)
        if evolution <= 0:
            arr = _wrap_simple(arr, ge.channel_flow_offset_np, intensity=intensity, max_px=max_px)
        else:
            t = _effect_time(params) * speed
            h, w, _ = arr.shape
            out = np.zeros_like(arr)
            orbit_radius = max_px * intensity * (0.35 + 0.65 * evolution)
            for ch in range(3):
                angle = t * (1.1 + 0.35 * ch) + ch * (2.0 * np.pi / 3.0)
                dx = int(round(np.cos(angle) * orbit_radius * (0.65 + 0.35 * _cycle01(t, 0.9, ch))))
                dy = int(round(np.sin(angle * 1.23 + 0.4) * orbit_radius * evolution))
                mat = np.float32([[1, 0, dx], [0, 1, dy]])
                out[..., ch] = ge.gpo.warpAffine(arr[..., ch], mat, (w, h), borderMode=cv2.BORDER_REFLECT)
            arr = out
        return arr, previous_frame, feedback_image, trail_image

    if label == "Mesh Tear":
        intensity = float(params.get("mesh_tear", 0.0) or 0.0)
        grid = int(params.get("mesh_grid", 6) or 6)
        evolution = float(params.get("mesh_tear_evolution", 0.0) or 0.0)
        speed = float(params.get("mesh_tear_evolution_speed", 0.9) or 0.9)
        arr = _animated_mesh_tear(arr, intensity=intensity, grid=grid, evolution=evolution, t=_effect_time(params) * speed)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Feedback / Trail":
        # Preserve untrailed snapshot as new_trail_image; then apply feedback, then trailing to output
        cur = arr.copy()
        new_trail = _np_to_pil(cur)
        out = cur
        fb_int = float(params.get("feedback_intensity", 0.0) or 0.0)
        tr_int = float(params.get("trailing_intensity", 0.0) or 0.0)
        if fb_int > 0 and feedback_image is not None:
            try:
                fb = _pil_to_rgb_np(feedback_image)
                if fb.shape != out.shape:
                    logging.warning(
                        "Discarding feedback history at %s; master shape is %s",
                        fb.shape,
                        out.shape,
                    )
                else:
                    out = cv2.addWeighted(out, 1.0 - fb_int, fb, fb_int, 0)
            except Exception as e:
                logging.error(f"Feedback blend failed: {e}")
        new_feedback = _np_to_pil(out)
        if tr_int > 0 and trail_image is not None:
            try:
                tr = _pil_to_rgb_np(trail_image)
                if tr.shape != out.shape:
                    logging.warning(
                        "Discarding trail history at %s; master shape is %s",
                        tr.shape,
                        out.shape,
                    )
                else:
                    out = cv2.addWeighted(out, 1.0 - tr_int, tr, tr_int, 0)
            except Exception as e:
                logging.error(f"Trailing blend failed: {e}")
        return _ensure_uint8_np(out), previous_frame, new_feedback, new_trail

    if label == "FrameDrop / RGB Split":
        out = arr
        # Simple RGB split (channel shifts)
        dist = int(params.get("rgb_split_distance", 0) or 0)
        if dist > 0:
            h, w, _ = out.shape
            r = np.roll(out[..., 0], -dist, axis=1)
            g = out[..., 1]
            b = np.roll(out[..., 2], dist, axis=1)
            out = np.stack([r, g, b], axis=-1)
        # Frame drop
        drop_chance = float(params.get("frame_drop_chance", 0) or 0)
        if drop_chance > 0 and previous_frame is not None:
            try:
                import numpy as _np
                if _np.random.rand() < drop_chance / 100.0:
                    # Output previous frame, keep feedback/trail unchanged
                    return previous_frame.copy(), previous_frame, feedback_image, trail_image
            except Exception:
                pass
        return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image

    if label == "Pixel Sorting":
        intensity = float(params.get("pixel_sorting_intensity", 0.0) or 0.0)
        if intensity <= 0:
            return img, previous_frame, feedback_image, trail_image
        direction = str(params.get("pixel_sorting_direction", "Horizontal") or "Horizontal")
        sort_chance = float(params.get("sort_chance", 0) or 0) / 100.0
        evolution = float(params.get("pixel_sort_evolution", 0.0) or 0.0)
        speed = float(params.get("pixel_sort_evolution_speed", 1.4) or 1.4)
        out = arr.copy()
        h, w = out.shape[:2]
        t = _effect_time(params) * speed
        anim_chance = float(np.clip(sort_chance + evolution * 0.35 * np.sin(t * 0.9 + 0.2), 0.0, 1.0))
        if direction == "Horizontal":
            row_phase = np.linspace(0.0, np.pi * (2.0 + 10.0 * evolution), h, dtype=np.float32)
            row_mask = 0.5 + 0.5 * np.sin(row_phase + t * (1.4 + 2.8 * evolution))
            for y in range(h):
                local_chance = anim_chance if evolution <= 0 else float(np.clip(anim_chance * (0.35 + 1.35 * row_mask[y]), 0.0, 1.0))
                if np.random.rand() < local_chance:
                    row = out[y, :, :]
                    bright = np.sum(row, axis=1)
                    sort_idx = np.argsort(bright)
                    row_sorted = row[sort_idx]
                    local_intensity = intensity if evolution <= 0 else float(np.clip(intensity * (0.6 + 0.4 * row_mask[y]), 0.0, 1.0))
                    out[y, :, :] = cv2.addWeighted(row, 1.0 - local_intensity, row_sorted, local_intensity, 0)
        else:
            col_phase = np.linspace(0.0, np.pi * (2.0 + 10.0 * evolution), w, dtype=np.float32)
            col_mask = 0.5 + 0.5 * np.sin(col_phase + t * (1.4 + 2.8 * evolution))
            for x in range(w):
                local_chance = anim_chance if evolution <= 0 else float(np.clip(anim_chance * (0.35 + 1.35 * col_mask[x]), 0.0, 1.0))
                if np.random.rand() < local_chance:
                    col = out[:, x, :]
                    bright = np.sum(col, axis=1)
                    sort_idx = np.argsort(bright)
                    col_sorted = col[sort_idx]
                    local_intensity = intensity if evolution <= 0 else float(np.clip(intensity * (0.6 + 0.4 * col_mask[x]), 0.0, 1.0))
                    out[:, x, :] = cv2.addWeighted(col, 1.0 - local_intensity, col_sorted, local_intensity, 0)
        return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image

    if label == "Text Overlay":
        text = str(params.get("glitch_text", "") or "")
        intensity = float(params.get("glitch_text_intensity", 0.0) or 0.0)
        if not text or intensity <= 0:
            return img, previous_frame, feedback_image, trail_image
        base = _np_to_pil(arr).convert('RGBA')
        overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        w, h = base.size
        # Draw centered with slight jitter for glitchy look
        alpha = int(np.clip(intensity * 255, 0, 255))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((w//2 + dx*2, h//2 + dy*2), text, fill=(255, 255, 255, alpha), anchor="mm", font=font)
        out = Image.alpha_composite(base, overlay)
        return np.array(out.convert("RGB")), previous_frame, feedback_image, trail_image

    if label == "Luma Key":
        mode = str(params.get("key_mode", "luma bright") or "luma bright").lower()
        threshold = float(np.clip(params.get("key_threshold", 0.62) or 0.62, 0.0, 1.0))
        softness = float(np.clip(params.get("key_softness", 0.12) or 0.12, 0.0, 1.0))
        opacity = float(np.clip(params.get("key_opacity", 1.0) or 1.0, 0.0, 1.0))
        if opacity <= 0:
            return img, previous_frame, feedback_image, trail_image

        src = arr.astype(np.float32)
        luma = (src[..., 0] * 0.2126 + src[..., 1] * 0.7152 + src[..., 2] * 0.0722) / 255.0

        if mode == "green screen":
            r = src[..., 0] / 255.0
            g = src[..., 1] / 255.0
            b = src[..., 2] / 255.0
            matte_source = np.clip((g - np.maximum(r, b)) * 2.0 + (g - 0.25), 0.0, 1.0)
        elif mode == "luma dark":
            matte_source = 1.0 - luma
        else:
            matte_source = luma

        if softness <= 1e-4:
            matte = (matte_source >= threshold).astype(np.float32)
        else:
            lo = threshold - softness * 0.5
            hi = threshold + softness * 0.5
            matte = np.clip((matte_source - lo) / max(1e-4, hi - lo), 0.0, 1.0)
            matte = matte * matte * (3.0 - 2.0 * matte)

        if bool(params.get("invert_matte", False)):
            matte = 1.0 - matte
        matte = np.clip(matte * opacity, 0.0, 1.0)[..., None]

        fill_mode = str(params.get("fill_mode", "color") or "color").lower()
        if fill_mode == "white":
            fill = np.full_like(src, 255.0)
        elif fill_mode == "black":
            fill = np.zeros_like(src)
        elif fill_mode == "invert":
            fill = 255.0 - src
        elif fill_mode == "posterize":
            fill = np.floor(src / 64.0) * 64.0 + 32.0
        else:
            color = np.array([
                int(np.clip(params.get("fill_red", 0) or 0, 0, 255)),
                int(np.clip(params.get("fill_green", 255) or 255, 0, 255)),
                int(np.clip(params.get("fill_blue", 180) or 180, 0, 255)),
            ], dtype=np.float32)
            fill = np.empty_like(src)
            fill[...] = color

        out = src * (1.0 - matte) + fill * matte
        edge_glow = float(np.clip(params.get("edge_glow", 0.0) or 0.0, 0.0, 1.0))
        if edge_glow > 0:
            matte_u8 = np.clip(matte[..., 0] * 255.0, 0, 255).astype(np.uint8)
            edges = cv2.Canny(matte_u8, 32, 128).astype(np.float32) / 255.0
            edges = cv2.GaussianBlur(edges, (0, 0), max(0.5, 3.0 * edge_glow))
            glow_color = fill if fill.ndim == 3 else np.full_like(src, 255.0)
            out = np.clip(out + glow_color * edges[..., None] * edge_glow, 0, 255)

        return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image

    if label == "Line Glitch":
        count = int(params.get("line_count", 0) or 0)
        max_offset = int(params.get("max_offset", 0) or 0)
        freq = float(params.get("glitch_frequency", 0.0) or 0.0) / 100.0
        intensity = float(params.get("glitch_intensity", 0.0) or 0.0) / 100.0
        if count <= 0 or max_offset <= 0 or freq <= 0:
            return img, previous_frame, feedback_image, trail_image
        out = arr.copy()
        h, w = out.shape[:2]
        rng = current_numpy_rng()
        lines = rng.choice(h, size=min(count, h), replace=False)
        for y in lines:
            if rng.random() < freq:
                offset = int(rng.integers(-max_offset, max_offset+1) * max(0.0, intensity))
                out[y] = np.roll(out[y], offset, axis=0)
        return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image

    if label == "Unfold / Euclid / Random":
        # Minimal: horizontal wave unfold combined with random brightness flicker
        out = arr.astype(np.float32)
        unfold = float(params.get("unfolding_speed", 0.0) or 0.0)
        euclid = float(params.get("euclidean_parameter", 0.0) or 0.0)
        rnd = float(params.get("randomize_intensity", 0.0) or 0.0) / 100.0
        if unfold > 0:
            h, w = out.shape[:2]
            y, x = np.indices((h, w), dtype=np.float32)
            dx = np.sin(2*np.pi * (y/h * max(1.0, euclid))) * unfold * 5.0
            map_x = np.clip(x + dx, 0, w-1)
            map_y = y
            out = ge.gpo.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        if rnd > 0:
            noise = np.random.uniform(-30, 30, out.shape).astype(np.float32) * rnd
            out = np.clip(out + noise, 0, 255)
        return _ensure_uint8_np(out.astype(np.uint8)), previous_frame, feedback_image, trail_image

    if label == "Frame Buffer":
        # No image change; allows setting buffer size limit via node
        if worker_state is not None and hasattr(worker_state, 'buffer_limit'):
            try:
                worker_state.buffer_limit = int(params.get('frame_buffer_size', 10) or 10)
            except Exception:
                pass
        return img, previous_frame, feedback_image, trail_image

    if label == "Pixelation / Chromatic":
        out = arr
        pix = int(params.get("pixelation_intensity", 1) or 1)
        if pix > 1:
            h, w = out.shape[:2]
            ds = max(1, pix)
            out = ge.gpo.resize(out, (max(1, w // ds), max(1, h // ds)), interpolation=cv2.INTER_AREA)
            out = ge.gpo.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)
        ca = float(params.get("chromatic_aberration", 0.0) or 0.0)
        if ca > 0:
            shift = int(max(1, round(ca)))
            h, w = out.shape[:2]
            r = np.roll(out[..., 0], -shift, axis=1)
            b = np.roll(out[..., 2], shift, axis=1)
            out = np.stack([r, out[..., 1], b], axis=-1)
        return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image

    if label == "Data Moshing":
        evolution = float(params.get("data_mosh_evolution", 0.0) or 0.0)
        speed = float(params.get("data_mosh_evolution_speed", 1.0) or 1.0)
        t = _effect_time(params) * speed
        out = arr.astype(np.float32)
        bs = int(params.get("block_size", 16) or 16)
        if evolution > 0:
            bs = max(1, int(round(bs * (1.0 + evolution * 0.5 * np.sin(t * 0.7 + 0.2)))))
        out = ge.block_displacement_np(
            out, bs,
            _modulate_amount(float(params.get("block_displacement", 0.0) or 0.0), evolution, t, 1.1, phase=0.1, swing=0.7),
            str(params.get("displacement_direction", 'random') or 'random')
        )
        out = ge.block_duplication_np(
            out, bs,
            _modulate_amount(float(params.get("block_duplication", 0.0) or 0.0), evolution, t, 1.7, phase=1.2, swing=0.85),
            str(params.get("duplication_pattern", 'random') or 'random')
        )
        out = ge.block_corruption_np(
            out, bs,
            _modulate_amount(float(params.get("block_corruption", 0.0) or 0.0), evolution, t, 1.4, phase=2.1, swing=0.75),
            str(params.get("corruption_type", 'noise') or 'noise')
        )
        out = ge.block_sorting_np(
            out, bs,
            _modulate_amount(float(params.get("block_sorting", 0.0) or 0.0), evolution, t, 0.9, phase=0.8, swing=0.9),
            str(params.get("sort_method", 'brightness') or 'brightness')
        )
        out = ge.motion_vectors_np(
            out,
            _modulate_amount(float(params.get("motion_vectors", 0.0) or 0.0), evolution, t, 1.25, phase=2.7, swing=0.8),
            max(1, int(round(int(params.get("vector_length", 20) or 20) * (1.0 + evolution * 0.45 * np.sin(t * 1.6 + 0.6)))))
        )
        out = ge.bitplane_corruption_np(
            out,
            _modulate_amount(float(params.get("bitplane_corruption", 0.0) or 0.0), evolution, t, 1.5, phase=1.5, swing=0.9),
            int(np.clip(round(int(params.get("bitplane", 1) or 1) + evolution * 2.5 * np.sin(t * 0.85 + 1.9)), 0, 7))
        )
        out = ge.dct_glitch_np(
            out,
            _modulate_amount(float(params.get("dct_glitch_intensity", 0.0) or 0.0), evolution, t, 1.05, phase=0.5, swing=0.85)
        )
        # Slit scan smear
        # Slit scan smear with worker-scoped state
        slit_state = None
        if worker_state is not None and hasattr(worker_state, 'slit_scan_buffer'):
            slit_state = {'buffer': getattr(worker_state, 'slit_scan_buffer', None), 'dir': getattr(worker_state, 'slit_scan_dir', None)}
        out = ge.slit_scan_smear_np(
            out,
            _modulate_amount(float(params.get('slit_scan_intensity', 0.0) or 0.0), evolution, t, 0.8, phase=1.0, swing=0.9),
            str(params.get('slit_scan_direction', 'right_to_left') or 'right_to_left'),
            int(params.get('slit_scan_size', 1) or 1),
            float(np.clip(float(params.get('slit_scan_pos', 0.5) or 0.5) + evolution * 0.35 * np.sin(t * 0.95 + 1.4), 0.0, 1.0)),
            external_state=slit_state
        )
        if slit_state is not None and worker_state is not None:
            try:
                worker_state.slit_scan_buffer = slit_state.get('buffer', None)
                worker_state.slit_scan_dir = slit_state.get('dir', None)
            except Exception:
                pass
        return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image

    if label == "Row Temporal Stutter":
        if frame_buffer:
            out = ge.row_temporal_stutter(frame_buffer, float(params.get('row_stutter', 0.0) or 0.0), int(params.get('row_stutter_max_delay', 4) or 4))
            return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image
        return img, previous_frame, feedback_image, trail_image

    if label == "P-Frame Glitches":
        # Use previous_frame as reference
        ref = _pil_to_rgb_np(previous_frame) if previous_frame is not None else None
        out_np = arr.copy()
        preview_python_rng = params.get('_preview_python_rng')
        preview_numpy_rng = params.get('_preview_numpy_rng')
        # Motion vector corruption
        mvc = float(params.get('motion_vector_corruption', 0.0) or 0.0)
        # Prediction error amplification
        pea = float(params.get('prediction_error_amplify', 0.0) or 0.0)
        block_size = int(str(params.get('macroblock_size', '16')))
        motion_direction = str(params.get('motion_vector_direction', 'random'))
        prediction_direction = str(params.get('prediction_error_direction', 'uniform'))
        error_threshold = int(params.get('error_threshold', 30) or 30)
        fused = None
        if mvc > 0 and pea > 0 and motion_direction == 'spiral' and prediction_direction == 'uniform':
            fused = ge.process_pframe_spiral_uniform(
                out_np,
                ref,
                mvc,
                block_size,
                pea,
                error_threshold,
                output=out_np,
                rng=params.get('_preview_python_rng'),
            )
        if fused is not None:
            out_np = fused
        else:
            if mvc > 0:
                out_np = ge.corrupt_motion_vectors(out_np, ref, mvc, block_size=block_size, direction=motion_direction, rng=preview_python_rng)
            if pea > 0:
                out_np = ge.amplify_prediction_errors(out_np, ref, pea, error_threshold=error_threshold, direction=prediction_direction)
        # Temporal bleeding (uses frame buffer)
        tb = float(params.get('temporal_bleeding', 0.0) or 0.0)
        if tb > 0 and frame_buffer:
            bleed_dist = int(params.get('bleed_distance', 3) or 3)
            needed_buffer = frame_buffer[-(bleed_dist + 1):]
            out_np = ge.temporal_bleeding_effect(needed_buffer, tb, bleed_distance=bleed_dist, direction=str(params.get('temporal_bleeding_direction', 'random')), rng=preview_numpy_rng)
        # Macroblock scrambling
        mbs = float(params.get('macroblock_scramble', 0.0) or 0.0)
        if mbs > 0:
            out_np = ge.macroblock_scrambling(out_np, ref, mbs, block_size=int(str(params.get('macroblock_size', '16'))), pattern=str(params.get('scrambling_pattern', 'random')), rng=preview_python_rng)
        # Residual injection
        ri = float(params.get('residual_injection', 0.0) or 0.0)
        if ri > 0:
            out_np = ge.residual_injection(out_np, ri, noise_type=str(params.get('residual_noise_type', 'gaussian')), direction=str(params.get('residual_direction', 'uniform')), python_rng=preview_python_rng, numpy_rng=preview_numpy_rng)
        # Reference corruption — approximate via macroblock scrambling with a pattern
        rc = float(params.get('reference_corruption', 0.0) or 0.0)
        if rc > 0 and ref is not None:
            out_np = ge.macroblock_scrambling(out_np, ref, rc, block_size=int(str(params.get('macroblock_size', '16'))), pattern=str(params.get('reference_corruption_pattern', 'scattered')), rng=preview_python_rng)
        # Next reference becomes current output
        next_ref = _ensure_uint8_np(out_np)
        return next_ref.copy(), next_ref.copy(), feedback_image, trail_image

    if label == "Invert Colors":
        chance = int(params.get("invert_chance", 0) or 0)
        if chance > 0:
            # Simple deterministic invert when slider > 0
            arr = 255 - arr
        return arr, previous_frame, feedback_image, trail_image

    if label == "Databending":
        intensity = float(params.get("databend_intensity", 0.0) or 0.0)
        if intensity <= 0:
            return img, previous_frame, feedback_image, trail_image
        mode = str(params.get("databend_mode", "random") or "random")
        layers = int(params.get("databend_layers", 1) or 1)
        fmt = str(params.get("databend_format", "jpeg") or "jpeg")
        qual = int(params.get("databend_quality", 75) or 75)
        try:
            out = ge.apply_databending_effect(_np_to_pil(arr), intensity, mode, layers, encoding_format=fmt, jpeg_quality=qual)
            return _ensure_uint8_np(out), previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Databending failed: {e}")
            return img, previous_frame, feedback_image, trail_image

    # Newly added effect placeholders
    if label == "JPEG Compression":
        arr = _wrap_simple(arr, ge.apply_jpeg_compression, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "WebP Corruption":
        arr = _wrap_simple(arr, ge.apply_webp_corruption, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "H264 Artifacts":
        arr = _wrap_simple(arr, ge.apply_h264_artifacts, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Frame Interpolation Glitch":
        arr = _wrap_simple(arr, ge.apply_frame_interpolation_glitch, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Temporal Slicing":
        arr = _wrap_simple(arr, ge.apply_temporal_slicing, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Time Displacement":
        arr = _wrap_simple(arr, ge.apply_time_displacement, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "CRT Phosphor Decay":
        arr = _wrap_simple(arr, ge.apply_crt_phosphor_decay, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Analog TV":
        arr = _wrap_simple(arr, ge.apply_analog_tv, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Tape Dropout":
        arr = _wrap_simple(arr, ge.apply_tape_dropout, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Network Packet Loss":
        arr = _wrap_simple(arr, ge.apply_network_packet_loss, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Buffer Underrun":
        arr = _wrap_simple(arr, ge.apply_buffer_underrun, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Digital Clipping":
        arr = _wrap_simple(arr, ge.apply_digital_clipping, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Wavelet Corruption":
        arr = _wrap_simple(arr, ge.apply_wavelet_corruption, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Fractal Noise":
        arr = _wrap_simple(arr, ge.apply_fractal_noise, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Cellular Automata":
        arr = _wrap_simple(arr, ge.apply_cellular_automata, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Spectral Shift":
        arr = _wrap_simple(arr, ge.apply_spectral_shift, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "ColorSpace Corruption":
        arr = _wrap_simple(arr, ge.apply_colorspace_corruption, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "LUT Corruption":
        arr = _wrap_simple(arr, ge.apply_lut_corruption, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Vertex Shader Glitch":
        arr = _wrap_simple(arr, ge.apply_vertex_shader_glitch, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Perspective Glitch":
        arr = _wrap_simple(arr, ge.apply_perspective_glitch, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Enhanced Data Moshing":
        arr = _wrap_simple(arr, ge.apply_enhanced_data_moshing, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Enhanced Pixel Sorting":
        arr = _wrap_simple(arr, ge.apply_enhanced_pixel_sorting, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Effect Blender":
        arr = _wrap_simple(arr, ge.apply_effect_blender, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Conditional Effect":
        arr = _wrap_simple(arr, ge.apply_conditional_effect, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Frame Difference":
        arr = _wrap_simple(arr, ge.apply_frame_difference, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Feedback Loop":
        arr = _wrap_simple(arr, ge.apply_feedback_loop, **params)
        return arr, previous_frame, feedback_image, trail_image

    if label == "Faultline Possession":
        if params is not None and str(params.get("enabled", True)).lower() == "false":
            return img, previous_frame, feedback_image, trail_image
        try:
            from faultline_possession import FaultlinePossessionEngine

            h, w = arr.shape[:2]
            node_id = params.get("_node_id", "default")
            engine_key = f"faultline_possession::{node_id}"
            effect_states = getattr(worker_state, "effect_states", None)

            if effect_states is not None:
                engine = effect_states.get(engine_key)
            else:
                engine = None

            target_territories = int(params.get("fp_territory_count", 8) or 8)
            if (
                engine is None or
                engine.width != w or
                engine.height != h or
                engine.n_territories != max(3, target_territories)
            ):
                seed = None
                if node_id is not None:
                    seed = stable_seed(node_id, w, h, "faultline_possession")
                engine = FaultlinePossessionEngine(
                    width=w,
                    height=h,
                    n_territories=target_territories,
                    mutation_rate=float(params.get("fp_mutation_rate", 0.025) or 0.025),
                    history_offset=float(params.get("history_offset", 4.0) or 4.0),
                    memory_decay=float(params.get("memory_decay", params.get("fp_memory_decay", 0.93)) or 0.93),
                    event_hazard=float(params.get("event_hazard", 0.05) or 0.05),
                    clock_quantization=float(params.get("clock_quantization", 0.0) or 0.0),
                    sync_desync=float(params.get("sync_desync", 0.0) or 0.0),
                    phase_reset_chance=float(params.get("phase_reset_chance", 0.01) or 0.01),
                    local_time_warp=float(params.get("local_time_warp", 0.0) or 0.0),
                    rupture_threshold=float(params.get("rupture_threshold", 0.5) or 0.5),
                    hysteresis_speed=float(params.get("hysteresis_speed", 0.08) or 0.08),
                    modulator_coupling=float(params.get("modulator_coupling", 0.0) or 0.0),
                    scar_decay=float(params.get("fp_scar_decay", 0.989) or 0.989),
                    takeover_rate=float(params.get("fp_takeover_rate", 0.055) or 0.055),
                    rupture_gain=float(params.get("fp_rupture_gain", 1.6) or 1.6),
                    reset_probability=float(params.get("fp_reset_probability", 0.002) or 0.002),
                    max_displacement=float(params.get("fp_max_displacement", 22.0) or 22.0),
                    seed=seed,
                )
                if effect_states is not None:
                    effect_states[engine_key] = engine
            else:
                engine.configure(
                    n_territories=target_territories,
                    mutation_rate=float(params.get("fp_mutation_rate", 0.025) or 0.025),
                    history_offset=float(params.get("history_offset", 4.0) or 4.0),
                    memory_decay=float(params.get("memory_decay", params.get("fp_memory_decay", 0.93)) or 0.93),
                    event_hazard=float(params.get("event_hazard", 0.05) or 0.05),
                    clock_quantization=float(params.get("clock_quantization", 0.0) or 0.0),
                    sync_desync=float(params.get("sync_desync", 0.0) or 0.0),
                    phase_reset_chance=float(params.get("phase_reset_chance", 0.01) or 0.01),
                    local_time_warp=float(params.get("local_time_warp", 0.0) or 0.0),
                    rupture_threshold=float(params.get("rupture_threshold", 0.5) or 0.5),
                    hysteresis_speed=float(params.get("hysteresis_speed", 0.08) or 0.08),
                    modulator_coupling=float(params.get("modulator_coupling", 0.0) or 0.0),
                    scar_decay=float(params.get("fp_scar_decay", 0.989) or 0.989),
                    takeover_rate=float(params.get("fp_takeover_rate", 0.055) or 0.055),
                    rupture_gain=float(params.get("fp_rupture_gain", 1.6) or 1.6),
                    reset_probability=float(params.get("fp_reset_probability", 0.002) or 0.002),
                    max_displacement=float(params.get("fp_max_displacement", 22.0) or 22.0),
                )

            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            result_bgr = engine.process(bgr)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            return _ensure_uint8_np(result_rgb), previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Faultline Possession effect failed: {e}", exc_info=True)
            return img, previous_frame, feedback_image, trail_image

    # ===== QUANTUM GESTALT COGNITIVE GLITCH EFFECTS =====
    
    if label == "Quantum Gestalt (AI)":
        try:
            from quantum_gestalt import QuantumGestaltEngine, CognitivePredictor
            
            # Initialize engine (cached in worker_state if available)
            engine = getattr(worker_state, 'qg_engine', None)
            if engine is None:
                engine = QuantumGestaltEngine(use_cuda=True, num_workers=2)
                if worker_state is not None:
                    worker_state.qg_engine = engine
            
            # Get parameters
            intensity = params.get('qg_intensity', 0.5)
            state_str = params.get('qg_state', 'Auto (AI)')
            progress = params.get('qg_progress', 0.0)
            auto_animate = params.get('qg_auto_animate', True)
            
            # Auto-increment progress if enabled
            if auto_animate:
                import time
                progress = (time.time() % 10.0) / 10.0  # 10-second cycle
            
            # Determine cognitive state
            if state_str == "Auto (AI)":
                # Use AI predictor
                predictor = getattr(worker_state, 'qg_predictor', None)
                if predictor is None:
                    predictor = CognitivePredictor()
                    if worker_state is not None:
                        worker_state.qg_predictor = predictor
                
                prediction = predictor.predict(img)
                cognitive_state = prediction['probabilities']
            else:
                # Manual state selection
                state_map = {
                    "Delta (Relaxed)": 0,
                    "Theta (Meditative)": 1,
                    "Alpha (Calm)": 2,
                    "Beta (Active)": 3,
                    "Gamma (Hyper-focus)": 4
                }
                cognitive_state = state_map.get(state_str, 2)
            
            # Apply cognitive glitch
            result = engine.apply_cognitive_glitch(img, cognitive_state, intensity, progress)
            # Ensure numpy output
            if isinstance(result, Image.Image):
                result = np.array(result.convert("RGB"))
            return result, previous_frame, feedback_image, trail_image
            
        except Exception as e:
            logging.error(f"Quantum Gestalt (AI) effect failed: {e}")
            return img, previous_frame, feedback_image, trail_image
    
    if label == "QG Delta Wave":
        try:
            from quantum_gestalt import QuantumGestaltEngine
            engine = getattr(worker_state, 'qg_engine', None)
            if engine is None:
                engine = QuantumGestaltEngine(use_cuda=True, num_workers=2)
                if worker_state is not None:
                    worker_state.qg_engine = engine
            
            intensity = params.get('qg_delta_intensity', 0.3)
            progress = params.get('qg_delta_progress', 0.0)
            result = engine.apply_cognitive_glitch(img, 0, intensity, progress)
            if isinstance(result, Image.Image):
                result = np.array(result.convert("RGB"))
            return result, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"QG Delta Wave effect failed: {e}")
            return img, previous_frame, feedback_image, trail_image
    
    if label == "QG Theta Flow":
        try:
            from quantum_gestalt import QuantumGestaltEngine
            engine = getattr(worker_state, 'qg_engine', None)
            if engine is None:
                engine = QuantumGestaltEngine(use_cuda=True, num_workers=2)
                if worker_state is not None:
                    worker_state.qg_engine = engine
            
            intensity = params.get('qg_theta_intensity', 0.4)
            progress = params.get('qg_theta_progress', 0.0)
            result = engine.apply_cognitive_glitch(img, 1, intensity, progress)
            if isinstance(result, Image.Image):
                result = np.array(result.convert("RGB"))
            return result, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"QG Theta Flow effect failed: {e}")
            return img, previous_frame, feedback_image, trail_image
    
    if label == "QG Alpha Balance":
        try:
            from quantum_gestalt import QuantumGestaltEngine
            engine = getattr(worker_state, 'qg_engine', None)
            if engine is None:
                engine = QuantumGestaltEngine(use_cuda=True, num_workers=2)
                if worker_state is not None:
                    worker_state.qg_engine = engine
            
            intensity = params.get('qg_alpha_intensity', 0.5)
            progress = params.get('qg_alpha_progress', 0.0)
            result = engine.apply_cognitive_glitch(img, 2, intensity, progress)
            if isinstance(result, Image.Image):
                result = np.array(result.convert("RGB"))
            return result, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"QG Alpha Balance effect failed: {e}")
            return img, previous_frame, feedback_image, trail_image
    
    if label == "QG Beta Pulse":
        try:
            from quantum_gestalt import QuantumGestaltEngine
            engine = getattr(worker_state, 'qg_engine', None)
            if engine is None:
                engine = QuantumGestaltEngine(use_cuda=True, num_workers=2)
                if worker_state is not None:
                    worker_state.qg_engine = engine
            
            intensity = params.get('qg_beta_intensity', 0.7)
            progress = params.get('qg_beta_progress', 0.0)
            result = engine.apply_cognitive_glitch(img, 3, intensity, progress)
            if isinstance(result, Image.Image):
                result = np.array(result.convert("RGB"))
            return result, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"QG Beta Pulse effect failed: {e}")
            return img, previous_frame, feedback_image, trail_image
    
    if label == "QG Gamma Chaos":
        try:
            from quantum_gestalt import QuantumGestaltEngine
            engine = getattr(worker_state, 'qg_engine', None)
            if engine is None:
                engine = QuantumGestaltEngine(use_cuda=True, num_workers=2)
                if worker_state is not None:
                    worker_state.qg_engine = engine
            
            intensity = params.get('qg_gamma_intensity', 0.9)
            progress = params.get('qg_gamma_progress', 0.0)
            result = engine.apply_cognitive_glitch(img, 4, intensity, progress)
            if isinstance(result, Image.Image):
                result = np.array(result.convert("RGB"))
            return result, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"QG Gamma Chaos effect failed: {e}")
            return img, previous_frame, feedback_image, trail_image
    
    # ===== GRAVITATIONAL PARTICLE SVG EFFECT =====
    
    if label == "Gravitational Particle":
        try:
            from svg_particle_engine import ParticlePhysicsEngine
            from svg_generator import SVGGenerator, composite_images
            # numpy already imported at module level as np
            
            # Get or initialize engine in worker state
            engine = getattr(worker_state, 'particle_engine', None)
            svg_gen = getattr(worker_state, 'svg_generator', None)
            
            w, h = img.size
            
            # Initialize if needed
            if engine is None or svg_gen is None:
                max_particles = int(params.get('max_particles', 500))
                gravity_x = float(params.get('gravity_x', 0.0))
                gravity_y = float(params.get('gravity_y', 0.5))
                damping = float(params.get('damping', 0.99))
                
                engine = ParticlePhysicsEngine(
                    width=w, height=h,
                    max_particles=max_particles,
                    gravity=(gravity_x, gravity_y),
                    damping=damping
                )
                svg_gen = SVGGenerator(width=w, height=h)
                
                if worker_state is not None:
                    worker_state.particle_engine = engine
                    worker_state.svg_generator = svg_gen
            
            # Update dimensions if changed
            if engine.width != w or engine.height != h:
                engine.set_dimensions(w, h)
                svg_gen.set_dimensions(w, h)
            
            # Reset if requested
            if params.get('reset_particles', False):
                engine.reset()
            
            # Update physics parameters
            engine.gravity = np.array([
                float(params.get('gravity_x', 0.0)),
                float(params.get('gravity_y', 0.5))
            ], dtype=np.float32)
            engine.damping = float(params.get('damping', 0.99))
            engine.particle_interaction_enabled = bool(params.get('particle_interaction', False))
            engine.particle_interaction_strength = float(params.get('interaction_strength', 0.1))
            engine.collision_enabled = bool(params.get('collision_enabled', False))
            engine.vector_field_enabled = bool(params.get('vector_field', False))
            engine.vector_field_scale = float(params.get('vector_field_scale', 50.0))
            engine.vector_field_intensity = float(params.get('vector_field_intensity', 1.0))
            engine.boundary_mode = str(params.get('boundary_mode', 'wrap'))
            engine.audio_reactive = bool(params.get('audio_reactive', True))
            
            # Update gravity wells
            num_wells = int(params.get('num_gravity_wells', 0))
            well_strength = float(params.get('well_strength', 20.0))
            well_radius = float(params.get('well_radius', 200.0))
            
            # Clear and recreate wells
            engine.clear_gravity_wells()
            if num_wells > 0:
                for i in range(num_wells):
                    # Distribute wells in a circle pattern
                    angle = 2 * np.pi * i / num_wells
                    radius = min(w, h) * 0.3
                    wx = w / 2 + radius * np.cos(angle)
                    wy = h / 2 + radius * np.sin(angle)
                    engine.add_gravity_well(wx, wy, well_strength, well_radius)
            
            # Audio reactivity - get audio data from worker state if available
            if engine.audio_reactive and worker_state is not None:
                audio_data = getattr(worker_state, 'audio_analysis', None)
                if audio_data:
                    engine.update_audio_data(audio_data)
            
            # Spawn new particles
            spawn_rate = int(params.get('spawn_rate', 5))
            size_min = float(params.get('size_min', 3.0))
            size_max = float(params.get('size_max', 10.0))
            lifetime = int(params.get('particle_lifetime', 120))
            
            # Determine shape type(s)
            shape_type = str(params.get('shape_type', 'circle'))
            if shape_type == 'mixed':
                shape_types = ['circle', 'polygon', 'star', 'curve']
            else:
                shape_types = [shape_type]
            
            # Color palette
            color_palette = [
                (
                    int(params.get('color_r_start', 100)),
                    int(params.get('color_g_start', 150)),
                    int(params.get('color_b_start', 255)),
                    int(params.get('opacity', 0.8) * 255)
                ),
                (
                    int(params.get('color_r_end', 255)),
                    int(params.get('color_g_end', 100)),
                    int(params.get('color_b_end', 150)),
                    int(params.get('opacity', 0.8) * 255)
                )
            ]
            
            # Spawn particles
            if spawn_rate > 0:
                engine.spawn_particles_batch(
                    count=spawn_rate,
                    spawn_area=(0, 0, w, h),
                    velocity_range=(-5.0, 5.0),
                    size_range=(size_min, size_max),
                    color_palette=color_palette,
                    shape_types=shape_types,
                    lifetime_range=(lifetime, lifetime) if lifetime > 0 else (-1, -1)
                )
            
            # Update physics
            engine.update(dt=1.0)
            
            # Get active particles
            particles = engine.get_active_particles()
            
            # Set background mode
            bg_mode = str(params.get('background_mode', 'transparent'))
            if bg_mode == 'black':
                svg_gen.background_color = (0, 0, 0, 255)
            elif bg_mode == 'white':
                svg_gen.background_color = (255, 255, 255, 255)
            else:
                svg_gen.background_color = None
            
            # Generate SVG
            include_trails = bool(params.get('include_trails', True))
            trail_fade = bool(params.get('trail_fade', True))
            use_gradients = bool(params.get('use_gradients', True))
            
            svg_string = svg_gen.generate_svg(
                particles,
                include_trails=include_trails,
                trail_fade=trail_fade,
                use_gradients=use_gradients
            )
            
            # Export SVG if requested
            if params.get('export_svg', False):
                svg_path = str(params.get('svg_output_path', 'svg_templates/particles.svg'))
                svg_gen.save_svg(svg_string, svg_path)
            
            # Rasterize SVG
            particle_img = svg_gen.rasterize_svg(svg_string, output_size=(w, h))
            
            # Composite with input image
            composite_mode = str(params.get('composite_mode', 'overlay'))
            
            if composite_mode == 'replace':
                # Just return particles
                result = particle_img
            elif composite_mode == 'overlay':
                result = composite_images(img, particle_img, blend_mode='normal')
            elif composite_mode in ['add', 'multiply', 'screen']:
                result = composite_images(img, particle_img, blend_mode=composite_mode)
            else:
                result = composite_images(img, particle_img, blend_mode='normal')
            
            return result, previous_frame, feedback_image, trail_image
            
        except Exception as e:
            logging.error(f"Gravitational Particle effect failed: {e}", exc_info=True)
            return img, previous_frame, feedback_image, trail_image

    # ===== QUALITY-FIRST OVERHAUL NODES =====

    if label == "Multimodal Analysis":
        try:
            node_id = params.get("_node_id", "1")
            effect_states = getattr(worker_state, "effect_states", None) if worker_state is not None else None
            state_key = f"multimodal_analysis::{node_id}"
            state = effect_states.setdefault(state_key, {}) if isinstance(effect_states, dict) else {}

            prev_source = state.get("previous_source")
            out_img, maps = ge.multimodal_analysis_np(arr, previous_frame=prev_source, params=params)
            
            # Update state history at full master resolution
            if isinstance(effect_states, dict):
                state["previous_source"] = arr.copy()
                state["analysis"] = maps
                # Mirror individual maps into effect_states matching quality benchmark keys
                for map_name, map_arr in maps.items():
                    effect_states[f"multimodal_analysis::{node_id}.analysis.{map_name}"] = map_arr
                effect_states[f"multimodal_analysis::{node_id}.previous_source"] = state["previous_source"]
                if worker_state is not None:
                    worker_state.latest_multimodal_analysis = maps

            return out_img, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Multimodal Analysis effect failed: {e}", exc_info=True)
            return arr, previous_frame, feedback_image, trail_image

    if label == "Optical Flow Remap":
        try:
            latest_maps = getattr(worker_state, "latest_multimodal_analysis", None) if worker_state is not None else None
            flow_vec = latest_maps.get("optical_flow_vector") if isinstance(latest_maps, dict) else None
            mod_mask = latest_maps.get("modulation_mask") if isinstance(latest_maps, dict) else None
            out_img = ge.optical_flow_remap_np(arr, flow_vector=flow_vec, modulation_mask=mod_mask, params=params)
            return out_img, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Optical Flow Remap effect failed: {e}", exc_info=True)
            return arr, previous_frame, feedback_image, trail_image

    if label == "Controlled Recursive Feedback":
        try:
            node_id = params.get("_node_id", "1")
            effect_states = getattr(worker_state, "effect_states", None) if worker_state is not None else None
            state_key = f"controlled_feedback::{node_id}"
            state = effect_states.setdefault(state_key, {}) if isinstance(effect_states, dict) else {}

            prev_fb = state.get("feedback._previous_output")
            clean_source = state.get("clean_source", arr)
            latest_maps = getattr(worker_state, "latest_multimodal_analysis", None) if worker_state is not None else None
            mot_mask = latest_maps.get("motion_mask") if isinstance(latest_maps, dict) else None

            out_img, next_fb = ge.controlled_recursive_feedback_np(
                arr,
                previous_feedback=prev_fb,
                clean_source=clean_source,
                motion_mask=mot_mask,
                params=params,
                state=state,
            )
            state["feedback._previous_output"] = next_fb.copy()
            if isinstance(effect_states, dict):
                effect_states[f"controlled_feedback::{node_id}.feedback._previous_output"] = state["feedback._previous_output"]

            return out_img, previous_frame, next_fb, trail_image
        except Exception as e:
            logging.error(f"Controlled Recursive Feedback effect failed: {e}", exc_info=True)
            return arr, previous_frame, feedback_image, trail_image

    if label == "Quantization / Solarization":
        try:
            latest_maps = getattr(worker_state, "latest_multimodal_analysis", None) if worker_state is not None else None
            mod_mask = latest_maps.get("modulation_mask") if isinstance(latest_maps, dict) else None
            out_img = ge.quantization_solarization_np(arr, modulation_mask=mod_mask, params=params)
            return out_img, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Quantization / Solarization effect failed: {e}", exc_info=True)
            return arr, previous_frame, feedback_image, trail_image

    if label == "Edge Reconstruction":
        try:
            latest_maps = getattr(worker_state, "latest_multimodal_analysis", None) if worker_state is not None else None
            edge_mask = latest_maps.get("edge_mask") if isinstance(latest_maps, dict) else None
            out_img = ge.edge_reconstruction_np(arr, edge_mask=edge_mask, params=params)
            return out_img, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Edge Reconstruction effect failed: {e}", exc_info=True)
            return arr, previous_frame, feedback_image, trail_image

    if label == "Structural Restoration":
        try:
            latest_maps = getattr(worker_state, "latest_multimodal_analysis", None) if worker_state is not None else None
            edge_mask = latest_maps.get("edge_mask") if isinstance(latest_maps, dict) else None
            out_img = ge.structural_restoration_np(arr, edge_mask=edge_mask, params=params)
            return out_img, previous_frame, feedback_image, trail_image
        except Exception as e:
            logging.error(f"Structural Restoration effect failed: {e}", exc_info=True)
            return arr, previous_frame, feedback_image, trail_image

    # Fallback to the orchestrator for complex or not-yet-mapped nodes
    try:
        # Orchestrator now handles both PIL and numpy (prefers numpy)
        glitched, p, f, t = ge.apply_glitch_effect(img, params, previous_frame, feedback_image, trail_image, frame_buffer)
        # Ensure numpy array output
        if isinstance(glitched, Image.Image):
            glitched = np.array(glitched.convert("RGB"))
        return glitched, p, f, t
    except Exception as e:
        logging.error(f"Fallback orchestrator failed: {e}")
        # Ensure we return numpy
        result = img if isinstance(img, np.ndarray) else np.array(img.convert("RGB"))
        return result, previous_frame, feedback_image, trail_image
