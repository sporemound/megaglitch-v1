"""
CUDA Effects Processing Module

Handles GPU-accelerated glitch effects using the megaglitchnode pipeline.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

# Import GPU operations module for CUDA acceleration
import gpu_ops as gpo

logger = logging.getLogger(__name__)

# Global state for temporal effects (per-effect instances)
_cuda_effect_states = {}
_current_cache_epoch = 0  # Track cache epoch for invalidation

# Global GPU context and nodes for persistence across frames
_gpu_context = None
_gpu_nodes = {}

# Production graph execution installs a stream-owned scope for every CUDA node.
# The legacy globals remain only as a compatibility fallback for callers that
# invoke individual CUDA helpers directly. A scope is backed by the owning
# FrameState.effect_states map, so decks, duplicate nodes, live sessions, and
# consecutive offline renders cannot share temporal accumulators.
_cuda_state_scope = ContextVar("megaglitch_cuda_state_scope", default=None)


@contextmanager
def use_cuda_state_store(state_store: Optional[dict], namespace: object):
    """Route CUDA temporal state into one node/stream-owned mapping."""
    if state_store is None:
        yield
        return
    token = _cuda_state_scope.set(
        {"store": state_store, "namespace": str(namespace)}
    )
    try:
        yield
    finally:
        _cuda_state_scope.reset(token)


def _scoped_runtime():
    scope = _cuda_state_scope.get()
    if scope is None:
        return None
    store = scope["store"]
    runtime = store.setdefault(
        "__cuda_runtime__",
        {"context": None, "nodes": {}, "cache_epoch": _current_cache_epoch},
    )
    return scope, runtime

def _get_shape(arr):
    if hasattr(cv2, 'cuda_GpuMat') and isinstance(arr, cv2.cuda_GpuMat):
        w, h = arr.size()
        return (h, w)
    try:
        from megaglitchnode.gpu import Frame
        if isinstance(arr, Frame):
            return (arr.height, arr.width)
    except Exception:
        pass
    if hasattr(arr, 'shape'):
        return arr.shape[:2]
    return (0, 0)

def get_gpu_context():
    """Get or create global GPU context."""
    global _gpu_context
    scoped = _scoped_runtime()
    if scoped is not None:
        _, runtime = scoped
        if runtime.get("context") is None:
            try:
                from megaglitchnode.gpu import Context
                runtime["context"] = Context(use_stream=False, frame_index=0)
            except ImportError:
                pass
        return runtime.get("context")
    if _gpu_context is None:
        try:
            from megaglitchnode.gpu import Context
            _gpu_context = Context(use_stream=False, frame_index=0)
        except ImportError:
            pass
    return _gpu_context

def get_gpu_node(node_type: str, node_id: str):
    """Get or create a GPU node (persistent across calls)."""
    global _gpu_nodes
    scoped = _scoped_runtime()
    if scoped is not None:
        scope, runtime = scoped
        node_id = f"{scope['namespace']}::{node_id}"
        node_cache = runtime["nodes"]
    else:
        node_cache = _gpu_nodes
    key = f"{node_type}_{node_id}"
    if key not in node_cache:
        try:
            from megaglitchnode.gpu.nodes import GpuWarpRemap, GpuFeedbackMixer, GpuBloomGhost
            
            if node_type == "warp":
                node_cache[key] = GpuWarpRemap(node_id)
            elif node_type == "feedback":
                node_cache[key] = GpuFeedbackMixer(node_id)
            elif node_type == "bloom":
                node_cache[key] = GpuBloomGhost(node_id)
        except ImportError:
            pass
    return node_cache.get(key)


def get_effect_state(effect_name: str, frame_shape: tuple):
    """Get or create effect state. Checks cache epoch to detect invalidation."""
    global _current_cache_epoch
    
    # Check if cache epoch changed (caches need to be rebuilt)
    try:
        from megaglitchnode.opencv_cuda_bootstrap import get_cache_epoch
        current_epoch = get_cache_epoch()
        if current_epoch != _current_cache_epoch:
            logger.info(f"Cache epoch changed ({_current_cache_epoch} -> {current_epoch}), clearing effect states")
            _cuda_effect_states.clear()
            _current_cache_epoch = current_epoch
    except ImportError:
        pass
    
    scope = _cuda_state_scope.get()
    if scope is None:
        state_store = _cuda_effect_states
        key = f"{effect_name}_{frame_shape}"
    else:
        state_store = scope["store"]
        runtime = state_store.setdefault(
            "__cuda_runtime__",
            {"context": None, "nodes": {}, "cache_epoch": _current_cache_epoch},
        )
        if runtime.get("cache_epoch") != _current_cache_epoch:
            state_store.clear()
            state_store["__cuda_runtime__"] = {
                "context": None,
                "nodes": {},
                "cache_epoch": _current_cache_epoch,
            }
        key = (scope["namespace"], effect_name, tuple(frame_shape))
    if key not in state_store:
        state_store[key] = {}
    return state_store[key]


def reset_cuda_states():
    """Reset all CUDA effect states and GPU node caches.
    
    This should be called:
    - On CUDA DLL update to avoid stale device state
    - Before batch rendering for consistent results
    - When changing GPU configuration
    """
    global _cuda_effect_states, _gpu_nodes, _gpu_context
    
    # Clear effect states (feedback buffers, flow accumulators, etc.)
    _cuda_effect_states = {}
    
    # Clear GPU nodes (will be recreated on next use)
    _gpu_nodes = {}
    
    # Clear GPU context
    _gpu_context = None
    
    # Also clear gpu_ops caches
    try:
        gpo._clear_cached_state()
    except Exception:
        pass
    
    logger.info("Reset all CUDA effect states and GPU caches")


def apply_cuda_flow_warp(img_array: np.ndarray, params: dict, prev_frame: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CUDA optical flow warping."""
    try:
        import gpu_ops as gpo
        
        h, w = img_array.shape[:2]
        state = get_effect_state("flow_warp", (h, w))
        
        flow_gain = params.get("flow_gain", 2.0)
        nonlinearity = params.get("nonlinearity", 1.2)
        motion_threshold = params.get("motion_threshold", 3.0)
        temporal_smooth = params.get("temporal_smooth", 0.2)
        border_mode = params.get("border_mode", "reflect")
        
        if (
            state.get("coord_size") != (h, w) or
            "x_coords" not in state or
            "y_coords" not in state
        ):
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            state["x_coords"] = x_coords
            state["y_coords"] = y_coords
            state["coord_size"] = (h, w)

        # Use frame difference to estimate motion (optical flow not available)
        if prev_frame is not None:

            # Match the original pseudo-flow energy: equal-channel mean of
            # absolute frame difference, not luma-weighted grayscale.
            diff = cv2.absdiff(img_array, prev_frame).astype(np.float32)
            if diff.ndim == 3:
                diff_gray = np.mean(diff, axis=2)
            else:
                diff_gray = diff
            grad_x = cv2.Sobel(diff_gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(diff_gray, cv2.CV_32F, 0, 1, ksize=3)

            # Apply temporal smoothing
            if (
                "flow_accum_x" not in state or
                "flow_accum_y" not in state or
                state["flow_accum_x"].shape != grad_x.shape
            ):
                state["flow_accum_x"] = grad_x
                state["flow_accum_y"] = grad_y
            else:
                state["flow_accum_x"] = cv2.addWeighted(
                    grad_x,
                    temporal_smooth,
                    state["flow_accum_x"],
                    1.0 - temporal_smooth,
                    0.0,
                )
                state["flow_accum_y"] = cv2.addWeighted(
                    grad_y,
                    temporal_smooth,
                    state["flow_accum_y"],
                    1.0 - temporal_smooth,
                    0.0,
                )
            
            # Apply nonlinearity
            flow_mag = np.sqrt(state["flow_accum_x"] ** 2 + state["flow_accum_y"] ** 2)
            flow_mag = np.where(flow_mag > motion_threshold, flow_mag, 0)
            
            if nonlinearity != 1.0 and flow_mag.max() > 0:
                flow_mag_norm = flow_mag / (flow_mag.max() + 1e-6)
                flow_mag_warped = flow_mag_norm ** nonlinearity
                scale = flow_mag_warped / (flow_mag_norm + 1e-6)
                state["flow_accum_x"] *= scale
                state["flow_accum_y"] *= scale
            
            # Apply gain and create displacement maps
            map_x = state["x_coords"] + state["flow_accum_x"] * flow_gain
            map_y = state["y_coords"] + state["flow_accum_y"] * flow_gain
            
            # Choose border mode
            border_dict = {"reflect": cv2.BORDER_REFLECT_101, "wrap": cv2.BORDER_WRAP, "constant": cv2.BORDER_CONSTANT}
            border_flag = border_dict.get(border_mode, cv2.BORDER_REFLECT_101)
            
            # Apply remap
            result = cv2.remap(img_array, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border_flag)
            return result, img_array
        
        return img_array, img_array
        
    except Exception as e:
        logger.error(f"CUDA Flow Warp error: {e}")
        return img_array, img_array


def apply_cuda_spectral_datamosh(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Apply CUDA DFT-based spectral manipulation."""
    try:
        mix = params.get("mix", 0.7)
        band_kill_low = params.get("band_kill_low", 0.05)
        band_kill_high = params.get("band_kill_high", 0.85)
        notch_count = int(params.get("notch_count", 8))
        notch_width = params.get("notch_width", 0.03)
        mask_lfo_rate = params.get("mask_lfo_rate", 0.7)
        
        h, w = img_array.shape[:2]
        state = get_effect_state("spectral", (h, w))
        
        # Use frame-based time_val for deterministic animation
        # Apply mask_lfo_rate directly to time_val for natural animation speed
        mask_phase = time_val * mask_lfo_rate * 2.0  # Multiply by 2 for reasonable speed
        
        # Process each channel
        result_channels = []
        for ch in cv2.split(img_array):
            ch_float = ch.astype(np.float32) / 255.0
            
            # FFT
            dft = np.fft.fft2(ch_float)
            dft_shifted = np.fft.fftshift(dft)
            
            # Create spectral mask
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[0:h, 0:w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            dist_norm = dist / max(cx, cy)
            
            mask = np.ones((h, w), dtype=np.float32)
            mask[dist_norm < band_kill_low] = 0.0
            mask[dist_norm > band_kill_high] = 0.0
            
            # Add animated notches
            if notch_count > 0:
                angle = np.arctan2(y - cy, x - cx)
                for i in range(notch_count):
                    notch_angle = (i / notch_count) * 2 * np.pi + mask_phase
                    # Fix: ensure notch_angle works with array broadcasting
                    angle_diff = np.abs(np.mod(angle - notch_angle + np.pi, 2 * np.pi) - np.pi)
                    notch_mask = 1.0 - np.exp(-angle_diff**2 / (notch_width**2 + 1e-6))
                    mask *= notch_mask
            
            # Apply mask
            dft_shifted *= mask
            
            # IFFT
            dft_unshifted = np.fft.ifftshift(dft_shifted)
            idft = np.fft.ifft2(dft_unshifted)
            result_ch = np.abs(idft)
            result_ch = np.clip(result_ch * 255, 0, 255).astype(np.uint8)
            result_channels.append(result_ch)
        
        result = cv2.merge(result_channels)
        
        # Blend with original
        final = cv2.addWeighted(img_array, 1.0 - mix, result, mix, 0.0)
        return final
        
    except Exception as e:
        logger.error(f"CUDA Spectral Datamosh error: {e}")
        return img_array


def apply_cuda_nonlinear_remap(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Apply CUDA nonlinear displacement field with GPU acceleration.
    
    Uses GPU-first approach when CUDA is enabled:
    - Caches displacement maps on GPU when phase doesn't change much
    - Only transfers result to CPU
    """
    try:
        h, w = _get_shape(img_array)
        state = get_effect_state("nonlinear_remap", (h, w))
        
        field_scale = float(params.get("field_scale", 0.25))
        amplitude_px = float(params.get("amplitude_px", 50.0))
        evolve_rate = float(params.get("evolve_rate", 1.0))
        
        # Use frame-based time_val for deterministic animation
        phase = time_val * evolve_rate
        
        # Generate displacement maps (CPU - fast for small images, cached for large)
        # Round phase to reduce map regeneration frequency
        phase_key = round(phase * 10) / 10  # Round to 0.1
        
        if state.get("phase_key") != phase_key or state.get("map_size") != (h, w):
            # Generate new displacement maps
            y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
            
            disp_x = np.sin(x_grid * field_scale * 0.1 + phase) * amplitude_px
            disp_y = np.cos(y_grid * field_scale * 0.1 + phase * 1.3) * amplitude_px
            
            map_x = (x_grid + disp_x).astype(np.float32)
            map_y = (y_grid + disp_y).astype(np.float32)
            
            state["map_x"] = map_x
            state["map_y"] = map_y
            state["phase_key"] = phase_key
            state["map_size"] = (h, w)
            
            # Upload to GPU if CUDA enabled
            if gpo.get_use_cuda():
                try:
                    state["gpu_map_x"] = gpo.upload(map_x)
                    state["gpu_map_y"] = gpo.upload(map_y)
                except Exception as e:
                    logger.debug(f"Failed to upload remap maps to GPU: {e}")
        
        # GPU-first path
        if gpo.get_use_cuda() and "gpu_map_x" in state:
            try:
                gpu_img = gpo.ensure_gpu(img_array)
                gpu_result = gpo.remap_gpu(gpu_img, state["gpu_map_x"], state["gpu_map_y"],
                                           interpolation=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REFLECT)
                return gpu_result
            except Exception as e:
                logger.debug(f"GPU remap failed, falling back: {e}")
        
        # CPU fallback
        img_cpu = gpo.ensure_cpu(img_array)
        result = gpo.remap(img_cpu, state["map_x"], state["map_y"],
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        
        return result
        
    except Exception as e:
        logger.error(f"CUDA Nonlinear Remap error: {e}", exc_info=True)
        return img_array


def _singularity_border_mode(name: str) -> int:
    border_modes = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
        "constant": cv2.BORDER_CONSTANT,
    }
    return border_modes.get(str(name or "reflect").lower(), cv2.BORDER_REFLECT_101)


def apply_cuda_singularity_engine(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Apply the Phase 2/3 SingularityEngine single-vortex CUDA remap.

    This is intentionally image-space only: no accretion disk, event horizon,
    or temporal feedback. It keeps the same proof path as the Lua prototype:
    generate map_x/map_y, upload maps to CUDA, remap, optional CUDA blur blend.
    """
    try:
        h, w = _get_shape(img_array)
        if h <= 0 or w <= 0:
            return img_array

        state = get_effect_state("singularity_engine", (h, w))

        position_x = float(params.get("position_x", 0.5) or 0.5)
        position_y = float(params.get("position_y", 0.5) or 0.5)
        orbit_x = float(params.get("orbit_x", 0.06) or 0.0)
        orbit_y = float(params.get("orbit_y", 0.04) or 0.0)
        mass = float(params.get("mass", 1.0) or 0.0)
        radius = max(0.01, float(params.get("radius", 0.42) or 0.42))
        spin = float(params.get("spin", 3.5) or 0.0)
        softness = max(0.05, float(params.get("softness", 1.8) or 1.8))
        pull = max(0.0, float(params.get("pull", 0.08) or 0.0))
        animation_speed = float(params.get("animation_speed", 1.0) or 0.0)
        blur_sigma = max(0.0, float(params.get("blur_sigma", 2.0) or 0.0))
        blur_mix = float(np.clip(float(params.get("blur_mix", 0.35) or 0.0), 0.0, 1.0))
        mix = float(np.clip(float(params.get("mix", 1.0) or 1.0), 0.0, 1.0))
        border_mode = _singularity_border_mode(params.get("border_mode", "reflect"))

        t = float(time_val) * animation_speed
        cx_norm = np.clip(position_x + orbit_x * np.sin(t * 0.37), 0.0, 1.0)
        cy_norm = np.clip(position_y + orbit_y * np.cos(t * 0.29), 0.0, 1.0)
        phase = 0.35 * np.sin(t * 1.3)

        map_key = (
            round(cx_norm, 4),
            round(cy_norm, 4),
            round(mass, 4),
            round(radius, 4),
            round(spin, 4),
            round(softness, 4),
            round(pull, 4),
            round(phase, 4),
            border_mode,
        )

        if state.get("map_key") != map_key or state.get("map_size") != (h, w):
            y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
            cx = np.float32(cx_norm * (w - 1))
            cy = np.float32(cy_norm * (h - 1))
            dx = x_grid - cx
            dy = y_grid - cy
            dist = np.sqrt(dx * dx + dy * dy)

            radius_px = max(1.0, radius * min(w, h))
            soft_inner = radius_px * float(np.clip(1.0 - (softness * 0.25), 0.05, 0.95))
            denom = max(1e-6, radius_px - soft_inner)
            fade = np.clip((dist - soft_inner) / denom, 0.0, 1.0)
            smooth = fade * fade * (3.0 - 2.0 * fade)
            influence = np.power(np.clip(1.0 - smooth, 0.0, 1.0), softness).astype(np.float32)

            angle = -((spin + phase) * mass * influence)
            ca = np.cos(angle)
            sa = np.sin(angle)
            pull_scale = 1.0 - (pull * mass * influence)

            map_x = (cx + ((dx * ca - dy * sa) * pull_scale)).astype(np.float32)
            map_y = (cy + ((dx * sa + dy * ca) * pull_scale)).astype(np.float32)

            state["map_x"] = map_x
            state["map_y"] = map_y
            state["map_key"] = map_key
            state["map_size"] = (h, w)

            if gpo.get_use_cuda():
                try:
                    state["gpu_map_x"] = gpo.upload(map_x)
                    state["gpu_map_y"] = gpo.upload(map_y)
                except Exception as e:
                    logger.debug(f"Failed to upload SingularityEngine maps to GPU: {e}")
                    state.pop("gpu_map_x", None)
                    state.pop("gpu_map_y", None)

        if gpo.get_use_cuda() and "gpu_map_x" in state and "gpu_map_y" in state:
            try:
                gpu_img = gpo.ensure_gpu(img_array)
                warped = cv2.cuda.remap(
                    gpu_img,
                    state["gpu_map_x"],
                    state["gpu_map_y"],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=border_mode,
                )

                if blur_sigma > 0.0 and blur_mix > 0.0:
                    k = max(3, int(round(blur_sigma * 4.0)) | 1)
                    blur_key = (gpu_img.type(), k, round(blur_sigma, 3))
                    if state.get("blur_key") != blur_key or "blur_filter" not in state:
                        state["blur_filter"] = cv2.cuda.createGaussianFilter(gpu_img.type(), gpu_img.type(), (k, k), blur_sigma)
                        state["blur_key"] = blur_key
                    blurred = state["blur_filter"].apply(warped)
                    warped = cv2.cuda.addWeighted(warped, 1.0 - blur_mix, blurred, blur_mix, 0.0)

                if mix < 1.0:
                    warped = cv2.cuda.addWeighted(gpu_img, 1.0 - mix, warped, mix, 0.0)
                return warped
            except Exception as e:
                logger.debug(f"CUDA SingularityEngine GPU path failed, falling back: {e}")

        img_cpu = gpo.ensure_cpu(img_array)
        warped_cpu = cv2.remap(
            img_cpu,
            state["map_x"],
            state["map_y"],
            interpolation=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
        if blur_sigma > 0.0 and blur_mix > 0.0:
            k = max(3, int(round(blur_sigma * 4.0)) | 1)
            blurred_cpu = cv2.GaussianBlur(warped_cpu, (k, k), blur_sigma)
            warped_cpu = cv2.addWeighted(warped_cpu, 1.0 - blur_mix, blurred_cpu, blur_mix, 0.0)
        if mix < 1.0:
            warped_cpu = cv2.addWeighted(img_cpu, 1.0 - mix, warped_cpu, mix, 0.0)
        return warped_cpu

    except Exception as e:
        logger.error(f"CUDA SingularityEngine error: {e}", exc_info=True)
        return img_array


def apply_cuda_persistent_feedback(img_array: np.ndarray, params: dict) -> np.ndarray:
    """Apply CUDA persistent feedback with GPU acceleration.
    
    Uses GPU-first approach when CUDA is enabled:
    - Keeps accumulator on GPU as GpuMat
    - Only downloads final result
    """
    try:
        h, w = _get_shape(img_array)
        state = get_effect_state("persistent_feedback", (h, w))
        
        feedback_gain = float(params.get("feedback_gain", 0.92))
        inject_gain = float(params.get("inject_gain", 0.25))
        decay_curve = params.get("decay_curve", "exp")
        
        # Apply decay curve
        if decay_curve == "exp":
            feedback_gain = feedback_gain ** 2
        
        if gpo.get_use_cuda():
            try:
                # Upload input to GPU
                gpu_input = gpo.ensure_gpu(img_array)
                
                # Initialize GPU accumulator if needed
                if "gpu_accum" not in state or state.get("gpu_accum") is None:
                    state["gpu_accum"] = gpu_input.clone() if hasattr(gpu_input, 'clone') else gpo.upload(gpo.ensure_cpu(img_array).copy())
                    state["accum_size"] = (h, w)
                
                # Reset accumulator if size changed
                if state.get("accum_size") != (h, w):
                    state["gpu_accum"] = gpu_input.clone() if hasattr(gpu_input, 'clone') else gpo.upload(gpo.ensure_cpu(img_array).copy())
                    state["accum_size"] = (h, w)
                
                # GPU blending (stays on GPU)
                gpu_result = gpo.add_weighted_gpu(state["gpu_accum"], feedback_gain, 
                                                   gpu_input, inject_gain, 0.0)
                
                # Update GPU accumulator (no download needed!)
                state["gpu_accum"] = gpu_result
                
                # Return the GpuMat directly
                return gpu_result
            except Exception as e:
                logger.debug(f"GPU-first feedback failed, falling back: {e}")
                # Fall through to CPU path
        
        # CPU fallback path
        img_cpu = gpo.ensure_cpu(img_array)
        # Initialize accumulator with FIRST FRAME (not black zeros!)
        if "accum" not in state or state["accum"] is None:
            state["accum"] = img_cpu.copy()
        
        # Ensure accumulator matches input size
        if state["accum"].shape[:2] != (h, w):
            state["accum"] = img_cpu.copy()
        
        # CPU blending via gpu_ops (will use CUDA if available)
        result = gpo.addWeighted(state["accum"], feedback_gain, img_cpu, inject_gain, 0.0)
        
        # Update accumulator
        state["accum"] = result.copy()
        
        return result
        
    except Exception as e:
        logger.error(f"CUDA Persistent Feedback error: {e}", exc_info=True)
        return img_array


def apply_cuda_temporal_median(img_array: np.ndarray, params: dict) -> np.ndarray:
    """Apply CUDA temporal median."""
    try:
        h, w = img_array.shape[:2]
        state = get_effect_state("temporal_median", (h, w))
        
        window = int(params.get("window", 5))
        if window % 2 == 0:
            window += 1
        window = max(3, min(15, window))
        
        mode = params.get("mode", "approx")
        mix = params.get("mix", 0.8)
        if mix <= 0:
            return img_array
        
        # Initialize buffer
        if "buffer" not in state or getattr(state["buffer"], "maxlen", None) != window:
            state["buffer"] = deque(maxlen=window)
        
        state["buffer"].append(img_array.copy())
        
        if len(state["buffer"]) < window:
            return img_array
        
        # Compute median
        stack = np.stack(list(state["buffer"]), axis=0)
        
        if mode == "approx":
            min_frame = np.min(stack, axis=0)
            max_frame = np.max(stack, axis=0)
            mean_frame = np.mean(stack, axis=0).astype(np.uint8)
            median = ((min_frame.astype(np.float32) + max_frame.astype(np.float32)) / 2).astype(np.uint8)
            result = cv2.addWeighted(median, 0.6, mean_frame, 0.4, 0.0)
        else:
            mid = len(state["buffer"]) // 2
            result = np.partition(stack, mid, axis=0)[mid].astype(np.uint8)
        
        # Blend
        final = cv2.addWeighted(img_array, 1.0 - mix, result, mix, 0.0)
        return final
        
    except Exception as e:
        logger.error(f"CUDA Temporal Median error: {e}")
        return img_array


def apply_cuda_flow_pixel_sort(img_array: np.ndarray, params: dict, prev_frame: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply flow-guided pixel sorting (CPU implementation)."""
    try:
        h, w = img_array.shape[:2]
        state = get_effect_state("flow_pixel_sort", (h, w))
        
        sort_axis = params.get("sort_axis", "x")
        region_mode = params.get("region_mode", "motion")
        threshold = float(params.get("threshold", 30.0))
        intensity = float(params.get("intensity", 0.5))
        
        # If no previous frame, just return original
        if prev_frame is None:
            return img_array
        
        # Compute motion (simple frame difference)
        diff = cv2.absdiff(img_array, prev_frame)
        motion_mag = np.mean(diff, axis=2).astype(np.float32)
        
        # Create sorting mask based on region mode
        if region_mode == "motion":
            mask = motion_mag > threshold
        elif region_mode == "edges":
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            mask = edges > 0
        else:  # luma
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            mask = gray > threshold
        
        result = img_array.copy()
        
        # Sort pixels in masked regions
        if sort_axis == "x":
            for y in range(h):
                if np.any(mask[y]):
                    # Find continuous regions in this row
                    row_mask = mask[y]
                    changes = np.diff(np.concatenate([[0], row_mask.astype(int), [0]]))
                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]
                    
                    for start, end in zip(starts, ends):
                        if end - start > 1:
                            # Sort this segment by brightness
                            segment = result[y, start:end]
                            brightness = np.mean(segment, axis=1)
                            sorted_indices = np.argsort(brightness)
                            result[y, start:end] = segment[sorted_indices]
        else:  # y axis
            for x in range(w):
                if np.any(mask[:, x]):
                    # Find continuous regions in this column
                    col_mask = mask[:, x]
                    changes = np.diff(np.concatenate([[0], col_mask.astype(int), [0]]))
                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]
                    
                    for start, end in zip(starts, ends):
                        if end - start > 1:
                            # Sort this segment by brightness
                            segment = result[start:end, x]
                            brightness = np.mean(segment, axis=1)
                            sorted_indices = np.argsort(brightness)
                            result[start:end, x] = segment[sorted_indices]
        
        # Blend with original
        final = cv2.addWeighted(img_array, 1.0 - intensity, result, intensity, 0.0)
        return final
        
    except Exception as e:
        logger.error(f"CUDA Flow Pixel Sort error: {e}", exc_info=True)
        return img_array


def apply_cuda_bitplane_drift(img_array: np.ndarray, params: dict) -> np.ndarray:
    """Apply bitplane phase drift (CPU implementation)."""
    try:
        h, w = img_array.shape[:2]
        state = get_effect_state("bitplane_drift", (h, w))
        
        recombine_mix = float(params.get("recombine_mix", 0.7))
        max_history = int(params.get("max_history", 30))
        drift_amount = int(params.get("drift_amount", 3))
        
        # Initialize history buffer
        if "history" not in state:
            state["history"] = deque(maxlen=max_history)
        
        # Extract bitplanes
        bitplanes = []
        for bit in range(8):
            plane = ((img_array >> bit) & 1).astype(np.uint8) * 255
            bitplanes.append(plane)
        
        # Store current bitplanes in history
        state["history"].append(bitplanes.copy())
        
        if len(state["history"]) < 2:
            return img_array
        
        # Recombine with historical bitplanes (temporal drift)
        result_planes = []
        for bit in range(8):
            # Mix current bitplane with a historical one
            history_idx = min(bit * drift_amount, len(state["history"]) - 1)
            historical_plane = state["history"][-1 - history_idx][bit]
            current_plane = bitplanes[bit]
            
            # Blend
            mixed = cv2.addWeighted(current_plane, 1.0 - recombine_mix, 
                                   historical_plane, recombine_mix, 0.0)
            result_planes.append((mixed > 127).astype(np.uint8))
        
        # Reconstruct image from bitplanes
        result = np.zeros_like(img_array)
        for bit in range(8):
            result += result_planes[bit] * (1 << bit)
        
        return result
        
    except Exception as e:
        logger.error(f"CUDA Bitplane Drift error: {e}", exc_info=True)
        return img_array


def apply_cuda_audio_flow(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Apply audio-driven flow injection (CPU implementation with fake audio)."""
    try:
        h, w = img_array.shape[:2]
        state = get_effect_state("audio_flow", (h, w))
        
        bass_force = float(params.get("bass_force", 2.5))
        treble_ripple = float(params.get("treble_ripple", 1.0))
        flow_decay = float(params.get("flow_decay", 0.85))
        
        # Use frame-based time_val for deterministic animation
        # Use time_val directly for smooth sine wave motion
        time_phase = time_val
        
        # Simulate audio (in real implementation, this would come from audio input)
        # Use simple sine waves to simulate bass and treble
        bass = np.sin(time_phase * 2.0) * 0.5 + 0.5  # 0-1
        treble = np.sin(time_phase * 8.0) * 0.5 + 0.5  # 0-1
        
        # Generate flow field based on "audio"
        y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Bass creates radial flow from center
        cy, cx = h // 2, w // 2
        dx_bass = (x_grid - cx) * bass * bass_force * 0.01
        dy_bass = (y_grid - cy) * bass * bass_force * 0.01
        
        # Treble creates ripple patterns
        ripple = np.sin(x_grid * 0.05 + time_phase * 5) * np.cos(y_grid * 0.05 + time_phase * 3)
        dx_treble = ripple * treble * treble_ripple * 5
        dy_treble = -ripple * treble * treble_ripple * 5
        
        # Combine
        dx_total = dx_bass + dx_treble
        dy_total = dy_bass + dy_treble
        
        # Initialize accumulated flow
        if "flow_accum_x" not in state:
            state["flow_accum_x"] = np.zeros((h, w), dtype=np.float32)
            state["flow_accum_y"] = np.zeros((h, w), dtype=np.float32)
        
        # Accumulate with decay
        state["flow_accum_x"] = state["flow_accum_x"] * flow_decay + dx_total
        state["flow_accum_y"] = state["flow_accum_y"] * flow_decay + dy_total
        
        # Create displacement maps
        map_x = (x_grid + state["flow_accum_x"]).astype(np.float32)
        map_y = (y_grid + state["flow_accum_y"]).astype(np.float32)
        
        # Apply remap
        result = gpo.remap(img_array, map_x, map_y, 
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
        
        return result
        
    except Exception as e:
        logger.error(f"CUDA Audio Flow error: {e}", exc_info=True)
        return img_array


def apply_cuda_edge_slicer(img_array, params, time_val):
    if not gpo.get_use_cuda(): return img_array

    edge_thresh = float(params.get("edge_thresh", 0.2))
    shift_x = float(params.get("shift_x", 50.0))
    shift_y = float(params.get("shift_y", 0.0))
    pre_blur = int(params.get("pre_blur", 3))
    if pre_blur % 2 == 0: pre_blur += 1
    
    h, w = _get_shape(img_array)
    state = get_effect_state("edge_slicer", (h, w))
    if "base_map_x" not in state:
        mx, my = np.meshgrid(np.arange(w), np.arange(h))
        state["base_map_x"] = gpo.upload(mx.astype(np.float32))
        state["base_map_y"] = gpo.upload(my.astype(np.float32))
        
    gpu_img = gpo.ensure_gpu(img_array)
    
    if pre_blur > 1:
        if "blur_filter" not in state or state.get("blur_size") != pre_blur:
            state["blur_filter"] = cv2.cuda.createGaussianFilter(gpu_img.type(), gpu_img.type(), (pre_blur, pre_blur), 0)
            state["blur_size"] = pre_blur
        gpu_blurred = state["blur_filter"].apply(gpu_img)
    else:
        gpu_blurred = gpu_img
        
    gpu_gray = cv2.cuda.cvtColor(gpu_blurred, cv2.COLOR_BGR2GRAY)
    
    if "sobel_x" not in state:
        state["sobel_x"] = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 1, 0, 3)
        state["sobel_y"] = cv2.cuda.createSobelFilter(cv2.CV_8UC1, cv2.CV_32F, 0, 1, 3)
        
    grad_x = state["sobel_x"].apply(gpu_gray)
    grad_y = state["sobel_y"].apply(gpu_gray)
    
    grad_x_sq = cv2.cuda.multiply(grad_x, grad_x)
    grad_y_sq = cv2.cuda.multiply(grad_y, grad_y)
    grad_mag_sq = cv2.cuda.add(grad_x_sq, grad_y_sq)
    
    thresh_val = (edge_thresh * 1000.0) ** 2
    _, gpu_mask = cv2.cuda.threshold(grad_mag_sq, thresh_val, 1.0, cv2.THRESH_BINARY)
    
    if "mask_blur" not in state:
        state["mask_blur"] = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (15, 15), 0)
    gpu_mask = state["mask_blur"].apply(gpu_mask)
    
    # Use addWeighted instead of multiply for safe scalar math
    gpu_map_x = cv2.cuda.addWeighted(state["base_map_x"], 1.0, gpu_mask, shift_x, 0.0)
    gpu_map_y = cv2.cuda.addWeighted(state["base_map_y"], 1.0, gpu_mask, shift_y, 0.0)
    
    gpu_out = cv2.cuda.remap(gpu_img, gpu_map_x, gpu_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return gpu_out

def apply_cuda_channel_drift(img_array, params, time_val):
    if not gpo.get_use_cuda(): return img_array

    r_shift_x = float(params.get("r_shift_x", -10.0))
    r_shift_y = float(params.get("r_shift_y", -5.0))
    g_shift_x = float(params.get("g_shift_x", 0.0))
    g_shift_y = float(params.get("g_shift_y", 0.0))
    b_shift_x = float(params.get("b_shift_x", 10.0))
    b_shift_y = float(params.get("b_shift_y", 5.0))
    scale_var = float(params.get("scale_var", 0.02))
    rot_var = float(params.get("rot_var", 0.5))
    drift_speed = float(params.get("drift_speed", 1.0))
    
    h, w = _get_shape(img_array)
    cx, cy = w / 2, h / 2
    gpu_img = gpo.ensure_gpu(img_array)
    
    def get_warp_mat(dx, dy, phase):
        # Increased multiplier for more visible effect
        angle = math.sin(t + phase) * rot_var * 5.0
        scale = 1.0 + math.cos(t * 1.3 + phase) * scale_var * 2.0
        sx = dx + math.sin(t * 0.7 + phase) * dx * 0.5
        sy = dy + math.cos(t * 0.5 + phase) * dy * 0.5
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        M[0, 2] += sx
        M[1, 2] += sy
        return M.astype(np.float32)
        
    import math
    t = time_val * drift_speed
    
    M_b = get_warp_mat(b_shift_x, b_shift_y, 0.0)
    M_g = get_warp_mat(g_shift_x, g_shift_y, 2.0)
    M_r = get_warp_mat(r_shift_x, r_shift_y, 4.0)
    
    channels = cv2.cuda.split(gpu_img)
    a_ch = None
    if len(channels) == 4:
        b_ch, g_ch, r_ch, a_ch = channels
    elif len(channels) == 3:
        b_ch, g_ch, r_ch = channels
    else:
        return gpu_img
    
    b_warp = cv2.cuda.warpAffine(b_ch, M_b, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    g_warp = cv2.cuda.warpAffine(g_ch, M_g, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    r_warp = cv2.cuda.warpAffine(r_ch, M_r, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    if a_ch is not None:
        gpu_out = cv2.cuda.merge([b_warp, g_warp, r_warp, a_ch])
    else:
        gpu_out = cv2.cuda.merge([b_warp, g_warp, r_warp])
        
    return gpu_out

def apply_cuda_digital_melt(img_array, params, time_val):
    if not gpo.get_use_cuda(): return img_array
    
    threshold = float(params.get("threshold", 0.6))
    melt_len = int(params.get("melt_length", 20))
    direction = str(params.get("direction", "down"))
    melt_mix = float(params.get("melt_mix", 0.8))
    
    h, w = _get_shape(img_array)
    state = get_effect_state("digital_melt", (h, w))
    gpu_img = gpo.ensure_gpu(img_array)
    
    gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
    _, gpu_mask = cv2.cuda.threshold(gpu_gray, int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    mask_cpu = gpu_mask.download()
    img_cpu = gpo.ensure_cpu(img_array)
    isolated = cv2.bitwise_and(img_cpu, img_cpu, mask=mask_cpu)
    gpu_isolated = gpo.upload(isolated)
    
    # morphology filter requires 1C or 4C, so convert BGR to BGRA
    gpu_isolated_bgra = cv2.cuda.cvtColor(gpu_isolated, cv2.COLOR_BGR2BGRA)
    
    if "morph" not in state or state.get("melt_len") != melt_len or state.get("dir") != direction:
        if direction == "down":
            kernel = np.ones((melt_len, 1), np.uint8)
            anchor = (0, 0)
        elif direction == "up":
            kernel = np.ones((melt_len, 1), np.uint8)
            anchor = (0, melt_len - 1)
        elif direction == "right":
            kernel = np.ones((1, melt_len), np.uint8)
            anchor = (0, 0)
        elif direction == "left":
            kernel = np.ones((1, melt_len), np.uint8)
            anchor = (melt_len - 1, 0)
        else:
            kernel = np.ones((melt_len, 1), np.uint8)
            anchor = (0, 0)
            
        state["morph"] = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, gpu_isolated_bgra.type(), kernel, anchor=anchor)
        state["melt_len"] = melt_len
        state["dir"] = direction
        
    gpu_melted_bgra = state["morph"].apply(gpu_isolated_bgra)
    gpu_melted = cv2.cuda.cvtColor(gpu_melted_bgra, cv2.COLOR_BGRA2BGR)
    gpu_out = cv2.cuda.addWeighted(gpu_img, 1.0, gpu_melted, melt_mix, 0.0)
    
    return gpu_out

def apply_cuda_vector_mosh(img_array: np.ndarray, params: dict, prev_frame: Optional[np.ndarray], time_val: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Authentic H.264 datamoshing using CUDA motion vectors (Optimized)."""
    try:
        if not gpo.get_use_cuda(): 
            return img_array, img_array
            
        flow_sens = float(params.get("flow_sensitivity", 1.0))
        i_frame_drop = float(params.get("i_frame_drop", 0.5))
        vector_decay = float(params.get("vector_decay", 0.9))
        x_bias = float(params.get("x_bias", 0.0))
        y_bias = float(params.get("y_bias", 0.0))
        
        h, w = _get_shape(img_array)
        state = get_effect_state("vector_mosh", (h, w))
        
        gpu_img = gpo.ensure_gpu(img_array)
        
        if prev_frame is None:
            state["last_out"] = gpu_img.clone() if hasattr(gpu_img, 'clone') else gpo.upload(gpo.ensure_cpu(img_array).copy())
            return gpu_img, img_array
            
        gpu_prev = gpo.ensure_gpu(prev_frame)
        
        if "flow" not in state:
            state["flow"] = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
            
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)
        gpu_prev_gray = cv2.cuda.cvtColor(gpu_prev, cv2.COLOR_BGR2GRAY)
        
        gpu_flow = state["flow"].calc(gpu_prev_gray, gpu_gray, None)
        
        if "accum_flow" not in state:
            state["accum_flow"] = gpu_flow
        else:
            state["accum_flow"] = cv2.cuda.addWeighted(state["accum_flow"], vector_decay, gpu_flow, flow_sens, 0.0)
            
        import random
        if random.random() < i_frame_drop and "last_out" in state:
            # GPU-based remap generation for 0 CPU copies!
            if "base_map_x" not in state or state.get("map_size") != (h, w):
                mx, my = np.meshgrid(np.arange(w), np.arange(h))
                state["base_map_x"] = gpo.upload(mx.astype(np.float32))
                state["base_map_y"] = gpo.upload(my.astype(np.float32))
                state["map_size"] = (h, w)
                
            channels = cv2.cuda.split(state["accum_flow"])
            flow_x = channels[0]
            flow_y = channels[1]
            
            gpu_map_x = cv2.cuda.addWeighted(state["base_map_x"], 1.0, flow_x, -1.0, x_bias)
            gpu_map_y = cv2.cuda.addWeighted(state["base_map_y"], 1.0, flow_y, -1.0, y_bias)
            
            gpu_out = cv2.cuda.remap(state["last_out"], gpu_map_x, gpu_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            state["last_out"] = gpu_out.clone()
            return gpu_out, img_array
        else:
            state["last_out"] = gpu_img.clone()
            state["accum_flow"] = cv2.cuda.multiply(state["accum_flow"], 0.5)
            return gpu_img, img_array
    except Exception as e:
        logger.error(f"CUDA Vector Mosh error: {e}", exc_info=True)
        return img_array, img_array


def apply_cuda_temporal_slit_scan(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Temporal Slit-Scan mapping depth to history buffer (Optimized Vectorized Array)."""
    try:
        buffer_size = int(params.get("buffer_size", 30))
        buffer_size = max(2, min(90, buffer_size))
        orientation = str(params.get("orientation", "y_axis"))
        mix = float(np.clip(params.get("mix", 1.0), 0.0, 1.0))
        direction = str(params.get("direction", "backward"))
        time_offset = float(params.get("time_offset", 0.0))
        time_scale = float(params.get("time_scale", 1.0))
        depth_curve = str(params.get("depth_curve", "linear"))
        luma_scale = float(params.get("luma_scale", 1.0))
        luma_gamma = max(0.01, float(params.get("luma_gamma", 1.0)))
        luma_invert = bool(params.get("luma_invert", False))
        luma_blur = max(0, int(params.get("luma_blur", 0)))
        wave_amount = float(params.get("wave_amount", 0.0))
        wave_frequency = max(0.01, float(params.get("wave_frequency", 4.0)))
        jitter_amount = float(params.get("jitter_amount", 0.0))
        jitter_scale = max(1.0, float(params.get("jitter_scale", 16.0)))
        posterize_steps = max(0, int(params.get("posterize_steps", 0)))
        freeze_history = bool(params.get("freeze_history", False))
        
        h, w = _get_shape(img_array)
        state = get_effect_state("temporal_slit_scan", (h, w))
        
        img_cpu = gpo.ensure_cpu(img_array)
        frame_shape = img_cpu.shape
        frame_dtype = img_cpu.dtype
        
        # Pre-allocate numpy array to avoid building it every frame
        if (
            "history" not in state or
            state.get("buffer_size") != buffer_size or
            state.get("map_size") != (h, w) or
            state.get("frame_shape") != frame_shape or
            state.get("frame_dtype") != frame_dtype
        ):
            state["history"] = np.zeros((buffer_size, *frame_shape), dtype=frame_dtype)
            state["buffer_size"] = buffer_size
            state["map_size"] = (h, w)
            state["frame_shape"] = frame_shape
            state["frame_dtype"] = frame_dtype
            state["idx"] = 0
            state["started"] = False
            state["luma_xmap"] = None
            state["luma_ybase"] = None
            
        current_idx = state["idx"]
        if not freeze_history or not state["started"]:
            state["history"][current_idx] = img_cpu
            state["idx"] = (current_idx + 1) % buffer_size
        else:
            current_idx = (current_idx - 1) % buffer_size
        
        if current_idx < buffer_size - 1 and not state["started"]:
            return img_cpu
        state["started"] = True

        no_offset_modulation = (
            abs(wave_amount) <= 1e-9 and
            jitter_amount <= 1e-9
        )
        fast_axis_path = (
            orientation in {"y_axis", "x_axis"} and
            depth_curve == "linear" and
            no_offset_modulation and
            posterize_steps <= 1
        )
        fast_luma_path = (
            orientation == "luma" and
            frame_dtype == np.uint8 and
            no_offset_modulation
        )

        if fast_axis_path:
            history_span = float(buffer_size - 1)
            if orientation == "y_axis":
                depth_1d = np.arange(h, dtype=np.float32) / max(1.0, float(h))
            else:
                depth_1d = np.arange(w, dtype=np.float32) / max(1.0, float(w))
            offsets = depth_1d * history_span * time_scale + time_offset
            if direction == "forward":
                hist_indices = (current_idx + offsets.astype(np.int32)) % buffer_size
            else:
                hist_indices = (current_idx - offsets.astype(np.int32)) % buffer_size

            if orientation == "y_axis":
                row_indices = np.arange(h)
                if len(frame_shape) == 2:
                    out_cpu = state["history"][hist_indices, row_indices, :]
                else:
                    out_cpu = state["history"][hist_indices, row_indices, :, :]
            else:
                col_indices = np.arange(w)
                if len(frame_shape) == 2:
                    out_cpu = state["history"][hist_indices, :, col_indices].T
                else:
                    out_cpu = state["history"][hist_indices, :, col_indices, :].transpose(1, 0, 2)
        elif fast_luma_path:
            if img_cpu.ndim == 3 and img_cpu.shape[2] == 4:
                gray = cv2.cvtColor(img_cpu, cv2.COLOR_BGRA2GRAY)
            elif img_cpu.ndim == 3:
                gray = cv2.cvtColor(img_cpu, cv2.COLOR_BGR2GRAY)
            else:
                gray = img_cpu
            if luma_blur > 0:
                kernel = luma_blur + 1 if luma_blur % 2 == 0 else luma_blur
                gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

            depth_lut = np.arange(256, dtype=np.float32) / 255.0
            if luma_invert:
                depth_lut = 1.0 - depth_lut
            depth_lut = np.power(np.clip(depth_lut, 0.0, 1.0), luma_gamma) * luma_scale

            if depth_curve == "ease_in":
                depth_lut = depth_lut * depth_lut
            elif depth_curve == "ease_out":
                depth_lut = 1.0 - (1.0 - depth_lut) * (1.0 - depth_lut)
            elif depth_curve == "sine":
                depth_lut = 0.5 - 0.5 * np.cos(np.clip(depth_lut, 0.0, 1.0) * np.pi)
            elif depth_curve == "center_peak":
                depth_lut = 1.0 - np.abs((np.clip(depth_lut, 0.0, 1.0) * 2.0) - 1.0)

            history_span = float(buffer_size - 1)
            offset_lut = depth_lut * history_span * time_scale + time_offset
            if posterize_steps > 1:
                step = history_span / float(posterize_steps - 1)
                if step > 0:
                    offset_lut = np.round(offset_lut / step) * step

            if direction == "forward":
                hist_lut = (current_idx + offset_lut.astype(np.int32)) % buffer_size
            else:
                hist_lut = (current_idx - offset_lut.astype(np.int32)) % buffer_size
            hist_lut = hist_lut.astype(np.uint8, copy=False)

            hist_map = cv2.LUT(gray, hist_lut.reshape(256, 1))
            if (
                state.get("luma_xmap") is None or
                state.get("luma_ybase") is None or
                state.get("luma_map_size") != (h, w)
            ):
                state["luma_xmap"] = np.broadcast_to(
                    np.arange(w, dtype=np.float32).reshape(1, w),
                    (h, w),
                ).copy()
                state["luma_ybase"] = np.broadcast_to(
                    np.arange(h, dtype=np.float32).reshape(h, 1),
                    (h, w),
                ).copy()
                state["luma_map_size"] = (h, w)

            # OpenCV remap is much faster than per-history masks, but source
            # atlas dimensions must stay below SHRT_MAX. Chunk the temporal
            # axis and composite only the chunks referenced by the luma map.
            xmap = state["luma_xmap"]
            ybase = state["luma_ybase"]
            max_chunk = max(1, min(32, 32000 // max(1, h)))
            chunk_size = min(buffer_size, max_chunk)
            out_cpu = None

            for start in range(0, buffer_size, chunk_size):
                end = min(buffer_size, start + chunk_size)
                atlas = state["history"][start:end].reshape((end - start) * h, w, *frame_shape[2:])
                if start == 0:
                    local_idx = np.minimum(hist_map, end - 1).astype(np.float32)
                    ymap = ybase + local_idx * float(h)
                    out_cpu = cv2.remap(
                        atlas,
                        xmap,
                        ymap,
                        interpolation=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    continue

                mask = cv2.inRange(hist_map, start, end - 1)
                if cv2.countNonZero(mask) <= 0:
                    continue
                local_idx = (hist_map - start).clip(0, end - start - 1).astype(np.float32)
                ymap = ybase + local_idx * float(h)
                chunk_out = cv2.remap(
                    atlas,
                    xmap,
                    ymap,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                cv2.copyTo(chunk_out, mask, out_cpu)

            if out_cpu is None:
                out_cpu = img_cpu
        else:
            Y, X = np.ogrid[:h, :w]

            if orientation == "y_axis":
                depth = np.arange(h, dtype=np.float32).reshape(h, 1) / max(1.0, float(h))
                depth = np.broadcast_to(depth, (h, w)).copy()
            elif orientation == "x_axis":
                depth = np.arange(w, dtype=np.float32).reshape(1, w) / max(1.0, float(w))
                depth = np.broadcast_to(depth, (h, w)).copy()
            else:
                if img_cpu.ndim == 3 and img_cpu.shape[2] == 4:
                    gray = cv2.cvtColor(img_cpu, cv2.COLOR_BGRA2GRAY)
                elif img_cpu.ndim == 3:
                    gray = cv2.cvtColor(img_cpu, cv2.COLOR_BGR2GRAY)
                else:
                    gray = img_cpu
                if luma_blur > 0:
                    kernel = luma_blur + 1 if luma_blur % 2 == 0 else luma_blur
                    gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
                depth = gray.astype(np.float32) / 255.0
                if luma_invert:
                    depth = 1.0 - depth
                depth = np.power(np.clip(depth, 0.0, 1.0), luma_gamma) * luma_scale

            if depth_curve == "ease_in":
                depth = depth * depth
            elif depth_curve == "ease_out":
                depth = 1.0 - (1.0 - depth) * (1.0 - depth)
            elif depth_curve == "sine":
                depth = 0.5 - 0.5 * np.cos(np.clip(depth, 0.0, 1.0) * np.pi)
            elif depth_curve == "center_peak":
                depth = 1.0 - np.abs((np.clip(depth, 0.0, 1.0) * 2.0) - 1.0)

            history_span = float(buffer_size - 1)
            offsets = depth * history_span * time_scale + time_offset

            if abs(wave_amount) > 0.0:
                yy = np.arange(h, dtype=np.float32).reshape(h, 1) / max(1.0, float(h))
                xx = np.arange(w, dtype=np.float32).reshape(1, w) / max(1.0, float(w))
                wave = np.sin((xx + yy) * wave_frequency * np.pi * 2.0 + float(time_val))
                offsets += wave * wave_amount * history_span

            if jitter_amount > 0.0:
                yy = np.arange(h, dtype=np.float32).reshape(h, 1)
                xx = np.arange(w, dtype=np.float32).reshape(1, w)
                jitter = np.sin((xx * 12.9898 + yy * 78.233 + float(time_val) * 37.719) / jitter_scale)
                offsets += jitter * jitter_amount * history_span

            if posterize_steps > 1:
                step = history_span / float(posterize_steps - 1)
                if step > 0:
                    offsets = np.round(offsets / step) * step

            if direction == "forward":
                hist_indices = (current_idx + offsets.astype(np.int32)) % buffer_size
            else:
                hist_indices = (current_idx - offsets.astype(np.int32)) % buffer_size

            out_cpu = state["history"][hist_indices, Y, X]
        
        if mix < 1.0:
            out_cpu = np.clip(
                img_cpu.astype(np.float32) * (1.0 - mix) + out_cpu.astype(np.float32) * mix,
                0,
                255,
            ).astype(frame_dtype)
            
        return out_cpu
    except Exception as e:
        logger.error(f"CUDA Temporal Slit Scan error: {e}", exc_info=True)
        return img_array


def apply_cuda_recursive_fractal_corrupter(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Warp UV coordinates with an iterated Julia-style fractal field, then remap on CUDA."""
    try:
        if not gpo.get_use_cuda():
            return img_array

        gpu_img = gpo.ensure_gpu(img_array)
        if hasattr(cv2, 'cuda_GpuMat') and isinstance(gpu_img, cv2.cuda_GpuMat):
            gpu_w, gpu_h = gpu_img.size()
            h, w = int(gpu_h), int(gpu_w)
        else:
            h, w = _get_shape(img_array)
        if h <= 0 or w <= 0:
            return img_array

        iterations = max(1, min(64, int(params.get("iterations", 18))))
        zoom = max(0.05, float(params.get("zoom", 1.4)))
        seed_x = float(params.get("seed_x", -0.72))
        seed_y = float(params.get("seed_y", 0.27))
        warp_intensity = float(params.get("warp_intensity", 34.0))
        evolve_rate = float(params.get("evolve_rate", 0.35))
        mix = float(np.clip(params.get("mix", 1.0), 0.0, 1.0))

        if warp_intensity <= 0.0 or mix <= 0.0:
            return img_array

        state = get_effect_state("recursive_fractal_corrupter", (h, w))
        if state.get("map_size") != (h, w):
            x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
            y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
            norm_x, norm_y = np.meshgrid(x, y)
            pix_x, pix_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            aspect = np.float32(w / max(1.0, float(h)))
            state["norm_x"] = norm_x * aspect
            state["norm_y"] = norm_y
            state["pix_x"] = pix_x
            state["pix_y"] = pix_y
            state["map_size"] = (h, w)

        phase = float(time_val) * evolve_rate
        c_re = np.float32(seed_x + 0.09 * np.sin(phase * 0.71))
        c_im = np.float32(seed_y + 0.09 * np.cos(phase * 0.63))

        zx = (state["norm_x"] / zoom).astype(np.float32, copy=True)
        zy = (state["norm_y"] / zoom).astype(np.float32, copy=True)
        escape = np.zeros((h, w), dtype=np.float32)
        active = np.ones((h, w), dtype=bool)

        for idx in range(iterations):
            zx2 = zx * zx
            zy2 = zy * zy
            mag2 = zx2 + zy2
            newly_escaped = active & (mag2 > 4.0)
            escape[newly_escaped] = float(idx) / float(iterations)
            active &= mag2 <= 4.0
            if not np.any(active):
                break
            zxy = zx * zy
            next_zx = zx2 - zy2 + c_re
            next_zy = (zxy + zxy) + c_im
            zx[active] = next_zx[active]
            zy[active] = next_zy[active]

        escape[active] = 1.0
        angle = np.arctan2(zy, zx).astype(np.float32)
        ripple = np.sin((escape * 18.0) + angle + phase).astype(np.float32)
        escape_weight = 0.35 + 0.65 * np.sqrt(np.clip(escape, 0.0, 1.0))
        displacement = ripple * escape_weight * warp_intensity
        map_x = state["pix_x"] + np.cos(angle) * displacement
        map_y = state["pix_y"] + np.sin(angle) * displacement

        gpu_map_x = gpo.upload(map_x.astype(np.float32, copy=False))
        gpu_map_y = gpo.upload(map_y.astype(np.float32, copy=False))
        gpu_out = cv2.cuda.remap(
            gpu_img,
            gpu_map_x,
            gpu_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        if mix >= 1.0:
            return gpu_out
        return cv2.cuda.addWeighted(gpu_img, 1.0 - mix, gpu_out, mix, 0.0)
    except Exception as e:
        logger.error(f"CUDA Recursive Fractal Corrupter error: {e}", exc_info=True)
        return img_array


def _voronoi_feature_points(img_cpu: np.ndarray, count: int, out_w: int, out_h: int) -> Optional[np.ndarray]:
    try:
        gray = cv2.cvtColor(img_cpu, cv2.COLOR_BGR2GRAY) if img_cpu.ndim == 3 else img_cpu
        gray_small = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA)
        points = cv2.goodFeaturesToTrack(
            gray_small,
            maxCorners=count,
            qualityLevel=0.01,
            minDistance=max(2, min(out_w, out_h) // 18),
        )
        if points is None or len(points) < 2:
            return None
        return points.reshape(-1, 2).astype(np.float32)
    except Exception:
        return None


def apply_cuda_voronoi_shatter(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Fracture a frame into animated Voronoi shards and execute the final remap on CUDA."""
    try:
        if not gpo.get_use_cuda():
            return img_array

        h, w = _get_shape(img_array)
        if h <= 0 or w <= 0:
            return img_array

        shard_count = max(2, min(128, int(params.get("shard_count", 32))))
        drift_speed = float(params.get("drift_speed", 0.8))
        rotation_variance = float(params.get("rotation_variance", 8.0))
        translation = float(params.get("translation", 28.0))
        feature_tracking = bool(params.get("feature_tracking", False))
        map_resolution = max(64, min(512, int(params.get("map_resolution", 256))))
        mix = float(np.clip(params.get("mix", 1.0), 0.0, 1.0))

        if mix <= 0.0 or (translation <= 0.0 and rotation_variance <= 0.0):
            return img_array

        map_w = int(min(map_resolution, max(16, w)))
        map_h = int(max(16, round(map_w * h / max(1, w))))
        state = get_effect_state("voronoi_shatter", (h, w))
        layout_key = (map_h, map_w, shard_count, feature_tracking)

        img_cpu = None
        if feature_tracking:
            img_cpu = gpo.ensure_cpu(img_array)
            seeds = _voronoi_feature_points(img_cpu, shard_count, map_w, map_h)
        else:
            seeds = None

        if seeds is None and state.get("layout_key") == layout_key and "seeds" in state:
            seeds = state["seeds"]

        if seeds is None:
            rng = np.random.default_rng(1337 + shard_count * 17 + map_w * 3 + map_h)
            seeds = np.column_stack([
                rng.uniform(0, map_w - 1, shard_count),
                rng.uniform(0, map_h - 1, shard_count),
            ]).astype(np.float32)

        if len(seeds) < shard_count:
            rng = np.random.default_rng(7331 + shard_count)
            extra = np.column_stack([
                rng.uniform(0, map_w - 1, shard_count - len(seeds)),
                rng.uniform(0, map_h - 1, shard_count - len(seeds)),
            ]).astype(np.float32)
            seeds = np.vstack([seeds, extra])
        seeds = seeds[:shard_count]

        if state.get("layout_key") != layout_key or "label_map" not in state or feature_tracking:
            yy, xx = np.mgrid[0:map_h, 0:map_w].astype(np.float32)
            best_dist = np.full((map_h, map_w), np.inf, dtype=np.float32)
            label_map = np.zeros((map_h, map_w), dtype=np.int32)
            for idx, (sx, sy) in enumerate(seeds):
                dist = (xx - sx) * (xx - sx) + (yy - sy) * (yy - sy)
                mask = dist < best_dist
                label_map[mask] = idx
                best_dist[mask] = dist[mask]
            rng = np.random.default_rng(2024 + shard_count)
            state["phase"] = rng.uniform(0.0, np.pi * 2.0, shard_count).astype(np.float32)
            state["dir_x"] = rng.uniform(-1.0, 1.0, shard_count).astype(np.float32)
            state["dir_y"] = rng.uniform(-1.0, 1.0, shard_count).astype(np.float32)
            state["rot_sign"] = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), shard_count)
            state["label_map"] = label_map
            state["seeds"] = seeds
            state["layout_key"] = layout_key

        label_map = state["label_map"]
        phase = state["phase"]
        t = float(time_val) * drift_speed
        yy, xx = np.mgrid[0:map_h, 0:map_w].astype(np.float32)
        map_x = xx * (w / max(1.0, float(map_w)))
        map_y = yy * (h / max(1.0, float(map_h)))

        scale_x = w / max(1.0, float(map_w))
        scale_y = h / max(1.0, float(map_h))
        for idx in range(shard_count):
            mask = label_map == idx
            if not np.any(mask):
                continue
            sx = float(state["seeds"][idx, 0]) * scale_x
            sy = float(state["seeds"][idx, 1]) * scale_y
            theta = np.deg2rad(rotation_variance * state["rot_sign"][idx] * np.sin(t + phase[idx]))
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            px = map_x[mask] - sx
            py = map_y[mask] - sy
            dx = translation * state["dir_x"][idx] * np.sin(t * 0.9 + phase[idx])
            dy = translation * state["dir_y"][idx] * np.cos(t * 1.1 + phase[idx])
            map_x[mask] = sx + px * cos_t - py * sin_t - dx
            map_y[mask] = sy + px * sin_t + py * cos_t - dy

        if (map_h, map_w) != (h, w):
            map_x = cv2.resize(map_x, (w, h), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(map_y, (w, h), interpolation=cv2.INTER_LINEAR)

        gpu_img = gpo.ensure_gpu(img_array)
        gpu_map_x = gpo.upload(map_x.astype(np.float32, copy=False))
        gpu_map_y = gpo.upload(map_y.astype(np.float32, copy=False))
        gpu_out = cv2.cuda.remap(
            gpu_img,
            gpu_map_x,
            gpu_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        if mix >= 1.0:
            return gpu_out
        return cv2.cuda.addWeighted(gpu_img, 1.0 - mix, gpu_out, mix, 0.0)
    except Exception as e:
        logger.error(f"CUDA Voronoi Shatter error: {e}", exc_info=True)
        return img_array


def _phase3_gray(img_cpu: np.ndarray) -> np.ndarray:
    if img_cpu.ndim == 2:
        return img_cpu
    if img_cpu.shape[2] == 4:
        return cv2.cvtColor(img_cpu, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img_cpu[:, :, :3], cv2.COLOR_BGR2GRAY)


def _phase3_rgb(img_cpu: np.ndarray) -> np.ndarray:
    if img_cpu.ndim == 2:
        return cv2.cvtColor(img_cpu, cv2.COLOR_GRAY2BGR)
    return img_cpu[:, :, :3]


def _phase3_match_channels(effect_rgb: np.ndarray, img_cpu: np.ndarray) -> np.ndarray:
    if img_cpu.ndim == 2:
        return cv2.cvtColor(effect_rgb, cv2.COLOR_BGR2GRAY)
    if img_cpu.shape[2] == 4:
        return np.dstack([effect_rgb, img_cpu[:, :, 3]])
    return effect_rgb


def _phase3_sim_size(width: int, height: int, target: int) -> Tuple[int, int]:
    sim_w = int(min(max(16, width), max(16, target)))
    sim_h = int(max(16, round(sim_w * height / max(1, width))))
    return sim_w, sim_h


def apply_cuda_reaction_diffusion(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Gray-Scott reaction-diffusion that drives a CUDA remap and growth-color blend."""
    try:
        if not gpo.get_use_cuda():
            return img_array

        h, w = _get_shape(img_array)
        if h <= 0 or w <= 0:
            return img_array

        feed_rate = float(params.get("feed_rate", 0.036))
        kill_rate = float(params.get("kill_rate", 0.060))
        diffusion_a = float(params.get("diffusion_a", 1.0))
        diffusion_b = float(params.get("diffusion_b", 0.5))
        iterations = max(1, min(20, int(params.get("iterations_per_frame", 4))))
        video_influence = float(np.clip(params.get("video_influence", 0.35), 0.0, 1.0))
        seed_amount = float(np.clip(params.get("seed_amount", 0.08), 0.0, 1.0))
        sim_resolution = max(32, min(512, int(params.get("sim_resolution", 192))))
        warp_strength = float(params.get("warp_strength", 18.0))
        growth_contrast = max(0.01, float(params.get("growth_contrast", 1.6)))
        color_mode = str(params.get("color_mode", "biofilm"))
        reset_requested = bool(params.get("reset_sim", False))
        mix = float(np.clip(params.get("mix", 0.85), 0.0, 1.0))

        if mix <= 0.0:
            return img_array

        img_cpu = gpo.ensure_cpu(img_array)
        gray = _phase3_gray(img_cpu)
        sim_w, sim_h = _phase3_sim_size(w, h, sim_resolution)
        state = get_effect_state("reaction_diffusion", (h, w))
        sim_key = (sim_h, sim_w)

        should_reset = (
            state.get("sim_key") != sim_key or
            "chem_a" not in state or
            (reset_requested and not state.get("reset_latched", False))
        )
        state["reset_latched"] = reset_requested
        if should_reset:
            rng = np.random.default_rng(4242 + sim_w * 3 + sim_h)
            a = np.ones((sim_h, sim_w), dtype=np.float32)
            b = np.zeros((sim_h, sim_w), dtype=np.float32)
            patch = max(4, min(sim_w, sim_h) // 9)
            cy, cx = sim_h // 2, sim_w // 2
            b[max(0, cy - patch):min(sim_h, cy + patch), max(0, cx - patch):min(sim_w, cx + patch)] = 1.0
            noise = rng.random((sim_h, sim_w), dtype=np.float32)
            b = np.maximum(b, (noise > (1.0 - max(0.01, seed_amount) * 0.08)).astype(np.float32))
            state["chem_a"] = a
            state["chem_b"] = b
            state["sim_key"] = sim_key

        a = state["chem_a"]
        b = state["chem_b"]
        gray_small = cv2.resize(gray, (sim_w, sim_h), interpolation=cv2.INTER_AREA)
        sobel_x = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(sobel_x, sobel_y)
        edge = edge / (float(edge.max()) + 1e-6)

        if seed_amount > 0.0:
            b = np.clip(b + edge * seed_amount * 0.04, 0.0, 1.0)

        laplacian_kernel = np.array(
            [[0.05, 0.20, 0.05], [0.20, -1.0, 0.20], [0.05, 0.20, 0.05]],
            dtype=np.float32,
        )
        feed_map = feed_rate + edge * video_influence * 0.025
        kill_map = kill_rate + (1.0 - edge) * video_influence * 0.010
        for _ in range(iterations):
            lap_a = cv2.filter2D(a, cv2.CV_32F, laplacian_kernel, borderType=cv2.BORDER_REFLECT)
            lap_b = cv2.filter2D(b, cv2.CV_32F, laplacian_kernel, borderType=cv2.BORDER_REFLECT)
            reaction = a * b * b
            a += diffusion_a * lap_a - reaction + feed_map * (1.0 - a)
            b += diffusion_b * lap_b + reaction - (kill_map + feed_map) * b
            a = np.clip(a, 0.0, 1.0)
            b = np.clip(b, 0.0, 1.0)

        state["chem_a"] = a
        state["chem_b"] = b
        growth = np.clip(b * growth_contrast, 0.0, 1.0)
        growth_full = cv2.resize(growth, (w, h), interpolation=cv2.INTER_CUBIC)

        grad_x = cv2.Sobel(growth_full, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(growth_full, cv2.CV_32F, 0, 1, ksize=3)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xx + grad_x * warp_strength
        map_y = yy + grad_y * warp_strength

        gpu_img = gpo.ensure_gpu(img_array)
        gpu_map_x = gpo.upload(map_x.astype(np.float32, copy=False))
        gpu_map_y = gpo.upload(map_y.astype(np.float32, copy=False))
        remapped = cv2.cuda.remap(gpu_img, gpu_map_x, gpu_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        mask_u8 = np.clip(growth_full * 255.0, 0, 255).astype(np.uint8)
        rgb = _phase3_rgb(img_cpu)
        if color_mode == "burn":
            tint_rgb = cv2.applyColorMap(mask_u8, cv2.COLORMAP_HOT)
            tint_rgb = cv2.addWeighted(rgb, 0.45, tint_rgb, 0.75, 0.0)
        elif color_mode == "luma_mask":
            mask = growth_full[:, :, None]
            tint_rgb = np.clip(rgb.astype(np.float32) * (1.0 - mask) + 255.0 * mask, 0, 255).astype(np.uint8)
        else:
            tint_rgb = cv2.applyColorMap(mask_u8, cv2.COLORMAP_TURBO)
            tint_rgb = cv2.addWeighted(rgb, 0.55, tint_rgb, 0.55, 0.0)
        tint_frame = _phase3_match_channels(tint_rgb.astype(img_cpu.dtype, copy=False), img_cpu)
        tint_gpu = gpo.upload(tint_frame)
        wet = cv2.cuda.addWeighted(remapped, 0.72, tint_gpu, 0.28, 0.0)
        if mix < 1.0:
            wet = cv2.cuda.addWeighted(gpu_img, 1.0 - mix, wet, mix, 0.0)
        return wet
    except Exception as e:
        logger.error(f"CUDA Reaction-Diffusion error: {e}", exc_info=True)
        return img_array


def _project_velocity(vx: np.ndarray, vy: np.ndarray, iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    if iterations <= 0:
        return vx, vy
    divergence = 0.5 * (
        np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1) +
        np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)
    )
    pressure = np.zeros_like(vx, dtype=np.float32)
    for _ in range(iterations):
        pressure = (
            np.roll(pressure, 1, axis=1) +
            np.roll(pressure, -1, axis=1) +
            np.roll(pressure, 1, axis=0) +
            np.roll(pressure, -1, axis=0) -
            divergence
        ) * 0.25
    vx = vx - 0.5 * (np.roll(pressure, -1, axis=1) - np.roll(pressure, 1, axis=1))
    vy = vy - 0.5 * (np.roll(pressure, -1, axis=0) - np.roll(pressure, 1, axis=0))
    return vx, vy


def apply_cuda_luma_keyed_fluid_dynamics(img_array: np.ndarray, params: dict, time_val: float = 0.0) -> np.ndarray:
    """Master-grid luma-keyed fluid velocity field with CUDA remap egress."""
    try:
        if not gpo.get_use_cuda():
            return img_array

        h, w = _get_shape(img_array)
        if h <= 0 or w <= 0:
            return img_array

        viscosity = float(params.get("viscosity", 0.8))
        flow_dissipation = float(np.clip(params.get("flow_dissipation", 0.94), 0.0, 1.0))
        inject_strength = float(params.get("velocity_inject_strength", 2.5))
        color_decay = float(np.clip(params.get("color_decay", 0.92), 0.0, 1.0))
        # Kept for preset compatibility; image/flow history is now always
        # master-resolution. Provider optimization may change the solver
        # implementation later, but may not reintroduce a reduced history ring.
        grid_resolution = max(32, min(512, int(params.get("grid_resolution", 192))))
        luma_threshold = float(np.clip(params.get("luma_threshold", 0.42), 0.0, 1.0))
        luma_softness = max(0.001, float(params.get("luma_softness", 0.16)))
        projection_iterations = max(0, min(30, int(params.get("projection_iterations", 8))))
        vorticity = float(params.get("vorticity", 0.6))
        warp_strength = float(params.get("warp_strength", 36.0))
        dye_mix = float(np.clip(params.get("dye_mix", 0.35), 0.0, 1.0))
        reset_requested = bool(params.get("reset_fluid", False))
        mix = float(np.clip(params.get("mix", 0.9), 0.0, 1.0))

        if mix <= 0.0:
            return img_array

        img_cpu = gpo.ensure_cpu(img_array)
        gray = _phase3_gray(img_cpu)
        rgb = _phase3_rgb(img_cpu)
        grid_w, grid_h = w, h
        state = get_effect_state("luma_keyed_fluid", (h, w))
        grid_key = (grid_h, grid_w)

        should_reset = (
            state.get("grid_key") != grid_key or
            "vel_x" not in state or
            (reset_requested and not state.get("reset_latched", False))
        )
        state["reset_latched"] = reset_requested
        if should_reset:
            state["vel_x"] = np.zeros((grid_h, grid_w), dtype=np.float32)
            state["vel_y"] = np.zeros((grid_h, grid_w), dtype=np.float32)
            state["dye"] = rgb.astype(np.float32, copy=True)
            state["prev_gray"] = None
            state["grid_key"] = grid_key

        gray_small_u8 = gray
        gray_small = gray_small_u8.astype(np.float32) / 255.0
        luma_gate = np.clip((gray_small - luma_threshold) / luma_softness, 0.0, 1.0)
        prev_gray = state.get("prev_gray")
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray_small_u8,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=13,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            inject_x = flow[:, :, 0]
            inject_y = flow[:, :, 1]
        else:
            inject_x = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
            inject_y = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
        state["prev_gray"] = gray_small_u8.copy()

        vx = state["vel_x"] * flow_dissipation + inject_x * inject_strength * luma_gate
        vy = state["vel_y"] * flow_dissipation + inject_y * inject_strength * luma_gate

        if vorticity > 0.0:
            curl = cv2.Sobel(vy, cv2.CV_32F, 1, 0, ksize=3) - cv2.Sobel(vx, cv2.CV_32F, 0, 1, ksize=3)
            abs_curl = np.abs(curl)
            grad_curl_x = cv2.Sobel(abs_curl, cv2.CV_32F, 1, 0, ksize=3)
            grad_curl_y = cv2.Sobel(abs_curl, cv2.CV_32F, 0, 1, ksize=3)
            norm = np.sqrt(grad_curl_x * grad_curl_x + grad_curl_y * grad_curl_y) + 1e-5
            vx += (grad_curl_y / norm) * curl * vorticity * 0.015
            vy -= (grad_curl_x / norm) * curl * vorticity * 0.015

        if viscosity > 0.0:
            sigma = max(0.01, viscosity * 0.65)
            vx = cv2.GaussianBlur(vx, (0, 0), sigma)
            vy = cv2.GaussianBlur(vy, (0, 0), sigma)

        vx, vy = _project_velocity(vx.astype(np.float32), vy.astype(np.float32), projection_iterations)
        vx = np.clip(vx, -grid_w * 0.33, grid_w * 0.33)
        vy = np.clip(vy, -grid_h * 0.33, grid_h * 0.33)
        state["vel_x"] = vx
        state["vel_y"] = vy

        yy, xx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        advect_x = np.clip(xx - vx, 0, grid_w - 1).astype(np.float32)
        advect_y = np.clip(yy - vy, 0, grid_h - 1).astype(np.float32)
        dye = cv2.remap(state["dye"], advect_x, advect_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        source_small = rgb.astype(np.float32, copy=False)
        gate_3 = luma_gate[:, :, None]
        dye = dye * color_decay + source_small * (1.0 - color_decay) * (0.25 + gate_3 * 0.75)
        state["dye"] = np.clip(dye, 0.0, 255.0)

        vx_full = vx
        vy_full = vy
        scale_x = warp_strength * 0.05
        scale_y = warp_strength * 0.05
        yy_full, xx_full = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xx_full - vx_full * scale_x
        map_y = yy_full - vy_full * scale_y

        gpu_img = gpo.ensure_gpu(img_array)
        gpu_map_x = gpo.upload(map_x.astype(np.float32, copy=False))
        gpu_map_y = gpo.upload(map_y.astype(np.float32, copy=False))
        warped = cv2.cuda.remap(gpu_img, gpu_map_x, gpu_map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if dye_mix > 0.0:
            dye_full = state["dye"]
            dye_frame = _phase3_match_channels(np.clip(dye_full, 0, 255).astype(img_cpu.dtype), img_cpu)
            dye_gpu = gpo.upload(dye_frame)
            warped = cv2.cuda.addWeighted(warped, 1.0 - dye_mix, dye_gpu, dye_mix, 0.0)

        if mix < 1.0:
            warped = cv2.cuda.addWeighted(gpu_img, 1.0 - mix, warped, mix, 0.0)
        return warped
    except Exception as e:
        logger.error(f"CUDA Luma-Keyed Fluid Dynamics error: {e}", exc_info=True)
        return img_array


def _apply_cuda_effect_scoped(effect_type: str, img_array: np.ndarray, params: dict,
                              prev_frame: Optional[np.ndarray] = None,
                              time_val: float = 0.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Route CUDA effect to appropriate handler.
    
    Returns:
        (processed_image, next_previous_frame)
    """
    
    try:
        # New GPU Pipeline Nodes
        if "CUDA GPU Upload" in effect_type:
            # Upload node doesn't modify the frame, just ensures it's on GPU in the node wrapper
            return img_array, img_array
        
        elif "CUDA GPU Warp Remap" in effect_type:
            # Route to GPU node wrapper with persistent context
            try:
                from megaglitchnode.gpu import Frame
                
                ctx = get_gpu_context()
                if ctx is None:
                    return img_array, img_array
                
                node = get_gpu_node("warp", "warp_remap_inline")
                if node is None:
                    return img_array, img_array
                
                node.set_params(**params)
                ctx.frame_index = int(time_val * 60)
                
                # Debug logging
                if ctx.frame_index % 30 == 0:
                    logger.info(f"[GPU WARP] frame_index={ctx.frame_index}, time_val={time_val:.3f}")
                
                if isinstance(img_array, Frame):
                    frame = img_array
                else:
                    img_cpu = gpo.ensure_cpu(img_array)
                    frame = Frame.from_numpy(img_cpu, fmt="bgr8")
                frame.ensure_gpu(ctx)
                
                result_frame = node.process(frame, ctx)
                # Keep on GPU
                return result_frame, result_frame
            except Exception as e:
                logger.error(f"GPU Warp Remap failed: {e}")
                return img_array, img_array
        
        elif "CUDA GPU Feedback Mixer" in effect_type:
            # Route to GPU node wrapper with persistent context
            try:
                from megaglitchnode.gpu import Frame
                
                ctx = get_gpu_context()
                if ctx is None:
                    return img_array, img_array
                
                node = get_gpu_node("feedback", "feedback_mixer_inline")
                if node is None:
                    return img_array, img_array
                
                node.set_params(**params)
                ctx.frame_index = int(time_val * 60)
                
                if isinstance(img_array, Frame):
                    frame = img_array
                else:
                    img_cpu = gpo.ensure_cpu(img_array)
                    frame = Frame.from_numpy(img_cpu, fmt="bgr8")
                frame.ensure_gpu(ctx)
                
                result_frame = node.process(frame, ctx)
                # Keep on GPU
                return result_frame, result_frame
            except Exception as e:
                logger.error(f"GPU Feedback Mixer failed: {e}")
                return img_array, img_array
        
        elif "CUDA GPU Bloom Ghost" in effect_type:
            # Route to GPU node wrapper with persistent context
            try:
                from megaglitchnode.gpu import Frame
                
                ctx = get_gpu_context()
                if ctx is None:
                    return img_array, img_array
                
                node = get_gpu_node("bloom", "bloom_ghost_inline")
                if node is None:
                    return img_array, img_array
                
                node.set_params(**params)
                ctx.frame_index = int(time_val * 60)
                
                if isinstance(img_array, Frame):
                    frame = img_array
                else:
                    img_cpu = gpo.ensure_cpu(img_array)
                    frame = Frame.from_numpy(img_cpu, fmt="bgr8")
                frame.ensure_gpu(ctx)
                
                result_frame = node.process(frame, ctx)
                # Keep on GPU
                return result_frame, result_frame
            except Exception as e:
                logger.error(f"GPU Bloom Ghost failed: {e}")
                return img_array, img_array
        elif "CUDA Recursive Fractal Corrupter" in effect_type:
            result = apply_cuda_recursive_fractal_corrupter(img_array, params, time_val)
            return result, img_array

        elif "CUDA Voronoi Shatter" in effect_type:
            result = apply_cuda_voronoi_shatter(img_array, params, time_val)
            return result, img_array

        elif "CUDA Reaction-Diffusion" in effect_type:
            result = apply_cuda_reaction_diffusion(img_array, params, time_val)
            return result, img_array

        elif "CUDA Luma-Keyed Fluid Dynamics" in effect_type:
            result = apply_cuda_luma_keyed_fluid_dynamics(img_array, params, time_val)
            return result, img_array

        # For other CUDA nodes (the hybrid ones), we must ensure CPU input
        # since their implementations heavily rely on numpy/CPU ops
        img_cpu = gpo.ensure_cpu(img_array)
        
        # Original CUDA Effects
        if "CUDA Flow Warp" in effect_type:
            result, next_prev = apply_cuda_flow_warp(img_cpu, params, prev_frame)
            return result, next_prev
        
        elif "CUDA Spectral Datamosh" in effect_type:
            result = apply_cuda_spectral_datamosh(img_cpu, params, time_val)
            return result, img_cpu
        
        elif "CUDA Nonlinear Remap" in effect_type:
            # We already updated this one to accept GpuMat natively!
            result = apply_cuda_nonlinear_remap(img_array, params, time_val)
            return result, img_array

        elif "CUDA SingularityEngine" in effect_type:
            result = apply_cuda_singularity_engine(img_array, params, time_val)
            return result, img_array
        
        elif "CUDA Persistent Feedback" in effect_type:
            # We already updated this one to accept GpuMat natively!
            result = apply_cuda_persistent_feedback(img_array, params)
            return result, img_array
        
        elif "CUDA Temporal Median" in effect_type:
            result = apply_cuda_temporal_median(img_cpu, params)
            return result, img_cpu
        
        elif "CUDA Flow Pixel Sort" in effect_type:
            result = apply_cuda_flow_pixel_sort(img_cpu, params, prev_frame)
            return result, img_cpu
        
        elif "CUDA Bitplane Drift" in effect_type:
            result = apply_cuda_bitplane_drift(img_cpu, params)
            return result, img_cpu
        
        elif "CUDA Audio Flow" in effect_type:
            result = apply_cuda_audio_flow(img_cpu, params, time_val)
            return result, img_cpu
            
        elif "CUDA Edge Slicer" in effect_type:
            result = apply_cuda_edge_slicer(img_array, params, time_val)
            return result, img_array
            
        elif "CUDA Channel Drift" in effect_type:
            result = apply_cuda_channel_drift(img_array, params, time_val)
            return result, img_array
            
        elif "CUDA Digital Melt" in effect_type:
            result = apply_cuda_digital_melt(img_array, params, time_val)
            return result, img_array
            
        elif "CUDA Vector Mosh" in effect_type:
            result, next_prev = apply_cuda_vector_mosh(img_array, params, prev_frame, time_val)
            return result, next_prev
            
        elif "CUDA Temporal Slit-Scan" in effect_type:
            result = apply_cuda_temporal_slit_scan(img_array, params, time_val)
            return result, img_array

        # For unimplemented CUDA effects, return original
        logger.warning(f"Unimplemented CUDA effect: {effect_type}, passing through")
        return img_array, img_array
        
    except Exception as e:
        logger.error(f"CUDA effect {effect_type} failed: {e}", exc_info=True)
        return img_array, img_array


def apply_cuda_effect(effect_type: str, img_array: np.ndarray, params: dict,
                      prev_frame: Optional[np.ndarray] = None,
                      time_val: float = 0.0, *,
                      state_store: Optional[dict] = None,
                      state_namespace: object = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply a CUDA node with optional stream-owned temporal state.

    Graph callers provide ``state_store``. Omitting it preserves direct-helper
    compatibility but deliberately uses the legacy process-global cache.
    """
    namespace = state_namespace
    if namespace is None:
        namespace = params.get("_node_id", effect_type)
    with use_cuda_state_store(state_store, namespace):
        return _apply_cuda_effect_scoped(
            effect_type,
            img_array,
            params,
            prev_frame=prev_frame,
            time_val=time_val,
        )
