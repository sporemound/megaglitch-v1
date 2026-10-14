import sys
import os
import faulthandler
from pathlib import Path

faulthandler.enable()

# ==========================================================================
# CRITICAL: Bootstrap CUDA DLLs BEFORE any cv2 import!
# This ensures the correct OpenCV-CUDA build is loaded, not pip opencv.
# ==========================================================================
try:
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
except Exception as _bootstrap_err:
    print(f"[BOOTSTRAP] Warning: {_bootstrap_err}")

import time
import threading
import queue
import multiprocessing as mp
from multiprocessing import Manager
import numpy as np
import cv2  # Safe to import AFTER bootstrap
import dearpygui.dearpygui as dpg
import logging
import wave
from tqdm import tqdm
import imageio
import pyvirtualcam
import pygetwindow as gw
import mss
import inspect


from PIL import Image

from akai_mpd26_controls import AkaiMPD26Controls
from audio_midi_manager import AudioMidiManager
from bpm import BPMPeakDetector
from hardware_mapping import HardwareMappingManager
from node_editor import NodeEditor
import effect_nodes
import effect_nodes_cuda
from node_modulation import NodeModulationMatrix
from parameter_timeline import ParameterTimeline
from glitch_effects import apply_glitch_effect, CorruptionMode
import effects_registry
from frame_state import FrameState
import gpu_ops
import ui_theme
from media_handler import open_image_file, open_gif_file, open_video_file, get_window_titles, capture_window_frame, update_display_texture
from node_pipeline_runtime import execute_flat_node_pipeline, record_frame_copy
from megaglitch_visual_analyzer import VisualAnalyzerWorker
from megaglitch_visual_bus import VisualBus
from megaglitch_midi_router import MidiDeviceManager
from megaglitch_live_sequencer import LiveSequencer
from megaglitch_sequencer_ui import build_live_sequencer_ui
from megaglitch.audio.media_audio_engine import (
    MEDIA_AUDIO_MOD_SOURCES,
    MEDIA_AUDIO_PRESETS,
    MediaAudioController,
    TX802_ALGORITHM_NAMES,
)
from megaglitch.audio.cdp_graph import CDP_EFFECT_SPECS
from megaglitch.render_contract import (
    FrameRole,
    MasterResolution,
    RenderMode,
    make_egress_frame,
    parse_render_mode,
    prepare_master_ingress,
    validate_master_frame,
)
from megaglitch.process_logging import create_process_log_handler
from srt_shader_cues import (
    apply_srt_shader_cues,
    find_video_srt_sidecar,
    parse_srt_shader_cues,
)


_OFFLINE_ENCODER_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
    "placebo",
)
_CODEC_POST_EDITABLE_TAGS = (
    "codec_post_command",
    "codec_post_work_directory",
    "codec_post_seed",
    "codec_post_long_gop",
    "codec_post_preparation_quality",
    "codec_post_final_codec",
    "codec_post_final_crf",
    "codec_post_final_preset",
)

# Configure logging with a process-isolated handler. Frozen UI and worker
# processes must never share a log or write into the immutable application
# bundle; source launches retain the established repository-local log.
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)
_handler = create_process_log_handler(
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
)
_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
_root_logger.handlers = [_handler]


# ==== HELPER FUNCTIONS ====

def apply_retro_mode(img_array, bits_per_channel=4, dither=True):
    """
    Apply retro 16-bit graphics mode (color quantization + dithering).
    OPTIMIZED: Uses fast ordered (Bayer) dithering instead of Floyd-Steinberg.
    
    Args:
        img_array: numpy array (H, W, 3) uint8
        bits_per_channel: Number of bits per color channel (1-8)
        dither: Whether to apply dithering
    
    Returns:
        numpy array (H, W, 3) uint8 with reduced color depth
    """
    if bits_per_channel >= 8:
        return img_array  # No reduction needed
    
    # Calculate number of levels for quantization
    levels = 2 ** bits_per_channel
    scale = 255.0 / (levels - 1)
    
    if dither:
        # Use 4x4 Bayer matrix for ordered dithering (MUCH faster than Floyd-Steinberg)
        bayer_matrix = np.array([
            [ 0,  8,  2, 10],
            [12,  4, 14,  6],
            [ 3, 11,  1,  9],
            [15,  7, 13,  5]
        ], dtype=np.float32) / 16.0 - 0.5  # Normalize to -0.5 to 0.5 range
        
        h, w, c = img_array.shape
        
        # Tile the Bayer matrix to cover the entire image
        tile_h = (h + 3) // 4
        tile_w = (w + 3) // 4
        threshold_map = np.tile(bayer_matrix, (tile_h, tile_w))[:h, :w]
        
        # Apply dithering: add threshold pattern before quantization
        img_float = img_array.astype(np.float32)
        for channel in range(c):
            img_float[:, :, channel] += threshold_map * scale
        
        # Quantize
        quantized = np.round(img_float / scale) * scale
        result = np.clip(quantized, 0, 255).astype(np.uint8)
    else:
        # Simple quantization without dithering
        quantized = np.round(img_array.astype(np.float32) / scale) * scale
        result = np.clip(quantized, 0, 255).astype(np.uint8)
    
    return result


# ==== PROCEDURAL GENERATOR FUNCTIONS ====

def apply_generator_mutation(frame, time, params):
    """Shared evolving mutation pass for all procedural generators."""
    arr = np.asarray(frame, dtype=np.uint8)
    strength = float(params.get("mutation_strength", 0.0) or 0.0)
    warp_strength = float(params.get("mutation_warp", 0.5) or 0.0)
    palette_drift = float(params.get("mutation_palette", 0.4) or 0.0)
    fracture = float(params.get("mutation_fracture", 0.4) or 0.0)
    echo = float(params.get("mutation_echo", 0.35) or 0.0)
    rate = float(params.get("mutation_rate", 1.0) or 0.0)

    if strength <= 0.0:
        return arr

    h, w = arr.shape[:2]
    t = float(time) * max(0.0, rate)
    max_dim = 360.0
    scale = min(1.0, max_dim / max(float(h), float(w), 1.0))
    work_w = max(1, int(round(w * scale)))
    work_h = max(1, int(round(h * scale)))

    if scale < 1.0:
        work = cv2.resize(arr, (work_w, work_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        work = arr.astype(np.float32)

    y, x = np.mgrid[0:work_h, 0:work_w].astype(np.float32)
    x_norm = x / max(1.0, float(work_w))
    y_norm = y / max(1.0, float(work_h))

    map_x = x + (
        np.sin((y_norm * (8.0 + fracture * 12.0) + t * 1.7) * np.pi * 2.0) +
        np.cos(((x_norm + y_norm) * (5.0 + warp_strength * 10.0) - t * 1.2) * np.pi * 2.0)
    ) * (strength * warp_strength * max(2.0, work_w * 0.03))
    map_y = y + (
        np.cos((x_norm * (7.0 + fracture * 9.0) - t * 1.4) * np.pi * 2.0) +
        np.sin(((x_norm - y_norm) * (4.0 + warp_strength * 8.0) + t * 1.9) * np.pi * 2.0)
    ) * (strength * warp_strength * max(2.0, work_h * 0.03))
    work = cv2.remap(
        work,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    if fracture > 0.0:
        band_shift = np.round(
            np.sin(np.linspace(0.0, np.pi * (10.0 + fracture * 12.0), work_h, dtype=np.float32) + t * 2.8) *
            strength * fracture * max(1.0, work_w * 0.05)
        ).astype(np.int32)
        fractured = work.copy()
        for row_idx, shift in enumerate(band_shift):
            fractured[row_idx] = np.roll(fractured[row_idx], int(shift), axis=0)
        work = fractured

    if echo > 0.0:
        shift_x = int(max(1, round((1.0 + echo * 10.0) * strength)))
        shift_y = int(max(1, round((1.0 + echo * 6.0) * strength)))
        ghost_a = np.roll(work, shift_x, axis=1)
        ghost_b = np.roll(work, -shift_y, axis=0)
        work = np.clip(
            work * (1.0 - 0.35 * echo) +
            ghost_a * (0.2 + 0.3 * echo) +
            ghost_b * (0.08 + 0.18 * echo),
            0,
            255,
        )

    if palette_drift > 0.0:
        hsv = cv2.cvtColor(np.clip(work, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hue_delta = (t * 18.0 * palette_drift + strength * 40.0) % 180.0
        sat_wave = 1.0 + np.sin((x_norm * 3.0 + y_norm * 5.0 + t) * np.pi * 2.0) * 0.18 * strength
        val_wave = 1.0 + np.cos((x_norm * 4.0 - y_norm * 2.0 - t * 1.3) * np.pi * 2.0) * 0.14 * strength
        hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_wave, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * val_wave, 0, 255)
        work = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    contrast = 1.0 + strength * 0.9
    work = np.clip((work - 127.5) * contrast + 127.5, 0, 255)
    out = work.astype(np.uint8)
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out

def generate_perlin_noise(width, height, time, params):
    """Generate Perlin noise field (optimized with GPU acceleration)."""
    import gpu_ops as gpo
    
    scale = params.get("scale", 80.0)
    octaves = int(params.get("octaves", 3))
    persistence = params.get("persistence", 0.6)
    lacunarity = params.get("lacunarity", 2.0)
    time_scale = params.get("time_scale", 1.5)
    color_variance = params.get("color_variance", 0.4)
    brightness = params.get("brightness", 1.2)
    contrast = params.get("contrast", 1.3)
    
    t = time * time_scale
    
    # Use lower resolution for speed, then upscale
    downsample = 2
    small_w = max(1, width // downsample)
    small_h = max(1, height // downsample)
    
    # Generate for each channel
    result = np.zeros((small_h, small_w, 3), dtype=np.float32)
    
    for c in range(3):
        channel_offset = c * color_variance * 10.0
        result[:, :, c] = _generate_octave_noise(
            small_w, small_h, scale, octaves, persistence, lacunarity,
            t + channel_offset
        )
    
    # Normalize to 0-1
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    
    # Apply brightness and contrast
    result = result * brightness
    result = (result - 0.5) * contrast + 0.5
    result = np.clip(result, 0, 1)
    
    # Convert to 0-255
    result = (result * 255).astype(np.uint8)
    
    # Upscale to target resolution using GPU-accelerated resize
    if (small_h, small_w) != (height, width):
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return apply_generator_mutation(result, time, params)


def _generate_octave_noise(width, height, scale, octaves, persistence, lacunarity, time_offset):
    """Generate multi-octave smooth noise."""
    result = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for octave in range(octaves):
        freq_scale = frequency / scale
        y = np.linspace(0, height * freq_scale, height, dtype=np.float32)
        x = np.linspace(0, width * freq_scale + time_offset, width, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        
        # Simplified noise (fewer terms for speed)
        noise = (
            np.sin(xx * 2.0 + octave) * np.cos(yy * 2.0) * 0.6 +
            np.sin(xx * 3.7 + octave * 1.3 + time_offset * 0.1) * 0.4
        )
        
        result += noise * amplitude
        max_value += amplitude
        
        amplitude *= persistence
        frequency *= lacunarity
    
    # Normalize
    if max_value > 0:
        result /= max_value
    
    return result


def generate_gradient(width, height, params):
    """Generate gradient (optimized with GPU acceleration)."""
    import gpu_ops as gpo
    
    mode = params.get("mode", "linear")
    angle = params.get("angle", 0.0)
    
    # Generate at lower resolution for speed
    downsample = 4
    small_w = max(1, width // downsample)
    small_h = max(1, height // downsample)
    
    color1 = np.array([
        params.get("color1_r", 0),
        params.get("color1_g", 0),
        params.get("color1_b", 0)
    ], dtype=np.float32)
    
    color2 = np.array([
        params.get("color2_r", 255),
        params.get("color2_g", 255),
        params.get("color2_b", 255)
    ], dtype=np.float32)
    
    if mode == "linear":
        angle_rad = np.radians(angle)
        y, x = np.ogrid[:small_h, :small_w]
        x_rot = (x - small_w/2) * np.cos(angle_rad) - (y - small_h/2) * np.sin(angle_rad)
        gradient = (x_rot - x_rot.min()) / (x_rot.max() - x_rot.min() + 1e-8)
    elif mode == "radial":
        y, x = np.ogrid[:small_h, :small_w]
        x = (x - small_w/2) / (small_w/2)
        y = (y - small_h/2) / (small_h/2)
        gradient = np.sqrt(x*x + y*y)
        gradient = np.clip(gradient, 0, 1)
    else:  # angle
        y, x = np.ogrid[:small_h, :small_w]
        x = x - small_w/2
        y = y - small_h/2
        gradient = (np.arctan2(y, x) + np.pi) / (2 * np.pi)
    
    # Apply gradient to colors (vectorized)
    gradient_3d = gradient[:, :, np.newaxis]
    result = (color1 * (1 - gradient_3d) + color2 * gradient_3d).astype(np.uint8)
    
    # Upscale using GPU-accelerated resize
    if (small_h, small_w) != (height, width):
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    
    mutation_time = float(params.get("time_accum", params.get("noise_warp_time", 0.0)) or 0.0)
    return apply_generator_mutation(result, mutation_time, params)


def generate_plasma(width, height, time, params):
    """Generate plasma effect with GPU acceleration."""
    import colorsys
    import gpu_ops as gpo
    
    freq = params.get("frequency", 2.0)
    speed = params.get("speed", 1.0)
    hue_shift = params.get("hue_shift", 0.0) / 360.0
    saturation = params.get("saturation", 0.9)
    complexity = int(params.get("complexity", 3))
    
    t = time * speed
    
    # Generate at lower resolution
    downsample = 2
    w = max(1, width // downsample)
    h = max(1, height // downsample)
    
    y, x = np.ogrid[:h, :w]
    x = x.astype(np.float32) / w
    y = y.astype(np.float32) / h
    
    # Multi-layer plasma
    plasma = np.zeros((h, w), dtype=np.float32)
    
    for i in range(complexity):
        phase = i * 2.1 + t * (0.5 + i * 0.2)
        plasma += np.sin(x * freq * (i+1) * np.pi * 4 + phase)
        plasma += np.sin(y * freq * (i+1) * np.pi * 4 + phase * 0.7)
        plasma += np.sin((x + y) * freq * (i+1) * np.pi * 2 + phase * 1.3)
        plasma += np.sin(np.sqrt((x-0.5)**2 + (y-0.5)**2) * freq * (i+1) * np.pi * 8 + phase * 0.9)
    
    # Normalize
    plasma = (plasma - plasma.min()) / (plasma.max() - plasma.min() + 1e-8)
    
    # Convert to RGB using HSV (vectorized)
    hue_array = (plasma + hue_shift) % 1.0
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[:,:,0] = hue_array * 179  # OpenCV HSV: H in [0,179]
    hsv[:,:,1] = saturation * 255
    hsv[:,:,2] = 255
    hsv = hsv.astype(np.uint8)
    
    # Convert HSV to RGB using GPU-accelerated cvtColor
    result = gpo.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Upscale using GPU-accelerated resize
    if (h, w) != (height, width):
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return apply_generator_mutation(result, time, params)


def generate_voronoi(width, height, time, params):
    """Generate Voronoi diagram with GPU acceleration."""
    import colorsys
    import gpu_ops as gpo
    
    num_cells = int(params.get("num_cells", 30))
    speed = params.get("speed", 0.5)
    color_mode = params.get("color_mode", "rainbow")
    brightness = params.get("brightness", 1.2)
    
    # Generate at lower resolution
    downsample = 3
    w = max(1, width // downsample)
    h = max(1, height // downsample)
    
    # Generate random cell positions (seeded for consistency)
    np.random.seed(42)
    cell_positions = np.random.rand(num_cells, 2)
    
    # Animate positions
    t = time * speed * 0.1
    for i in range(num_cells):
        cell_positions[i, 0] += np.sin(t + i * 0.5) * 0.1
        cell_positions[i, 1] += np.cos(t + i * 0.7) * 0.1
    cell_positions = cell_positions % 1.0
    
    # Scale to image dimensions
    cell_coords = (cell_positions * [w, h]).astype(np.int32)
    
    # Create Voronoi diagram
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            distances = np.sqrt((cell_coords[:, 0] - j)**2 + (cell_coords[:, 1] - i)**2)
            closest_idx = np.argmin(distances)
            closest_dist = distances[closest_idx]
            
            if color_mode == "rainbow":
                hue = (closest_idx / num_cells) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.8, brightness)
            elif color_mode == "distance":
                val = min(1.0, closest_dist / (max(w, h) * 0.3))
                r, g, b = val * brightness, val * brightness, val * brightness
            else:  # cell_id
                val = (closest_idx * 137) % 255
                r, g, b = val/255.0 * brightness, (255-val)/255.0 * brightness, ((val+128)%255)/255.0 * brightness
            
            result[i, j] = [int(r*255), int(g*255), int(b*255)]
    
    # Upscale using GPU-accelerated resize
    if (h, w) != (height, width):
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return apply_generator_mutation(result, time, params)


def generate_fractal(width, height, time, params):
    """Generate Mandelbrot fractal with infinite zoom and GPU acceleration."""
    import colorsys
    import gpu_ops as gpo
    
    zoom = params.get("zoom", 1.0)
    center_x = params.get("center_x", -0.5)
    center_y = params.get("center_y", 0.0)
    max_iter = int(params.get("max_iter", 30))
    color_speed = params.get("color_speed", 1.0)
    continuous_zoom = params.get("continuous_zoom", False)
    zoom_speed = params.get("zoom_speed", 1.0)
    
    # Debug: Log zoom parameters
    logging.debug(f"Fractal params: continuous_zoom={continuous_zoom}, zoom={zoom}, zoom_speed={zoom_speed}, time={time}")
    
    # Apply continuous zoom if enabled
    if continuous_zoom:
        # Exponential zoom for smooth infinite effect
        zoom = zoom * np.exp(time * zoom_speed * 0.1)
        
        # Optionally drift the center point for more interesting zoom
        center_x += np.sin(time * 0.1) * 0.01 / zoom
        center_y += np.cos(time * 0.15) * 0.01 / zoom
        
        logging.debug(f"Infinite zoom active! Applied zoom: {zoom}")
    
    # Generate at lower resolution
    downsample = 2
    w = max(1, width // downsample)
    h = max(1, height // downsample)
    
    # Create complex plane
    xmin, xmax = center_x - 2.0/zoom, center_x + 2.0/zoom
    ymin, ymax = center_y - 2.0/zoom, center_y + 2.0/zoom
    
    x = np.linspace(xmin, xmax, w)
    y = np.linspace(ymin, ymax, h)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y
    
    # Calculate Mandelbrot set
    Z = np.zeros_like(C)
    M = np.zeros(C.shape)
    
    for i in range(max_iter):
        mask = np.abs(Z) <= 2
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] = i
    
    # Normalize
    M = M / max_iter
    
    # Apply time-based color cycling
    hue_offset = (time * color_speed * 0.1) % 1.0
    
    # Convert to RGB
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if M[i, j] >= 0.99:  # Inside set
                result[i, j] = [0, 0, 0]
            else:
                hue = (M[i, j] + hue_offset) % 1.0
                r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
                result[i, j] = [int(r*255), int(g*255), int(b*255)]
    
    # Upscale using GPU-accelerated resize
    if (h, w) != (height, width):
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return apply_generator_mutation(result, time, params)


def generate_checkerboard(width, height, time, params):
    """Generate checkerboard pattern."""
    grid_size = int(params.get("grid_size", 8))
    rotation = params.get("rotation", 0.0)
    rotation_speed = params.get("rotation_speed", 0.0)
    
    color1 = np.array([
        params.get("color1_r", 255),
        params.get("color1_g", 0),
        params.get("color1_b", 255)
    ], dtype=np.uint8)
    
    color2 = np.array([
        params.get("color2_r", 0),
        params.get("color2_g", 255),
        params.get("color2_b", 255)
    ], dtype=np.uint8)
    
    # Animated rotation
    angle = np.radians(rotation + time * rotation_speed * 10)
    
    # Create coordinate grid
    y, x = np.ogrid[:height, :width]
    
    # Center and rotate
    cx, cy = width / 2, height / 2
    x_rot = (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle)
    y_rot = (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    
    # Create checkerboard
    cell_size = max(width, height) / grid_size
    checker = ((x_rot // cell_size).astype(int) + (y_rot // cell_size).astype(int)) % 2
    
    # Apply colors
    result = np.where(checker[:, :, np.newaxis], color1, color2)
    
    return apply_generator_mutation(result.astype(np.uint8), time, params)


def generate_scan_feedback(width, height, time, params):
    """Generate unstable scan/sync feedback with retrace bands and tearing."""
    import gpu_ops as gpo

    stripe_density = float(params.get("stripe_density", 72.0))
    drift_speed = float(params.get("drift_speed", 1.2))
    tear_amount = float(params.get("tear_amount", 0.35))
    feedback = float(params.get("feedback", 0.45))
    noise_amount = float(params.get("noise_amount", 0.12))
    channel_split = float(params.get("channel_split", 0.3))

    downsample = 2
    w = max(1, width // downsample)
    h = max(1, height // downsample)

    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    xf = x / max(1.0, float(w))
    yf = y / max(1.0, float(h))
    t = time * drift_speed

    line_phase = yf * stripe_density * np.pi * 2.0 + t * 6.0
    scan = 0.5 + 0.5 * np.sin(line_phase)
    field = 0.5 + 0.5 * np.sin((yf * (stripe_density * 0.5) - t * 1.7) * np.pi * 2.0)

    tear_field = np.sin((yf * 19.0 - t * 2.3) * np.pi * 2.0)
    offset = tear_amount * (
        0.05 * np.sign(tear_field) * (0.5 + 0.5 * np.sin((yf * 6.0 + t) * np.pi * 2.0))
        + 0.03 * np.sin((yf * 31.0 + t * 3.0) * np.pi * 2.0)
    )
    x_warp = (xf + offset) % 1.0

    carrier = 0.5 + 0.5 * np.sin((x_warp * 28.0 + yf * 8.0 + t * 4.0) * np.pi * 2.0)
    diagonal = 0.5 + 0.5 * np.sin(((x_warp + yf) * 14.0 - t * 2.6) * np.pi * 2.0)
    retrace = np.exp(-(((np.mod(yf * 4.0 - t * 0.55, 1.0)) - 0.02) ** 2) / 0.0018)

    base = scan * 0.45 + field * 0.15 + carrier * 0.25 + diagonal * 0.15
    echo = np.roll(base, int(max(1, w * (0.01 + feedback * 0.04))), axis=1)
    sparkle = (
        np.sin((x_warp * 97.0 + yf * 53.0 + t * 8.0) * np.pi * 2.0) *
        np.sin((x_warp * 23.0 - yf * 71.0 - t * 5.0) * np.pi * 2.0)
    )
    base = np.clip(base * (1.0 - feedback) + echo * feedback + retrace * 0.85 + sparkle * noise_amount, 0.0, 1.0)

    split_px = int(max(1, round(channel_split * 6)))
    r = np.roll(base, split_px, axis=1)
    g = 0.5 + 0.5 * np.sin((x_warp * 22.0 + yf * 13.0 + t * 3.4) * np.pi * 2.0)
    b = np.roll(base, -split_px, axis=1)
    result = np.stack([r, g * 0.8 + base * 0.2, b], axis=-1)
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    if (h, w) != (height, width):
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    return apply_generator_mutation(result, time, params)


def generate_xor_lattice(width, height, time, params):
    """Generate digital lattice patterns from bitwise interference and warped coordinate logic."""
    import gpu_ops as gpo

    lattice_scale = float(params.get("lattice_scale", 18.0))
    warp = float(params.get("warp", 0.55))
    speed = float(params.get("speed", 1.0))
    palette_shift = float(params.get("palette_shift", 0.0))
    contrast = float(params.get("contrast", 1.3))
    bit_depth = int(params.get("bit_depth", 8))

    downsample = 2
    w = max(1, width // downsample)
    h = max(1, height // downsample)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    xf = x / max(1.0, float(w))
    yf = y / max(1.0, float(h))
    t = time * speed

    xw = xf + warp * (
        0.16 * np.sin((yf * 7.0 + t * 0.9) * np.pi * 2.0) +
        0.05 * np.cos((yf * 19.0 - t * 1.4) * np.pi * 2.0)
    )
    yw = yf + warp * (
        0.16 * np.cos((xf * 9.0 - t * 1.1) * np.pi * 2.0) +
        0.05 * np.sin((xf * 15.0 + t * 1.7) * np.pi * 2.0)
    )

    ix = np.floor(xw * lattice_scale * 16.0 + t * 11.0).astype(np.int32)
    iy = np.floor(yw * lattice_scale * 16.0 - t * 9.0).astype(np.int32)
    mask = (1 << max(1, min(8, bit_depth))) - 1

    xor_a = np.bitwise_xor(ix & 255, iy & 255)
    xor_b = np.bitwise_xor((ix * 5) & 255, ((iy << 1) + int(t * 17)) & 255)
    xor_c = np.bitwise_xor(((ix >> 1) + (iy * 3)) & 255, ((iy >> 2) + int(t * 31)) & 255)

    pulse = 0.5 + 0.5 * np.sin(((xf + yf) * lattice_scale * 0.75 + t * 2.4) * np.pi * 2.0)
    r = ((xor_a & mask) / max(1, mask)) * 255.0
    g = ((xor_b & mask) / max(1, mask)) * 255.0
    b = ((xor_c & mask) / max(1, mask)) * 255.0

    result = np.stack([r, g, b], axis=-1)
    result = (result - 127.5) * contrast + 127.5
    result[..., 1] = result[..., 1] * (0.65 + 0.35 * pulse)
    result[..., 2] = result[..., 2] * (1.1 - 0.25 * pulse)
    result = np.roll(result.astype(np.uint8), int(round(palette_shift)) % 3, axis=2)

    if (h, w) != (height, width):
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_NEAREST)
    return apply_generator_mutation(np.clip(result, 0, 255).astype(np.uint8), time, params)


def generate_macroblock_echo(width, height, time, params):
    """Generate compression-style macroblocks with drift, echo ghosts, and corruption islands."""
    import gpu_ops as gpo

    block_size = int(params.get("block_size", 18))
    drift = float(params.get("drift", 1.0))
    echo_strength = float(params.get("echo_strength", 0.5))
    corruption = float(params.get("corruption", 0.35))
    palette_levels = int(params.get("palette_levels", 6))

    block_size = max(4, block_size)
    grid_w = max(1, width // block_size)
    grid_h = max(1, height // block_size)
    gy, gx = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
    t = time * drift

    row_shift = np.round(np.sin(gy * 0.9 + t * 2.1) * (1.0 + drift * 2.5)).astype(np.int32)
    col_shift = np.round(np.cos(gx * 1.1 - t * 1.7) * (0.5 + drift * 1.8)).astype(np.int32)

    block_id = ((gx + row_shift) * 37 + (gy + col_shift) * 61 + int(t * 23)) % 256
    ghost_a = np.roll(block_id, int(max(1, round(1 + echo_strength * 3))), axis=1)
    ghost_b = np.roll(block_id, -int(max(1, round(1 + echo_strength * 2))), axis=0)
    corrupt_mask = (
        0.5 + 0.5 * np.sin((gx * 1.7 + gy * 2.3 + t * 4.1))
    ) > (0.92 - corruption * 0.35)

    r = block_id
    g = np.bitwise_xor(block_id.astype(np.uint8), ghost_a.astype(np.uint8)).astype(np.float32)
    b = (ghost_b * (0.6 + echo_strength * 0.4) + ghost_a * 0.4) % 256
    grid = np.stack([r, g, b], axis=-1).astype(np.float32)

    if palette_levels > 1:
        step = 255.0 / (palette_levels - 1)
        grid = np.round(grid / step) * step

    grid[corrupt_mask] = np.roll(grid[corrupt_mask], 1, axis=-1)
    grid = np.clip(grid, 0, 255).astype(np.uint8)
    result = np.repeat(np.repeat(grid, block_size, axis=0), block_size, axis=1)
    result = result[:height, :width]

    if result.shape[0] != height or result.shape[1] != width:
        result = gpo.resize(result, (width, height), interpolation=cv2.INTER_NEAREST)
    return apply_generator_mutation(result.astype(np.uint8), time, params)


def generate_bubble_sim_preview(width, height, time, params):
    """CPU preview for the Bubble Sim Gen menu node.

    The OpenGL BubbleSimGeneratorNode owns the real ping-pong texture subsystem;
    this keeps the existing CPU pipeline able to display/select the generator.
    """
    density = float(np.clip(params.get("density", 0.55) or 0.55, 0.0, 1.0))
    tex_size = int(np.clip(params.get("particle_texture_size", 128) or 128, 16, 512))
    seed = int(params.get("seed", 1977) or 1977)
    base_radius = float(np.clip(params.get("base_radius", 0.018) or 0.018, 0.001, 0.08))
    radius_variation = float(np.clip(params.get("radius_variation", 0.65) or 0.65, 0.0, 1.0))
    buoyancy = float(np.clip(params.get("buoyancy", 0.16) or 0.16, 0.0, 2.0))
    turbulence = float(np.clip(params.get("turbulence", 0.35) or 0.35, 0.0, 2.0))
    flow_speed = float(np.clip(params.get("flow_speed", 0.25) or 0.25, 0.0, 4.0))
    depth_spread = float(np.clip(params.get("depth_spread", 0.8) or 0.8, 0.0, 1.0))
    depth_motion = float(np.clip(params.get("depth_motion", 0.7) or 0.7, 0.0, 2.0))
    orbit_strength = float(np.clip(params.get("orbit_strength", 0.8) or 0.8, 0.0, 2.0))
    perspective_strength = float(np.clip(params.get("perspective", 1.0) or 1.0, 0.0, 2.0))
    refraction = float(np.clip(params.get("refraction_strength", 0.025) or 0.025, 0.0, 0.15))
    highlight_strength = float(np.clip(params.get("highlight_strength", 0.45) or 0.45, 0.0, 2.0))
    sparkle = float(np.clip(params.get("sparkle", 0.25) or 0.25, 0.0, 1.0))
    show_energy = float(np.clip(params.get("show_energy", 0.0) or 0.0, 0.0, 1.0))
    show_chaos = float(np.clip(params.get("show_chaos", 0.0) or 0.0, 0.0, 1.0))
    view = str(params.get("debug_mode", "composite") or "composite")

    preview_scale = min(1.0, 480.0 / max(float(width), float(height), 1.0))
    work_w = max(64, int(round(width * preview_scale)))
    work_h = max(64, int(round(height * preview_scale)))
    count = int(min(tex_size * tex_size * max(density, 0.02), 320))
    rng = np.random.default_rng(seed)
    ids = np.arange(count, dtype=np.float32)
    x0 = rng.random(count, dtype=np.float32) * 2.0 - 1.0
    y0 = rng.random(count, dtype=np.float32)
    z0 = rng.random(count, dtype=np.float32)
    radius_rand = rng.random(count, dtype=np.float32)
    orbit_phase = rng.random(count, dtype=np.float32) * np.pi * 2.0
    orbit_rate = 0.35 + rng.random(count, dtype=np.float32) * 1.15
    age = (rng.random(count, dtype=np.float32) + time * (0.035 + buoyancy * 0.04)) % 1.0

    flow_t = time * max(flow_speed, 0.01)
    depth = np.mod(
        z0 +
        np.sin(flow_t * 0.23 + orbit_phase) * turbulence * 0.055 * depth_motion +
        np.cos(flow_t * 0.11 + ids * 0.071) * show_chaos * 0.08 * depth_motion,
        1.0,
    ) * max(depth_spread, 0.001)
    depth = np.clip(depth, 0.0, 1.0)

    rise_speed = (0.10 + buoyancy * 0.22) * (1.25 - depth * 0.70)
    world_y = np.mod(y0 + time * rise_speed + age * 0.16, 1.35) - 0.18
    world_y = world_y * 2.0 - 1.0
    orbit = orbit_phase + flow_t * orbit_rate + age * np.pi * 2.0
    lateral = np.sin(orbit) * turbulence * (0.045 + show_chaos * 0.035) * orbit_strength
    depth_lateral = np.cos(orbit * 0.71) * (0.035 + show_chaos * 0.045) * orbit_strength
    world_x = x0 + lateral + depth_lateral * (0.5 - depth)

    perspective = np.clip(
        1.0 / (1.0 + perspective_strength * (depth * 1.15 - 0.38)),
        0.42,
        2.6,
    )
    camera_sway = np.sin(time * 0.09) * 0.035
    projected_x = 0.5 + (world_x + camera_sway * depth) * 0.43 * perspective
    projected_y = 0.55 - world_y * 0.43 * perspective
    radius_px = (
        base_radius * work_h *
        (0.35 + radius_rand * (1.0 + radius_variation)) *
        perspective *
        (1.0 + show_energy * 0.22)
    )

    mask = np.zeros((work_h, work_w), dtype=np.float32)
    depth_plane = np.zeros((work_h, work_w), dtype=np.float32)
    age_plane = np.zeros((work_h, work_w), dtype=np.float32)
    highlight_plane = np.zeros((work_h, work_w), dtype=np.float32)
    shade_plane = np.zeros((work_h, work_w), dtype=np.float32)

    # Painter's order gives near bubbles proper overlap and stronger presence.
    for i in np.argsort(depth)[::-1]:
        r = max(1, int(round(radius_px[i])))
        cx = int(projected_x[i] * (work_w - 1))
        cy = int(projected_y[i] * (work_h - 1))
        if cx + r < 0 or cx - r >= work_w or cy + r < 0 or cy - r >= work_h:
            continue

        depth_gain = float(1.0 - depth[i] * 0.68)
        pop = np.clip((age[i] - 0.78) / 0.22, 0.0, 1.0)
        glint = ((np.sin(ids[i] * 19.7 + np.floor(time * 18.0)) * 0.5 + 0.5) > 0.86) * pop * sparkle
        alpha = depth_gain * (0.46 + show_energy * 0.38)
        squash = 0.88 + 0.10 * np.sin(orbit[i] * 0.53)
        axes = (r, max(1, int(round(r * squash))))
        light_offset = max(1, int(round(r * 0.32)))
        light_center = (cx - light_offset, cy - light_offset)

        cv2.ellipse(mask, (cx, cy), axes, 0.0, 0.0, 360.0, alpha, thickness=-1, lineType=cv2.LINE_AA)
        cv2.ellipse(depth_plane, (cx, cy), axes, 0.0, 0.0, 360.0, float(depth[i]), thickness=-1, lineType=cv2.LINE_AA)
        cv2.ellipse(age_plane, (cx, cy), axes, 0.0, 0.0, 360.0, float(age[i]), thickness=-1, lineType=cv2.LINE_AA)

        # Dark displaced core plus bright opposing rim creates a refractive sphere.
        cv2.ellipse(
            shade_plane,
            (cx + light_offset // 2, cy + light_offset // 2),
            (max(1, int(r * 0.72)), max(1, int(axes[1] * 0.72))),
            0.0, 0.0, 360.0,
            float(0.18 + depth[i] * 0.20),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.ellipse(
            highlight_plane,
            (cx, cy),
            axes,
            0.0, 0.0, 360.0,
            float((0.26 + glint) * highlight_strength * depth_gain),
            thickness=max(1, r // 5),
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            highlight_plane,
            light_center,
            max(1, int(round(r * 0.16))),
            float((0.55 + glint) * highlight_strength * depth_gain),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    mask = cv2.GaussianBlur(mask, (0, 0), 0.65)
    grad_x = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    inv_mag = 1.0 / np.maximum(grad_mag, 1e-5)
    nx = np.clip(-grad_x * inv_mag, -1.0, 1.0)
    ny = np.clip(-grad_y * inv_mag, -1.0, 1.0)
    nz = np.sqrt(np.clip(1.0 - np.minimum(1.0, nx * nx + ny * ny), 0.0, 1.0))

    mask3 = np.dstack([mask, mask, mask])
    normal = np.clip(np.dstack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5]) * mask[..., None], 0.0, 1.0)
    depth_img = np.clip(np.dstack([depth_plane, age_plane, mask]), 0.0, 1.0)
    refr_img = np.clip(np.dstack([
        0.5 + nx * refraction * 8.0,
        0.5 + ny * refraction * 8.0,
        highlight_plane,
    ]) * mask[..., None], 0.0, 1.0)
    high = np.clip(np.dstack([
        highlight_plane * 0.72,
        highlight_plane * 0.95,
        highlight_plane,
    ]), 0.0, 1.0)

    if view == "mask":
        result = mask3
    elif view == "normal":
        result = normal
    elif view == "depth":
        result = depth_img
    elif view == "refraction":
        result = refr_img
    elif view == "highlight":
        result = high
    else:
        deep = np.dstack([
            mask * 15.0 + high[..., 0] * 255.0 - shade_plane * 22.0,
            mask * 44.0 + high[..., 1] * 255.0 - shade_plane * 30.0,
            mask * 64.0 + high[..., 2] * 255.0 - shade_plane * 34.0,
        ]) / 255.0
        result = np.clip(deep, 0.0, 1.0)

    result = (result * 255.0).astype(np.uint8)
    if (work_h, work_w) != (height, width):
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)
    return apply_generator_mutation(result, time, params)


def _colony_reef_neighbor_mean(field):
    return (
        np.roll(field, 1, axis=0) +
        np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) +
        np.roll(field, -1, axis=1) +
        0.7 * np.roll(np.roll(field, 1, axis=0), 1, axis=1) +
        0.7 * np.roll(np.roll(field, 1, axis=0), -1, axis=1) +
        0.7 * np.roll(np.roll(field, -1, axis=0), 1, axis=1) +
        0.7 * np.roll(np.roll(field, -1, axis=0), -1, axis=1)
    ) / 6.8


def _colony_reef_neighbor_mean_stack(fields):
    return (
        np.roll(fields, 1, axis=1) +
        np.roll(fields, -1, axis=1) +
        np.roll(fields, 1, axis=2) +
        np.roll(fields, -1, axis=2) +
        0.7 * np.roll(np.roll(fields, 1, axis=1), 1, axis=2) +
        0.7 * np.roll(np.roll(fields, 1, axis=1), -1, axis=2) +
        0.7 * np.roll(np.roll(fields, -1, axis=1), 1, axis=2) +
        0.7 * np.roll(np.roll(fields, -1, axis=1), -1, axis=2)
    ) / 6.8


def _colony_reef_phase_profile(phase_name):
    profiles = {
        "bloom": {"regen": 0.090, "growth": 1.40, "conflict": 0.55, "signal": 0.70, "predator": 0.45, "mutation": 0.70},
        "war": {"regen": 0.040, "growth": 0.95, "conflict": 1.65, "signal": 1.20, "predator": 0.90, "mutation": 1.30},
        "famine": {"regen": 0.020, "growth": 0.58, "conflict": 1.15, "signal": 0.45, "predator": 1.10, "mutation": 0.85},
        "spore": {"regen": 0.060, "growth": 1.15, "conflict": 0.85, "signal": 1.35, "predator": 0.55, "mutation": 1.55},
        "trance": {"regen": 0.045, "growth": 0.86, "conflict": 0.42, "signal": 1.85, "predator": 0.35, "mutation": 1.05},
    }
    return profiles.get(phase_name, profiles["bloom"])


def _colony_reef_pick_phase(rng, current_phase, instability):
    phases = ["bloom", "war", "famine", "spore", "trance"]
    weights = {
        "bloom": np.array([0.0, 0.18, 0.12, 0.30, 0.40], dtype=np.float32),
        "war": np.array([0.22, 0.0, 0.28, 0.18, 0.32], dtype=np.float32),
        "famine": np.array([0.30, 0.24, 0.0, 0.12, 0.34], dtype=np.float32),
        "spore": np.array([0.34, 0.16, 0.10, 0.0, 0.40], dtype=np.float32),
        "trance": np.array([0.26, 0.26, 0.18, 0.30, 0.0], dtype=np.float32),
    }
    w = weights.get(current_phase, weights["bloom"]).copy()
    w += np.array([0.10, 0.22, 0.12, 0.18, 0.28], dtype=np.float32) * float(instability)
    w = np.clip(w, 0.0, None)
    w /= max(1e-6, float(np.sum(w)))
    return phases[int(rng.choice(len(phases), p=w))]


def _seed_colony_reef_patch(state, colony_idx, cx, cy, strength=0.7, radius=2.6):
    y_grid = state["y_grid"]
    x_grid = state["x_grid"]
    blob = np.exp(-(((x_grid - cx) ** 2) + ((y_grid - cy) ** 2)) / max(1e-3, radius ** 2)).astype(np.float32)
    state["fields"][colony_idx] = np.clip(state["fields"][colony_idx] + blob * strength, 0.0, 1.25)
    state["signal"] = np.clip(state["signal"] + blob * strength * 0.22, 0.0, 1.0)


def _init_colony_reef_state(width, height, params, node_key):
    cell_size = max(4, int(round(max(width, height) / 170.0)))
    grid_w = max(48, width // cell_size)
    grid_h = max(32, height // cell_size)
    colony_count = max(3, int(params.get("colony_count", 9) or 9))
    seed = (hash((str(node_key), grid_w, grid_h, colony_count)) & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)

    x_grid = np.broadcast_to(np.arange(grid_w, dtype=np.float32)[None, :], (grid_h, grid_w))
    y_grid = np.broadcast_to(np.arange(grid_h, dtype=np.float32)[:, None], (grid_h, grid_w))
    xn = x_grid / max(1.0, float(grid_w - 1))
    yn = y_grid / max(1.0, float(grid_h - 1))

    nutrient = (
        0.42
        + 0.20 * np.sin((xn * 2.7 + yn * 1.9) * np.pi * 2.0)
        + 0.13 * np.cos((xn * 5.3 - yn * 3.1) * np.pi * 2.0)
        + rng.normal(0.0, 0.08, size=(grid_h, grid_w))
    ).astype(np.float32)
    nutrient = np.clip(nutrient, 0.05, 1.0)

    state = {
        "version": 1,
        "size": (width, height),
        "grid_size": (grid_w, grid_h),
        "cell_size": cell_size,
        "fields": np.zeros((colony_count, grid_h, grid_w), dtype=np.float32),
        "nutrient": nutrient,
        "scar": np.zeros((grid_h, grid_w), dtype=np.float32),
        "signal": np.zeros((grid_h, grid_w), dtype=np.float32),
        "x_grid": x_grid,
        "y_grid": y_grid,
        "xn": xn,
        "yn": yn,
        "rng": rng,
        "phase": "bloom",
        "phase_timer": 0.0,
        "phase_duration": float(rng.uniform(7.0, 13.0)),
        "phase_shift_flash": 0.0,
        "last_time": None,
        "rule_types": rng.integers(0, 5, size=colony_count, dtype=np.int32),
        "render_modes": rng.integers(0, 3, size=colony_count, dtype=np.int32),
        "hues": rng.random(colony_count, dtype=np.float32),
        "aggression": rng.uniform(0.25, 1.0, size=colony_count).astype(np.float32),
        "appetite": rng.uniform(0.35, 1.0, size=colony_count).astype(np.float32),
        "resilience": rng.uniform(0.25, 1.0, size=colony_count).astype(np.float32),
        "oscillators": rng.uniform(0.6, 2.8, size=colony_count).astype(np.float32),
        "anchor_x": rng.uniform(0.0, float(grid_w - 1), size=colony_count).astype(np.float32),
        "anchor_y": rng.uniform(0.0, float(grid_h - 1), size=colony_count).astype(np.float32),
    }

    scavenger_count = max(4, colony_count // 2 + 2)
    state["scavenger_pos"] = rng.uniform([0.0, 0.0], [max(1.0, grid_w - 1.0), max(1.0, grid_h - 1.0)], size=(scavenger_count, 2)).astype(np.float32)
    state["scavenger_vel"] = rng.normal(0.0, 0.35, size=(scavenger_count, 2)).astype(np.float32)

    for colony_idx in range(colony_count):
        _seed_colony_reef_patch(
            state,
            colony_idx,
            state["anchor_x"][colony_idx],
            state["anchor_y"][colony_idx],
            strength=float(rng.uniform(0.45, 0.95)),
            radius=float(rng.uniform(1.6, 3.4)),
        )

    return state


def _render_colony_reef(state, width, height, params, time_value):
    import colorsys
    import gpu_ops as gpo

    fields = np.clip(state["fields"], 0.0, 1.25)
    nutrient = np.clip(state["nutrient"], 0.0, 1.0)
    scar = np.clip(state["scar"], 0.0, 1.0)
    signal = np.clip(state["signal"], 0.0, 1.0)
    total = np.sum(fields, axis=0)
    dominant = np.max(fields, axis=0)
    if fields.shape[0] > 1:
        sorted_fields = np.sort(fields, axis=0)
        second = sorted_fields[-2]
    else:
        second = np.zeros_like(dominant)
    frontier = np.clip(1.0 - (dominant - second) * 5.5, 0.0, 1.0)

    phase_name = state["phase"]
    luminous_residue = float(params.get("luminous_residue", 0.9) or 0.9)
    phase_shift = {
        "bloom": 0.00,
        "war": 0.08,
        "famine": -0.04,
        "spore": 0.14,
        "trance": 0.22,
    }.get(phase_name, 0.0)

    colors = []
    for i, base_hue in enumerate(state["hues"]):
        hue = float((base_hue + phase_shift + 0.015 * np.sin(time_value * state["oscillators"][i] + i * 1.7)) % 1.0)
        sat = 0.70 + 0.25 * ((state["render_modes"][i] % 3) / 2.0)
        val = 0.68 + 0.26 * state["resilience"][i]
        colors.append(colorsys.hsv_to_rgb(hue, sat, val))
    color_table = np.array(colors, dtype=np.float32)

    weighted = np.tensordot(fields, color_table, axes=(0, 0))
    weighted /= np.maximum(total[..., None], 1e-4)

    base = weighted * (0.35 + 0.9 * np.clip(total, 0.0, 1.0)[..., None])
    nutrient_glow = nutrient[..., None] * np.array([0.06, 0.16, 0.10], dtype=np.float32)
    signal_glow = signal[..., None] * np.array([0.10, 0.16, 0.22], dtype=np.float32) * luminous_residue
    scar_tint = scar[..., None] * np.array([0.05, 0.06, 0.08], dtype=np.float32)
    frontier_glow = frontier[..., None] * np.array([0.18, 0.20, 0.13], dtype=np.float32) * (0.4 + 0.8 * state["phase_shift_flash"])

    phase_tint = {
        "bloom": np.array([0.02, 0.08, 0.03], dtype=np.float32),
        "war": np.array([0.14, 0.03, 0.02], dtype=np.float32),
        "famine": np.array([0.08, 0.07, 0.03], dtype=np.float32),
        "spore": np.array([0.07, 0.11, 0.03], dtype=np.float32),
        "trance": np.array([0.04, 0.05, 0.14], dtype=np.float32),
    }.get(phase_name, np.zeros(3, dtype=np.float32))

    out = base + nutrient_glow + signal_glow + frontier_glow + phase_tint - scar_tint * 0.8

    agent_map = np.zeros_like(dominant)
    for pos in state["scavenger_pos"]:
        ix = int(np.clip(round(float(pos[0])), 0, agent_map.shape[1] - 1))
        iy = int(np.clip(round(float(pos[1])), 0, agent_map.shape[0] - 1))
        agent_map[iy, ix] += 1.0
    agent_map = cv2.GaussianBlur(agent_map.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
    out += agent_map[..., None] * np.array([0.16, 0.22, 0.20], dtype=np.float32)

    out = np.clip(out, 0.0, 1.0)
    small = (out * 255.0).astype(np.uint8)
    if small.shape[1] != width or small.shape[0] != height:
        smooth = gpo.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
        edge = gpo.resize((frontier * 255.0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        smooth = np.clip(smooth.astype(np.float32) + edge[..., None].astype(np.float32) * np.array([0.10, 0.12, 0.07], dtype=np.float32), 0, 255)
        return smooth.astype(np.uint8)
    return small


def generate_colony_reef(width, height, time, params, worker_state=None, node_key=None):
    """Persistent ecological generator with territorial colonies, scavengers, and phase mutations."""
    state_key = f"colony_reef::{node_key if node_key is not None else 'default'}"
    generator_states = getattr(worker_state, "generator_states", None)
    if generator_states is None:
        generator_states = {}
        if worker_state is not None:
            worker_state.generator_states = generator_states

    colony_count = max(3, int(params.get("colony_count", 9) or 9))
    state = generator_states.get(state_key)
    if (
        state is None or
        state.get("size") != (width, height) or
        state.get("fields", np.empty((0,))).shape[0] != colony_count
    ):
        state = _init_colony_reef_state(width, height, params, node_key)
        generator_states[state_key] = state

    rng = state["rng"]
    last_time = state.get("last_time")
    if last_time is None:
        dt = 1.0 / 60.0
    else:
        dt = float(np.clip(time - float(last_time), 1.0 / 120.0, 0.25))
    state["last_time"] = float(time)

    mutation_rate = float(params.get("mutation_rate", 0.28) or 0.28)
    memory_decay = float(params.get("memory_decay", 0.14) or 0.14)
    phase_instability = float(params.get("phase_instability", 0.42) or 0.42)
    region_spread = float(params.get("region_spread_speed", 1.0) or 1.0)
    catastrophic_reset_probability = float(params.get("catastrophic_reset_probability", 0.01) or 0.01)
    nutrient_flux = float(params.get("nutrient_flux", 1.0) or 1.0)
    predator_pressure = float(params.get("predator_pressure", 0.7) or 0.7)

    step_count = max(1, min(4, int(np.ceil(dt / (1.0 / 45.0)))))
    sub_dt = dt / float(step_count)

    for step_idx in range(step_count):
        sim_time = time - dt + sub_dt * (step_idx + 1)
        phase_name = state["phase"]
        phase_profile = _colony_reef_phase_profile(phase_name)
        state["phase_timer"] += sub_dt
        state["phase_shift_flash"] *= max(0.0, 1.0 - sub_dt * 1.7)

        should_shift = state["phase_timer"] >= state["phase_duration"]
        should_shift = should_shift or (rng.random() < sub_dt * (0.02 + phase_instability * 0.16))
        if should_shift:
            state["phase"] = _colony_reef_pick_phase(rng, phase_name, phase_instability)
            state["phase_timer"] = 0.0
            state["phase_duration"] = float(rng.uniform(6.0, 16.0) * (0.8 + (1.0 - phase_instability) * 0.6))
            state["phase_shift_flash"] = 1.0
            phase_name = state["phase"]
            phase_profile = _colony_reef_phase_profile(phase_name)

        fields = state["fields"]
        nutrient = state["nutrient"]
        scar = state["scar"]
        signal = state["signal"]
        xn = state["xn"]
        yn = state["yn"]

        total_before = np.sum(fields, axis=0)
        field_neighbors = _colony_reef_neighbor_mean_stack(fields)
        nutrient_neighbors = _colony_reef_neighbor_mean(nutrient)
        signal_neighbors = _colony_reef_neighbor_mean(signal)
        scar_neighbors = _colony_reef_neighbor_mean(scar)

        nutrient += (nutrient_neighbors - nutrient) * (0.22 + nutrient_flux * 0.12) * sub_dt
        nutrient += phase_profile["regen"] * nutrient_flux * sub_dt
        nutrient -= np.clip(total_before, 0.0, 1.0) * (0.020 + 0.050 * region_spread) * sub_dt
        nutrient = np.clip(nutrient, 0.0, 1.0)

        signal += (signal_neighbors - signal) * 0.18 * sub_dt
        signal += np.clip(total_before, 0.0, 1.5) * (0.06 + 0.12 * phase_profile["signal"]) * sub_dt
        signal *= max(0.0, 1.0 - (0.08 + 0.22 * memory_decay) * sub_dt)
        signal = np.clip(signal, 0.0, 1.0)

        scar += (scar_neighbors - scar) * 0.05 * sub_dt
        scar += np.clip(total_before - nutrient * 0.7, 0.0, 1.0) * (0.015 + 0.05 * predator_pressure) * sub_dt
        scar *= max(0.0, 1.0 - (0.015 + 0.10 * memory_decay) * sub_dt)
        scar = np.clip(scar, 0.0, 1.0)

        total = np.sum(fields, axis=0)
        for colony_idx in range(fields.shape[0]):
            if rng.random() < mutation_rate * phase_profile["mutation"] * sub_dt * 0.20:
                state["rule_types"][colony_idx] = int(rng.integers(0, 5))
                state["render_modes"][colony_idx] = int(rng.integers(0, 3))
                state["hues"][colony_idx] = float((state["hues"][colony_idx] + rng.uniform(-0.18, 0.18)) % 1.0)
                state["aggression"][colony_idx] = float(np.clip(state["aggression"][colony_idx] + rng.normal(0.0, 0.12), 0.05, 1.6))
                state["appetite"][colony_idx] = float(np.clip(state["appetite"][colony_idx] + rng.normal(0.0, 0.12), 0.05, 1.5))
                state["resilience"][colony_idx] = float(np.clip(state["resilience"][colony_idx] + rng.normal(0.0, 0.10), 0.05, 1.4))
                state["oscillators"][colony_idx] = float(np.clip(state["oscillators"][colony_idx] + rng.normal(0.0, 0.28), 0.35, 4.0))
                state["anchor_x"][colony_idx] = float(np.clip(state["anchor_x"][colony_idx] + rng.normal(0.0, 4.5), 0.0, fields.shape[2] - 1))
                state["anchor_y"][colony_idx] = float(np.clip(state["anchor_y"][colony_idx] + rng.normal(0.0, 4.5), 0.0, fields.shape[1] - 1))
                state["phase_shift_flash"] = max(state["phase_shift_flash"], 0.35)

            field = fields[colony_idx]
            edge = np.clip(field_neighbors[colony_idx] - field * 0.72, 0.0, 1.0)
            rivalry = np.clip(total - field, 0.0, 1.5)
            appetite = state["appetite"][colony_idx]
            aggression = state["aggression"][colony_idx]
            resilience = state["resilience"][colony_idx]
            osc = state["oscillators"][colony_idx]
            anchor_dx = xn - (state["anchor_x"][colony_idx] / max(1.0, fields.shape[2] - 1.0))
            anchor_dy = yn - (state["anchor_y"][colony_idx] / max(1.0, fields.shape[1] - 1.0))
            anchor_pull = np.exp(-(anchor_dx * anchor_dx + anchor_dy * anchor_dy) / 0.040).astype(np.float32)
            wave = 0.5 + 0.5 * np.sin((xn * (2.0 + colony_idx * 0.11) + yn * (1.3 + colony_idx * 0.07) + sim_time * osc) * np.pi * 2.0)
            veins = 0.5 + 0.5 * np.cos((xn * 7.0 - yn * 5.0 + sim_time * (0.4 + osc * 0.8)) * np.pi * 2.0)
            rule_type = int(state["rule_types"][colony_idx])

            if rule_type == 0:
                target = edge * nutrient * (1.0 - scar) * (0.45 + 0.55 * wave)
            elif rule_type == 1:
                target = np.clip(field_neighbors[colony_idx] * 1.25 - field * 0.55, 0.0, 1.0) * (0.35 + 0.65 * signal) * (0.3 + 0.7 * anchor_pull)
            elif rule_type == 2:
                target = edge * (0.25 + 0.85 * rivalry) * (0.30 + 0.70 * signal) * (0.45 + 0.55 * veins)
            elif rule_type == 3:
                target = edge * nutrient * (0.15 + 0.85 * ((wave > 0.58).astype(np.float32))) * (1.0 - 0.5 * scar)
            else:
                target = np.clip(edge + anchor_pull * 0.8, 0.0, 1.0) * (0.25 + 0.75 * nutrient) * (0.25 + 0.75 * signal)

            growth = target * (0.06 + 0.18 * appetite) * region_spread * phase_profile["growth"] * sub_dt
            conflict = rivalry * (0.012 + 0.035 * aggression * phase_profile["conflict"]) * sub_dt
            fade = (0.010 + 0.055 * memory_decay * (1.3 - resilience)) * sub_dt
            field *= max(0.0, 1.0 - fade)
            field += growth
            field -= conflict
            field -= scar * (0.004 + 0.010 * memory_decay) * sub_dt
            fields[colony_idx] = np.clip(field, 0.0, 1.35)

            nutrient -= growth * (0.08 + 0.12 * appetite)
            signal += growth * (0.05 + 0.10 * phase_profile["signal"])
            scar += conflict * (0.18 + 0.25 * aggression)

            if float(np.mean(fields[colony_idx])) < 0.003 and rng.random() < sub_dt * 0.22:
                _seed_colony_reef_patch(
                    state,
                    colony_idx,
                    float(rng.uniform(0.0, fields.shape[2] - 1.0)),
                    float(rng.uniform(0.0, fields.shape[1] - 1.0)),
                    strength=float(rng.uniform(0.20, 0.55)),
                    radius=float(rng.uniform(1.5, 3.8)),
                )

        if phase_name == "spore" and rng.random() < sub_dt * (0.40 + phase_instability):
            spore_colony = int(rng.integers(0, fields.shape[0]))
            hotspot = signal + nutrient * 0.5 - scar * 0.3
            flat_idx = int(np.argmax(hotspot + rng.normal(0.0, 0.03, size=hotspot.shape)))
            cy, cx = np.unravel_index(flat_idx, hotspot.shape)
            _seed_colony_reef_patch(
                state,
                spore_colony,
                float(cx) + float(rng.normal(0.0, 1.8)),
                float(cy) + float(rng.normal(0.0, 1.8)),
                strength=float(rng.uniform(0.18, 0.42)),
                radius=float(rng.uniform(1.8, 4.2)),
            )

        hunt_field = np.sum(fields, axis=0) * (0.65 + 0.35 * phase_profile["predator"]) + signal * phase_profile["signal"] - nutrient * 0.18
        for agent_idx in range(state["scavenger_pos"].shape[0]):
            pos = state["scavenger_pos"][agent_idx]
            vel = state["scavenger_vel"][agent_idx]
            ix = int(np.clip(round(float(pos[0])), 0, fields.shape[2] - 1))
            iy = int(np.clip(round(float(pos[1])), 0, fields.shape[1] - 1))
            x0 = max(0, ix - 1)
            x1 = min(fields.shape[2] - 1, ix + 1)
            y0 = max(0, iy - 1)
            y1 = min(fields.shape[1] - 1, iy + 1)
            grad_x = float(hunt_field[iy, x1] - hunt_field[iy, x0])
            grad_y = float(hunt_field[y1, ix] - hunt_field[y0, ix])
            turn = np.array([grad_x, grad_y], dtype=np.float32)
            vel = vel * 0.82 + turn * (0.34 + 0.30 * predator_pressure) + rng.normal(0.0, 0.18, size=2).astype(np.float32)
            speed = np.linalg.norm(vel) + 1e-6
            vel = vel / speed * min(speed, 1.25 + predator_pressure * 0.8)
            pos = pos + vel * (0.35 + 0.45 * predator_pressure)
            if pos[0] < 0 or pos[0] > fields.shape[2] - 1:
                vel[0] *= -1
            if pos[1] < 0 or pos[1] > fields.shape[1] - 1:
                vel[1] *= -1
            pos[0] = np.clip(pos[0], 0.0, fields.shape[2] - 1.0)
            pos[1] = np.clip(pos[1], 0.0, fields.shape[1] - 1.0)
            state["scavenger_pos"][agent_idx] = pos
            state["scavenger_vel"][agent_idx] = vel

            ix = int(np.clip(round(float(pos[0])), 0, fields.shape[2] - 1))
            iy = int(np.clip(round(float(pos[1])), 0, fields.shape[1] - 1))
            radius = 2 if predator_pressure > 0.9 else 1
            xs = slice(max(0, ix - radius), min(fields.shape[2], ix + radius + 1))
            ys = slice(max(0, iy - radius), min(fields.shape[1], iy + radius + 1))
            fields[:, ys, xs] *= max(0.0, 1.0 - (0.07 + 0.10 * predator_pressure * phase_profile["predator"]) * sub_dt * 12.0)
            scar[ys, xs] = np.clip(scar[ys, xs] + (0.05 + 0.07 * predator_pressure) * sub_dt * 12.0, 0.0, 1.0)
            signal[ys, xs] = np.clip(signal[ys, xs] + (0.04 + 0.08 * phase_profile["signal"]) * sub_dt * 12.0, 0.0, 1.0)
            nutrient[ys, xs] = np.clip(nutrient[ys, xs] + 0.012 * phase_profile["regen"] * sub_dt * 12.0, 0.0, 1.0)

        if rng.random() < catastrophic_reset_probability * sub_dt:
            wipe_noise = (
                0.5
                + 0.5 * np.sin((state["xn"] * rng.uniform(3.0, 7.0) + state["yn"] * rng.uniform(2.0, 8.0) + sim_time * rng.uniform(0.7, 2.0)) * np.pi * 2.0)
                + 0.35 * np.cos((state["xn"] * rng.uniform(8.0, 13.0) - state["yn"] * rng.uniform(5.0, 11.0)) * np.pi * 2.0)
            )
            wipe_mask = wipe_noise > (0.62 - phase_instability * 0.18)
            fields[:, wipe_mask] *= rng.uniform(0.02, 0.28)
            nutrient[wipe_mask] = np.clip(nutrient[wipe_mask] + rng.uniform(0.15, 0.45), 0.0, 1.0)
            scar[wipe_mask] = np.clip(scar[wipe_mask] + rng.uniform(0.10, 0.35), 0.0, 1.0)
            for _ in range(max(1, fields.shape[0] // 4)):
                colony_idx = int(rng.integers(0, fields.shape[0]))
                _seed_colony_reef_patch(
                    state,
                    colony_idx,
                    float(rng.uniform(0.0, fields.shape[2] - 1.0)),
                    float(rng.uniform(0.0, fields.shape[1] - 1.0)),
                    strength=float(rng.uniform(0.15, 0.40)),
                    radius=float(rng.uniform(1.4, 3.5)),
                )
            state["phase"] = "spore" if rng.random() < 0.55 else "famine"
            state["phase_timer"] = 0.0
            state["phase_duration"] = float(rng.uniform(5.0, 10.0))
            state["phase_shift_flash"] = 1.0

        state["fields"] = np.clip(fields, 0.0, 1.35)
        state["nutrient"] = np.clip(nutrient, 0.0, 1.0)
        state["scar"] = np.clip(scar, 0.0, 1.0)
        state["signal"] = np.clip(signal, 0.0, 1.0)

    return _render_colony_reef(state, width, height, params, float(time))


def glitch_process_worker(raw_queue, glitched_queue, stop_event, node_params_getter,
                          shared_transport_spec=None, lifecycle_queue=None):
    """This function runs in a separate process to handle glitching."""
    from megaglitch.worker_lifecycle import report_worker_status
    report_worker_status(lifecycle_queue, "warming_up", backend="python_worker")
    
    # ==========================================================================
    # CRITICAL: Bootstrap CUDA DLLs BEFORE any cv2 import!
    # On Windows, os.add_dll_directory() doesn't carry to child processes
    # ==========================================================================
    try:
        from megaglitchnode.opencv_cuda_bootstrap import (
            bootstrap_for_dev_defaults,
            invalidate_gpu_caches,
            run_cuda_selftest
        )
        bootstrap_for_dev_defaults()
        invalidate_gpu_caches()  # Clear any stale GPU state
    except Exception as e:
        print(f"[WORKER] Bootstrap warning: {e}")
    
    # NOW safe to import cv2
    import cv2
    
    # ==========================================================================
    # DIAGNOSTICS: Prove we're using CUDA OpenCV build (not pip opencv)
    # ==========================================================================
    print("=" * 60)
    print("[WORKER] OpenCV CUDA Diagnostics")
    print("=" * 60)
    print(f"[WORKER] cv2.__file__: {cv2.__file__}")
    print(f"[WORKER] hasattr(cv2, 'cuda'): {hasattr(cv2, 'cuda')}")
    if hasattr(cv2, 'cuda'):
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            print(f"[WORKER] cv2.cuda.getCudaEnabledDeviceCount(): {device_count}")
        except Exception as e:
            print(f"[WORKER] cv2.cuda.getCudaEnabledDeviceCount() FAILED: {e}")
    else:
        print("[WORKER] cv2.cuda NOT AVAILABLE - using CPU-only OpenCV!")
    print("=" * 60)
    
    # Configure logging for worker process (console only, no file to avoid conflicts)
    import logging
    logger = logging.getLogger()
    logger.handlers.clear()  # Remove all handlers from parent process
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[WORKER] %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    logging.info("Worker process started with clean logging configuration")
    
    # ==========================================================================
    # OpenCV-CUDA Self-Test: Validate CUDA works in worker (not just deviceQuery)
    # ==========================================================================
    import numpy as np
    worker_cuda_ok = False
    try:
        if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Test 1: GaussianFilter create + apply + download
            test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(test_img)
            
            gauss = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.0)
            gpu_result = gauss.apply(gpu_img)
            result = gpu_result.download()
            
            if result.shape == test_img.shape:
                print("[WORKER] OpenCV-CUDA GaussianFilter test: PASS")
                worker_cuda_ok = True
            else:
                print("[WORKER] OpenCV-CUDA GaussianFilter test: FAIL (shape mismatch)")
        else:
            print("[WORKER] OpenCV-CUDA test: SKIPPED (no CUDA devices)")
    except Exception as e:
        print(f"[WORKER] OpenCV-CUDA GaussianFilter test: FAIL ({e})")
    
    # Run full CUDA self-test
    try:
        results = run_cuda_selftest()
        if results['success']:
            logging.info(f"[WORKER] CUDA verified: {results['device_count']} device(s), all tests passed")
            worker_cuda_ok = True
        else:
            logging.warning(f"[WORKER] CUDA self-test failed: {results['errors']}")
            logging.warning("[WORKER] GPU effects will use CPU fallback")
    except Exception as e:
        logging.warning(f"[WORKER] CUDA self-test error: {e}")
    
    # Set global CUDA flag based on test results
    try:
        import gpu_ops
        gpu_ops.set_use_cuda(worker_cuda_ok)
        print(f"[WORKER] gpu_ops.USE_CUDA set to: {worker_cuda_ok}")
    except Exception as e:
        print(f"[WORKER] Failed to set gpu_ops.USE_CUDA: {e}")
    
    shared_transport = None
    if shared_transport_spec is not None:
        try:
            from megaglitch.shared_frame_transport import SharedFrameTransportEndpoint
            shared_transport = SharedFrameTransportEndpoint(shared_transport_spec)
            logging.info("Worker attached to fixed shared-memory frame transport")
        except Exception as exc:
            # The parent will send descriptors when shared mode is selected. Continuing
            # with legacy payload handling would corrupt that contract.
            raise RuntimeError("failed to attach required shared frame transport") from exc

    state = FrameState()
    state_b = FrameState()
    report_worker_status(
        lifecycle_queue,
        "ready",
        backend="python_worker",
        cuda_available=bool(worker_cuda_ok),
        frame_transport="shared_v1" if shared_transport is not None else "legacy",
    )

    def reset_state():
        nonlocal state, state_b
        state.reset()
        state_b.reset()
        # Clear the glitched queue to remove stale frames
        while not glitched_queue.empty():
            try:
                stale_result = glitched_queue.get_nowait()
                if shared_transport is not None:
                    shared_transport.release_output_result(stale_result)
            except queue.Empty:
                break
        logging.info("Glitch process state and output queue have been reset.")

    while not stop_event.is_set():
        try:
            source_item = raw_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error retrieving from raw queue: {e}")
            continue

        shared_input_slots = []
        try:
            if source_item == "RESET":
                reset_state()
                # Echo RESET to output queue
                try:
                    glitched_queue.put_nowait("RESET")
                except queue.Full:
                    pass
                continue

            if source_item:
                render_started = time.perf_counter()
                frame_copy_stats = {
                    str(name): {
                        "count": int(values.get("count", 0)),
                        "bytes": int(values.get("bytes", 0)),
                    }
                    for name, values in (source_item.get("frame_copy_stats") or {}).items()
                    if isinstance(values, dict)
                }
                source_frame = source_item["image"]
                source_copy_policy = str(source_item.get("source_copy_policy", "copy"))
                if source_copy_policy not in {"copy", "borrow_unique"}:
                    source_copy_policy = "copy"
                benchmark_seed = source_item.get("_benchmark_seed")
                if benchmark_seed is not None:
                    import random
                    seed = int(benchmark_seed)
                    random.seed(seed)
                    np.random.seed(seed & 0xFFFFFFFF)
                secondary_source_frame = source_item.get("secondary_image")
                if shared_transport is not None:
                    source_frame, source_slot = shared_transport.resolve_input(source_frame)
                    secondary_source_frame, secondary_slot = shared_transport.resolve_input(secondary_source_frame)
                    shared_input_slots.extend(
                        slot for slot in (source_slot, secondary_slot) if slot is not None
                    )
                processing_pipeline = source_item.get("pipeline")
                explicit_params = source_item.get("params")
                genesis_debug_request = source_item.get("genesis_debug_request")
                
                # Convert source to numpy if needed
                if isinstance(source_frame, Image.Image):
                    current_glitched_frame = np.array(source_frame.convert("RGB"))
                elif isinstance(source_frame, np.ndarray):
                    current_glitched_frame = source_frame
                else:
                    current_glitched_frame = None

                if isinstance(secondary_source_frame, Image.Image):
                    secondary_source_frame = np.array(secondary_source_frame.convert("RGB"))
                elif not isinstance(secondary_source_frame, np.ndarray):
                    secondary_source_frame = None

                fallback_width = int(source_item.get("fallback_width", 1280) or 1280)
                fallback_height = int(source_item.get("fallback_height", 720) or 720)

                # The graph is a master-only domain.  Legacy callers without
                # metadata are inferred once for compatibility; current callers
                # declare the full-resolution contract explicitly.
                render_mode = parse_render_mode(
                    source_item.get("render_mode", RenderMode.LIVE.value)
                )
                try:
                    frame_role = FrameRole(
                        str(source_item.get("frame_role", FrameRole.MASTER_INGRESS.value))
                    )
                except ValueError as exc:
                    raise ValueError("worker received an unknown frame role") from exc
                if frame_role is not FrameRole.MASTER_INGRESS:
                    raise ValueError(
                        f"worker accepts only master_ingress frames, got {frame_role.value}"
                    )

                declared_resolution = source_item.get("master_resolution")
                if declared_resolution is not None:
                    master_resolution = MasterResolution.from_mapping(declared_resolution)
                else:
                    target_reference = (
                        current_glitched_frame
                        if current_glitched_frame is not None
                        else secondary_source_frame
                    )
                    if target_reference is None:
                        master_resolution = MasterResolution(
                            width=fallback_width,
                            height=fallback_height,
                        )
                    else:
                        master_resolution = MasterResolution(
                            width=int(target_reference.shape[1]),
                            height=int(target_reference.shape[0]),
                        )

                current_glitched_frame = validate_master_frame(
                    current_glitched_frame,
                    master_resolution,
                    role=frame_role,
                    name="worker primary source",
                )
                secondary_source_frame = validate_master_frame(
                    secondary_source_frame,
                    master_resolution,
                    role=frame_role,
                    name="worker secondary source",
                )
                frame_shape = master_resolution.shape
                resolution_changed = state.bind_master_shape(
                    frame_shape,
                    allow_resolution_change=True,
                )
                resolution_changed_b = state_b.bind_master_shape(
                    frame_shape,
                    allow_resolution_change=True,
                )
                if resolution_changed or resolution_changed_b:
                    logging.info(
                        "Master resolution changed to %sx%s; temporal state reset",
                        master_resolution.width,
                        master_resolution.height,
                    )
                
                current_previous_frame = state.previous_frame
                current_feedback_image = state.feedback_image
                current_trail_image = state.trail_image
                current_frame_buffer = state.frame_buffer

                # Non-destructive param injection from the producer (e.g., global time)
                extra_params = source_item.get("extra_params")

                # Configure GPU usage per frame in worker
                try:
                    use_cuda = bool(source_item.get("use_cuda", False))
                    import gpu_ops as _gpu_ops
                    _gpu_ops.set_use_cuda(use_cuda)
                    # Removed noisy per-frame logging
                except Exception as e:
                    logging.error(f"[WORKER GPU] Failed to set CUDA: {e}")

                from node_pipeline_runtime import blend_decks
                
                pipeline_b = source_item.get("pipeline_b")
                for effect_config in pipeline_b or []:
                    if effect_config.get("node_type") == "source":
                        effect_config["source_key"] = "secondary"
                deck_a_uses_secondary = any(
                    effect_config.get("node_type") == "source"
                    and effect_config.get("source_key") == "secondary"
                    for effect_config in processing_pipeline or []
                )
                master_mix = float(source_item.get("master_mix", 0.5))
                master_transition = source_item.get("master_transition", "Normal Mix")
                
                extra_params_with_frame = dict(extra_params or {})
                extra_params_with_frame.setdefault('time_accum', float(state.frame_count) / 60.0)
                extra_params_with_frame.setdefault('noise_warp_time', float(extra_params_with_frame.get('time_accum', 0.0)))
                
                frame_a = current_glitched_frame
                frame_b = None
                
                if processing_pipeline:
                    buffer_limit = getattr(state, 'buffer_limit', 10)
                    for effect_config in processing_pipeline:
                        node_params = dict(effect_config.get("params", {}) or {})
                        try:
                            if 'frame_buffer_size' in node_params:
                                buffer_limit = int(node_params.get('frame_buffer_size', buffer_limit) or buffer_limit)
                            elif 'buffer_size' in node_params:
                                buffer_limit = int(node_params.get('buffer_size', buffer_limit) or buffer_limit)
                        except Exception:
                            pass

                    (
                        frame_a,
                        current_previous_frame,
                        current_feedback_image,
                        current_trail_image,
                    ) = execute_flat_node_pipeline(
                        processing_pipeline,
                        current_glitched_frame,
                        secondary_frame=secondary_source_frame if deck_a_uses_secondary else None,
                        extra_params=extra_params_with_frame,
                        previous_frame=current_previous_frame,
                        feedback_image=current_feedback_image,
                        trail_image=current_trail_image,
                        frame_buffer=current_frame_buffer,
                        worker_state=state,
                        fallback_size=(fallback_width, fallback_height),
                        copy_stats=frame_copy_stats,
                        source_copy_policy=source_copy_policy,
                    )
                    try:
                        history_frame = frame_a.copy()
                        record_frame_copy(frame_copy_stats, "history", history_frame)
                        state.frame_buffer.append(history_frame)
                        while len(state.frame_buffer) > int(buffer_limit):
                            state.frame_buffer.pop(0)
                        state.buffer_limit = int(buffer_limit)
                    except Exception:
                        pass
                
                if pipeline_b:
                    buffer_limit_b = getattr(state_b, 'buffer_limit', 10)
                    for effect_config in pipeline_b:
                        node_params = dict(effect_config.get("params", {}) or {})
                        try:
                            if 'frame_buffer_size' in node_params:
                                buffer_limit_b = int(node_params.get('frame_buffer_size', buffer_limit_b) or buffer_limit_b)
                        except Exception:
                            pass

                    (
                        frame_b,
                        state_b.previous_frame,
                        state_b.feedback_image,
                        state_b.trail_image,
                    ) = execute_flat_node_pipeline(
                        pipeline_b,
                        secondary_source_frame,
                        secondary_frame=secondary_source_frame,
                        extra_params=extra_params_with_frame,
                        previous_frame=state_b.previous_frame,
                        feedback_image=state_b.feedback_image,
                        trail_image=state_b.trail_image,
                        frame_buffer=state_b.frame_buffer,
                        worker_state=state_b,
                        fallback_size=(fallback_width, fallback_height),
                        copy_stats=frame_copy_stats,
                        source_copy_policy=source_copy_policy,
                    )
                    try:
                        history_frame_b = frame_b.copy()
                        record_frame_copy(frame_copy_stats, "history", history_frame_b)
                        state_b.frame_buffer.append(history_frame_b)
                        while len(state_b.frame_buffer) > int(buffer_limit_b):
                            state_b.frame_buffer.pop(0)
                        state_b.buffer_limit = int(buffer_limit_b)
                    except Exception:
                        pass
                
                if processing_pipeline or pipeline_b:
                    # Blend the decks
                    current_glitched_frame = blend_decks(frame_a, frame_b, master_mix, master_transition, fallback_size=(fallback_width, fallback_height))
                else:
                    # Fallback: merge all params and apply monolithically
                    if explicit_params is not None:
                        merged_params = explicit_params
                    else:
                        merged_params = {}
                    if isinstance(processing_pipeline, list):
                        for effect_config in processing_pipeline:
                            if effect_config.get("effect_type") == "Display":
                                continue
                            effect_params = effect_config.get("params", {}) or {}
                            merged_params.update(effect_params)
                    if isinstance(extra_params, dict):
                        try:
                            merged_params.update(extra_params)
                        except Exception:
                            pass
                    glitched, new_prev, new_feed, new_trail = apply_glitch_effect(
                        current_glitched_frame, merged_params, current_previous_frame, current_feedback_image, current_trail_image, current_frame_buffer
                    )
                    current_glitched_frame = glitched
                    current_previous_frame = new_prev
                    current_feedback_image = new_feed
                    current_trail_image = new_trail

                # Effects may return PIL or non-contiguous arrays, but they may
                # not change the spatial contract.  Normalize representation
                # and reject any hidden resize before state is persisted.
                current_glitched_frame = validate_master_frame(
                    current_glitched_frame,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="worker master output",
                )
                frame_a = validate_master_frame(
                    frame_a,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="deck A output",
                )
                frame_b = validate_master_frame(
                    frame_b,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="deck B output",
                )
                current_previous_frame = validate_master_frame(
                    current_previous_frame,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="previous-output buffer",
                )
                if current_previous_frame is None:
                    current_previous_frame = np.zeros(frame_shape, dtype=np.uint8)
                current_feedback_image = validate_master_frame(
                    current_feedback_image,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="feedback buffer",
                )
                if current_feedback_image is None:
                    current_feedback_image = np.zeros(frame_shape, dtype=np.uint8)
                current_trail_image = validate_master_frame(
                    current_trail_image,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="trail buffer",
                )
                if current_trail_image is None:
                    current_trail_image = np.zeros(frame_shape, dtype=np.uint8)
                state_b.previous_frame = validate_master_frame(
                    state_b.previous_frame,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="deck B previous-output buffer",
                )
                if state_b.previous_frame is None:
                    state_b.previous_frame = np.zeros(frame_shape, dtype=np.uint8)
                state_b.feedback_image = validate_master_frame(
                    state_b.feedback_image,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="deck B feedback buffer",
                )
                if state_b.feedback_image is None:
                    state_b.feedback_image = np.zeros(frame_shape, dtype=np.uint8)
                state_b.trail_image = validate_master_frame(
                    state_b.trail_image,
                    master_resolution,
                    role=FrameRole.MASTER_PROGRAM,
                    name="deck B trail buffer",
                )
                if state_b.trail_image is None:
                    state_b.trail_image = np.zeros(frame_shape, dtype=np.uint8)

                # Persist state across frames
                state.previous_frame = current_previous_frame
                state.feedback_image = current_feedback_image
                state.trail_image = current_trail_image
                state.assert_master_integrity()
                state_b.assert_master_integrity()
                
                # Increment frame counter for consistent temporal effects
                state.frame_count += 1
                state_b.frame_count += 1

                # CRITICAL: Ensure final output is numpy array, no PIL Images in worker output
                assert isinstance(current_glitched_frame, np.ndarray), \
                    f"Worker output must be np.ndarray, got {type(current_glitched_frame)}"
                assert current_glitched_frame.dtype == np.uint8, \
                    f"Worker output must be uint8, got {current_glitched_frame.dtype}"

                temporal_inventory_a = state.temporal_buffer_inventory()
                temporal_inventory_b = state_b.temporal_buffer_inventory()
                multimodal_telemetry = state.effect_states.get(
                    "multimodal_telemetry"
                )
                feedback_telemetry = state.effect_states.get(
                    "controlled_feedback_telemetry"
                )
                try:
                    from dataclasses import asdict, is_dataclass

                    if is_dataclass(multimodal_telemetry):
                        multimodal_telemetry = asdict(multimodal_telemetry)
                    elif multimodal_telemetry is not None:
                        multimodal_telemetry = dict(multimodal_telemetry)
                    if is_dataclass(feedback_telemetry):
                        feedback_telemetry = asdict(feedback_telemetry)
                    elif feedback_telemetry is not None:
                        feedback_telemetry = dict(feedback_telemetry)
                except Exception:
                    multimodal_telemetry = None
                    feedback_telemetry = None

                # Enqueue result with timestamp for latency tracking
                result_data = {
                    "frame": current_glitched_frame,
                    "frame_a": frame_a,
                    "frame_b": frame_b,
                    "timestamp": source_item.get("timestamp", time.perf_counter()),
                    "frame_copy_stats": frame_copy_stats,
                    "render_mode": render_mode.value,
                    "frame_role": FrameRole.MASTER_PROGRAM.value,
                    "master_resolution": master_resolution.to_dict(),
                    "history_buffer_resolution": master_resolution.to_dict(),
                    "measured_temporal_inventory": {
                        "deck_a": temporal_inventory_a,
                        "deck_b": temporal_inventory_b,
                    },
                    "multimodal_telemetry": multimodal_telemetry,
                    "controlled_feedback_telemetry": feedback_telemetry,
                    "history_depth": len(state.frame_buffer),
                    "random_seed": source_item.get("random_seed", benchmark_seed),
                    "frame_index": source_item.get("frame_index"),
                    "render_time_ms": (time.perf_counter() - render_started) * 1000.0,
                    "temporal_resolution_resets": int(state.resolution_change_count),
                }
                if isinstance(genesis_debug_request, dict):
                    debug_node_id = genesis_debug_request.get("node_id")
                    debug_output = str(genesis_debug_request.get("output") or "priority_mask")
                    debug_cache_key = f"genesis_outputs::{debug_node_id}"
                    debug_outputs = state.effect_states.get(debug_cache_key, {}) if hasattr(state, "effect_states") else {}
                    debug_frame = debug_outputs.get(debug_output)
                    if debug_frame is not None:
                        result_data["genesis_debug_node_id"] = debug_node_id
                        result_data["genesis_debug_output"] = debug_output
                        result_data["genesis_debug_frame"] = np.asarray(debug_frame, dtype=np.uint8).copy()
                
                if shared_transport is not None:
                    canonical_metadata = {
                        key: value for key, value in result_data.items()
                        if key not in {"frame", "frame_a", "frame_b"}
                    }
                    shared_transport.publish_result(
                        glitched_queue, current_glitched_frame, canonical_metadata
                    )
                else:
                    # Latest-frame behavior: if output queue is full, drop oldest
                    try:
                        glitched_queue.put_nowait(result_data)
                    except queue.Full:
                        # Queue full - drop oldest and enqueue latest
                        try:
                            glitched_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            glitched_queue.put_nowait(result_data)
                        except queue.Full:
                            pass  # Still full, drop this frame
        except Exception as e:
            logging.error(f"Error in glitch_process: {e}")
        finally:
            if shared_transport is not None:
                for slot in shared_input_slots:
                    try:
                        shared_transport.release_input_slot(slot)
                    except Exception:
                        logging.debug("Failed to release shared input slot", exc_info=True)

    for worker_frame_state in (state, state_b):
        worker_frame_state.reset()
        resource = getattr(worker_frame_state, "lua_gl_input_tex", None)
        if resource is not None:
            try:
                resource.release()
            except Exception:
                pass
        context = getattr(worker_frame_state, "lua_gl_ctx", None)
        if context is not None:
            try:
                context.release()
            except Exception:
                pass
    if shared_transport is not None:
        shared_transport.close()

class MegaGlitchFMApp:
    def __init__(self):
        self.display_width = 1200
        self.display_height = 800
        self.image = None
        self.original_image_with_alpha = None  # Store original for BG color changes
        self.gif_frames = []
        self.gif_durations = []
        self.gif_index = 0
        self.gif_running = False
        self.last_gif_frame_time = time.time()
        self.video_reader = None
        self.primary_media_path = None
        self.video_playing = False
        self.video_refresh_requested = False
        self._primary_video_last_frame = None
        self._primary_video_last_frame_index = None
        self._video_seek_lock = threading.Lock()
        self._pending_video_seek_seconds = None
        self._pending_video_refresh = False
        self.srt_shader_cues = []
        self.srt_shader_path = ""
        self.srt_shader_enabled = True
        self.srt_shader_last_change_key = None
        self.srt_shader_default_preset = None
        self.window_capture_running = False
        self.selected_window = None
        self._window_capture_start_pending = False
        self._window_capture_window = None
        self._window_capture_sct = None
        self._window_capture_last_resolve = 0.0
        self._window_capture_last_bbox = None
        self._window_capture_last_ms = 0.0
        self._window_capture_fps_samples = 0
        self._window_capture_fps_last_time = time.perf_counter()
        self.last_window_capture_image = None
        self.webcam_capture = None
        self.webcam_running = False
        self.webcam_index = 0
        self.webcam_device_options = []
        self.source_selector_mode = "Image"
        self.processing_scale = 1.0
        self.adaptive_processing_scale = True
        self.adaptive_processing_min_scale = 0.2
        self.adaptive_processing_max_scale = 1.0
        self.adaptive_processing_target_latency_ms = 0.0
        self._adaptive_latency_ema_ms = None
        self._adaptive_last_adjust_time = 0.0
        self._adaptive_queue_pressure = 0
        self._adaptive_last_result_time = time.perf_counter()
        self._adaptive_pipeline_signature = None
        self.framerate = 30
        self._framerate_user_overridden = False
        # Time accumulator for animated effects and a user scale
        self.time_accum = 0.0
        self.time_scale = 1.0
        self.global_time_domain_macros = {
            param["name"]: 0.0
            for param in self._get_global_time_domain_param_defs()
        }
        self.global_time_domain_slider_tags = {}
        self.frame_counter = 0  # Frame counter for generators
        # Retro mode settings (from Display menu)
        self.retro_mode_enabled = False
        self.retro_bits_per_channel = 8
        self.retro_dither_enabled = False
        # GPU
        try:
            from megaglitchnode.opencv_cuda_bootstrap import has_cuda_build as bootstrap_cuda_build
            self.gpu_available, _ = bootstrap_cuda_build()
        except Exception:
            self.gpu_available = gpu_ops.is_cuda_available()
        self.use_cuda = False
        self._gpu_toggle_syncing = False
        self.virtual_cam = None
        self.virtual_cam_running = False
        self.virtual_cam_queue = queue.Queue(maxsize=1)
        self.virtual_cam_stop_event = threading.Event()
        self.virtual_cam_thread = None
        
        # OpenCV/CUDA backend state
        self.opencv_enabled = True  # Default enabled if available
        self.cuda_enabled = False  # Default disabled (manual opt-in)
        self._cuda_warning_shown = False
        
        # Latency-optimized queues (configurable maxsize)
        self.queue_max_size = 2  # Minimize latency: only 2 frames in flight
        self.raw_frame_queue = mp.Queue(maxsize=self.queue_max_size)
        self.glitched_frame_queue = mp.Queue(maxsize=self.queue_max_size)
        self.stop_glitch_process_event = mp.Event()
        self.shared_frame_transport = None
        self.processing_lifecycle = None
        self.glitch_process = None
        self._processing_failure_reported = False
        
        # Latency tracking
        self.latency_samples = []
        self.latency_report_interval = 120  # frames
        self.processed_fps_samples = 0
        self.processed_fps_last_time = time.perf_counter()
        
        # Recording (silent)
        self.is_recording_gif = False
        self.is_recording_mp4 = False
        self.recording_frames = []
        self.export_duration = 10.0  # Default 10 seconds
        self.recording_start_time = None
        self.recording_elapsed = 0.0
        self.last_recorded_frame = None
        self.recording_frame_times = []  # Track when each frame was captured
        self.recording_target_frames = 0  # Total frames needed for recording
        self.recording_frame_count = 0  # Current frame count during recording
        self.use_batch_recording = True  # Use fast parallel batch processing by default
        self.cancel_render_requested = False
        # Deterministic quality-first export.  These settings are serialized in
        # dual-deck session presets and never affect the live graph.
        self.offline_source_fps_auto = True
        self.offline_source_fps = 30.0
        self.offline_source_range = "entire_source"
        self.offline_output_fps = 55.0
        self.offline_fps_semantics = "true_accelerated"
        self.offline_master_codec = "h264"
        self.offline_master_crf = 16
        self.offline_master_preset = "slow"
        self.offline_random_seed = 0
        self.offline_ffmpeg_executable = "ffmpeg"
        self.offline_ffprobe_executable = "ffprobe"
        self.offline_analyze_source_audio = True
        self.offline_audio_sample_rate = 48000
        self.codec_post_enabled = False
        self.codec_post_command = (
            'pframe-mosh --input "{prepared}" --output "{corrupted}" --seed {seed}'
        )
        self.codec_post_work_directory = "build/codec-domain"
        self.codec_post_seed = 0
        self.codec_post_long_gop = 120
        self.codec_post_preparation_quality = 2
        self.codec_post_final_codec = "h264"
        self.codec_post_final_crf = 16
        self.codec_post_final_preset = "slow"
        self._offline_master_settings_open = False
        self.last_offline_master_telemetry = None
        # Modulation Matrix (non-audio)
        self.mod_matrix = None
        self.parameter_timeline = None
        self.audio_midi = None
        self.bpm_detector = None
        self.latest_audio_bands = {}
        self.latest_onset_value = 0.0
        self.latest_onset_event_id = 0
        self.audio_input_devices = []
        self.audio_input_device_map = {"System Default": None}
        self.audio_input_combo_tag = "audio_input_device_combo"
        self.audio_input_status_tag = "audio_input_status"
        self.media_audio = MediaAudioController(sample_rate=48000, block_size=2048)
        self.media_audio_control_tags = {}
        self.media_audio_mod_source_tags = {}
        self.media_audio_seq_tags = {}
        self.media_audio_seq_step_tags = {}
        self.media_audio_cdp_tags = {}
        self.media_audio_midi_input_tag = "media_audio_midi_input_combo"
        self.media_audio_midi_status_tag = "media_audio_midi_status"
        self.media_audio_last_status = None
        self.media_audio_last_ir_status = None
        self.media_audio_ir_max_seconds = 4.0
        self.media_audio_ir_normalize = True
        # Live Generative Sequencer: rendered frames feed VisualBus; the clocked
        # sequencer reads snapshots from that bus and owns MIDI note output.
        self.visual_bus = VisualBus()
        self.visual_analyzer_worker = VisualAnalyzerWorker(
            self.visual_bus,
            on_metrics=self.media_audio.set_live_metrics,
        )
        self.midi_device_manager = MidiDeviceManager()
        self.live_sequencer = LiveSequencer(self.visual_bus, self.midi_device_manager)
        self.mpd26_controls = AkaiMPD26Controls(self.live_sequencer)
        self.hardware_mapping_manager = None
        self.midi_mapping_status_tag = "midi_mapping_status"
        self.live_sequencer_ui = None
        # Navigation
        self.pipeline_navigator_visible = False
        # Genesis auxiliary buffer preview
        self.genesis_aux_window_visible = False
        self.genesis_aux_preview_size = (384, 240)
        self.genesis_aux_texture_tag = None
        self.genesis_aux_output_descriptions = {
            "final_frame": "The finished Genesis frame after CRT/composite finishing.",
            "plane_a": "Foreground-biased tile plane before sprite composition.",
            "plane_b": "Background-biased tile plane before sprite composition.",
            "sprite_layer": "Sprite-only buffer with current sprite placement and glitching.",
            "priority_mask": "Priority routing mask used to decide front/back layering.",
            "palette_index_buffer": "Palette-bank/index field for palette-driven routing or inspection.",
            "luma_mask": "Luminance-derived mask from the pre-CRT scene.",
            "corruption_mask": "Regions currently affected by corruption or instability.",
            "scanline_mod_buffer": "Scanline modulation field used for raster-driven motion.",
            "crt_prepass": "Scene before CRT/composite finish is applied.",
        }
        self.genesis_aux_node_map = {}
        self.genesis_aux_selected_node_id = None
        self.genesis_aux_selected_output = "priority_mask"
        self.genesis_aux_false_color = True
        # Image background color for transparent images/vectors
        self.image_bg_color = (0, 0, 0, 255)  # Default black RGBA
        self.bg_media = None  # Background image/video/gif
        self.bg_gif_frames = []
        self.bg_gif_index = 0
        self.bg_video_reader = None
        self.background_media_path = None
        self.image_bg_media_opacity = 0.75
        self.mixer_enabled = False
        self.mixer_mix = 0.5
        self.mixer_mode = "Crossfade"
        self.mixer_channel_b_image = None
        self.mixer_channel_b_gif_frames = []
        self.mixer_channel_b_gif_durations = []
        self.mixer_channel_b_gif_index = 0
        self.mixer_channel_b_last_gif_frame_time = time.time()
        self.mixer_channel_b_video_reader = None
        self.mixer_channel_b_media_path = None
        self.mixer_channel_b_video_refresh_requested = False
        self.mixer_channel_b_last_video_frame = None
        self.mixer_channel_b_label = "No Channel B media loaded"
        self.processing_paused = False  # Pause heavy processing during dialogs
        self.batch_rendering = False  # Flag to prevent display updates during batch rendering
        self.setup_ui()
        self.visual_analyzer_worker.start()
        self.live_sequencer.start()

    
    
    def generate_node_chain(self, sender=None, app_data=None, user_data=None):
        import random
        import dearpygui.dearpygui as dpg
        import effect_nodes
        import effect_nodes_cuda
        from node_editor import SourceNode, DisplayNode, SecondarySourceNode
        
        complexity = dpg.get_value("ml_rand_complexity")
        profile = dpg.get_value("ml_rand_profile")
        editor = self.get_active_editor()
        
        nodes_to_remove = [n for n in editor.nodes if not isinstance(n, (SourceNode, DisplayNode, SecondarySourceNode))]
        for node in nodes_to_remove:
            try:
                dpg.delete_item(node.node_tag)
            except Exception: pass
            if node in editor.nodes:
                editor.nodes.remove(node)
                
        editor.clear_all_links()
        
        source = editor.get_source_node()
        
        complexity = dpg.get_value("ml_rand_complexity")
        profile = dpg.get_value("ml_rand_profile")

        available_nodes = [
            "PixelSortingNode", "DatabendingNode", "DataMoshingNode", "PFrameGlitchesNode",
            "ScanlineShuffleNode", "ChromaLeakNode", "ChannelFlowOffsetNode",
            "PixelationChromaticNode", "ColorSpaceCorruptionNode", "CRTEmulationNode",
            "FeedbackTrailNode", "FrameDropRGBSplitNode", "CUDAFlowWarpNode",
            "CUDAChannelDriftNode", "CUDADigitalMeltNode", "CUDABitplaneDriftNode",
            "CUDATemporalMedianNode", "CUDASpectralDatamoshNode",
            "CUDARecursiveFractalCorrupterNode", "CUDAVoronoiShatterNode",
            "CUDAReactionDiffusionNode", "CUDALumaKeyedFluidDynamicsNode",
        ]
        
        if "Delta" in profile:
            available_nodes = [
                "BlurSharpenNode", "ColorAdjustNode", "BrightnessContrastNode",
                "BloomNode", "EdgeEmbossNode", "VignetteNode", "CUDATemporalMedianNode",
                "CUDAChannelDriftNode"
            ]
        elif "Alpha" in profile:
            available_nodes = [
                "PixelSortingNode", "EnhancedPixelSortingNode", "CRTEmulationNode",
                "ColorSpaceCorruptionNode", "CUDAFlowWarpNode", "CUDADigitalMeltNode"
            ]
        elif "Gamma" in profile:
            available_nodes = [
                "FeedbackTrailNode", "DatabendingNode", "DataMoshingNode",
                "ScanlineShuffleNode", "CUDASpectralDatamoshNode", "CUDABitplaneDriftNode",
                "CUDARecursiveFractalCorrupterNode", "CUDAVoronoiShatterNode",
                "CUDAReactionDiffusionNode", "CUDALumaKeyedFluidDynamicsNode",
            ]

        def node_class_exists(node_type):
            node_cls = getattr(effect_nodes, node_type, None) or getattr(effect_nodes_cuda, node_type, None)
            return isinstance(node_cls, type) and issubclass(node_cls, effect_nodes.GlitchEffectNode)

        available_nodes = [node_type for node_type in available_nodes if node_class_exists(node_type)]
        if not available_nodes:
            logging.warning("Generative node builder found no valid single-input effect nodes")
            return

        layout_top = 80
        source_x = 80
        effect_x = 380
        display_x = 680
        vertical_spacing = 150

        dpg.set_item_pos(source.node_tag, [source_x, layout_top])
            
        prev_node = source
        created_nodes = []
        for i in range(complexity):
            node_type = random.choice(available_nodes)
            new_node = editor.add_node(node_type)
            if new_node is None:
                logging.warning(f"Generative node builder skipped unavailable node type: {node_type}")
                continue
            
            x_pos = effect_x
            y_pos = layout_top + (len(created_nodes) * vertical_spacing)
            dpg.set_item_pos(new_node.node_tag, [x_pos, y_pos])
            
            try:
                out_tag = list(prev_node.output_attrs.values())[0]
                in_tag = list(new_node.input_attrs.values())[0]
                editor.link_callback(editor.node_editor_tag, [out_tag, in_tag])
            except Exception: pass
            
            prev_node = new_node
            created_nodes.append(new_node)
            
        try:
            out_tag = list(prev_node.output_attrs.values())[0]
            in_tag = list(display.input_attrs.values())[0]
            editor.link_callback(editor.node_editor_tag, [out_tag, in_tag])
        except Exception: pass
        
        display_y = layout_top
        if created_nodes:
            display_y = layout_top + ((len(created_nodes) - 1) * vertical_spacing // 2)
        dpg.set_item_pos(display.node_tag, [display_x, display_y])
        
        # Mute severity a little for generation so it's not totally broken
        orig_sev = dpg.get_value("ml_rand_severity")
        dpg.set_value("ml_rand_severity", orig_sev * 0.5)
        self.trigger_ml_mutation()
        dpg.set_value("ml_rand_severity", orig_sev)
        
    def trigger_ml_mutation(
self, sender=None, app_data=None, user_data=None):
        import random
        import dearpygui.dearpygui as dpg
        
        chance = dpg.get_value("ml_rand_chance")
        severity = dpg.get_value("ml_rand_severity")
        profile = dpg.get_value("ml_rand_profile")
        
        # Adjust severity/chance based on profile
        profile_mult = 1.0
        if "Delta" in profile: profile_mult = 0.5
        elif "Beta" in profile: profile_mult = 1.5
        elif "Gamma" in profile: profile_mult = 2.5
        
        actual_chance = min(1.0, chance * profile_mult)
        actual_severity = min(1.0, severity * profile_mult)
        
        if dpg.get_value("ml_rand_nodes"):
            editor = self.get_active_editor()
            for node in getattr(editor, "nodes", []):
                params = getattr(node, "params", None)
                if not params:
                    params = getattr(node, "params_def", None)
                if params and hasattr(node, "param_values"):
                    if isinstance(params, dict):
                        params = params.values()
                    for param_def in params:
                        if random.random() < actual_chance:
                            p_name = param_def.get("name")
                            p_type = param_def.get("type", "float")
                            if p_type in ["float", "int"]:
                                p_min = param_def.get("min_value", 0.0)
                                p_max = param_def.get("max_value", 1.0)
                                current_val = node.param_values.get(p_name, param_def.get("default_value"))
                                
                                # Mutate
                                range_span = p_max - p_min
                                delta = (random.random() * 2.0 - 1.0) * range_span * actual_severity
                                new_val = current_val + delta
                                
                                # Clamp
                                new_val = max(p_min, min(p_max, new_val))
                                if p_type == "int": new_val = int(new_val)
                                
                                node.param_values[p_name] = new_val
                                if hasattr(node, "param_controls") and p_name in node.param_controls:
                                    try:
                                        dpg.set_value(node.param_controls[p_name], new_val)
                                    except Exception: pass
                            elif p_type == "bool":
                                node.param_values[p_name] = not node.param_values.get(p_name, False)
                                if hasattr(node, "param_controls") and p_name in node.param_controls:
                                    try:
                                        dpg.set_value(node.param_controls[p_name], node.param_values[p_name])
                                    except Exception: pass
                                    
        if dpg.get_value("ml_rand_time"):
            if random.random() < actual_chance:
                # Mutate time_scale or local_time_warp
                macro = random.choice(["time_scale", "local_time_warp", "clock_quantization", "sync_desync"])
                if macro == "time_scale":
                    val = self.time_scale + (random.random()*2-1) * actual_severity * 2.0
                    self.time_scale = max(0.0, min(5.0, val))
                    if hasattr(self, "time_scale_slider"):
                        try: dpg.set_value(self.time_scale_slider, self.time_scale)
                        except Exception: pass
                # (Can expand to update other macros if UI elements are mapped)
                
        if dpg.get_value("ml_rand_mod"):
            if hasattr(self, "mod_matrix"):
                for conn in self.mod_matrix.connections:
                    if random.random() < actual_chance:
                        conn["depth"] += (random.random()*2-1) * actual_severity
                        conn["depth"] = max(-1.0, min(1.0, conn["depth"]))
                # Need to refresh UI if mod matrix is open, but depth is applied in backend anyway
        
        # Reset worker state to prevent hard jumps if we mutated pipeline params severely
        if hasattr(self, "raw_frame_queue"):
            try: self.raw_frame_queue.put("RESET")
            except Exception: pass

    def get_active_editor(self):
        import dearpygui.dearpygui as dpg
        try:
            if hasattr(self, 'deck_tab_bar') and dpg.does_item_exist(self.deck_tab_bar):
                active_tab = dpg.get_value(self.deck_tab_bar)
                if active_tab == self.deck_b_tab:
                    return self.node_editor_b
        except Exception:
            pass
        return self.node_editor_a

    def _deck_b_is_active(self):
        return self.get_active_editor() is self.node_editor_b
        
    def _get_both_node_params(self):
        params_a = self.node_editor_a.get_node_params()
        params_b = self.node_editor_b.get_node_params()
        return {**params_a, **params_b}

    def setup_ui(self):
        dpg.create_context()
        ui_theme.apply_modern_theme()
        self.node_editor_a = NodeEditor(primary_source_key="primary")
        self.node_editor_b = NodeEditor(primary_source_key="secondary")
        self.hardware_mapping_manager = HardwareMappingManager(
            (("Deck A", self.node_editor_a), ("Deck B", self.node_editor_b))
        )
        # New sessions start on the quality-first Deck A graph. Keep Deck B out
        # of the master until the user deliberately mixes it in.
        self.master_mix = 0.0
        self.master_transition = "Normal Mix" 
        self.genesis_aux_texture_tag = dpg.generate_uuid()
        # Connect node_editor to reset queue (for state reset on graph changes)
        self.node_editor_a.reset_queue = self.raw_frame_queue
        self.node_editor_b.reset_queue = self.raw_frame_queue
        dpg.create_viewport(title="MegaGlitchFM Node Editor", width=self.display_width, height=self.display_height)
        dpg.setup_dearpygui()
        dpg.show_viewport()

        with dpg.texture_registry() as reg_id:
            self.texture_registry = reg_id
            aux_w, aux_h = self.genesis_aux_preview_size
            empty_aux = np.zeros((aux_h, aux_w, 4), dtype=np.float32).flatten()
            dpg.add_dynamic_texture(aux_w, aux_h, empty_aux, tag=self.genesis_aux_texture_tag, parent=reg_id)

        with dpg.window(label="Mega Glitch FM Node Editor", width=self.display_width, height=self.display_height, tag="main_window"):
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Select Source...", callback=self.show_source_selector)
                    dpg.add_menu_item(label="Open Image", callback=self.open_image)
                    dpg.add_menu_item(label="Open GIF", callback=self.open_gif)
                    dpg.add_menu_item(label="Open Video", callback=self.open_video)
                    dpg.add_menu_item(label="Load SRT Shader Cues", callback=self.open_srt_shader_cues)
                    dpg.add_menu_item(label="Start Webcam Capture", callback=self.start_webcam_capture)
                    dpg.add_menu_item(label="Stop Webcam Capture", callback=self.stop_webcam_capture)
                    dpg.add_menu_item(label="Start Window Capture", callback=self.start_window_capture)
                    dpg.add_menu_item(label="Stop Window Capture", callback=self.stop_window_capture)
                    dpg.add_menu_item(label="Stop All Media", callback=self.stop_all_media)
                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(label="Copy Selected Node", callback=lambda: self.get_active_editor().copy_selected_node())
                    dpg.add_menu_item(label="Delete Selected Node", callback=lambda: self.get_active_editor().delete_selected_node())
                    dpg.add_menu_item(label="Delete Selected Links", callback=lambda: self.get_active_editor().delete_selected_links())
                    dpg.add_menu_item(label="Clear All Links", callback=lambda: self.get_active_editor().clear_all_links())
                with dpg.menu(label="Display"):
                    self.display_width_input = dpg.add_input_int(label="Width", default_value=self.display_width, min_value=100, max_value=3840)
                    self.display_height_input = dpg.add_input_int(label="Height", default_value=self.display_height, min_value=100, max_value=2160)
                    dpg.add_button(label="Apply Resolution", callback=self.apply_display_resolution)
                    self.processing_scale_slider = dpg.add_slider_float(label="Preview Scale (egress only)", min_value=0.1, max_value=1.0, default_value=self.processing_scale, callback=self._set_processing_scale, user_data=None)
                    dpg.add_checkbox(label="Adaptive Preview Scale", default_value=self.adaptive_processing_scale, callback=self._toggle_adaptive_processing_scale)
                    self.framerate_slider = dpg.add_slider_int(label="Framerate", min_value=1, max_value=60, default_value=self.framerate, callback=self._set_preview_framerate, user_data=None)
                    # Global time scale for animated effects (e.g., Noise Warp)
                    self.time_scale_slider = dpg.add_slider_float(label="Time Scale", min_value=0.0, max_value=5.0, default_value=self.time_scale, callback=lambda s,a,u: setattr(self, 'time_scale', a), user_data=None)
                    dpg.add_separator()
                    dpg.add_text("Image Background:")
                    dpg.add_combo(label="BG Preset", items=["Black", "White", "Transparent", "Custom", "Media"], 
                                 default_value="Black", callback=self.change_image_background, width=150, tag="bg_preset_combo")
                    dpg.add_color_edit(label="Custom BG", default_value=(0, 0, 0, 255), 
                                      tag="custom_bg_color", callback=self.update_custom_bg_color, width=150,
                                      no_alpha=False, display_mode=dpg.mvColorEdit_rgb)
                    dpg.add_slider_float(label="Image Over BG Opacity", min_value=0.0, max_value=1.0,
                                         default_value=self.image_bg_media_opacity,
                                         callback=self.update_bg_media_opacity, width=150,
                                         tag="bg_media_opacity")
                    dpg.add_button(label="Load BG Media", callback=self.open_bg_media, width=150)
                    dpg.add_text("No BG media loaded", tag="bg_media_status", color=(150, 150, 150))
                    dpg.add_separator()
                    dpg.add_text("2-Channel Mixer:")

                    dpg.add_checkbox(label="Enable Mixer", default_value=False, tag="mixer_enabled_checkbox",
                                     callback=lambda s, a: setattr(self, 'mixer_enabled', bool(a)))
                    dpg.add_slider_float(label="Channel B Mix", min_value=0.0, max_value=1.0, default_value=self.mixer_mix,
                                         callback=lambda s, a: setattr(self, 'mixer_mix', float(a)), width=150)
                    dpg.add_combo(label="Mixer Mode",
                                  items=["Crossfade", "Add", "Multiply", "Screen", "Difference", "Lighten", "Darken"],
                                  default_value=self.mixer_mode,
                                  callback=lambda s, a: setattr(self, 'mixer_mode', str(a)),
                                  width=150)
                    dpg.add_button(label="Load Channel B", callback=self.open_mixer_channel_b, width=150)
                    dpg.add_button(label="Clear Channel B", callback=self.clear_mixer_channel_b, width=150)
                    dpg.add_text(self.mixer_channel_b_label, tag="mixer_channel_b_status", color=(150, 150, 150))
                    dpg.add_separator()
                    # Retro Mode / Bit Depth Controls
                    dpg.add_text("Retro Mode (Bit Depth):")
                    self.retro_mode_checkbox = dpg.add_checkbox(
                        label="Enable Bit Reduction", 
                        default_value=False, 
                        callback=lambda s, v: setattr(self, 'retro_mode_enabled', v)
                    )
                    self.bits_per_channel_slider = dpg.add_slider_int(
                        label="Bits per Channel", 
                        default_value=8, 
                        min_value=1, 
                        max_value=8, 
                        callback=lambda s, v: setattr(self, 'retro_bits_per_channel', v),
                        width=150
                    )
                    self.retro_dither_checkbox = dpg.add_checkbox(
                        label="Dither", 
                        default_value=False, 
                        callback=lambda s, v: setattr(self, 'retro_dither_enabled', v)
                    )
                    dpg.add_separator()
                    # Visible checkbox inside the menu for clarity (disabled if CUDA not available)
                    self.cuda_checkbox = dpg.add_checkbox(label="Use GPU (CUDA)", default_value=False, callback=self.toggle_gpu, enabled=self.gpu_available)
                    # GPU toggle as a checkable menu item (works inside menu bars)
                    self.cuda_menu_item = dpg.add_menu_item(label="Use GPU (CUDA)", check=True, default_value=False, callback=self.toggle_gpu, enabled=self.gpu_available)
                    # If CUDA isn't available, add a helpful tooltip explaining why the toggle is disabled
                    if not self.gpu_available:
                        try:
                            with dpg.tooltip(parent=self.cuda_checkbox):
                                dpg.add_text("CUDA not available: install a CUDA-enabled OpenCV build to use GPU.")
                        except Exception:
                            pass
                with dpg.menu(label="Virtual Cam"):
                    cam_checkbox = dpg.add_checkbox(label="Enable Virtual Camera", default_value=False, callback=self.toggle_virtual_camera)
                    with dpg.tooltip(parent=cam_checkbox):
                        dpg.add_text("Note: If using OBS Virtual Camera, OBS hides it from its own sources to prevent loops.\nTo capture this in OBS, use a 'Window Capture' source instead, or install 'Unity Capture'.")
                with dpg.menu(label="Export"):
                    self.export_duration_slider = dpg.add_slider_float(
                        label="Duration (sec)",
                        default_value=self.export_duration,
                        min_value=1.0,
                        max_value=60.0,
                        callback=self._on_export_duration_slider,
                        width=150,
                    )
                    self.export_duration_input = dpg.add_input_text(
                        label="Export Length",
                        default_value=self._format_timecode(self.export_duration),
                        callback=self._on_export_duration_text,
                        on_enter=True,
                        width=150,
                    )
                    dpg.add_checkbox(label="Batch Mode (Fast)", default_value=True, 
                                   callback=lambda s, a: setattr(self, 'use_batch_recording', a))
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Offline Master Settings...",
                        callback=self.show_offline_master_settings,
                    )
                    dpg.add_menu_item(
                        label="Render Offline Master...",
                        callback=self.render_offline_master,
                    )
                    dpg.add_separator()
                    dpg.add_menu_item(label="Render GIF", callback=self.batch_render_gif)
                    dpg.add_menu_item(label="Render Legacy Batch MP4", callback=self.batch_render_mp4)
                    dpg.add_menu_item(label="Cancel Active Render", callback=self.cancel_active_render)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Start Realtime Recording", callback=self.start_gif_recording)
                    dpg.add_menu_item(label="Stop && Save", callback=self.stop_and_save_gif)
                with dpg.menu(label="Modulation"):
                    dpg.add_menu_item(label="Open Matrix", callback=self.open_modulation_matrix)
                    dpg.add_menu_item(label="Parameter Timeline", callback=self.open_parameter_timeline)
                    dpg.add_menu_item(label="Live Sequencer", callback=self.open_live_sequencer_tab)
                    dpg.add_separator()
                    dpg.add_menu_item(label="MIDI Mapping Mode", check=True, callback=self.toggle_midi_mapping_mode)
                    dpg.add_menu_item(label="Clear MIDI Mappings", callback=self.clear_midi_mappings)
                    dpg.add_text("MIDI mapping mode off", tag=self.midi_mapping_status_tag)
                with dpg.menu(label="Audio"):
                    dpg.add_combo(
                        label="Input Source",
                        items=["System Default"],
                        default_value="System Default",
                        tag=self.audio_input_combo_tag,
                        callback=self.on_audio_input_selected,
                        width=320,
                    )
                    dpg.add_menu_item(label="Refresh Input Sources", callback=self.refresh_audio_input_devices)
                    dpg.add_menu_item(label="Restart Audio Input", callback=self.restart_audio_input)
                    dpg.add_text("Audio input: System Default", tag=self.audio_input_status_tag)
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Media-to-Audio Page...",
                        callback=self.open_media_audio_page,
                    )
                    dpg.add_menu_item(
                        label="Stop Media Audio",
                        callback=self.stop_media_audio,
                    )
                
                # Backend menu for OpenCV/CUDA controls
                with dpg.menu(label="Backend"):
                    dpg.add_menu_item(label="Backend Info...", callback=self.show_backend_info)
                    dpg.add_separator()
                    
                    # Check OpenCV availability
                    opencv_available = False
                    try:
                        import opencv_backend as ocv
                        opencv_available = ocv.is_cv2_available()
                    except:
                        pass
                    
                    # OpenCV toggle
                    self.opencv_toggle = dpg.add_checkbox(
                        label="Use OpenCV Backend",
                        default_value=self.opencv_enabled and opencv_available,
                        enabled=opencv_available,
                        callback=self.toggle_opencv_backend
                    )
                    if not opencv_available:
                        dpg.add_text("  (OpenCV not available)", color=(255, 100, 100), indent=20)
                    
                    # CUDA toggle
                    cuda_available = False
                    if opencv_available:
                        try:
                            from megaglitchnode.opencv_cuda_bootstrap import has_cuda_build as bootstrap_cuda_build
                            cuda_available, _ = bootstrap_cuda_build()
                        except:
                            pass
                    
                    self.cuda_toggle_menu = dpg.add_checkbox(
                        label="Use CUDA Acceleration (Experimental)",
                        default_value=False,
                        enabled=cuda_available,
                        callback=self.toggle_cuda_backend
                    )
                    if not cuda_available:
                        dpg.add_text("  (CUDA not available)", color=(255, 150, 100), indent=20)
                    
                    dpg.add_separator()
                    dpg.add_menu_item(label="Performance Settings...", callback=self.show_performance_settings)
                    dpg.add_menu_item(label="Run Backend Test", callback=self.run_backend_test)
                
                with dpg.menu(label="Presets"):
                    dpg.add_menu_item(label="Save Preset...", callback=self.show_save_preset_dialog)
                    dpg.add_menu_item(label="Load Preset...", callback=self.show_load_preset_dialog)
                    dpg.add_separator()
                    dpg.add_menu_item(
                        label="Load Quality-First Master Graph",
                        callback=self.load_quality_first_graph,
                    )
                    dpg.add_separator()
                    dpg.add_text("Presets save:")
                    dpg.add_text("  • Node types & connections", indent=10, bullet=False)
                    dpg.add_text("  • Parameter values", indent=10, bullet=False)
                    dpg.add_text("  • Node positions", indent=10, bullet=False)
                    dpg.add_text("  • Modulation matrix state", indent=10, bullet=False)
                with dpg.menu(label="View"):
                    dpg.add_menu_item(label="Pipeline Navigator", callback=self.toggle_pipeline_navigator)
                    dpg.add_menu_item(label="Genesis Aux Inspector", callback=self.toggle_genesis_aux_inspector)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Navigation Help", callback=self.show_navigation_help)
                with dpg.menu(label="Nodes"):
                    node_categories = {
                        "Analysis & Quality": [
                            "MultimodalAnalysisNode",
                            "OpticalFlowRemapNode",
                            "ControlledRecursiveFeedbackNode",
                            "QuantizationSolarizationNode",
                            "EdgeReconstructionNode",
                            "StructuralRestorationNode",
                        ],
                        "GPU Accelerated (CUDA)": [
                            "CUDAFlowWarpNode", "CUDASpectralDatamoshNode", "CUDAFlowPixelSortNode",
                            "CUDANonlinearRemapNode", "CUDASingularityEngineNode", "CUDAPersistentFeedbackNode", "CUDABitplaneDriftNode",
                            "CUDATemporalMedianNode", "CUDAAudioFlowNode", "CUDAVectorMoshNode", "CUDATemporalSlitScanNode",
                            "CUDARecursiveFractalCorrupterNode", "CUDAVoronoiShatterNode",
                            "CUDAReactionDiffusionNode", "CUDALumaKeyedFluidDynamicsNode",
                        ],
                        "Generators": [
                            "PerlinNoiseGeneratorNode", "GradientGeneratorNode", "PlasmaGeneratorNode",
                            "VoronoiGeneratorNode", "FractalGeneratorNode", "CheckerboardGeneratorNode",
                            "ScanFeedbackGeneratorNode", "XorLatticeGeneratorNode", "MacroblockEchoGeneratorNode",
                            "BubbleSimGeneratorNode", "BubbleSimGenerator2Node", "ColonyReefGeneratorNode", "GenesisVDPGeneratorNode",
                        ],
                        "Audio": [
                            "FMSynthNode",
                        ],
                        "Compositors": [
                            "GenesisPriorityCompositeNode",
                        ],
                        "Keyers / Masks": [
                            "GenesisMaskKeyNode", "LumaKeyNode",
                        ],
                        "Core": [
                            "GlitchAmountNode", "DatabendingNode", "DataMoshingNode", "PFrameGlitchesNode", "EnhancedDataMoshingNode",
                        ],
                        "Visual": [
                            "PixelSortingNode", "EnhancedPixelSortingNode", "RollingShutterWobbleNode", "ScanlineShuffleNode", "ChromaLeakNode",
                            "RowTemporalStutterNode", "PaletteCrunchNode", "ChannelFlowOffsetNode", "PixelationChromaticNode",
                            "ScanLinesNoiseWaveNode", "InvertColorNode", "LineGlitchNode", "SimpleDataMoshingIntensityNode",
                        ],
                        "Geometric": [
                            "MeshTearNode", "RotationKaleidoscopeNode", "UnfoldEuclidRandomNode", "BarrelDistortionNode", "SwirlNode",
                            "RippleNode", "KaleidoscopeNode", "NoiseWarpNode", "VectorscopeNode", "VertexShaderGlitchNode", "PerspectiveGlitchNode",
                        ],
                        "Temporal": [
                            "FeedbackTrailNode", "FrameDropRGBSplitNode", "FrameBufferNode", "FrameInterpolationGlitchNode",
                            "TemporalSlicingNode", "TimeDisplacementNode", "BufferUnderrunNode",
                        ],
                        "Analog & Hardware": [
                            "FFTPhaseScrambleNode", "VignetteNode", "HalftoneNode", "VHSJitterNode", "CRTEmulationNode",
                            "VHSDamageNode", "CRTPhosphorNode", "AnalogTVNode", "TapeDropoutNode",
                        ],
                        "Digital & Compression": [
                            "JPEGCompressionNode", "WebPCorruptionNode", "H264ArtifactsNode", "NetworkPacketLossNode",
                            "DigitalCLIPNode", "WaveletCorruptionNode", "ColorSpaceCorruptionNode", "LUTCorruptionNode", "SpectralShiftNode",
                        ],
                        "Generative & Algorithmic": [
                            "FractalNoiseNode", "CellularAutomataNode", "FaultlinePossessionNode", "LuaShaderNode",
                            "KaleidoFeedbackShaderNode", "AbyssalSignalFeedbackShaderNode",
                        ],
                        "Utility": [
                            "TextOverlayNode", "BrightnessContrastNode", "ColorAdjustNode", "BlurSharpenNode", "BloomNode",
                            "EdgeEmbossNode", "EffectBlenderNode", "ConditionalEffectNode", "FrameDifferenceNode", "FeedbackLoopNode",
                            "VideoMixerNode",
                            "WatercolorPosterizeNode",
                        ]
                    }

                    # Collect nodes from both effect_nodes and effect_nodes_cuda
                    all_nodes = {}
                    for module in [effect_nodes, effect_nodes_cuda]:
                        for name, obj in inspect.getmembers(module):
                            if inspect.isclass(obj) and (
                                (issubclass(obj, effect_nodes.GlitchEffectNode) or 
                                 issubclass(obj, effect_nodes.MultiInputNode) or
                                 issubclass(obj, effect_nodes.GeneratorNode) or
                                 obj is effect_nodes.VideoMixerNode) and 
                                obj not in (
                                    effect_nodes.GlitchEffectNode,
                                    effect_nodes.MultiInputNode,
                                    effect_nodes.Node,
                                    effect_nodes.GeneratorNode,
                                )
                            ):
                                all_nodes[name] = obj
                    
                    dpg.add_menu_item(label="SecondarySourceNode", callback=self.add_effect_node, user_data="SecondarySourceNode")
                    dpg.add_separator()
                    categorized_nodes = {node for category in node_categories.values() for node in category}
                    node_menu_labels = {
                        "GenesisVDPGeneratorNode": "Genesis VDP Gen",
                        "BubbleSimGeneratorNode": "Bubble Sim Gen",
                        "BubbleSimGenerator2Node": "Bubble Sim Gen (2)",
                        "FMSynthNode": "FM Synth",
                        "AbyssalSignalFeedbackShaderNode": "Abyssal Signal Feedback",
                        "KaleidoFeedbackShaderNode": "Kaleido Feedback",
                        "GenesisPriorityCompositeNode": "Genesis Priority Composite",
                        "GenesisMaskKeyNode": "Genesis Mask Key",
                        "CUDASingularityEngineNode": "SingularityEngine",
                    }

                    for category, nodes in node_categories.items():
                        with dpg.menu(label=category):
                            for node_name in nodes:
                                if node_name in all_nodes:
                                    dpg.add_menu_item(
                                        label=node_menu_labels.get(node_name, node_name),
                                        callback=self.add_effect_node,
                                        user_data=node_name,
                                    )

                    uncategorized_nodes = sorted([name for name in all_nodes if name not in categorized_nodes])
                    if uncategorized_nodes:
                        with dpg.menu(label="Uncategorized"):
                            for node_name in uncategorized_nodes:
                                dpg.add_menu_item(
                                    label=node_menu_labels.get(node_name, node_name),
                                    callback=self.add_effect_node,
                                    user_data=node_name,
                                )


            
            self.deck_tab_bar = dpg.add_tab_bar(parent="main_window")
            self.deck_a_tab = dpg.add_tab(label="Deck A", parent=self.deck_tab_bar)
            
            self.deck_b_tab = dpg.add_tab(label="Deck B", parent=self.deck_tab_bar)
            
            self.ml_rand_tab = dpg.add_tab(label="ML Randomizer", parent=self.deck_tab_bar)
            dpg.add_text("Cognitive Mutation Engine", color=(255, 150, 200), parent=self.ml_rand_tab)
            dpg.add_separator(parent=self.ml_rand_tab)
            
            dpg.add_combo(label="Cognitive Profile", items=["Delta (Relaxed)", "Theta (Flow)", "Alpha (Balanced)", "Beta (Active)", "Gamma (Chaotic)"], default_value="Alpha (Balanced)", tag="ml_rand_profile", parent=self.ml_rand_tab)
            dpg.add_slider_float(label="Mutation Severity (%)", min_value=0.0, max_value=1.0, default_value=0.3, tag="ml_rand_severity", parent=self.ml_rand_tab)
            dpg.add_slider_float(label="Mutation Chance (%)", min_value=0.0, max_value=1.0, default_value=0.15, tag="ml_rand_chance", parent=self.ml_rand_tab)
            
            dpg.add_separator(parent=self.ml_rand_tab)
            dpg.add_text("Mutation Targets:", parent=self.ml_rand_tab)
            dpg.add_checkbox(label="Node Parameters (Active Deck)", default_value=True, tag="ml_rand_nodes", parent=self.ml_rand_tab)
            dpg.add_checkbox(label="Global Time Macros", default_value=False, tag="ml_rand_time", parent=self.ml_rand_tab)
            dpg.add_checkbox(label="Modulation Matrix Depths", default_value=False, tag="ml_rand_mod", parent=self.ml_rand_tab)
            
            dpg.add_separator(parent=self.ml_rand_tab)
            dpg.add_button(label="Trigger Mutation (Burst)", callback=self.trigger_ml_mutation, width=250, height=40, parent=self.ml_rand_tab)
            
            dpg.add_separator(parent=self.ml_rand_tab)
            dpg.add_checkbox(label="Enable Continuous ML Evolution", default_value=False, tag="ml_rand_continuous", parent=self.ml_rand_tab)
            dpg.add_slider_float(label="Evolution Rate (s)", min_value=0.1, max_value=10.0, default_value=2.0, tag="ml_rand_rate", parent=self.ml_rand_tab)
            

            dpg.add_separator(parent=self.ml_rand_tab)
            dpg.add_text("Generative Node Builder:", parent=self.ml_rand_tab)
            dpg.add_slider_int(label="Chain Complexity (Nodes)", min_value=1, max_value=8, default_value=3, tag="ml_rand_complexity", parent=self.ml_rand_tab)
            dpg.add_button(label="Generate Node Chain (Active Deck)", callback=self.generate_node_chain, width=250, height=40, parent=self.ml_rand_tab)

            self.live_sequencer_tab = dpg.add_tab(label="Live Sequencer", parent=self.deck_tab_bar)
            self.live_sequencer_ui = build_live_sequencer_ui(
                self.live_sequencer_tab,
                self.live_sequencer,
                self.visual_bus,
                self.midi_device_manager,
                self.mpd26_controls,
            )
            
            self.node_editor_a.create_node_editor(parent_window=self.deck_a_tab)


            self.node_editor_b.create_node_editor(parent_window=self.deck_b_tab)

            # Install the requested controlled-destruction topology for a new
            # session. Existing projects/presets still replace it through the
            # normal compatibility loader.
            self.load_quality_first_graph(
                editor=self.node_editor_a,
                announce=False,
            )

                        
            self._init_audio_inputs()
            # Create modulation matrix after nodes exist (to discover params)
            self.mod_matrix = NodeModulationMatrix(
                self.node_editor_a,
                providers=self._build_modulation_providers(),
                extra_targets_provider=self._get_global_time_domain_param_defs,
            )
            self.parameter_timeline = ParameterTimeline(
                self.get_active_editor,
                lambda: float(self.time_accum),
            )

            display_node_a = self.node_editor_a.get_display_node()
            if display_node_a:
                display_node_a.display_width = self.display_width
                display_node_a.display_height = self.display_height
                empty_img = np.zeros((display_node_a.display_height, display_node_a.display_width, 4), dtype=np.float32).flatten()
                dpg.add_dynamic_texture(display_node_a.display_width, display_node_a.display_height, empty_img, tag=display_node_a.texture_tag, parent=self.texture_registry)
                
                with dpg.node_attribute(label="Display", attribute_type=dpg.mvNode_Attr_Static, parent=display_node_a.node_tag):
                    display_node_a.display_image_item = dpg.add_image(display_node_a.texture_tag, width=display_node_a.display_width, height=display_node_a.display_height)
                
            display_node_b = self.node_editor_b.get_display_node()
            if display_node_b:
                display_node_b.display_width = self.display_width
                display_node_b.display_height = self.display_height
                empty_img = np.zeros((display_node_b.display_height, display_node_b.display_width, 4), dtype=np.float32).flatten()
                dpg.add_dynamic_texture(display_node_b.display_width, display_node_b.display_height, empty_img, tag=display_node_b.texture_tag, parent=self.texture_registry)

                with dpg.node_attribute(label="Display", attribute_type=dpg.mvNode_Attr_Static, parent=display_node_b.node_tag):
                    display_node_b.display_image_item = dpg.add_image(display_node_b.texture_tag, width=display_node_b.display_width, height=display_node_b.display_height)

        # Treat the editor window as the viewport background so floating tool
        # windows like transport remain above it during normal interaction.
        dpg.set_primary_window("main_window", True)

        import analog_clock
        with dpg.window(label="MainClock", no_title_bar=True, no_resize=True, no_move=True, no_background=True, tag="main_clock_window", pos=[20, 80]):
            analog_clock.create_clock(parent="main_clock_window", radius=30)

        # Global Time Macros Window (floating, minimizable)
        self._ensure_global_time_macros_window(recreate=True)
        
        # Master Mixer Window (floating, persistent)
        with dpg.window(label="Master Mixer", pos=[600, 30], width=350, height=125, tag="master_mixer_window", no_close=True):
            dpg.add_text("A/B Deck Crossfader:")
            # Use a tag so we can update it from external controllers later
            dpg.add_slider_float(label="A <---> B", min_value=0.0, max_value=1.0, default_value=self.master_mix, callback=lambda s,a,u: setattr(self, 'master_mix', a), width=200, tag="master_mix_slider")
            dpg.add_combo(
                label="Transition Mask",
                items=[
                    "Normal Mix",
                    "Luma Wipe",
                    "Glitch Dissolve",
                    "Horizontal Wipe",
                    "Vertical Wipe",
                    "Radial Wipe",
                    "Checker Wipe",
                    "Hard Cut",
                    "Additive",
                    "Multiply",
                    "Screen",
                    "Difference",
                ],
                default_value=self.master_transition,
                callback=lambda s, a, u: setattr(self, 'master_transition', a),
                width=200,
                tag="master_transition_combo",
            )

        # Hidden window for video controls; shown when a video is loaded
        with dpg.window(label="Video Controls", show=False, tag="video_transport_window", width=320, height=170, pos=[20, self.display_height-210], no_resize=True, no_collapse=True):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Play", callback=self.play_video)
                dpg.add_button(label="Pause", callback=self.pause_video)
                dpg.add_button(label="Stop", callback=self.stop_video)
            with dpg.group(horizontal=True):
                dpg.add_button(label="<< Rewind 1s", callback=self.rewind_video)
                dpg.add_button(label=">> Fwd 1s", callback=self.ffwd_video)
            dpg.add_text("0:00.000 / 0:00.000 | frame 0", tag="video_transport_status")
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="video_transport_time_input",
                    default_value="0:00.000",
                    width=150,
                    hint="ss or mm:ss",
                    on_enter=True,
                    callback=self.seek_video_to_time_input,
                )
                dpg.add_button(label="Go To Time", callback=self.seek_video_to_time_input)
        
        # Recording status overlay - shown during export
        with dpg.window(label="Recording Status", show=False, tag="recording_status_window", 
                       width=300, height=100, pos=[self.display_width-320, 60],
                       no_collapse=True, no_resize=True):
            dpg.add_text("🔴 RECORDING", tag="recording_status_text", color=(255, 50, 50))
            dpg.add_progress_bar(tag="recording_progress_bar", default_value=0.0, width=-1)
            dpg.add_text("0.0s / 10.0s", tag="recording_time_text")

        aux_w, aux_h = self.genesis_aux_preview_size
        with dpg.window(
            label="Genesis Aux Inspector",
            show=False,
            tag="genesis_aux_window",
            width=430,
            height=430,
            pos=[self.display_width - 470, 190],
            no_collapse=True,
            no_close=True,
        ):
            dpg.add_text("Inspect semantic Genesis VDP buffers from a live node instance.")
            dpg.add_combo(
                label="Node",
                items=[],
                width=280,
                tag="genesis_aux_node_combo",
                callback=self.on_genesis_aux_node_changed,
            )
            dpg.add_combo(
                label="Output",
                items=list(self.genesis_aux_output_descriptions.keys()),
                default_value=self.genesis_aux_selected_output,
                width=220,
                tag="genesis_aux_output_combo",
                callback=self.on_genesis_aux_output_changed,
            )
            dpg.add_checkbox(
                label="False Color Masks",
                default_value=self.genesis_aux_false_color,
                callback=self.on_genesis_aux_false_color_changed,
            )
            dpg.add_button(label="Refresh Nodes", callback=self.refresh_genesis_aux_inspector)
            dpg.add_text("Waiting for a rendered Genesis node.", tag="genesis_aux_status")
            dpg.add_text("", tag="genesis_aux_desc", wrap=390)
            dpg.add_image(self.genesis_aux_texture_tag, width=aux_w, height=aux_h, tag="genesis_aux_image")

    def _init_audio_inputs(self):
        if self.bpm_detector is None:
            try:
                self.bpm_detector = BPMPeakDetector()
            except Exception as exc:
                logging.warning("Failed to construct BPM detector: %s", exc)
                self.bpm_detector = None
        if self.audio_midi:
            return
        try:
            self.audio_midi = AudioMidiManager(
                sample_rate=44100,
                buffer_size=1024,
                fm_params={},
                fm_state={},
                effect_params={},
                audio_callbacks=[self._on_audio_block],
                midi_callbacks=[self._on_midi_message],
            )
            midi_input_name = self.mpd26_controls.choose_input(self.audio_midi.get_midi_input_names())
            if midi_input_name:
                self.audio_midi.select_midi_input(midi_input_name)
            self.audio_midi.start_input_stream()
            self.audio_midi.start_midi_thread()
            self.refresh_audio_input_devices()
        except Exception as exc:
            logging.warning("Audio input initialisation failed: %s", exc)
            self.audio_midi = None

    def refresh_audio_input_devices(self, sender=None, app_data=None, user_data=None):
        if not self.audio_midi:
            try:
                self._init_audio_inputs()
            except Exception:
                pass
        labels = ["System Default"]
        device_map = {"System Default": None}
        sources = []
        if self.audio_midi:
            try:
                if hasattr(self.audio_midi, "get_input_sources"):
                    sources = self.audio_midi.get_input_sources()
                else:
                    sources = [
                        {
                            "index": device.get("index"),
                            "name": device.get("name"),
                            "label": f"Input: {device.get('index')}: {device.get('name')}",
                            "loopback": False,
                            "channels": 1,
                        }
                        for device in self.audio_midi.get_input_devices()
                    ]
            except Exception as exc:
                logging.warning("Could not query audio input sources: %s", exc)
                sources = []

        for source in sources:
            try:
                label = str(source.get("label") or source.get("name") or "Audio Source")
                labels.append(label)
                device_map[label] = dict(source)
            except Exception:
                continue

        self.audio_input_devices = sources
        self.audio_input_device_map = device_map
        if dpg.does_item_exist(self.audio_input_combo_tag):
            current = dpg.get_value(self.audio_input_combo_tag)
            if current not in device_map:
                current = "System Default"
            dpg.configure_item(self.audio_input_combo_tag, items=labels, default_value=current)
            dpg.set_value(self.audio_input_combo_tag, current)
        self._set_audio_input_status()

    def on_audio_input_selected(self, sender, app_data, user_data=None):
        label = str(app_data or "System Default")
        source = self.audio_input_device_map.get(label)
        if not self.audio_midi:
            self._init_audio_inputs()
        if not self.audio_midi:
            self._set_audio_input_status("Audio input unavailable")
            return
        try:
            if isinstance(source, dict) and hasattr(self.audio_midi, "select_input_source"):
                started = self.audio_midi.select_input_source(source)
            else:
                started = self.audio_midi.select_input_device(source)
            if started is False or self.audio_midi.input_stream is None:
                error = getattr(self.audio_midi, "last_input_error", "") or "capture stream did not start"
                self._set_audio_input_status(f"Audio input error: {error}")
            else:
                self._set_audio_input_status(f"Audio input active: {label}")
        except Exception as exc:
            logging.warning("Could not select audio input '%s': %s", label, exc)
            self._set_audio_input_status(f"Audio input error: {label}")

    def restart_audio_input(self, sender=None, app_data=None, user_data=None):
        if not self.audio_midi:
            self._init_audio_inputs()
        if not self.audio_midi:
            self._set_audio_input_status("Audio input unavailable")
            return
        try:
            self.audio_midi.restart_input_stream()
            label = "System Default"
            if dpg.does_item_exist(self.audio_input_combo_tag):
                label = str(dpg.get_value(self.audio_input_combo_tag) or label)
            self._set_audio_input_status(f"Audio input restarted: {label}")
        except Exception as exc:
            logging.warning("Could not restart audio input: %s", exc)
            self._set_audio_input_status("Audio input restart failed")

    def _set_audio_input_status(self, text=None):
        if text is None:
            label = "System Default"
            if dpg.does_item_exist(self.audio_input_combo_tag):
                label = str(dpg.get_value(self.audio_input_combo_tag) or label)
            text = f"Audio input: {label}"
        if dpg.does_item_exist(self.audio_input_status_tag):
            try:
                dpg.set_value(self.audio_input_status_tag, text)
            except Exception:
                pass

    def open_media_audio_page(self, sender=None, app_data=None, user_data=None):
        window_tag = "media_audio_window"
        if dpg.does_item_exist(window_tag):
            dpg.delete_item(window_tag)

        snapshot = self.media_audio.snapshot()
        params = snapshot["params"]
        mod_sources = snapshot.get("param_mod_sources", {})
        self.media_audio_control_tags = {}
        self.media_audio_mod_source_tags = {}
        self.media_audio_seq_tags = {}
        self.media_audio_seq_step_tags = {}
        with dpg.window(
            label="Synthesis Engine - Media-to-Audio",
            tag=window_tag,
            width=560,
            height=760,
            pos=[60, 90],
            no_collapse=False,
        ):
            dpg.add_text(
                "Synthesize live audio from the processed program display, a "
                "pre-analyzed MP4/GIF timeline, or a blend of both.",
                wrap=520,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Load MP4 / GIF",
                    callback=self.open_media_audio_file_dialog,
                )
                dpg.add_button(
                    label="Import TX802/DX7 SysEx",
                    callback=self.open_media_audio_sysex_dialog,
                )
                dpg.add_button(
                    label="CDP Node Graph",
                    callback=self.open_media_audio_cdp_graph_page,
                )
                dpg.add_button(
                    label="Start / Restart",
                    callback=self.start_media_audio,
                )
                dpg.add_button(
                    label="Stop",
                    callback=self.stop_media_audio,
                )
            dpg.add_input_text(
                label="Media",
                readonly=True,
                width=430,
                tag="media_audio_path",
            )
            dpg.add_text(
                snapshot["status"],
                tag="media_audio_status",
                wrap=520,
            )
            dpg.add_text(
                snapshot.get("sysex_status", "No TX802/DX7 SysEx imported."),
                tag="media_audio_sysex_status",
                wrap=520,
                color=(170, 210, 255),
            )
            dpg.add_progress_bar(
                default_value=0.0,
                width=510,
                overlay="00:00.00 / 00:00.00",
                tag="media_audio_progress",
            )
            dpg.add_separator()
            dpg.add_combo(
                label="Synthesis Engine Source",
                items=["Live Display", "Loaded Media", "Hybrid"],
                default_value=snapshot["source_mode"],
                width=220,
                callback=self.set_media_audio_source,
                tag="media_audio_source_mode",
            )
            dpg.add_text(
                (
                    "Live display feed active."
                    if snapshot["live_metrics_received"]
                    else "Waiting for processed display frames..."
                ),
                tag="media_audio_live_status",
                color=(140, 210, 165),
            )
            dpg.add_combo(
                label="Media Audio Preset",
                items=list(MEDIA_AUDIO_PRESETS.keys()),
                default_value="432 Hz - Golden Spiral",
                width=310,
                callback=self.apply_media_audio_preset,
                tag="media_audio_preset",
            )
            dpg.add_checkbox(
                label="Loop",
                default_value=bool(params["loop"]),
                callback=self.set_media_audio_param,
                user_data="loop",
            )
            dpg.add_separator()
            dpg.add_text("BeatStep Pro MIDI Input", color=(180, 205, 235))
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    label="MIDI Input",
                    items=self._media_audio_midi_input_items(),
                    default_value=self._current_media_audio_midi_input_label(),
                    width=320,
                    callback=self.select_media_audio_midi_input,
                    tag=self.media_audio_midi_input_tag,
                )
                dpg.add_button(label="Refresh", callback=self.refresh_media_audio_midi_inputs, width=85)
            dpg.add_text("", tag=self.media_audio_midi_status_tag, color=(170, 210, 255), wrap=520)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="MIDI Enabled",
                    default_value=bool(params.get("midi_input_enabled", True)),
                    callback=self.set_media_audio_param,
                    user_data="midi_input_enabled",
                )
                dpg.add_checkbox(
                    label="Note Follow",
                    default_value=bool(params.get("midi_note_follow", True)),
                    callback=self.set_media_audio_param,
                    user_data="midi_note_follow",
                )
                dpg.add_checkbox(
                    label="Gate Audio",
                    default_value=bool(params.get("midi_gate_enabled", False)),
                    callback=self.set_media_audio_param,
                    user_data="midi_gate_enabled",
                )
                dpg.add_checkbox(
                    label="Retrigger Phase",
                    default_value=bool(params.get("midi_retrigger_phase", True)),
                    callback=self.set_media_audio_param,
                    user_data="midi_retrigger_phase",
                )
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    label="MIDI Style",
                    items=["Musical", "Drone"],
                    default_value=str(params.get("midi_style", "Musical")),
                    width=160,
                    callback=self.set_media_audio_param,
                    user_data="midi_style",
                )
                self.media_audio_control_tags["midi_transpose"] = dpg.add_slider_float(
                    label="MIDI Transpose",
                    min_value=-48.0,
                    max_value=48.0,
                    default_value=float(params.get("midi_transpose", 0.0)),
                    format="%.0f st",
                    width=220,
                    callback=self.set_media_audio_param,
                    user_data="midi_transpose",
                )
                self.media_audio_control_tags["midi_velocity_depth"] = dpg.add_slider_float(
                    label="Velocity Depth",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=float(params.get("midi_velocity_depth", 0.6)),
                    format="%.3f",
                    width=220,
                    callback=self.set_media_audio_param,
                    user_data="midi_velocity_depth",
                )
            with dpg.group(horizontal=True):
                self.media_audio_control_tags["midi_timbre_depth"] = dpg.add_slider_float(
                    label="MIDI Timbre",
                    min_value=0.0,
                    max_value=2.0,
                    default_value=float(params.get("midi_timbre_depth", 0.45)),
                    format="%.3f",
                    width=220,
                    callback=self.set_media_audio_param,
                    user_data="midi_timbre_depth",
                )
                self.media_audio_control_tags["midi_note_punch"] = dpg.add_slider_float(
                    label="Note Punch",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=float(params.get("midi_note_punch", 0.28)),
                    format="%.3f",
                    width=220,
                    callback=self.set_media_audio_param,
                    user_data="midi_note_punch",
                )
                self.media_audio_control_tags["midi_pitch_slew_ms"] = dpg.add_slider_float(
                    label="Pitch Glide",
                    min_value=0.0,
                    max_value=500.0,
                    default_value=float(params.get("midi_pitch_slew_ms", 45.0)),
                    format="%.0f ms",
                    width=220,
                    callback=self.set_media_audio_param,
                    user_data="midi_pitch_slew_ms",
                )
                self.media_audio_control_tags["midi_note_focus"] = dpg.add_slider_float(
                    label="Note Focus",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=float(params.get("midi_note_focus", 0.55)),
                    format="%.3f",
                    width=220,
                    callback=self.set_media_audio_param,
                    user_data="midi_note_focus",
                )
            dpg.add_text("Synthesis Engine", color=(180, 205, 235))
            with dpg.group(horizontal=True):
                algorithm_value = int(params.get("algorithm", 1))
                self.media_audio_control_tags["algorithm"] = dpg.add_combo(
                    label="TX802 Algorithm",
                    items=TX802_ALGORITHM_NAMES,
                    default_value=TX802_ALGORITHM_NAMES[max(0, min(31, algorithm_value - 1))],
                    width=190,
                    callback=self.set_media_audio_algorithm,
                    user_data="algorithm",
                    tag="media_audio::algorithm",
                )
                self.media_audio_mod_source_tags["algorithm"] = dpg.add_combo(
                    label="Mod Source",
                    items=list(MEDIA_AUDIO_MOD_SOURCES),
                    default_value=mod_sources.get("algorithm", "None"),
                    width=150,
                    callback=self.set_media_audio_param_mod_source,
                    user_data="algorithm",
                    tag="media_audio_mod::algorithm",
                )
            control_defs = [
                ("master_gain", "Master Gain", 0.0, 1.0, "%.3f"),
                ("carrier_freq", "Carrier Frequency", 20.0, 2000.0, "%.2f Hz"),
                ("modulator_ratio", "Modulator Ratio", 0.125, 4.0, "%.5f"),
                ("modulation_index", "FM Modulation Index", 0.0, 12.0, "%.3f"),
                ("feedback", "Operator Feedback", 0.0, 0.95, "%.3f"),
                ("poly_spread", "Polyphonic Detune Spread", 0.0, 1.0, "%.3f"),
                ("chord_depth", "Visual Chord Depth", 0.0, 1.0, "%.3f"),
                ("timbre_spread", "Multitimbral Spread", 0.0, 1.0, "%.3f"),
                ("visual_drive", "Display Modulation Drive", 0.0, 2.0, "%.3f"),
                ("visual_influence", "Visual Influence", 0.0, 2.0, "%.3f"),
                ("visual_evolution", "Visual Evolution", 0.0, 1.0, "%.3f"),
                ("visual_pitch_motion", "Visual Pitch Motion", 0.0, 1.0, "%.3f"),
                ("visual_ir_motion", "Visual IR Motion", 0.0, 1.0, "%.3f"),
                ("feedback_color", "Color Feedback", 0.0, 1.0, "%.3f"),
                ("motion_gating", "Motion Amplitude Gate", 0.0, 1.0, "%.3f"),
                ("motion_response", "Motion Response", 0.0, 4.0, "%.3f"),
                ("brightness_response", "Brightness Amplitude", 0.0, 1.0, "%.3f"),
                ("color_stereo", "Color Stereo Width", 0.0, 1.0, "%.3f"),
                ("visual_morph_depth", "Visual Morph Depth", 0.0, 1.0, "%.3f"),
                ("lfo_rate", "TX802 LFO Rate", 0.0, 30.0, "%.3f Hz"),
                ("lfo_depth", "TX802 LFO Depth", 0.0, 1.0, "%.3f"),
                ("lfo_delay", "TX802 LFO Delay", 0.0, 10.0, "%.3f s"),
                ("idm_intensity", "IDM Intensity", 0.0, 1.0, "%.3f"),
                ("idm_bpm", "IDM BPM", 40.0, 260.0, "%.1f"),
                ("idm_grid_rate", "IDM Grid Rate", 4.0, 64.0, "%.0f"),
                ("idm_probability", "IDM Probability", 0.0, 1.0, "%.3f"),
                ("idm_microtiming", "IDM Microtiming", 0.0, 1.0, "%.3f"),
                ("idm_chaos", "IDM Chaos", 0.0, 1.0, "%.3f"),
                ("idm_phrase_depth", "IDM Phrase Mutation", 0.0, 1.0, "%.3f"),
                ("idm_polyrhythm_depth", "IDM Polyrhythm", 0.0, 1.0, "%.3f"),
                ("idm_scene_reactivity", "IDM Scene Reactivity", 0.0, 1.0, "%.3f"),
                ("idm_density_drift", "IDM Density Drift", 0.0, 1.0, "%.3f"),
                ("idm_retrigger_depth", "IDM Retrigger", 0.0, 1.0, "%.3f"),
                ("idm_stutter_depth", "IDM Stutter", 0.0, 1.0, "%.3f"),
                ("idm_env_slice_depth", "IDM Env Slice", 0.0, 1.0, "%.3f"),
                ("idm_carrier_chop_depth", "IDM Carrier Chop", 0.0, 1.0, "%.3f"),
                ("idm_modulator_chop_depth", "IDM Modulator Chop", 0.0, 1.0, "%.3f"),
                ("idm_ratio_snap_depth", "IDM Ratio Snap", 0.0, 1.0, "%.3f"),
                ("idm_algorithm_morph_depth", "IDM Algorithm Morph", 0.0, 1.0, "%.3f"),
                ("idm_bitcrush_depth", "IDM Bitcrush", 0.0, 1.0, "%.3f"),
                ("idm_sample_rate_fold", "IDM Sample Fold", 0.0, 1.0, "%.3f"),
                ("idm_wavefold_depth", "IDM Wavefold", 0.0, 1.0, "%.3f"),
                ("amp_attack_ms", "Amp Attack", 0.0, 10000.0, "%.0f ms"),
                ("amp_decay_ms", "Amp Decay", 0.0, 10000.0, "%.0f ms"),
                ("amp_sustain", "Amp Sustain", 0.0, 1.0, "%.3f"),
                ("amp_release_ms", "Amp Release", 0.0, 15000.0, "%.0f ms"),
                ("filter_cutoff", "Filter Cutoff", 40.0, 18000.0, "%.0f Hz"),
                ("filter_resonance", "Filter Resonance", 0.0, 0.95, "%.3f"),
                ("filter_env_amount", "Filter Env Amount", -1.0, 1.0, "%.3f"),
                ("filter_visual_amount", "Filter Visual Amount", 0.0, 1.0, "%.3f"),
                ("filter_drive", "Filter Drive", 0.0, 2.0, "%.3f"),
                ("playback_rate", "Analysis Playback Rate", 0.05, 4.0, "%.3fx"),
                ("hybrid_mix", "Hybrid Loaded-Media Mix", 0.0, 1.0, "%.3f"),
            ]
            with dpg.group(horizontal=True):
                self.media_audio_control_tags["amp_env_enabled"] = dpg.add_checkbox(
                    label="Amp ADSR",
                    default_value=bool(params.get("amp_env_enabled", True)),
                    callback=self.set_media_audio_param,
                    user_data="amp_env_enabled",
                    tag="media_audio::amp_env_enabled",
                )
                self.media_audio_control_tags["filter_enabled"] = dpg.add_checkbox(
                    label="Filter",
                    default_value=bool(params.get("filter_enabled", True)),
                    callback=self.set_media_audio_param,
                    user_data="filter_enabled",
                    tag="media_audio::filter_enabled",
                )
                self.media_audio_control_tags["idm_mode"] = dpg.add_checkbox(
                    label="IDM Mode",
                    default_value=bool(params.get("idm_mode", False)),
                    callback=self.set_media_audio_param,
                    user_data="idm_mode",
                    tag="media_audio::idm_mode",
                )
            with dpg.group(horizontal=True):
                self.media_audio_control_tags["voice_count"] = dpg.add_slider_int(
                    label="FM Voices",
                    min_value=1,
                    max_value=16,
                    default_value=int(params["voice_count"]),
                    width=270,
                    callback=self.set_media_audio_param,
                    user_data="voice_count",
                )
                self.media_audio_mod_source_tags["voice_count"] = dpg.add_combo(
                    label="Mod Source",
                    items=list(MEDIA_AUDIO_MOD_SOURCES),
                    default_value=mod_sources.get("voice_count", "None"),
                    width=150,
                    callback=self.set_media_audio_param_mod_source,
                    user_data="voice_count",
                    tag="media_audio_mod::voice_count",
                )
            with dpg.collapsing_header(label="TX802 Operators", default_open=True):
                for operator_index in range(1, 7):
                    dpg.add_text(f"Operator {operator_index}", color=(170, 195, 225))
                    operator_defs = [
                        (f"operator_ratio_{operator_index}", "Ratio", 0.125, 32.0, "%.4f"),
                        (f"operator_output_level_{operator_index}", "Output", 0.0, 1.0, "%.3f"),
                        (f"operator_detune_{operator_index}", "Detune", -100.0, 100.0, "%.2f cents"),
                        (f"operator_pan_{operator_index}", "Pan", -1.0, 1.0, "%.3f"),
                        (f"operator_fixed_freq_{operator_index}", "Fixed Hz", 0.1, 12000.0, "%.2f Hz"),
                        (f"operator_velocity_{operator_index}", "Velocity", 0.0, 1.0, "%.3f"),
                        (f"operator_key_level_{operator_index}", "Key Level", 0.0, 1.0, "%.3f"),
                        (f"operator_key_rate_{operator_index}", "Key Rate", 0.0, 1.0, "%.3f"),
                        (f"operator_glitch_depth_{operator_index}", "IDM Glitch", 0.0, 1.0, "%.3f"),
                        (f"operator_burst_probability_{operator_index}", "IDM Burst", 0.0, 1.0, "%.3f"),
                        (f"operator_level_jitter_{operator_index}", "IDM Level Jitter", 0.0, 1.0, "%.3f"),
                        (f"operator_ratio_snap_{operator_index}", "IDM Ratio Snap", 0.0, 1.0, "%.3f"),
                    ]
                    with dpg.group(horizontal=True):
                        fixed_name = f"operator_fixed_{operator_index}"
                        enabled_name = f"operator_enabled_{operator_index}"
                        self.media_audio_control_tags[fixed_name] = dpg.add_checkbox(
                            label="Fixed",
                            default_value=bool(params.get(fixed_name, False)),
                            callback=self.set_media_audio_param,
                            user_data=fixed_name,
                            tag=f"media_audio::{fixed_name}",
                        )
                        self.media_audio_control_tags[enabled_name] = dpg.add_checkbox(
                            label="Enabled",
                            default_value=bool(params.get(enabled_name, True)),
                            callback=self.set_media_audio_param,
                            user_data=enabled_name,
                            tag=f"media_audio::{enabled_name}",
                        )
                    for name, label, minimum, maximum, format_string in operator_defs:
                        tag = f"media_audio::{name}"
                        self.media_audio_control_tags[name] = tag
                        with dpg.group(horizontal=True):
                            dpg.add_slider_float(
                                label=f"Op {operator_index} {label}",
                                min_value=minimum,
                                max_value=maximum,
                                default_value=float(params.get(name, minimum)),
                                format=format_string,
                                width=270,
                                callback=self.set_media_audio_param,
                                user_data=name,
                                tag=tag,
                            )
                            if name in mod_sources:
                                mod_tag = f"media_audio_mod::{name}"
                                self.media_audio_mod_source_tags[name] = mod_tag
                                dpg.add_combo(
                                    label="Mod Source",
                                    items=list(MEDIA_AUDIO_MOD_SOURCES),
                                    default_value=mod_sources.get(name, "None"),
                                    width=150,
                                    callback=self.set_media_audio_param_mod_source,
                                    user_data=name,
                                    tag=mod_tag,
                                )
                    dpg.add_text(f"Operator {operator_index} EG", color=(140, 165, 205))
                    for eg_kind, eg_label, minimum, maximum, format_string in (
                        ("rate", "R", 0.0, 99.0, "%.1f"),
                        ("level", "L", 0.0, 1.0, "%.3f"),
                    ):
                        with dpg.group(horizontal=True):
                            for stage in range(1, 5):
                                name = f"operator_eg_{eg_kind}_{operator_index}_{stage}"
                                tag = f"media_audio::{name}"
                                self.media_audio_control_tags[name] = tag
                                dpg.add_slider_float(
                                    label=f"{eg_label}{stage}",
                                    min_value=minimum,
                                    max_value=maximum,
                                    default_value=float(params.get(name, minimum)),
                                    format=format_string,
                                    width=105,
                                    callback=self.set_media_audio_param,
                                    user_data=name,
                                    tag=tag,
                                )
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    label="LFO Wave",
                    items=["sine", "triangle", "saw_up", "saw_down", "square", "sample_hold"],
                    default_value=str(params.get("lfo_wave", "sine")),
                    width=170,
                    callback=self.set_media_audio_param,
                    user_data="lfo_wave",
                    tag="media_audio::lfo_wave",
                )
                self.media_audio_control_tags["lfo_wave"] = "media_audio::lfo_wave"
            for name, label, minimum, maximum, format_string in control_defs:
                tag = f"media_audio::{name}"
                self.media_audio_control_tags[name] = tag
                with dpg.group(horizontal=True):
                    dpg.add_slider_float(
                        label=label,
                        min_value=minimum,
                        max_value=maximum,
                        default_value=float(params[name]),
                        format=format_string,
                        width=270,
                        callback=self.set_media_audio_param,
                        user_data=name,
                        tag=tag,
                    )
                    if name in mod_sources:
                        mod_tag = f"media_audio_mod::{name}"
                        self.media_audio_mod_source_tags[name] = mod_tag
                        dpg.add_combo(
                            label="Mod Source",
                            items=list(MEDIA_AUDIO_MOD_SOURCES),
                            default_value=mod_sources.get(name, "None"),
                            width=150,
                            callback=self.set_media_audio_param_mod_source,
                            user_data=name,
                            tag=mod_tag,
                        )
            self._build_media_audio_sequencer_ui()
            dpg.add_separator()
            dpg.add_text("Impulse Response Convolution", color=(180, 205, 235))
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Load IR WAV",
                    callback=self.open_media_audio_ir_dialog,
                )
                dpg.add_button(
                    label="Clear IR",
                    callback=self.clear_media_audio_ir,
                )
            dpg.add_input_text(
                label="IR File",
                readonly=True,
                width=430,
                tag="media_audio_ir_path",
            )
            dpg.add_text(
                snapshot["ir_status"],
                tag="media_audio_ir_status",
                wrap=520,
            )
            self.media_audio_control_tags["ir_wet"] = dpg.add_slider_float(
                label="IR Space",
                min_value=0.0,
                max_value=1.0,
                default_value=float(params["ir_wet"]),
                format="%.3f",
                width=320,
                callback=self.set_media_audio_param,
                user_data="ir_wet",
            )
            self.media_audio_control_tags["ir_gain"] = dpg.add_slider_float(
                label="IR Return",
                min_value=0.0,
                max_value=1.25,
                default_value=float(params["ir_gain"]),
                format="%.3f",
                width=320,
                callback=self.set_media_audio_param,
                user_data="ir_gain",
            )
            dpg.add_slider_float(
                label="Maximum IR Length",
                min_value=0.1,
                max_value=8.0,
                default_value=self.media_audio_ir_max_seconds,
                format="%.2f sec",
                width=320,
                callback=self.set_media_audio_ir_max_seconds,
            )
            dpg.add_checkbox(
                label="Normalize IR",
                default_value=self.media_audio_ir_normalize,
                callback=self.set_media_audio_ir_normalize,
            )
            dpg.add_separator()
            dpg.add_text(
                "The live output uses the system default audio output device. "
                "Media analysis and IR preparation run on background threads.",
                wrap=520,
                color=(160, 175, 190),
            )

    def _media_audio_midi_input_items(self):
        try:
            if not self.audio_midi:
                self._init_audio_inputs()
            names = self.audio_midi.get_midi_input_names() if self.audio_midi else []
        except Exception:
            names = []
        return ["System Default"] + list(names)

    def _current_media_audio_midi_input_label(self):
        if self.audio_midi:
            selected = str(getattr(self.audio_midi, "selected_midi_input_name", "") or "")
            if selected:
                return selected
        return "System Default"

    def refresh_media_audio_midi_inputs(self, sender=None, app_data=None, user_data=None):
        items = self._media_audio_midi_input_items()
        current = self._current_media_audio_midi_input_label()
        if current not in items:
            current = "System Default"
        if dpg.does_item_exist(self.media_audio_midi_input_tag):
            dpg.configure_item(self.media_audio_midi_input_tag, items=items)
            dpg.set_value(self.media_audio_midi_input_tag, current)
        if dpg.does_item_exist(self.media_audio_midi_status_tag):
            dpg.set_value(self.media_audio_midi_status_tag, f"{len(items) - 1} MIDI input(s) available")

    def select_media_audio_midi_input(self, sender=None, app_data=None, user_data=None):
        label = str(app_data or "System Default")
        try:
            if not self.audio_midi:
                self._init_audio_inputs()
            if not self.audio_midi:
                status = "MIDI input unavailable"
            else:
                self.audio_midi.select_midi_input("" if label == "System Default" else label)
                self.audio_midi.start_midi_thread()
                status = f"Media Audio MIDI input: {label}"
        except Exception as exc:
            logging.warning("Could not select Media Audio MIDI input '%s': %s", label, exc)
            status = f"MIDI input error: {label}"
        if dpg.does_item_exist(self.media_audio_midi_status_tag):
            dpg.set_value(self.media_audio_midi_status_tag, status)

    def open_media_audio_file_dialog(self, sender=None, app_data=None, user_data=None):
        dialog_tag = "media_audio_file_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self.load_media_audio_callback,
            tag=dialog_tag,
            modal=True,
            file_count=1,
            width=760,
            height=480,
        ):
            for extension in (
                ".mp4",
                ".mov",
                ".avi",
                ".mkv",
                ".m4v",
                ".webm",
                ".gif",
            ):
                dpg.add_file_extension(extension)

    def load_media_audio_callback(self, sender, app_data, user_data=None):
        path = str((app_data or {}).get("file_path_name") or "")
        if dpg.does_item_exist("media_audio_file_dialog"):
            dpg.delete_item("media_audio_file_dialog")
        if not path:
            return
        self.media_audio.load_async(path)
        if dpg.does_item_exist("media_audio_path"):
            dpg.set_value("media_audio_path", path)
        self._update_media_audio_ui(force=True)

    def open_media_audio_sysex_dialog(self, sender=None, app_data=None, user_data=None):
        dialog_tag = "media_audio_sysex_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self.load_media_audio_sysex_callback,
            tag=dialog_tag,
            modal=True,
            file_count=1,
            width=760,
            height=480,
        ):
            dpg.add_file_extension(".syx")

    def load_media_audio_sysex_callback(self, sender, app_data, user_data=None):
        path = str((app_data or {}).get("file_path_name") or "")
        if dpg.does_item_exist("media_audio_sysex_dialog"):
            dpg.delete_item("media_audio_sysex_dialog")
        if not path:
            return
        self.media_audio.load_tx802_sysex_async(path)
        self._update_media_audio_ui(force=True)

    def _build_media_audio_sequencer_ui(self):
        sequencer = self.media_audio.sequencer
        target_items = sorted(sequencer.param_limits.keys())
        visual_items = [
            "brightness",
            "contrast",
            "motion",
            "entropy",
            "edge_density",
            "red_energy",
            "green_energy",
            "blue_energy",
            "glitch_spike",
            "center_mass_x",
            "center_mass_y",
        ]
        with dpg.collapsing_header(label="BeatStep / Media Step Sequencer", default_open=True):
            dpg.add_text("16-step parameter sequencer for Media-to-Audio synthesis.", color=(160, 175, 190))
            with dpg.group(horizontal=True):
                self.media_audio_seq_tags["enabled"] = dpg.add_checkbox(
                    label="Enabled",
                    default_value=sequencer.enabled,
                    callback=self.set_media_audio_seq_global,
                    user_data="enabled",
                )
                dpg.add_button(label="Play", width=60, callback=self.play_media_audio_sequencer)
                dpg.add_button(label="Stop", width=60, callback=self.stop_media_audio_sequencer)
                dpg.add_button(label="Reset", width=60, callback=self.reset_media_audio_sequencer)
                dpg.add_button(label="Mutate From Visual", width=145, callback=self.mutate_media_audio_sequence)
                self.media_audio_seq_tags["step_status"] = dpg.add_text("Step 1", color=(170, 210, 255))
            with dpg.group(horizontal=True):
                self.media_audio_seq_tags["bpm"] = dpg.add_slider_float(
                    label="Seq BPM",
                    min_value=20.0,
                    max_value=300.0,
                    default_value=sequencer.bpm,
                    width=180,
                    callback=self.set_media_audio_seq_global,
                    user_data="bpm",
                )
                self.media_audio_seq_tags["random_strength"] = dpg.add_slider_float(
                    label="Random Strength",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=sequencer.random_strength,
                    width=180,
                    callback=self.set_media_audio_seq_global,
                    user_data="random_strength",
                )
                self.media_audio_seq_tags["chaos_ceiling"] = dpg.add_slider_float(
                    label="Chaos Ceiling",
                    min_value=0.0,
                    max_value=1.0,
                    default_value=sequencer.chaos_ceiling,
                    width=180,
                    callback=self.set_media_audio_seq_global,
                    user_data="chaos_ceiling",
                )
                self.media_audio_seq_tags["freeze_randomization"] = dpg.add_checkbox(
                    label="Freeze Random",
                    default_value=sequencer.freeze_randomization,
                    callback=self.set_media_audio_seq_global,
                    user_data="freeze_randomization",
                )
            dpg.add_text("BeatStep Pro: Start/Stop/Clock sync, notes 36-51 toggle lane 1 steps, notes 52-59 toggle lanes, CC10/11/12 set random/chaos/BPM.", wrap=520)
            with dpg.table(header_row=True, policy=dpg.mvTable_SizingStretchProp, borders_innerV=True, borders_outerV=True, borders_innerH=True):
                dpg.add_table_column(label="Lane")
                dpg.add_table_column(label="On")
                dpg.add_table_column(label="Target")
                dpg.add_table_column(label="Min")
                dpg.add_table_column(label="Max")
                dpg.add_table_column(label="Visual")
                dpg.add_table_column(label="Amt")
                dpg.add_table_column(label="Rnd")
                for step in range(16):
                    dpg.add_table_column(label=str(step + 1))
                for lane_index, lane in enumerate(sequencer.lanes):
                    with dpg.table_row():
                        dpg.add_text(str(lane_index + 1))
                        self.media_audio_seq_tags[(lane_index, "enabled")] = dpg.add_checkbox(
                            label="",
                            default_value=lane.enabled,
                            callback=self.set_media_audio_seq_lane,
                            user_data=(lane_index, "enabled"),
                        )
                        self.media_audio_seq_tags[(lane_index, "target")] = dpg.add_combo(
                            label="",
                            items=target_items,
                            default_value=lane.target,
                            width=145,
                            callback=self.set_media_audio_seq_lane,
                            user_data=(lane_index, "target"),
                        )
                        self.media_audio_seq_tags[(lane_index, "min_value")] = dpg.add_input_float(
                            label="",
                            default_value=lane.min_value,
                            width=75,
                            format="%.3f",
                            callback=self.set_media_audio_seq_lane,
                            user_data=(lane_index, "min_value"),
                        )
                        self.media_audio_seq_tags[(lane_index, "max_value")] = dpg.add_input_float(
                            label="",
                            default_value=lane.max_value,
                            width=75,
                            format="%.3f",
                            callback=self.set_media_audio_seq_lane,
                            user_data=(lane_index, "max_value"),
                        )
                        self.media_audio_seq_tags[(lane_index, "visual_source")] = dpg.add_combo(
                            label="",
                            items=visual_items,
                            default_value=lane.visual_source,
                            width=115,
                            callback=self.set_media_audio_seq_lane,
                            user_data=(lane_index, "visual_source"),
                        )
                        self.media_audio_seq_tags[(lane_index, "visual_amount")] = dpg.add_slider_float(
                            label="",
                            min_value=0.0,
                            max_value=1.0,
                            default_value=lane.visual_amount,
                            width=75,
                            callback=self.set_media_audio_seq_lane,
                            user_data=(lane_index, "visual_amount"),
                        )
                        self.media_audio_seq_tags[(lane_index, "random_amount")] = dpg.add_slider_float(
                            label="",
                            min_value=0.0,
                            max_value=1.0,
                            default_value=lane.random_amount,
                            width=75,
                            callback=self.set_media_audio_seq_lane,
                            user_data=(lane_index, "random_amount"),
                        )
                        for step_index, step in enumerate(lane.steps):
                            tag = dpg.add_checkbox(
                                label="",
                                default_value=step.enabled,
                                callback=self.set_media_audio_seq_step,
                                user_data=(lane_index, step_index),
                            )
                            self.media_audio_seq_step_tags[(lane_index, step_index)] = tag

    def set_media_audio_seq_global(self, sender=None, app_data=None, user_data=None):
        seq = self.media_audio.sequencer
        name = str(user_data or "")
        if name in {"enabled", "freeze_randomization"}:
            setattr(seq, name, bool(app_data))
        elif name in {"bpm", "random_strength", "chaos_ceiling"}:
            setattr(seq, name, float(app_data))

    def play_media_audio_sequencer(self, sender=None, app_data=None, user_data=None):
        self.media_audio.sequencer.play()
        if dpg.does_item_exist(self.media_audio_seq_tags.get("enabled", "")):
            dpg.set_value(self.media_audio_seq_tags["enabled"], True)

    def stop_media_audio_sequencer(self, sender=None, app_data=None, user_data=None):
        self.media_audio.sequencer.stop()

    def reset_media_audio_sequencer(self, sender=None, app_data=None, user_data=None):
        self.media_audio.sequencer.reset()
        self._update_media_audio_sequencer_ui(force=True)

    def mutate_media_audio_sequence(self, sender=None, app_data=None, user_data=None):
        self.media_audio.mutate_sequence_from_visual(self.get_latest_visual_metrics())
        self._update_media_audio_sequencer_ui(force=True)

    def set_media_audio_seq_lane(self, sender=None, app_data=None, user_data=None):
        if not user_data:
            return
        lane_index, field_name = user_data
        self.media_audio.sequencer.update_lane(int(lane_index), str(field_name), app_data)
        if field_name == "target":
            self._update_media_audio_sequencer_ui(force=True)

    def set_media_audio_seq_step(self, sender=None, app_data=None, user_data=None):
        if not user_data:
            return
        lane_index, step_index = user_data
        self.media_audio.sequencer.update_step(int(lane_index), int(step_index), "enabled", bool(app_data))

    def apply_media_audio_preset(self, sender=None, app_data=None, user_data=None):
        preset_name = str(app_data or "")
        try:
            values = self.media_audio.apply_preset(preset_name)
        except Exception as exc:
            logging.warning("Could not apply Media Audio preset '%s': %s", preset_name, exc)
            return
        for name, value in values.items():
            tag = self.media_audio_control_tags.get(name)
            if tag and dpg.does_item_exist(tag):
                if name == "algorithm":
                    index = max(0, min(31, int(value) - 1))
                    dpg.set_value(tag, TX802_ALGORITHM_NAMES[index])
                else:
                    dpg.set_value(tag, value)

    def set_media_audio_source(self, sender=None, app_data=None, user_data=None):
        try:
            self.media_audio.set_source_mode(str(app_data or "Live Display"))
        except Exception:
            logging.exception("Could not set Media Audio source mode")
        self._update_media_audio_ui(force=True)

    def open_media_audio_cdp_graph_page(self, sender=None, app_data=None, user_data=None):
        window_tag = "media_audio_cdp_graph_window"
        if dpg.does_item_exist(window_tag):
            dpg.delete_item(window_tag)
        snapshot = self.media_audio.snapshot().get("cdp_graph", {})
        self.media_audio_cdp_tags = {}
        with dpg.window(label="Media Audio CDP Node Graph", tag=window_tag, width=620, height=720, pos=[650, 90], no_collapse=False):
            dpg.add_text(
                "Nonblocking CDP 8 graph for live Media Audio. CPU Preview is the live effect path; CDP Worker is experimental and only runs raw-WAV-compatible CDP recipes.",
                wrap=580,
            )
            dpg.add_text(str(snapshot.get("status", "CDP graph bypassed.")), tag="media_audio_cdp_status", wrap=580, color=(170, 210, 255))
            with dpg.group(horizontal=True):
                self.media_audio_cdp_tags["enabled"] = dpg.add_checkbox(label="Enable Graph", default_value=bool(snapshot.get("enabled", False)), callback=self.set_media_audio_cdp_graph_option, user_data="enabled")
                self.media_audio_cdp_tags["cpu_preview"] = dpg.add_checkbox(label="CPU Preview", default_value=bool(snapshot.get("cpu_preview", True)), callback=self.set_media_audio_cdp_graph_option, user_data="cpu_preview")
                self.media_audio_cdp_tags["use_cdp"] = dpg.add_checkbox(label="Use CDP Worker", default_value=bool(snapshot.get("use_cdp", False)), callback=self.set_media_audio_cdp_graph_option, user_data="use_cdp")
            self.media_audio_cdp_tags["wet_mix"] = dpg.add_slider_float(label="Wet Mix", min_value=0.0, max_value=1.0, default_value=float(snapshot.get("wet_mix", 0.0)), format="%.3f", width=300, callback=self.set_media_audio_cdp_graph_option, user_data="wet_mix")
            self.media_audio_cdp_tags["cdp_programs_path"] = dpg.add_input_text(label="CDP _cdprogs", default_value=str(snapshot.get("cdp_programs_path", "")), width=500, callback=self.set_media_audio_cdp_graph_option, user_data="cdp_programs_path")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_combo(label="Effect", items=list(CDP_EFFECT_SPECS), default_value=next(iter(CDP_EFFECT_SPECS)), width=210, tag="media_audio_cdp_add_effect")
                dpg.add_button(label="Add Node", callback=self.add_media_audio_cdp_graph_node)
                dpg.add_button(label="Refresh", callback=lambda *_: self._update_media_audio_cdp_graph_ui(force=True))
            dpg.add_child_window(tag="media_audio_cdp_nodes", width=590, height=480, border=True)
        self._update_media_audio_cdp_graph_ui(force=True)

    def set_media_audio_cdp_graph_option(self, sender=None, app_data=None, user_data=None):
        if not user_data:
            return
        try:
            self.media_audio.configure_cdp_graph(**{str(user_data): app_data})
        except Exception:
            logging.exception("Could not update CDP graph option '%s'", user_data)
        self._update_media_audio_cdp_graph_ui(force=True)

    def add_media_audio_cdp_graph_node(self, sender=None, app_data=None, user_data=None):
        effect = "Blur Average"
        if dpg.does_item_exist("media_audio_cdp_add_effect"):
            effect = str(dpg.get_value("media_audio_cdp_add_effect") or effect)
        self.media_audio.add_cdp_graph_node(effect)
        self._update_media_audio_cdp_graph_ui(force=True)

    def remove_media_audio_cdp_graph_node(self, sender=None, app_data=None, user_data=None):
        if user_data is None:
            return
        self.media_audio.remove_cdp_graph_node(int(user_data))
        self._update_media_audio_cdp_graph_ui(force=True)

    def move_media_audio_cdp_graph_node(self, sender=None, app_data=None, user_data=None):
        if not user_data:
            return
        index, delta = user_data
        self.media_audio.move_cdp_graph_node(int(index), int(delta))
        self._update_media_audio_cdp_graph_ui(force=True)

    def set_media_audio_cdp_graph_node(self, sender=None, app_data=None, user_data=None):
        if not user_data:
            return
        index, field_name = user_data
        updates = {}
        if field_name == "enabled":
            updates["enabled"] = bool(app_data)
        elif field_name == "effect":
            updates["effect"] = str(app_data or "")
        else:
            updates["params"] = {str(field_name): app_data}
        self.media_audio.update_cdp_graph_node(int(index), **updates)
        if field_name == "effect":
            self._update_media_audio_cdp_graph_ui(force=True)

    def _update_media_audio_cdp_graph_ui(self, force=False):
        if not dpg.does_item_exist("media_audio_cdp_graph_window"):
            return
        snapshot = self.media_audio.snapshot().get("cdp_graph", {})
        if dpg.does_item_exist("media_audio_cdp_status"):
            dpg.set_value("media_audio_cdp_status", str(snapshot.get("status", "")))
        for key in ("enabled", "cpu_preview", "use_cdp", "wet_mix", "cdp_programs_path"):
            tag = self.media_audio_cdp_tags.get(key)
            if tag and dpg.does_item_exist(tag):
                dpg.set_value(tag, snapshot.get(key, False if key != "wet_mix" else 0.0))
        nodes_tag = "media_audio_cdp_nodes"
        if force and dpg.does_item_exist(nodes_tag):
            dpg.delete_item(nodes_tag, children_only=True)
            nodes = snapshot.get("graph", {}).get("nodes", [])
            if not nodes:
                with dpg.group(parent=nodes_tag):
                    dpg.add_text("No CDP graph nodes. Add an effect node to start.")
                return
            for index, node in enumerate(nodes):
                effect = str(node.get("effect", "Blur Average"))
                params = dict(node.get("params", {}))
                with dpg.collapsing_header(label=f"{index + 1}. {effect}", default_open=True, parent=nodes_tag):
                    with dpg.group(horizontal=True):
                        dpg.add_checkbox(label="On", default_value=bool(node.get("enabled", True)), callback=self.set_media_audio_cdp_graph_node, user_data=(index, "enabled"))
                        dpg.add_combo(label="Effect", items=list(CDP_EFFECT_SPECS), default_value=effect if effect in CDP_EFFECT_SPECS else "Blur Average", width=180, callback=self.set_media_audio_cdp_graph_node, user_data=(index, "effect"))
                        dpg.add_button(label="Up", callback=self.move_media_audio_cdp_graph_node, user_data=(index, -1))
                        dpg.add_button(label="Down", callback=self.move_media_audio_cdp_graph_node, user_data=(index, 1))
                        dpg.add_button(label="Remove", callback=self.remove_media_audio_cdp_graph_node, user_data=index)
                    for param_name, value in params.items():
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            numeric = 0.0
                        max_value = 1.0
                        if "freq" in param_name:
                            max_value = 12000.0
                        elif "grain" in param_name or "amount" in param_name or "cycles" in param_name:
                            max_value = 128.0
                        elif "gain" in param_name:
                            max_value = 4.0
                        dpg.add_slider_float(label=param_name, min_value=0.0, max_value=max_value, default_value=max(0.0, min(max_value, numeric)), format="%.3f", width=320, callback=self.set_media_audio_cdp_graph_node, user_data=(index, param_name))

    def set_media_audio_param(self, sender=None, app_data=None, user_data=None):
        if not user_data:
            return
        try:
            self.media_audio.set_param(str(user_data), app_data)
        except Exception:
            logging.exception("Could not update Media Audio parameter '%s'", user_data)

    def set_media_audio_algorithm(self, sender=None, app_data=None, user_data=None):
        try:
            label = str(app_data or "")
            if label in TX802_ALGORITHM_NAMES:
                value = TX802_ALGORITHM_NAMES.index(label) + 1
            else:
                value = int(label.split()[1])
            self.media_audio.set_param("algorithm", value)
        except Exception:
            logging.exception("Could not update Media Audio TX802 algorithm")

    def set_media_audio_param_mod_source(self, sender=None, app_data=None, user_data=None):
        if not user_data:
            return
        try:
            self.media_audio.set_param_mod_source(str(user_data), str(app_data or "None"))
        except Exception:
            logging.exception("Could not update Media Audio modulation source '%s'", user_data)

    def start_media_audio(self, sender=None, app_data=None, user_data=None):
        self.media_audio.start(restart=True)
        self._update_media_audio_ui(force=True)

    def open_media_audio_ir_dialog(self, sender=None, app_data=None, user_data=None):
        dialog_tag = "media_audio_ir_dialog"
        if dpg.does_item_exist(dialog_tag):
            dpg.delete_item(dialog_tag)
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self.load_media_audio_ir_callback,
            tag=dialog_tag,
            modal=True,
            file_count=1,
            width=760,
            height=480,
        ):
            dpg.add_file_extension(".wav")

    def load_media_audio_ir_callback(self, sender, app_data, user_data=None):
        path = str((app_data or {}).get("file_path_name") or "")
        if dpg.does_item_exist("media_audio_ir_dialog"):
            dpg.delete_item("media_audio_ir_dialog")
        if not path:
            return
        self.media_audio.load_ir_async(
            path,
            max_seconds=self.media_audio_ir_max_seconds,
            normalize=self.media_audio_ir_normalize,
        )
        if dpg.does_item_exist("media_audio_ir_path"):
            dpg.set_value("media_audio_ir_path", path)
        self._update_media_audio_ui(force=True)

    def clear_media_audio_ir(self, sender=None, app_data=None, user_data=None):
        self.media_audio.clear_ir()
        if dpg.does_item_exist("media_audio_ir_path"):
            dpg.set_value("media_audio_ir_path", "")
        self._update_media_audio_ui(force=True)

    def set_media_audio_ir_max_seconds(self, sender=None, app_data=None, user_data=None):
        try:
            self.media_audio_ir_max_seconds = min(
                8.0, max(0.1, float(app_data))
            )
        except (TypeError, ValueError):
            self.media_audio_ir_max_seconds = 4.0

    def set_media_audio_ir_normalize(self, sender=None, app_data=None, user_data=None):
        self.media_audio_ir_normalize = bool(app_data)

    def stop_media_audio(self, sender=None, app_data=None, user_data=None):
        self.media_audio.stop()
        self._update_media_audio_ui(force=True)

    @staticmethod
    def _format_media_audio_time(seconds):
        seconds = max(0.0, float(seconds or 0.0))
        minutes = int(seconds // 60.0)
        remainder = seconds - minutes * 60.0
        return f"{minutes:02d}:{remainder:05.2f}"

    def _update_media_audio_ui(self, force=False):
        if not dpg.does_item_exist("media_audio_window"):
            self.media_audio.tick_sequencer(self.get_latest_visual_metrics())
            return
        self.media_audio.tick_sequencer(self.get_latest_visual_metrics())
        snapshot = self.media_audio.snapshot()
        status = str(snapshot["status"])
        if force or status != self.media_audio_last_status:
            if dpg.does_item_exist("media_audio_status"):
                dpg.set_value("media_audio_status", status)
            self.media_audio_last_status = status
        if dpg.does_item_exist("media_audio_sysex_status"):
            dpg.set_value(
                "media_audio_sysex_status",
                snapshot.get("sysex_status", "No TX802/DX7 SysEx imported."),
            )
        if dpg.does_item_exist("media_audio_preset"):
            dpg.configure_item("media_audio_preset", items=list(MEDIA_AUDIO_PRESETS.keys()))
        ir_status = str(snapshot["ir_status"])
        if force or ir_status != self.media_audio_last_ir_status:
            if dpg.does_item_exist("media_audio_ir_status"):
                dpg.set_value("media_audio_ir_status", ir_status)
            self.media_audio_last_ir_status = ir_status
        duration = float(snapshot["duration"])
        position = float(snapshot["position"])
        progress = position / duration if duration > 0.0 else 0.0
        if dpg.does_item_exist("media_audio_progress"):
            dpg.set_value("media_audio_progress", min(1.0, max(0.0, progress)))
            dpg.configure_item(
                "media_audio_progress",
                overlay=(
                    f"{self._format_media_audio_time(position)} / "
                    f"{self._format_media_audio_time(duration)}"
                ),
            )
        source_path = str(snapshot.get("source_path") or "")
        if source_path and dpg.does_item_exist("media_audio_path"):
            dpg.set_value("media_audio_path", source_path)
        ir_path = str(snapshot.get("ir_path") or "")
        if dpg.does_item_exist("media_audio_ir_path"):
            dpg.set_value("media_audio_ir_path", ir_path)
        if dpg.does_item_exist("media_audio_live_status"):
            dpg.set_value(
                "media_audio_live_status",
                (
                    "Live display feed active."
                    if snapshot["live_metrics_received"]
                    else "Waiting for processed display frames..."
                ),
            )
        if dpg.does_item_exist(self.media_audio_midi_status_tag):
            seq_state = snapshot.get("sequencer", {})
            selected = self._current_media_audio_midi_input_label()
            last_midi = str(seq_state.get("last_midi_status") or "")
            suffix = f" | Last: {last_midi}" if last_midi else ""
            dpg.set_value(self.media_audio_midi_status_tag, f"Media Audio MIDI input: {selected}{suffix}")
        display_params = snapshot.get("params", {})
        mod_sources = snapshot.get("param_mod_sources", {})
        for name, tag in self.media_audio_control_tags.items():
            if not dpg.does_item_exist(tag):
                continue
            if name == "algorithm":
                value = int(display_params.get(name, 1))
                index = max(0, min(31, value - 1))
                dpg.set_value(tag, TX802_ALGORITHM_NAMES[index])
            elif name in display_params:
                dpg.set_value(tag, display_params[name])
        for name, tag in self.media_audio_mod_source_tags.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, mod_sources.get(name, "None"))
        self._update_media_audio_sequencer_ui(force=force)
        self._update_media_audio_cdp_graph_ui(force=False)

    def _update_media_audio_sequencer_ui(self, force=False):
        seq = self.media_audio.sequencer
        status_tag = self.media_audio_seq_tags.get("step_status")
        if status_tag and dpg.does_item_exist(status_tag):
            mode = "BSP Clock" if seq.clock_source == "beatstep" else "Internal"
            state = "Playing" if seq.playing else "Stopped"
            dpg.set_value(status_tag, f"{state} {mode} Step {seq.step_index + 1}  {seq.last_midi_status}")
        for key in ("enabled", "freeze_randomization", "bpm", "random_strength", "chaos_ceiling"):
            tag = self.media_audio_seq_tags.get(key)
            if force and tag and dpg.does_item_exist(tag):
                dpg.set_value(tag, getattr(seq, key))
        if not force:
            return
        for lane_index, lane in enumerate(seq.lanes):
            for field_name in ("enabled", "target", "min_value", "max_value", "visual_source", "visual_amount", "random_amount"):
                tag = self.media_audio_seq_tags.get((lane_index, field_name))
                if tag and dpg.does_item_exist(tag):
                    dpg.set_value(tag, getattr(lane, field_name))
            for step_index, step in enumerate(lane.steps):
                tag = self.media_audio_seq_step_tags.get((lane_index, step_index))
                if tag and dpg.does_item_exist(tag):
                    dpg.set_value(tag, step.enabled)

    def _on_audio_block(self, block, fft_res, sample_rate, timestamp):
        if not self.bpm_detector:
            return
        try:
            self.bpm_detector.update_from_block(block, fft_res, sample_rate, timestamp)
        except Exception:
            logging.debug("BPM detector update failed", exc_info=True)

    def _get_fft_snapshot(self):
        if not self.audio_midi:
            return None
        with self.audio_midi.data_lock:
            data = getattr(self.audio_midi, 'fft_data', None)
            if data is None:
                return None
            return data.copy()

    def _get_fft_bands_snapshot(self):
        if not self.audio_midi:
            return dict(self.latest_audio_bands)
        with self.audio_midi.data_lock:
            bands = getattr(self.audio_midi, 'fft_bands', None)
            if isinstance(bands, dict):
                self.latest_audio_bands = dict(bands)
            return dict(self.latest_audio_bands)

    def _get_onset_value(self):
        if not self.audio_midi:
            return float(self.latest_onset_value)
        with self.audio_midi.data_lock:
            self.latest_onset_value = float(getattr(self.audio_midi, 'onset_value', 0.0) or 0.0)
            return float(self.latest_onset_value)

    def _get_onset_event_id(self):
        if not self.audio_midi:
            return int(self.latest_onset_event_id)
        with self.audio_midi.data_lock:
            self.latest_onset_event_id = int(getattr(self.audio_midi, 'onset_event_id', 0) or 0)
            return int(self.latest_onset_event_id)

    def _get_global_time_domain_param_defs(self):
        return [
            {"name": "history_offset", "type": "float", "label": "History Offset", "min_value": 0.0, "max_value": 32.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 8.0},
            {"name": "memory_decay", "type": "float", "label": "Memory Decay", "min_value": 0.0, "max_value": 0.999, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.08},
            {"name": "event_hazard", "type": "float", "label": "Event Hazard", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.25},
            {"name": "clock_quantization", "type": "float", "label": "Clock Quantization", "min_value": 0.0, "max_value": 0.25, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.08},
            {"name": "sync_desync", "type": "float", "label": "Sync / Desync", "min_value": -1.0, "max_value": 1.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.6},
            {"name": "phase_reset_chance", "type": "float", "label": "Phase Reset Chance", "min_value": 0.0, "max_value": 0.5, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.12},
            {"name": "local_time_warp", "type": "float", "label": "Local Time Warp", "min_value": -2.0, "max_value": 2.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 1.0},
            {"name": "rupture_threshold", "type": "float", "label": "Rupture Threshold", "min_value": 0.0, "max_value": 2.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.6},
            {"name": "hysteresis_speed", "type": "float", "label": "Hysteresis Speed", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.25},
            {"name": "modulator_coupling", "type": "float", "label": "Modulator Coupling", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 0.4},
            {"name": "time_reversal", "type": "float", "label": "Time Reversal", "min_value": 0.0, "max_value": 1.0, "default_value": 0.0,
             "mod_mode": "offset", "mod_span": 1.0},
        ]

    def _set_global_time_domain_macro(self, name, value):
        if name not in self.global_time_domain_macros:
            return
        self.global_time_domain_macros[name] = float(value)

    def _ensure_global_time_macros_window(self, recreate=False):
        window_tag = "global_time_macros_window"
        if recreate and dpg.does_item_exist(window_tag):
            try:
                dpg.delete_item(window_tag)
            except Exception:
                pass

        if not dpg.does_item_exist(window_tag):
            with dpg.window(
                label="Global Time Macros",
                show=True,
                tag=window_tag,
                width=320,
                height=300,
                pos=[20, self.display_height - 550],
                no_resize=False,
                no_collapse=False,
            ):
                for param_info in self._get_global_time_domain_param_defs():
                    self._add_global_time_domain_slider(param_info["name"])
        else:
            try:
                dpg.configure_item(window_tag, show=True)
            except Exception:
                pass

    def _add_global_time_domain_slider(self, name):
        param_map = {param["name"]: param for param in self._get_global_time_domain_param_defs()}
        param = param_map.get(name)
        if not param:
            return None
        tag = f"global_time_domain::{name}"
        self.global_time_domain_slider_tags[name] = tag
        return dpg.add_slider_float(
            label=str(param["label"]),
            min_value=float(param["min_value"]),
            max_value=float(param["max_value"]),
            default_value=float(self.global_time_domain_macros.get(name, param["default_value"])),
            callback=lambda s, a, u: self._set_global_time_domain_macro(u, a),
            user_data=name,
            tag=tag,
            width=180,
        )

    def _get_global_time_domain_macro_values(self):
        return {name: float(value) for name, value in self.global_time_domain_macros.items()}

    def _set_processing_scale(self, sender=None, app_data=None, user_data=None):
        try:
            self.processing_scale = float(app_data)
        except (TypeError, ValueError):
            self.processing_scale = 1.0

    def _set_preview_framerate(self, sender=None, app_data=None, user_data=None):
        try:
            self.framerate = int(np.clip(int(app_data), 1, 60))
            self._framerate_user_overridden = True
        except (TypeError, ValueError):
            self.framerate = 30

    def _toggle_adaptive_processing_scale(self, sender=None, app_data=None, user_data=None):
        self.adaptive_processing_scale = bool(app_data)
        self._adaptive_latency_ema_ms = None
        self._adaptive_queue_pressure = 0
        self._adaptive_pipeline_signature = None
        self._adaptive_last_result_time = time.perf_counter()
        self._adaptive_last_adjust_time = time.perf_counter()

    def _effective_processing_scale(self, is_recording=False):
        try:
            scale = float(np.clip(self.processing_scale, 0.1, 1.0))
            if self.adaptive_processing_scale:
                scale = self._quantize_adaptive_processing_scale(scale)
            return scale
        except Exception:
            return 1.0

    def _resize_preview_frame(self, frame, scale):
        if frame is None:
            return None
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            return frame
        if not (0.1 <= scale < 1.0):
            return frame

        if isinstance(frame, Image.Image):
            w, h = frame.size
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return frame.resize((new_w, new_h), Image.BILINEAR)

        arr = np.asarray(frame)
        if arr.ndim < 2:
            return frame
        h, w = arr.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(arr, (new_w, new_h), interpolation=interpolation)

    def _quantize_adaptive_processing_scale(self, scale):
        """Keep adaptive preview sizes in stable buckets to avoid reallocating every adjustment."""
        try:
            min_scale = float(np.clip(self.adaptive_processing_min_scale, 0.1, 1.0))
            max_scale = float(np.clip(self.adaptive_processing_max_scale, min_scale, 1.0))
            scale = float(np.clip(scale, min_scale, max_scale))
            step = 0.05
            return float(np.clip(round(scale / step) * step, min_scale, max_scale))
        except Exception:
            return float(np.clip(scale, 0.1, 1.0))

    def _set_adaptive_processing_scale_value(self, scale):
        new_scale = self._quantize_adaptive_processing_scale(scale)
        current = self._quantize_adaptive_processing_scale(self.processing_scale)
        if abs(new_scale - current) < 0.024:
            return False
        self.processing_scale = new_scale
        if hasattr(self, "processing_scale_slider") and dpg.does_item_exist(self.processing_scale_slider):
            try:
                dpg.set_value(self.processing_scale_slider, new_scale)
            except Exception:
                pass
        return True

    def _mark_adaptive_queue_pressure(self):
        if self.adaptive_processing_scale:
            self._adaptive_queue_pressure = min(12, int(self._adaptive_queue_pressure) + 1)

    def _processing_pipeline_signature(self, pipeline):
        signature = []
        for step in pipeline or []:
            effect_type = str(step.get("effect_type", ""))
            if effect_type == "Display":
                continue
            params = step.get("params", {}) or {}
            heavy_params = []
            for name in ("buffer_size", "frame_buffer_size", "temporal_range", "time_offset_range"):
                if name in params:
                    try:
                        heavy_params.append((name, int(float(params.get(name) or 0))))
                    except Exception:
                        heavy_params.append((name, repr(params.get(name))))
            signature.append((effect_type, step.get("node_type"), tuple(heavy_params)))
        return tuple(signature)

    def _estimate_pipeline_preview_scale(self, pipeline):
        cost = 0.0
        for step in pipeline or []:
            label = str(step.get("effect_type", ""))
            if label == "Display":
                continue
            params = step.get("params", {}) or {}
            cost += 1.0
            lower = label.lower()
            shader_preset = str(params.get("shader_preset", "") or "").strip().lower()
            shader_path = str(params.get("shader_path", "") or "").replace("\\", "/").strip().lower()
            if label == "Lua Shader":
                cost += 1.4
            if shader_preset == "enhance" or shader_path.endswith("/enhance.frag"):
                cost += 5.0
            if "cuda" in lower:
                cost += 1.6
            if any(word in lower for word in ("temporal", "flow", "slit", "mosh", "feedback", "genesis", "fractal")):
                cost += 1.4
            if any(word in lower for word in ("p-frame", "pframe", "prediction")):
                cost += 2.8
            for name in (
                "motion_vector_corruption",
                "prediction_error_amplify",
                "macroblock_scramble",
                "reference_corruption",
            ):
                try:
                    cost += max(0.0, float(params.get(name, 0.0) or 0.0)) * 1.2
                except Exception:
                    pass
            for name, weight in (
                ("buffer_size", 0.035),
                ("frame_buffer_size", 0.08),
                ("temporal_range", 0.05),
                ("time_offset_range", 0.04),
            ):
                try:
                    cost += max(0.0, float(params.get(name, 0.0) or 0.0)) * weight
                except Exception:
                    pass

        if cost >= 7.0:
            return 0.35
        if cost >= 5.0:
            return 0.58
        if cost >= 3.5:
            return 0.72
        return 1.0

    def _prime_adaptive_processing_scale_for_pipeline(self, pipeline, is_recording=False):
        if not self.adaptive_processing_scale or is_recording:
            return
        signature = self._processing_pipeline_signature(pipeline)
        if signature == self._adaptive_pipeline_signature:
            return
        self._adaptive_pipeline_signature = signature
        self._adaptive_latency_ema_ms = None
        self._adaptive_queue_pressure = 0
        self._adaptive_last_result_time = time.perf_counter()
        suggested = self._estimate_pipeline_preview_scale(pipeline)
        suggested = self._quantize_adaptive_processing_scale(suggested)
        if suggested < self._quantize_adaptive_processing_scale(self.processing_scale):
            self._set_adaptive_processing_scale_value(suggested)
            logging.info("Adaptive preview scale primed to %.2f for current graph", suggested)

    def _update_adaptive_processing_without_result(self):
        if not self.adaptive_processing_scale:
            return
        now = time.perf_counter()
        target_fps = max(1.0, float(self.framerate or 30))
        frame_budget = 1.0 / target_fps
        if now - float(self._adaptive_last_result_time or now) < frame_budget * 5.0:
            return
        if now - float(self._adaptive_last_adjust_time or 0.0) < 0.75:
            return
        min_scale = float(np.clip(self.adaptive_processing_min_scale, 0.1, 1.0))
        current_scale = self._quantize_adaptive_processing_scale(np.clip(self.processing_scale, min_scale, 1.0))
        new_scale = float(np.clip(current_scale - 0.05, min_scale, 1.0))
        if new_scale < current_scale and self._set_adaptive_processing_scale_value(new_scale):
            self._adaptive_last_adjust_time = now
            self._adaptive_queue_pressure = min(12, int(self._adaptive_queue_pressure) + 2)

    def _update_adaptive_processing_scale(self, latency_ms):
        if not self.adaptive_processing_scale:
            return
        try:
            latency_ms = float(latency_ms)
        except (TypeError, ValueError):
            return

        target_fps = max(1.0, float(self.framerate or 30))
        frame_budget_ms = 1000.0 / target_fps
        target_ms = float(self.adaptive_processing_target_latency_ms or (frame_budget_ms * 1.75))
        min_scale = float(np.clip(self.adaptive_processing_min_scale, 0.1, 1.0))
        max_scale = float(np.clip(self.adaptive_processing_max_scale, min_scale, 1.0))
        current_scale = self._quantize_adaptive_processing_scale(np.clip(self.processing_scale, min_scale, max_scale))

        if self._adaptive_latency_ema_ms is None:
            self._adaptive_latency_ema_ms = latency_ms
        else:
            self._adaptive_latency_ema_ms = (self._adaptive_latency_ema_ms * 0.82) + (latency_ms * 0.18)

        now = time.perf_counter()
        if now - float(self._adaptive_last_adjust_time or 0.0) < 0.75:
            return

        pressure = int(self._adaptive_queue_pressure)
        ema = float(self._adaptive_latency_ema_ms)
        new_scale = current_scale

        if pressure >= 3 or ema > target_ms * 1.35:
            severity = max(1.0, ema / max(1.0, target_ms))
            step = 0.10 if pressure >= 5 else 0.05
            new_scale = current_scale - step * min(2.0, severity)
        elif ema < target_ms * 0.55 and pressure == 0:
            new_scale = current_scale + 0.05

        new_scale = self._quantize_adaptive_processing_scale(np.clip(new_scale, min_scale, max_scale))
        if self._set_adaptive_processing_scale_value(new_scale):
            self._adaptive_last_adjust_time = now
            logging.debug(
                "Adaptive preview scale: %.3f (latency_ema=%.1fms target=%.1fms pressure=%s)",
                new_scale,
                ema,
                target_ms,
                pressure,
            )

        self._adaptive_queue_pressure = max(0, pressure - 1)

    def _get_active_time_domain_macros(self, merged_params):
        active_macros = self._get_global_time_domain_macro_values()
        for name in active_macros:
            if name in merged_params:
                try:
                    active_macros[name] = float(merged_params[name])
                except Exception:
                    active_macros[name] = merged_params[name]
        return active_macros

    def _is_passthrough_pipeline(self, pipeline):
        if not isinstance(pipeline, list) or len(pipeline) != 2:
            return False
        source_step, display_step = pipeline[0], pipeline[1]
        return (
            source_step.get("node_type") == "source"
            and display_step.get("node_type") == "display"
        )

    def _load_global_time_domain_macros_from_preset(self, preset_name):
        import json

        filepath = Path("node_presets") / f"{preset_name}.json"
        if not filepath.exists():
            return
        try:
            with filepath.open("r", encoding="utf-8") as handle:
                preset_data = json.load(handle)
        except Exception as e:
            logging.warning(f"Could not read display macros from preset '{preset_name}': {e}")
            return

        macro_values = preset_data.get("display_macros", {})
        if not isinstance(macro_values, dict):
            return

        for param in self._get_global_time_domain_param_defs():
            name = param["name"]
            value = macro_values.get(name, param["default_value"])
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = float(param["default_value"])
            self.global_time_domain_macros[name] = numeric_value
            if name == "local_time_warp" and abs(numeric_value) > 0.5:
                logging.warning(
                    "Preset '%s' loaded with local_time_warp=%s; video sources may seek frequently and lose FPS.",
                    preset_name,
                    numeric_value,
                )
            slider_tag = self.global_time_domain_slider_tags.get(name)
            if slider_tag and dpg.does_item_exist(slider_tag):
                try:
                    dpg.set_value(slider_tag, numeric_value)
                except Exception:
                    pass

    def _save_global_time_domain_macros_to_preset(self, preset_name):
        import json

        filepath = Path("node_presets") / f"{preset_name}.json"
        if not filepath.exists():
            return
        try:
            with filepath.open("r", encoding="utf-8") as handle:
                preset_data = json.load(handle)
        except Exception as e:
            logging.warning(f"Could not update preset '{preset_name}' with display macros: {e}")
            return

        preset_data["display_macros"] = self._get_global_time_domain_macro_values()

        try:
            with filepath.open("w", encoding="utf-8") as handle:
                json.dump(preset_data, handle, indent=2)
        except Exception as e:
            logging.warning(f"Could not write display macros to preset '{preset_name}': {e}")

    def _build_modulation_providers(self):
        return {
            'fft_data': self._get_fft_snapshot,
            'fft_bands': self._get_fft_bands_snapshot,
            'onset_value': self._get_onset_value,
            'onset_event_id': self._get_onset_event_id,
            'midi_note': lambda: (self.audio_midi.latest_midi_note_velocity if self.audio_midi else 0.0),
            'midi_cc': lambda: (self.audio_midi.latest_midi_cc_value if self.audio_midi else 0.0),
            'bpm_phase': lambda: (self.bpm_detector.get_beat_phase() if self.bpm_detector else 0.0),
            'bpm_pulse': lambda: (self.bpm_detector.get_pulse_value() if self.bpm_detector else 0.0),
            'bpm_normalized': lambda: (self.bpm_detector.get_normalized_bpm() if self.bpm_detector else 0.0),
            'bpm_value': lambda: (self.bpm_detector.get_bpm() if self.bpm_detector else 0.0),
            'beat_event_id': lambda: (self.bpm_detector.get_beat_event_id() if self.bpm_detector else 0),
            'peak_event_id': lambda: (self.bpm_detector.get_peak_event_id() if self.bpm_detector else 0),
        }

    def add_effect_node(self, sender, app_data, user_data):
        self.get_active_editor().add_node(user_data)
        # Auto-refresh modulation targets when nodes are added
        try:
            if self.mod_matrix:
                self.mod_matrix.refresh_targets_ui()
        except Exception:
            pass
        # Auto-refresh pipeline navigator
        try:
            if self.pipeline_navigator_visible and dpg.does_item_exist("pipeline_navigator_window"):
                self.update_pipeline_navigator()
        except Exception:
            pass
        self.refresh_genesis_aux_inspector()

    def _get_genesis_aux_nodes(self):
        return [
            node for node in getattr(self.node_editor_a, "nodes", [])
            if isinstance(node, effect_nodes.GenesisVDPGeneratorNode)
        ]

    def _clear_genesis_aux_preview(self):
        if not self.genesis_aux_texture_tag:
            return
        aux_w, aux_h = self.genesis_aux_preview_size
        blank = np.zeros((aux_h, aux_w, 3), dtype=np.uint8)
        try:
            update_display_texture(blank, self.genesis_aux_texture_tag, aux_w, aux_h)
        except Exception:
            logging.debug("Failed to clear Genesis aux preview", exc_info=True)

    def _sync_genesis_aux_description(self):
        output_name = str(self.genesis_aux_selected_output or "priority_mask")
        description = self.genesis_aux_output_descriptions.get(output_name, "")
        if self.genesis_aux_false_color and output_name in {
            "priority_mask",
            "palette_index_buffer",
            "luma_mask",
            "corruption_mask",
            "scanline_mod_buffer",
        }:
            description = f"{description} False-color preview enabled for readability."
        if dpg.does_item_exist("genesis_aux_desc"):
            dpg.set_value("genesis_aux_desc", description)

    def refresh_genesis_aux_inspector(self, sender=None, app_data=None):
        if not dpg.does_item_exist("genesis_aux_node_combo"):
            return
        nodes = self._get_genesis_aux_nodes()
        self.genesis_aux_node_map = {}
        node_items = []
        selected_label = None
        for index, node in enumerate(nodes, start=1):
            item_label = f"{node.label} #{index}"
            node_items.append(item_label)
            self.genesis_aux_node_map[item_label] = node.node_tag
            if node.node_tag == self.genesis_aux_selected_node_id:
                selected_label = item_label

        if selected_label is None and node_items:
            selected_label = node_items[0]
            self.genesis_aux_selected_node_id = self.genesis_aux_node_map[selected_label]

        dpg.configure_item("genesis_aux_node_combo", items=node_items)
        if selected_label is not None:
            dpg.set_value("genesis_aux_node_combo", selected_label)
            dpg.set_value("genesis_aux_status", "Live preview reads the selected Genesis node's cached auxiliary buffer.")
        else:
            self.genesis_aux_selected_node_id = None
            try:
                dpg.set_value("genesis_aux_node_combo", "")
            except Exception:
                pass
            dpg.set_value("genesis_aux_status", "No Genesis VDP Gen nodes found in the graph.")
            self._clear_genesis_aux_preview()

        if dpg.does_item_exist("genesis_aux_output_combo"):
            dpg.configure_item("genesis_aux_output_combo", items=list(self.genesis_aux_output_descriptions.keys()))
            dpg.set_value("genesis_aux_output_combo", self.genesis_aux_selected_output)
        self._sync_genesis_aux_description()

    def on_genesis_aux_node_changed(self, sender, app_data):
        self.genesis_aux_selected_node_id = self.genesis_aux_node_map.get(app_data)
        self._sync_genesis_aux_description()

    def on_genesis_aux_output_changed(self, sender, app_data):
        self.genesis_aux_selected_output = str(app_data or "priority_mask")
        self._sync_genesis_aux_description()

    def on_genesis_aux_false_color_changed(self, sender, app_data):
        self.genesis_aux_false_color = bool(app_data)
        self._sync_genesis_aux_description()

    def toggle_genesis_aux_inspector(self, sender=None, app_data=None):
        self.genesis_aux_window_visible = not self.genesis_aux_window_visible
        if dpg.does_item_exist("genesis_aux_window"):
            dpg.configure_item("genesis_aux_window", show=self.genesis_aux_window_visible)
        if self.genesis_aux_window_visible:
            self.refresh_genesis_aux_inspector()

    def _get_genesis_aux_debug_request(self):
        if not self.genesis_aux_window_visible:
            return None
        if self.genesis_aux_selected_node_id is None:
            return None
        active_ids = {node.node_tag for node in self._get_genesis_aux_nodes()}
        if self.genesis_aux_selected_node_id not in active_ids:
            self.refresh_genesis_aux_inspector()
        if self.genesis_aux_selected_node_id is None:
            return None
        return {
            "node_id": self.genesis_aux_selected_node_id,
            "output": str(self.genesis_aux_selected_output or "priority_mask"),
        }

    def _prepare_genesis_aux_preview_frame(self, frame, output_name):
        if frame is None:
            return None
        arr = np.asarray(frame, dtype=np.uint8)
        mask_like_outputs = {
            "priority_mask",
            "palette_index_buffer",
            "luma_mask",
            "corruption_mask",
            "scanline_mod_buffer",
        }
        if not self.genesis_aux_false_color or output_name not in mask_like_outputs:
            return arr
        if arr.ndim == 3 and arr.shape[2] >= 3:
            gray = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
        else:
            gray = arr
        colorized = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    def _update_genesis_aux_preview_from_result(self, glitched_result):
        if not self.genesis_aux_window_visible or not isinstance(glitched_result, dict):
            return
        preview_frame = glitched_result.get("genesis_debug_frame")
        preview_output = str(glitched_result.get("genesis_debug_output") or self.genesis_aux_selected_output or "priority_mask")
        preview_node_id = glitched_result.get("genesis_debug_node_id")
        if preview_frame is None:
            if dpg.does_item_exist("genesis_aux_status") and self.genesis_aux_selected_node_id is not None:
                dpg.set_value("genesis_aux_status", "No live auxiliary buffer is available for the selected Genesis node.")
            return
        preview_frame = self._prepare_genesis_aux_preview_frame(preview_frame, preview_output)
        aux_w, aux_h = self.genesis_aux_preview_size
        try:
            update_display_texture(preview_frame, self.genesis_aux_texture_tag, aux_w, aux_h)
            if dpg.does_item_exist("genesis_aux_status"):
                dpg.set_value(
                    "genesis_aux_status",
                    f"Previewing {preview_output} from Genesis node {preview_node_id}.",
                )
        except Exception:
            logging.error("Genesis aux preview update failed", exc_info=True)

    def open_image(self, sender, app_data):
        self.processing_paused = True  # Pause during dialog
        if dpg.does_item_exist("file_dialog_image"): dpg.delete_item("file_dialog_image")
        with dpg.file_dialog(directory_selector=False, show=True, callback=self.load_image_callback, 
                            tag="file_dialog_image", modal=True, file_count=1,
                            cancel_callback=lambda: setattr(self, 'processing_paused', False)):
            dpg.add_file_extension(".png"); dpg.add_file_extension(".jpg"); dpg.add_file_extension(".*")

    def load_image_callback(self, sender, app_data):
        self.processing_paused = False  # Resume processing
        path = app_data['file_path_name']
        if path and self._deck_b_is_active():
            self.load_mixer_channel_b_callback(sender, app_data)
        elif path:
            self.stop_all_media()
            self.primary_media_path = str(path)
            loaded_img = open_image_file(path)
            if loaded_img:
                # Store original with alpha for real-time BG changes
                self.original_image_with_alpha = loaded_img.copy()
                # Apply background color if image has transparency
                self.image = self._apply_background_to_image(loaded_img)
        dpg.delete_item("file_dialog_image")
    
    def change_image_background(self, sender, app_data):
        """Change the background color preset and update image in real-time."""
        preset = app_data
        
        if preset == "Black":
            self.image_bg_color = (0, 0, 0, 255)
        elif preset == "White":
            self.image_bg_color = (255, 255, 255, 255)
        elif preset == "Transparent":
            self.image_bg_color = (0, 0, 0, 0)
        elif preset == "Custom":
            # Use custom color from color picker
            try:
                custom_color = dpg.get_value("custom_bg_color")
                self.image_bg_color = tuple(int(c * 255) for c in custom_color)
            except Exception:
                self.image_bg_color = (0, 0, 0, 255)
        elif preset == "Media":
            # Use loaded media as background - handled in _apply_background_to_image
            pass
        
        # Re-apply background using stored original image
        if self.original_image_with_alpha:
            self.image = self._apply_background_to_image(self.original_image_with_alpha.copy())
            logging.info(f"Background changed to: {preset}")
    
    def update_custom_bg_color(self, sender, app_data):
        """Update custom background color in real-time."""
        # Convert from 0-1 float to 0-255 int
        self.image_bg_color = tuple(int(c * 255) for c in app_data)
        
        # Apply immediately if we have an image with alpha
        if self.original_image_with_alpha:
            self.image = self._apply_background_to_image(self.original_image_with_alpha.copy())

    def update_bg_media_opacity(self, sender, app_data):
        """Update loaded background media blend amount for the source image."""
        try:
            self.image_bg_media_opacity = max(0.0, min(1.0, float(app_data)))
        except (TypeError, ValueError):
            self.image_bg_media_opacity = 1.0
        if self.original_image_with_alpha:
            self.image = self._apply_background_to_image(self.original_image_with_alpha.copy())
    
    def _apply_background_to_image(self, img):
        """Apply display background color/media to the current source image."""
        if img is None:
            return None

        # Check if using media background
        preset = dpg.get_value("bg_preset_combo") if dpg.does_item_exist("bg_preset_combo") else "Black"
        
        # Check if ANY background media is loaded (image, gif, or video)
        has_bg_media = self.bg_media or self.bg_gif_frames or self.bg_video_reader
        
        if preset == "Media" and has_bg_media:
            # Use media as background
            background = self._get_current_bg_frame()
            if background:
                foreground = img.convert('RGBA')
                # Ensure background matches image size
                if background.size != foreground.size:
                    background = background.resize(foreground.size, Image.Resampling.LANCZOS)
                background = background.convert('RGBA')
                opacity = max(0.0, min(1.0, float(self.image_bg_media_opacity)))
                if opacity < 1.0:
                    foreground = Image.blend(background, foreground, opacity)
                # Composite respects real PNG alpha; blend above makes opaque images
                # partially reveal the display background when requested.
                result = Image.alpha_composite(background, foreground)
                return result

        # If there is no media background and image has no alpha, return as-is.
        if img.mode != 'RGBA':
            return img
        
        # Use solid color background
        if self.image_bg_color[3] == 0:
            return img  # Transparent background
        
        bg_color = self.image_bg_color
        background = Image.new('RGBA', img.size, bg_color)
        
        # Composite image over background
        result = Image.alpha_composite(background, img)
        
        return result
    
    def _get_current_bg_frame(self):
        """Get the current frame from background media (handles image/video/gif)."""
        if not self.bg_media and not self.bg_gif_frames and not self.bg_video_reader:
            return None
        
        # Static image background
        if self.bg_media:
            return self.bg_media
        
        # GIF background
        if self.bg_gif_frames:
            frame = self.bg_gif_frames[self.bg_gif_index % len(self.bg_gif_frames)]
            self.bg_gif_index += 1
            return frame
        
        # Video background
        if self.bg_video_reader:
            try:
                ok, frame_bgr = self.bg_video_reader.read()
                if ok and frame_bgr is not None:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(frame_rgb)
                else:
                    # Loop video
                    self.bg_video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame_bgr = self.bg_video_reader.read()
                    if ok and frame_bgr is not None:
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(frame_rgb)
            except Exception as e:
                logging.error(f"Error reading background video frame: {e}")
        
        return None

    def _background_media_active(self):
        preset = dpg.get_value("bg_preset_combo") if dpg.does_item_exist("bg_preset_combo") else "Black"
        return preset == "Media" and bool(self.bg_media or self.bg_gif_frames or self.bg_video_reader)

    def open_gif(self, sender, app_data):
        self.processing_paused = True  # Pause during dialog
        if dpg.does_item_exist("file_dialog_gif"): dpg.delete_item("file_dialog_gif")
        with dpg.file_dialog(directory_selector=False, show=True, callback=self.load_gif_callback, 
                            tag="file_dialog_gif", modal=True,
                            cancel_callback=lambda: setattr(self, 'processing_paused', False)):
            dpg.add_file_extension(".gif")

    def load_gif_callback(self, sender, app_data):
        self.processing_paused = False  # Resume processing
        path = app_data['file_path_name']
        if path and self._deck_b_is_active():
            self.load_mixer_channel_b_callback(sender, app_data)
        elif path:
            self.stop_all_media()
            self.primary_media_path = str(path)
            self.gif_frames, self.gif_durations = open_gif_file(path)
            if self.gif_frames:
                self.gif_index = 0
                self.gif_running = True
                self.last_gif_frame_time = time.time()
        dpg.delete_item("file_dialog_gif")

    def open_video(self, sender, app_data):
        self.processing_paused = True  # Pause during dialog
        if dpg.does_item_exist("file_dialog_video"): dpg.delete_item("file_dialog_video")
        with dpg.file_dialog(directory_selector=False, show=True, callback=self.load_video_callback, 
                            tag="file_dialog_video", modal=True, file_count=1,
                            cancel_callback=lambda: setattr(self, 'processing_paused', False)):
            dpg.add_file_extension(".mp4")
            dpg.add_file_extension(".avi")
            dpg.add_file_extension(".mov")
            dpg.add_file_extension(".mkv")
            dpg.add_file_extension(".*")

    def load_video_callback(self, sender, app_data):
        self.processing_paused = False  # Resume processing
        try:
            path = app_data['file_path_name']
            if path and self._deck_b_is_active():
                self.load_mixer_channel_b_callback(sender, app_data)
            elif path:
                self.stop_all_media()
                self.primary_media_path = str(path)
                self.video_reader = open_video_file(path)
                if self.video_reader:
                    self._clear_primary_video_frame_cache()
                    self.video_transport_frame = 0.0
                    self._load_srt_shader_sidecar(path)
                    # Source FPS still drives video frame selection; this only
                    # controls render cadence. Keep automatic preview capped at
                    # 30 FPS so heavy GLSL graphs do not starve media audio.
                    try:
                        meta = self.video_reader.get_meta_data()
                        fps = meta.get('fps')
                        if fps and not self._framerate_user_overridden:
                            self.framerate = int(max(1, min(30, round(float(fps)))))
                            try:
                                dpg.set_value(self.framerate_slider, self.framerate)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    self.video_playing = True
                    self.video_refresh_requested = True
                    self._update_video_transport_visibility()
                    self._sync_video_transport_ui(update_input=True)
            dpg.delete_item("file_dialog_video")
        except Exception as e:
            logging.error(f"Error loading video: {e}")

    def open_srt_shader_cues(self, sender=None, app_data=None):
        self.processing_paused = True
        if dpg.does_item_exist("file_dialog_srt_shader"):
            dpg.delete_item("file_dialog_srt_shader")
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self.load_srt_shader_cues_callback,
            tag="file_dialog_srt_shader",
            modal=True,
            file_count=1,
            cancel_callback=lambda: setattr(self, "processing_paused", False),
        ):
            dpg.add_file_extension(".srt")
            dpg.add_file_extension(".*")

    def load_srt_shader_cues_callback(self, sender, app_data):
        self.processing_paused = False
        try:
            path = app_data.get("file_path_name")
            if path:
                self._load_srt_shader_cues(path)
            if dpg.does_item_exist("file_dialog_srt_shader"):
                dpg.delete_item("file_dialog_srt_shader")
        except Exception as e:
            logging.error(f"Error loading SRT shader cues: {e}")

    def _load_srt_shader_sidecar(self, video_path):
        sidecar = find_video_srt_sidecar(video_path)
        if sidecar:
            self._load_srt_shader_cues(sidecar, default_preset=self._infer_srt_shader_preset(video_path, sidecar))
        else:
            self.srt_shader_cues = []
            self.srt_shader_path = ""
            self.srt_shader_last_change_key = None
            self.srt_shader_default_preset = None

    def _load_srt_shader_cues(self, path, default_preset=None):
        cues = parse_srt_shader_cues(path, default_preset=default_preset)
        self.srt_shader_cues = cues
        self.srt_shader_path = str(path)
        self.srt_shader_last_change_key = None
        self.srt_shader_default_preset = default_preset
        suffix = f" with default preset {default_preset}" if default_preset else ""
        logging.info("Loaded %s SRT shader cues from %s%s", len(cues), path, suffix)

    def _infer_srt_shader_preset(self, video_path, srt_path):
        video_stem = Path(video_path).stem
        srt_stem = Path(srt_path).stem
        prefix = f"{video_stem}."
        if not srt_stem.startswith(prefix):
            return None
        preset = srt_stem[len(prefix):].strip()
        return preset or None

    def _apply_srt_shader_cues_to_pipelines(self, pipeline_a, pipeline_b, seconds):
        if not self.srt_shader_enabled or not self.srt_shader_cues:
            return {}
        base_dir = Path(__file__).resolve().parent
        changes = {}
        try:
            changes.update(
                apply_srt_shader_cues(
                    pipeline_a,
                    self.srt_shader_cues,
                    float(seconds),
                    base_dir,
                    deck="a",
                )
            )
            changes.update(
                apply_srt_shader_cues(
                    pipeline_b,
                    self.srt_shader_cues,
                    float(seconds),
                    base_dir,
                    deck="b",
                )
            )
        except Exception as e:
            logging.debug(f"Could not apply SRT shader cues: {e}")
        if changes:
            change_key = tuple(sorted(changes.keys()))
            if change_key != self.srt_shader_last_change_key:
                logging.debug("Applied SRT shader cues at %.3fs: %s", seconds, changes)
                self.srt_shader_last_change_key = change_key
        elif self.srt_shader_last_change_key is not None:
            self.srt_shader_last_change_key = None
        return changes

    def show_window_list(self, sender, app_data):
        windows = get_window_titles()
        if not windows: return
        if dpg.does_item_exist("window_select_modal"): dpg.delete_item("window_select_modal")
        with dpg.window(label="Select Window", modal=True, tag="window_select_modal", width=400, height=300):
            for wtitle in windows: dpg.add_button(label=wtitle, callback=self.select_window_callback, user_data=wtitle)
            dpg.add_button(label="Close", callback=lambda: dpg.delete_item("window_select_modal"))

    def show_source_selector(self, sender, app_data):
        if dpg.does_item_exist("source_select_modal"):
            dpg.delete_item("source_select_modal")
        self._refresh_webcam_devices()
        source_items = ["Image", "GIF", "Video", "Window Capture", "Webcam"]
        if self.source_selector_mode not in source_items:
            self.source_selector_mode = "Image"
        with dpg.window(label="Select Source", modal=True, tag="source_select_modal", width=420, height=240):
            dpg.add_text("Source Type")
            dpg.add_combo(
                items=source_items,
                default_value=self.source_selector_mode,
                callback=self._on_source_type_changed,
                width=260,
                tag="source_type_combo",
            )
            dpg.add_spacer(height=8)
            dpg.add_text("Webcam Device", tag="webcam_device_label")
            dpg.add_combo(
                items=self.webcam_device_options or ["0"],
                default_value=str(self.webcam_index),
                width=260,
                tag="webcam_device_combo",
            )
            self._on_source_type_changed(None, self.source_selector_mode, None)
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", callback=self._start_selected_source, width=120)
                dpg.add_button(label="Close", callback=lambda: dpg.delete_item("source_select_modal"), width=120)

    def _on_source_type_changed(self, sender, app_data, user_data):
        self.source_selector_mode = str(app_data or "Image")
        show_webcam = self.source_selector_mode == "Webcam"
        for item_tag in ("webcam_device_label", "webcam_device_combo"):
            if dpg.does_item_exist(item_tag):
                dpg.configure_item(item_tag, show=show_webcam)
        if show_webcam and dpg.does_item_exist("webcam_device_combo"):
            dpg.configure_item("webcam_device_combo", items=self.webcam_device_options or ["0"])
            dpg.set_value("webcam_device_combo", str(self.webcam_index))

    def _refresh_webcam_devices(self):
        found_indices = self._probe_webcam_indices(max_indices=10)
        self.webcam_device_options = [str(i) for i in found_indices]
        if not self.webcam_device_options:
            self.webcam_device_options = ["0"]
        if str(self.webcam_index) not in self.webcam_device_options:
            self.webcam_index = int(self.webcam_device_options[0])

    def _open_webcam_capture(self, index, quiet=False):
        backends = [None]
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)

        old_log_level = None
        if quiet:
            try:
                old_log_level = cv2.getLogLevel()
                cv2.setLogLevel(0)
            except Exception:
                old_log_level = None

        try:
            for backend in backends:
                cap = None
                try:
                    cap = cv2.VideoCapture(int(index)) if backend is None else cv2.VideoCapture(int(index), backend)
                    if cap and cap.isOpened():
                        return cap
                except Exception:
                    pass
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        finally:
            if old_log_level is not None:
                try:
                    cv2.setLogLevel(old_log_level)
                except Exception:
                    pass
        return None

    def _probe_webcam_indices(self, max_indices=10):
        found = []
        for idx in range(max(1, int(max_indices))):
            cap = None
            try:
                cap = self._open_webcam_capture(idx, quiet=True)
                if cap and cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None and getattr(frame, "size", 0) > 0:
                        found.append(idx)
            except Exception:
                pass
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
        return found

    def _start_selected_source(self, sender, app_data):
        source_mode = str(dpg.get_value("source_type_combo")) if dpg.does_item_exist("source_type_combo") else self.source_selector_mode
        self.source_selector_mode = source_mode
        if dpg.does_item_exist("source_select_modal"):
            dpg.delete_item("source_select_modal")
        if source_mode == "Image":
            self.open_image(sender, app_data)
            return
        if source_mode == "GIF":
            self.open_gif(sender, app_data)
            return
        if source_mode == "Video":
            self.open_video(sender, app_data)
            return
        if source_mode == "Window Capture":
            self.start_window_capture(sender, app_data)
            return
        if source_mode == "Webcam":
            combo_value = dpg.get_value("webcam_device_combo") if dpg.does_item_exist("webcam_device_combo") else str(self.webcam_index)
            try:
                self.webcam_index = int(str(combo_value).strip())
            except Exception:
                self.webcam_index = 0
            self.start_webcam_capture(sender, app_data)

    def select_window_callback(self, sender, app_data, user_data):
        self.selected_window = user_data
        if dpg.does_item_exist("window_select_modal"): dpg.delete_item("window_select_modal")
        if self._window_capture_start_pending:
            self._window_capture_start_pending = False
            self.start_window_capture(sender, app_data)

    def start_window_capture(self, sender, app_data):
        if not self.selected_window:
            self._window_capture_start_pending = True
            self.show_window_list(sender, app_data)
            return
        self.stop_all_media()
        self.last_window_capture_image = None
        self._release_window_capture_resources()
        self._window_capture_fps_samples = 0
        self._window_capture_fps_last_time = time.perf_counter()
        self._window_capture_last_ms = 0.0
        self.window_capture_running = True
        try:
            w = self._resolve_window_capture_target(force=True)
            if w is not None:
                if w.isMinimized:
                    w.restore()
                try:
                    w.activate()
                except Exception:
                    pass
            self._window_capture_sct = mss.mss()
        except Exception:
            logging.debug("Window capture initialization fallback will retry in loop", exc_info=True)
        logging.info(f"Window capture started: {self.selected_window}")

    def stop_window_capture(self, sender, app_data):
        self._window_capture_start_pending = False
        self.window_capture_running = False
        self.last_window_capture_image = None
        self._release_window_capture_resources()

    def _release_window_capture_resources(self):
        self._window_capture_window = None
        self._window_capture_last_bbox = None
        self._window_capture_last_resolve = 0.0
        sct = getattr(self, "_window_capture_sct", None)
        self._window_capture_sct = None
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass

    def _resolve_window_capture_target(self, force=False):
        if not self.selected_window:
            self._window_capture_window = None
            return None
        now = time.perf_counter()
        cached = self._window_capture_window
        if not force and cached is not None:
            try:
                if getattr(cached, "width", 0) > 0 and getattr(cached, "height", 0) > 0:
                    return cached
            except Exception:
                pass
        if not force and now - float(self._window_capture_last_resolve or 0.0) < 1.0:
            return cached
        self._window_capture_last_resolve = now
        try:
            wins = gw.getWindowsWithTitle(self.selected_window)
            self._window_capture_window = wins[0] if wins else None
        except Exception:
            self._window_capture_window = None
        return self._window_capture_window

    def _capture_selected_window_frame(self):
        capture_started = time.perf_counter()
        w = self._resolve_window_capture_target()
        if w is None:
            if self.frame_counter % 120 == 0:
                logging.warning(f"Window capture target not found: {self.selected_window}")
            return None
        try:
            if w.isMinimized:
                if self.frame_counter % 120 == 0:
                    logging.warning(
                        "Window capture paused while target is minimized: %s",
                        self.selected_window,
                    )
                return self.last_window_capture_image.copy() if self.last_window_capture_image is not None else None
            width = int(getattr(w, "width", 0))
            height = int(getattr(w, "height", 0))
            if width <= 0 or height <= 0:
                if self.frame_counter % 120 == 0:
                    logging.warning(
                        "Window capture target has invalid size: %s (%sx%s)",
                        self.selected_window,
                        width,
                        height,
                    )
                return None
            bbox = {
                "top": int(getattr(w, "top", 0)),
                "left": int(getattr(w, "left", 0)),
                "width": width,
                "height": height,
            }
            self._window_capture_last_bbox = bbox
            if self._window_capture_sct is None:
                self._window_capture_sct = mss.mss()
            shot = self._window_capture_sct.grab(bbox)
            frame_bgra = np.asarray(shot, dtype=np.uint8)
            frame_rgb = np.ascontiguousarray(frame_bgra[:, :, 2::-1])
            frame_rgb = self._cap_window_capture_frame_size(frame_rgb)
            self.last_window_capture_image = frame_rgb
            self._record_window_capture_timing(capture_started)
            return frame_rgb
        except Exception:
            self._resolve_window_capture_target(force=True)
            try:
                if self._window_capture_sct is not None:
                    self._window_capture_sct.close()
            except Exception:
                pass
            self._window_capture_sct = None
            raise

    def _cap_window_capture_frame_size(self, frame):
        try:
            arr = np.asarray(frame)
            h, w = arr.shape[:2]
            max_w = max(1, int(self.display_width))
            max_h = max(1, int(self.display_height))
            if w <= max_w and h <= max_h:
                return frame
            scale = min(max_w / float(w), max_h / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame

    def _record_window_capture_timing(self, started):
        try:
            now = time.perf_counter()
            self._window_capture_last_ms = (now - float(started)) * 1000.0
            self._window_capture_fps_samples += 1
            elapsed = now - float(self._window_capture_fps_last_time or now)
            if elapsed >= 2.0:
                fps = self._window_capture_fps_samples / max(elapsed, 1e-6)
                logging.info(
                    "WINDOW CAPTURE FPS: %.1f grab/sec (last %.1fms, bbox=%s, scale=%.2f)",
                    fps,
                    self._window_capture_last_ms,
                    self._window_capture_last_bbox,
                    float(self.processing_scale),
                )
                self._window_capture_fps_samples = 0
                self._window_capture_fps_last_time = now
        except Exception:
            pass

    def start_webcam_capture(self, sender, app_data):
        self._release_webcam_capture()
        self.stop_all_media()
        try:
            cam_index = int(self.webcam_index)
        except Exception:
            cam_index = 0
            self.webcam_index = 0
        try:
            cap = self._open_webcam_capture(cam_index, quiet=False)
            if not cap or not cap.isOpened():
                raise RuntimeError(f"Unable to open webcam index {cam_index}")
            self.webcam_capture = cap
            self.webcam_running = True
            logging.info(f"Webcam capture started on index {cam_index}")
        except Exception as e:
            logging.error(f"Failed to start webcam capture: {e}")
            self.webcam_capture = None
            self.webcam_running = False

    def stop_webcam_capture(self, sender, app_data):
        self.webcam_running = False
        self._release_webcam_capture()

    def _release_webcam_capture(self):
        if self.webcam_capture is not None:
            try:
                self.webcam_capture.release()
            except Exception:
                pass
            self.webcam_capture = None

    def stop_all_media(self):
        self.gif_running = False
        self.window_capture_running = False
        self.last_window_capture_image = None
        self._release_window_capture_resources()
        self.webcam_running = False
        self._release_webcam_capture()
        if self.video_reader:
            self._close_media_reader(self.video_reader)
            self.video_reader = None
        self.video_playing = False
        self.video_refresh_requested = False
        self._clear_primary_video_frame_cache()
        with self._video_seek_lock:
            self._pending_video_seek_seconds = None
            self._pending_video_refresh = False
        self.image = None
        self.original_image_with_alpha = None
        self.gif_frames = []
        self.gif_durations = []
        self.gif_index = 0
        self.primary_media_path = None
        self._update_video_transport_visibility()
        self._sync_video_transport_ui(update_input=True)

    def _close_media_reader(self, reader):
        if not reader:
            return
        try:
            reader.close()
        except Exception:
            try:
                reader.release()
            except Exception:
                pass

    def _reader_get_fps(self, reader, fallback=None):
        fallback_fps = max(1.0, float(fallback if fallback is not None else (self.framerate or 30)))
        if not reader:
            return fallback_fps

        try:
            meta = reader.get_meta_data()
            fps = float(meta.get("fps") or 0.0)
            if fps > 0:
                return fps
        except Exception:
            pass

        try:
            fps = float(reader.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps > 0:
                return fps
        except Exception:
            pass

        return fallback_fps

    def _reader_get_frame_count(self, reader):
        if not reader:
            return 0

        try:
            meta = reader.get_meta_data()
            frame_count = int(meta.get("nframes"))
            if frame_count > 0:
                return frame_count
        except Exception:
            pass

        try:
            frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 0:
                return frame_count
        except Exception:
            pass

        return 0

    def _reader_get_frame_index(self, reader):
        if not reader:
            return 0

        try:
            frame_index = int(reader.get_next_index())
        except Exception:
            try:
                frame_index = int(reader.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            except Exception:
                frame_index = 0

        frame_count = self._reader_get_frame_count(reader)
        if frame_count > 0:
            frame_index = min(max(0, frame_index), max(0, frame_count - 1))
        return max(0, frame_index)

    def _reader_set_frame_index(self, reader, frame_index):
        if not reader:
            return False

        target_index = max(0, int(round(frame_index)))
        frame_count = self._reader_get_frame_count(reader)
        if frame_count > 0:
            target_index = min(target_index, max(0, frame_count - 1))

        try:
            reader.set_image_index(target_index)
            return True
        except Exception:
            pass

        try:
            return bool(reader.set(cv2.CAP_PROP_POS_FRAMES, target_index))
        except Exception:
            return False

    def _reader_get_time_seconds(self, reader, fallback_fps=None):
        fps = self._reader_get_fps(reader, fallback=fallback_fps)
        if fps <= 0:
            return 0.0
        return float(self._reader_get_frame_index(reader)) / fps

    def _seek_reader_to_seconds(self, reader, seconds):
        if not reader:
            return False

        fps = self._reader_get_fps(reader)
        frame_index = int(round(max(0.0, float(seconds)) * fps))
        return self._reader_set_frame_index(reader, frame_index)

    def _reader_get_duration_seconds(self, reader, fallback_fps=None):
        fps = self._reader_get_fps(reader, fallback=fallback_fps)
        frame_count = self._reader_get_frame_count(reader)
        if fps <= 0 or frame_count <= 0:
            return 0.0
        return float(frame_count) / fps

    def _format_timecode(self, seconds):
        total_seconds = max(0.0, float(seconds or 0.0))
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds_part = total_seconds - (hours * 3600) - (minutes * 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds_part:06.3f}"
        return f"{minutes}:{seconds_part:06.3f}"

    def _parse_timecode(self, raw_value):
        text = str(raw_value or "").strip()
        if not text:
            raise ValueError("Time value is empty")

        if ":" not in text:
            return max(0.0, float(text))

        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError(f"Unsupported timecode: {text}")

        multipliers = (1.0, 60.0, 3600.0)
        total_seconds = 0.0
        for idx, part in enumerate(reversed(parts)):
            total_seconds += float(part) * multipliers[idx]
        return max(0.0, total_seconds)

    def _set_export_duration(self, seconds, update_slider=True, update_input=True):
        duration = max(0.001, float(seconds))
        self.export_duration = duration
        if update_slider and hasattr(self, "export_duration_slider"):
            if dpg.does_item_exist(self.export_duration_slider):
                dpg.set_value(self.export_duration_slider, min(60.0, max(1.0, duration)))
        if update_input and hasattr(self, "export_duration_input"):
            if dpg.does_item_exist(self.export_duration_input):
                dpg.set_value(self.export_duration_input, self._format_timecode(duration))

    def _on_export_duration_slider(self, sender, app_data, user_data=None):
        self._set_export_duration(app_data, update_slider=False, update_input=True)

    def _on_export_duration_text(self, sender, app_data, user_data=None):
        try:
            self._set_export_duration(
                self._parse_timecode(app_data),
                update_slider=True,
                update_input=True,
            )
        except (TypeError, ValueError) as exc:
            logging.warning("Invalid export duration '%s': %s", app_data, exc)
            self._set_export_duration(
                self.export_duration,
                update_slider=True,
                update_input=True,
            )

    def _sync_video_transport_ui(self, update_input=False):
        if not dpg.does_item_exist("video_transport_status"):
            return

        transport_reader = self.video_reader or self.mixer_channel_b_video_reader
        if transport_reader:
            fps = self._reader_get_fps(transport_reader)
            frame_count = self._reader_get_frame_count(transport_reader)
            if transport_reader is self.video_reader and hasattr(self, "video_transport_frame"):
                if frame_count > 0:
                    frame_index = int(np.clip(int(self.video_transport_frame), 0, max(0, frame_count - 1)))
                else:
                    frame_index = max(0, int(self.video_transport_frame))
            else:
                frame_index = self._reader_get_frame_index(transport_reader)
            current_seconds = float(frame_index) / max(1.0, float(fps))
            duration_seconds = (float(frame_count) / max(1.0, float(fps))) if frame_count > 0 else 0.0
            frame_suffix = f" / {max(0, frame_count - 1)}" if frame_count > 0 else ""
            status_text = (
                f"{self._format_timecode(current_seconds)} / "
                f"{self._format_timecode(duration_seconds)} | "
                f"frame {frame_index}{frame_suffix}"
            )
            input_value = self._format_timecode(current_seconds)
        else:
            status_text = "0:00.000 / 0:00.000 | frame 0"
            input_value = "0:00.000"

        dpg.set_value("video_transport_status", status_text)
        if update_input and dpg.does_item_exist("video_transport_time_input"):
            dpg.set_value("video_transport_time_input", input_value)

    def _clear_primary_video_frame_cache(self):
        self._primary_video_last_frame = None
        self._primary_video_last_frame_index = None

    def _update_video_transport_visibility(self):
        try:
            if dpg.does_item_exist("video_transport_window"):
                dpg.configure_item(
                    "video_transport_window",
                    show=bool(self.video_reader or self.mixer_channel_b_video_reader),
                )
        except Exception:
            pass

    def _seek_video_to_seconds(self, seconds, refresh_frame=True):
        if not self.video_reader and not self.mixer_channel_b_video_reader:
            return False

        sought = False
        if self.video_reader:
            sought = self._seek_reader_to_seconds(self.video_reader, seconds) or sought
            self._clear_primary_video_frame_cache()
            self.video_transport_frame = float(self._reader_get_frame_index(self.video_reader))
            self.video_refresh_requested = bool(refresh_frame)
        if self.mixer_channel_b_video_reader:
            sought = self._seek_reader_to_seconds(self.mixer_channel_b_video_reader, seconds) or sought
            self.mixer_channel_b_video_refresh_requested = bool(refresh_frame)
            self.mixer_channel_b_last_video_frame = None
        self._sync_video_transport_ui(update_input=True)
        return sought

    def _queue_video_seek(self, seconds, refresh_frame=True):
        if not self.video_reader and not self.mixer_channel_b_video_reader:
            return False

        with self._video_seek_lock:
            self._pending_video_seek_seconds = max(0.0, float(seconds))
            self._pending_video_refresh = bool(refresh_frame)
        return True

    def _apply_pending_video_seek(self):
        if not self.video_reader and not self.mixer_channel_b_video_reader:
            with self._video_seek_lock:
                self._pending_video_seek_seconds = None
                self._pending_video_refresh = False
            return False

        with self._video_seek_lock:
            pending_seconds = self._pending_video_seek_seconds
            refresh_frame = self._pending_video_refresh
            self._pending_video_seek_seconds = None
            self._pending_video_refresh = False

        if pending_seconds is None:
            return False

        return self._seek_video_to_seconds(pending_seconds, refresh_frame=refresh_frame)

    def _capture_media_playback_state(self):
        return {
            "primary_video_index": self._reader_get_frame_index(self.video_reader),
            "primary_video_playing": bool(self.video_playing),
            "primary_gif_index": int(self.gif_index),
            "bg_video_index": self._reader_get_frame_index(self.bg_video_reader),
            "bg_gif_index": int(self.bg_gif_index),
            "mixer_video_index": self._reader_get_frame_index(self.mixer_channel_b_video_reader),
            "mixer_gif_index": int(self.mixer_channel_b_gif_index),
            "video_refresh_requested": bool(self.video_refresh_requested),
        }

    def _prime_media_for_export(self, playback_state):
        if not playback_state:
            return

        if self.video_reader:
            self._reader_set_frame_index(self.video_reader, playback_state.get("primary_video_index", 0))
        self.gif_index = int(playback_state.get("primary_gif_index", 0))
        self.last_gif_frame_time = time.time()

        if self.bg_video_reader:
            self._reader_set_frame_index(self.bg_video_reader, playback_state.get("bg_video_index", 0))
        self.bg_gif_index = int(playback_state.get("bg_gif_index", 0))

        if self.mixer_channel_b_video_reader:
            self._reader_set_frame_index(self.mixer_channel_b_video_reader, playback_state.get("mixer_video_index", 0))
        self.mixer_channel_b_gif_index = int(playback_state.get("mixer_gif_index", 0))
        self.mixer_channel_b_last_gif_frame_time = time.time()

    def _restore_media_playback_state(self, playback_state):
        if not playback_state:
            return

        self._prime_media_for_export(playback_state)
        self.video_playing = bool(playback_state.get("primary_video_playing", self.video_playing))
        self.video_refresh_requested = bool(playback_state.get("video_refresh_requested", False))
        self._sync_video_transport_ui(update_input=True)

    def _get_export_timeline_start(self, playback_state, target_fps):
        if self.video_reader and not self.image and not self.gif_frames:
            fps = self._reader_get_fps(self.video_reader, fallback=target_fps)
            return float(playback_state.get("primary_video_index", 0)) / max(1.0, fps)

        if self.gif_frames:
            return float(playback_state.get("primary_gif_index", 0)) / max(1.0, float(target_fps))

        return max(0.0, float(self.time_accum))

    def seek_video_to_time_input(self, sender, app_data):
        if not self.video_reader and not self.mixer_channel_b_video_reader:
            return

        try:
            raw_value = app_data if isinstance(app_data, str) else dpg.get_value("video_transport_time_input")
            seconds = self._parse_timecode(raw_value)
            if self._queue_video_seek(seconds, refresh_frame=True):
                if dpg.does_item_exist("video_transport_time_input"):
                    dpg.set_value("video_transport_time_input", self._format_timecode(seconds))
                logging.info("Video seek: %.3fs", seconds)
        except Exception as exc:
            logging.warning("Invalid video seek value: %s", exc)
            self._sync_video_transport_ui(update_input=True)

    def _set_mixer_channel_b_status(self, text, color=(150, 150, 150)):
        self.mixer_channel_b_label = text
        if dpg.does_item_exist("mixer_channel_b_status"):
            dpg.set_value("mixer_channel_b_status", text)
            dpg.configure_item("mixer_channel_b_status", color=color)

    def _clear_mixer_channel_b_media(self):
        self.mixer_channel_b_image = None
        self.mixer_channel_b_gif_frames = []
        self.mixer_channel_b_gif_durations = []
        self.mixer_channel_b_gif_index = 0
        self.mixer_channel_b_last_gif_frame_time = time.time()
        self._close_media_reader(self.mixer_channel_b_video_reader)
        self.mixer_channel_b_video_reader = None
        self.mixer_channel_b_video_refresh_requested = False
        self.mixer_channel_b_last_video_frame = None
        self.mixer_channel_b_media_path = None

    def clear_mixer_channel_b(self, sender, app_data):
        self._clear_mixer_channel_b_media()
        if not self.video_reader:
            self.video_playing = False
        self._update_video_transport_visibility()
        self._sync_video_transport_ui(update_input=True)
        self._set_mixer_channel_b_status("No Channel B media loaded")

    def open_mixer_channel_b(self, sender, app_data):
        self.processing_paused = True
        if dpg.does_item_exist("file_dialog_mixer_channel_b"):
            dpg.delete_item("file_dialog_mixer_channel_b")
        with dpg.file_dialog(directory_selector=False, show=True, callback=self.load_mixer_channel_b_callback,
                            tag="file_dialog_mixer_channel_b", modal=True, file_count=1,
                            cancel_callback=lambda: setattr(self, 'processing_paused', False)):
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".gif")
            dpg.add_file_extension(".mp4")
            dpg.add_file_extension(".avi")
            dpg.add_file_extension(".mov")
            dpg.add_file_extension(".mkv")
            dpg.add_file_extension(".*")

    def load_mixer_channel_b_callback(self, sender, app_data):
        self.processing_paused = False
        path = app_data['file_path_name']
        if not path:
            return

        self._clear_mixer_channel_b_media()

        try:
            self.mixer_channel_b_media_path = str(path)
            if path.lower().endswith('.gif'):
                self.mixer_channel_b_gif_frames, self.mixer_channel_b_gif_durations = open_gif_file(path)
                self.mixer_channel_b_last_gif_frame_time = time.time()
                self._set_mixer_channel_b_status(
                    f"Channel B GIF: {len(self.mixer_channel_b_gif_frames)} frames",
                    (100, 255, 100)
                )
            elif path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.mixer_channel_b_video_reader = open_video_file(path)
                if not self.mixer_channel_b_video_reader:
                    raise RuntimeError("Failed to open Channel B video")
                self.mixer_channel_b_video_refresh_requested = True
                self.mixer_channel_b_last_video_frame = None
                self.video_playing = True

                label = "Channel B video loaded"
                try:
                    meta = self.mixer_channel_b_video_reader.get_meta_data()
                    fps = meta.get('fps')
                    if fps:
                        label = f"Channel B video: {fps:.1f} fps"
                except Exception:
                    pass
                self._set_mixer_channel_b_status(label, (100, 255, 100))
            else:
                self.mixer_channel_b_image = Image.open(path).convert('RGBA')
                self._set_mixer_channel_b_status(
                    f"Channel B image: {self.mixer_channel_b_image.size[0]}x{self.mixer_channel_b_image.size[1]}",
                    (100, 255, 100)
                )

            self.mixer_enabled = True
            if dpg.does_item_exist("mixer_enabled_checkbox"):
                dpg.set_value("mixer_enabled_checkbox", True)
            self.node_editor_b.mark_pipeline_dirty()
            self._update_video_transport_visibility()
            self._sync_video_transport_ui(update_input=True)
            try:
                while not self.raw_frame_queue.empty():
                    stale_input = self.raw_frame_queue.get_nowait()
                    if self.shared_frame_transport is not None:
                        self.shared_frame_transport.release_input_payload(stale_input)
            except Exception:
                pass
            try:
                while not self.glitched_frame_queue.empty():
                    stale_result = self.glitched_frame_queue.get_nowait()
                    if self.shared_frame_transport is not None:
                        self.shared_frame_transport.release_output_result(stale_result)
            except Exception:
                pass
            try:
                self.raw_frame_queue.put_nowait("RESET")
            except Exception:
                pass
            logging.info(f"Loaded Channel B mixer media: {path}")
        except Exception as e:
            logging.error(f"Error loading Channel B media: {e}", exc_info=True)
            self._set_mixer_channel_b_status(f"Channel B error: {str(e)[:40]}", (255, 100, 100))
        finally:
            try:
                dpg.delete_item("file_dialog_mixer_channel_b")
            except Exception:
                pass

    def _has_mixer_channel_b_media(self):
        return bool(
            self.mixer_channel_b_image is not None or
            self.mixer_channel_b_gif_frames or
            self.mixer_channel_b_video_reader is not None
        )

    def _get_current_mixer_channel_b_frame(self, export_mode=False):
        if self.mixer_channel_b_image:
            return self.mixer_channel_b_image.copy()

        if self.mixer_channel_b_gif_frames:
            if export_mode:
                frame = self.mixer_channel_b_gif_frames[
                    self.mixer_channel_b_gif_index % len(self.mixer_channel_b_gif_frames)
                ].copy()
                self.mixer_channel_b_gif_index = (self.mixer_channel_b_gif_index + 1) % len(self.mixer_channel_b_gif_frames)
                return frame

            frame_duration_sec = 0.1
            if self.mixer_channel_b_gif_durations:
                frame_duration_sec = max(
                    0.001,
                    self.mixer_channel_b_gif_durations[self.mixer_channel_b_gif_index % len(self.mixer_channel_b_gif_durations)] / 1000.0
                )
            if time.time() - self.mixer_channel_b_last_gif_frame_time >= frame_duration_sec:
                self.mixer_channel_b_gif_index = (self.mixer_channel_b_gif_index + 1) % len(self.mixer_channel_b_gif_frames)
                self.mixer_channel_b_last_gif_frame_time = time.time()
            return self.mixer_channel_b_gif_frames[self.mixer_channel_b_gif_index].copy()

        if self.mixer_channel_b_video_reader:
            if not export_mode and not self.video_playing and not self.mixer_channel_b_video_refresh_requested:
                if self.mixer_channel_b_last_video_frame is not None:
                    return self.mixer_channel_b_last_video_frame.copy()
                return None
            try:
                raw = self.mixer_channel_b_video_reader.get_next_data()
                frame = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_RGB2RGBA))
                self.mixer_channel_b_last_video_frame = frame.copy()
                self.mixer_channel_b_video_refresh_requested = False
                return frame
            except (IndexError, StopIteration):
                try:
                    self.mixer_channel_b_video_reader.set_image_index(0)
                    raw = self.mixer_channel_b_video_reader.get_next_data()
                    frame = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_RGB2RGBA))
                    self.mixer_channel_b_last_video_frame = frame.copy()
                    self.mixer_channel_b_video_refresh_requested = False
                    return frame
                except Exception as e:
                    logging.error(f"Error restarting Channel B video: {e}")
                    if self.mixer_channel_b_video_reader:
                        self._close_media_reader(self.mixer_channel_b_video_reader)
                        self.mixer_channel_b_video_reader = None
                    return None
            except Exception as e:
                logging.error(f"Error reading Channel B video frame: {e}")
                self._close_media_reader(self.mixer_channel_b_video_reader)
                self.mixer_channel_b_video_reader = None
                return None

        return None

    def _mix_source_frames(self, channel_a, channel_b):
        if channel_a is None or channel_b is None:
            return channel_a

        mix = float(max(0.0, min(1.0, self.mixer_mix)))
        if mix <= 0.0:
            return channel_a

        frame_a = channel_a.convert('RGBA')
        frame_b = channel_b.convert('RGBA')
        if frame_b.size != frame_a.size:
            frame_b = frame_b.resize(frame_a.size, Image.Resampling.LANCZOS)

        arr_a = np.asarray(frame_a, dtype=np.float32)
        arr_b = np.asarray(frame_b, dtype=np.float32)
        rgb_a = arr_a[..., :3]
        rgb_b = arr_b[..., :3]
        alpha_a = arr_a[..., 3:4]
        alpha_b = arr_b[..., 3:4]

        mode = str(self.mixer_mode or "Crossfade")
        if mode == "Add":
            mixed_rgb = np.clip(rgb_a + rgb_b, 0, 255)
            mixed_alpha = np.maximum(alpha_a, alpha_b)
        elif mode == "Multiply":
            mixed_rgb = (rgb_a * rgb_b) / 255.0
            mixed_alpha = np.maximum(alpha_a, alpha_b)
        elif mode == "Screen":
            mixed_rgb = 255.0 - ((255.0 - rgb_a) * (255.0 - rgb_b) / 255.0)
            mixed_alpha = np.maximum(alpha_a, alpha_b)
        elif mode == "Difference":
            mixed_rgb = np.abs(rgb_a - rgb_b)
            mixed_alpha = np.maximum(alpha_a, alpha_b)
        elif mode == "Lighten":
            mixed_rgb = np.maximum(rgb_a, rgb_b)
            mixed_alpha = np.maximum(alpha_a, alpha_b)
        elif mode == "Darken":
            mixed_rgb = np.minimum(rgb_a, rgb_b)
            mixed_alpha = np.maximum(alpha_a, alpha_b)
        else:
            mixed_rgb = rgb_b
            mixed_alpha = alpha_b

        out_rgb = rgb_a * (1.0 - mix) + mixed_rgb * mix
        out_alpha = alpha_a * (1.0 - mix) + mixed_alpha * mix
        out = np.concatenate([
            np.clip(out_rgb, 0, 255),
            np.clip(out_alpha, 0, 255)
        ], axis=2).astype(np.uint8)
        return Image.fromarray(out, mode='RGBA')

    def _apply_two_channel_mixer(self, frame, export_mode=False):
        if not self.mixer_enabled or not self._has_mixer_channel_b_media() or frame is None:
            return frame

        if not isinstance(frame, Image.Image):
            arr = np.asarray(frame)
            if arr.ndim == 3 and arr.shape[2] == 4:
                frame = Image.fromarray(arr.astype(np.uint8), mode='RGBA')
            else:
                frame = Image.fromarray(self._to_virtual_cam_rgb(arr), mode='RGB').convert('RGBA')

        channel_b = self._get_current_mixer_channel_b_frame(export_mode=export_mode)
        if channel_b is None:
            return frame
        return self._mix_source_frames(frame, channel_b)
    
    def open_bg_media(self, sender, app_data):
        """Open file dialog to load background media."""
        self.processing_paused = True  # Pause during dialog
        if dpg.does_item_exist("file_dialog_bg_media"):
            dpg.delete_item("file_dialog_bg_media")
        with dpg.file_dialog(directory_selector=False, show=True, callback=self.load_bg_media_callback, 
                            tag="file_dialog_bg_media", modal=True, file_count=1,
                            cancel_callback=lambda: setattr(self, 'processing_paused', False)):
            dpg.add_file_extension(".png")
            dpg.add_file_extension(".jpg")
            dpg.add_file_extension(".jpeg")
            dpg.add_file_extension(".gif")
            dpg.add_file_extension(".mp4")
            dpg.add_file_extension(".avi")
            dpg.add_file_extension(".mov")
            dpg.add_file_extension(".mkv")
            dpg.add_file_extension(".*")
    
    def load_bg_media_callback(self, sender, app_data):
        """Load background media (image/video/gif)."""
        self.processing_paused = False  # Resume processing
        path = app_data['file_path_name']
        if not path:
            return
        
        # Clear previous background media
        self.bg_media = None
        self.bg_gif_frames = []
        self.bg_gif_index = 0
        if self.bg_video_reader:
            self.bg_video_reader.release()
            self.bg_video_reader = None
        self.background_media_path = None
        
        try:
            self.background_media_path = str(path)
            # Detect file type
            if path.lower().endswith('.gif'):
                # Load GIF
                from PIL import ImageSequence
                gif = Image.open(path)
                self.bg_gif_frames = [frame.copy().convert('RGBA') for frame in ImageSequence.Iterator(gif)]
                if dpg.does_item_exist("bg_media_status"):
                    dpg.set_value("bg_media_status", f"GIF loaded: {len(self.bg_gif_frames)} frames")
                    dpg.configure_item("bg_media_status", color=(100, 255, 100))
                logging.info(f"Background GIF loaded: {path}")
                
            elif path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Load video
                self.bg_video_reader = cv2.VideoCapture(path)
                if self.bg_video_reader.isOpened():
                    frame_count = int(self.bg_video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
                    if dpg.does_item_exist("bg_media_status"):
                        dpg.set_value("bg_media_status", f"Video loaded: {frame_count} frames")
                        dpg.configure_item("bg_media_status", color=(100, 255, 100))
                    logging.info(f"Background video loaded: {path}")
                else:
                    raise Exception("Failed to open video")
                    
            else:
                # Load static image
                self.bg_media = Image.open(path).convert('RGBA')
                if dpg.does_item_exist("bg_media_status"):
                    dpg.set_value("bg_media_status", f"Image loaded: {self.bg_media.size[0]}x{self.bg_media.size[1]}")
                    dpg.configure_item("bg_media_status", color=(100, 255, 100))
                logging.info(f"Background image loaded: {path}")
            
            # Auto-switch to Media preset
            if dpg.does_item_exist("bg_preset_combo"):
                dpg.set_value("bg_preset_combo", "Media")
                self.change_image_background(None, "Media")
                
        except Exception as e:
            logging.error(f"Error loading background media: {e}")
            if dpg.does_item_exist("bg_media_status"):
                dpg.set_value("bg_media_status", f"Error: {str(e)[:30]}")
                dpg.configure_item("bg_media_status", color=(255, 100, 100))
        
        dpg.delete_item("file_dialog_bg_media")
        # Reset glitch worker state and clear queues
        try:
            if self.raw_frame_queue:
                self.raw_frame_queue.put("RESET")
        except Exception:
            pass
        try:
            while not self.raw_frame_queue.empty():
                stale_input = self.raw_frame_queue.get_nowait()
                if self.shared_frame_transport is not None:
                    self.shared_frame_transport.release_input_payload(stale_input)
        except Exception:
            pass
        try:
            while not self.glitched_frame_queue.empty():
                stale_result = self.glitched_frame_queue.get_nowait()
                if self.shared_frame_transport is not None:
                    self.shared_frame_transport.release_output_result(stale_result)
        except Exception:
            pass
        self._update_video_transport_visibility()

    def _cleanup_failed_worker_startup(self):
        """Release host resources when the processing worker never becomes ready."""
        self.processing_paused = True
        try:
            if self.processing_lifecycle is not None:
                self.processing_lifecycle.stop()
        except Exception:
            logging.debug("Failed worker lifecycle cleanup encountered an issue", exc_info=True)
        for cleanup in (
            self._release_window_capture_resources,
            self._release_webcam_capture,
            self._stop_virtual_cam_sender,
            self._close_virtual_camera,
        ):
            try:
                cleanup()
            except Exception:
                logging.debug("Partial startup resource cleanup failed", exc_info=True)
        for reader_name in ("video_reader", "mixer_channel_b_video_reader", "bg_video_reader"):
            reader = getattr(self, reader_name, None)
            try:
                self._close_media_reader(reader)
            except Exception:
                try:
                    if reader is not None:
                        reader.release()
                except Exception:
                    pass
            setattr(self, reader_name, None)
        try:
            self.stop_live_sequencer()
        except Exception:
            pass
        try:
            if self.audio_midi:
                self.audio_midi.shutdown()
        except Exception:
            pass
        self.audio_midi = None
        try:
            if self.media_audio:
                self.media_audio.destroy()
        except Exception:
            pass
        for editor_name in ("node_editor_a", "node_editor_b"):
            try:
                getattr(self, editor_name).destroy_audio_nodes()
            except Exception:
                pass
        if self.shared_frame_transport is not None:
            try:
                self.shared_frame_transport.close()
            except Exception:
                pass
            self.shared_frame_transport = None
        for processing_queue in (self.raw_frame_queue, self.glitched_frame_queue):
            try:
                processing_queue.close()
                processing_queue.cancel_join_thread()
            except Exception:
                pass
        try:
            dpg.destroy_context()
        except Exception:
            pass

    def run(self):
        runtime_backend = os.getenv("MEGAGLITCH_RUNTIME_BACKEND", "python").strip().lower()
        transport_backend = os.getenv("MEGAGLITCH_FRAME_TRANSPORT", "shared").strip().lower()
        worker_target = glitch_process_worker
        worker_args = (self.raw_frame_queue, self.glitched_frame_queue, self.stop_glitch_process_event, None)
        if runtime_backend == "native":
            from megaglitch.native_runtime_worker import native_runtime_process_worker
            worker_target = native_runtime_process_worker
            worker_args = worker_args + (glitch_process_worker,)
            logging.info("Native Rust runtime requested; unsupported graphs will fall back to Python")
        elif transport_backend == "shared":
            try:
                from megaglitch.shared_frame_transport import SharedFrameTransportOwner
                max_width = max(
                    int(self.display_width),
                    int(os.getenv("MEGAGLITCH_SHARED_FRAME_MAX_WIDTH", "1920")),
                )
                max_height = max(
                    int(self.display_height),
                    int(os.getenv("MEGAGLITCH_SHARED_FRAME_MAX_HEIGHT", "1080")),
                )
                self.shared_frame_transport = SharedFrameTransportOwner(
                    mp.get_context(), max_width, max_height, channels=3,
                    input_slots=self.queue_max_size * 2,
                    output_slots=self.queue_max_size,
                )
                worker_args = worker_args + (self.shared_frame_transport.spec(),)
                logging.info(
                    "Shared frame transport enabled: %sx%s, %s bytes",
                    max_width, max_height,
                    self.shared_frame_transport.get_statistics()["allocated_bytes"],
                )
            except Exception:
                self.shared_frame_transport = None
                logging.exception("Shared frame transport initialization failed; using legacy queues")
        from megaglitch.worker_lifecycle import WorkerProcessLifecycle
        startup_timeout = float(os.getenv("MEGAGLITCH_WORKER_STARTUP_TIMEOUT", "30"))
        self.processing_lifecycle = WorkerProcessLifecycle(
            mp.get_context(), self.stop_glitch_process_event,
            startup_timeout=startup_timeout,
            graceful_timeout=5.0,
            terminate_timeout=2.0,
        )
        try:
            self.glitch_process = self.processing_lifecycle.start(worker_target, worker_args)
            logging.info(
                "Processing worker ready: %s",
                self.processing_lifecycle.snapshot(),
            )
        except Exception:
            logging.exception("Processing worker failed during startup")
            self._cleanup_failed_worker_startup()
            raise
        last_frame_time = time.time()
        try:
            import analog_clock
            while dpg.is_dearpygui_running():
                lifecycle_state = self.processing_lifecycle.poll()
                if lifecycle_state.value == "failed":
                    if not self._processing_failure_reported:
                        logging.error(
                            "Processing worker failed while running: %s",
                            self.processing_lifecycle.error_message(),
                        )
                        self._processing_failure_reported = True
                    break
                time_macros = self._get_global_time_domain_macro_values()
                analog_clock.update_clocks(self.time_accum, time_macros)
                target_framerate = max(1, int(self.framerate))
                frame_delay = 1.0 / target_framerate
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # Skip frame delay when recording to process at maximum speed
                is_recording = self.is_recording_gif or self.is_recording_mp4
                should_process_frame = is_recording or elapsed >= frame_delay
                seek_applied = self._apply_pending_video_seek()
                if seek_applied:
                    current_time = time.time()
                    last_frame_time = current_time
                    elapsed = 0.0
                    should_process_frame = True

                if should_process_frame:
                    advance_clock = is_recording or elapsed >= frame_delay
                    # Update frame time - important for maintaining timing after recording
                    if advance_clock:
                        last_frame_time = current_time
                    # Advance the time accumulator in seconds scaled by user control
                    if advance_clock:
                        global_macros = self._get_global_time_domain_macro_values()
                        reversal = float(global_macros.get("time_reversal", 0.0))
                        warp = float(global_macros.get("local_time_warp", 0.0))
                        flow_rate = (1.0 - 2.0 * reversal) * (1.0 + warp)
                        
                        try:
                            self.time_accum += elapsed * float(self.time_scale) * flow_rate
                        except Exception:
                            self.time_accum += elapsed * flow_rate
                        
                        # Increment frame counter
                        self.frame_counter += 1

                        if dpg.does_item_exist("ml_rand_continuous") and dpg.get_value("ml_rand_continuous"):
                            now = time.time()
                            if not hasattr(self, "last_ml_mutate_time"):
                                self.last_ml_mutate_time = now
                            rate = dpg.get_value("ml_rand_rate")
                            if now - self.last_ml_mutate_time > rate:
                                self.trigger_ml_mutation()
                                self.last_ml_mutate_time = now

                        
                        # Debug logging for timing (every 30 frames)
                        if self.frame_counter % 30 == 0:
                            logging.debug(f"[MAIN] Frame {self.frame_counter}: time_accum_wallclock={self.time_accum:.3f}s, time_accum_frame_based={self.frame_counter/60.0:.3f}s")

                    audio_dt = max(0.0, min(float(elapsed), 0.25))
                    self.node_editor_a.update_audio_nodes(self.time_accum, audio_dt, self.time_accum)
                    self.node_editor_b.update_audio_nodes(self.time_accum, audio_dt, self.time_accum)
                    
                    # Skip heavy processing if paused (e.g., during file dialogs or batch rendering)
                    if self.processing_paused or self.batch_rendering:
                        dpg.render_dearpygui_frame()
                        continue
                    
                    current_image = None
                    secondary_image = None
                    timeline_seconds = None
                    timeline_frame = None
                    # Only produce frames if the pipeline reaches the Display node
                    
                    processing_pipeline = self.node_editor_a.get_processing_pipeline()
                    pipeline_b = self.node_editor_b.get_processing_pipeline()

                    # Deck B's ordinary Source socket always represents Channel B
                    # media. Normalize cached/legacy pipelines as well as newly
                    # compiled ones so old presets cannot silently read Deck A.
                    for effect_config in pipeline_b or []:
                        if effect_config.get("node_type") == "source":
                            effect_config["source_key"] = "secondary"

                    deck_a_connected = bool(processing_pipeline) and processing_pipeline[-1].get("effect_type") == "Display"
                    deck_b_connected = bool(pipeline_b) and pipeline_b[-1].get("effect_type") == "Display"
                    connected_to_display = deck_a_connected or deck_b_connected
                    
                    # A generator on either deck can drive the master without source media.
                    has_generator_a = self._pipeline_starts_with_generator(processing_pipeline)
                    has_generator_b = self._pipeline_starts_with_generator(pipeline_b)
                    has_generator = has_generator_a or has_generator_b
                    uses_secondary_source_a = self._pipeline_uses_secondary_source(processing_pipeline)
                    uses_secondary_source_b = self._pipeline_uses_secondary_source(pipeline_b)
                    uses_secondary_source = uses_secondary_source_a or uses_secondary_source_b

                    if connected_to_display and self.image:
                        # Re-apply background for animated media (GIF/video backgrounds)
                        preset = dpg.get_value("bg_preset_combo") if dpg.does_item_exist("bg_preset_combo") else "Black"
                        has_animated_bg = (preset == "Media") and (self.bg_gif_frames or self.bg_video_reader)
                        
                        if has_animated_bg and self.original_image_with_alpha:
                            # Update background frame for animated backgrounds
                            current_image = self._apply_background_to_image(self.original_image_with_alpha.copy())
                            # Also update self.image for consistency
                            self.image = current_image.copy()
                        else:
                            current_image = self.image.copy()
                    elif connected_to_display and self._background_media_active():
                        bg_frame = self._get_current_bg_frame()
                        if bg_frame is not None:
                            if bg_frame.size != (self.display_width, self.display_height):
                                bg_frame = bg_frame.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
                            current_image = bg_frame.convert('RGBA')
                    elif connected_to_display and self.gif_running and self.gif_frames:
                        global_macros = self._get_global_time_domain_macro_values()
                        reversal = float(global_macros.get("time_reversal", 0.0))
                        warp = float(global_macros.get("local_time_warp", 0.0))
                        flow_rate = (1.0 - 2.0 * reversal) * (1.0 + warp)
                        
                        total_gif_dur = sum(self.gif_durations) / 1000.0
                        if total_gif_dur > 0:
                            if not hasattr(self, 'gif_transport_time'):
                                self.gif_transport_time = 0.0
                            self.gif_transport_time += elapsed * flow_rate
                            # Wrap around
                            self.gif_transport_time %= total_gif_dur
                            if self.gif_transport_time < 0:
                                self.gif_transport_time += total_gif_dur
                                
                            accum = 0.0
                            target_index = 0
                            for i, dur in enumerate(self.gif_durations):
                                dur_sec = dur / 1000.0
                                if accum <= self.gif_transport_time < accum + dur_sec:
                                    target_index = i
                                    break
                                accum += dur_sec
                                
                            self.gif_index = target_index
                            current_image = self.gif_frames[self.gif_index].copy()
                            timeline_frame = int(self.gif_index)
                            timeline_seconds = float(self.gif_index) / max(1.0, float(self.framerate))

                    elif connected_to_display and self.video_reader and (self.video_playing or self.video_refresh_requested):
                        try:
                            current_video_fps = self._reader_get_fps(self.video_reader)
                            frame_count = self._reader_get_frame_count(self.video_reader)
                            
                            global_macros = self._get_global_time_domain_macro_values()
                            reversal = float(global_macros.get("time_reversal", 0.0))
                            warp = float(global_macros.get("local_time_warp", 0.0))
                            flow_rate = (1.0 - 2.0 * reversal) * (1.0 + warp)
                            
                            if not hasattr(self, 'video_transport_frame'):
                                self.video_transport_frame = float(self._reader_get_frame_index(self.video_reader))
                                
                            if self.video_playing:
                                # Keep live preview sequential. Chasing wall-clock time after
                                # a slow seek/decode frame can force a seek every frame.
                                transport_elapsed = frame_delay
                                self.video_transport_frame += (transport_elapsed * flow_rate) * current_video_fps
                                if frame_count > 0:
                                    self.video_transport_frame %= frame_count
                                    if self.video_transport_frame < 0:
                                        self.video_transport_frame += frame_count
                                        
                            target_frame_index = int(self.video_transport_frame)
                            use_cached_frame = (
                                not self.video_refresh_requested
                                and self._primary_video_last_frame is not None
                                and self._primary_video_last_frame_index == target_frame_index
                            )

                            if use_cached_frame:
                                raw = self._primary_video_last_frame
                            else:
                                current_reader_index = self._reader_get_frame_index(self.video_reader)

                                # Only seek when transport jumps/reverses. Duplicate 60fps preview
                                # frames should reuse the last decoded source frame instead.
                                if target_frame_index != current_reader_index or self.video_refresh_requested:
                                    self._reader_set_frame_index(self.video_reader, target_frame_index)

                                raw = self.video_reader.get_next_data()
                                raw = self._to_virtual_cam_rgb(raw)
                                self._primary_video_last_frame = raw
                                self._primary_video_last_frame_index = target_frame_index

                            current_image = raw
                            timeline_frame = target_frame_index
                            timeline_seconds = float(target_frame_index) / max(1.0, current_video_fps)
                            self.video_refresh_requested = False
                        except (IndexError, StopIteration):
                            try:
                                self.video_reader.set_image_index(0) # Loop
                                if hasattr(self, 'video_transport_frame'):
                                    self.video_transport_frame = 0.0
                                raw = self.video_reader.get_next_data()
                                raw = self._to_virtual_cam_rgb(raw)
                                self._primary_video_last_frame = raw
                                self._primary_video_last_frame_index = 0
                                current_image = raw
                                timeline_frame = 0
                                timeline_seconds = 0.0
                                self.video_refresh_requested = False
                            except Exception as loop_exc:
                                logging.error(f"Error restarting video frame read: {loop_exc}")
                                self._close_media_reader(self.video_reader)
                                self.video_reader = None
                                self.video_playing = False
                                self.video_refresh_requested = False
                                self._clear_primary_video_frame_cache()
                                self._update_video_transport_visibility()
                        except Exception as e:
                            logging.error(f"Error reading video frame: {e}")
                            self._close_media_reader(self.video_reader)
                            self.video_reader = None
                            self.video_playing = False
                            self.video_refresh_requested = False
                            self._clear_primary_video_frame_cache()
                            self._update_video_transport_visibility()
                    elif connected_to_display and self.window_capture_running:
                        try:
                            current_image = self._capture_selected_window_frame()
                        except Exception as capture_err:
                            if self.frame_counter % 120 == 0:
                                logging.warning(f"Window capture failed for '{self.selected_window}': {capture_err}")
                    elif connected_to_display and self.webcam_running and self.webcam_capture is not None:
                        try:
                            ok, frame_bgr = self.webcam_capture.read()
                            if ok and frame_bgr is not None and getattr(frame_bgr, "size", 0) > 0:
                                current_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA))
                            else:
                                logging.warning("Webcam frame read failed; stopping webcam capture.")
                                self.stop_webcam_capture(None, None)
                        except Exception as webcam_err:
                            logging.error(f"Webcam capture error: {webcam_err}")
                            self.stop_webcam_capture(None, None)
                    elif connected_to_display and has_generator:
                        # Generator doesn't need source media - create a dummy frame
                        # The generator will replace this entirely
                        current_image = Image.new('RGB', (self.display_width, self.display_height), (0, 0, 0))

                    needs_channel_b_frame = (
                        connected_to_display
                        and (
                            uses_secondary_source
                            or (deck_b_connected and self._has_mixer_channel_b_media())
                        )
                    )
                    if needs_channel_b_frame:
                        secondary_image = self._get_current_mixer_channel_b_frame(export_mode=False)

                    if current_image is not None or secondary_image is not None:
                        # Legacy live recording requests bounded backpressure.  It
                        # can still drop after the one-second pool/queue timeout;
                        # Offline Master is the ordered, genuinely no-drop path.
                        is_recording = self.is_recording_gif or self.is_recording_mp4
                        self._prime_adaptive_processing_scale_for_pipeline(processing_pipeline, is_recording=is_recording)
                        self._prime_adaptive_processing_scale_for_pipeline(pipeline_b, is_recording=is_recording)

                        passthrough_active = (
                            not is_recording
                            and deck_a_connected
                            and not deck_b_connected
                            and current_image is not None
                            and secondary_image is None
                            and not has_generator_a
                            and not uses_secondary_source
                            and self._is_passthrough_pipeline(processing_pipeline)
                        )
                        if passthrough_active:
                            try:
                                passthrough_master = prepare_master_ingress(
                                    current_image,
                                    fallback_size=(self.display_width, self.display_height),
                                )
                                master_frame = passthrough_master.primary
                                if master_frame is not None:
                                    display_frame = master_frame
                                    if self.retro_mode_enabled:
                                        display_frame = apply_retro_mode(
                                            display_frame,
                                            self.retro_bits_per_channel,
                                            self.retro_dither_enabled,
                                        )
                                    self.update_visual_metrics_from_frame(display_frame)
                                    preview_frame = make_egress_frame(
                                        display_frame,
                                        passthrough_master.resolution,
                                        role=FrameRole.PREVIEW,
                                        scale=self._effective_processing_scale(is_recording=False),
                                    ).data
                                    for editor in (self.node_editor_a, self.node_editor_b):
                                        display_node = editor.get_display_node()
                                        if display_node is None:
                                            continue
                                        update_display_texture(
                                            preview_frame,
                                            display_node.texture_tag,
                                            display_node.display_width,
                                            display_node.display_height,
                                        )
                                    if self.virtual_cam_running and self.virtual_cam is not None:
                                        self._enqueue_virtual_cam_frame(master_frame)
                                self._sync_video_transport_ui(update_input=False)
                                dpg.render_dearpygui_frame()
                                continue
                            except Exception as passthrough_err:
                                logging.debug(f"Passthrough fast-path failed, falling back to worker: {passthrough_err}")
                        
                        # Only drain stale input for low latency when not recording.
                        # Legacy live capture instead waits for bounded backpressure.
                        if not is_recording and not self.raw_frame_queue.full():
                            # Drop stale frames to keep latency low and sliders responsive
                            try:
                                while self.raw_frame_queue.qsize() > 0:
                                    self.raw_frame_queue.get_nowait()
                            except Exception:
                                pass
                        
                        timeline_eval_seconds = float(timeline_seconds) if timeline_seconds is not None else float(self.time_accum)
                        if self.parameter_timeline:
                            try:
                                timeline_changes = self.parameter_timeline.apply_to_pipeline(
                                    processing_pipeline,
                                    timeline_eval_seconds,
                                )
                                timeline_changes.update(
                                    self.parameter_timeline.apply_to_pipeline(
                                        pipeline_b,
                                        timeline_eval_seconds,
                                    )
                                )
                                self.parameter_timeline.sync_controls(
                                    [self.node_editor_a, self.node_editor_b],
                                    timeline_changes,
                                )
                            except Exception as e:
                                logging.debug(f"Could not apply parameter timeline: {e}")

                        self._apply_srt_shader_cues_to_pipelines(
                            processing_pipeline,
                            pipeline_b,
                            timeline_eval_seconds,
                        )

                        # Keep deck parameter namespaces isolated. Merging both
                        # dictionaries lets common names from Deck B overwrite
                        # Deck A before either pipeline reaches the worker.
                        global_macro_values = self._get_global_time_domain_macro_values()

                        def collect_deck_params(deck_pipeline):
                            collected = {}
                            for effect_config in deck_pipeline:
                                if effect_config.get("effect_type") == "Display":
                                    continue
                                effect_params = effect_config.get("params", {}) or {}
                                collected.update(effect_params)
                            return collected

                        deck_a_params = collect_deck_params(processing_pipeline)
                        deck_b_params = collect_deck_params(pipeline_b)
                        modulated_a = dict(deck_a_params)
                        modulated_b = dict(deck_b_params)
                        modulated_globals = dict(global_macro_values)
                        if self.mod_matrix:
                            try:
                                matrix_editor = getattr(self.mod_matrix, "node_editor", self.node_editor_a)
                                if matrix_editor is self.node_editor_a:
                                    modulated_a = self.mod_matrix.apply(
                                        {**deck_a_params, **global_macro_values},
                                        time_scale=self.time_scale,
                                    )
                                elif matrix_editor is self.node_editor_b:
                                    modulated_b = self.mod_matrix.apply(
                                        {**deck_b_params, **global_macro_values},
                                        time_scale=self.time_scale,
                                    )
                                modulated_globals = self.mod_matrix.apply(
                                    global_macro_values,
                                    time_scale=self.time_scale,
                                )
                                # Visually update modulated parameters
                                if self.mod_matrix.enabled:
                                    for editor, deck_values in (
                                        (self.node_editor_a, modulated_a),
                                        (self.node_editor_b, modulated_b),
                                    ):
                                        for node in getattr(editor, "nodes", []):
                                            if hasattr(node, "param_controls"):
                                                for p_name, ctrl_tag in node.param_controls.items():
                                                    if p_name in deck_values:
                                                        is_targeted = False
                                                        for conn in getattr(self.mod_matrix, "connections", []):
                                                            if conn.endswith(f"-> {p_name}") or conn.endswith(f"->{p_name}"):
                                                                is_targeted = True
                                                                break
                                                        if is_targeted:
                                                            try:
                                                                dpg.set_value(ctrl_tag, deck_values[p_name])
                                                            except Exception:
                                                                pass
                            except Exception:
                                pass
                        active_time_domain_macros = self._get_active_time_domain_macros(modulated_globals)
                        # Push only each deck's own values back into its pipeline.
                        try:
                            for deck_pipeline, deck_values in (
                                (processing_pipeline, modulated_a),
                                (pipeline_b, modulated_b),
                            ):
                                for effect_config in deck_pipeline:
                                    if effect_config.get("effect_type") == "Display":
                                        continue
                                    params_dict = effect_config.get("params") or {}
                                    for k in list(params_dict.keys()):
                                        if k in deck_values:
                                            params_dict[k] = deck_values[k]
                                    effect_config["params"] = params_dict
                        except Exception:
                            pass
                        # Bind the full-resolution master path after source
                        # acquisition.  This is the only ingress resize point:
                        # Deck B is normalized once when it differs from the
                        # primary master.  Preview scale is applied only after
                        # the worker returns a master program frame.
                        master_ingress = prepare_master_ingress(
                            current_image,
                            secondary_image,
                            fallback_size=(self.display_width, self.display_height),
                        )
                        current_image = master_ingress.primary
                        secondary_image = master_ingress.secondary
                        # Enqueue frame with ordered pipeline and extra parameters for animated effects.
                        # Video-backed playback uses the transport position so export/record starts from the
                        # actual media timeline instead of rewinding effect time back to zero.
                        if timeline_seconds is None:
                            effective_frame = self.recording_frame_count if is_recording else self.frame_counter
                            timeline_frame = effective_frame if timeline_frame is None else timeline_frame
                            timeline_seconds = float(effective_frame) / 60.0
                        elif timeline_frame is None:
                            timeline_frame = int(round(float(timeline_seconds) * max(1.0, float(self.framerate))))
                        
                        # Visible logging for recording (every 10 frames)
                        if is_recording and int(timeline_frame) % 10 == 0:
                            logging.info(f"[RECORDING] Queueing frame {int(timeline_frame)}, time_val={float(timeline_seconds):.3f}s")
                        
                        frame_data = {
                            "image": current_image,
                            "secondary_image": secondary_image,
                            # Provide node-ordered pipeline so worker can apply effects in sequence
                            "pipeline": processing_pipeline,
                            "pipeline_b": pipeline_b,
                            "master_mix": float(self.master_mix),
                            "master_transition": str(self.master_transition),
                            # Non-destructive param injection merged per-node in worker (e.g., global time)
                            "extra_params": {
                                "noise_warp_time": float(self.time_accum),
                                "time_accum": float(self.time_accum),
                                "frame_num": int(timeline_frame),
                                **active_time_domain_macros,
                            },
                            # Worker-side GPU usage
                            "use_cuda": bool(self.use_cuda),
                            "fallback_width": int(master_ingress.resolution.width),
                            "fallback_height": int(master_ingress.resolution.height),
                            "render_mode": RenderMode.LIVE.value,
                            "frame_role": FrameRole.MASTER_INGRESS.value,
                            "master_resolution": master_ingress.resolution.to_dict(),
                            "frame_index": int(timeline_frame),
                            "random_seed": int(getattr(self, "render_seed", 0) or 0),
                            "frame_copy_stats": {
                                "ingress_resize": {
                                    "count": int(master_ingress.ingress_resize_count),
                                    "bytes": int(master_ingress.ingress_resize_bytes),
                                }
                            },
                            # Latency tracking: timestamp at capture
                            "timestamp": time.perf_counter()
                        }
                        source_copy_policy = os.environ.get(
                            "MEGAGLITCH_SOURCE_COPY_POLICY", "copy"
                        ).strip().lower()
                        if source_copy_policy not in {"copy", "borrow_unique"}:
                            source_copy_policy = "copy"
                        frame_data["source_copy_policy"] = source_copy_policy
                        genesis_debug_request = self._get_genesis_aux_debug_request()
                        if genesis_debug_request is not None:
                            frame_data["genesis_debug_request"] = genesis_debug_request
                        
                        queued_frame_data = frame_data
                        if self.shared_frame_transport is not None:
                            queued_frame_data = self.shared_frame_transport.encode_payload(
                                frame_data, block=is_recording,
                                timeout=1.0 if is_recording else None,
                            )
                        if queued_frame_data is None:
                            self._mark_adaptive_queue_pressure()
                            if is_recording:
                                logging.warning("[RECORDING] Shared frame pool timeout - frame dropped")
                        # Legacy live capture waits up to one second; a timeout is
                        # reported as a drop.  Preview uses latest-frame semantics.
                        elif is_recording:
                            try:
                                self.raw_frame_queue.put(queued_frame_data, timeout=1.0)
                            except queue.Full:
                                if self.shared_frame_transport is not None:
                                    self.shared_frame_transport.release_input_payload(queued_frame_data)
                                logging.warning("[RECORDING] Queue timeout - frame may be dropped")
                        else:
                            try:
                                self.raw_frame_queue.put_nowait(queued_frame_data)
                            except queue.Full:
                                self._mark_adaptive_queue_pressure()
                                try:
                                    stale_input = self.raw_frame_queue.get_nowait()
                                    if self.shared_frame_transport is not None:
                                        self.shared_frame_transport.release_input_payload(stale_input)
                                except queue.Empty:
                                    pass
                                try:
                                    self.raw_frame_queue.put_nowait(queued_frame_data)
                                except queue.Full:
                                    if self.shared_frame_transport is not None:
                                        self.shared_frame_transport.release_input_payload(queued_frame_data)
                                    self._mark_adaptive_queue_pressure()

                    try:
                        # Always display the most recent glitched frame
                        glitched_result = None
                        drained_result_count = 0
                        while True:
                            try:
                                candidate_result = self.glitched_frame_queue.get_nowait()
                                if self.shared_frame_transport is not None:
                                    candidate_result = self.shared_frame_transport.decode_result(candidate_result)
                                glitched_result = candidate_result
                                drained_result_count += 1
                            except queue.Empty:
                                break
                        if drained_result_count > 1:
                            self._mark_adaptive_queue_pressure()
                        
                        if glitched_result is not None:
                            self._adaptive_last_result_time = time.perf_counter()
                            try:
                                self.processed_fps_samples += 1
                                fps_now = time.perf_counter()
                                fps_elapsed = fps_now - float(self.processed_fps_last_time or fps_now)
                                if fps_elapsed >= 2.0:
                                    processed_fps = self.processed_fps_samples / max(fps_elapsed, 1e-6)
                                    logging.info("PROCESSED FPS: %.1f unique frames/sec", processed_fps)
                                    self.processed_fps_samples = 0
                                    self.processed_fps_last_time = fps_now
                            except Exception:
                                pass
                            # Handle RESET message
                            if glitched_result == "RESET":
                                continue
                            
                            # Extract frame and timestamp
                            if isinstance(glitched_result, dict):
                                glitched_frame = glitched_result.get("frame")
                                deck_a_frame = glitched_result.get("frame_a")
                                deck_b_frame = glitched_result.get("frame_b")
                                capture_timestamp = glitched_result.get("timestamp", 0)
                                result_resolution_data = glitched_result.get("master_resolution")
                                if result_resolution_data is not None:
                                    result_resolution = MasterResolution.from_mapping(
                                        result_resolution_data
                                    )
                                elif glitched_frame is not None:
                                    result_array = np.asarray(glitched_frame)
                                    result_resolution = MasterResolution(
                                        width=int(result_array.shape[1]),
                                        height=int(result_array.shape[0]),
                                    )
                                else:
                                    result_resolution = None
                                result_role = FrameRole(
                                    str(
                                        glitched_result.get(
                                            "frame_role",
                                            FrameRole.MASTER_PROGRAM.value,
                                        )
                                    )
                                )
                            else:
                                # Backward compatibility: if worker returns bare frame
                                glitched_frame = glitched_result
                                deck_a_frame = None
                                deck_b_frame = None
                                capture_timestamp = 0
                                result_array = np.asarray(glitched_frame)
                                result_resolution = MasterResolution(
                                    width=int(result_array.shape[1]),
                                    height=int(result_array.shape[0]),
                                )
                                result_role = FrameRole.MASTER_PROGRAM
                            if glitched_frame is not None and result_resolution is not None:
                                glitched_frame = validate_master_frame(
                                    glitched_frame,
                                    result_resolution,
                                    role=result_role,
                                    name="live master program frame",
                                )
                                self.last_master_render_metrics = {
                                    key: glitched_result.get(key)
                                    for key in (
                                        "render_mode",
                                        "master_resolution",
                                        "history_buffer_resolution",
                                        "measured_temporal_inventory",
                                        "history_depth",
                                        "multimodal_telemetry",
                                        "controlled_feedback_telemetry",
                                        "render_time_ms",
                                        "random_seed",
                                        "frame_index",
                                        "temporal_resolution_resets",
                                        "frame_copy_stats",
                                    )
                                    if isinstance(glitched_result, dict) and key in glitched_result
                                }
                            self._update_genesis_aux_preview_from_result(glitched_result)

                            # The worker may have blended this result before the user moved
                            # the crossfader. Enforce hard endpoint cuts from the current UI
                            # value so stale queued frames cannot leak the opposite deck.
                            live_master_mix = float(np.clip(self.master_mix, 0.0, 1.0))
                            endpoint_epsilon = 1e-4
                            if live_master_mix <= endpoint_epsilon:
                                if deck_a_frame is not None:
                                    glitched_frame = deck_a_frame
                                elif deck_b_frame is not None:
                                    glitched_frame = np.zeros_like(deck_b_frame)
                            elif live_master_mix >= 1.0 - endpoint_epsilon:
                                if deck_b_frame is not None:
                                    glitched_frame = deck_b_frame
                                elif deck_a_frame is not None:
                                    glitched_frame = np.zeros_like(deck_a_frame)
                            
                            # Compute end-to-end latency
                            if capture_timestamp > 0:
                                latency_ms = (time.perf_counter() - capture_timestamp) * 1000
                                self.latency_samples.append(latency_ms)
                                if not (self.is_recording_gif or self.is_recording_mp4):
                                    self._update_adaptive_processing_scale(latency_ms)
                                
                                # Report every N frames
                                if len(self.latency_samples) >= self.latency_report_interval:
                                    avg_latency = np.mean(self.latency_samples)
                                    max_latency = np.max(self.latency_samples)
                                    min_latency = np.min(self.latency_samples)
                                    logging.info(f"LATENCY: avg={avg_latency:.1f}ms, min={min_latency:.1f}ms, max={max_latency:.1f}ms (n={len(self.latency_samples)})")
                                    self.latency_samples.clear()
                            if glitched_frame is not None:
                                # Apply display bit reduction if enabled (from Display menu)
                                display_frame = glitched_frame
                                if self.retro_mode_enabled:
                                    # Convert to numpy if needed
                                    if isinstance(display_frame, Image.Image):
                                        display_array = np.array(display_frame.convert('RGB'))
                                    else:
                                        display_array = display_frame
                                    
                                    # Apply bit reduction
                                    display_array = apply_retro_mode(
                                        display_array, 
                                        self.retro_bits_per_channel, 
                                        self.retro_dither_enabled
                                    )
                                    display_frame = display_array
                                self.update_visual_metrics_from_frame(display_frame)

                                # Push the latest frame into the DearPyGui texture for on-screen preview.
                                # During realtime recording, throttle preview uploads; recording still
                                # captures every processed frame below.
                                recording_active = self.is_recording_gif or self.is_recording_mp4
                                update_recording_preview = (
                                    not recording_active or
                                    self.recording_frame_count == 0 or
                                    self.recording_frame_count % 4 == 0
                                )
                                if update_recording_preview:
                                    preview_scale = self._effective_processing_scale(
                                        is_recording=recording_active
                                    )
                                    preview_branch = make_egress_frame(
                                        display_frame,
                                        result_resolution,
                                        role=FrameRole.PREVIEW,
                                        scale=preview_scale,
                                    ).data
                                    deck_previews = (
                                        # Deck A is the master/program monitor.
                                        (self.node_editor_a, preview_branch),
                                        # Both displays monitor the same master output.
                                        (self.node_editor_b, preview_branch),
                                    )
                                    for editor, deck_frame in deck_previews:
                                        display_node = editor.get_display_node()
                                        if display_node is None:
                                            continue
                                        try:
                                            preview_frame = (
                                                deck_frame
                                                if deck_frame is not None
                                                else display_frame
                                            )
                                            update_display_texture(
                                                preview_frame,
                                                display_node.texture_tag,
                                                display_node.display_width,
                                                display_node.display_height,
                                            )
                                        except Exception as e:
                                            logging.error(f"Display texture update failed: {e}", exc_info=True)

                                # Record frame if recording
                                if self.is_recording_gif or self.is_recording_mp4:
                                    try:
                                        # Capture frame and increment counter
                                        self.recording_frames.append(glitched_frame.copy())
                                        self.recording_frame_count += 1
                                        current_time = time.time()
                                        self.recording_elapsed = current_time - self.recording_start_time
                                        
                                        # Log first frame and every 10th frame - also write to file
                                        debug_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording_debug.log")
                                        if self.recording_frame_count == 1:
                                            try:
                                                with open(debug_log_path, "a") as f:
                                                    f.write(f"First frame captured! Total: {len(self.recording_frames)}\n")
                                            except: pass
                                            print(f"[RECORDING] First frame captured! Total: {len(self.recording_frames)}")
                                        elif self.recording_frame_count % 10 == 0:
                                            try:
                                                with open(debug_log_path, "a") as f:
                                                    f.write(f"Frame {self.recording_frame_count}, total={len(self.recording_frames)}\n")
                                            except: pass
                                            print(f"[RECORDING] Captured frame {self.recording_frame_count}, total={len(self.recording_frames)}")
                                        
                                        # Update recording status display
                                        update_recording_status = (
                                            self.recording_frame_count == 1 or
                                            self.recording_frame_count % 5 == 0 or
                                            self.recording_frame_count >= self.recording_target_frames
                                        )
                                        if update_recording_status and dpg.does_item_exist("recording_progress_bar"):
                                            progress = min(1.0, self.recording_frame_count / max(1, self.recording_target_frames))
                                            dpg.set_value("recording_progress_bar", progress)
                                            elapsed_str = f"{self.recording_elapsed:.1f}s"
                                            dpg.set_value("recording_time_text", 
                                                        f"{self.recording_frame_count} / {self.recording_target_frames} frames ({elapsed_str})")
                                        
                                        # Check if we've reached target frame count
                                        if self.recording_frame_count >= self.recording_target_frames:
                                            # Auto-stop recording
                                            if self.is_recording_gif:
                                                print(f"[RECORDING] GIF complete: {self.recording_frame_count} frames in {self.recording_elapsed:.1f}s")
                                                logging.info(f"GIF recording complete: {self.recording_frame_count} frames in {self.recording_elapsed:.1f}s")
                                                self.stop_and_save_gif(None, None)
                                            elif self.is_recording_mp4:
                                                print(f"[RECORDING] MP4 complete: {self.recording_frame_count} frames in {self.recording_elapsed:.1f}s")
                                                logging.info(f"MP4 recording complete: {self.recording_frame_count} frames in {self.recording_elapsed:.1f}s")
                                                self.stop_and_save_mp4(None, None)
                                    except Exception as e:
                                        logging.error(f"Recording frame error: {e}")
                            # Send to virtual camera if enabled
                            if self.virtual_cam_running and self.virtual_cam is not None:
                                try:
                                    self._enqueue_virtual_cam_frame(glitched_frame)
                                except Exception as e:
                                    frame_shape = None
                                    try:
                                        frame_shape = np.asarray(glitched_frame).shape
                                    except Exception:
                                        pass
                                    logging.error(f"Virtual cam send error: {e} (frame_shape={frame_shape})")
                        else:
                            self._update_adaptive_processing_without_result()
                    except queue.Empty:
                        pass
                self._sync_video_transport_ui(update_input=False)
                self.update_live_sequencer_ui()
                self._update_media_audio_ui()
                dpg.render_dearpygui_frame()
        except Exception as e:
            logging.error(f"Error in main loop: {e}", exc_info=True)
        finally:
            # Reject new frame work before releasing any producer or consumer.
            self.processing_paused = True
            # Stop capture sources before draining and stopping the processing worker.
            self._release_window_capture_resources()
            self.window_capture_running = False
            self._release_webcam_capture()
            self.webcam_running = False
            self._close_media_reader(self.video_reader)
            self.video_reader = None
            self._close_media_reader(self.mixer_channel_b_video_reader)
            self.mixer_channel_b_video_reader = None
            if self.bg_video_reader:
                try:
                    self.bg_video_reader.release()
                except Exception:
                    pass
                self.bg_video_reader = None
            # Reclaim queued input descriptors before asking the idle worker to exit.
            try:
                while True:
                    stale_input = self.raw_frame_queue.get_nowait()
                    if self.shared_frame_transport is not None:
                        self.shared_frame_transport.release_input_payload(stale_input)
            except queue.Empty:
                pass
            if self.processing_lifecycle is not None:
                forced = self.processing_lifecycle.stop()
                if forced:
                    logging.warning("Processing worker required forced termination")
            # The worker is now stopped, so all remaining output descriptors are stable
            # and can be reclaimed before shared regions are unlinked.
            try:
                while True:
                    stale_result = self.glitched_frame_queue.get_nowait()
                    if self.shared_frame_transport is not None:
                        self.shared_frame_transport.release_output_result(stale_result)
            except queue.Empty:
                pass
            self._stop_virtual_cam_sender()
            self._close_virtual_camera()
            self.virtual_cam_running = False
            if self.shared_frame_transport is not None:
                try:
                    self.shared_frame_transport.close()
                except Exception:
                    logging.debug("Shared frame transport shutdown failed", exc_info=True)
                finally:
                    self.shared_frame_transport = None
            try:
                self.stop_live_sequencer()
            except Exception:
                logging.debug("Live sequencer shutdown encountered an issue", exc_info=True)
            try:
                if self.audio_midi:
                    self.audio_midi.shutdown()
            except Exception:
                logging.debug("Audio shutdown encountered an issue", exc_info=True)
            finally:
                self.audio_midi = None
            try:
                if self.media_audio:
                    self.media_audio.destroy()
            except Exception:
                logging.debug("Media Audio shutdown encountered an issue", exc_info=True)
            self.node_editor_a.destroy_audio_nodes()
            self.node_editor_b.destroy_audio_nodes()
            self.bpm_detector = None
            for processing_queue in (self.raw_frame_queue, self.glitched_frame_queue):
                try:
                    processing_queue.close()
                    processing_queue.cancel_join_thread()
                except Exception:
                    pass
            dpg.destroy_context()
            lifecycle_snapshot = (
                None if self.processing_lifecycle is None
                else self.processing_lifecycle.snapshot()
            )
            logging.info("Shutdown complete: processing=%s", lifecycle_snapshot)

    def update_visual_metrics_from_frame(self, frame):
        try:
            if self.visual_analyzer_worker:
                self.visual_analyzer_worker.submit_frame(frame)
        except Exception:
            logging.debug("Could not submit frame to visual analyzer", exc_info=True)

    def get_latest_visual_metrics(self):
        try:
            return self.visual_bus.get_latest()
        except Exception:
            return {}

    def start_live_sequencer(self):
        try:
            if self.visual_analyzer_worker:
                self.visual_analyzer_worker.start()
            if self.live_sequencer:
                self.live_sequencer.start()
        except Exception:
            logging.warning("Could not start live sequencer", exc_info=True)

    def stop_live_sequencer(self):
        try:
            if self.live_sequencer:
                self.live_sequencer.stop()
        finally:
            try:
                if self.visual_analyzer_worker:
                    self.visual_analyzer_worker.stop()
            finally:
                try:
                    if self.midi_device_manager:
                        self.midi_device_manager.close_all()
                except Exception:
                    logging.debug("MIDI output close failed", exc_info=True)

    def panic_all_midi(self):
        try:
            if self.live_sequencer:
                self.live_sequencer.panic_all()
        except Exception:
            logging.debug("MIDI panic failed", exc_info=True)

    def build_live_sequencer_ui(self):
        if self.live_sequencer_ui is not None:
            return self.live_sequencer_ui
        if not hasattr(self, "live_sequencer_tab"):
            return None
        self.live_sequencer_ui = build_live_sequencer_ui(
            self.live_sequencer_tab,
            self.live_sequencer,
            self.visual_bus,
            self.midi_device_manager,
            self.mpd26_controls,
        )
        return self.live_sequencer_ui

    def update_live_sequencer_ui(self):
        try:
            if self.hardware_mapping_manager:
                self.hardware_mapping_manager.update()
                if dpg.does_item_exist(self.midi_mapping_status_tag):
                    dpg.set_value(self.midi_mapping_status_tag, self.hardware_mapping_manager.status_message)
            if self.live_sequencer_ui:
                self.live_sequencer_ui.update()
        except Exception:
            logging.debug("Live sequencer UI update failed", exc_info=True)

    def _on_midi_message(self, msg):
        try:
            if self.media_audio and self.media_audio.handle_midi(msg, self.get_latest_visual_metrics()):
                return
        except Exception:
            logging.debug("Media Audio BeatStep MIDI handling failed", exc_info=True)
        consumed = False
        if self.hardware_mapping_manager:
            consumed = self.hardware_mapping_manager.handle_midi(msg)
        if not consumed and self.mpd26_controls:
            self.mpd26_controls.handle_midi(msg)

    def toggle_midi_mapping_mode(self, sender=None, app_data=None, user_data=None):
        if not self.hardware_mapping_manager:
            return
        self.hardware_mapping_manager.set_enabled(bool(app_data))
        if dpg.does_item_exist(self.midi_mapping_status_tag):
            dpg.set_value(self.midi_mapping_status_tag, self.hardware_mapping_manager.status_message)

    def clear_midi_mappings(self, sender=None, app_data=None, user_data=None):
        if not self.hardware_mapping_manager:
            return
        self.hardware_mapping_manager.clear()
        if dpg.does_item_exist(self.midi_mapping_status_tag):
            dpg.set_value(self.midi_mapping_status_tag, self.hardware_mapping_manager.status_message)

    def open_live_sequencer_tab(self, sender=None, app_data=None, user_data=None):
        try:
            if hasattr(self, "deck_tab_bar") and hasattr(self, "live_sequencer_tab"):
                dpg.set_value(self.deck_tab_bar, self.live_sequencer_tab)
        except Exception:
            logging.debug("Could not switch to Live Sequencer tab", exc_info=True)

    def open_modulation_matrix(self, sender, app_data):
        try:
            providers = self._build_modulation_providers()
            active_editor = self.get_active_editor()
            if not self.mod_matrix:
                self.mod_matrix = NodeModulationMatrix(
                    active_editor,
                    providers=providers,
                    extra_targets_provider=self._get_global_time_domain_param_defs,
                )
            else:
                self.mod_matrix.node_editor = active_editor
                self.mod_matrix.set_providers(providers)
            self.mod_matrix._collect_targets_and_ranges()
            self.mod_matrix.show_window()
        except Exception as e:
            logging.error(f"Failed to open modulation matrix: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            # Try to show an error popup to the user
            try:
                if dpg.does_item_exist("modulation_error_window"):
                    dpg.delete_item("modulation_error_window")
                with dpg.window(label="Modulation Matrix Error", modal=True, show=True, 
                               tag="modulation_error_window", pos=[400, 300], width=400, height=150):
                    dpg.add_text(f"Could not open modulation matrix:")
                    dpg.add_text(f"{str(e)}", wrap=380)
                    dpg.add_button(label="OK", callback=lambda: dpg.delete_item("modulation_error_window"))
            except Exception:
                pass

    def open_parameter_timeline(self, sender=None, app_data=None):
        try:
            if self.parameter_timeline is None:
                self.parameter_timeline = ParameterTimeline(
                    self.get_active_editor,
                    lambda: float(self.time_accum),
                )
            self.parameter_timeline.show_window()
        except Exception as e:
            logging.error(f"Failed to open parameter timeline: {e}", exc_info=True)

    def _get_media_audio_preset_state(self):
        if not self.media_audio:
            return None
        try:
            state = self.media_audio.export_preset_state()
            state["ui_preset"] = (
                dpg.get_value("media_audio_preset")
                if dpg.does_item_exist("media_audio_preset")
                else ""
            )
            return state
        except Exception:
            logging.warning("Could not collect Media Audio preset state", exc_info=True)
            return None

    def _restore_media_audio_preset_state(self, state):
        if not self.media_audio or not isinstance(state, dict):
            return
        try:
            self.media_audio.import_preset_state(state, reload_media=True, reload_ir=True)
            ui_preset = str(state.get("ui_preset") or "")
            if ui_preset and dpg.does_item_exist("media_audio_preset"):
                dpg.set_value("media_audio_preset", ui_preset)
            self._update_media_audio_ui(force=True)
            self._update_media_audio_cdp_graph_ui(force=True)
        except Exception:
            logging.warning("Could not restore Media Audio preset state", exc_info=True)

    def show_save_preset_dialog(self, sender, app_data):
        """Show dialog to save current node configuration as a preset."""
        if dpg.does_item_exist("save_preset_window"):
            dpg.delete_item("save_preset_window")
        
        with dpg.window(label="Save Node Preset", modal=True, show=True, 
                       tag="save_preset_window", pos=[400, 300], width=400, height=200):
            dpg.add_text("Session preset: Deck A + Deck B")
            dpg.add_text("Enter preset name:")
            dpg.add_input_text(tag="preset_name_input", default_value="", width=-1, on_enter=True, callback=self.save_node_preset_callback)
            dpg.add_spacer(height=15)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self.save_node_preset_callback, width=180)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("save_preset_window"), width=180)
    
    def save_node_preset_callback(self, sender, app_data):
        """Callback to save the preset with the entered name."""
        import json

        preset_name = dpg.get_value("preset_name_input").strip()
        
        if not preset_name:
            logging.warning("Please enter a preset name")
            return
        
        try:
            preset_data = {
                "version": "2.0",
                "preset_type": "dual_deck_session",
                "decks": {
                    "a": self.node_editor_a.get_node_preset_data(),
                    "b": self.node_editor_b.get_node_preset_data(),
                },
                "master_mix": float(self.master_mix),
                "master_transition": str(self.master_transition),
                "display_macros": self._get_global_time_domain_macro_values(),
                "offline_render_settings": self._get_offline_render_settings(),
                "media_audio": self._get_media_audio_preset_state(),
                "mod_matrix_state": {
                    "enabled": bool(self.mod_matrix.enabled),
                    "connections": list(self.mod_matrix.connections),
                } if self.mod_matrix else None,
                "parameter_timeline_state": (
                    self.parameter_timeline.to_dict() if self.parameter_timeline else None
                ),
            }
            filepath = Path("node_presets") / f"{preset_name}.json"
            temporary_path = filepath.with_suffix(".json.tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(preset_data, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            with temporary_path.open("r", encoding="utf-8") as handle:
                verified = json.load(handle)
            verified_decks = verified.get("decks", {})
            if (
                verified.get("preset_type") != "dual_deck_session"
                or not isinstance(verified_decks.get("a"), dict)
                or not isinstance(verified_decks.get("b"), dict)
            ):
                raise ValueError("dual-deck session verification failed after save")
            temporary_path.replace(filepath)
            logging.info(f"Preset '{preset_name}' saved successfully")
            dpg.delete_item("save_preset_window")
        except Exception as e:
            try:
                temporary_path = Path("node_presets") / f"{preset_name}.json.tmp"
                if temporary_path.exists():
                    temporary_path.unlink()
            except Exception:
                pass
            logging.error(f"Failed to save preset '{preset_name}': {e}", exc_info=True)
    
    def load_quality_first_graph(
        self,
        sender=None,
        app_data=None,
        user_data=None,
        *,
        editor=None,
        announce=True,
    ):
        """Install the master-resolution controlled-destruction graph."""
        from megaglitch.quality_graph import build_quality_first_graph

        target_editor = editor or self.get_active_editor() or self.node_editor_a
        loaded = target_editor.load_node_preset(
            "Quality-First Master",
            build_quality_first_graph(),
        )
        if not loaded:
            logging.error("Could not load the Quality-First Master graph")
            return False
        target_editor.mark_pipeline_dirty()
        if target_editor is self.node_editor_a:
            self.master_mix = 0.0
            if dpg.does_item_exist("master_mix_slider"):
                dpg.set_value("master_mix_slider", 0.0)
        try:
            self.raw_frame_queue.put_nowait("RESET")
        except Exception:
            pass
        if announce:
            logging.info(
                "Loaded Quality-First Master graph: analysis -> controlled "
                "corruption -> structural restoration"
            )
        return True

    def show_load_preset_dialog(self, sender, app_data):
        """Show dialog to load a saved preset."""
        self._preset_target_editor = self.get_active_editor()
        target_name = "Deck B" if self._preset_target_editor is self.node_editor_b else "Deck A"
        if dpg.does_item_exist("load_preset_window"):
            dpg.delete_item("load_preset_window")
        
        presets = self._preset_target_editor.get_available_presets()
        
        if not presets:
            with dpg.window(label="Load Node Preset", modal=True, show=True, 
                           tag="load_preset_window", pos=[400, 300], width=400, height=100):
                dpg.add_text("No presets found. Save a preset first!")
                dpg.add_spacer(height=15)
                dpg.add_button(label="OK", callback=lambda: dpg.delete_item("load_preset_window"), width=-1)
            return
        
        with dpg.window(label="Load Node Preset", modal=True, show=True, 
                       tag="load_preset_window", pos=[400, 300], width=400, height=300):
            dpg.add_text(f"Target: {target_name}")
            dpg.add_text("Select a preset, then press Load:")
            dpg.add_listbox(
                items=presets,
                default_value=presets[0],
                tag="preset_listbox",
                width=-1,
                num_items=10,
            )
            dpg.add_text("", tag="preset_load_status", color=(255, 120, 120))
            dpg.add_spacer(height=12)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load", callback=self.load_node_preset_callback, width=125)
                dpg.add_button(label="Delete", callback=self.delete_node_preset_callback, width=125)
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("load_preset_window"), width=125)
    
    def load_node_preset_callback(self, sender, app_data):
        """Callback to load the selected preset."""
        preset_name = dpg.get_value("preset_listbox")
        
        if not preset_name:
            logging.warning("Please select a preset to load")
            return
        
        import json
        filepath = f"node_presets/{preset_name}.json"
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                preset_data = json.load(handle)
        except Exception as e:
            logging.error(f"Failed to read preset '{preset_name}': {e}")
            if dpg.does_item_exist("preset_load_status"):
                dpg.set_value("preset_load_status", f"Could not read '{preset_name}': {e}")
            return

        decks = preset_data.get("decks") if isinstance(preset_data, dict) else None
        target_editor = getattr(self, "_preset_target_editor", None) or self.get_active_editor()
        if isinstance(decks, dict):
            loaded = (
                self.node_editor_a.load_node_preset(preset_name, decks.get("a", {}))
                and self.node_editor_b.load_node_preset(preset_name, decks.get("b", {}))
            )
            self.master_mix = float(np.clip(preset_data.get("master_mix", self.master_mix), 0.0, 1.0))
            self.master_transition = str(preset_data.get("master_transition", self.master_transition))
            if dpg.does_item_exist("master_mix_slider"):
                dpg.set_value("master_mix_slider", self.master_mix)
            if dpg.does_item_exist("master_transition_combo"):
                dpg.set_value("master_transition_combo", self.master_transition)
        else:
            loaded = target_editor.load_node_preset(preset_name, preset_data)

        if loaded:
            self._ensure_global_time_macros_window(recreate=False)
            self._load_global_time_domain_macros_from_preset(preset_name)
            
            # Load modulation matrix state
            try:
                mod_state = preset_data.get('mod_matrix_state')
                if mod_state and self.mod_matrix:
                    self.mod_matrix.enabled = mod_state.get('enabled', True)
                    self.mod_matrix.connections = list(mod_state.get('connections', []))
                    if dpg.does_item_exist(self.mod_matrix.conn_list_tag):
                        dpg.configure_item(self.mod_matrix.conn_list_tag, items=self.mod_matrix.connections)
                    self.mod_matrix._sync_checks_from_connections()

                timeline_state = preset_data.get('parameter_timeline_state')
                if timeline_state:
                    if self.parameter_timeline is None:
                        self.parameter_timeline = ParameterTimeline(
                            self.get_active_editor,
                            lambda: float(self.time_accum),
                        )
                    self.parameter_timeline.from_dict(timeline_state)
                self._restore_media_audio_preset_state(preset_data.get("media_audio"))
                self._restore_offline_render_settings(
                    preset_data.get("offline_render_settings")
                )
            except Exception as e:
                logging.warning(f"Failed to load modulation/timeline state for preset '{preset_name}': {e}")
                
            destination = "Deck A + Deck B" if isinstance(decks, dict) else (
                "Deck B" if target_editor is self.node_editor_b else "Deck A"
            )
            logging.info(f"Preset '{preset_name}' loaded successfully into {destination}")
            dpg.delete_item("load_preset_window")
            
            # Refresh modulation matrix targets with new node parameters
            if self.mod_matrix:
                try:
                    self.mod_matrix.refresh_targets_ui()
                    logging.info("Modulation targets refreshed")
                except Exception as e:
                    logging.warning(f"Could not refresh modulation targets: {e}")

            try:
                while self.raw_frame_queue.qsize() > 0:
                    self.raw_frame_queue.get_nowait()
            except Exception:
                pass
            try:
                self.raw_frame_queue.put("RESET")
            except Exception:
                pass
            
            # Update pipeline navigator if visible
            if self.pipeline_navigator_visible:
                self.update_pipeline_navigator()
        else:
            logging.error("Failed to load preset")
            if dpg.does_item_exist("preset_load_status"):
                dpg.set_value("preset_load_status", f"Failed to load '{preset_name}'.")
    
    def delete_node_preset_callback(self, sender, app_data):
        """Callback to delete the selected preset."""
        import os
        preset_name = dpg.get_value("preset_listbox")
        
        if not preset_name:
            logging.warning("Please select a preset to delete")
            return
        
        try:
            os.remove(f"node_presets/{preset_name}.json")
            logging.info(f"Preset '{preset_name}' deleted")
            # Refresh the dialog
            dpg.delete_item("load_preset_window")
            self.show_load_preset_dialog(None, None)
        except Exception as e:
            logging.error(f"Failed to delete preset: {e}")

    def show_navigation_help(self, sender, app_data):
        """Show node editor navigation controls help."""
        if dpg.does_item_exist("nav_help_window"):
            dpg.delete_item("nav_help_window")
        
        with dpg.window(label="Node Editor Navigation", modal=True, show=True,
                       tag="nav_help_window", pos=[400, 250], width=450, height=300):
            dpg.add_text("Node Editor Controls:", color=(100, 200, 255))
            dpg.add_spacer(height=8)
            
            dpg.add_text("PANNING:")
            dpg.add_text("  • Middle Mouse Button + Drag - Pan the canvas")
            dpg.add_text("  • Right Click + Drag - Alternative pan method")
            dpg.add_spacer(height=8)
            
            dpg.add_text("ZOOMING:")
            dpg.add_text("  • Mouse Wheel - Zoom in/out")
            dpg.add_spacer(height=8)
            
            dpg.add_text("MINIMAP:")
            dpg.add_text("  • Bottom-right corner shows full canvas overview")
            dpg.add_text("  • Click minimap to jump to that area")
            dpg.add_text("  • Drag rectangle in minimap to pan")
            dpg.add_spacer(height=8)
            
            dpg.add_text("NODE OPERATIONS:")
            dpg.add_text("  • Left Click + Drag Node - Move node")
            dpg.add_text("  • Click Output → Input - Create connection")
            dpg.add_text("  • Right Click Link - Delete connection")
            dpg.add_spacer(height=12)
            
            dpg.add_button(label="Got it!", callback=lambda: dpg.delete_item("nav_help_window"), width=-1)
    
    def toggle_pipeline_navigator(self, sender, app_data):
        """Toggle the pipeline navigator window visibility."""
        self.pipeline_navigator_visible = not self.pipeline_navigator_visible
        if self.pipeline_navigator_visible:
            self.show_pipeline_navigator()
        else:
            if dpg.does_item_exist("pipeline_navigator_window"):
                dpg.hide_item("pipeline_navigator_window")
    
    def show_pipeline_navigator(self):
        """Show or update the pipeline navigator window."""
        if dpg.does_item_exist("pipeline_navigator_window"):
            dpg.show_item("pipeline_navigator_window")
            self.update_pipeline_navigator()
        else:
            self.create_pipeline_navigator()
    
    def create_pipeline_navigator(self):
        """Create the pipeline navigator window."""
        with dpg.window(label="Pipeline Navigator", tag="pipeline_navigator_window",
                       pos=[10, 100], width=300, height=500, show=True,
                       on_close=lambda: setattr(self, 'pipeline_navigator_visible', False)):
            dpg.add_text("Current Pipeline:", tag="pipeline_nav_title")
            dpg.add_separator()
            with dpg.child_window(tag="pipeline_nav_list", height=-30):
                dpg.add_text("No pipeline configured", tag="pipeline_empty_text")
            dpg.add_separator()
            dpg.add_button(label="Refresh", callback=lambda: self.update_pipeline_navigator(), width=-1)
        
        self.update_pipeline_navigator()
    
    def update_pipeline_navigator(self):
        """Update the pipeline navigator with current nodes."""
        if not dpg.does_item_exist("pipeline_nav_list"):
            return
        
        # Clear existing items
        dpg.delete_item("pipeline_nav_list", children_only=True)
        
        # Get current pipeline
        try:
            pipeline = self.get_active_editor().get_processing_pipeline()
            
            if not pipeline:
                dpg.add_text("No nodes in pipeline", parent="pipeline_nav_list", color=(150, 150, 150))
                return
            
            # Display each node in the pipeline
            for i, node_config in enumerate(pipeline):
                effect_type = node_config.get("effect_type", "Unknown")
                
                with dpg.group(parent="pipeline_nav_list", horizontal=True):
                    # Arrow indicator
                    if i > 0:
                        dpg.add_text("  ↓", color=(100, 200, 255))
                    
                with dpg.group(parent="pipeline_nav_list"):
                    # Node box
                    node_text = f"{i+1}. {effect_type}"
                    dpg.add_text(node_text, color=(255, 255, 255))
                    
                    # Show some key parameters
                    params = node_config.get("params", {})
                    if params and effect_type != "Display":
                        param_count = len([k for k, v in params.items() if v != 0])
                        if param_count > 0:
                            dpg.add_text(f"   ({param_count} active params)", color=(150, 150, 150))
                    
                    dpg.add_spacer(height=8)
            
            # Update title with count
            if dpg.does_item_exist("pipeline_nav_title"):
                node_count = len([n for n in pipeline if n.get("effect_type") != "Display"])
                dpg.set_value("pipeline_nav_title", f"Current Pipeline ({node_count} nodes):")
                
        except Exception as e:
            dpg.add_text(f"Error: {str(e)}", parent="pipeline_nav_list", color=(255, 100, 100))
            logging.error(f"Pipeline navigator update error: {e}")

    # ---------- Recording / Export ----------
    def _get_offline_render_settings(self):
        return {
            "version": 1,
            "master": {
                "source_fps_auto": bool(self.offline_source_fps_auto),
                "source_fps": float(self.offline_source_fps),
                "source_range": str(self.offline_source_range),
                "output_fps": float(self.offline_output_fps),
                "fps_semantics": str(self.offline_fps_semantics),
                "codec": str(self.offline_master_codec),
                "crf": int(self.offline_master_crf),
                "preset": str(self.offline_master_preset),
                "random_seed": int(self.offline_random_seed),
                "ffmpeg_executable": str(self.offline_ffmpeg_executable),
                "ffprobe_executable": str(self.offline_ffprobe_executable),
                "analyze_source_audio": bool(self.offline_analyze_source_audio),
                "audio_sample_rate": int(self.offline_audio_sample_rate),
            },
            "codec_post": {
                "enabled": bool(self.codec_post_enabled),
                "command": str(self.codec_post_command),
                "work_directory": str(self.codec_post_work_directory),
                "seed": int(self.codec_post_seed),
                "long_gop": int(self.codec_post_long_gop),
                "preparation_quality": int(self.codec_post_preparation_quality),
                "final_codec": str(self.codec_post_final_codec),
                "final_crf": int(self.codec_post_final_crf),
                "final_preset": str(self.codec_post_final_preset),
            },
        }

    def _restore_offline_render_settings(self, settings):
        if not isinstance(settings, dict):
            return False
        try:
            validated = self._validate_offline_render_settings(settings)
        except (TypeError, ValueError):
            logging.warning("Ignoring invalid offline mastering settings", exc_info=True)
            return False
        self._commit_offline_render_settings(validated)
        self._sync_offline_master_settings_ui()
        return True

    @staticmethod
    def _offline_setting_bool(value, name):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"{name} must be a boolean")

    def _validate_offline_render_settings(self, settings):
        """Return one complete, validated settings snapshot without mutating state."""

        if not isinstance(settings, dict):
            raise TypeError("offline render settings must be a dictionary")
        current = self._get_offline_render_settings()
        supplied_master = settings.get("master", {})
        supplied_codec = settings.get("codec_post", {})
        if not isinstance(supplied_master, dict):
            raise TypeError("master settings must be a dictionary")
        if not isinstance(supplied_codec, dict):
            raise TypeError("codec_post settings must be a dictionary")
        master = {**current["master"], **supplied_master}
        codec_post = {**current["codec_post"], **supplied_codec}

        source_fps = float(master["source_fps"])
        output_fps = float(master["output_fps"])
        if not np.isfinite(source_fps) or source_fps <= 0.0:
            raise ValueError("Source FPS must be finite and greater than zero")
        if not np.isfinite(output_fps) or output_fps <= 0.0:
            raise ValueError("Output FPS must be finite and greater than zero")
        source_range = str(master["source_range"]).strip()
        if source_range not in {"entire_source", "from_playhead"}:
            raise ValueError("unsupported source range")
        fps_semantics = str(master["fps_semantics"]).strip()
        if fps_semantics not in {"duration_preserving", "true_accelerated"}:
            raise ValueError("unsupported FPS semantics")
        master_codec = str(master["codec"]).strip().lower()
        if master_codec not in {"h264", "h265"}:
            raise ValueError("Master Codec must be h264 or h265")
        master_crf = int(master["crf"])
        if not 0 <= master_crf <= 51:
            raise ValueError("Master CRF must be between 0 and 51")
        master_preset = str(master["preset"]).strip().lower()
        if master_preset not in _OFFLINE_ENCODER_PRESETS:
            raise ValueError("unsupported Master Preset")
        ffmpeg_executable = str(master["ffmpeg_executable"]).strip()
        ffprobe_executable = str(master["ffprobe_executable"]).strip()
        if not ffmpeg_executable:
            raise ValueError("FFmpeg executable cannot be empty")
        if not ffprobe_executable:
            raise ValueError("FFprobe executable cannot be empty")
        audio_sample_rate = int(master["audio_sample_rate"])
        if not 8000 <= audio_sample_rate <= 192000:
            raise ValueError("Audio Analysis Sample Rate must be 8000..192000")

        codec_enabled = self._offline_setting_bool(
            codec_post["enabled"], "Enable external CLI codec pass"
        )
        codec_command = str(codec_post["command"]).strip()
        work_directory = str(codec_post["work_directory"]).strip()
        if not work_directory:
            raise ValueError("Codec Work Directory cannot be empty")
        long_gop = int(codec_post["long_gop"])
        if long_gop < 2:
            raise ValueError("Long GOP must be at least 2")
        preparation_quality = int(codec_post["preparation_quality"])
        if not 1 <= preparation_quality <= 31:
            raise ValueError("MPEG-4 preparation q:v must be between 1 and 31")
        final_codec = str(codec_post["final_codec"]).strip().lower()
        if final_codec not in {"h264", "h265"}:
            raise ValueError("Final Codec must be h264 or h265")
        final_crf = int(codec_post["final_crf"])
        if not 0 <= final_crf <= 51:
            raise ValueError("Final CRF must be between 0 and 51")
        final_preset = str(codec_post["final_preset"]).strip().lower()
        if final_preset not in _OFFLINE_ENCODER_PRESETS:
            raise ValueError("unsupported Final Preset")

        from megaglitch.codec_domain_post import (
            CodecDomainPostConfig,
            parse_corruption_command_template,
        )
        from megaglitch.offline_master import MasterCodec

        if codec_command:
            parsed_command = parse_corruption_command_template(codec_command)
            if any(
                placeholder in parsed_command[0]
                for placeholder in ("{prepared}", "{corrupted}", "{seed}")
            ):
                raise ValueError("External CLI template must start with an executable")
            CodecDomainPostConfig(
                deterministic_input="validation-input.mp4",
                final_output="validation-output.mp4",
                work_directory=work_directory,
                corruption_command=parsed_command,
                ffmpeg_executable=ffmpeg_executable,
                long_gop_size=long_gop,
                preparation_quality=preparation_quality,
                corruption_seed=int(codec_post["seed"]),
                final_codec=MasterCodec(final_codec),
                final_crf=final_crf,
                final_preset=final_preset,
            )
        elif codec_enabled:
            raise ValueError("codec corruption command cannot be empty")

        return {
            "version": 1,
            "master": {
                "source_fps_auto": self._offline_setting_bool(
                    master["source_fps_auto"], "Use media source FPS"
                ),
                "source_fps": source_fps,
                "source_range": source_range,
                "output_fps": output_fps,
                "fps_semantics": fps_semantics,
                "codec": master_codec,
                "crf": master_crf,
                "preset": master_preset,
                "random_seed": int(master["random_seed"]),
                "ffmpeg_executable": ffmpeg_executable,
                "ffprobe_executable": ffprobe_executable,
                "analyze_source_audio": self._offline_setting_bool(
                    master["analyze_source_audio"], "Analyze source audio"
                ),
                "audio_sample_rate": audio_sample_rate,
            },
            "codec_post": {
                "enabled": codec_enabled,
                "command": codec_command,
                "work_directory": work_directory,
                "seed": int(codec_post["seed"]),
                "long_gop": long_gop,
                "preparation_quality": preparation_quality,
                "final_codec": final_codec,
                "final_crf": final_crf,
                "final_preset": final_preset,
            },
        }

    def _commit_offline_render_settings(self, settings):
        master = settings["master"]
        codec_post = settings["codec_post"]
        self.offline_source_fps_auto = master["source_fps_auto"]
        self.offline_source_fps = master["source_fps"]
        self.offline_source_range = master["source_range"]
        self.offline_output_fps = master["output_fps"]
        self.offline_fps_semantics = master["fps_semantics"]
        self.offline_master_codec = master["codec"]
        self.offline_master_crf = master["crf"]
        self.offline_master_preset = master["preset"]
        self.offline_random_seed = master["random_seed"]
        self.offline_ffmpeg_executable = master["ffmpeg_executable"]
        self.offline_ffprobe_executable = master["ffprobe_executable"]
        self.offline_analyze_source_audio = master["analyze_source_audio"]
        self.offline_audio_sample_rate = master["audio_sample_rate"]
        self.codec_post_enabled = codec_post["enabled"]
        self.codec_post_command = codec_post["command"]
        self.codec_post_work_directory = codec_post["work_directory"]
        self.codec_post_seed = codec_post["seed"]
        self.codec_post_long_gop = codec_post["long_gop"]
        self.codec_post_preparation_quality = codec_post["preparation_quality"]
        self.codec_post_final_codec = codec_post["final_codec"]
        self.codec_post_final_crf = codec_post["final_crf"]
        self.codec_post_final_preset = codec_post["final_preset"]

    def _offline_render_settings_from_ui(self):
        return {
            "version": 1,
            "master": {
                "source_fps_auto": dpg.get_value("offline_source_fps_auto"),
                "source_fps": dpg.get_value("offline_source_fps"),
                "source_range": dpg.get_value("offline_source_range"),
                "output_fps": dpg.get_value("offline_output_fps"),
                "fps_semantics": dpg.get_value("offline_fps_semantics"),
                "codec": dpg.get_value("offline_master_codec"),
                "crf": dpg.get_value("offline_master_crf"),
                "preset": dpg.get_value("offline_master_preset"),
                "random_seed": dpg.get_value("offline_random_seed"),
                "ffmpeg_executable": dpg.get_value("offline_ffmpeg_executable"),
                "ffprobe_executable": dpg.get_value("offline_ffprobe_executable"),
                "analyze_source_audio": dpg.get_value("offline_analyze_source_audio"),
                "audio_sample_rate": dpg.get_value("offline_audio_sample_rate"),
            },
            "codec_post": {
                "enabled": dpg.get_value("codec_post_enabled"),
                "command": dpg.get_value("codec_post_command"),
                "work_directory": dpg.get_value("codec_post_work_directory"),
                "seed": dpg.get_value("codec_post_seed"),
                "long_gop": dpg.get_value("codec_post_long_gop"),
                "preparation_quality": dpg.get_value(
                    "codec_post_preparation_quality"
                ),
                "final_codec": dpg.get_value("codec_post_final_codec"),
                "final_crf": dpg.get_value("codec_post_final_crf"),
                "final_preset": dpg.get_value("codec_post_final_preset"),
            },
        }

    def _update_offline_master_control_states(
        self, sender=None, app_data=None, user_data=None
    ):
        try:
            if dpg.does_item_exist("offline_source_fps"):
                dpg.configure_item(
                    "offline_source_fps",
                    enabled=not bool(dpg.get_value("offline_source_fps_auto")),
                )
            codec_enabled = bool(dpg.get_value("codec_post_enabled"))
            for tag in _CODEC_POST_EDITABLE_TAGS:
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, enabled=codec_enabled)
        except Exception:
            logging.debug("Could not update Offline Master control states", exc_info=True)

    def _sync_offline_master_settings_ui(self):
        if not bool(getattr(self, "_offline_master_settings_open", False)):
            return
        try:
            if not dpg.does_item_exist("offline_master_settings_window"):
                return
        except Exception:
            return
        settings = self._get_offline_render_settings()
        values = {
            "offline_source_fps_auto": settings["master"]["source_fps_auto"],
            "offline_source_fps": settings["master"]["source_fps"],
            "offline_source_range": settings["master"]["source_range"],
            "offline_output_fps": settings["master"]["output_fps"],
            "offline_fps_semantics": settings["master"]["fps_semantics"],
            "offline_master_codec": settings["master"]["codec"],
            "offline_master_crf": settings["master"]["crf"],
            "offline_master_preset": settings["master"]["preset"],
            "offline_random_seed": settings["master"]["random_seed"],
            "offline_ffmpeg_executable": settings["master"]["ffmpeg_executable"],
            "offline_ffprobe_executable": settings["master"]["ffprobe_executable"],
            "offline_analyze_source_audio": settings["master"][
                "analyze_source_audio"
            ],
            "offline_audio_sample_rate": settings["master"]["audio_sample_rate"],
            "codec_post_enabled": settings["codec_post"]["enabled"],
            "codec_post_command": settings["codec_post"]["command"],
            "codec_post_work_directory": settings["codec_post"]["work_directory"],
            "codec_post_seed": settings["codec_post"]["seed"],
            "codec_post_long_gop": settings["codec_post"]["long_gop"],
            "codec_post_preparation_quality": settings["codec_post"][
                "preparation_quality"
            ],
            "codec_post_final_codec": settings["codec_post"]["final_codec"],
            "codec_post_final_crf": settings["codec_post"]["final_crf"],
            "codec_post_final_preset": settings["codec_post"]["final_preset"],
        }
        for tag, value in values.items():
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)
        self._update_offline_master_control_states()

    def _build_codec_domain_post_config(self, deterministic_input, final_output):
        from megaglitch.codec_domain_post import (
            CodecDomainPostConfig,
            parse_corruption_command_template,
        )
        from megaglitch.offline_master import MasterCodec

        settings = self._validate_offline_render_settings(
            self._get_offline_render_settings()
        )
        master = settings["master"]
        codec_post = settings["codec_post"]
        return CodecDomainPostConfig(
            deterministic_input=deterministic_input,
            final_output=final_output,
            work_directory=codec_post["work_directory"],
            corruption_command=parse_corruption_command_template(
                codec_post["command"]
            ),
            ffmpeg_executable=master["ffmpeg_executable"],
            long_gop_size=codec_post["long_gop"],
            preparation_quality=codec_post["preparation_quality"],
            corruption_seed=codec_post["seed"],
            final_codec=MasterCodec(codec_post["final_codec"]),
            final_crf=codec_post["final_crf"],
            final_preset=codec_post["final_preset"],
        )

    def _build_offline_master_encoder_config(
        self, output_path, width, height, fps
    ):
        from megaglitch.offline_master import FFmpegMasterConfig, MasterCodec

        master = self._validate_offline_render_settings(
            self._get_offline_render_settings()
        )["master"]
        return FFmpegMasterConfig(
            output_path=output_path,
            width=int(width),
            height=int(height),
            fps=float(fps),
            codec=MasterCodec(master["codec"]),
            crf=master["crf"],
            preset=master["preset"],
            ffmpeg_executable=master["ffmpeg_executable"],
            ffprobe_executable=master["ffprobe_executable"],
        )

    def _apply_offline_master_settings(self, sender=None, app_data=None, user_data=None):
        try:
            validated = self._validate_offline_render_settings(
                self._offline_render_settings_from_ui()
            )
            self._commit_offline_render_settings(validated)
            self._sync_offline_master_settings_ui()
            if dpg.does_item_exist("offline_master_settings_status"):
                dpg.set_value(
                    "offline_master_settings_status",
                    "Validated and applied. Direct image, temporal, and codec domains remain isolated.",
                )
            return True
        except Exception as exc:
            logging.error("Invalid Offline Master settings: %s", exc)
            if dpg.does_item_exist("offline_master_settings_status"):
                dpg.set_value(
                    "offline_master_settings_status", f"Invalid settings: {exc}"
                )
            return False

    def _close_offline_master_settings(self, sender=None, app_data=None, user_data=None):
        self._offline_master_settings_open = False
        try:
            if dpg.does_item_exist("offline_master_settings_window"):
                dpg.delete_item("offline_master_settings_window")
        except Exception:
            pass

    def show_offline_master_settings(self, sender=None, app_data=None, user_data=None):
        tag = "offline_master_settings_window"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        self._offline_master_settings_open = True
        with dpg.window(
            label="Offline Master + Codec Domain",
            modal=True,
            show=True,
            tag=tag,
            pos=[260, 90],
            width=760,
            height=780,
            no_scrollbar=True,
            on_close=self._close_offline_master_settings,
        ):
            with dpg.child_window(
                tag="offline_master_settings_scroll",
                height=-88,
                border=False,
            ):
                dpg.add_text("DIRECT IMAGE DOMAIN", color=(100, 230, 210))
                dpg.add_text(
                    "The connected image-effect graph runs at master resolution; preview, OBS, and virtual-camera scaling are isolated.",
                    wrap=700,
                )
                dpg.add_text("TEMPORAL NETWORK DOMAIN", color=(120, 190, 255))
                dpg.add_text(
                    "Feedback, frame history, and recursive state stay at master resolution and are reset for this deterministic render.",
                    wrap=700,
                )

                dpg.add_separator()
                dpg.add_text("SOURCE ENUMERATION + TIMEBASE", color=(190, 210, 225))
                dpg.add_checkbox(
                    label="Use media source FPS when available",
                    default_value=self.offline_source_fps_auto,
                    tag="offline_source_fps_auto",
                    callback=self._update_offline_master_control_states,
                )
                dpg.add_input_float(
                    label="Manual Source FPS",
                    default_value=float(self.offline_source_fps),
                    min_value=0.001,
                    min_clamped=True,
                    format="%.3f",
                    tag="offline_source_fps",
                )
                dpg.add_combo(
                    label="Media Source Range",
                    items=["entire_source", "from_playhead"],
                    default_value=self.offline_source_range,
                    tag="offline_source_range",
                    width=260,
                )
                dpg.add_text(
                    "Video/GIF media is enumerated once to EOF; duration applies only to stills and generators.",
                    color=(150, 170, 185),
                    wrap=700,
                )
                dpg.add_input_float(
                    label="Output FPS",
                    default_value=float(self.offline_output_fps),
                    min_value=0.001,
                    min_clamped=True,
                    format="%.3f",
                    tag="offline_output_fps",
                )
                dpg.add_combo(
                    label="FPS Semantics",
                    items=["true_accelerated", "duration_preserving"],
                    default_value=self.offline_fps_semantics,
                    tag="offline_fps_semantics",
                    width=260,
                )
                dpg.add_text(
                    "true_accelerated keeps every source frame once; 30 frames at 55 fps last 30/55 s.",
                    color=(255, 190, 90),
                    wrap=700,
                )

                dpg.add_separator()
                dpg.add_text("DETERMINISTIC MASTER ENCODE", color=(160, 220, 170))
                dpg.add_text(
                    "This is the final encode when the codec-domain pass is off, or its clean deterministic input when the pass is on.",
                    wrap=700,
                )
                dpg.add_combo(
                    label="Deterministic Master Codec",
                    items=["h264", "h265"],
                    default_value=self.offline_master_codec,
                    tag="offline_master_codec",
                    width=160,
                )
                dpg.add_input_int(
                    label="Master CRF (14-18 target)",
                    default_value=int(self.offline_master_crf),
                    min_value=0,
                    max_value=51,
                    min_clamped=True,
                    max_clamped=True,
                    tag="offline_master_crf",
                )
                dpg.add_combo(
                    label="Master Encoder Preset",
                    items=list(_OFFLINE_ENCODER_PRESETS),
                    default_value=self.offline_master_preset,
                    tag="offline_master_preset",
                    width=160,
                )
                dpg.add_input_int(
                    label="Deterministic Seed",
                    default_value=int(self.offline_random_seed),
                    tag="offline_random_seed",
                )
                dpg.add_input_text(
                    label="FFmpeg executable (master + codec pass)",
                    default_value=self.offline_ffmpeg_executable,
                    tag="offline_ffmpeg_executable",
                    width=440,
                )
                dpg.add_input_text(
                    label="FFprobe executable (master verification)",
                    default_value=self.offline_ffprobe_executable,
                    tag="offline_ffprobe_executable",
                    width=440,
                )
                dpg.add_checkbox(
                    label="Analyze source audio deterministically for multimodal control",
                    default_value=self.offline_analyze_source_audio,
                    tag="offline_analyze_source_audio",
                )
                dpg.add_input_int(
                    label="Audio Analysis Sample Rate",
                    default_value=int(self.offline_audio_sample_rate),
                    min_value=8000,
                    max_value=192000,
                    min_clamped=True,
                    max_clamped=True,
                    tag="offline_audio_sample_rate",
                )

                dpg.add_separator()
                dpg.add_text(
                    "CODEC DOMAIN — OPTIONAL EXTERNAL CLI PASS",
                    color=(255, 180, 80),
                )
                dpg.add_text(
                    "Domain order: direct image graph → temporal network → MPEG-4 long-GOP preparation → external packet/P-frame CLI → tolerant decode → final encode.",
                    wrap=700,
                )
                dpg.add_checkbox(
                    label="Enable external CLI codec pass",
                    default_value=self.codec_post_enabled,
                    tag="codec_post_enabled",
                    callback=self._update_offline_master_control_states,
                )
                dpg.add_input_text(
                    label="External CLI Command Template",
                    default_value=self.codec_post_command,
                    multiline=True,
                    height=72,
                    width=600,
                    tag="codec_post_command",
                )
                dpg.add_text(
                    "Required placeholders: {prepared}, {corrupted}. Optional deterministic value: {seed}. Quoted executable paths are supported.",
                    color=(190, 190, 190),
                    wrap=700,
                )
                dpg.add_input_text(
                    label="Codec Work Directory",
                    default_value=self.codec_post_work_directory,
                    width=500,
                    tag="codec_post_work_directory",
                )
                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        label="CLI Seed",
                        default_value=int(self.codec_post_seed),
                        tag="codec_post_seed",
                        width=120,
                    )
                    dpg.add_input_int(
                        label="Long GOP Frames",
                        default_value=int(self.codec_post_long_gop),
                        min_value=2,
                        min_clamped=True,
                        tag="codec_post_long_gop",
                        width=120,
                    )
                    dpg.add_input_int(
                        label="Preparation MPEG-4 q:v",
                        default_value=int(self.codec_post_preparation_quality),
                        min_value=1,
                        max_value=31,
                        min_clamped=True,
                        max_clamped=True,
                        tag="codec_post_preparation_quality",
                        width=120,
                    )
                with dpg.group(horizontal=True):
                    dpg.add_combo(
                        label="Final Codec",
                        items=["h264", "h265"],
                        default_value=self.codec_post_final_codec,
                        tag="codec_post_final_codec",
                        width=110,
                    )
                    dpg.add_input_int(
                        label="Final CRF",
                        default_value=int(self.codec_post_final_crf),
                        min_value=0,
                        max_value=51,
                        min_clamped=True,
                        max_clamped=True,
                        tag="codec_post_final_crf",
                        width=100,
                    )
                    dpg.add_combo(
                        label="Final Encoder Preset",
                        items=list(_OFFLINE_ENCODER_PRESETS),
                        default_value=self.codec_post_final_preset,
                        tag="codec_post_final_preset",
                        width=130,
                    )

            dpg.add_separator()
            dpg.add_text(
                "",
                tag="offline_master_settings_status",
                wrap=700,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Validate && Apply",
                    callback=self._apply_offline_master_settings,
                    width=180,
                )
                dpg.add_button(
                    label="Close Without Applying",
                    callback=self._close_offline_master_settings,
                    width=180,
                )
        self._update_offline_master_control_states()

    def start_gif_recording(self, sender, app_data):
        # Write to file for debugging (visible even without console)
        # Use absolute path next to this script file
        import os
        debug_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording_debug.log")
        try:
            with open(debug_log_path, "a") as f:
                f.write(f"\n========== GIF RECORDING STARTED at {time.strftime('%H:%M:%S')} ==========\n")
                f.write(f"Log file location: {debug_log_path}\n")
        except Exception as e:
            print(f"[DEBUG LOG ERROR] Could not write to {debug_log_path}: {e}")
        print("[RECORDING] ========== GIF RECORDING STARTED ==========")
        logging.info("[RECORDING] ========== GIF RECORDING STARTED ==========")
        self.is_recording_gif = True
        self.recording_frames = []
        self.recording_frame_times = []
        self.last_recorded_frame = None
        self.recording_start_time = time.time()
        self.recording_elapsed = 0.0
        self.recording_frame_count = 0
        # Calculate total frames needed
        target_fps = max(1, int(self.framerate))
        self.recording_target_frames = int(self.export_duration * target_fps)
        # Show recording status window
        if dpg.does_item_exist("recording_status_window"):
            dpg.configure_item("recording_status_window", show=True)
            dpg.set_value("recording_status_text", "🔴 RECORDING GIF")
            dpg.set_value("recording_time_text", f"0 / {self.recording_target_frames} frames")
        logging.info(f"GIF recording started. Target: {self.recording_target_frames} frames ({self.export_duration:.1f}s at {target_fps} fps)")

    def stop_and_save_gif(self, sender, app_data):
        if not self.is_recording_gif and not self.recording_frames:
            return
        self.is_recording_gif = False
        # Hide recording status window
        if dpg.does_item_exist("recording_status_window"):
            dpg.configure_item("recording_status_window", show=False)
        # Open save dialog
        if dpg.does_item_exist("save_gif_dialog"): dpg.delete_item("save_gif_dialog")
        with dpg.file_dialog(directory_selector=False, show=True, callback=self._save_gif_callback, tag="save_gif_dialog", modal=True):
            dpg.add_file_extension(".gif")

    def _save_gif_callback(self, sender, app_data):
        try:
            path = app_data['file_path_name']
            if not path.lower().endswith('.gif'):
                path += '.gif'
            
            num_frames = len(self.recording_frames)
            
            # Write to debug log file (same location as script)
            debug_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording_debug.log")
            try:
                with open(debug_log_path, "a") as f:
                    f.write(f"SAVE GIF: {num_frames} frames to save\n")
                    if num_frames <= 1:
                        f.write(f"ERROR: Only {num_frames} frame(s)! Expected {self.recording_target_frames}\n")
            except: pass
            
            print(f"[SAVE GIF] Starting save with {num_frames} frames")
            logging.info(f"[SAVE GIF] Starting save with {num_frames} frames")
            
            # Check for single frame issue
            if num_frames <= 1:
                print(f"[SAVE GIF] ERROR: Only {num_frames} frame(s) captured! Expected {self.recording_target_frames}")
                logging.error(f"[SAVE GIF] ERROR: Only {num_frames} frame(s) captured! Expected {self.recording_target_frames}")
                logging.error("[SAVE GIF] This will create a static image, not an animation")
            
            # Use TARGET framerate for smooth playback
            # Don't use actual capture time as processing speed varies
            target_fps = max(1, int(self.framerate))
            duration_ms = int(1000.0 / target_fps)
            
            # Log actual vs target for debugging
            if self.recording_elapsed > 0 and num_frames > 0:
                actual_capture_fps = num_frames / self.recording_elapsed
                logging.info(f"Saving GIF: {num_frames} frames")
                logging.info(f"  Capture rate: {actual_capture_fps:.1f} fps (processing speed)")
                logging.info(f"  Export rate: {target_fps} fps (target playback speed)")
            else:
                logging.info(f"Saving GIF: {num_frames} frames at {target_fps} fps")
            
            frames_rgb = [self._prepare_frame_for_export(img) for img in self.recording_frames]
            
            # Use PIL to save GIF with proper framerate control
            from PIL import Image
            pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
            
            # Save with PIL using TARGET framerate for smooth playback
            pil_frames[0].save(
                path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,  # milliseconds per frame at target FPS
                loop=0,  # 0 = loop forever (infinite loop)
                optimize=False
            )
            logging.info(f"Saved GIF: {path} ({target_fps} fps, {num_frames} frames)")
        except Exception as e:
            logging.error(f"Error saving GIF: {e}")
        finally:
            try:
                dpg.delete_item("save_gif_dialog")
            except Exception:
                pass
            self.recording_frames = []
            self.recording_frame_times = []
            self.last_recorded_frame = None
            self.recording_frames = []
            self.recording_frame_times = []
            self.last_recorded_frame = None

    def start_mp4_recording(self, sender, app_data):
        self.is_recording_mp4 = True
        self.recording_frames = []
        self.recording_frame_times = []
        self.last_recorded_frame = None
        self.recording_start_time = time.time()
        self.recording_elapsed = 0.0
        self.recording_frame_count = 0
        # Calculate total frames needed
        target_fps = max(1, int(self.framerate))
        self.recording_target_frames = int(self.export_duration * target_fps)
        # Show recording status window
        if dpg.does_item_exist("recording_status_window"):
            dpg.configure_item("recording_status_window", show=True)
            dpg.set_value("recording_status_text", "🔴 RECORDING MP4")
            dpg.set_value("recording_time_text", f"0 / {self.recording_target_frames} frames")
        logging.info(f"MP4 recording started. Target: {self.recording_target_frames} frames ({self.export_duration:.1f}s at {target_fps} fps)")

    def stop_and_save_mp4(self, sender, app_data):
        if not self.is_recording_mp4 and not self.recording_frames:
            return
        self.is_recording_mp4 = False
        # Hide recording status window
        if dpg.does_item_exist("recording_status_window"):
            dpg.configure_item("recording_status_window", show=False)
        if dpg.does_item_exist("save_mp4_dialog"): dpg.delete_item("save_mp4_dialog")
        with dpg.file_dialog(directory_selector=False, show=True, callback=self._save_mp4_callback, tag="save_mp4_dialog", modal=True):
            dpg.add_file_extension(".mp4")

    def _save_mp4_callback(self, sender, app_data):
        try:
            path = app_data['file_path_name']
            if not path.lower().endswith('.mp4'):
                path += '.mp4'
            
            num_frames = len(self.recording_frames)
            
            # Use TARGET framerate for smooth playback
            # Don't use actual capture time as processing speed varies
            target_fps = max(1, int(self.framerate))
            
            # Log actual vs target for debugging
            if self.recording_elapsed > 0 and num_frames > 0:
                actual_capture_fps = num_frames / self.recording_elapsed
                logging.info(f"Saving MP4: {num_frames} frames")
                logging.info(f"  Capture rate: {actual_capture_fps:.1f} fps (processing speed)")
                logging.info(f"  Export rate: {target_fps} fps (target playback speed)")
            else:
                logging.info(f"Saving MP4: {num_frames} frames at {target_fps} fps")
            
            import cv2
            w, h = self.display_width, self.display_height
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(path, fourcc, target_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Could not open VideoWriter with mp4v")
            for img in self.recording_frames:
                frame_rgb = self._prepare_frame_for_export(img)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
            writer.release()
            logging.info(f"Saved MP4: {path} ({target_fps} fps, {num_frames} frames)")
        except Exception as e:
            logging.error(f"Error saving MP4: {e}")
        finally:
            try:
                dpg.delete_item("save_mp4_dialog")
            except Exception:
                pass
            self.recording_frames = []
            self.recording_frame_times = []
            self.last_recorded_frame = None
            self.recording_frames = []
            self.recording_frame_times = []
            self.last_recorded_frame = None

    def _prepare_frame_for_export(self, frame):
        # Returns RGB numpy array sized to display dims, with letterboxing like display
        import numpy as np

        if isinstance(frame, Image.Image):
            arr = np.asarray(frame.convert('RGBA'), dtype=np.uint8)
        elif isinstance(frame, np.ndarray):
            arr = frame
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim != 3 or arr.shape[2] not in (3, 4):
                raise ValueError(f"Unsupported frame shape for export: {arr.shape}")
            if arr.shape[2] == 3:
                alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
                arr = np.concatenate([arr, alpha], axis=2)
        else:
            raise ValueError(f"Frame must be numpy array or PIL Image, got {type(frame)}")

        rgb = arr[..., :3]
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        if np.any(alpha < 1.0):
            rgb = (rgb.astype(np.float32) * alpha).astype(np.uint8)

        disp_w, disp_h = int(self.display_width), int(self.display_height)
        h, w = rgb.shape[:2]
        aspect = w / float(h)
        target_aspect = disp_w / float(disp_h)
        out = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)

        if aspect > target_aspect:
            new_w = disp_w
            new_h = max(1, int(new_w / aspect))
        else:
            new_h = disp_h
            new_w = max(1, int(new_h * aspect))

        if new_w != w or new_h != h:
            interpolation = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_LINEAR
            resized = cv2.resize(rgb, (new_w, new_h), interpolation=interpolation)
        else:
            resized = rgb

        px = (disp_w - new_w) // 2
        py = (disp_h - new_h) // 2
        out[py:py + new_h, px:px + new_w] = resized
        return out

    def _pipeline_starts_with_generator(self, pipeline):
        return bool(pipeline) and bool(pipeline[0].get("is_generator", False))

    def _pipeline_uses_secondary_source(self, pipeline):
        return any(
            effect_config.get("node_type") == "source" and effect_config.get("source_key") == "secondary"
            for effect_config in (pipeline or [])
        )

    def _build_batch_chunk(self, start_idx, end_idx, target_fps, base_pipeline, timeline_start_seconds):
        """Build source frames, time params, and per-frame pipelines for one export chunk."""
        import copy

        source_frames = []
        secondary_source_frames = []
        extra_params_list = []
        modulated_pipelines = []
        video_input = self.video_reader and not self.image and not self.gif_frames
        has_generator = self._pipeline_starts_with_generator(base_pipeline)
        uses_secondary_source = self._pipeline_uses_secondary_source(base_pipeline)

        for i in range(start_idx, end_idx):
            if video_input:
                try:
                    raw = self.video_reader.get_next_data()
                except (IndexError, StopIteration):
                    self.video_reader.set_image_index(0)
                    raw = self.video_reader.get_next_data()
                frame = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_RGB2RGBA))
            else:
                if self.original_image_with_alpha:
                    frame = self._apply_background_to_image(self.original_image_with_alpha.copy())
                elif self.image:
                    frame = self.image.copy()
                elif self._background_media_active():
                    bg_frame = self._get_current_bg_frame()
                    if bg_frame is None:
                        raise RuntimeError("Background media is active but no frame was available")
                    if bg_frame.size != (self.display_width, self.display_height):
                        bg_frame = bg_frame.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
                    frame = bg_frame.convert('RGBA')
                elif self.gif_frames:
                    frame = self.gif_frames[self.gif_index % len(self.gif_frames)].copy()
                    self.gif_index = (self.gif_index + 1) % len(self.gif_frames)
                elif has_generator or uses_secondary_source:
                    # Match live preview: generator-first pipelines do not require source media.
                    frame = Image.new('RGB', (self.display_width, self.display_height), (0, 0, 0))
                else:
                    raise RuntimeError("No media loaded for batch rendering")

            secondary_frame = None
            if uses_secondary_source:
                secondary_frame = self._get_current_mixer_channel_b_frame(export_mode=True)

            source_frames.append(frame)
            secondary_source_frames.append(secondary_frame)

            global_macros = self._get_global_time_domain_macro_values()
            reversal = float(global_macros.get("time_reversal", 0.0))
            warp = float(global_macros.get("local_time_warp", 0.0))
            flow_rate = (1.0 - 2.0 * reversal) * (1.0 + warp)
            
            frame_time = timeline_start_seconds + (i / target_fps) * flow_rate
            extra_params = {
                "noise_warp_time": frame_time * self.time_scale,
                "time_accum": frame_time,
                "frame_num": int(round(frame_time * target_fps)),
            }

            frame_pipeline = copy.deepcopy(base_pipeline)
            if self.parameter_timeline:
                try:
                    self.parameter_timeline.apply_to_pipeline(frame_pipeline, frame_time)
                except Exception as e:
                    logging.warning(f"Could not apply parameter timeline for frame {i}: {e}")
            if self.srt_shader_enabled and self.srt_shader_cues:
                try:
                    apply_srt_shader_cues(
                        frame_pipeline,
                        self.srt_shader_cues,
                        frame_time,
                        Path(__file__).resolve().parent,
                        deck="a",
                    )
                except Exception as e:
                    logging.warning(f"Could not apply SRT shader cues for frame {i}: {e}")

            if self.mod_matrix:
                try:
                    explicit_time = frame_time
                    merged_params = {}
                    for effect_config in frame_pipeline:
                        if effect_config.get("effect_type") == "Display":
                            continue
                        effect_params = effect_config.get("params", {}) or {}
                        merged_params.update(effect_params)
                    merged_params.update(self._get_global_time_domain_macro_values())

                    modulated_params = self.mod_matrix.apply(
                        merged_params,
                        explicit_time=explicit_time,
                        time_scale=self.time_scale,
                    )
                    extra_params.update(self._get_active_time_domain_macros(modulated_params))

                    for effect_config in frame_pipeline:
                        if effect_config.get("effect_type") == "Display":
                            continue
                        params_dict = effect_config.get("params") or {}
                        for key in list(params_dict.keys()):
                            if key in modulated_params:
                                params_dict[key] = modulated_params[key]
                        effect_config["params"] = params_dict
                except Exception as e:
                    logging.warning(f"Could not apply modulation for frame {i}: {e}")
                    extra_params.update(self._get_global_time_domain_macro_values())
            else:
                extra_params.update(self._get_global_time_domain_macro_values())

            extra_params_list.append(extra_params)

            modulated_pipelines.append(frame_pipeline)

        return source_frames, secondary_source_frames, modulated_pipelines, extra_params_list

    def _process_batch_chunk(self, source_frames, secondary_source_frames, modulated_pipelines, extra_params_list,
                             has_temporal, sequential_state=None, progress_bar=None):
        from parallel_processor import _batch_process_parallel_modulated, _batch_process_sequential_modulated

        if has_temporal:
            processed_frames, sequential_state = _batch_process_sequential_modulated(
                source_frames,
                secondary_source_frames,
                modulated_pipelines,
                extra_params_list,
                bool(self.use_cuda),
                state=sequential_state,
                return_state=True,
                progress_bar=progress_bar,
            )
            return processed_frames, sequential_state

        processed_frames = _batch_process_parallel_modulated(
            source_frames,
            secondary_source_frames,
            modulated_pipelines,
            extra_params_list,
            bool(self.use_cuda),
            progress_bar=progress_bar,
        )
        return processed_frames, sequential_state

    def _effective_offline_source_fps(self):
        if self.offline_source_fps_auto:
            if self.video_reader is not None:
                return max(0.001, float(self._reader_get_fps(self.video_reader)))
            if self.gif_frames and self.gif_durations:
                mean_duration_ms = float(np.mean(self.gif_durations))
                if mean_duration_ms > 0.0:
                    return 1000.0 / mean_duration_ms
        return max(0.001, float(self.offline_source_fps))

    def _build_offline_source_plan(self, playback_state, base_pipeline_a, base_pipeline_b):
        """Freeze the exact primary source range; media never loops at EOF."""
        from megaglitch.offline_source import (
            OfflineSourceKind,
            OfflineSourceRange,
            plan_gif_source,
            plan_timed_source,
            plan_video_source,
        )

        source_range = OfflineSourceRange(self.offline_source_range)
        has_generator = self._pipeline_starts_with_generator(
            base_pipeline_a
        ) or self._pipeline_starts_with_generator(base_pipeline_b)

        if self.video_reader is not None and not self.image and not self.gif_frames:
            frame_count = self._reader_get_frame_count(self.video_reader)
            plan = plan_video_source(
                frame_count=frame_count,
                fps=self._reader_get_fps(self.video_reader),
                playhead_index=int(playback_state.get("primary_video_index", 0)),
                source_range=source_range,
            )
            return plan, "primary_video"

        if self.gif_frames:
            durations = list(self.gif_durations or ())
            if len(durations) != len(self.gif_frames):
                fallback_ms = 1000.0 / self._effective_offline_source_fps()
                durations = [fallback_ms] * len(self.gif_frames)
            plan = plan_gif_source(
                durations,
                playhead_index=int(playback_state.get("primary_gif_index", 0)),
                source_range=source_range,
            )
            return plan, "primary_gif"

        if self.original_image_with_alpha or self.image:
            return (
                plan_timed_source(
                    duration_seconds=self.export_duration,
                    fps=self._effective_offline_source_fps(),
                    kind=OfflineSourceKind.STILL,
                ),
                "primary_still",
            )

        if self._background_media_active() and self.bg_video_reader is not None:
            frame_count = self._reader_get_frame_count(self.bg_video_reader)
            plan = plan_video_source(
                frame_count=frame_count,
                fps=self._reader_get_fps(self.bg_video_reader),
                playhead_index=int(playback_state.get("bg_video_index", 0)),
                source_range=source_range,
            )
            return plan, "background_video"

        if self._background_media_active() and self.bg_gif_frames:
            plan = plan_gif_source(
                [100.0] * len(self.bg_gif_frames),
                playhead_index=int(playback_state.get("bg_gif_index", 0)),
                source_range=source_range,
            )
            return plan, "background_gif"

        if self._background_media_active() and self.bg_media:
            return (
                plan_timed_source(
                    duration_seconds=self.export_duration,
                    fps=self._effective_offline_source_fps(),
                    kind=OfflineSourceKind.STILL,
                ),
                "background_still",
            )

        if has_generator or self._pipeline_uses_secondary_source(base_pipeline_a):
            return (
                plan_timed_source(
                    duration_seconds=self.export_duration,
                    fps=self._effective_offline_source_fps(),
                    kind=OfflineSourceKind.GENERATOR,
                ),
                "generator",
            )
        raise RuntimeError("No media or generator is available for Offline Master")

    def _read_offline_reader_frame(self, reader, frame_index):
        """Read one exact RGB frame by index from either reader API."""
        if not self._reader_set_frame_index(reader, frame_index):
            raise RuntimeError(f"could not seek offline media to frame {frame_index}")
        if hasattr(reader, "get_next_data"):
            try:
                return self._to_virtual_cam_rgb(reader.get_next_data())
            except (IndexError, StopIteration) as exc:
                raise RuntimeError(
                    f"offline media ended before declared frame {frame_index}"
                ) from exc
        result = reader.read()
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("offline media reader returned an unsupported payload")
        ok, frame_bgr = result
        if not ok or frame_bgr is None:
            raise RuntimeError(
                f"offline media ended before declared frame {frame_index}"
            )
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _get_offline_secondary_frame_at_time(self, timestamp_seconds):
        """Timestamp-sample Deck B media without primary-rate stepping or loops."""
        from megaglitch.offline_source import (
            sample_cfr_index_at_time,
            sample_gif_index_at_time,
        )

        if self.mixer_channel_b_image is not None:
            return self.mixer_channel_b_image.copy()
        if self.mixer_channel_b_gif_frames:
            durations = list(self.mixer_channel_b_gif_durations or ())
            if len(durations) != len(self.mixer_channel_b_gif_frames):
                durations = [100.0] * len(self.mixer_channel_b_gif_frames)
            index = sample_gif_index_at_time(timestamp_seconds, durations)
            return self.mixer_channel_b_gif_frames[index].copy()
        if self.mixer_channel_b_video_reader is not None:
            count = self._reader_get_frame_count(self.mixer_channel_b_video_reader)
            index = sample_cfr_index_at_time(
                timestamp_seconds,
                fps=self._reader_get_fps(self.mixer_channel_b_video_reader),
                frame_count=count,
            )
            return self._read_offline_reader_frame(
                self.mixer_channel_b_video_reader, index
            )
        return None

    def _capture_offline_control_snapshot(self):
        """Copy every low-frequency control consumed by the offline graph."""
        import copy

        from megaglitch.offline_controls import FrozenModulation

        global_macros = copy.deepcopy(self._get_global_time_domain_macro_values())
        timeline = {
            "enabled": bool(
                self.parameter_timeline
                and getattr(self.parameter_timeline, "enabled", False)
            ),
            "tracks": copy.deepcopy(
                getattr(self.parameter_timeline, "tracks", {}) or {}
            ),
        }
        source_values = {
            "fft_bands": copy.deepcopy(dict(self.latest_audio_bands or {})),
            "onset_value": float(self.latest_onset_value or 0.0),
            "onset_event_id": int(self.latest_onset_event_id or 0),
            "beat_event_id": 0,
            "bpm_pulse": 0.0,
            "bpm_phase": 0.0,
            "bpm_normalized": 0.0,
            "midi_note": 0.0,
            "midi_cc": 0.0,
        }
        modulator = None
        if self.mod_matrix is not None:
            for key in tuple(source_values):
                provider = getattr(self.mod_matrix, "providers", {}).get(key)
                if callable(provider):
                    try:
                        source_values[key] = copy.deepcopy(provider())
                    except Exception:
                        pass
            target_editor = getattr(self.mod_matrix, "node_editor", None)
            target_deck = (
                "b" if target_editor is self.node_editor_b else
                "a" if target_editor is self.node_editor_a else None
            )
            modulator = FrozenModulation(
                enabled=bool(getattr(self.mod_matrix, "enabled", False)),
                connections=tuple(getattr(self.mod_matrix, "connections", ()) or ()),
                parameter_ranges=copy.deepcopy(
                    getattr(self.mod_matrix, "param_ranges", {}) or {}
                ),
                parameter_types=copy.deepcopy(
                    getattr(self.mod_matrix, "param_types", {}) or {}
                ),
                parameter_items={
                    key: tuple(values)
                    for key, values in copy.deepcopy(
                        getattr(self.mod_matrix, "param_items", {}) or {}
                    ).items()
                },
                parameter_modes=copy.deepcopy(
                    getattr(self.mod_matrix, "param_mod_modes", {}) or {}
                ),
                parameter_spans=copy.deepcopy(
                    getattr(self.mod_matrix, "param_mod_spans", {}) or {}
                ),
                source_values=source_values,
                target_deck=target_deck,
            )
        return {
            "global_macros": global_macros,
            "timeline": timeline,
            "srt_enabled": bool(self.srt_shader_enabled),
            "srt_cues": copy.deepcopy(self.srt_shader_cues or []),
            "time_scale": float(self.time_scale),
            "audio": source_values,
            "modulator": modulator,
        }

    def _offline_control_document(self, snapshot):
        modulator = snapshot.get("modulator")
        return {
            "global_macros": snapshot.get("global_macros", {}),
            "timeline": snapshot.get("timeline", {}),
            "srt_enabled": bool(snapshot.get("srt_enabled", False)),
            "srt_cues": snapshot.get("srt_cues", []),
            "time_scale": float(snapshot.get("time_scale", 1.0)),
            "audio": snapshot.get("audio", {}),
            "source_audio_analysis": snapshot.get("source_audio_document"),
            "modulation": modulator.document() if modulator is not None else None,
        }

    def _prepare_offline_pipeline(
        self,
        base_pipeline,
        frame_time,
        deck_name,
        control_snapshot,
        python_rng,
        frame_audio=None,
    ):
        """Evaluate a frozen control document for one deterministic frame."""
        import copy

        from megaglitch.offline_controls import apply_frozen_timeline

        pipeline = copy.deepcopy(base_pipeline or [])
        if deck_name == "b":
            for effect_config in pipeline:
                if effect_config.get("node_type") == "source":
                    effect_config["source_key"] = "secondary"
        timeline = control_snapshot.get("timeline", {})
        apply_frozen_timeline(
            pipeline,
            timeline.get("tracks", {}),
            frame_time,
            enabled=bool(timeline.get("enabled", False)),
        )
        if control_snapshot.get("srt_enabled") and control_snapshot.get("srt_cues"):
            apply_srt_shader_cues(
                pipeline,
                control_snapshot["srt_cues"],
                frame_time,
                Path(__file__).resolve().parent,
                deck=deck_name,
            )

        global_macros = dict(control_snapshot.get("global_macros", {}))
        extra_params = dict(global_macros)
        modulator = control_snapshot.get("modulator")
        if modulator is not None and modulator.target_deck == deck_name:
            merged_params = dict(global_macros)
            for effect_config in pipeline:
                if effect_config.get("effect_type") == "Display":
                    continue
                merged_params.update(effect_config.get("params") or {})
            modulated = modulator.apply(
                merged_params,
                seconds=frame_time,
                time_scale=float(control_snapshot.get("time_scale", 1.0)),
                rng=python_rng,
                source_values=frame_audio,
            )
            extra_params.update(self._get_active_time_domain_macros(modulated))
            for effect_config in pipeline:
                params = effect_config.get("params") or {}
                for key in list(params):
                    if key in modulated:
                        params[key] = modulated[key]
                effect_config["params"] = params

        audio = frame_audio or control_snapshot.get("audio", {})
        onset = float(audio.get("onset_value", 0.0) or 0.0)
        extra_params.update(
            {
                "time_accum": float(frame_time),
                "noise_warp_time": float(frame_time)
                * float(control_snapshot.get("time_scale", 1.0)),
                "audio_fft_bands": copy.deepcopy(dict(audio.get("fft_bands") or {})),
                "audio_transient": onset,
                "beat_state": bool(audio.get("beat_state", onset >= 0.5)),
            }
        )
        return pipeline, extra_params

    def _build_offline_frame_payload(
        self,
        frame_ordinal,
        source_plan,
        source_slot,
        timeline_origin_seconds,
        base_pipeline_a,
        base_pipeline_b,
        control_snapshot,
    ):
        has_generator = self._pipeline_starts_with_generator(
            base_pipeline_a
        ) or self._pipeline_starts_with_generator(base_pipeline_b)
        uses_secondary = self._pipeline_uses_secondary_source(
            base_pipeline_a
        ) or self._pipeline_uses_secondary_source(base_pipeline_b)
        source_index = int(source_plan.source_indices[frame_ordinal])
        relative_timestamp = float(source_plan.timestamps_seconds[frame_ordinal])
        absolute_timestamp = float(timeline_origin_seconds) + relative_timestamp

        if source_slot == "primary_video":
            primary = self._read_offline_reader_frame(
                self.video_reader, source_index
            )
        elif source_slot == "primary_gif":
            primary = self.gif_frames[source_index].copy()
        elif source_slot == "primary_still" and self.original_image_with_alpha:
            primary = self._apply_background_to_image(
                self.original_image_with_alpha.copy()
            )
        elif source_slot == "primary_still" and self.image:
            primary = self.image.copy()
        elif source_slot == "background_video":
            primary = self._read_offline_reader_frame(
                self.bg_video_reader, source_index
            )
        elif source_slot == "background_gif":
            primary = self.bg_gif_frames[source_index].copy()
        elif source_slot == "background_still":
            primary = self.bg_media.copy()
        elif source_slot == "generator" or has_generator or uses_secondary:
            primary = Image.new(
                "RGB", (self.display_width, self.display_height), (0, 0, 0)
            )
        else:
            raise RuntimeError("No media or generator is available for Offline Master")

        secondary = None
        if uses_secondary or base_pipeline_b:
            secondary = self._get_offline_secondary_frame_at_time(
                absolute_timestamp
            )

        ingress = prepare_master_ingress(
            primary,
            secondary,
            fallback_size=(self.display_width, self.display_height),
        )
        global_macros = dict(control_snapshot.get("global_macros", {}))
        reversal = float(global_macros.get("time_reversal", 0.0))
        warp = float(global_macros.get("local_time_warp", 0.0))
        flow_rate = (1.0 - 2.0 * reversal) * (1.0 + warp)
        frame_time = float(timeline_origin_seconds) + relative_timestamp * flow_rate
        return {
            "ingress": ingress,
            "frame_time": frame_time,
            "frame_ordinal": int(frame_ordinal),
            "source_index": source_index,
            "source_timestamp_seconds": absolute_timestamp,
            "relative_source_timestamp_seconds": relative_timestamp,
        }

    def render_offline_master(self, sender=None, app_data=None, user_data=None):
        if dpg.does_item_exist("offline_master_dialog"):
            dpg.delete_item("offline_master_dialog")
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self._render_offline_master_callback,
            tag="offline_master_dialog",
            modal=True,
        ):
            dpg.add_file_extension(".mp4")

    def _render_offline_master_callback(self, sender, app_data):
        import json

        from megaglitch.codec_domain_post import (
            run_codec_domain_post_network,
        )
        from megaglitch.master_graph import (
            MasterGraphExecutor,
            describe_provider_assignments,
        )
        from megaglitch.offline_master import (
            FFmpegMasterEncoder,
            FpsSemantics,
            OfflineMasterConfig,
            OfflineMasterRenderer,
            stable_graph_hash,
        )

        selected_path = Path(app_data["file_path_name"])
        if selected_path.suffix.lower() != ".mp4":
            selected_path = selected_path.with_suffix(".mp4")
        playback_state = None
        deterministic_output = selected_path
        deterministic_output_preexisted = selected_path.exists()
        deterministic_master_complete = False
        render_cancelled = False
        previous_processing_paused = bool(self.processing_paused)
        self.batch_rendering = True
        self.processing_paused = True
        self.cancel_render_requested = False
        try:
            base_pipeline_a = self.node_editor_a.get_processing_pipeline()
            base_pipeline_b = self.node_editor_b.get_processing_pipeline()
            if not base_pipeline_a and not base_pipeline_b:
                raise RuntimeError("No processing graph is connected to an output")

            playback_state = self._capture_media_playback_state()
            source_plan, source_slot = self._build_offline_source_plan(
                playback_state, base_pipeline_a, base_pipeline_b
            )
            source_fps = float(source_plan.source_rate)
            output_fps = max(0.001, float(self.offline_output_fps))
            source_frame_count = int(source_plan.frame_count)
            if source_slot in {"primary_video", "background_video"}:
                timeline_start_seconds = float(source_plan.start_index) / source_fps
            elif source_slot == "primary_gif":
                timeline_start_seconds = sum(
                    float(value) / 1000.0
                    for value in (self.gif_durations or ())[: source_plan.start_index]
                )
            else:
                timeline_start_seconds = 0.0
            self._prime_media_for_export(playback_state)
            source_audio_analysis = None
            source_audio_path = None
            if source_slot == "primary_video":
                source_audio_path = self.primary_media_path
            elif source_slot == "background_video":
                source_audio_path = self.background_media_path
            if self.offline_analyze_source_audio and source_audio_path:
                try:
                    from megaglitch.offline_audio import decode_source_audio

                    source_audio_analysis = decode_source_audio(
                        source_audio_path,
                        start_seconds=timeline_start_seconds,
                        duration_seconds=float(source_plan.duration_seconds),
                        sample_rate=int(self.offline_audio_sample_rate),
                        ffmpeg_executable=str(self.offline_ffmpeg_executable),
                    )
                except Exception as exc:
                    logging.warning(
                        "Offline source-audio analysis unavailable; using frozen "
                        "control snapshot: %s",
                        exc,
                    )
            control_snapshot = self._capture_offline_control_snapshot()
            control_snapshot["source_audio_document"] = (
                source_audio_analysis.document()
                if source_audio_analysis is not None
                else None
            )

            first_payload = self._build_offline_frame_payload(
                0,
                source_plan,
                source_slot,
                timeline_start_seconds,
                base_pipeline_a,
                base_pipeline_b,
                control_snapshot,
            )
            resolution = first_payload["ingress"].resolution
            executor = MasterGraphExecutor(resolution, use_cuda=bool(self.use_cuda))
            frozen_master_mix = float(self.master_mix)
            frozen_master_transition = str(self.master_transition)
            graph_document = {
                "deck_a": base_pipeline_a,
                "deck_b": base_pipeline_b,
                "master_mix": frozen_master_mix,
                "master_transition": frozen_master_transition,
                "source": {
                    "kind": source_plan.kind.value,
                    "slot": source_slot,
                    "range": str(self.offline_source_range),
                    "start_index": int(source_plan.start_index),
                    "frame_count": source_frame_count,
                    "source_rate": str(source_plan.source_rate),
                    "duration_seconds": str(source_plan.duration_seconds),
                },
                "controls": self._offline_control_document(control_snapshot),
            }
            graph_json_safe = json.loads(
                json.dumps(graph_document, sort_keys=True, default=repr)
            )
            graph_hash = stable_graph_hash(graph_json_safe)
            providers = describe_provider_assignments(
                base_pipeline_a, base_pipeline_b
            )

            work_directory = Path(self.codec_post_work_directory)
            if self.codec_post_enabled:
                work_directory.mkdir(parents=True, exist_ok=True)
                deterministic_output = work_directory / (
                    selected_path.stem + ".deterministic-master.mp4"
                )
            deterministic_output_preexisted = deterministic_output.exists()

            offline_config = OfflineMasterConfig(
                width=resolution.width,
                height=resolution.height,
                source_fps=source_fps,
                output_fps=output_fps,
                fps_semantics=FpsSemantics(self.offline_fps_semantics),
                graph_hash=graph_hash,
                random_seed=int(self.offline_random_seed),
                history_buffer_resolution=executor.history_buffer_resolution,
                node_provider_assignment=providers,
            )
            encoder = FFmpegMasterEncoder(
                self._build_offline_master_encoder_config(
                    deterministic_output,
                    resolution.width,
                    resolution.height,
                    output_fps,
                )
            )

            pending_payloads = {}
            ingress_resize_count = 0
            ingress_resize_bytes = 0
            copy_stats_total = {}
            measured_temporal_inventory = {"deck_a": {}, "deck_b": {}}

            def source_frames():
                nonlocal ingress_resize_count, ingress_resize_bytes, render_cancelled
                with tqdm(
                    total=source_frame_count,
                    desc="Offline Master",
                    unit="master frame",
                ) as progress:
                    for index in range(source_frame_count):
                        if self.cancel_render_requested:
                            render_cancelled = True
                            raise RuntimeError("Offline Master render cancelled")
                        payload = (
                            first_payload
                            if index == 0
                            else self._build_offline_frame_payload(
                                index,
                                source_plan,
                                source_slot,
                                timeline_start_seconds,
                                base_pipeline_a,
                                base_pipeline_b,
                                control_snapshot,
                            )
                        )
                        ingress = payload["ingress"]
                        if ingress.resolution != resolution:
                            raise RuntimeError(
                                "source resolution changed during Offline Master; "
                                "temporal state was not resized"
                            )
                        ingress_resize_count += int(ingress.ingress_resize_count)
                        ingress_resize_bytes += int(ingress.ingress_resize_bytes)
                        pending_payloads[index] = payload
                        primary = ingress.primary
                        if primary is None:
                            primary = np.zeros(resolution.shape, dtype=np.uint8)
                        yield primary
                        progress.update(1)

            def process_master(frame, context):
                payload = pending_payloads.pop(context.source_index)
                ingress = payload["ingress"]
                frame_audio = None
                if source_audio_analysis is not None:
                    ordinal = int(payload["frame_ordinal"])
                    if ordinal + 1 < source_plan.frame_count:
                        frame_duration = float(
                            source_plan.timestamps_seconds[ordinal + 1]
                            - source_plan.timestamps_seconds[ordinal]
                        )
                    else:
                        frame_duration = float(source_plan.duration_seconds) - float(
                            source_plan.timestamps_seconds[ordinal]
                        )
                    features = source_audio_analysis.features_at(
                        payload["relative_source_timestamp_seconds"],
                        frame_duration_seconds=max(
                            frame_duration, 1.0 / float(source_plan.source_rate)
                        ),
                    )
                    frame_audio = {
                        "fft_bands": features.fft_bands,
                        "onset_value": features.transient,
                        "beat_state": features.beat_state,
                        "bpm_pulse": float(features.beat_state),
                        "onset_event_id": (
                            ordinal + 1 if features.transient >= 0.35 else 0
                        ),
                        "beat_event_id": (
                            ordinal + 1 if features.beat_state else 0
                        ),
                    }
                pipeline_a, extra_a = self._prepare_offline_pipeline(
                    base_pipeline_a,
                    payload["frame_time"],
                    "a",
                    control_snapshot,
                    context.python_rng,
                    frame_audio,
                )
                pipeline_b, extra_b = self._prepare_offline_pipeline(
                    base_pipeline_b,
                    payload["frame_time"],
                    "b",
                    control_snapshot,
                    context.python_rng,
                    frame_audio,
                )
                extra_a["frame_num"] = int(payload["frame_ordinal"])
                extra_b["frame_num"] = int(payload["frame_ordinal"])
                result = executor.process(
                    frame,
                    ingress.secondary,
                    pipeline_a=pipeline_a,
                    pipeline_b=pipeline_b,
                    extra_params_a=extra_a,
                    extra_params_b=extra_b,
                    master_mix=frozen_master_mix,
                    master_transition=frozen_master_transition,
                    instrumentation=context.instrumentation,
                    source_index=context.source_index,
                )
                for category, values in result.frame_copy_stats.items():
                    bucket = copy_stats_total.setdefault(
                        category, {"count": 0, "bytes": 0}
                    )
                    bucket["count"] += int(values.get("count", 0))
                    bucket["bytes"] += int(values.get("bytes", 0))
                measured_temporal_inventory["deck_a"] = dict(
                    result.temporal_inventory_a
                )
                measured_temporal_inventory["deck_b"] = dict(
                    result.temporal_inventory_b
                )
                return result.frame

            offline_result = OfflineMasterRenderer(offline_config).render(
                source_frames(),
                source_frame_count=source_frame_count,
                processor=process_master,
                encoder=encoder,
            )
            deterministic_master_complete = True
            telemetry = dict(offline_result.telemetry)
            telemetry["frame_domains"] = {
                "image": "master-resolution graph",
                "temporal_network": "master-resolution graph state",
                "codec": "external post-network" if self.codec_post_enabled else "bypassed",
            }
            telemetry["ingress_resize"] = {
                "count": ingress_resize_count,
                "bytes": ingress_resize_bytes,
            }
            telemetry["frame_copy_stats"] = copy_stats_total
            telemetry["measured_temporal_inventory"] = measured_temporal_inventory
            measured_resolutions = {}
            for deck_name, inventory in measured_temporal_inventory.items():
                for path, record in inventory.items():
                    resolution_value = record.get("resolution")
                    if resolution_value:
                        measured_resolutions[f"{deck_name}.{path}"] = list(
                            resolution_value
                        )
            telemetry["history_buffer_resolution"].update(measured_resolutions)
            telemetry["control_snapshot"] = {
                "included_in_graph_hash": True,
                "audio_analysis": (
                    "deterministic_source_audio"
                    if source_audio_analysis is not None
                    else "frozen_snapshot"
                ),
                "document": self._offline_control_document(control_snapshot),
            }
            telemetry["source_enumeration"] = {
                "kind": source_plan.kind.value,
                "slot": source_slot,
                "range": str(self.offline_source_range),
                "start_index": int(source_plan.start_index),
                "end_index": int(source_plan.source_indices[-1]),
                "declared_source_frames": source_frame_count,
                "unique_source_indices": len(set(source_plan.source_indices)),
                "looped_at_eof": False,
                "timestamps_seconds": [
                    float(value) for value in source_plan.timestamps_seconds
                ],
            }

            if self.codec_post_enabled:
                post_result = run_codec_domain_post_network(
                    self._build_codec_domain_post_config(
                        deterministic_output,
                        selected_path,
                    )
                )
                telemetry["codec_domain_post"] = {
                    "enabled": True,
                    "domains": list(post_result.domains),
                    "stages": [
                        {
                            "stage": stage.stage,
                            "command": list(stage.command),
                            "returncode": stage.returncode,
                            "elapsed_seconds": stage.elapsed_seconds,
                            "stderr": stage.stderr,
                        }
                        for stage in post_result.stages
                    ],
                    "final_output": post_result.final_output,
                }
            else:
                telemetry["codec_domain_post"] = {"enabled": False}

            telemetry_path = selected_path.with_suffix(
                selected_path.suffix + ".telemetry.json"
            )
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            telemetry_path.write_text(
                json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8"
            )
            self.last_offline_master_telemetry = telemetry
            logging.info(
                "Offline Master complete: %s (%s source frames, %s encoded)",
                selected_path,
                telemetry["frame_count"]["source"],
                telemetry["frame_count"]["encoded"],
            )
        except Exception:
            if not render_cancelled:
                logging.error("Offline Master render failed", exc_info=True)
            if (
                not deterministic_master_complete
                and not deterministic_output_preexisted
                and deterministic_output.exists()
            ):
                try:
                    deterministic_output.unlink()
                except OSError:
                    logging.warning(
                        "Could not remove incomplete Offline Master output",
                        exc_info=True,
                    )
        finally:
            self._restore_media_playback_state(playback_state)
            self.batch_rendering = False
            self.processing_paused = previous_processing_paused
            self.cancel_render_requested = False
            if dpg.does_item_exist("offline_master_dialog"):
                dpg.delete_item("offline_master_dialog")

    def cancel_active_render(self, sender, app_data):
        if not self.batch_rendering:
            logging.info("Cancel render requested, but no batch render is active")
            return

        self.cancel_render_requested = True
        logging.info("Cancel requested for active batch render")
        print("[BATCH RENDER] Cancel requested. Stopping after the current chunk.")

    def batch_render_gif(self, sender, app_data):
        """Batch render GIF using parallel processing (much faster than realtime recording)."""
        # Open file dialog to choose save location
        if dpg.does_item_exist("batch_gif_dialog"): 
            dpg.delete_item("batch_gif_dialog")
        with dpg.file_dialog(directory_selector=False, show=True, 
                           callback=self._batch_render_gif_callback, 
                           tag="batch_gif_dialog", modal=True):
            dpg.add_file_extension(".gif")
    
    def _batch_render_gif_callback(self, sender, app_data):
        """Execute batch GIF rendering with chunked processing."""
        print("[BATCH RENDER] ========== BATCH GIF RENDER STARTED ==========")

        self.batch_rendering = True
        self.cancel_render_requested = False
        writer = None
        path = None
        total_written = 0
        render_cancelled = False
        playback_state = None
        try:
            from parallel_processor import analyze_temporal_effects

            try:
                import cuda_effects
                cuda_effects.reset_cuda_states()
            except Exception as e:
                print(f"[BATCH RENDER] Could not reset CUDA states: {e}")

            path = app_data['file_path_name']
            if not path.lower().endswith('.gif'):
                path += '.gif'

            target_fps = max(1, int(self.framerate))
            duration = self.export_duration
            num_frames = int(target_fps * duration)
            playback_state = self._capture_media_playback_state()
            timeline_start_seconds = self._get_export_timeline_start(playback_state, target_fps)
            base_pipeline = self.get_active_editor().get_processing_pipeline()
            if not base_pipeline:
                logging.error("No processing pipeline configured")
                return

            has_temporal, temporal_reasons = analyze_temporal_effects(base_pipeline)
            chunk_size = 30 if has_temporal else 120
            if has_temporal:
                logging.info(f"GIF render using sequential mode: {', '.join(temporal_reasons[:5])}")
            else:
                logging.info("GIF render using parallel mode")
            mode_label = "sequential" if has_temporal else "parallel"
            reason_text = f" | Reasons: {', '.join(temporal_reasons[:5])}" if temporal_reasons else ""
            print(f"[BATCH RENDER] GIF export: {num_frames} frames at {target_fps}fps using {mode_label} mode{reason_text}")
            processing_time = 0.0
            sequential_state = None

            self._prime_media_for_export(playback_state)

            writer = imageio.get_writer(path, mode='I', duration=(1.0 / target_fps), loop=0)
            with tqdm(total=num_frames, desc=f"Rendering GIF ({mode_label})", unit="frame") as progress_bar:
                for start_idx in range(0, num_frames, chunk_size):
                    if self.cancel_render_requested:
                        render_cancelled = True
                        break

                    end_idx = min(num_frames, start_idx + chunk_size)
                    source_frames, secondary_source_frames, modulated_pipelines, extra_params_list = self._build_batch_chunk(
                        start_idx, end_idx, target_fps, base_pipeline, timeline_start_seconds
                    )

                    chunk_start = time.time()
                    processed_frames, sequential_state = self._process_batch_chunk(
                        source_frames,
                        secondary_source_frames,
                        modulated_pipelines,
                        extra_params_list,
                        has_temporal,
                        sequential_state,
                        progress_bar=progress_bar,
                    )
                    processing_time += time.time() - chunk_start

                    if self.cancel_render_requested:
                        render_cancelled = True

                    for img in processed_frames:
                        if self.cancel_render_requested:
                            render_cancelled = True
                            break
                        if img is None:
                            continue
                        writer.append_data(self._prepare_frame_for_export(img))
                        total_written += 1

                    logging.info(f"GIF chunk complete: {end_idx}/{num_frames} frames processed")
                    if render_cancelled:
                        break

            if render_cancelled:
                logging.warning(
                    f"GIF render cancelled after {total_written}/{num_frames} frames"
                )
                print(f"[BATCH RENDER] GIF render cancelled after {total_written}/{num_frames} frames")
                return

            speedup = (duration / processing_time) if processing_time > 0 else 0
            logging.info(
                f"GIF saved: {path} | {total_written} frames @ {target_fps}fps | "
                f"{processing_time:.1f}s ({speedup:.1f}x)"
            )
            print(f"[BATCH RENDER] GIF saved: {path}")

        except Exception as e:
            logging.error(f"Error in batch GIF rendering: {e}", exc_info=True)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
            if render_cancelled and path and os.path.exists(path):
                try:
                    os.remove(path)
                    logging.info(f"Removed partial GIF output after cancel: {path}")
                except Exception:
                    logging.warning(f"Could not remove partial GIF output after cancel: {path}", exc_info=True)
            self._restore_media_playback_state(playback_state)
            self.batch_rendering = False
            self.cancel_render_requested = False
            try:
                dpg.delete_item("batch_gif_dialog")
            except Exception:
                pass
    
    def batch_render_mp4(self, sender, app_data):
        """Batch render MP4 using parallel processing (much faster than realtime recording)."""
        # Open file dialog to choose save location
        if dpg.does_item_exist("batch_mp4_dialog"): 
            dpg.delete_item("batch_mp4_dialog")
        with dpg.file_dialog(directory_selector=False, show=True, 
                           callback=self._batch_render_mp4_callback, 
                           tag="batch_mp4_dialog", modal=True):
            dpg.add_file_extension(".mp4")
    
    def _batch_render_mp4_callback(self, sender, app_data):
        """Execute batch MP4 rendering with chunked processing."""
        print("[BATCH RENDER] ========== BATCH MP4 RENDER STARTED ==========")

        self.batch_rendering = True
        self.cancel_render_requested = False
        writer = None
        path = None
        total_written = 0
        render_cancelled = False
        playback_state = None
        try:
            from parallel_processor import analyze_temporal_effects
            import config

            try:
                import cuda_effects
                cuda_effects.reset_cuda_states()
                print("[BATCH RENDER] Reset CUDA effect states for fresh render")
            except Exception as e:
                print(f"[BATCH RENDER] Could not reset CUDA states: {e}")

            path = app_data['file_path_name']
            if not path.lower().endswith('.mp4'):
                path += '.mp4'

            target_fps = max(1, int(self.framerate))
            duration = self.export_duration
            num_frames = int(target_fps * duration)
            codec = str(getattr(config, 'OPENCV_VIDEO_CODEC', 'mp4v') or 'mp4v')
            playback_state = self._capture_media_playback_state()
            timeline_start_seconds = self._get_export_timeline_start(playback_state, target_fps)

            base_pipeline = self.get_active_editor().get_processing_pipeline()
            if not base_pipeline:
                logging.error("No processing pipeline configured")
                return

            has_temporal, temporal_reasons = analyze_temporal_effects(base_pipeline)
            chunk_size = 30 if has_temporal else 120
            if has_temporal:
                logging.info(f"MP4 render using sequential mode: {', '.join(temporal_reasons[:5])}")
            else:
                logging.info("MP4 render using parallel mode")
            mode_label = "sequential" if has_temporal else "parallel"
            reason_text = f" | Reasons: {', '.join(temporal_reasons[:5])}" if temporal_reasons else ""
            print(f"[BATCH RENDER] MP4 export: {num_frames} frames at {target_fps}fps using {mode_label} mode{reason_text}")
            processing_time = 0.0
            sequential_state = None

            self._prime_media_for_export(playback_state)

            w, h = self.display_width, self.display_height
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, target_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open VideoWriter with codec {codec}")

            with tqdm(total=num_frames, desc=f"Rendering MP4 ({mode_label})", unit="frame") as progress_bar:
                for start_idx in range(0, num_frames, chunk_size):
                    if self.cancel_render_requested:
                        render_cancelled = True
                        break

                    end_idx = min(num_frames, start_idx + chunk_size)
                    source_frames, secondary_source_frames, modulated_pipelines, extra_params_list = self._build_batch_chunk(
                        start_idx, end_idx, target_fps, base_pipeline, timeline_start_seconds
                    )

                    chunk_start = time.time()
                    processed_frames, sequential_state = self._process_batch_chunk(
                        source_frames,
                        secondary_source_frames,
                        modulated_pipelines,
                        extra_params_list,
                        has_temporal,
                        sequential_state,
                        progress_bar=progress_bar,
                    )
                    processing_time += time.time() - chunk_start

                    if self.cancel_render_requested:
                        render_cancelled = True

                    for img in processed_frames:
                        if self.cancel_render_requested:
                            render_cancelled = True
                            break
                        if img is None:
                            continue
                        frame_rgb = self._prepare_frame_for_export(img)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        writer.write(frame_bgr)
                        total_written += 1

                    logging.info(f"MP4 chunk complete: {end_idx}/{num_frames} frames processed")
                    if render_cancelled:
                        break

            if render_cancelled:
                logging.warning(
                    f"MP4 render cancelled after {total_written}/{num_frames} frames"
                )
                print(f"[BATCH RENDER] MP4 render cancelled after {total_written}/{num_frames} frames")
                return

            speedup = (duration / processing_time) if processing_time > 0 else 0
            logging.info(
                f"MP4 saved: {path} | {total_written} frames @ {target_fps}fps | "
                f"{processing_time:.1f}s ({speedup:.1f}x)"
            )

        except Exception as e:
            logging.error(f"Error in batch MP4 rendering: {e}", exc_info=True)
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if render_cancelled and path and os.path.exists(path):
                try:
                    os.remove(path)
                    logging.info(f"Removed partial MP4 output after cancel: {path}")
                except Exception:
                    logging.warning(f"Could not remove partial MP4 output after cancel: {path}", exc_info=True)
            self._restore_media_playback_state(playback_state)
            self.batch_rendering = False
            self.cancel_render_requested = False
            try:
                dpg.delete_item("batch_mp4_dialog")
            except Exception:
                pass

    def apply_display_resolution(self, sender, app_data):
        try:
            new_w = int(dpg.get_value(self.display_width_input))
            new_h = int(dpg.get_value(self.display_height_input))
            if new_w <= 0 or new_h <= 0:
                return
            self.display_width, self.display_height = new_w, new_h
            display_node = self.get_active_editor().get_display_node()
            if not display_node:
                return
            display_node.display_width = new_w
            display_node.display_height = new_h
            # Recreate dynamic texture to new size
            try:
                if dpg.does_item_exist(display_node.texture_tag):
                    dpg.delete_item(display_node.texture_tag)
            except Exception:
                pass
            empty_img = np.zeros((new_h, new_w, 4), dtype=np.float32).flatten()
            dpg.add_dynamic_texture(new_w, new_h, empty_img, tag=display_node.texture_tag, parent=self.texture_registry)
            if display_node.display_image_item is not None:
                try:
                    dpg.configure_item(display_node.display_image_item, texture_tag=display_node.texture_tag)
                    dpg.configure_item(display_node.display_image_item, width=new_w, height=new_h)
                except Exception:
                    pass
        except Exception as e:
            logging.error(f"Failed to apply display resolution: {e}")

    def toggle_virtual_camera(self, sender, app_data):
        enable = bool(app_data)
        if enable and not self.virtual_cam_running:
            try:
                self.virtual_cam = self._create_virtual_camera()
                self._start_virtual_cam_sender()
                self.virtual_cam_running = True
                logging.info(f"Virtual camera started: {self.virtual_cam.device}")
            except Exception as e:
                logging.error(f"Failed to start virtual camera: {e}")
                self._stop_virtual_cam_sender()
                self.virtual_cam = None
                self.virtual_cam_running = False
        elif not enable and self.virtual_cam_running:
            self._stop_virtual_cam_sender()
            self._close_virtual_camera()
            self.virtual_cam_running = False

    def _close_virtual_camera(self):
        try:
            if self.virtual_cam:
                self.virtual_cam.close()
        except Exception:
            pass
        self.virtual_cam = None

    def _start_virtual_cam_sender(self):
        self._stop_virtual_cam_sender(close_camera=False)
        self.virtual_cam_stop_event = threading.Event()
        self.virtual_cam_queue = queue.Queue(maxsize=1)
        self.virtual_cam_thread = threading.Thread(
            target=self._virtual_cam_sender_loop,
            name="VirtualCameraSender",
            daemon=True,
        )
        self.virtual_cam_thread.start()

    def _stop_virtual_cam_sender(self, close_camera=True):
        try:
            self.virtual_cam_stop_event.set()
        except Exception:
            pass
        try:
            if self.virtual_cam_thread and self.virtual_cam_thread.is_alive():
                self.virtual_cam_thread.join(timeout=1.0)
        except Exception:
            pass
        self.virtual_cam_thread = None
        try:
            while not self.virtual_cam_queue.empty():
                self.virtual_cam_queue.get_nowait()
        except Exception:
            pass
        if close_camera:
            self.virtual_cam_stop_event = threading.Event()

    def _enqueue_virtual_cam_frame(self, frame):
        if not self.virtual_cam_running or self.virtual_cam is None:
            return
        try:
            arr = self._to_virtual_cam_rgb(frame)
            if arr is None:
                return
            arr = np.ascontiguousarray(arr)
            try:
                while True:
                    self.virtual_cam_queue.get_nowait()
            except queue.Empty:
                pass
            self.virtual_cam_queue.put_nowait(arr)
        except queue.Full:
            pass
        except Exception as exc:
            logging.debug(f"Virtual cam enqueue failed: {exc}")

    def _virtual_cam_sender_loop(self):
        last_log_time = 0.0
        last_frame = None
        prepared_frame = None
        while not self.virtual_cam_stop_event.is_set():
            latest = None
            try:
                while True:
                    latest = self.virtual_cam_queue.get_nowait()
            except queue.Empty:
                pass
            if latest is not None:
                last_frame = latest
                prepared_frame = None
            if last_frame is None:
                time.sleep(0.002)
                continue

            cam = self.virtual_cam
            if cam is None:
                continue
            try:
                if prepared_frame is None:
                    frame = last_frame
                    if (frame.shape[1], frame.shape[0]) != (cam.width, cam.height):
                        frame = cv2.resize(frame, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)
                    prepared_frame = np.ascontiguousarray(frame, dtype=np.uint8)
                cam.send(prepared_frame)
                cam.sleep_until_next_frame()
            except Exception as exc:
                now = time.perf_counter()
                if now - last_log_time > 2.0:
                    logging.error(f"Virtual cam sender error: {exc}")
                    last_log_time = now

    def _create_virtual_camera(self):
        """
        Prefer Unity Capture on Windows so OBS can consume the output as a source.
        Fall back to pyvirtualcam defaults if Unity Capture is unavailable.
        """
        width = int(self.display_width)
        height = int(self.display_height)
        fps = max(1, int(self.framerate))

        camera_sig = inspect.signature(pyvirtualcam.Camera)
        supported = set(camera_sig.parameters.keys())

        def make_cam(**kwargs):
            filtered = {k: v for k, v in kwargs.items() if k in supported and v is not None}
            return pyvirtualcam.Camera(width=width, height=height, fps=fps, **filtered)

        attempts = []
        # Unity Capture preferred path
        attempts.append({"backend": "unitycapture", "device": "Unity Video Capture"})
        attempts.append({"backend": "unitycapture", "device": "UnityCapture"})
        attempts.append({"backend": "unitycapture"})
        # Generic fallback
        attempts.append({})

        last_error = None
        for attempt in attempts:
            try:
                cam = make_cam(**attempt)
                backend = attempt.get("backend", "auto")
                device = attempt.get("device", getattr(cam, "device", "unknown"))
                logging.info(f"Virtual camera backend selected: {backend}, device: {device}")
                return cam
            except Exception as e:
                last_error = e
                logging.warning(f"Virtual camera attempt failed ({attempt}): {e}")

        raise RuntimeError(f"No virtual camera backend could be started. Last error: {last_error}")

    def _to_virtual_cam_rgb(self, frame):
        """
        Normalize frame input to RGB uint8 ndarray expected by pyvirtualcam.
        Accepts PIL images or numpy arrays (H,W), (H,W,3), (H,W,4).
        """
        if frame is None:
            return None
        if isinstance(frame, Image.Image):
            arr = np.array(frame.convert("RGB"), dtype=np.uint8)
            return arr

        arr = np.asarray(frame)
        if arr.ndim == 2:
            return cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_GRAY2RGB)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                # Worker output is RGBA when alpha is present.
                return cv2.cvtColor(arr.astype(np.uint8, copy=False), cv2.COLOR_RGBA2RGB)
            if arr.shape[2] == 3:
                return arr.astype(np.uint8, copy=False)
        raise ValueError(f"Unsupported frame shape for virtual cam: {getattr(arr, 'shape', None)}")

    def play_video(self, sender, app_data):
        if not self.video_reader and not self.mixer_channel_b_video_reader:
            return
        self.video_playing = True
        self.video_refresh_requested = True
        self.mixer_channel_b_video_refresh_requested = True
        self._sync_video_transport_ui(update_input=False)

    def pause_video(self, sender, app_data):
        if not self.video_reader and not self.mixer_channel_b_video_reader:
            return
        self.video_playing = False
        self._sync_video_transport_ui(update_input=False)

    def stop_video(self, sender, app_data):
        if not self.video_reader and not self.mixer_channel_b_video_reader:
            return
        self.video_playing = False
        self._queue_video_seek(0.0, refresh_frame=True)
        self._sync_video_transport_ui(update_input=False)

    def rewind_video(self, sender, app_data):
        try:
            transport_reader = self.video_reader or self.mixer_channel_b_video_reader
            if transport_reader:
                current_seconds = self._reader_get_time_seconds(transport_reader)
                self._queue_video_seek(current_seconds - 1.0, refresh_frame=True)
                self._sync_video_transport_ui(update_input=False)
        except Exception:
            pass

    def ffwd_video(self, sender, app_data):
        try:
            transport_reader = self.video_reader or self.mixer_channel_b_video_reader
            if transport_reader:
                current_seconds = self._reader_get_time_seconds(transport_reader)
                duration_seconds = self._reader_get_duration_seconds(transport_reader)
                target_seconds = current_seconds + 1.0
                if duration_seconds > 0:
                    target_seconds = min(target_seconds, duration_seconds)
                self._queue_video_seek(target_seconds, refresh_frame=True)
                self._sync_video_transport_ui(update_input=False)
        except Exception:
            pass

    def toggle_gpu(self, sender, app_data):
        try:
            if self._gpu_toggle_syncing:
                return
            # Determine desired state from control value
            if isinstance(app_data, (bool, int)):
                desired = bool(app_data)
            else:
                try:
                    desired = bool(dpg.get_value(sender))
                except Exception:
                    desired = False
            if not self.gpu_available and desired:
                # Revert checkbox visually and warn
                self._gpu_toggle_syncing = True
                for ctrl in (getattr(self, 'cuda_checkbox', None), getattr(self, 'cuda_menu_item', None)):
                    try:
                        if ctrl is not None:
                            dpg.set_value(ctrl, False)
                    except Exception:
                        pass
                self._gpu_toggle_syncing = False
                logging.warning("CUDA not available in this OpenCV build; ignoring GPU toggle.")
                self.use_cuda = False
                # Show a small modal to inform the user
                try:
                    if dpg.does_item_exist("cuda_unavailable_modal"):
                        dpg.delete_item("cuda_unavailable_modal")
                    with dpg.window(label="CUDA Unavailable", modal=True, tag="cuda_unavailable_modal", width=420, height=160, no_resize=True):
                        dpg.add_text("OpenCV is not built with CUDA on this system.\nInstall a CUDA-enabled OpenCV build to use GPU acceleration.")
                        dpg.add_separator()
                        dpg.add_button(label="OK", callback=lambda: dpg.delete_item("cuda_unavailable_modal"))
                except Exception:
                    pass
                return
            self.use_cuda = desired
            # Sync both controls to the same value
            self._gpu_toggle_syncing = True
            for ctrl in (getattr(self, 'cuda_checkbox', None), getattr(self, 'cuda_menu_item', None)):
                try:
                    if ctrl is not None:
                        dpg.set_value(ctrl, self.use_cuda)
                except Exception:
                    pass
            self._gpu_toggle_syncing = False
            try:
                gpu_ops.set_use_cuda(self.use_cuda)
            except Exception:
                pass
        except Exception:
            pass

    # =========================================================================
    # Backend Menu Callbacks (OpenCV/CUDA)
    # =========================================================================
    
    def toggle_opencv_backend(self, sender, app_data):
        """Toggle OpenCV backend on/off."""
        self.opencv_enabled = app_data
        try:
            import config
            config.OPENCV_ENABLED = app_data
            status = "enabled" if app_data else "disabled"
            logging.info(f"OpenCV backend {status}")
            print(f"✓ OpenCV backend {status}")
        except Exception as e:
            logging.error(f"Failed to toggle OpenCV: {e}")
    
    def toggle_cuda_backend(self, sender, app_data):
        """Toggle CUDA acceleration on/off."""
        self.cuda_enabled = app_data
        self.use_cuda = app_data  # Update existing use_cuda flag
        try:
            import config
            config.USE_CUDA_IF_AVAILABLE = app_data
            status = "enabled" if app_data else "disabled"
            logging.info(f"CUDA acceleration {status}")
            print(f"✓ CUDA acceleration {status}")
            
            if app_data:
                # Show warning for first-time enable
                if not self._cuda_warning_shown:
                    self._show_cuda_warning()
                    self._cuda_warning_shown = True
        except Exception as e:
            logging.error(f"Failed to toggle CUDA: {e}")
    
    def show_backend_info(self):
        """Show backend information window."""
        if dpg.does_item_exist("backend_info_window"):
            dpg.delete_item("backend_info_window")
        
        try:
            import opencv_backend as ocv
            info = ocv.get_info()
            
            with dpg.window(label="Backend Information", modal=True, show=True,
                           tag="backend_info_window", pos=[400, 200], width=550, height=450):
                dpg.add_text("OpenCV Backend Status", color=(100, 200, 255))
                dpg.add_separator()
                
                # Status
                available_text = "Yes" if info['available'] else "No"
                available_color = (100, 255, 100) if info['available'] else (255, 100, 100)
                dpg.add_text(f"OpenCV Available:     {available_text}", color=available_color)
                
                if info['available']:
                    dpg.add_text(f"OpenCV Version:       {info['version']}")
                    
                    cuda_text = "Yes" if info.get('cuda_available', False) else "No"
                    cuda_color = (100, 255, 100) if info.get('cuda_available', False) else (255, 150, 100)
                    dpg.add_text(f"CUDA Available:       {cuda_text}", color=cuda_color)
                    
                    # Show detailed CUDA status message
                    if info.get('cuda_check_message'):
                        dpg.add_text(f"CUDA Status:          {info['cuda_check_message'][:60]}", 
                                   color=(150, 150, 150), wrap=530)
                    
                    if info.get('cuda_available'):
                        dpg.add_text(f"CUDA Devices:         {info.get('cuda_device_count', 0)}")
                        if info.get('cuda_device_name'):
                            dpg.add_text(f"Device Name:          {info['cuda_device_name']}")
                
                env_ok = info.get('env_ok', False)
                env_color = (100, 255, 100) if env_ok else (255, 200, 100)
                dpg.add_text(f"Environment:          {info.get('env_check', 'Unknown')[:30]}", color=env_color)
                
                backend_name = "OpenCV" if self.opencv_enabled and info['available'] else "PIL/imageio"
                dpg.add_text(f"Current Backend:      {backend_name}")
                
                gpu_status = "Enabled" if self.cuda_enabled else "Disabled"
                gpu_color = (100, 255, 100) if self.cuda_enabled else (200, 200, 200)
                dpg.add_text(f"GPU Acceleration:     {gpu_status}", color=gpu_color)
                
                # Performance estimates
                if info['available']:
                    dpg.add_separator()
                    dpg.add_text("Performance Estimates:", color=(100, 200, 255))
                    dpg.add_text("  Image Load:    ~15ms  (1.67x faster than PIL)")
                    dpg.add_text("  Video Decode:  ~83fps (2.08x faster than imageio)")
                    dpg.add_text("  Resize/Rotate: ~2ms   (4x faster than PIL)")
                    
                    if info.get('cuda_available') and self.cuda_enabled:
                        dpg.add_separator()
                        dpg.add_text("CUDA Acceleration Active:", color=(100, 255, 100))
                        dpg.add_text("  • GPU-accelerated resize")
                        dpg.add_text("  • GPU-accelerated rotation")
                        dpg.add_text("  • GPU-accelerated blur")
                        dpg.add_text("  • GPU-accelerated kaleidoscope")
                
                # Error info
                if info.get('error'):
                    dpg.add_separator()
                    dpg.add_text("Error:", color=(255, 100, 100))
                    error_text = info['error'][:300]
                    dpg.add_text(error_text, wrap=530)
                
                dpg.add_separator()
                dpg.add_button(label="Close", width=150, callback=lambda: dpg.delete_item("backend_info_window"))
        
        except Exception as e:
            # Fallback if opencv_backend not available
            with dpg.window(label="Backend Information", modal=True, show=True,
                           tag="backend_info_window", pos=[400, 200], width=450, height=250):
                dpg.add_text("Backend Status", color=(100, 200, 255))
                dpg.add_separator()
                dpg.add_text("OpenCV backend not installed", color=(255, 100, 100))
                dpg.add_text(f"Error: {e}")
                dpg.add_separator()
                dpg.add_text("To install OpenCV:", color=(200, 200, 200))
                dpg.add_text("  conda activate cv311")
                dpg.add_text("  conda install -c conda-forge opencv -y")
                dpg.add_separator()
                dpg.add_button(label="Close", width=150, callback=lambda: dpg.delete_item("backend_info_window"))
    
    def show_performance_settings(self):
        """Show performance settings window."""
        if dpg.does_item_exist("perf_settings_window"):
            dpg.delete_item("perf_settings_window")
        
        try:
            import config
            
            with dpg.window(label="Performance Settings", modal=True, show=True,
                           tag="perf_settings_window", pos=[350, 150], width=560, height=560):
                dpg.add_text("Backend Performance Settings", color=(100, 200, 255))
                dpg.add_separator()
                
                # OpenCV preferences
                dpg.add_checkbox(
                    label="Prefer OpenCV for Images",
                    default_value=getattr(config, 'OPENCV_PREFERRED_FOR_IMAGES', True),
                    tag="perf_opencv_images"
                )
                dpg.add_text("  (Use OpenCV for image I/O when available)", indent=20, color=(150, 150, 150))
                
                dpg.add_checkbox(
                    label="Prefer OpenCV for Video",
                    default_value=getattr(config, 'OPENCV_PREFERRED_FOR_VIDEO', True),
                    tag="perf_opencv_video"
                )
                dpg.add_text("  (Use OpenCV for video decode when available)", indent=20, color=(150, 150, 150))
                
                dpg.add_checkbox(
                    label="Enable CUDA Acceleration (Experimental)",
                    default_value=self.cuda_enabled,
                    tag="perf_cuda_enable"
                )
                dpg.add_text("  (GPU acceleration for effects - may be unstable)", indent=20, color=(255, 150, 100))
                
                dpg.add_separator()
                dpg.add_text("Video Export:", color=(100, 200, 255))
                
                # Video codec selection
                codecs = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
                current_codec = getattr(config, 'OPENCV_VIDEO_CODEC', 'mp4v')
                dpg.add_combo(
                    label="Codec",
                    items=codecs,
                    default_value=current_codec,
                    width=150,
                    tag="perf_video_codec"
                )
                
                # Quality slider
                dpg.add_slider_int(
                    label="Quality",
                    min_value=50,
                    max_value=100,
                    default_value=getattr(config, 'VIDEO_EXPORT_QUALITY', 95),
                    width=200,
                    tag="perf_video_quality"
                )
                
                dpg.add_separator()
                dpg.add_text("Worker Settings:", color=(100, 200, 255))
                
                # Worker count
                dpg.add_input_int(
                    label="Worker Processes",
                    default_value=getattr(config, 'NUM_WORKER_PROCESSES', 1),
                    min_value=1,
                    max_value=8,
                    width=100,
                    tag="perf_num_workers"
                )
                dpg.add_text("  (Number of parallel processing workers)", indent=20, color=(150, 150, 150))
                
                # Queue size
                dpg.add_input_int(
                    label="Queue Size",
                    default_value=self.queue_max_size,
                    min_value=1,
                    max_value=10,
                    width=100,
                    tag="perf_queue_size"
                )
                dpg.add_text("  (Lower = less latency, higher = smoother)", indent=20, color=(150, 150, 150))
                
                dpg.add_separator()
                dpg.add_text("Adaptive Preview Scale:", color=(100, 200, 255))
                dpg.add_slider_float(
                    label="Minimum Scale",
                    min_value=0.1,
                    max_value=1.0,
                    default_value=float(self.adaptive_processing_min_scale),
                    width=200,
                    tag="perf_adaptive_min_scale",
                    format="%.2f",
                )
                dpg.add_slider_float(
                    label="Maximum Scale",
                    min_value=0.1,
                    max_value=1.0,
                    default_value=float(self.adaptive_processing_max_scale),
                    width=200,
                    tag="perf_adaptive_max_scale",
                    format="%.2f",
                )
                dpg.add_input_float(
                    label="Target Latency ms",
                    default_value=float(self.adaptive_processing_target_latency_ms),
                    width=120,
                    tag="perf_adaptive_target_latency",
                    format="%.1f",
                )
                dpg.add_text("  (0 = derive from display framerate)", indent=20, color=(150, 150, 150))

                dpg.add_separator()
                
                # Buttons
                dpg.add_button(label="Apply", callback=self._apply_performance_settings, width=100)
                dpg.add_same_line()
                dpg.add_button(label="Reset to Defaults", callback=self._reset_performance_settings, width=150)
                dpg.add_same_line()
                dpg.add_button(label="Cancel", callback=lambda: dpg.delete_item("perf_settings_window"), width=100)
        
        except Exception as e:
            logging.error(f"Failed to show performance settings: {e}")
    
    def _apply_performance_settings(self):
        """Apply performance settings from the window."""
        try:
            import config
            
            # Update config values
            config.OPENCV_PREFERRED_FOR_IMAGES = dpg.get_value("perf_opencv_images")
            config.OPENCV_PREFERRED_FOR_VIDEO = dpg.get_value("perf_opencv_video")
            cuda_enable = dpg.get_value("perf_cuda_enable")
            config.USE_CUDA_IF_AVAILABLE = cuda_enable
            config.OPENCV_VIDEO_CODEC = dpg.get_value("perf_video_codec")
            config.VIDEO_EXPORT_QUALITY = dpg.get_value("perf_video_quality")
            config.NUM_WORKER_PROCESSES = dpg.get_value("perf_num_workers")
            config.QUEUE_MAX_SIZE = dpg.get_value("perf_queue_size")
            min_scale = float(dpg.get_value("perf_adaptive_min_scale"))
            max_scale = float(dpg.get_value("perf_adaptive_max_scale"))
            if min_scale > max_scale:
                min_scale, max_scale = max_scale, min_scale
            
            # Update app state
            self.cuda_enabled = cuda_enable
            self.use_cuda = cuda_enable
            self.queue_max_size = config.QUEUE_MAX_SIZE
            self.adaptive_processing_min_scale = float(np.clip(min_scale, 0.1, 1.0))
            self.adaptive_processing_max_scale = float(np.clip(max_scale, self.adaptive_processing_min_scale, 1.0))
            self.adaptive_processing_target_latency_ms = max(0.0, float(dpg.get_value("perf_adaptive_target_latency") or 0.0))
            self.processing_scale = float(np.clip(self.processing_scale, self.adaptive_processing_min_scale, self.adaptive_processing_max_scale))
            if hasattr(self, "processing_scale_slider") and dpg.does_item_exist(self.processing_scale_slider):
                try:
                    dpg.set_value(self.processing_scale_slider, self.processing_scale)
                except Exception:
                    pass
            
            # Update menu checkbox
            if hasattr(self, 'cuda_toggle_menu'):
                dpg.set_value(self.cuda_toggle_menu, cuda_enable)
            
            logging.info("Performance settings applied")
            print("✓ Performance settings applied successfully!")
            
            if cuda_enable and not self._cuda_warning_shown:
                self._show_cuda_warning()
                self._cuda_warning_shown = True
            
            dpg.delete_item("perf_settings_window")
        
        except Exception as e:
            logging.error(f"Failed to apply performance settings: {e}")
    
    def _reset_performance_settings(self):
        """Reset performance settings to defaults."""
        dpg.set_value("perf_opencv_images", True)
        dpg.set_value("perf_opencv_video", True)
        dpg.set_value("perf_cuda_enable", False)
        dpg.set_value("perf_video_codec", "mp4v")
        dpg.set_value("perf_video_quality", 95)
        dpg.set_value("perf_num_workers", 1)
        dpg.set_value("perf_queue_size", 2)
        if dpg.does_item_exist("perf_adaptive_min_scale"):
            dpg.set_value("perf_adaptive_min_scale", 0.2)
        if dpg.does_item_exist("perf_adaptive_max_scale"):
            dpg.set_value("perf_adaptive_max_scale", 1.0)
        if dpg.does_item_exist("perf_adaptive_target_latency"):
            dpg.set_value("perf_adaptive_target_latency", 0.0)
        print("✓ Settings reset to defaults")
    
    def run_backend_test(self):
        """Run backend smoketest and show results."""
        import subprocess
        import sys
        
        if dpg.does_item_exist("backend_test_window"):
            dpg.delete_item("backend_test_window")
        
        with dpg.window(label="Backend Test", modal=True, show=True,
                       tag="backend_test_window", pos=[300, 150], width=650, height=450):
            dpg.add_text("Running Backend Test...", color=(100, 200, 255))
            dpg.add_separator()
            
            test_output = dpg.add_text("Starting test...\nThis may take up to 30 seconds...", 
                                      wrap=630, tag="test_output_text")
            
            dpg.add_separator()
            dpg.add_button(label="Close", callback=lambda: dpg.delete_item("backend_test_window"))
        
        # Run test in background thread to avoid blocking UI
        def run_test_thread():
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/opencv_smoketest.py"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=Path(__file__).parent
                )
                
                # Parse results
                output = result.stdout
                if result.returncode == 0:
                    # Success
                    summary_start = output.find("TEST SUMMARY")
                    if summary_start != -1:
                        summary = output[summary_start:summary_start+500]
                    else:
                        summary = output[-600:]
                    
                    dpg.set_value("test_output_text", 
                                 "[SUCCESS] All tests passed!\n\n" + summary)
                else:
                    # Failure
                    dpg.set_value("test_output_text", 
                                 "[FAILED] Some tests failed\n\n" + output[-600:])
            
            except subprocess.TimeoutExpired:
                dpg.set_value("test_output_text", 
                             "[TIMEOUT] Test took longer than 30 seconds")
            except FileNotFoundError:
                dpg.set_value("test_output_text", 
                             "[ERROR] Test script not found at scripts/opencv_smoketest.py")
            except Exception as e:
                dpg.set_value("test_output_text", 
                             f"[ERROR] Test failed: {e}")
        
        import threading
        test_thread = threading.Thread(target=run_test_thread, daemon=True)
        test_thread.start()
    
    def _show_cuda_warning(self):
        """Show CUDA warning on first enable."""
        if dpg.does_item_exist("cuda_warning_window"):
            return
        
        with dpg.window(label="CUDA Warning", modal=True, show=True,
                       tag="cuda_warning_window", pos=[400, 250], width=480, height=250):
            dpg.add_text("CUDA Acceleration Enabled", color=(255, 200, 100))
            dpg.add_separator()
            dpg.add_text("Warning: CUDA acceleration is experimental!", color=(255, 150, 100))
            dpg.add_text("\nThis feature may cause:")
            dpg.add_text("  • Application instability or crashes")
            dpg.add_text("  • Incorrect rendering in some effects")
            dpg.add_text("  • Increased GPU memory usage")
            dpg.add_text("  • Driver compatibility issues")
            dpg.add_separator()
            dpg.add_text("Recommendation:", color=(200, 200, 255))
            dpg.add_text("Start with CPU mode. Enable CUDA only if you need")
            dpg.add_text("maximum performance and have a compatible GPU.")
            dpg.add_separator()
            dpg.add_button(label="I Understand", width=150, 
                          callback=lambda: dpg.delete_item("cuda_warning_window"))

def main():
    mp.freeze_support()
    app = MegaGlitchFMApp()
    app.run()

if __name__ == "__main__":
    main()
