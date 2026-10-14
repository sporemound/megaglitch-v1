import numpy as np
import cv2
import gpu_ops as gpo
from megaglitch.determinism import current_numpy_rng
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import math
import time
import logging
import random
import io
from enum import Enum
from collections import defaultdict
from functools import lru_cache
from typing import Optional, Tuple, Dict, Any, Union, List
import math as _math

class CorruptionMode(Enum):
    RANDOM = "random"
    WAVE = "wave"
    BLOCKS = "blocks"
    SELECTIVE = "selective"
    RHYTHMIC = "rhythmic"
    GRADIENT = "gradient"
    TEMPORAL = "temporal"
    MOTION_BASED = "motion_based"
    FREQUENCY_DOMAIN = "frequency_domain"
    DATAMOSHING = "datamoshing"

# ---------- New Effect Helpers (Kaleidoscope / Noise Warp / Watercolor / CRT / VHS) ----------

def _center_crop_np(arr, target_w, target_h):
    h, w = arr.shape[:2]
    x0 = max(0, (w - target_w) // 2)
    y0 = max(0, (h - target_h) // 2)
    return arr[y0:y0 + target_h, x0:x0 + target_w]


def _pad_reflect_np(arr, pad):
    return cv2.copyMakeBorder(arr, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)


def _kaleidoscope_core_np(arr, segments=6, angle_offset=0.0, radius_factor=1.0):
    """Fold image angularly into N segments like a kaleidoscope."""
    h, w = arr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y, x = np.indices((h, w), dtype=np.float32)
    x = x - cx
    y = y - cy
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x) + float(angle_offset)

    seg = max(2, int(segments))
    seg_angle = 2.0 * np.pi / seg
    theta_mod = np.mod(theta, seg_angle)
    theta_fold = np.minimum(theta_mod, seg_angle - theta_mod)

    # Use the rectangular frame bounds instead of a fixed circular aperture.
    # This avoids the bright circular edge ring that shows up when the effect
    # is combined with rotation or animated angle offsets.
    dir_x = np.cos(theta_fold)
    dir_y = np.sin(theta_fold)
    eps = 1e-6
    limit_x = (cx - 1.0) / np.maximum(np.abs(dir_x), eps)
    limit_y = (cy - 1.0) / np.maximum(np.abs(dir_y), eps)
    rmax = np.minimum(limit_x, limit_y) * max(0.01, float(radius_factor))
    r_clamped = np.minimum(r, rmax)

    xn = cx + r_clamped * np.cos(theta_fold)
    yn = cy + r_clamped * np.sin(theta_fold)

    map_x = np.clip(xn, 0, w - 1).astype(np.float32)
    map_y = np.clip(yn, 0, h - 1).astype(np.float32)
    out = gpo.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def kaleidoscope_np(arr, segments=6, angle_offset=0.0, radius_factor=1.0):
    # Overscan the source before folding so edge pixels come from real nearby
    # content instead of repeated border reflections.
    h, w = arr.shape[:2]
    pad = max(2, int(round(max(h, w) * 0.18)))
    padded = _pad_reflect_np(arr, pad)
    folded = _kaleidoscope_core_np(padded, segments=segments, angle_offset=angle_offset, radius_factor=radius_factor)
    return _center_crop_np(folded, w, h)


def rotate_in_frame_np(arr, angle=0.0, overscan=0.18):
    if not angle:
        return arr
    h, w = arr.shape[:2]
    pad = max(2, int(round(max(h, w) * max(0.0, float(overscan)))))
    padded = _pad_reflect_np(arr, pad)
    ph, pw = padded.shape[:2]
    center = (pw * 0.5, ph * 0.5)
    mat = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rotated = gpo.warpAffine(padded, mat, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return _center_crop_np(rotated, w, h)


def _smooth_noise(h, w, scale=64, seed=None):
    rng = np.random.default_rng(seed)
    gh = max(2, int(round(h / max(2, scale))))
    gw = max(2, int(round(w / max(2, scale))))
    grid = rng.standard_normal((gh, gw)).astype(np.float32)
    up = gpo.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    _sigma = float(scale) * 0.25
    _ksize = int(max(1, round(_sigma))) * 2 + 1
    up = gpo.gaussian_blur(up, _ksize, sigmaX=_sigma, sigmaY=_sigma)
    m, s = float(up.mean()), float(up.std() + 1e-6)
    norm = (up - m) / s
    return np.clip(norm, -2.0, 2.0) / 2.0


def noise_field_warp_np(arr, intensity=0.0, scale=64.0, t=0.0):
    """Displace pixels using two smooth noise fields as a vector flow."""
    if intensity <= 0:
        return arr
    h, w = arr.shape[:2]
    nx = _smooth_noise(h, w, scale=max(4.0, float(scale)), seed=int(123 + t * 13))
    ny = _smooth_noise(h, w, scale=max(4.0, float(scale)), seed=int(987 + t * 17))
    max_disp = max(1.0, 0.05 * (h + w) * float(intensity))
    dx = (nx * max_disp).astype(np.float32)
    dy = (ny * max_disp).astype(np.float32)
    coords = np.indices((h, w), dtype=np.float32)
    x_coords = np.clip(coords[1] + dx, 0, w - 1)
    y_coords = np.clip(coords[0] + dy, 0, h - 1)
    return gpo.remap(arr, x_coords, y_coords, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def bilateral_posterize_np(arr, levels=6, bilateral_sigma=12, passes=2):
    """Bilateral smoothing + color quantization for a watercolor/cartoon look."""
    img = arr.astype(np.uint8)
    sm = img
    for _ in range(max(1, int(passes))):
        sm = gpo.bilateral_filter(sm, d=0, sigmaColor=float(bilateral_sigma), sigmaSpace=float(bilateral_sigma))
    levels = max(2, int(levels))
    step = 255.0 / (levels - 1)
    q = np.round(sm.astype(np.float32) / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)


def _rgb_mask(w, h, pattern='rgb'):
    mask = np.ones((h, w, 3), dtype=np.float32)
    if pattern == 'rgb':
        stripes = np.zeros((h, w, 3), dtype=np.float32)
        for x in range(w):
            stripes[:, x, x % 3] = 1.0
        return stripes
    elif pattern == 'shadow':
        dots = np.zeros((h, w, 3), dtype=np.float32)
        for y in range(0, h, 3):
            for x in range(0, w, 3):
                dots[y % h, x % w, 0] = 1.0
                dots[(y + 1) % h, (x + 1) % w, 1] = 1.0
                dots[(y + 2) % h, (x + 2) % w, 2] = 1.0
        return dots
    return mask


def crt_emulation_np(arr, scanline=0.0, mask_type='rgb', curvature=0.0, bloom=0.0):
    """Simple CRT look: scanlines, subpixel mask, optional curvature + bloom."""
    h, w = arr.shape[:2]
    out = arr.astype(np.float32)
    if scanline > 0:
        line = (np.sin(np.arange(h, dtype=np.float32) * np.pi) * 0.5 + 0.5)
        line = 1.0 - float(scanline) * (1.0 - line)
        if out.ndim == 3:
            line = line[:, None, None]
        elif out.ndim == 2:
            line = line[:, None]
        out *= line
    if mask_type and mask_type != 'none':
        mask = _rgb_mask(w, h, pattern=mask_type)
        out *= (0.7 + 0.3 * mask)
    if bloom > 0:
        gray = gpo.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        bright = cv2.threshold(gray, 220, 255, cv2.THRESH_TOZERO)[1]
        bright_rgb = np.stack([bright] * 3, axis=-1)
        blurred = gpo.gaussian_blur(bright_rgb, 7, sigmaX=3, sigmaY=3)
        out = np.clip(out + blurred * float(bloom), 0, 255)
    try:
        if curvature != 0:
            out = barrel_pincushion_np(out, amount=float(curvature))
    except Exception:
        pass
    return out.astype(np.uint8)


def vhs_damage_np(arr, damage=0.0, band_min=4, band_max=24, jitter=0.2, noise=0.2):
    """Horizontal dropouts, head-switching wobble, and additive noise."""
    if damage <= 0:
        return arr
    rng = current_numpy_rng()
    h, w, _ = arr.shape
    out = arr.astype(np.float32).copy()
    num_bands = int(damage * 8)
    for _ in range(num_bands):
        bh = int(rng.integers(max(1, band_min), max(band_min + 1, band_max)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        shift = int(rng.integers(-w // 50, w // 50))
        band = out[y0:y0 + bh]
        band = np.roll(band, shift, axis=1)
        band = band * (0.8 + 0.2 * rng.random()) + rng.uniform(-10, 10)
        out[y0:y0 + bh] = band
    hs = int(h * 0.08)
    for yi in range(h - hs, h):
        phase = (yi - (h - hs)) / max(1, hs)
        sx = int((phase ** 2) * (w * 0.02) * damage * np.sin(phase * 10))
        out[yi] = np.roll(out[yi], sx, axis=0)
    if jitter > 0:
        prob = min(1.0, float(jitter)) * 0.2
        for yi in range(h):
            if rng.random() < prob:
                out[yi] = np.roll(out[yi], int(rng.integers(-3, 4)), axis=0)
    if noise > 0:
        n = rng.normal(0, 12 * float(noise), out.shape).astype(np.float32)
        out = np.clip(out + n, 0, 255)
    return out.astype(np.uint8)

# --- Databending Functions (adapted from databend.py) ---

def _get_header_size(data: bytearray) -> int:
    """Detect and return header size to preserve file structure"""
    file_signatures = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF8': 'GIF',
        b'RIFF': 'WAV',
        b'BM': 'BMP'
    }
    for signature, file_type in file_signatures.items():
        if data.startswith(signature):
            if file_type == 'JPEG':
                for i in range(len(data) - 1):
                    if data[i:i+2] == b'\xff\xda': return i + 12
            elif file_type == 'PNG': return 33
            elif file_type == 'GIF': return 13
            elif file_type == 'WAV': return 44
            elif file_type == 'BMP': return 54
    return 0

def _random_corruption(data: bytearray, start: int, end: int, intensity: float):
    num_corruptions = int((end - start) * intensity)
    for _ in range(num_corruptions):
        idx = random.randint(start, end - 1)
        data[idx] = random.randint(0, 255)

def _wave_corruption(data: bytearray, start: int, end: int, intensity: float):
    length = end - start
    frequency = random.uniform(0.01, 0.1)
    for i in range(start, end):
        wave_val = math.sin((i - start) * frequency * 2 * math.pi)
        if random.random() < intensity * (0.5 + 0.5 * wave_val):
            data[i] = int(128 + 127 * wave_val) % 256

def _block_corruption(data: bytearray, start: int, end: int, intensity: float):
    block_size = random.randint(8, 64)
    num_blocks = int((end - start) * intensity / block_size)
    for _ in range(num_blocks):
        block_start = random.randint(start, max(start, end - block_size))
        corruption_value = random.randint(0, 255)
        for i in range(block_start, min(block_start + block_size, end)):
            data[i] = corruption_value

def _selective_corruption(data: bytearray, start: int, end: int, intensity: float):
    target_values = [0x00, 0xFF, 0x80, 0x40, 0xC0]
    for i in range(start, end):
        if data[i] in target_values and random.random() < intensity * 2:
            data[i] ^= random.randint(1, 255)

def _rhythmic_corruption(data: bytearray, start: int, end: int, intensity: float):
    interval = max(1, int(1 / intensity))
    for i in range(start, end, interval):
        if random.random() < 0.7:
            data[i] = random.randint(0, 255)

def _gradient_corruption(data: bytearray, start: int, end: int, intensity: float):
    length = end - start
    for i in range(start, end):
        position_ratio = (i - start) / length
        if random.random() < intensity * (0.1 + 0.9 * position_ratio):
            data[i] = random.randint(0, 255)

corruption_functions = {
    "random": _random_corruption,
    "wave": _wave_corruption,
    "blocks": _block_corruption,
    "selective": _selective_corruption,
    "rhythmic": _rhythmic_corruption,
    "gradient": _gradient_corruption,
}

def apply_databending_effect(img, intensity, mode, layers, encoding_format='jpeg', jpeg_quality=75, preserve_keyframes=True, frame_skip=0, start_time=0, duration=None):
    if intensity == 0:
        return np.array(img)

    img_rgb = img.convert("RGB")
    
    try:
        # 1. Save image to in-memory buffer
        buffer = io.BytesIO()
        img_rgb.save(buffer, format=encoding_format, quality=jpeg_quality)
        buffer.seek(0)
        data = bytearray(buffer.read())

        # 2. Corrupt the byte data
        header_size = _get_header_size(data)
        corruption_func = corruption_functions.get(mode, _random_corruption)
        
        for layer in range(layers):
            layer_intensity = intensity * (0.8 ** layer)
            corruption_func(data, header_size, len(data), layer_intensity)

        # 3. Load corrupted data back into an image
        corrupted_buffer = io.BytesIO(data)
        bent_img = Image.open(corrupted_buffer)
        
        # Ensure the image has the same size
        if bent_img.size != img.size:
            bent_img = bent_img.resize(img.size, Image.LANCZOS)

        return np.array(bent_img.convert("RGB"))

    except Exception as e:
        logging.warning(f"Databending failed: {e}. Returning original image.")
        return np.array(img_rgb)


slit_scan_buffer = None
slit_scan_buffer_dir = None

# --- Data Moshing Effects (from pureglitch, adapted for MegaGlitchFM) ---

def block_displacement_np(arr, block_size, intensity, direction):
    if intensity == 0 or block_size == 0: return arr
    arr_copy = arr.copy()
    height, width, _ = arr_copy.shape
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            if random.random() < intensity:
                block = arr_copy[y:y+block_size, x:x+block_size].copy()
                if direction == 'horizontal': sx, sy = random.randint(-block_size, block_size), 0
                elif direction == 'vertical': sx, sy = 0, random.randint(-block_size, block_size)
                else: sx, sy = random.randint(-block_size, block_size), random.randint(-block_size, block_size)
                ny, nx = y + sy, x + sx
                if (0 <= ny < height - block_size and 0 <= nx < width - block_size):
                    arr_copy[ny:ny+block_size, nx:nx+block_size] = block
    return arr_copy

def block_duplication_np(arr, block_size, intensity, pattern):
    if intensity == 0 or block_size == 0: return arr
    arr_copy = arr.copy()
    height, width, _ = arr_copy.shape
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            if random.random() < intensity:
                block = arr_copy[y:y+block_size, x:x+block_size].copy()
                if pattern == 'horizontal':
                    for i in range(1, min(4, (width - x) // block_size)):
                        if x + i * block_size < width - block_size: arr_copy[y:y+block_size, x+i*block_size:x+(i+1)*block_size] = block
                elif pattern == 'vertical':
                    for i in range(1, min(4, (height - y) // block_size)):
                        if y + i * block_size < height - block_size: arr_copy[y+i*block_size:y+(i+1)*block_size, x:x+block_size] = block
                elif pattern == 'grid':
                     for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0: continue
                            ny, nx = y + dy * block_size, x + dx * block_size
                            if (0 <= ny < height - block_size and 0 <= nx < width - block_size): arr_copy[ny:ny+block_size, nx:nx+block_size] = block
    return arr_copy

def block_corruption_np(arr, block_size, intensity, corruption_type):
    if intensity == 0 or block_size == 0: return arr
    arr_copy = arr.copy()
    height, width, _ = arr_copy.shape
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            if random.random() < intensity:
                block = arr_copy[y:y+block_size, x:x+block_size]
                if corruption_type == 'noise':
                    noise = np.random.uniform(-50, 50, block.shape)
                    corrupted_block = block + noise
                elif corruption_type == 'invert':
                    corrupted_block = 255.0 - block
                else:
                    block_uint8 = np.clip(block, 0, 255).astype(np.uint8)
                    block_img = Image.fromarray(block_uint8)
                    if corruption_type == 'blur':
                        corrupted_block = np.array(block_img.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.float32)
                    elif corruption_type == 'median':
                        median_color = np.median(block.reshape(-1, 3), axis=0)
                        corrupted_block = np.full(block.shape, median_color, dtype=np.float32)
                    else:
                        corrupted_block = block
                arr_copy[y:y+block_size, x:x+block_size] = corrupted_block
    return arr_copy

def block_sorting_np(arr, block_size, intensity, sort_method):
    if intensity == 0 or block_size == 0: return arr
    arr_copy = arr.copy()
    height, width, _ = arr_copy.shape
    blocks, positions = [], []
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            blocks.append(arr_copy[y:y+block_size, x:x+block_size])
            positions.append((y, x))
    if random.random() < intensity and blocks:
        if sort_method == 'brightness': key = lambda b: np.mean(b)
        else: key = lambda b: np.mean(b[:,:,{'red':0,'green':1,'blue':2}[sort_method]])
        sorted_blocks = sorted(blocks, key=key)
        for i, (y, x) in enumerate(positions):
            arr_copy[y:y+block_size, x:x+block_size] = sorted_blocks[i]
    return arr_copy

def motion_vectors_np(arr, intensity, vector_length):
    if intensity == 0: return arr
    arr_copy = arr.copy()
    height, width, _ = arr_copy.shape
    num_vectors = int((width * height) / 1000 * intensity)
    for _ in range(num_vectors):
        start_x, start_y = random.randint(0, width - 1), random.randint(0, height - 1)
        angle = random.uniform(0, 2 * np.pi)
        end_x, end_y = int(start_x + vector_length * np.cos(angle)), int(start_y + vector_length * np.sin(angle))
        end_x, end_y = max(0, min(width - 1, end_x)), max(0, min(height - 1, end_y))
        steps = max(1, int(np.hypot(end_x - start_x, end_y - start_y)))
        source_pixel = arr_copy[start_y, start_x].copy()
        for step in range(steps):
            t = step / steps
            x, y = int(start_x + t * (end_x - start_x)), int(start_y + t * (end_y - start_y))
            if 0 <= x < width and 0 <= y < height:
                arr_copy[y, x] = arr_copy[y, x] * 0.7 + source_pixel * 0.3
    return arr_copy

def bitplane_corruption_np(arr, intensity, bitplane):
    if intensity == 0 or random.random() >= intensity: return arr
    int_arr = np.clip(arr, 0, 255).astype(np.uint8)
    mask = np.uint8(1 << bitplane)
    corruption = np.random.randint(0, 2, int_arr.shape, dtype=np.uint8) * mask
    int_arr = (int_arr & ~mask) | corruption
    return int_arr.astype(np.float32)

def dct_glitch_np(arr, intensity):
    if intensity == 0:
        return arr

    arr_copy = arr.copy()
    height, width, _ = arr_copy.shape
    block_size = 8

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            if random.random() < intensity:
                for c in range(3): # For each color channel
                    block = arr_copy[y:y+block_size, x:x+block_size, c]
                    
                    if block.shape[0] < block_size or block.shape[1] < block_size:
                        continue

                    # Apply DCT
                    dct_block = cv2.dct(block.astype(np.float32))

                    # Manipulate DCT coefficients
                    # For example, zero out some high-frequency coefficients
                    if dct_block.shape[0] > 4 and dct_block.shape[1] > 4:
                        dct_block[4:, 4:] = 0

                    # Apply inverse DCT
                    idct_block = cv2.idct(dct_block)

                    arr_copy[y:y+block_size, x:x+block_size, c] = idct_block

    return arr_copy

def slit_scan_smear_np(arr, intensity, direction='right_to_left', size=1, pos=0.5, external_state=None):
    """Slit-scan smear effect supporting multiple directions.
    direction: 'right_to_left', 'left_to_right', 'top_to_bottom', 'bottom_to_top'
    
    IMPORTANT: 
    - size should be >= 16 for visibility
    - intensity should be >= 0.8 for strong effect
    - Works best with moving/changing sources
    """
    import logging
    global slit_scan_buffer, slit_scan_buffer_dir
    
    if intensity == 0:
        return arr

    h, w, _ = arr.shape

    # Select state storage
    if external_state is not None and isinstance(external_state, dict):
        buf = external_state.get('buffer', None)
        sdir = external_state.get('dir', None)
    else:
        buf = slit_scan_buffer
        sdir = slit_scan_buffer_dir

    # Reset buffer if shape or direction changed
    if buf is None or getattr(buf, 'shape', None) != arr.shape or sdir != direction:
        logging.info(f"SLIT SCAN: Initializing buffer {arr.shape}, direction={direction}, size={size}")
        # Initialize with current frame (not zeros) for immediate effect
        buf = arr.copy()
        sdir = direction

    # Normalize inputs
    try:
        sw = max(1, int(size))
    except Exception:
        sw = 1
    try:
        pos = float(pos)
    except Exception:
        pos = 0.5
    pos = min(1.0, max(0.0, pos))

    # Slice width/height
    if direction == 'right_to_left':
        sw = min(sw, w)
        x0 = int(round(pos * (w - sw)))
        slice_img = arr[:, x0:x0 + sw, :]
        buf = np.roll(buf, -sw, axis=1)
        buf[:, -sw:, :] = slice_img
    elif direction == 'left_to_right':
        sw = min(sw, w)
        x0 = int(round(pos * (w - sw)))
        slice_img = arr[:, x0:x0 + sw, :]
        buf = np.roll(buf, sw, axis=1)
        buf[:, :sw, :] = slice_img
    elif direction == 'top_to_bottom':
        sh = min(sw, h)
        y0 = int(round(pos * (h - sh)))
        slice_img = arr[y0:y0 + sh, :, :]
        buf = np.roll(buf, sh, axis=0)
        buf[:sh, :, :] = slice_img
    else:  # 'bottom_to_top'
        sh = min(sw, h)
        y0 = int(round(pos * (h - sh)))
        slice_img = arr[y0:y0 + sh, :, :]
        buf = np.roll(buf, -sh, axis=0)
        buf[-sh:, :, :] = slice_img

    # Blend the original image with the slit scan buffer
    # Use more aggressive blending for stronger effect
    blend_intensity = float(intensity)
    output_arr = gpo.addWeighted(arr, 1 - blend_intensity, buf, blend_intensity, 0)

    # Write back state
    if external_state is not None and isinstance(external_state, dict):
        external_state['buffer'] = buf
        external_state['dir'] = direction
    else:
        slit_scan_buffer = buf
        slit_scan_buffer_dir = direction

    return output_arr

# ---- Conventional Image Adjustments ----

def apply_brightness_contrast(arr, brightness=0.0, contrast=0.0):
    if brightness == 0 and contrast == 0:
        return arr
    img = arr.astype(np.float32)
    # brightness in [-100,100], contrast in [-100,100]
    b = float(brightness)
    c = float(contrast)
    alpha = 1.0 + (c / 100.0)
    beta = b
    out = img * alpha + beta
    return np.clip(out, 0, 255)

def apply_gamma(arr, gamma=1.0):
    if gamma == 1.0 or gamma <= 0:
        return arr
    g = float(gamma)
    img = np.clip(arr, 0, 255) / 255.0
    img = np.power(img, 1.0 / g)
    return np.clip(img * 255.0, 0, 255)

def apply_hsv_adjust(arr, hue_shift=0.0, saturation=1.0):
    if hue_shift == 0.0 and saturation == 1.0:
        return arr
    img8 = np.clip(arr, 0, 255).astype(np.uint8)
    hsv = gpo.cvtColor(img8, cv2.COLOR_RGB2HSV).astype(np.float32)
    if hue_shift:
        hsv[..., 0] = (hsv[..., 0] + float(hue_shift)) % 180.0
    if saturation != 1.0:
        hsv[..., 1] = np.clip(hsv[..., 1] * float(saturation), 0, 255)
    out = gpo.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    return out

@lru_cache(maxsize=32)
def _color_adjust_tables(hue_shift, saturation, gamma):
    """Immutable, bounded tables using the legacy operations' exact precision."""
    hsv_active = hue_shift != 0.0 or saturation != 1.0
    hsv_table = None
    if hsv_active:
        values = np.arange(256, dtype=np.float32)
        hsv_table = np.repeat(values[:, None], 3, axis=1)
        if hue_shift:
            hsv_table[:, 0] = (values + float(hue_shift)) % 180.0
        if saturation != 1.0:
            hsv_table[:, 1] = np.clip(values * float(saturation), 0, 255)
        hsv_table = hsv_table.astype(np.uint8).reshape(256, 1, 3)
        hsv_table.setflags(write=False)
    gamma_table = None
    if gamma != 1.0 and gamma > 0:
        # HSV returns float32; bypassing HSV leaves RGB8, whose division in
        # apply_gamma promotes to float64. Preserve both rounding paths.
        values = np.arange(256, dtype=np.float32 if hsv_active else np.uint8)
        gamma_table = apply_gamma(values, gamma).astype(np.uint8)
        gamma_table.setflags(write=False)
    return hsv_table, gamma_table


def apply_color_adjust_rgb8(arr, hue_shift=0.0, saturation=1.0, gamma=1.0):
    """CPU RGB8 equivalent of HSV then gamma, without full-frame float math."""
    hsv_table, gamma_table = _color_adjust_tables(hue_shift, saturation, gamma)
    output = arr
    if hsv_table is not None:
        hsv = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
        hsv = cv2.LUT(hsv, hsv_table)
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if gamma_table is not None:
        output = cv2.LUT(output, gamma_table)
    return output


def apply_vignette(arr, intensity=0.0, radius=0.75):
    if intensity <= 0:
        return arr
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rmax = np.sqrt(cx ** 2 + cy ** 2) * max(0.001, float(radius))
    mask = 1.0 - np.clip(r / rmax, 0, 1)
    mask = (mask * (1.0 - float(intensity)) + float(intensity) * mask ** 2).astype(np.float32)
    return arr * mask[..., None]

def apply_blur(arr, radius=0.0, use_cuda=False):
    """Apply Gaussian blur with optional CUDA acceleration."""
    if radius <= 0:
        return arr
    # derive kernel size from radius
    k = int(max(1, round(radius))) * 2 + 1
    
    # Try CUDA path if enabled
    if use_cuda:
        try:
            import opencv_backend as ocv
            if ocv.is_cuda_available():
                # Convert to uint8 for CUDA processing
                arr_u8 = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
                result = ocv.gaussian_blur_cuda(arr_u8, k, 0)
                # Convert back to original dtype
                if arr.dtype == np.float32:
                    return result.astype(np.float32)
                return result
        except Exception as e:
            logging.warning(f"CUDA blur failed, using CPU: {e}")
    
    # CPU path
    return gpo.gaussian_blur(arr, k, 0)

def apply_sharpen(arr, amount=0.0, radius=2):
    if amount <= 0:
        return arr
    k = int(max(1, round(radius))) * 2 + 1
    blurred = gpo.gaussian_blur(arr, k, 0)
    out = np.clip(arr + (arr - blurred) * float(amount), 0, 255)
    return out

def apply_bloom(arr, strength=0.0, threshold=200):
    if strength <= 0:
        return arr
    gray = gpo.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = (gray > float(threshold)).astype(np.float32)
    bright = arr * mask[..., None]
    blurred = gpo.gaussian_blur(bright, 7, sigmaX=5, sigmaY=5)
    return np.clip(arr + blurred * float(strength), 0, 255)

def apply_edges_emboss(arr, edge_mix=0.0, emboss=0.0):
    out = arr
    if edge_mix > 0:
        gray = gpo.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200).astype(np.float32)
        edges_rgb = np.stack([edges, edges, edges], axis=-1)
        out = np.clip(out * (1.0 - edge_mix) + edges_rgb * edge_mix, 0, 255)
    if emboss > 0:
        kernel = np.array([[-2, -1, 0],
                           [-1, 1, 1],
                           [0, 1, 2]], dtype=np.float32)
        emb = cv2.filter2D(out, -1, kernel) + 128.0
        out = np.clip(out * (1.0 - emboss) + emb * emboss, 0, 255)
    return out

# ---- Geometric/Obscure Effects ----

def barrel_pincushion_np(arr, amount=0.0):
    if amount == 0:
        return arr
    h, w = arr.shape[:2]
    # create coordinate grid normalized to [-1,1]
    y, x = np.indices((h, w), dtype=np.float32)
    x = (2.0 * x / (w - 1)) - 1.0
    y = (2.0 * y / (h - 1)) - 1.0
    r2 = x * x + y * y
    k = float(amount)  # positive -> barrel, negative -> pincushion
    x1 = x * (1 + k * r2)
    y1 = y * (1 + k * r2)
    # map back to image coords
    map_x = ((x1 + 1.0) * 0.5 * (w - 1)).astype(np.float32)
    map_y = ((y1 + 1.0) * 0.5 * (h - 1)).astype(np.float32)
    return gpo.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def swirl_np(arr, amount=0.0, radius_factor=0.75):
    if amount == 0:
        return arr
    h, w = arr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    y, x = np.indices((h, w), dtype=np.float32)
    x -= cx; y -= cy
    r = np.sqrt(x * x + y * y)
    rmax = min(cx, cy) * max(0.01, float(radius_factor))
    theta = np.arctan2(y, x) + (1.0 - np.clip(r / rmax, 0, 1)) * float(amount)
    rn = np.minimum(r, rmax)
    xn = cx + rn * np.cos(theta)
    yn = cy + rn * np.sin(theta)
    map_x = xn.astype(np.float32)
    map_y = yn.astype(np.float32)
    return gpo.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def ripple_np(arr, amp=0.0, freq=3.0, t=0.0):
    if amp == 0:
        return arr
    h, w = arr.shape[:2]
    y, x = np.indices((h, w), dtype=np.float32)
    dx = (np.sin(2 * np.pi * (y / h * freq) + t) * float(amp)).astype(np.float32)
    dy = (np.sin(2 * np.pi * (x / w * freq) + t * 0.7) * float(amp)).astype(np.float32)
    map_x = np.clip(x + dx, 0, w - 1)
    map_y = np.clip(y + dy, 0, h - 1)
    return gpo.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def halftone_np(arr, cell=8, contrast=1.0):
    cell = int(max(2, round(cell)))
    h, w = arr.shape[:2]
    out = np.ones((h, w, 3), dtype=np.float32) * 255.0
    gray = gpo.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            block = gray[y:y+cell, x:x+cell]
            if block.size == 0:
                continue
            m = 1.0 - (np.mean(block) / 255.0)
            r = int(m * (cell / 2.0) * float(contrast))
            cx, cy = x + cell // 2, y + cell // 2
            cv2.circle(out, (cx, cy), max(0, r), (0, 0, 0), -1, lineType=cv2.LINE_AA)
    return out

def vhs_jitter_np(arr, intensity=0.0, max_shift=8):
    if intensity <= 0:
        return arr
    rng = current_numpy_rng()
    h, w, c = arr.shape
    out = arr.copy()
    # randomly shift some scanlines
    prob = min(1.0, float(intensity)) * 0.2
    for y in range(h):
        if rng.random() < prob:
            shift = int(rng.integers(-max_shift, max_shift + 1))
            out[y] = np.roll(out[y], shift, axis=0)
    return out

def rolling_shutter_wobble_np(arr, intensity=0.0, freq=3.0, phase=0.0):
  if intensity <= 0: return arr
  h, w, c = arr.shape
  y = np.arange(h, dtype=np.float32)[:, None]
  shift = (np.sin(2*np.pi*(y/h*freq) + phase) * intensity * w * 0.02).astype(np.float32)
  x = np.arange(w, dtype=np.float32)[None, :]
  map_x = (x + shift) % w
  map_y = np.repeat(y, w, axis=1)
  return gpo.remap(arr, map_x.astype(np.float32), map_y.astype(np.float32),
                   interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def scanline_shuffle_np(arr, intensity=0.0, max_span=32, prob=0.1, rng=None):
    if intensity <= 0: return arr
    rng = rng or current_numpy_rng()
    out = arr.copy()
    h, w, _ = out.shape
    spans = rng.integers(1, max(2, max_span), size=h)
    for y in range(h):
        if rng.random() < prob * intensity:
            span = int(spans[y])
            start = int(rng.integers(0, max(1, w-span)))
            seg = out[y, start:start+span].copy()
            shift = int(rng.integers(-span, span))
            dst = np.clip(start+shift, 0, w-span)
            out[y, dst:dst+span] = seg
    return out

def chroma_leak_np(arr, intensity=0.0, subsample='420', drift_px=2, rng=None):
    """Chroma leak using YUV space with safe blending to avoid color channel dominance."""
    if intensity <= 0:
        return arr
    rng = rng or current_numpy_rng()
    # Convert to YUV (Y, U, V) which is unambiguous in OpenCV
    yuv = gpo.cvtColor(arr, cv2.COLOR_RGB2YUV).astype(np.float32)
    Y, U, V = yuv[..., 0], yuv[..., 1], yuv[..., 2]

    def down_up(ch, mode):
        if mode == '420':
            ch_d = gpo.resize(ch, (max(1, ch.shape[1]//2), max(1, ch.shape[0]//2)), interpolation=cv2.INTER_AREA)
        elif mode == '422':
            ch_d = gpo.resize(ch, (max(1, ch.shape[1]//2), ch.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            ch_d = ch
        return gpo.resize(ch_d, (ch.shape[1], ch.shape[0]), interpolation=cv2.INTER_LINEAR)

    U_s = down_up(U, subsample)
    V_s = down_up(V, subsample)

    dx = int(rng.integers(-drift_px, drift_px + 1))
    dy = int(rng.integers(-drift_px, drift_px + 1))
    M = np.float32([[1, 0, dx * float(intensity)], [0, 1, dy * float(intensity)]])
    U_s = gpo.warpAffine(U_s, M, (U_s.shape[1], U_s.shape[0]), borderMode=cv2.BORDER_REFLECT)
    V_s = gpo.warpAffine(V_s, M, (V_s.shape[1], V_s.shape[0]), borderMode=cv2.BORDER_REFLECT)

    out_yuv = np.stack([Y, U_s, V_s], axis=-1)
    out_yuv = np.clip(out_yuv, 0, 255).astype(np.uint8)
    out_rgb = gpo.cvtColor(out_yuv, cv2.COLOR_YUV2RGB)
    # Blend with original to prevent extreme casts; scale blend by intensity
    alpha = float(max(0.0, min(1.0, intensity))) * 0.8
    return gpo.addWeighted(arr.astype(np.uint8), 1.0 - alpha, out_rgb.astype(np.uint8), alpha, 0)

def fft_phase_scramble_np(arr, intensity=0.0, bands=3, rng=None):
    if intensity <= 0: return arr
    rng = rng or current_numpy_rng()
    out = np.empty_like(arr)
    for c in range(3):
        f = np.fft.rfft2(arr[...,c])
        mag, ph = np.abs(f), np.angle(f)
        # bandwise jitter
        H, W = ph.shape
        splits = np.linspace(0, min(H,W), bands+1, dtype=int)
        for i in range(bands):
            y0,y1 = splits[i], splits[i+1]
            x0,x1 = splits[i], splits[i+1]
            jitter = (rng.standard_normal((y1-y0, x1-x0)) * np.pi * intensity).astype(np.float32)
            ph[y0:y1, x0:x1] += jitter
        f2 = mag * np.exp(1j*ph)
        out[...,c] = np.fft.irfft2(f2, s=arr.shape[:2]).real
    return np.clip(out, 0, 255).astype(np.uint8)

def row_temporal_stutter(frame_buffer, intensity=0.0, max_delay=4, rng=None):
    if intensity <= 0 or not frame_buffer:
        return np.array(frame_buffer[-1])
    rng = rng or current_numpy_rng()
    stack = [np.array(f) for f in frame_buffer[-(max_delay+1):]]
    curr = stack[-1]
    h, w, c = curr.shape
    out = curr.copy()
    delays = (rng.random(h) * max_delay * intensity).astype(int)
    for y in range(h):
        d = delays[y]
        src = stack[-1-d]
        out[y:y+1] = src[y:y+1]
    return out

def palette_crunch_np(arr, levels=6, dither=True, rng=None):
    rng = rng or current_numpy_rng()
    if levels <= 1: return arr
    step = 255.0 / (levels-1)
    if dither:
        noise = rng.uniform(-step/2, step/2, arr.shape).astype(np.float32)
        arrf = np.clip(arr.astype(np.float32) + noise, 0, 255)
    else:
        arrf = arr.astype(np.float32)
    q = np.round(arrf / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)

def channel_flow_offset_np(arr, intensity=0.0, max_px=8, rng=None):
    if intensity <= 0: return arr
    rng = rng or current_numpy_rng()
    h, w, _ = arr.shape
    out = np.zeros_like(arr)
    for ch in range(3):
        dx = int(rng.integers(-max_px, max_px+1) * intensity)
        dy = int(rng.integers(-max_px, max_px+1) * intensity)
        M = np.float32([[1,0,dx],[0,1,dy]])
        out[...,ch] = gpo.warpAffine(arr[...,ch], M, (w,h), borderMode=cv2.BORDER_REFLECT)
    return out

def mesh_tear_np(arr, intensity=0.0, grid=6, rng=None):
    if intensity <= 0: return arr
    rng = rng or current_numpy_rng()
    h, w, _ = arr.shape
    ys = np.linspace(0, h, grid+1, dtype=np.int32)
    xs = np.linspace(0, w, grid+1, dtype=np.int32)
    src_pts, dst_pts = [], []
    jitter = int(max(h, w) * 0.02 * intensity)
    for y in ys:
        for x in xs:
            src_pts.append([x, y])
            dx = int(rng.integers(-jitter, jitter+1))
            dy = int(rng.integers(-jitter, jitter+1))
            dst_pts.append([np.clip(x+dx,0,w-1), np.clip(y+dy,0,h-1)])
    src = np.float32(src_pts).reshape((grid+1, grid+1, 2))
    dst = np.float32(dst_pts).reshape((grid+1, grid+1, 2))
    # warp via dense map
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
            t = (np.arange(ny, dtype=np.float64) / max(1.0, float(ny)))[:, None, None]
            s = (np.arange(nx, dtype=np.float64) / max(1.0, float(nx)))[None, :, None]
            left = tl * (1.0 - t) + bl * t
            right = tr * (1.0 - t) + br * t
            p = left * (1.0 - s) + right * s
            map_x[ys0:ys1, xs0:xs1] = p[..., 0].astype(np.float32)
            map_y[ys0:ys1, xs0:xs1] = p[..., 1].astype(np.float32)
    return gpo.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# --- P-Frame Specific Manipulation Functions ---

def corrupt_motion_vectors(current_frame, reference_frame, intensity, block_size=16, direction='random', rng=None):
    """
    Simulates corrupted motion vectors by applying displacements in specified directions
    that would occur if P-frame motion vectors were damaged.

    Args:
        direction: 'random', 'horizontal', 'vertical', 'diagonal', 'radial', 'spiral', 'wave'
    """
    if intensity == 0 or reference_frame is None:
        return current_frame

    rng = rng or random

    current = np.asarray(current_frame, dtype=np.uint8)
    reference = np.asarray(reference_frame, dtype=np.uint8)

    if current.shape != reference.shape:
        logging.warning(
            "Discarding motion-vector reference at %s; master shape is %s",
            reference.shape,
            current.shape,
        )
        return current_frame

    height, width = current.shape[:2]
    result = current.copy()
    center_x, center_y = width // 2, height // 2

    # Process in macroblocks (typical for video compression)
    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            if rng.random() < intensity:
                max_displacement = int(block_size * intensity * 2)

                # Calculate displacement based on direction
                if direction == 'horizontal':
                    dx = rng.randint(-max_displacement, max_displacement)
                    dy = 0
                elif direction == 'vertical':
                    dx = 0
                    dy = rng.randint(-max_displacement, max_displacement)
                elif direction == 'diagonal':
                    displacement = rng.randint(-max_displacement, max_displacement)
                    dx = displacement
                    dy = displacement if rng.random() < 0.5 else -displacement
                elif direction == 'radial':
                    # Move away from or toward center
                    block_center_x = x + block_size // 2
                    block_center_y = y + block_size // 2
                    angle = math.atan2(block_center_y - center_y, block_center_x - center_x)
                    displacement = rng.randint(-max_displacement, max_displacement)
                    dx = int(displacement * math.cos(angle))
                    dy = int(displacement * math.sin(angle))
                elif direction == 'spiral':
                    # Spiral motion based on distance from center
                    block_center_x = x + block_size // 2
                    block_center_y = y + block_size // 2
                    dist = math.sqrt((block_center_x - center_x)**2 + (block_center_y - center_y)**2)
                    angle = math.atan2(block_center_y - center_y, block_center_x - center_x)
                    spiral_angle = angle + (dist * 0.01 * intensity * 10)  # Spiral factor
                    displacement = max_displacement * 0.5
                    dx = int(displacement * math.cos(spiral_angle))
                    dy = int(displacement * math.sin(spiral_angle))
                elif direction == 'wave':
                    # Wave-based displacement
                    wave_freq = 0.1
                    wave_x = math.sin(x * wave_freq) * max_displacement
                    wave_y = math.cos(y * wave_freq) * max_displacement
                    dx = int(wave_x)
                    dy = int(wave_y)
                else:  # 'random'
                    dx = rng.randint(-max_displacement, max_displacement)
                    dy = rng.randint(-max_displacement, max_displacement)

                # Find source block in reference frame
                src_x = max(0, min(width - block_size, x + dx))
                src_y = max(0, min(height - block_size, y + dy))

                # Copy block from displaced location (simulating bad motion vector)
                source_block = reference[src_y:src_y + block_size, src_x:src_x + block_size]
                result[y:y + block_size, x:x + block_size] = source_block

    return result


def process_pframe_spiral_uniform(current_frame, reference_frame, motion_intensity,
                                  block_size, prediction_intensity, error_threshold=30,
                                  output=None, rng=None):
    """Try the coarse native spiral-motion + uniform-prediction segment."""
    if reference_frame is None or motion_intensity <= 0 or prediction_intensity <= 0:
        return None
    current = np.asarray(current_frame)
    reference = np.asarray(reference_frame)
    if (
        current.dtype != np.uint8
        or reference.dtype != np.uint8
        or current.shape != reference.shape
        or current.ndim != 3
        or current.shape[2] != 3
        or not current.flags.c_contiguous
        or not reference.flags.c_contiguous
        or int(block_size) <= 0
    ):
        return None
    try:
        from megaglitch.native_temporal import native_available, process_pframe_spiral_uniform as native_segment
        if not native_available():
            return None
    except ImportError:
        return None

    block_size = int(block_size)
    height, width = current.shape[:2]
    blocks_x = 0 if width <= block_size else (width - 1) // block_size
    blocks_y = 0 if height <= block_size else (height - 1) // block_size
    block_count = blocks_x * blocks_y
    rng = rng or random
    random_state = rng.getstate()
    random_values = np.fromiter(
        (rng.random() for _ in range(block_count)), dtype=np.float64,
        count=block_count,
    )
    try:
        return native_segment(
            current, reference, motion_intensity, block_size, random_values,
            prediction_intensity, error_threshold, output,
        )
    except Exception:
        # The Python fallback must consume the same random sequence it always did.
        rng.setstate(random_state)
        return None

def amplify_prediction_errors(current_frame, reference_frame, intensity, error_threshold=30, direction='uniform'):
    """
    Amplifies prediction errors with directional bias.

    Args:
        direction: 'uniform', 'horizontal', 'vertical', 'diagonal', 'radial'
    """
    if intensity == 0 or reference_frame is None:
        return current_frame

    if direction == 'uniform':
        current_u8 = np.asarray(current_frame)
        reference_u8 = np.asarray(reference_frame)
        if (
            current_u8.dtype == np.uint8
            and reference_u8.dtype == np.uint8
            and current_u8.shape == reference_u8.shape
            and current_u8.flags.c_contiguous
            and reference_u8.flags.c_contiguous
        ):
            try:
                from megaglitch.native_temporal import amplify_prediction_errors_uniform
                native_result = amplify_prediction_errors_uniform(
                    current_u8, reference_u8, intensity, error_threshold
                )
                if native_result is not None:
                    return native_result
            except Exception:
                # The source tree remains fully functional before/without a native build.
                pass

    current = np.array(current_frame, dtype=np.float32)
    reference = np.array(reference_frame, dtype=np.float32)

    if current.shape != reference.shape:
        logging.warning(
            "Discarding prediction reference at %s; master shape is %s",
            reference.shape,
            current.shape,
        )
        return current_frame

    height, width = current.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Calculate prediction error (residual)
    residual = current - reference

    # Create directional amplification mask
    if direction == 'horizontal':
        # Amplify more on left/right edges
        x_coords = np.arange(width)
        amplification_mask = np.abs(x_coords - center_x) / (width / 2)
        amplification_mask = np.tile(amplification_mask, (height, 1))
    elif direction == 'vertical':
        # Amplify more on top/bottom edges
        y_coords = np.arange(height)
        amplification_mask = np.abs(y_coords - center_y) / (height / 2)
        amplification_mask = np.tile(amplification_mask.reshape(-1, 1), (1, width))
    elif direction == 'diagonal':
        # Amplify more on diagonal edges
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        diagonal_dist = np.abs(x_coords - y_coords * width / height) / (width / 2)
        amplification_mask = np.clip(diagonal_dist, 0, 1)
    elif direction == 'radial':
        # Amplify more towards edges
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        radial_dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        amplification_mask = radial_dist / max_dist
    else:  # 'uniform'
        amplification_mask = np.ones((height, width))

    # Apply directional amplification
    error_mask = np.abs(residual) > error_threshold
    amplification_factor = 1 + intensity * 3 * amplification_mask[..., np.newaxis]
    amplified_residual = residual * amplification_factor

    # Apply amplification only where errors exist
    result = reference + np.where(error_mask, amplified_residual, residual)

    return np.clip(result, 0, 255).astype(np.uint8)

def temporal_bleeding_effect(frame_buffer, intensity, bleed_distance=3, direction='random', rng=None):
    """
    Creates temporal bleeding with directional patterns.

    Args:
        direction: 'random', 'forward', 'backward', 'ping_pong', 'center_out'
    """
    if intensity == 0 or len(frame_buffer) < 2:
        return frame_buffer[-1] if frame_buffer else None

    rng = rng if rng is not None else np.random

    current = np.asarray(frame_buffer[-1], dtype=np.uint8)
    result = current.copy()
    height, width = current.shape[:2]

    # Mix with previous frames based on direction
    num_frames_to_mix = min(len(frame_buffer) - 1, bleed_distance)

    for i in range(1, num_frames_to_mix + 1):
        if i < len(frame_buffer):
            prev_frame = np.asarray(frame_buffer[-(i+1)], dtype=np.uint8)
            if prev_frame.shape != current.shape:
                logging.warning(
                    "Skipping temporal-bleed history at %s; master shape is %s",
                    prev_frame.shape,
                    current.shape,
                )
                continue

            # Weight decreases with temporal distance
            base_weight = intensity * (0.7 ** i)

            # Create directional mask for temporal bleeding
            if direction == 'forward':
                # Bleeding moves from left to right
                mask_strength = np.linspace(base_weight, 0, width, dtype=np.float64)
                mask = np.tile(mask_strength, (height, 1))
            elif direction == 'backward':
                # Bleeding moves from right to left
                mask_strength = np.linspace(0, base_weight, width, dtype=np.float64)
                mask = np.tile(mask_strength, (height, 1))
            elif direction == 'ping_pong':
                # Alternating direction based on frame index
                if i % 2 == 0:
                    mask_strength = np.linspace(base_weight, 0, width, dtype=np.float64)
                else:
                    mask_strength = np.linspace(0, base_weight, width, dtype=np.float64)
                mask = np.tile(mask_strength, (height, 1))
            elif direction == 'center_out':
                # Bleeding from center outward
                center_x = width // 2
                x_coords = np.arange(width)
                dist_from_center = np.abs(x_coords - center_x) / center_x
                mask_strength = base_weight * (1 - dist_from_center)
                mask = np.tile(mask_strength, (height, 1))
            else:  # 'random'
                mask = rng.random(current.shape[:2]) * base_weight

            # Apply temporal bleeding with directional mask
            mask_3d = np.expand_dims(mask, axis=2) if len(current.shape) == 3 else mask
            blend_mask = mask_3d > rng.random(current.shape)
            result[blend_mask] = prev_frame[blend_mask]

    return result

def macroblock_scrambling(current_frame, reference_frame, intensity, block_size=16, pattern='random', rng=None):
    """
    Scrambles macroblocks with specific patterns.

    Args:
        pattern: 'random', 'horizontal_shift', 'vertical_shift', 'diagonal', 'spiral', 'checkerboard'
    """
    if intensity == 0:
        return current_frame

    rng = rng or random

    current = np.asarray(current_frame, dtype=np.uint8)
    if reference_frame is not None:
        reference = np.asarray(reference_frame, dtype=np.uint8)
        if reference.shape != current.shape:
            logging.warning(
                "Discarding macroblock reference at %s; master shape is %s",
                reference.shape,
                current.shape,
            )
            reference = current
    else:
        reference = current

    height, width = current.shape[:2]
    result = current.copy()

    # Collect all block views with their positions
    blocks = []
    positions = []

    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            # Mix current and reference blocks
            if rng.random() < 0.5:
                blocks.append(current[y:y + block_size, x:x + block_size])
            else:
                blocks.append(reference[y:y + block_size, x:x + block_size])

            positions.append((y, x))

    if len(blocks) <= 1:
        return result

    # Apply scrambling pattern
    for i, (y, x) in enumerate(positions):
        if rng.random() < intensity:
            if pattern == 'horizontal_shift':
                # Shift blocks horizontally
                shift = int(intensity * 5)  # Max 5 blocks shift
                target_idx = (i + shift) % len(blocks)
            elif pattern == 'vertical_shift':
                # Shift blocks vertically
                blocks_per_row = width // block_size
                shift = int(intensity * 3) * blocks_per_row  # Max 3 rows shift
                target_idx = (i + shift) % len(blocks)
            elif pattern == 'diagonal':
                # Diagonal scrambling
                blocks_per_row = width // block_size
                row = i // blocks_per_row
                col = i % blocks_per_row
                new_row = (row + int(intensity * 3)) % (height // block_size)
                new_col = (col + int(intensity * 3)) % blocks_per_row
                target_idx = new_row * blocks_per_row + new_col
                target_idx = min(target_idx, len(blocks) - 1)
            elif pattern == 'spiral':
                # Spiral pattern scrambling
                center_idx = len(blocks) // 2
                distance = abs(i - center_idx)
                spiral_shift = int(distance * intensity * 0.5)
                target_idx = (center_idx + spiral_shift) % len(blocks)
            elif pattern == 'checkerboard':
                # Checkerboard pattern
                blocks_per_row = width // block_size
                row = i // blocks_per_row
                col = i % blocks_per_row
                if (row + col) % 2 == 0:  # Even squares
                    target_idx = rng.randint(0, len(blocks) - 1)
                else:
                    target_idx = i  # Keep odd squares in place
            else:  # 'random'
                target_idx = rng.randint(0, len(blocks) - 1)

            # Apply the scrambled block
            result[y:y + block_size, x:x + block_size] = blocks[target_idx]

    return result

def residual_injection(current_frame, intensity, noise_type='gaussian', direction='uniform', python_rng=None, numpy_rng=None):
    """
    Injects artificial residual data with directional patterns.

    Args:
        direction: 'uniform', 'horizontal_gradient', 'vertical_gradient', 'radial', 'corner_focus'
    """
    if intensity == 0:
        return current_frame

    python_rng = python_rng or random
    numpy_rng = numpy_rng if numpy_rng is not None else np.random

    current = np.asarray(current_frame)
    height, width = current.shape[:2]

    if direction == 'uniform' and noise_type == 'gaussian':
        residual_noise = numpy_rng.normal(0, 1, current.shape)
        residual_noise *= float(intensity * 50.0)
        result = current.astype(np.float32) + residual_noise
        return np.clip(result, 0, 255).astype(np.uint8)

    current_f32 = np.array(current_frame, dtype=np.float32)
    center_x, center_y = width // 2, height // 2

    # Create directional intensity mask
    if direction == 'horizontal_gradient':
        x_coords = np.arange(width)
        direction_mask = x_coords / width
        direction_mask = np.tile(direction_mask, (height, 1))
    elif direction == 'vertical_gradient':
        y_coords = np.arange(height)
        direction_mask = y_coords / height
        direction_mask = np.tile(direction_mask.reshape(-1, 1), (1, width))
    elif direction == 'radial':
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        radial_dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        direction_mask = radial_dist / max_dist
    elif direction == 'corner_focus':
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        corner_dist = np.minimum(
            np.minimum(x_coords, width - x_coords),
            np.minimum(y_coords, height - y_coords)
        )
        direction_mask = 1 - (corner_dist / np.max(corner_dist))
    else:  # 'uniform'
        direction_mask = np.ones((height, width))

    # Apply directional scaling to intensity
    scaled_intensity = intensity * direction_mask

    # Generate noise based on type
    if noise_type == 'gaussian':
        residual_noise = numpy_rng.normal(0, 1, current_f32.shape)
        residual_noise *= (scaled_intensity * 50)[..., np.newaxis]
    elif noise_type == 'block_artifacts':
        residual_noise = np.zeros_like(current_f32)
        block_size = 8
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                local_intensity = np.mean(scaled_intensity[y:y+block_size, x:x+block_size])
                if python_rng.random() < local_intensity:
                    block_noise = numpy_rng.uniform(-local_intensity * 100, local_intensity * 100)
                    residual_noise[y:y + block_size, x:x + block_size] = block_noise
    else:  # Salt and pepper noise
        residual_noise = np.zeros_like(current_f32)
        mask = numpy_rng.random(current_f32.shape) < (scaled_intensity * 0.1)[..., np.newaxis]
        residual_noise[mask] = numpy_rng.choice([-127, 127], size=np.sum(mask))

    result = current_f32 + residual_noise
    return np.clip(result, 0, 255).astype(np.uint8)

def reference_frame_corruption(reference_frame, intensity, corruption_type='random_blocks', pattern='scattered'):
    """
    Corrupts the reference frame with specific patterns.

    Args:
        pattern: 'scattered', 'horizontal_bands', 'vertical_bands', 'diagonal_sweep', 'center_corruption'
    """
    if intensity == 0 or reference_frame is None:
        return reference_frame

    reference = np.array(reference_frame, dtype=np.float32)
    height, width = reference.shape[:2]
    center_x, center_y = width // 2, height // 2

    if corruption_type == 'random_blocks':
        block_size = 16

        if pattern == 'horizontal_bands':
            # Corrupt in horizontal bands
            band_height = height // 5
            for band in range(5):
                if random.random() < intensity:
                    y_start = band * band_height
                    y_end = min((band + 1) * band_height, height)
                    for y in range(y_start, y_end - block_size, block_size):
                        for x in range(0, width - block_size, block_size):
                            if random.random() < 0.5:
                                reference[y:y + block_size, x:x + block_size] = np.random.randint(0, 256,
                                                                                                (block_size, block_size, reference.shape[2]))

        elif pattern == 'vertical_bands':
            # Corrupt in vertical bands
            band_width = width // 5
            for band in range(5):
                if random.random() < intensity:
                    x_start = band * band_width
                    x_end = min((band + 1) * band_width, width)
                    for y in range(0, height - block_size, block_size):
                        for x in range(x_start, x_end - block_size, block_size):
                            if random.random() < 0.5:
                                reference[y:y + block_size, x:x + block_size] = np.random.randint(0, 256,
                                                                                                (block_size, block_size, reference.shape[2]))

        elif pattern == 'diagonal_sweep':
            # Corrupt along diagonal
            for y in range(0, height - block_size, block_size):
                for x in range(0, width - block_size, block_size):
                    # Distance from main diagonal
                    diag_dist = abs((x / width) - (y / height))
                    corruption_prob = intensity * (1 - diag_dist * 2)  # Higher prob near diagonal
                    if corruption_prob > 0 and random.random() < corruption_prob:
                        reference[y:y + block_size, x:x + block_size] = np.random.randint(0, 256,
                                                                                        (block_size, block_size, reference.shape[2]))

        elif pattern == 'center_corruption':
            # Corrupt more near center, less at edges
            for y in range(0, height - block_size, block_size):
                for x in range(0, width - block_size, block_size):
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    center_factor = 1 - (dist_from_center / max_dist)
                    corruption_prob = intensity * center_factor * 0.5
                    if random.random() < corruption_prob:
                        reference[y:y + block_size, x:x + block_size] = np.random.randint(0, 256,
                                                                                        (block_size, block_size, reference.shape[2]))

        else:  # 'scattered'
            # Original scattered corruption
            for y in range(0, height - block_size, block_size):
                for x in range(0, width - block_size, block_size):
                    if random.random() < intensity * 0.3:
                        if random.random() < 0.5:
                            reference[y:y + block_size, x:x + block_size] = np.random.randint(0, 256,
                                                                                            (block_size, block_size, reference.shape[2]))
                        else:
                            # Duplicate a nearby block
                            src_y = max(0, min(height - block_size, y + random.randint(-block_size*2, block_size*2)))
                            src_x = max(0, min(width - block_size, x + random.randint(-block_size*2, block_size*2)))
                            reference[y:y + block_size, x:x + block_size] = reference[src_y:src_y + block_size, src_x:src_x + block_size]

    elif corruption_type == 'temporal_artifacts':
        # Directional temporal artifacts
        noise_strength = intensity * 100
        if pattern == 'horizontal_sweep':
            # Noise sweeps from left to right
            x_coords = np.arange(width)
            noise_mask = (x_coords / width)[np.newaxis, :, np.newaxis]
            temporal_noise = np.random.uniform(-noise_strength, noise_strength, reference.shape) * noise_mask
        elif pattern == 'vertical_sweep':
            # Noise sweeps from top to bottom
            y_coords = np.arange(height)
            noise_mask = (y_coords / height)[:, np.newaxis, np.newaxis]
            temporal_noise = np.random.uniform(-noise_strength, noise_strength, reference.shape) * noise_mask
        else:
            # Uniform temporal noise
            temporal_noise = np.random.uniform(-noise_strength, noise_strength, reference.shape)

        reference += temporal_noise

    return np.clip(reference, 0, 255).astype(np.uint8)

def apply_pframe_effects(current_frame, reference_frame, frame_buffer, params):
    """
    Apply P-frame specific effects. Effects are blended over the incoming frame
    using their intensity, to avoid wiping earlier pipeline results.
    """
    # Helper: ensure 3-channel uint8 array
    def to_rgb_np(img_or_arr):
        arr = np.array(img_or_arr)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        if arr.shape[2] == 4:
            arr = gpo.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr.astype(np.uint8)

    # Helper: alpha blend two arrays (same shape, uint8)
    def blend(base, overlay, alpha):
        a = float(max(0.0, min(1.0, alpha)))
        if a == 0.0:
            return base
        return np.clip(base.astype(np.float32)*(1.0-a) + overlay.astype(np.float32)*a, 0, 255).astype(np.uint8)

    result_frame = to_rgb_np(current_frame)
    ref_np = to_rgb_np(reference_frame) if reference_frame is not None else None
    updated_reference = reference_frame

    # Normalize/cast some params
    try:
        mbs = params.get('macroblock_size', 16)
        macroblock_size = int(mbs) if not isinstance(mbs, int) else mbs
    except Exception:
        macroblock_size = 16

    # Motion vector corruption with direction control (blended)
    mv_int = float(params.get('motion_vector_corruption', 0) or 0)
    if mv_int > 0:
        out = corrupt_motion_vectors(result_frame, ref_np, mv_int, macroblock_size, params.get('motion_vector_direction', 'random'))
        result_frame = blend(result_frame, out, mv_int)

    # Prediction error amplification with direction (blended)
    pea_int = float(params.get('prediction_error_amplify', 0) or 0)
    if pea_int > 0:
        out = amplify_prediction_errors(result_frame, ref_np, pea_int, params.get('error_threshold', 30), params.get('prediction_error_direction', 'uniform'))
        result_frame = blend(result_frame, out, pea_int)

    # Temporal bleeding with direction (blended)
    tb_int = float(params.get('temporal_bleeding', 0) or 0)
    if tb_int > 0 and len(frame_buffer) > 1:
        # Build buffer of RGB arrays
        fb_arrays = []
        for f in frame_buffer:
            try:
                fb_arrays.append(to_rgb_np(f))
            except Exception:
                pass
        fb_arrays.append(result_frame.copy())
        out = temporal_bleeding_effect(fb_arrays, tb_int, params.get('bleed_distance', 3), params.get('temporal_bleeding_direction', 'random'))
        result_frame = blend(result_frame, out, tb_int)

    # Macroblock scrambling with patterns (blended)
    mbs_int = float(params.get('macroblock_scramble', 0) or 0)
    if mbs_int > 0:
        out = macroblock_scrambling(result_frame, ref_np, mbs_int, macroblock_size, params.get('scrambling_pattern', 'random'))
        result_frame = blend(result_frame, out, mbs_int)

    # Residual injection with direction (blended)
    ri_int = float(params.get('residual_injection', 0) or 0)
    if ri_int > 0:
        out = residual_injection(result_frame, ri_int, params.get('residual_noise_type', 'gaussian'), params.get('residual_direction', 'uniform'))
        result_frame = blend(result_frame, out, ri_int)

    # Reference frame corruption with patterns (updates reference only)
    rc_int = float(params.get('reference_corruption', 0) or 0)
    if rc_int > 0 and ref_np is not None:
        updated_ref_np = reference_frame_corruption(ref_np, rc_int, params.get('reference_corruption_type', 'random_blocks'), params.get('reference_corruption_pattern', 'scattered'))
        updated_reference = Image.fromarray(updated_ref_np)

    return Image.fromarray(result_frame), updated_reference


def apply_glitch_effect(img, params, previous_frame, feedback_image, trail_image, frame_buffer):
    """
    Apply glitch effects to an image.
    OPTIMIZED: Uses numpy arrays throughout, no PIL conversions in hot loop.
    
    Args:
        img: Input image as PIL.Image OR numpy array (H,W,3) uint8
        params: Effect parameters dict
        previous_frame: numpy array (H,W,3) uint8 or None
        feedback_image: numpy array (H,W,3) uint8 or None
        trail_image: numpy array (H,W,3) uint8 or None
        frame_buffer: List of numpy arrays
        
    Returns:
        (output_array, next_reference, new_feedback, new_trail)
        All as numpy arrays (H,W,3) uint8
    """
    from performance_profiler import get_profiler
    import time
    
    profiler = get_profiler()
    frame_start = time.perf_counter()
    
    try:
        # Guard against missing keys causing KeyError anywhere below
        params = defaultdict(lambda: 0, params or {})
        
        # Convert input to numpy if needed (only happens at boundary)
        with profiler.time_stage("input_conversion"):
            if isinstance(img, Image.Image):
                img_array = np.array(img.convert("RGB"))
            else:
                img_array = img  # Already numpy
        
        rows, cols = img_array.shape[:2]
        
        # --- DATABENDING ---
        with profiler.time_stage("databending"):
            if params.get('databend_intensity', 0) > 0:
                # Databending still needs PIL temporarily
                img_pil_temp = Image.fromarray(img_array)
                img_array = apply_databending_effect(
                    img_pil_temp,
                    params.get('databend_intensity', 0),
                    params.get('databend_mode', 'random'),
                    params.get('databend_layers', 1),
                    params.get('databend_format', 'jpeg'),
                    params.get('databend_quality', 75)
                )

        # --- NEW EFFECTS: Kaleidoscope / Noise Warp / Watercolor Posterize / CRT / VHS ---
        try:
            seg = int(params.get('kaleido_segments', 0))
        except Exception:
            seg = 0
        if seg >= 2 or float(params.get('kaleido_angle', 0.0)) != 0.0:
            img_array = kaleidoscope_np(img_array,
                                        segments=max(2, seg),
                                        angle_offset=float(params.get('kaleido_angle', 0.0)),
                                        radius_factor=float(params.get('kaleido_radius', 1.0)))

        nw_int = float(params.get('noise_warp_intensity', 0.0))
        if nw_int > 0.0:
            img_array = noise_field_warp_np(img_array,
                                            intensity=nw_int,
                                            scale=float(params.get('noise_warp_scale', 64.0)),
                                            t=float(params.get('noise_warp_time', 0.0)))

        levels = int(params.get('watercolor_levels', 0))
        if levels >= 2 or float(params.get('watercolor_sigma', 0.0)) > 0.0:
            img_array = bilateral_posterize_np(img_array,
                                               levels=max(2, int(params.get('watercolor_levels', 6))),
                                               bilateral_sigma=float(params.get('watercolor_sigma', 12.0)),
                                               passes=int(params.get('watercolor_passes', 2)))

        sl = float(params.get('crt_scanline', 0.0))
        if sl > 0.0 or params.get('crt_mask', 'none') != 'none' or float(params.get('crt_curvature', 0.0)) != 0.0 or float(params.get('crt_bloom', 0.0)) > 0.0:
            img_array = crt_emulation_np(img_array,
                                         scanline=sl,
                                         mask_type=str(params.get('crt_mask', 'rgb')),
                                         curvature=float(params.get('crt_curvature', 0.0)),
                                         bloom=float(params.get('crt_bloom', 0.0)))

        vhs = float(params.get('vhs_damage', 0.0))
        if vhs > 0.0:
            img_array = vhs_damage_np(img_array,
                                      damage=vhs,
                                      band_min=int(params.get('vhs_band_min', 4)),
                                      band_max=int(params.get('vhs_band_max', 24)),
                                      jitter=float(params.get('vhs_jitter', 0.2)),
                                      noise=float(params.get('vhs_noise', 0.2)))

        # --- GEOMETRIC AND FULL-FRAME EFFECTS ---
        with profiler.time_stage("rotation"):
            if params.get("rotation_speed", 0) != 0:
                params["rotation_angle"] = (params.get("rotation_angle", 0) + params.get("rotation_speed", 0)) % 360.0
            if params.get("rotation_angle", 0) != 0:
                angle = params["rotation_angle"]
                img_array = rotate_in_frame_np(img_array, angle, overscan=0.18)
                rows, cols = img_array.shape[:2]

        # Kaleidoscope (optimized for procedural generators)
        with profiler.time_stage("kaleidoscope"):
            try:
                seg_raw = params.get("kaleidoscope_segments", 1)
                seg = max(1, int(seg_raw)) if isinstance(seg_raw, (int, float, np.integer)) else 1
            except Exception:
                seg = 1
            if seg > 1:
                speed = float(params.get('kaleidoscope_speed', 0.2) or 0.0)
                angle_offset = float(params.get('noise_warp_time', 0.0)) * speed
                img_array = kaleidoscope_np(
                    img_array,
                    segments=seg,
                    angle_offset=angle_offset,
                    radius_factor=float(params.get("kaleidoscope_radius", 1.0) or 1.0),
                )
                rows, cols = img_array.shape[:2]

        # ... (Other geometric effects like glitch_amount, unfolding, wave_distortion)
        if params.get("glitch_amount_x", 0) > 0 or params.get("glitch_amount_y", 0) > 0:
            map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)
            ts = params.get("time_scale", 1.0)
            time_multiplier = 2.0**ts
            base_speed = 1.0
            # Use noise_warp_time for proper animation in batch rendering
            t_off = float(params.get('noise_warp_time', 0.0)) * base_speed * time_multiplier
            if params.get("glitch_freq_y", 0) * params.get("uv_size", 1.0) != 0:
                glitch_x = params.get("glitch_amount_x", 0) * np.sin(2.0 * np.pi * (map_y / (params.get("glitch_freq_y", 1.0) * params.get("uv_size", 1.0))) + t_off)
            else:
                glitch_x = np.zeros_like(map_x)
            if params.get("glitch_freq_x", 0) * params.get("uv_size", 1.0) != 0:
                glitch_y = params.get("glitch_amount_y", 0) * np.sin(2.0 * np.pi * (map_x / (params.get("glitch_freq_x", 1.0) * params.get("uv_size", 1.0))) + t_off / 2.0)
            else:
                glitch_y = np.zeros_like(map_y)
            map_x += glitch_x
            map_y += glitch_y
            img_array = gpo.remap(img_array, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # --- PIXEL-LEVEL AND BLOCK-BASED EFFECTS ---
        with profiler.time_stage("block_effects"):
            img_array_float = img_array.astype(np.float32)
            img_array_float = block_displacement_np(img_array_float, params.get('block_size', 16), params.get('block_displacement', 0), params.get('displacement_direction', 'random'))
            img_array_float = block_duplication_np(img_array_float, params.get('block_size', 16), params.get('block_duplication', 0), params.get('duplication_pattern', 'random'))
            img_array_float = block_corruption_np(img_array_float, params.get('block_size', 16), params.get('block_corruption', 0), params.get('corruption_type', 'noise'))
            img_array_float = block_sorting_np(img_array_float, params.get('block_size', 16), params.get('block_sorting', 0), params.get('sort_method', 'brightness'))
            img_array_float = motion_vectors_np(img_array_float, params.get('motion_vectors', 0), params.get('vector_length', 20))
            img_array_float = bitplane_corruption_np(img_array_float, params.get('bitplane_corruption', 0), params.get('bitplane', 1))
            img_array_float = dct_glitch_np(img_array_float, params.get('dct_glitch_intensity', 0))
            # Slit scan with state persistence
            # CRITICAL: Convert to uint8 for slit scan to avoid float32 accumulation issues
            slit_scan_intensity = params.get('slit_scan_intensity', 0)
            if slit_scan_intensity > 0:
                if not hasattr(frame_state, 'slit_scan_state'):
                    frame_state.slit_scan_state = {}
                
                # Convert to uint8, apply slit scan, convert back to float32
                img_array_uint8 = np.clip(img_array_float, 0, 255).astype(np.uint8)
                img_array_uint8 = slit_scan_smear_np(
                    img_array_uint8,
                    slit_scan_intensity,
                    str(params.get('slit_scan_direction', 'right_to_left')),
                    int(params.get('slit_scan_size', 1) or 1),
                    float(params.get('slit_scan_pos', 0.5) or 0.5),
                    external_state=frame_state.slit_scan_state
                )
                img_array_float = img_array_uint8.astype(np.float32)
        
        # New effects - WRAP IN PROFILING + SKIP IF DISABLED
        with profiler.time_stage("additional_effects"):
            # Only process if ANY effect is enabled (huge performance win)
            additional_effects_active = any([
                params.get('wobble_intensity', 0.0) > 0,
                params.get('scan_shuffle', 0.0) > 0,
                params.get('chroma_intensity', 0.0) > 0,
                params.get('fft_phase', 0.0) > 0,
                params.get('channel_flow', 0.0) > 0,
                params.get('mesh_tear', 0.0) > 0,
                params.get('palette_levels', 6) != 6,
                params.get('brightness', 0.0) != 0,
                params.get('contrast', 0.0) != 0,
                params.get('gamma', 1.0) != 1.0,
                params.get('hue_shift', 0.0) != 0,
                params.get('saturation', 1.0) != 1.0,
                params.get('vignette_intensity', 0.0) > 0,
                params.get('blur_radius', 0.0) > 0,
                params.get('sharpen_amount', 0.0) > 0,
                params.get('bloom_strength', 0.0) > 0,
                params.get('edge_mix', 0.0) > 0,
                params.get('emboss_strength', 0.0) > 0,
                params.get('barrel_amount', 0.0) != 0,
                params.get('swirl_amount', 0.0) != 0,
                params.get('ripple_amp', 0.0) > 0,
                params.get('halftone_cell', 0) not in (None, 0),
                params.get('vhs_jitter', 0.0) > 0,
            ])
            
            if additional_effects_active:
                rng = current_numpy_rng()
                # Use noise_warp_time for proper animation in batch rendering
                t = float(params.get('noise_warp_time', 0.0))

                img_array_float = rolling_shutter_wobble_np(img_array_float, params.get('wobble_intensity', 0.0),
                                              params.get('wobble_freq', 3.0),
                                              phase=t*params.get('wobble_speed', 0.5))

                img_array_float = scanline_shuffle_np(img_array_float, params.get('scan_shuffle', 0.0),
                                        params.get('scan_span', 32),
                                        params.get('scan_prob', 0.12), rng)

                img_array_float = chroma_leak_np(img_array_float, params.get('chroma_intensity', 0.0),
                                   params.get('chroma_mode', '420'),
                                   params.get('chroma_drift', 2), rng)

                img_array_float = fft_phase_scramble_np(img_array_float, params.get('fft_phase', 0.0),
                                          params.get('fft_bands', 3), rng)

                img_array_float = channel_flow_offset_np(img_array_float, params.get('channel_flow', 0.0),
                                           params.get('channel_flow_max', 8), rng)

                img_array_float = mesh_tear_np(img_array_float, params.get('mesh_tear', 0.0),
                                         params.get('mesh_grid', 6), rng)

                img_array_float = palette_crunch_np(img_array_float, params.get('palette_levels', 6),
                                      params.get('palette_dither', True), rng)

                # ---- Newly added conventional adjustments ----
                img_array_float = apply_brightness_contrast(
                    img_array_float,
                    params.get('brightness', 0.0),
                    params.get('contrast', 0.0)
                )

                img_array_float = apply_gamma(img_array_float, params.get('gamma', 1.0))

                img_array_float = apply_hsv_adjust(
                    img_array_float,
                    params.get('hue_shift', 0.0),
                    params.get('saturation', 1.0)
                )

                img_array_float = apply_vignette(
                    img_array_float,
                    params.get('vignette_intensity', 0.0),
                    params.get('vignette_radius', 0.75)
                )

                img_array_float = apply_blur(img_array_float, params.get('blur_radius', 0.0), use_cuda=gpo.get_use_cuda())
                img_array_float = apply_sharpen(img_array_float, params.get('sharpen_amount', 0.0), radius=2)

                img_array_float = apply_bloom(img_array_float, params.get('bloom_strength', 0.0), params.get('bloom_threshold', 200))

                img_array_float = apply_edges_emboss(img_array_float, params.get('edge_mix', 0.0), params.get('emboss_strength', 0.0))

                # ---- Newly added geometric/obscure effects ----
                img_array_float = barrel_pincushion_np(img_array_float, params.get('barrel_amount', 0.0))
                img_array_float = swirl_np(img_array_float, params.get('swirl_amount', 0.0), params.get('swirl_radius_factor', 0.75))
                img_array_float = ripple_np(img_array_float, params.get('ripple_amp', 0.0), params.get('ripple_freq', 3.0), t)
                if params.get('halftone_cell', 0) not in (None, 0):
                    img_array_float = halftone_np(img_array_float, params.get('halftone_cell', 8), params.get('halftone_contrast', 1.0))
                img_array_float = vhs_jitter_np(img_array_float, params.get('vhs_jitter', 0.0), params.get('vhs_max_shift', 8))

                img_array = np.clip(img_array_float, 0, 255).astype(np.uint8)
            else:
                # No additional effects active - keep img_array as-is
                pass

        # ... (Other pixel effects like pixel_sorting, pixelation, etc.)
        with profiler.time_stage("pixel_sorting"):
            if params.get("pixel_sorting_intensity", 0) > 0:
                direction = params.get("pixel_sorting_direction", "Horizontal")
                for idx in range(rows if direction == "Horizontal" else cols):
                    if np.random.rand() < params.get("sort_chance", 0) / 100.0:
                        if direction == "Horizontal":
                            row = img_array[idx, :, :]; bright = np.sum(row, axis=1); sort_idx = np.argsort(bright); row_sorted = row[sort_idx]
                            img_array[idx, :, :] = gpo.addWeighted(row, 1.0 - params.get("pixel_sorting_intensity", 0), row_sorted, params.get("pixel_sorting_intensity", 0), 0)
                        else:
                            col_arr = img_array[:, idx, :]; bright = np.sum(col_arr, axis=1); sort_idx = np.argsort(bright); col_sorted = col_arr[sort_idx]
                            img_array[:, idx, :] = gpo.addWeighted(col_arr, 1.0 - params.get("pixel_sorting_intensity", 0), col_sorted, params.get("pixel_sorting_intensity", 0), 0)

        # --- FRAME-LEVEL AND FEEDBACK EFFECTS ---
        # These effects need to be applied before P-frame logic
        
        # --- FEEDBACK AND TRAILING (OPTIMIZED - NO PIL CONVERSIONS) ---
        with profiler.time_stage("feedback_trail"):
            # Store clean version before feedback (numpy copy, not PIL)
            new_trail_image = img_array.copy()
            
            # Feedback blend (all numpy, no shape checks - enforced at boundary)
            if params.get("feedback_intensity", 0) > 0 and feedback_image is not None:
                intensity = float(params.get("feedback_intensity", 0))
                # ASSERT shape matches - should be enforced at worker init
                if feedback_image.shape == img_array.shape:
                    # In-place blend for speed (CUDA-accelerated when enabled)
                    gpo.addWeighted(img_array, 1.0 - intensity, 
                                  feedback_image, intensity, 
                                  0, dst=img_array)
                else:
                    logging.warning(
                        "Discarding feedback history at %s; master shape is %s",
                        feedback_image.shape,
                        img_array.shape,
                    )
            
            # Store feedback for next frame (numpy copy)
            new_feedback_image = img_array.copy()
            
            # Trailing blend (all numpy)
            if params.get("trailing_intensity", 0) > 0 and trail_image is not None:
                intensity = float(params.get("trailing_intensity", 0))
                if trail_image.shape == img_array.shape:
                    gpo.addWeighted(img_array, 1.0 - intensity, 
                                  trail_image, intensity, 
                                  0, dst=img_array)
                else:
                    logging.warning(
                        "Discarding trail history at %s; master shape is %s",
                        trail_image.shape,
                        img_array.shape,
                    )

        # Output is numpy array (H, W, 3) uint8
        glitched_img_array = img_array

        # --- P-FRAME AND FINAL STATE MANAGEMENT ---
        with profiler.time_stage("pframe_effects"):
            # Reference for next frame (numpy copy, not PIL)
            next_reference_frame = glitched_img_array.copy()

            pframe_active = any(params.get(key, 0) > 0 for key in [
                'motion_vector_corruption', 'prediction_error_amplify', 
                'temporal_bleeding', 'macroblock_scramble', 
                'residual_injection', 'reference_corruption'
            ])

            if pframe_active:
                # P-frame effects need PIL for now (legacy compatibility)
                # TODO: Refactor apply_pframe_effects to use numpy
                glitched_pil_temp = Image.fromarray(glitched_img_array)
                prev_pil_temp = Image.fromarray(previous_frame) if previous_frame is not None else None
                fb_pil_temp = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frame_buffer]
                
                glitched_pil_temp, next_ref_pil = apply_pframe_effects(
                    glitched_pil_temp, 
                    prev_pil_temp,
                    fb_pil_temp, 
                    params
                )
                
                # Convert back to numpy
                glitched_img_array = np.array(glitched_pil_temp.convert("RGB"))
                next_reference_frame = np.array(next_ref_pil.convert("RGB"))

            # Frame drop happens *after* all other effects
            if params.get("frame_drop_chance", 0) > 0 and np.random.rand() < params.get("frame_drop_chance", 0) / 100.0:
                if previous_frame is not None:
                    # If we drop, output is previous frame
                    glitched_img_array = previous_frame.copy()
                    next_reference_frame = previous_frame.copy()

        # Update frame buffer (all numpy arrays)
        with profiler.time_stage("frame_buffer"):
            frame_buffer.append(glitched_img_array.copy())
            buffer_size = params.get('buffer_size', 10)
            while len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)

            if params.get('row_stutter', 0.0) > 0 and frame_buffer:
                # row_temporal_stutter expects PIL images for now
                # TODO: Refactor to use numpy
                fb_pil = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in frame_buffer]
                rng = current_numpy_rng()
                stuttered = row_temporal_stutter(fb_pil, params['row_stutter'],
                                        params.get('row_stutter_max_delay', 4), rng)
                glitched_img_array = stuttered

        # Record total frame time
        frame_elapsed = time.perf_counter() - frame_start
        profiler.record_frame_time(frame_elapsed)
        
        # Return all as numpy arrays (H, W, 3) uint8
        return glitched_img_array, next_reference_frame, new_feedback_image, new_trail_image

    except Exception as e:
        logging.error(f"Error in apply_glitch_effect: {e}", exc_info=True)
        # Return original state on error (convert to numpy if needed)
        img_array = np.array(img.convert("RGB")) if isinstance(img, Image.Image) else img
        prev_array = np.array(previous_frame.convert("RGB")) if isinstance(previous_frame, Image.Image) else previous_frame
        fb_array = np.array(feedback_image.convert("RGB")) if isinstance(feedback_image, Image.Image) else feedback_image
        trail_array = np.array(trail_image.convert("RGB")) if isinstance(trail_image, Image.Image) else trail_image
        return img_array, prev_array, fb_array, trail_array

def apply_kaleidoscope_effect(img_array, segments):
    try:
        h, w = img_array.shape[:2]; center = (w // 2, h // 2); angle = 360 / segments
        result = np.zeros_like(img_array); wedge = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([[center[0], center[1]], [w, center[1]], [w, 0], [center[0], 0]], np.int32)
        cv2.fillConvexPoly(wedge, pts, 255)
        for i in range(segments):
            M = cv2.getRotationMatrix2D(center, angle * i, 1.0)
            rotated = gpo.warpAffine(img_array, M, (w, h))
            M_mask = cv2.getRotationMatrix2D(center, angle * i, 1.0)
            rot_wedge = gpo.warpAffine(wedge, M_mask, (w, h))
            seg = rotated if i % 2 == 0 else gpo.flip(rotated, 1)
            mask_idx = (rot_wedge == 255)
            result[mask_idx] = seg[mask_idx]
        return result
    except Exception as e:
        print(f"Error in apply_kaleidoscope_effect: {e}")
        return img_array

# --- Implementations for new effects ---

def vectorscope_xy_np(arr, intensity=0.5, freq_x=1.0, freq_y=2.0, 
                     phase=0.0, circular=0.3, t=0.0):
    """
    Vectorscope/XY oscilloscope effect with Lissajous patterns.
    Creates circular distortions driven by image brightness data.
    """
    h, w = arr.shape[:2]
    
    # Convert to grayscale for intensity mapping
    if len(arr.shape) == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    
    # Normalize brightness
    gray_norm = gray / 255.0
    
    # Create coordinate grids
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    x_norm = (x_coords / w - 0.5) * 2  # -1 to 1
    y_norm = (y_coords / h - 0.5) * 2  # -1 to 1
    
    # Lissajous pattern (XY oscilloscope mode)
    # x(t) = sin(freq_x * t + phase)
    # y(t) = sin(freq_y * t)
    lissajous_x = np.sin(freq_x * t + phase + x_norm * np.pi)
    lissajous_y = np.sin(freq_y * t + y_norm * np.pi)
    
    # Circular distortion based on brightness
    radius = np.sqrt(x_norm**2 + y_norm**2)
    angle = np.arctan2(y_norm, x_norm)
    
    # Modulate distortion by image brightness
    distortion_amount = gray_norm * intensity * circular
    
    # Apply Lissajous + circular distortion
    dx = (lissajous_x * intensity * 20) + (np.cos(angle) * distortion_amount * radius * 30)
    dy = (lissajous_y * intensity * 20) + (np.sin(angle) * distortion_amount * radius * 30)
    
    # Create displacement map
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)
    
    # Apply remap
    result = gpo.remap(arr, map_x, map_y)
    
    return result

def apply_jpeg_compression(arr, **kwargs):
    """Applies JPEG compression artifacts by encoding and decoding the image."""
    quality = int(kwargs.get('jpeg_quality', 75))
    passes = int(kwargs.get('jpeg_passes', 1))
    if quality >= 100 and passes == 1:
        return arr
    
    img = arr
    for _ in range(passes):
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            if result:
                img = cv2.imdecode(encimg, 1)
            else:
                return arr # Return original if encoding fails
        except Exception as e:
            logging.error(f"JPEG compression failed: {e}")
            return arr
            
    # DCT/Quantization corruption can be simulated with noise on block level
    dct_corr = float(kwargs.get('dct_corruption', 0.0))
    if dct_corr > 0:
        h, w = img.shape[:2]
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if random.random() < dct_corr:
                    noise = np.random.randint(-10, 10, (8, 8, 3))
                    block = img[y:y+8, x:x+8].astype(np.int16)
                    block += noise
                    img[y:y+8, x:x+8] = np.clip(block, 0, 255).astype(np.uint8)
    return img

def apply_webp_corruption(arr, **kwargs):
    """Applies WebP compression artifacts."""
    quality = int(kwargs.get('webp_quality', 80))
    if quality >= 100:
        return arr
    try:
        pil_img = Image.fromarray(arr)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='WEBP', quality=quality)
        buffer.seek(0)
        
        # Optional: corrupt the lossless stream if requested
        lossless_corr = float(kwargs.get('lossless_corruption', 0.0))
        if lossless_corr > 0:
            data = bytearray(buffer.read())
            header = 30 # Approximate header size for webp
            for i in range(int(len(data) * lossless_corr)):
                idx = random.randint(header, len(data)-1)
                data[idx] = random.randint(0, 255)
            buffer = io.BytesIO(data)

        return np.array(Image.open(buffer))
    except Exception as e:
        logging.error(f"WebP corruption failed: {e}")
        return arr

def apply_h264_artifacts(arr, **kwargs):
    """Simulates H.264 video compression artifacts like macroblocking."""
    p_corr = float(kwargs.get('pframe_corruption', 0.0))
    if p_corr <= 0:
        return arr
    
    h, w = arr.shape[:2]
    out = arr.copy()
    block_size = 16 # Common macroblock size
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if random.random() < p_corr:
                # Replace block with a slightly offset version of itself or a nearby block
                dx = random.randint(-block_size//2, block_size//2)
                dy = random.randint(-block_size//2, block_size//2)
                nx, ny = np.clip(x+dx, 0, w-block_size), np.clip(y+dy, 0, h-block_size)
                out[y:y+block_size, x:x+block_size] = arr[ny:ny+block_size, nx:nx+block_size]
    return out

def apply_temporal_slicing(arr, **kwargs):
    """Spatially slices the image and shifts the slices, simulating a temporal effect."""
    slice_width = int(kwargs.get('slice_width', 8))
    direction = kwargs.get('slice_direction', 'vertical')
    time_offset = int(kwargs.get('time_offset_range', 5)) # Use as spatial offset
    
    if slice_width <= 0:
        return arr
        
    h, w = arr.shape[:2]
    out = arr.copy()
    
    python_rng = kwargs.get('_preview_python_rng') or random
    preview_numpy_rng = kwargs.get('_preview_numpy_rng')

    if direction == 'vertical':
        for x in range(0, w, slice_width):
            offset = python_rng.randint(-time_offset, time_offset)
            col = out[:, x:x+slice_width]
            out[:, x:x+slice_width] = np.roll(col, offset, axis=0)
    elif direction == 'horizontal':
        for y in range(0, h, slice_width):
            offset = python_rng.randint(-time_offset, time_offset)
            row = out[y:y+slice_width, :]
            out[y:y+slice_width, :] = np.roll(row, offset, axis=1)
    elif direction == 'diagonal':
        import cv2
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        band_idx = (map_x + map_y) // slice_width
        max_band = (w + h) // slice_width + 1
        if preview_numpy_rng is None:
            offsets = np.random.randint(-time_offset, time_offset + 1, size=int(max_band))
        else:
            offsets = preview_numpy_rng.integers(
                -time_offset, time_offset + 1, size=int(max_band)
            )
        shift_vals = offsets[band_idx.astype(int)]
        
        # Shift diagonally (both x and y)
        map_x = (map_x + shift_vals).astype(np.float32)
        map_y = (map_y + shift_vals).astype(np.float32)
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    elif direction == 'radial':
        import cv2
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        cx, cy = w / 2, h / 2
        dist = np.sqrt((map_x - cx)**2 + (map_y - cy)**2)
        band_idx = (dist // slice_width).astype(int)
        max_band = int(np.max(band_idx)) + 1
        if preview_numpy_rng is None:
            offsets = np.random.randint(-time_offset, time_offset + 1, size=max_band)
        else:
            offsets = preview_numpy_rng.integers(
                -time_offset, time_offset + 1, size=max_band
            )
        shift_vals = offsets[band_idx]
        
        angle = np.arctan2(map_y - cy, map_x - cx)
        map_x = (map_x + shift_vals * np.cos(angle)).astype(np.float32)
        map_y = (map_y + shift_vals * np.sin(angle)).astype(np.float32)
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        
    return out

def apply_analog_tv(arr, **kwargs):
    """Simulates various analog TV signal defects."""
    noise = float(kwargs.get('signal_noise', 0.0))
    vsync = float(kwargs.get('vsync_corruption', 0.0))
    hsync = float(kwargs.get('hsync_corruption', 0.0))
    
    out = arr.astype(np.float32)
    h, w = out.shape[:2]
    
    if noise > 0:
        noise_arr = np.random.normal(0, 25 * noise, (h, w, 3))
        out += noise_arr
        
    if vsync > 0 and random.random() < vsync:
        shift = int(h * (random.random() * 0.2 + 0.05)) # 5-25% vertical roll
        out = np.roll(out, shift, axis=0)
        
    if hsync > 0:
        for y in range(h):
            if random.random() < hsync:
                shift = int(w * (random.random() * 0.1))
                out[y,:] = np.roll(out[y,:], shift, axis=0)
                
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_tape_dropout(arr, **kwargs):
    """Simulates magnetic tape dropouts as black or noisy horizontal lines."""
    freq = float(kwargs.get('dropout_frequency', 0.0))
    if freq <= 0:
        return arr
        
    out = arr.copy()
    h, w = out.shape[:2]
    num_dropouts = int(h * freq * 0.2)
    
    for _ in range(num_dropouts):
        y = random.randint(0, h-1)
        h_line = random.randint(kwargs.get('dropout_size_min', 1), kwargs.get('dropout_size_max', 10))
        y_end = min(h, y + h_line)
        
        # Make it a noisy dropout instead of pure black
        noise = np.random.randint(0, 50, (y_end-y, w, 3))
        out[y:y_end, :] = out[y:y_end, :] * 0.2 + noise
        
    return out

def apply_digital_clipping(arr, **kwargs):
    """Clips color channels to a threshold, simulating digital signal clipping."""
    thresh = int(kwargs.get('clip_threshold', 235))
    color_str = kwargs.get('clip_color', 'white')
    
    if color_str == 'white':
        clip_val = 255
    elif color_str == 'black':
        clip_val = 0
    else:
        clip_val = random.randint(0, 255)

    # Create a mask where any channel is over the threshold
    mask = np.any(arr > thresh, axis=-1)
    
    out = arr.copy()
    out[mask] = clip_val
    return out

def apply_fractal_noise(arr, **kwargs):
    """Displaces pixels using fractal noise, similar to noise_field_warp_np."""
    strength = float(kwargs.get('displacement_strength', 0.0))
    scale = float(kwargs.get('fractal_scale', 20.0))
    if strength <= 0:
        return arr
    # Use t parameter if provided, otherwise default to 0
    t_val = float(kwargs.get('t', 0.0))
    return noise_field_warp_np(arr, intensity=strength/50.0, scale=scale, t=t_val)

def apply_colorspace_corruption(arr, **kwargs):
    """Corrupts image by converting between color spaces and manipulating channels."""
    # This is a simplified version. Real color space conversion is complex.
    # We'll simulate it by swapping and scaling channels in HSV space.
    swap = kwargs.get('channel_swap', False)
    quantize = int(kwargs.get('quantization_levels', 256))
    
    hsv = gpo.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    if swap:
        # Swap H and S channels
        hsv[:,:,[0,1]] = hsv[:,:,[1,0]]
        
    if quantize < 256:
        step = 255 / (quantize -1)
        hsv = np.round(hsv / step) * step
        
    return gpo.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_frame_interpolation_glitch(arr, **kwargs):
    """Simulates interpolation errors by blending shifted versions of the image."""
    strength = float(kwargs.get('interp_corruption', 0.0))
    if strength <= 0: return arr
    h, w = arr.shape[:2]
    dx, dy = int(w * strength * 0.05), int(h * strength * 0.05)
    M = np.float32([[1, 0, random.randint(-dx, dx)], [0, 1, random.randint(-dy, dy)]])
    shifted = gpo.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return gpo.addWeighted(arr, 0.5, shifted, 0.5, 0)

def apply_time_displacement(arr, **kwargs):
    """Alias for Fractal Noise displacement."""
    strength = float(kwargs.get('displacement_strength', 0.0))
    scale = float(kwargs.get('displacement_map_scale', 1.0)) * 20.0
    return apply_fractal_noise(arr, displacement_strength=strength, fractal_scale=scale)

def apply_crt_phosphor_decay(arr, **kwargs):
    """Simulates phosphor decay by blurring and tinting the image."""
    decay = float(kwargs.get('phosphor_decay', 0.0))
    if decay <= 0: return arr
    color = kwargs.get('phosphor_color', 'green')
    
    blurred = gpo.gaussian_blur(arr, 0, sigmaX=5 * decay)
    
    if color == 'green':
        tint = np.array([0.8, 1.0, 0.8], dtype=np.float32)
    elif color == 'amber':
        tint = np.array([1.0, 0.8, 0.6], dtype=np.float32)
    else:
        tint = np.array([0.9, 0.9, 1.0], dtype=np.float32)
        
    return np.clip(blurred * tint, 0, 255).astype(np.uint8)

def apply_network_packet_loss(arr, **kwargs):
    """Simulates packet loss by dropping horizontal strips of the image."""
    loss_rate = float(kwargs.get('packet_loss_rate', 0.0)) / 100.0
    if loss_rate <= 0: return arr
    
    out = arr.copy()
    h, w = out.shape[:2]
    strip_height = 16
    for y in range(0, h, strip_height):
        if random.random() < loss_rate:
            out[y:y+strip_height, :] = 0 # Black strip for lost packet
    return out

def apply_buffer_underrun(arr, **kwargs):
    """Simulates a buffer underrun by freezing the top part of the image."""
    freq = float(kwargs.get('underrun_frequency', 0.0))
    if freq <= 0 or random.random() > freq: return arr
    
    h, w = arr.shape[:2]
    freeze_point = random.randint(h//4, h//2)
    top_part = arr[:freeze_point, :]
    out = arr.copy()
    out[freeze_point:, :] = cv2.resize(top_part, (w, h-freeze_point), interpolation=cv2.INTER_NEAREST)
    return out

def apply_wavelet_corruption(arr, **kwargs):
    """Corrupts wavelet coefficients to create artifacts."""
    # This is complex; we'll simulate it with a block-based noise approach
    strength = float(kwargs.get('coefficient_corruption', 0.0))
    if strength <= 0: return arr
    
    out = arr.astype(np.float32)
    h, w = out.shape[:2]
    for _ in range(int(50 * strength)):
        size = random.choice([8, 16, 32])
        x, y = random.randint(0, w-size), random.randint(0, h-size)
        noise = np.random.uniform(-50, 50)
        out[y:y+size, x:x+size] += noise
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_cellular_automata(arr, **kwargs):
    """
    Applies 1D cellular automata rules independently to each RGB channel.
    Creates full-color psychedelic patterns driven by Wolfram-style CA rules.
    """
    rule = int(kwargs.get('ca_rule', 30))
    generations = int(kwargs.get('ca_generations', 10))
    seed_from_image = bool(kwargs.get('seed_from_image', True))
    blend_mode = kwargs.get('blend_mode', 'overlay')
    animation_speed = float(kwargs.get('ca_animation_speed', 1.0))
    time_offset = float(kwargs.get('ca_time_offset', 0.0))
    
    if rule == 0 or generations == 0: 
        return arr
    
    h, w = arr.shape[:2]
    
    # Get time parameter for animation (auto time + manual offset + speed control)
    t = (float(kwargs.get('noise_warp_time', 0.0)) * animation_speed) + time_offset
    
    # Rule lookup table (Wolfram-style)
    rule_bin = format(rule, '08b')
    lookup = np.array([int(b) for b in rule_bin[::-1]], dtype=np.uint8)
    
    # Process each color channel independently for full-color CA
    ca_channels = []
    
    for channel in range(3):  # R, G, B
        # Initialize from image channel with time-varying threshold
        if seed_from_image:
            channel_data = arr[:, :, channel]
            # Vary threshold with time to animate the CA
            base_threshold = 85 + (channel * 85)  # R:85, G:170, B:255
            time_offset = int(np.sin(t + channel) * 50)  # -50 to +50 variation
            threshold = np.clip(base_threshold + time_offset, 0, 255)
            binary = (channel_data > threshold).astype(np.uint8)
        else:
            # Random seed (changes each frame)
            np.random.seed(int((t * 1000 + channel * 100) % 2**32))
            binary = np.random.randint(0, 2, (h, w), dtype=np.uint8)
        
        # Evolve cellular automata for this channel
        state = binary.copy()
        for gen in range(min(generations, h)):
            # Create neighborhood using vectorized operations
            left = np.roll(state, 1, axis=1)
            center = state
            right = np.roll(state, -1, axis=1)
            
            # Convert 3-cell neighborhood to index (0-7)
            neighborhood_idx = (left * 4 + center * 2 + right * 1).astype(np.uint8)
            
            # Apply CA rule
            new_state = lookup[neighborhood_idx]
            
            # Propagate to next row
            if gen < h - 1:
                state[gen + 1, :] = new_state[gen, :]
        
        # Convert to 0-255 range
        ca_channels.append(state * 255)
    
    # Stack channels back into RGB image
    ca_rgb = np.stack(ca_channels, axis=2).astype(np.uint8)
    
    # Blend with original image
    if blend_mode == 'overlay':
        # Overlay blend: multiply if dark, screen if bright
        result = np.where(
            arr < 128,
            (2 * arr.astype(np.float32) * ca_rgb.astype(np.float32) / 255.0).astype(np.uint8),
            (255 - 2 * (255 - arr.astype(np.float32)) * (255 - ca_rgb.astype(np.float32)) / 255.0).astype(np.uint8)
        )
    elif blend_mode == 'multiply':
        result = (arr.astype(np.float32) * ca_rgb.astype(np.float32) / 255.0).astype(np.uint8)
    elif blend_mode == 'screen':
        result = (255 - ((255 - arr.astype(np.float32)) * (255 - ca_rgb.astype(np.float32)) / 255.0)).astype(np.uint8)
    elif blend_mode == 'difference':
        result = cv2.absdiff(arr, ca_rgb)
    else:
        result = ca_rgb
    
    return result

def apply_spectral_shift(arr, **kwargs):
    """Shifts frequencies in the FFT domain."""
    strength = float(kwargs.get('frequency_shift', 0.0))
    if strength == 0: return arr
    
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(3):
        dft = cv2.dft(arr[:,:,c].astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        rows, cols = arr.shape[:2]
        crow, ccol = rows//2 , cols//2
        dx = int(ccol * strength * 0.5)
        
        # Shift the frequency spectrum
        shifted = np.roll(dft_shift, dx, axis=1)
        
        f_ishift = np.fft.ifftshift(shifted)
        img_back = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        out[:,:,c] = img_back
        
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_lut_corruption(arr, **kwargs):
    """Simulates a corrupted 3D Look-Up Table (LUT) with controlled intensity."""
    strength = float(kwargs.get('corruption_amount', 0.0))
    interp_error = float(kwargs.get('interpolation_error', 0.0))
    lut_size_str = kwargs.get('lut_size', '32')
    
    if strength <= 0: 
        return arr
    
    try:
        lut_size = int(lut_size_str)
    except (ValueError, TypeError):
        lut_size = 32
    
    # Create identity LUT (no change)
    identity = np.arange(256, dtype=np.float32)
    
    # Add corruption - controlled random offsets scaled properly
    corruption_scale = strength * 50  # Max ±50 at full strength
    
    # Create corrupted LUT for each channel
    out = arr.copy().astype(np.float32)
    
    for c in range(3):
        # Create LUT with random corruption
        corrupted_lut = identity + np.random.uniform(-corruption_scale, corruption_scale, 256)
        
        # Add interpolation errors (quantization)
        if interp_error > 0:
            # Quantize LUT to simulate reduced precision
            steps = max(4, int(256 * (1.0 - interp_error * 0.9)))
            corrupted_lut = np.round(corrupted_lut / 255.0 * steps) * (255.0 / steps)
        
        # Clamp to valid range
        corrupted_lut = np.clip(corrupted_lut, 0, 255).astype(np.uint8)
        
        # Apply LUT to this channel
        out[:, :, c] = cv2.LUT(arr[:, :, c], corrupted_lut)
    
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_vertex_shader_glitch(arr, **kwargs):
    """Simulates vertex displacement by warping the image with a grid."""
    strength = float(kwargs.get('vertex_displacement', 0.0))
    if strength <= 0: return arr
    return mesh_tear_np(arr, intensity=strength, grid=8)

def apply_perspective_glitch(arr, **kwargs):
    """Simulates perspective errors by applying a random perspective transform."""
    strength = float(kwargs.get('perspective_error', 0.0))
    if strength <= 0: return arr
    
    h, w = arr.shape[:2]
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    
    dx = w * strength * 0.1
    dy = h * strength * 0.1
    
    pts2 = np.float32([
        [random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), random.uniform(-dy, dy)],
        [random.uniform(-dx, dx), h + random.uniform(-dy, dy)],
        [w + random.uniform(-dx, dx), h + random.uniform(-dy, dy)]
    ])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(arr, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def apply_enhanced_data_moshing(arr, **kwargs):
    """A wrapper for existing block-based effects to simulate data moshing."""
    return block_displacement_np(arr, 
        block_size=int(kwargs.get('block_size', 16)),
        intensity=float(kwargs.get('block_displacement', 0.0)),
        direction='random'
    )

def apply_enhanced_pixel_sorting(arr, **kwargs):
    """Sorts pixels within rows or columns based on a criterion."""
    intensity = float(kwargs.get('pixel_sorting_intensity', 0.0))
    direction = kwargs.get('pixel_sorting_direction', 'Horizontal')
    criteria = kwargs.get('sorting_criteria', 'brightness')
    if intensity <= 0: return arr

    out = arr.copy()
    h, w = out.shape[:2]

    if direction == "Horizontal":
        for y in range(h):
            if random.random() < intensity:
                row = out[y, :, :]
                if criteria == 'brightness':
                    key = np.sum(row.astype(np.float32), axis=1)
                elif criteria == 'hue':
                    key = gpo.cvtColor(row[np.newaxis,:,:], cv2.COLOR_RGB2HSV)[0,:,0]
                else: # red, green, blue
                    key = row[:, {'red':0, 'green':1, 'blue':2}.get(criteria, 0)]
                
                sort_indices = np.argsort(key)
                out[y, :, :] = row[sort_indices]
    else: # Vertical
        for x in range(w):
            if random.random() < intensity:
                col = out[:, x, :]
                if criteria == 'brightness':
                    key = np.sum(col.astype(np.float32), axis=1)
                else:
                    key = col[:, {'red':0, 'green':1, 'blue':2}.get(criteria, 0)]
                
                sort_indices = np.argsort(key)
                out[:, x, :] = col[sort_indices]
    return out

def apply_effect_blender(arr, **kwargs):
    """Blends the image with a noisy version of itself."""
    amount = float(kwargs.get('blend_amount', 0.0))
    if amount <= 0: return arr
    noise = np.random.normal(0, 50, arr.shape).astype(np.uint8)
    return gpo.addWeighted(arr, 1.0 - amount, noise, amount, 0)

def apply_conditional_effect(arr, **kwargs):
    """Applies a simple threshold effect based on luminance."""
    thresh = float(kwargs.get('threshold_value', 0.5))
    gray = gpo.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mask = (gray / 255.0) > thresh
    out = arr.copy()
    out[mask] = 255 - out[mask] # Invert pixels above threshold
    return out

def apply_frame_difference(arr, **kwargs):
    """Returns a grayscale visualization of the image's edges (Sobel)."""
    gray = gpo.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    sobelx = gpo.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = gpo.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return gpo.cvtColor(magnitude, cv2.COLOR_GRAY2RGB)

def apply_feedback_loop(arr, **kwargs):
    """Simulates feedback by blending a transformed version of the image."""
    amount = float(kwargs.get('Feedback Amount', 0.5)) # Note: param name from node
    zoom = float(kwargs.get('Zoom', 1.0))
    if amount <= 0: return arr
    
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, zoom)
    transformed = gpo.warpAffine(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return gpo.addWeighted(arr, 1.0 - amount, transformed, amount, 0)


# ==============================================================================
# Render-Quality-First Multimodal Analysis, Modulation & Restoration Pipeline
# ==============================================================================

def compute_multimodal_analysis_maps(arr: np.ndarray, previous_arr: Optional[np.ndarray], params: Optional[dict] = None) -> dict:
    """
    Compute deterministic master-resolution multimodal analysis maps.
    All returned 2D maps have shape (H, W) dtype float32 (except optical_flow_vector (H, W, 2)).
    """
    h, w = arr.shape[:2]
    params = params or {}

    # Luminance map (0.0 .. 1.0)
    arr_f = arr.astype(np.float32)
    luma = (0.2126 * arr_f[..., 0] + 0.7152 * arr_f[..., 1] + 0.0722 * arr_f[..., 2]) / 255.0
    luma = np.clip(luma, 0.0, 1.0).astype(np.float32)

    # Grayscale uint8 for CV operations
    gray = (luma * 255.0).astype(np.uint8)

    # Edge strength and edge mask (Sobel)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(gx * gx + gy * gy)
    edge_strength = np.clip(edge_mag / 255.0, 0.0, 1.0).astype(np.float32)
    edge_threshold = float(params.get("edge_threshold", 0.15))
    edge_mask = np.clip((edge_strength - edge_threshold) / max(1e-4, 1.0 - edge_threshold), 0.0, 1.0).astype(np.float32)

    # Frame difference and Optical Flow against previous frame
    if previous_arr is not None and previous_arr.shape[:2] == (h, w):
        prev_f = previous_arr.astype(np.float32)
        prev_luma = (0.2126 * prev_f[..., 0] + 0.7152 * prev_f[..., 1] + 0.0722 * prev_f[..., 2]) / 255.0
        frame_diff = np.abs(luma - prev_luma).astype(np.float32)

        prev_gray = (np.clip(prev_luma, 0.0, 1.0) * 255.0).astype(np.uint8)
        try:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0
            ).astype(np.float32)
        except Exception:
            flow = np.zeros((h, w, 2), dtype=np.float32)
    else:
        frame_diff = np.zeros((h, w), dtype=np.float32)
        flow = np.zeros((h, w, 2), dtype=np.float32)

    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
    max_flow = max(1.0, float(np.percentile(flow_mag, 99)))
    flow_mag_norm = np.clip(flow_mag / max_flow, 0.0, 1.0).astype(np.float32)

    # Motion mask: blend normalized flow magnitude with frame difference
    motion_mask = np.clip(flow_mag_norm * 0.7 + frame_diff * 0.3, 0.0, 1.0).astype(np.float32)

    # Local Entropy: approximate spatial entropy via local standard deviation filter
    blurred_luma = cv2.blur(luma, (9, 9))
    blurred_sq_luma = cv2.blur(luma * luma, (9, 9))
    local_var = np.maximum(0.0, blurred_sq_luma - blurred_luma * blurred_luma)
    local_entropy = np.clip(np.sqrt(local_var) * 3.5, 0.0, 1.0).astype(np.float32)

    # Spectrum Energy Map: high-frequency spatial energy via Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    spectrum_energy_map = np.clip(np.abs(lap) / 128.0, 0.0, 1.0).astype(np.float32)

    # Synthesize multimodal modulation mask
    audio_boost = float(params.get("audio_boost", 0.0))
    mod_mask = (
        motion_mask * 0.35 +
        edge_strength * 0.25 +
        local_entropy * 0.20 +
        spectrum_energy_map * 0.20 +
        audio_boost * 0.20
    )
    modulation_mask = np.clip(mod_mask, 0.0, 1.0).astype(np.float32)

    return {
        "optical_flow_vector": flow,
        "optical_flow_magnitude": flow_mag,
        "frame_difference": frame_diff,
        "edge_strength": edge_strength,
        "luminance": luma,
        "local_entropy": local_entropy,
        "spectrum_energy_map": spectrum_energy_map,
        "motion_mask": motion_mask,
        "edge_mask": edge_mask,
        "modulation_mask": modulation_mask,
    }


def multimodal_analysis_np(arr: np.ndarray, previous_frame=None, params: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
    """Execute multimodal visual analysis and return pass-through image with analysis maps."""
    prev_np = None
    if previous_frame is not None:
        if isinstance(previous_frame, np.ndarray):
            prev_np = previous_frame
        elif hasattr(previous_frame, "convert"):
            prev_np = np.array(previous_frame.convert("RGB"))

    maps = compute_multimodal_analysis_maps(arr, prev_np, params)
    debug_mode = str(params.get("debug_view", "passthrough") if params else "passthrough").lower()
    
    if debug_mode == "optical_flow":
        h, w = arr.shape[:2]
        flow = maps["optical_flow_vector"]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    elif debug_mode == "modulation_mask":
        m = (maps["modulation_mask"] * 255.0).astype(np.uint8)
        out_img = np.stack([m, m, m], axis=-1)
    elif debug_mode == "motion_mask":
        m = (maps["motion_mask"] * 255.0).astype(np.uint8)
        out_img = np.stack([m, m, m], axis=-1)
    elif debug_mode == "edges":
        m = (maps["edge_mask"] * 255.0).astype(np.uint8)
        out_img = np.stack([m, m, m], axis=-1)
    elif debug_mode == "local_entropy":
        m = (maps["local_entropy"] * 255.0).astype(np.uint8)
        out_img = np.stack([m, m, m], axis=-1)
    else:
        out_img = arr

    return out_img, maps


def optical_flow_remap_np(arr: np.ndarray, flow_vector: Optional[np.ndarray] = None,
                          modulation_mask: Optional[np.ndarray] = None,
                          params: Optional[dict] = None) -> np.ndarray:
    """Displace pixels along dense optical flow field modulated by multimodal mask."""
    params = params or {}
    intensity = float(params.get("intensity", 1.0))
    if intensity == 0.0 or flow_vector is None:
        return arr

    h, w = arr.shape[:2]
    if flow_vector.shape[:2] != (h, w):
        return arr

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    flow_scale = intensity * float(params.get("flow_scale", 4.0))

    dx = flow_vector[..., 0] * flow_scale
    dy = flow_vector[..., 1] * flow_scale

    if modulation_mask is not None and modulation_mask.shape[:2] == (h, w):
        mod = modulation_mask.astype(np.float32)
        dx *= mod
        dy *= mod

    map_x = np.clip(grid_x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)

    return cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def controlled_recursive_feedback_np(arr: np.ndarray,
                                     previous_feedback: Optional[np.ndarray],
                                     clean_source: Optional[np.ndarray],
                                     motion_mask: Optional[np.ndarray] = None,
                                     params: Optional[dict] = None,
                                     state: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounded recursive feedback with opacity caps, decay, scene change resets,
    clean source mix, and motion sensitivity.
    """
    params = params or {}
    feedback_amount = float(np.clip(params.get("feedback_amount", 0.65), 0.0, 0.95))
    decay = float(np.clip(params.get("decay", 0.92), 0.0, 1.0))
    clean_mix = float(np.clip(params.get("clean_mix", 0.20), 0.0, 1.0))
    scene_threshold = float(params.get("scene_change_threshold", 0.40))
    zoom = float(params.get("zoom", 1.01))
    rotation = float(params.get("rotation", 0.0))
    offset_x = float(params.get("offset_x", 0.0))
    offset_y = float(params.get("offset_y", 0.0))

    h, w = arr.shape[:2]
    clean = clean_source if clean_source is not None and clean_source.shape == arr.shape else arr

    if previous_feedback is None or previous_feedback.shape != arr.shape:
        return arr.copy(), arr.copy()

    # Scene change detection to prevent dirty ghost smearing across cuts
    prev_source = state.get("prev_source") if state else None
    if prev_source is not None and prev_source.shape == arr.shape:
        diff = np.mean(np.abs(arr.astype(np.float32) - prev_source.astype(np.float32))) / 255.0
        if diff > scene_threshold:
            # Scene cut detected - reset feedback
            if state is not None:
                state["prev_source"] = arr.copy()
            return arr.copy(), arr.copy()

    if state is not None:
        state["prev_source"] = arr.copy()

    # Transform previous feedback
    center = (w * 0.5 + offset_x, h * 0.5 + offset_y)
    M = cv2.getRotationMatrix2D(center, rotation, zoom)
    transformed_fb = cv2.warpAffine(previous_feedback, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Apply decay
    decayed_fb = transformed_fb.astype(np.float32) * decay

    # Motion masking
    if motion_mask is not None and motion_mask.shape[:2] == (h, w):
        mot = motion_mask[..., None].astype(np.float32)
        effective_feedback = feedback_amount * (0.3 + 0.7 * mot)
    else:
        effective_feedback = feedback_amount

    # Blend current frame with decayed feedback and clean source
    blended = (
        arr.astype(np.float32) * (1.0 - effective_feedback) +
        decayed_fb * effective_feedback
    )
    if clean_mix > 0.0:
        blended = blended * (1.0 - clean_mix) + clean.astype(np.float32) * clean_mix

    out = np.clip(blended, 0, 255).astype(np.uint8)
    return out, out


def quantization_solarization_np(arr: np.ndarray,
                                 modulation_mask: Optional[np.ndarray] = None,
                                 params: Optional[dict] = None) -> np.ndarray:
    """Multimodally-modulated color quantization and solarization."""
    params = params or {}
    levels = max(2, min(256, int(params.get("levels", 16))))
    solarize_thresh = float(np.clip(params.get("solarize_threshold", 0.65), 0.0, 1.0)) * 255.0
    solarize_mix = float(np.clip(params.get("solarize_mix", 0.4), 0.0, 1.0))
    intensity = float(np.clip(params.get("intensity", 1.0), 0.0, 1.0))

    if intensity <= 0.0:
        return arr

    arr_f = arr.astype(np.float32)
    step = 255.0 / float(levels - 1)
    quantized = np.round(arr_f / step) * step

    # Solarization: invert components where luminance or value exceeds threshold
    solarized = np.where(arr_f > solarize_thresh, 255.0 - arr_f, arr_f)

    combined = quantized * (1.0 - solarize_mix) + solarized * solarize_mix

    if modulation_mask is not None and modulation_mask.shape[:2] == arr.shape[:2]:
        mod = modulation_mask[..., None].astype(np.float32) * intensity
        out = arr_f * (1.0 - mod) + combined * mod
    else:
        out = arr_f * (1.0 - intensity) + combined * intensity

    return np.clip(out, 0, 255).astype(np.uint8)


def edge_reconstruction_np(arr: np.ndarray,
                           edge_mask: Optional[np.ndarray] = None,
                           params: Optional[dict] = None) -> np.ndarray:
    """Reinforce destroyed structural edges and sharpen key contours."""
    params = params or {}
    strength = float(params.get("strength", 0.75))
    if strength <= 0.0:
        return arr

    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

    if edge_mask is not None and edge_mask.shape[:2] == (h, w):
        mask = edge_mask.astype(np.float32)
    else:
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sobelx * sobelx + sobely * sobely)
        mask = np.clip(mag / 128.0, 0.0, 1.0)

    edge_detail = np.clip(np.abs(lap), 0, 255).astype(np.float32)
    boost = (edge_detail * mask)[..., None] * (strength * 0.6)

    out = arr.astype(np.float32) + boost
    return np.clip(out, 0, 255).astype(np.uint8)


def structural_restoration_np(arr: np.ndarray,
                             edge_mask: Optional[np.ndarray] = None,
                             params: Optional[dict] = None) -> np.ndarray:
    """
    Final structural restoration node supporting restrained unsharp masking,
    local-contrast recovery (CLAHE), edge-derived luminance reinforcement,
    final clamping, and optional grain normalization.
    """
    params = params or {}
    unsharp_amount = float(params.get("unsharp_amount", 0.70))
    unsharp_radius = max(0.5, float(params.get("unsharp_radius", 1.2)))
    unsharp_threshold = float(params.get("unsharp_threshold", 4.0))
    clahe_clip = float(params.get("local_contrast_clip", 1.5))
    edge_reinforce = float(params.get("edge_reinforce", 0.35))
    grain_amount = float(np.clip(params.get("grain_amount", 0.02), 0.0, 0.20))
    grain_seed = params.get("grain_seed")

    arr_f = arr.astype(np.float32)
    h, w = arr.shape[:2]

    # 1. Restrained Unsharp Masking with Thresholding
    if unsharp_amount > 0.0:
        ksize = int(round(unsharp_radius * 3.0)) * 2 + 1
        blurred = cv2.GaussianBlur(arr_f, (ksize, ksize), unsharp_radius)
        diff = arr_f - blurred
        mask = np.abs(diff) > unsharp_threshold
        arr_f = arr_f + (diff * mask) * unsharp_amount

    # 2. Local Contrast Recovery (CLAHE on L-channel of LAB)
    if clahe_clip > 0.0:
        arr_uint8 = np.clip(arr_f, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        arr_f = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)

    # 3. Edge-derived luminance reinforcement
    if edge_reinforce > 0.0:
        gray = (0.2126 * arr_f[..., 0] + 0.7152 * arr_f[..., 1] + 0.0722 * arr_f[..., 2])
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_detail = np.sqrt(gx * gx + gy * gy)
        edge_norm = np.clip(edge_detail / 128.0, 0.0, 1.0)[..., None]
        arr_f = arr_f * (1.0 + edge_norm * (edge_reinforce * 0.15))

    # 4. Optional Grain Normalization
    if grain_amount > 0.0:
        rng = np.random.default_rng(grain_seed)
        grain = rng.standard_normal((h, w, 3)).astype(np.float32) * (grain_amount * 255.0)
        arr_f = arr_f + grain

    # 5. Final Clamping
    return np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
