"""Deterministic CPU fallback for Bubble Sim Generator (2)."""

from __future__ import annotations

import cv2
import numpy as np


DEFAULTS = {
    "seed": 1977,
    "particle_texture_size": 64,
    "density": 0.12,
    "base_radius": 0.021,
    "radius_variation": 0.8,
    "buoyancy": 0.02,
    "drag": 0.94,
    "turbulence": 0.30,
    "flow_speed": 0.22,
    "motion_speed": 1.0,
    "motion_style": "Vortex",
    "depth_spread": 0.95,
    "depth_motion": 0.35,
    "orbit_strength": 0.70,
    "perspective": 1.2,
    "refraction_strength": 0.040,
    "highlight_strength": 0.34,
    "sparkle": 0.04,
    "show_energy": 0.0,
    "show_chaos": 0.0,
    "mutation_strength": 0.0,
    "mutation_rate": 1.0,
    "mutation_warp": 0.5,
    "mutation_palette": 0.4,
    "mutation_fracture": 0.4,
}


def _f(params, name, low, high):
    return float(np.clip(params.get(name, DEFAULTS[name]), low, high))


def _i(params, name, low, high):
    return int(np.clip(params.get(name, DEFAULTS[name]), low, high))


def _resize_source(source, width, height):
    if source is None:
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
        top = np.array([18.0, 54.0, 68.0], dtype=np.float32)[None, None, :]
        bottom = np.array([2.0, 10.0, 18.0], dtype=np.float32)[None, None, :]
        return np.clip(top * (1.0 - y) + bottom * y, 0, 255).astype(np.uint8).repeat(width, axis=1)
    arr = np.asarray(source)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[:2] != (height, width):
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
    return arr


def generate_bubble_sim2_cpu_outputs(width, height, time_value, params, source=None):
    """Return distinct uint8 RGB buffers for the V2 bubble generator."""
    width, height = max(1, int(width)), max(1, int(height))
    seed = _i(params, "seed", 0, 999999)
    tex_size = _i(params, "particle_texture_size", 16, 512)
    density = _f(params, "density", 0.0, 1.0)
    base_radius = _f(params, "base_radius", 0.001, 0.08)
    radius_variation = _f(params, "radius_variation", 0.0, 1.0)
    buoyancy = _f(params, "buoyancy", 0.0, 2.0)
    drag = _f(params, "drag", 0.0, 1.0)
    turbulence = _f(params, "turbulence", 0.0, 2.0)
    flow_speed = _f(params, "flow_speed", 0.0, 4.0)
    motion_speed = _f(params, "motion_speed", 0.0, 4.0)
    motion_style = str(params.get("motion_style", DEFAULTS["motion_style"]))
    depth_spread = _f(params, "depth_spread", 0.0, 1.0)
    depth_motion = _f(params, "depth_motion", 0.0, 2.0)
    orbit_strength = _f(params, "orbit_strength", 0.0, 2.0)
    perspective_strength = _f(params, "perspective", 0.0, 2.0)
    refraction_strength = _f(params, "refraction_strength", 0.0, 0.15)
    highlight_strength = _f(params, "highlight_strength", 0.0, 2.0)
    sparkle = _f(params, "sparkle", 0.0, 1.0)
    energy = _f(params, "show_energy", 0.0, 1.0)
    chaos = _f(params, "show_chaos", 0.0, 1.0)
    mutation = float(np.clip(params.get("mutation_strength", 0.0), 0.0, 1.0))
    mutation_rate = float(np.clip(params.get("mutation_rate", 1.0), 0.0, 5.0))
    mutation_warp = float(np.clip(params.get("mutation_warp", 0.5), 0.0, 2.0))
    mutation_palette = float(np.clip(params.get("mutation_palette", 0.4), 0.0, 2.0))
    mutation_fracture = float(np.clip(params.get("mutation_fracture", 0.4), 0.0, 1.0))

    scale = min(1.0, 640.0 / max(width, height))
    ww, wh = max(64, int(round(width * scale))), max(64, int(round(height * scale)))
    count = min(int(tex_size * tex_size * density), 900)
    rng = np.random.default_rng(seed)
    ids = np.arange(count, dtype=np.float32)
    x0 = rng.random(count, dtype=np.float32)
    y0 = rng.random(count, dtype=np.float32)
    z0 = rng.random(count, dtype=np.float32)
    age0 = rng.random(count, dtype=np.float32)
    rr = rng.random(count, dtype=np.float32)
    phase = rng.random(count, dtype=np.float32) * np.pi * 2.0

    drag_motion = 0.2 + 0.8 * drag
    age = np.mod(age0 + time_value * motion_speed * (0.04 + buoyancy * 0.12), 1.0)
    depth = np.mod(
        z0
        + np.sin(time_value * motion_speed * flow_speed * 0.37 + phase) * depth_motion * turbulence * 0.07
        + np.sin(time_value * motion_speed * 0.19 + ids * 0.13) * chaos * depth_motion * 0.04,
        1.0,
    )
    depth = np.clip(depth * max(depth_spread, 0.001), 0.0, 1.0)
    rise = (0.006 + buoyancy * 0.34) * (1.25 - depth * 0.65)
    world_y = np.mod(y0 + time_value * motion_speed * rise * 0.95 + age * 0.06, 1.3) - 0.15
    orbit = phase + time_value * motion_speed * flow_speed * (0.9 + rr) + age * np.pi
    curl = np.sin(orbit) * orbit_strength * turbulence * 0.12 * drag_motion
    drift_rate = 0.35 + flow_speed * 0.65
    projected_orbit_x = np.sin(
        time_value * motion_speed * drift_rate * (0.75 + rr * 0.45) + phase
    ) * (0.006 + orbit_strength * 0.022 + turbulence * 0.010) * (1.0 - depth * 0.55)
    projected_orbit_y = np.cos(
        time_value * motion_speed * drift_rate * (0.52 + rr * 0.32) + phase * 0.7
    ) * (0.006 + orbit_strength * 0.022 + turbulence * 0.010) * (1.0 - depth * 0.55)
    chaos_jitter = np.sin(time_value * motion_speed * 8.0 + ids * 0.37) * chaos * 0.025
    depth_bob = np.sin(time_value * motion_speed * (0.9 + flow_speed * 0.2) + phase * 2.3) * depth_motion * 0.018
    flow = np.sin(time_value * motion_speed * 0.16 + y0 * 9.0 + depth * 4.0) * turbulence * 0.055
    fracture = np.sign(np.sin(ids * 1.71 + np.floor(time_value * 3.0 * mutation_rate))) * mutation * mutation_fracture * 0.025
    world_x = x0 + curl + flow + fracture + projected_orbit_x + chaos_jitter
    centered_x = world_x - 0.5
    centered_y = world_y - 0.35
    if motion_style == "Vortex":
        angle = time_value * motion_speed * (0.12 + orbit_strength * 0.18) * np.clip(
            1.0 - np.sqrt(centered_x * centered_x + centered_y * centered_y),
            0.0,
            1.0,
        )
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        world_x = 0.5 + centered_x * cos_a - centered_y * sin_a
        world_y = 0.35 + centered_x * sin_a + centered_y * cos_a
    elif motion_style == "Burst":
        radial_length = np.maximum(
            np.sqrt(centered_x * centered_x + centered_y * centered_y),
            1e-5,
        )
        pulse = np.sin(time_value * motion_speed * 1.7 + phase * 9.0) * (
            0.015 + chaos * 0.04
        )
        world_x += centered_x / radial_length * pulse
        world_y += centered_y / radial_length * pulse
    elif motion_style == "Rising":
        world_y += np.mod(age + time_value * motion_speed * 0.08, 1.0) * 0.08 - 0.04

    perspective = np.clip(1.0 / (1.0 + perspective_strength * (depth * 1.1 - 0.32)), 0.42, 2.5)
    px = np.mod(world_x, 1.0) * (ww - 1)
    py = (1.05 - world_y + projected_orbit_y + depth_bob) * (wh - 1)
    radius_norm = base_radius * (0.35 + rr * (1.0 + radius_variation))
    radius_px = radius_norm * wh * perspective * (1.0 + energy * 0.25)

    mask = np.zeros((wh, ww), np.float32)
    depth_plane = np.zeros_like(mask)
    age_plane = np.zeros_like(mask)
    radius_plane = np.zeros_like(mask)
    rim_plane = np.zeros_like(mask)
    highlight = np.zeros((wh, ww, 3), np.float32)

    for index in np.argsort(depth)[::-1]:
        radius = max(1, int(round(radius_px[index])))
        cx, cy = int(px[index]), int(py[index])
        if cx + radius < 0 or cx - radius >= ww or cy + radius < 0 or cy - radius >= wh:
            continue
        axes = (radius, max(1, int(radius * (0.9 + 0.08 * np.sin(orbit[index])))))
        alpha = float((0.3 + (1.0 - depth[index]) * 0.55) * (0.72 + energy * 0.28))
        cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, alpha, -1, cv2.LINE_AA)
        cv2.ellipse(depth_plane, (cx, cy), axes, 0, 0, 360, float(depth[index]), -1, cv2.LINE_AA)
        cv2.ellipse(age_plane, (cx, cy), axes, 0, 0, 360, float(age[index]), -1, cv2.LINE_AA)
        cv2.ellipse(radius_plane, (cx, cy), axes, 0, 0, 360, float(np.clip(radius_norm[index] / 0.08, 0, 1)), -1, cv2.LINE_AA)
        cv2.ellipse(rim_plane, (cx, cy), axes, 0, 0, 360, alpha, max(1, radius // 6), cv2.LINE_AA)
        pop = np.clip((age[index] - 0.8) / 0.2, 0.0, 1.0)
        glint = float(
            (np.sin(ids[index] * 19.7 + np.floor(time_value * 18.0)) > 0.40)
            * (0.25 + pop * 0.75)
            * sparkle
            * 1.8
        )
        tint_shift = mutation * mutation_palette * 0.25 * np.sin(time_value * 0.3 * mutation_rate + ids[index])
        color = np.array([0.62 + tint_shift, 0.82, 0.92 - tint_shift], np.float32)
        rim_color = tuple(float(v) for v in color * highlight_strength * 0.18)
        cv2.ellipse(highlight, (cx, cy), axes, 0, 0, 360, rim_color, max(1, radius // 7), cv2.LINE_AA)
        light = (cx - max(1, radius // 3), cy - max(1, radius // 3))
        glint_color = tuple(float(v) for v in color * highlight_strength * (0.35 + glint))
        cv2.circle(highlight, light, max(1, radius // 8), glint_color, -1, cv2.LINE_AA)

    mask = np.clip(cv2.GaussianBlur(mask, (0, 0), 0.65), 0.0, 1.0)
    gx = cv2.Sobel(mask, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    nx = np.where(mag > 1e-5, -gx / np.maximum(mag, 1e-5), 0.0)
    ny = np.where(mag > 1e-5, -gy / np.maximum(mag, 1e-5), 0.0)
    nz = np.sqrt(np.clip(1.0 - np.minimum(1.0, nx * nx + ny * ny), 0.0, 1.0))
    normal = np.dstack((nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5))
    normal = normal * mask[..., None] + (1.0 - mask[..., None]) * np.array([0.5, 0.5, 1.0])
    mask_rgb = np.repeat(mask[..., None], 3, axis=2)
    depth_rgb = np.dstack((depth_plane, age_plane, radius_plane))
    refract_rgb = np.dstack((
        np.clip(0.5 + nx * refraction_strength * 4.0, 0.0, 1.0),
        np.clip(0.5 + ny * refraction_strength * 4.0, 0.0, 1.0),
        np.clip(rim_plane, 0.0, 1.0),
    ))
    highlight = np.clip(highlight, 0.0, 1.0)

    background = _resize_source(source, ww, wh)
    yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
    warp_gain = 1.0 + mutation * mutation_warp
    map_x = np.clip(xx + nx * refraction_strength * ww * 0.35 * warp_gain * mask, 0, ww - 1).astype(np.float32)
    map_y = np.clip(yy + ny * refraction_strength * wh * 0.10 * warp_gain * mask, 0, wh - 1).astype(np.float32)
    refracted = cv2.remap(background, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fade = (0.05 + depth_plane * 0.12) * mask
    beauty = refracted.astype(np.float32) * (1.0 - fade[..., None])
    beauty += highlight * 255.0
    beauty += rim_plane[..., None] * np.array([8.0, 15.0, 18.0], np.float32)
    beauty = np.clip(beauty, 0, 255)

    outputs_float = {
        "image_out": beauty / 255.0,
        "final_frame": beauty / 255.0,
        "bubble_mask": mask_rgb,
        "bubble_normal": normal,
        "bubble_depth": depth_rgb,
        "bubble_refraction": refract_rgb,
        "bubble_highlight": highlight,
    }
    debug_mode = str(params.get("debug_mode", "composite") or "composite")
    debug_key = {
        "mask": "bubble_mask",
        "normal": "bubble_normal",
        "depth": "bubble_depth",
        "refraction": "bubble_refraction",
        "highlight": "bubble_highlight",
    }.get(debug_mode, "image_out")
    outputs_float["bubble_debug"] = outputs_float[debug_key]

    outputs = {}
    for name, value in outputs_float.items():
        frame = np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        outputs[name] = frame
    return outputs
