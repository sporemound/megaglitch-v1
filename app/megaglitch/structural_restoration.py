"""Deterministic, master-resolution structural restoration.

This module deliberately contains no resize operation and no temporal state.  It
is intended to be the final image-domain node on the master render path, after
all recursive, remap, blur, and corruption stages.  Preview and output-branch
scaling must happen after this function returns.
"""

from __future__ import annotations

import cv2
import numpy as np


_LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _bounded_float(value: float, minimum: float, maximum: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return float(np.clip(result, minimum, maximum))


def _gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Blur without changing dimensions or implicitly converting formats."""

    return cv2.GaussianBlur(
        image,
        (0, 0),
        sigmaX=max(0.01, float(sigma)),
        sigmaY=max(0.01, float(sigma)),
        borderType=cv2.BORDER_REFLECT101,
    )


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return np.sum(rgb * _LUMA_WEIGHTS, axis=2, dtype=np.float32)


def apply_structural_restoration(
    frame: np.ndarray,
    *,
    unsharp_amount: float = 0.20,
    unsharp_radius: float = 1.10,
    local_contrast: float = 0.12,
    local_contrast_radius: float = 10.0,
    edge_luma_reinforcement: float = 0.05,
    edge_sensitivity: float = 0.05,
    grain_normalization: float = 0.0,
    grain_target: float = 0.012,
    final_clamp_low: int = 0,
    final_clamp_high: int = 255,
) -> np.ndarray:
    """Restore restrained structure on an RGB8 master frame.

    Sharpening, broad local-contrast recovery, and edge reinforcement operate in
    luminance so the node does not invent channel offsets or chromatic halos.
    Grain normalization deterministically moves flat-area fine-detail energy
    toward ``grain_target``; it never synthesizes random noise.  The input and
    output resolution are always identical.

    A zero-strength configuration with the full 0..255 clamp is an exact pixel
    identity (apart from returning an owned, C-contiguous output buffer).
    """

    if not isinstance(frame, np.ndarray):
        raise TypeError("structural restoration requires a numpy RGB frame")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("structural restoration requires an HxWx3 RGB frame")
    if frame.dtype != np.uint8:
        raise TypeError("structural restoration requires uint8 master frames")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("structural restoration requires a non-empty frame")

    unsharp_amount = _bounded_float(unsharp_amount, 0.0, 1.0, "unsharp_amount")
    unsharp_radius = _bounded_float(unsharp_radius, 0.4, 3.0, "unsharp_radius")
    local_contrast = _bounded_float(local_contrast, 0.0, 0.75, "local_contrast")
    local_contrast_radius = _bounded_float(
        local_contrast_radius, 2.0, 32.0, "local_contrast_radius"
    )
    edge_luma_reinforcement = _bounded_float(
        edge_luma_reinforcement, 0.0, 0.5, "edge_luma_reinforcement"
    )
    edge_sensitivity = _bounded_float(
        edge_sensitivity, 0.005, 0.25, "edge_sensitivity"
    )
    grain_normalization = _bounded_float(
        grain_normalization, 0.0, 1.0, "grain_normalization"
    )
    grain_target = _bounded_float(grain_target, 0.0, 0.08, "grain_target")
    clamp_a = int(np.clip(int(final_clamp_low), 0, 255))
    clamp_b = int(np.clip(int(final_clamp_high), 0, 255))
    clamp_low, clamp_high = sorted((clamp_a, clamp_b))

    if (
        unsharp_amount == 0.0
        and local_contrast == 0.0
        and edge_luma_reinforcement == 0.0
        and grain_normalization == 0.0
        and clamp_low == 0
        and clamp_high == 255
    ):
        return np.ascontiguousarray(frame.copy())

    rgb = frame.astype(np.float32) * (1.0 / 255.0)
    source_luma = _luminance(rgb)
    restoration = np.zeros_like(source_luma)

    if unsharp_amount > 0.0:
        fine_base = _gaussian(source_luma, unsharp_radius)
        restoration += unsharp_amount * (source_luma - fine_base)

    if local_contrast > 0.0:
        local_base = _gaussian(source_luma, local_contrast_radius)
        restoration += local_contrast * (source_luma - local_base)

    needs_edge_map = edge_luma_reinforcement > 0.0 or grain_normalization > 0.0
    if needs_edge_map:
        gradient_x = cv2.Sobel(
            source_luma, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT101
        )
        gradient_y = cv2.Sobel(
            source_luma, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT101
        )
        gradient_x64 = gradient_x.astype(np.float64)
        gradient_y64 = gradient_y.astype(np.float64)
        gradient = np.sqrt(
            gradient_x64 * gradient_x64 + gradient_y64 * gradient_y64
        ).astype(np.float32)
        edge_weight = gradient / (gradient + edge_sensitivity)
    else:
        edge_weight = None

    if edge_luma_reinforcement > 0.0:
        edge_detail = source_luma - _gaussian(source_luma, 0.8)
        restoration += edge_luma_reinforcement * edge_detail * edge_weight

    restored = rgb + restoration[..., None]

    if grain_normalization > 0.0:
        grain_base = _gaussian(restored, 0.8)
        grain = restored - grain_base
        flat_weight = np.square(1.0 - edge_weight)[..., None]
        weighted_samples = float(np.sum(flat_weight)) * 3.0
        if weighted_samples > 1e-6:
            grain_rms = float(
                np.sqrt(np.sum(np.square(grain) * flat_weight) / weighted_samples)
            )
            if grain_rms > 1e-6:
                target_scale = float(np.clip(grain_target / grain_rms, 0.25, 2.0))
                restored += (
                    grain_normalization
                    * (target_scale - 1.0)
                    * grain
                    * flat_weight
                )

    restored = np.clip(restored, clamp_low / 255.0, clamp_high / 255.0)
    output = np.rint(restored * 255.0).astype(np.uint8)
    return np.ascontiguousarray(output)


__all__ = ["apply_structural_restoration"]
