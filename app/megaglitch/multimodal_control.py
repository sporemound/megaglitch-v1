"""Deterministic master-resolution analysis and controlled recursive feedback.

This module deliberately has no preview, display, OBS, virtual-camera, or encoder
dependencies.  Every image-shaped input and output is tied to the source master
shape; a resolution change must be handled by calling :meth:`reset` at ingress.

The analyzer exposes feature maps as a typed side-channel rather than encoding
them into an RGB frame.  That lets optical-flow remap, feedback, temporal
corruption, tearing, and restoration share one analysis pass without sending an
analysis texture through the image graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


class MasterResolutionError(ValueError):
    """Raised when an image or mask violates the bound master-frame contract."""


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("smoothstep high threshold must be greater than low threshold")
    normalized = np.clip((value - low) / (high - low), 0.0, 1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _as_contiguous_float32(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


def _validate_master_frame(
    frame: np.ndarray,
    *,
    expected_shape: tuple[int, int, int] | None = None,
    name: str = "master frame",
) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise MasterResolutionError(
            f"{name} must have shape (height, width, 3); got {frame.shape!r}"
        )
    if frame.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8 RGB; got {frame.dtype}")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise MasterResolutionError(f"{name} cannot be empty")
    if not frame.flags.c_contiguous:
        raise MasterResolutionError(
            f"{name} must be contiguous; make the ingress copy explicit"
        )
    if expected_shape is not None and tuple(frame.shape) != tuple(expected_shape):
        raise MasterResolutionError(
            f"{name} shape {tuple(frame.shape)} does not match master shape "
            f"{tuple(expected_shape)}"
        )
    return frame


def _validate_mask(
    mask: np.ndarray,
    master_shape: tuple[int, int, int],
    *,
    name: str,
) -> np.ndarray:
    expected = master_shape[:2]
    if not isinstance(mask, np.ndarray) or mask.shape != expected:
        raise MasterResolutionError(
            f"{name} shape {getattr(mask, 'shape', None)} does not match master "
            f"mask shape {expected}"
        )
    if not np.issubdtype(mask.dtype, np.floating):
        raise TypeError(f"{name} must be floating point; got {mask.dtype}")
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"{name} contains non-finite values")
    return mask


@dataclass(frozen=True)
class MultimodalAnalysisConfig:
    """Fixed analysis parameters used for both live and offline master paths."""

    entropy_window: int = 9
    entropy_bins: int = 16
    flow_scale_pixels: float = 12.0
    spectrum_local_scale: float = 0.12
    spectrum_high_frequency_cutoff: float = 0.18
    motion_low: float = 0.035
    motion_high: float = 0.42
    edge_low: float = 0.025
    edge_high: float = 0.32
    scene_change_threshold: float = 0.32
    modulation_gamma: float = 1.12

    def __post_init__(self) -> None:
        if self.entropy_window < 3 or self.entropy_window % 2 == 0:
            raise ValueError("entropy_window must be an odd integer >= 3")
        if self.entropy_bins < 2 or self.entropy_bins > 64:
            raise ValueError("entropy_bins must be in [2, 64]")
        if self.flow_scale_pixels <= 0.0:
            raise ValueError("flow_scale_pixels must be positive")
        if self.spectrum_local_scale <= 0.0:
            raise ValueError("spectrum_local_scale must be positive")
        if not 0.0 < self.spectrum_high_frequency_cutoff < 0.71:
            raise ValueError("spectrum_high_frequency_cutoff must be in (0, 0.71)")
        if not 0.0 <= self.motion_low < self.motion_high <= 1.0:
            raise ValueError("motion thresholds must be ordered inside [0, 1]")
        if not 0.0 <= self.edge_low < self.edge_high <= 1.0:
            raise ValueError("edge thresholds must be ordered inside [0, 1]")
        if not 0.0 <= self.scene_change_threshold <= 1.0:
            raise ValueError("scene_change_threshold must be in [0, 1]")
        if self.modulation_gamma <= 0.0:
            raise ValueError("modulation_gamma must be positive")


@dataclass(frozen=True)
class MultimodalAnalysis:
    """Master-resolution feature maps plus scalar frame telemetry.

    Maps are normalized ``float32`` arrays in ``[0, 1]``.  Their shape is
    always exactly ``master_shape[:2]``; no analysis stage downsamples or
    reconstructs a preview-sized approximation.
    """

    master_shape: tuple[int, int, int]
    optical_flow_vector: np.ndarray
    optical_flow_magnitude: np.ndarray
    frame_difference: np.ndarray
    edge_strength: np.ndarray
    luminance: np.ndarray
    local_entropy: np.ndarray
    spectrum_energy_map: np.ndarray
    motion_mask: np.ndarray
    edge_mask: np.ndarray
    modulation_mask: np.ndarray
    telemetry: Mapping[str, float | int | bool]

    def assert_master_integrity(self) -> None:
        expected_vector_shape = self.master_shape[:2] + (2,)
        if (
            not isinstance(self.optical_flow_vector, np.ndarray)
            or self.optical_flow_vector.shape != expected_vector_shape
            or self.optical_flow_vector.dtype != np.float32
            or not np.all(np.isfinite(self.optical_flow_vector))
        ):
            raise MasterResolutionError(
                "optical_flow_vector must be finite float32 at master shape "
                f"{expected_vector_shape}; got "
                f"{getattr(self.optical_flow_vector, 'shape', None)}"
            )
        for name in (
            "optical_flow_magnitude",
            "frame_difference",
            "edge_strength",
            "luminance",
            "local_entropy",
            "spectrum_energy_map",
            "motion_mask",
            "edge_mask",
            "modulation_mask",
        ):
            value = _validate_mask(getattr(self, name), self.master_shape, name=name)
            minimum = float(np.min(value))
            maximum = float(np.max(value))
            if minimum < -1e-6 or maximum > 1.0 + 1e-6:
                raise ValueError(
                    f"{name} must be normalized to [0, 1]; got [{minimum}, {maximum}]"
                )


import functools


@functools.lru_cache(maxsize=16)
def _get_entropy_lut(window: int) -> np.ndarray:
    """Precompute exact single-precision -p*log2(p) lookup table for box filter counts."""
    area = float(window * window)
    lut = np.zeros(int(area) + 1, dtype=np.float32)
    for k in range(1, int(area) + 1):
        p = np.float32(k) / np.float32(area)
        if p > np.float32(1e-12):
            lut[k] = p * np.log2(p)
    return lut


def _local_entropy(
    luminance_u8: np.ndarray,
    *,
    window: int,
    bins: int,
) -> np.ndarray:
    """Compute deterministic local Shannon entropy without changing resolution."""

    lut = _get_entropy_lut(int(window))
    quantized = np.minimum(
        (luminance_u8.astype(np.uint16) * bins) // 256,
        bins - 1,
    ).astype(np.uint8)
    entropy = np.zeros(luminance_u8.shape, dtype=np.float32)
    for index in range(bins):
        occ = (quantized == index).view(np.uint8)
        count = cv2.boxFilter(
            occ,
            ddepth=cv2.CV_8U,
            ksize=(window, window),
            normalize=False,
            borderType=cv2.BORDER_REFLECT_101,
        )
        entropy -= lut[count]
    entropy /= np.log2(float(bins))
    return _as_contiguous_float32(np.clip(entropy, 0.0, 1.0))


def _spectrum_energy(
    luminance: np.ndarray,
    *,
    cutoff: float,
) -> float:
    """Return the normalized non-DC image-spectrum energy above ``cutoff``."""

    centered = luminance.astype(np.float32) - float(np.mean(luminance))
    spectrum = np.fft.rfft2(centered)
    power = np.abs(spectrum) ** 2
    total = float(np.sum(power))
    if total <= 1e-12:
        return 0.0

    fy = np.fft.fftfreq(luminance.shape[0])[:, None]
    fx = np.fft.rfftfreq(luminance.shape[1])[None, :]
    high_frequency = np.sqrt(fx * fx + fy * fy) >= cutoff
    return _clamp01(float(np.sum(power[high_frequency])) / total)


def _band_means(
    audio_fft_bands: Mapping[str, float] | Sequence[float] | np.ndarray | None,
) -> tuple[float, float, float, float]:
    if audio_fft_bands is None:
        return 0.0, 0.0, 0.0, 0.0

    if isinstance(audio_fft_bands, Mapping):
        entries = [
            (str(name).strip().lower(), _clamp01(value))
            for name, value in sorted(
                audio_fft_bands.items(), key=lambda item: str(item[0])
            )
        ]
        low_names = ("sub", "bass", "low", "kick")
        mid_names = ("mid", "body", "voice", "vocal")
        high_names = ("high", "treble", "presence", "air", "hat")
        classified = {
            "low": [
                value
                for name, value in entries
                if any(token in name for token in low_names)
            ],
            "mid": [
                value
                for name, value in entries
                if any(token in name for token in mid_names)
            ],
            "high": [
                value
                for name, value in entries
                if any(token in name for token in high_names)
            ],
        }
        ordered = [value for _, value in entries]
    else:
        array = np.asarray(audio_fft_bands, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return 0.0, 0.0, 0.0, 0.0
        if not np.all(np.isfinite(array)):
            raise ValueError("audio_fft_bands contains non-finite values")
        ordered = [float(np.clip(value, 0.0, 1.0)) for value in array]
        classified = {"low": [], "mid": [], "high": []}

    if not ordered:
        return 0.0, 0.0, 0.0, 0.0
    thirds = np.array_split(np.asarray(ordered, dtype=np.float32), 3)
    fallback = [float(np.mean(group)) if group.size else 0.0 for group in thirds]
    low = float(np.mean(classified["low"])) if classified["low"] else fallback[0]
    mid = float(np.mean(classified["mid"])) if classified["mid"] else fallback[1]
    high = float(np.mean(classified["high"])) if classified["high"] else fallback[2]
    return (
        _clamp01(low),
        _clamp01(mid),
        _clamp01(high),
        _clamp01(float(np.mean(ordered))),
    )


class MasterMultimodalAnalyzer:
    """Analyze consecutive full-resolution master frames deterministically."""

    def __init__(self, config: MultimodalAnalysisConfig | None = None) -> None:
        self.config = config or MultimodalAnalysisConfig()
        self._master_shape: tuple[int, int, int] | None = None

    @property
    def master_shape(self) -> tuple[int, int, int] | None:
        return self._master_shape

    def reset(self) -> None:
        """Unbind resolution; callers may then establish a new ingress shape."""

        self._master_shape = None

    def analyze(
        self,
        master_frame: np.ndarray,
        previous_master_frame: np.ndarray | None = None,
        *,
        audio_fft_bands: Mapping[str, float]
        | Sequence[float]
        | np.ndarray
        | None = None,
        transient: float = 0.0,
        beat_state: float | bool = 0.0,
    ) -> MultimodalAnalysis:
        frame = _validate_master_frame(
            master_frame,
            expected_shape=self._master_shape,
        )
        if self._master_shape is None:
            self._master_shape = tuple(frame.shape)
        master_shape = self._master_shape

        previous = None
        if previous_master_frame is not None:
            previous = _validate_master_frame(
                previous_master_frame,
                expected_shape=master_shape,
                name="previous master frame",
            )

        luma_u8 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        luminance = _as_contiguous_float32(luma_u8.astype(np.float32) / 255.0)

        grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
        grad_x64 = grad_x.astype(np.float64)
        grad_y64 = grad_y.astype(np.float64)
        edge_strength = _as_contiguous_float32(
            np.clip(np.sqrt(grad_x64 * grad_x64 + grad_y64 * grad_y64) / 4.0, 0.0, 1.0)
        )

        if previous is None:
            difference = np.zeros(luminance.shape, dtype=np.float32)
            flow_vector = np.zeros(luminance.shape + (2,), dtype=np.float32)
            flow_magnitude = np.zeros(luminance.shape, dtype=np.float32)
        else:
            previous_luma_u8 = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
            difference = _as_contiguous_float32(
                np.mean(
                    cv2.absdiff(frame, previous),
                    axis=2,
                    dtype=np.float32,
                )
                / 255.0
            )
            flow = cv2.calcOpticalFlowFarneback(
                previous_luma_u8,
                luma_u8,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            # Optimized OpenCV kernels can differ by a few float32 ULPs across
            # processes.  Quantize the canonical vector before either analysis
            # or remapping consumes it.  A 1/256-pixel grid is substantially
            # finer than OpenCV remap's interpolation table while preventing
            # sub-ULP provider variance from crossing the final RGB8 rounding
            # boundary in deterministic Offline Master renders.
            flow_fixed = np.rint(
                flow.astype(np.float32) * np.float32(256.0)
            ).astype(np.int32)
            flow_vector = _as_contiguous_float32(
                flow_fixed.astype(np.float32) / np.float32(256.0)
            )
            # ``cv2.magnitude`` itself may vary by a few ULPs with optimized
            # worker scheduling.  Derive magnitude from the fixed-point vector
            # with NumPy float64 arithmetic so the analysis masks do not regain
            # provider variance after the vector has been canonicalized.
            flow_x = flow_fixed[..., 0].astype(np.float64)
            flow_y = flow_fixed[..., 1].astype(np.float64)
            raw_flow_magnitude = np.sqrt(flow_x * flow_x + flow_y * flow_y) / 256.0
            # Quantize the normalized analysis value too, so spatial masks and
            # telemetry share the same reproducible flow evidence.
            normalized_flow = np.clip(
                raw_flow_magnitude / self.config.flow_scale_pixels, 0.0, 1.0
            )
            normalized_flow = (
                np.rint(normalized_flow * 100000.0).astype(np.float32)
                / np.float32(100000.0)
            )
            flow_magnitude = _as_contiguous_float32(normalized_flow)

        entropy = _local_entropy(
            luma_u8,
            window=self.config.entropy_window,
            bins=self.config.entropy_bins,
        )
        local_base = cv2.GaussianBlur(
            luminance,
            (0, 0),
            sigmaX=1.25,
            sigmaY=1.25,
            borderType=cv2.BORDER_REFLECT_101,
        )
        spectrum_map = _as_contiguous_float32(
            np.clip(
                np.abs(luminance - local_base) / self.config.spectrum_local_scale,
                0.0,
                1.0,
            )
        )
        spectrum_scalar = _spectrum_energy(
            luminance,
            cutoff=self.config.spectrum_high_frequency_cutoff,
        )

        motion_evidence = np.clip(flow_magnitude * 0.62 + difference * 0.58, 0.0, 1.0)
        motion_mask = _as_contiguous_float32(
            _smoothstep(
                motion_evidence,
                self.config.motion_low,
                self.config.motion_high,
            )
        )
        edge_mask = _as_contiguous_float32(
            _smoothstep(edge_strength, self.config.edge_low, self.config.edge_high)
        )

        audio_low, audio_mid, audio_high, audio_energy = _band_means(audio_fft_bands)
        transient_value = _clamp01(transient)
        beat_value = _clamp01(float(beat_state))

        # Uniflora-style selection is deliberately spatial: rhythmic scalars
        # excite locally meaningful motion, contour, entropy, and spectral
        # structures rather than becoming a globally uniform intensity knob.
        luminance_structure = np.abs(luminance - 0.5) * 2.0
        intersection = np.sqrt(np.clip(motion_mask * edge_mask, 0.0, 1.0))
        image_drive = (
            motion_mask * 0.25
            + difference * 0.12
            + edge_mask * 0.19
            + entropy * 0.14
            + spectrum_map * 0.11
            + luminance_structure * 0.07
            + intersection * 0.12
        )
        band_drive = audio_low * 0.45 + audio_mid * 0.35 + audio_high * 0.20
        rhythmic_gain = 0.72 + 0.18 * band_drive + 0.10 * spectrum_scalar
        event_detail = (
            transient_value * (difference * 0.68 + motion_mask * 0.32) * 0.24
            + beat_value * (edge_mask * 0.60 + entropy * 0.40) * 0.16
        )
        modulation_mask = np.clip(image_drive * rhythmic_gain + event_detail, 0.0, 1.0)
        modulation_mask = _as_contiguous_float32(
            np.power(modulation_mask, self.config.modulation_gamma)
        )

        difference_mean = float(np.mean(difference))
        scene_change = difference_mean >= self.config.scene_change_threshold
        telemetry: dict[str, float | int | bool] = {
            "master_width": int(master_shape[1]),
            "master_height": int(master_shape[0]),
            "optical_flow_mean": float(np.mean(flow_magnitude)),
            "optical_flow_peak": float(np.max(flow_magnitude)),
            "frame_difference_mean": difference_mean,
            "edge_strength_mean": float(np.mean(edge_strength)),
            "edge_coverage": float(np.mean(edge_mask > 0.5)),
            "luminance_mean": float(np.mean(luminance)),
            "luminance_std": float(np.std(luminance)),
            "local_entropy_mean": float(np.mean(entropy)),
            "image_spectrum_energy": spectrum_scalar,
            "audio_fft_low": audio_low,
            "audio_fft_mid": audio_mid,
            "audio_fft_high": audio_high,
            "audio_fft_energy": audio_energy,
            "transient": transient_value,
            "beat_state": beat_value,
            "motion_coverage": float(np.mean(motion_mask > 0.5)),
            "modulation_mean": float(np.mean(modulation_mask)),
            "modulation_coverage": float(np.mean(modulation_mask > 0.5)),
            "scene_change_score": difference_mean,
            "scene_change": scene_change,
        }
        result = MultimodalAnalysis(
            master_shape=master_shape,
            optical_flow_vector=flow_vector,
            optical_flow_magnitude=flow_magnitude,
            frame_difference=difference,
            edge_strength=edge_strength,
            luminance=luminance,
            local_entropy=entropy,
            spectrum_energy_map=spectrum_map,
            motion_mask=motion_mask,
            edge_mask=edge_mask,
            modulation_mask=modulation_mask,
            telemetry=telemetry,
        )
        result.assert_master_integrity()
        return result


def apply_master_optical_flow_remap(
    master_frame: np.ndarray,
    analysis: MultimodalAnalysis,
    *,
    strength: float = 0.65,
    displacement_cap_pixels: float = 24.0,
    direction: str = "backward",
    border_mode: str = "reflect",
) -> np.ndarray:
    """Remap one master frame with the canonical full-resolution flow field.

    This is a provider-independent correctness path. It neither owns history
    nor changes resolution; CUDA/GL implementations may replace the kernel only
    if they consume the same field and preserve these semantics.
    """
    frame = _validate_master_frame(
        master_frame,
        expected_shape=analysis.master_shape,
        name="optical-flow remap master frame",
    )
    analysis.assert_master_integrity()
    amount = float(np.clip(strength, 0.0, 2.0))
    if amount <= 0.0:
        return frame.copy()
    cap = max(0.0, float(displacement_cap_pixels))
    flow = analysis.optical_flow_vector.astype(np.float32, copy=True)
    if cap > 0.0:
        flow_x64 = flow[..., 0].astype(np.float64)
        flow_y64 = flow[..., 1].astype(np.float64)
        magnitude = np.sqrt(flow_x64 * flow_x64 + flow_y64 * flow_y64)
        scale = np.minimum(1.0, cap / np.maximum(magnitude, 1e-6)).astype(
            np.float32
        )
        flow *= scale[..., None]
    sign = -1.0 if str(direction).lower() == "backward" else 1.0
    height, width = frame.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    map_x = np.ascontiguousarray(xx + sign * flow[..., 0] * amount)
    map_y = np.ascontiguousarray(yy + sign * flow[..., 1] * amount)
    borders = {
        "reflect": cv2.BORDER_REFLECT_101,
        "replicate": cv2.BORDER_REPLICATE,
        "wrap": cv2.BORDER_WRAP,
        "constant": cv2.BORDER_CONSTANT,
    }
    result = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=borders.get(str(border_mode).lower(), cv2.BORDER_REFLECT_101),
    )
    return np.ascontiguousarray(result, dtype=np.uint8)


@dataclass(frozen=True)
class ControlledFeedbackConfig:
    """Hard safety bounds for recursive master-path feedback."""

    opacity: float = 0.42
    opacity_cap: float = 0.68
    decay: float = 0.92
    clean_source_floor: float = 0.32
    motion_suppression: float = 0.72
    edge_protection: float = 0.28
    scene_change_threshold: float = 0.32

    def __post_init__(self) -> None:
        for name in (
            "opacity",
            "opacity_cap",
            "decay",
            "clean_source_floor",
            "motion_suppression",
            "edge_protection",
            "scene_change_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class ControlledFeedbackTelemetry:
    requested_opacity: float
    bounded_opacity: float
    mean_effective_opacity: float
    peak_effective_opacity: float
    clean_source_floor: float
    scene_reset: bool
    initialized: bool


@dataclass(frozen=True)
class ControlledFeedbackResult:
    frame: np.ndarray
    telemetry: ControlledFeedbackTelemetry


class ControlledRecursiveFeedback:
    """Bounded recursive feedback with master-resolution state ownership."""

    def __init__(self, config: ControlledFeedbackConfig | None = None) -> None:
        self.config = config or ControlledFeedbackConfig()
        self._master_shape: tuple[int, int, int] | None = None
        self._previous_output: np.ndarray | None = None

    @property
    def master_shape(self) -> tuple[int, int, int] | None:
        return self._master_shape

    @property
    def has_history(self) -> bool:
        return self._previous_output is not None

    def reset(self) -> None:
        """Drop recursive history and permit an explicit ingress resolution change."""

        self._master_shape = None
        self._previous_output = None

    def process(
        self,
        clean_master_frame: np.ndarray,
        analysis: MultimodalAnalysis,
        *,
        enabled: bool = True,
    ) -> ControlledFeedbackResult:
        frame = _validate_master_frame(
            clean_master_frame,
            expected_shape=self._master_shape,
            name="clean master frame",
        )
        if self._master_shape is None:
            self._master_shape = tuple(frame.shape)
        if tuple(analysis.master_shape) != self._master_shape:
            raise MasterResolutionError(
                f"analysis master shape {analysis.master_shape} does not match feedback "
                f"master shape {self._master_shape}"
            )
        analysis.assert_master_integrity()

        requested = _clamp01(self.config.opacity)
        bounded = min(
            requested,
            _clamp01(self.config.opacity_cap),
            1.0 - _clamp01(self.config.clean_source_floor),
        ) * _clamp01(self.config.decay)
        scene_score = float(analysis.telemetry.get("scene_change_score", 0.0))
        scene_reset = scene_score >= self.config.scene_change_threshold
        initialized = self._previous_output is not None

        # A bypass/reset returns the exact input object and pixels.  History is
        # reseeded from the clean master so re-enabling cannot resurrect stale
        # content from before a bypass or hard cut.
        if not enabled or bounded <= 0.0 or not initialized or scene_reset:
            self._previous_output = frame.copy()
            return ControlledFeedbackResult(
                frame=frame,
                telemetry=ControlledFeedbackTelemetry(
                    requested_opacity=requested,
                    bounded_opacity=bounded,
                    mean_effective_opacity=0.0,
                    peak_effective_opacity=0.0,
                    clean_source_floor=self.config.clean_source_floor,
                    scene_reset=scene_reset,
                    initialized=initialized,
                ),
            )

        motion_gate = 1.0 - analysis.motion_mask * self.config.motion_suppression
        edge_gate = 1.0 - analysis.edge_mask * self.config.edge_protection
        opacity_map = np.clip(
            analysis.modulation_mask * motion_gate * edge_gate * bounded,
            0.0,
            bounded,
        ).astype(np.float32)
        opacity_3 = opacity_map[..., None]
        output = np.rint(
            frame.astype(np.float32) * (1.0 - opacity_3)
            + self._previous_output.astype(np.float32) * opacity_3
        )
        output = np.ascontiguousarray(np.clip(output, 0.0, 255.0).astype(np.uint8))
        self._previous_output = output.copy()
        return ControlledFeedbackResult(
            frame=output,
            telemetry=ControlledFeedbackTelemetry(
                requested_opacity=requested,
                bounded_opacity=bounded,
                mean_effective_opacity=float(np.mean(opacity_map)),
                peak_effective_opacity=float(np.max(opacity_map)),
                clean_source_floor=self.config.clean_source_floor,
                scene_reset=False,
                initialized=True,
            ),
        )


@dataclass(frozen=True)
class MultimodalFeedbackFrame:
    frame: np.ndarray
    analysis: MultimodalAnalysis
    feedback: ControlledFeedbackTelemetry


class MasterMultimodalFeedbackStage:
    """Convenience owner for consecutive-frame analysis and feedback state."""

    def __init__(
        self,
        *,
        analysis_config: MultimodalAnalysisConfig | None = None,
        feedback_config: ControlledFeedbackConfig | None = None,
    ) -> None:
        self.analyzer = MasterMultimodalAnalyzer(analysis_config)
        self.feedback = ControlledRecursiveFeedback(feedback_config)
        self._previous_source: np.ndarray | None = None

    @property
    def master_shape(self) -> tuple[int, int, int] | None:
        return self.analyzer.master_shape

    def reset(self) -> None:
        self.analyzer.reset()
        self.feedback.reset()
        self._previous_source = None

    def process(
        self,
        master_frame: np.ndarray,
        *,
        audio_fft_bands: Mapping[str, float]
        | Sequence[float]
        | np.ndarray
        | None = None,
        transient: float = 0.0,
        beat_state: float | bool = 0.0,
        feedback_enabled: bool = True,
    ) -> MultimodalFeedbackFrame:
        analysis = self.analyzer.analyze(
            master_frame,
            self._previous_source,
            audio_fft_bands=audio_fft_bands,
            transient=transient,
            beat_state=beat_state,
        )
        feedback = self.feedback.process(
            master_frame,
            analysis,
            enabled=feedback_enabled,
        )
        self._previous_source = master_frame.copy()
        return MultimodalFeedbackFrame(
            frame=feedback.frame,
            analysis=analysis,
            feedback=feedback.telemetry,
        )


__all__ = [
    "ControlledFeedbackConfig",
    "ControlledFeedbackResult",
    "ControlledFeedbackTelemetry",
    "ControlledRecursiveFeedback",
    "MasterMultimodalAnalyzer",
    "MasterMultimodalFeedbackStage",
    "MasterResolutionError",
    "MultimodalAnalysis",
    "MultimodalAnalysisConfig",
    "MultimodalFeedbackFrame",
]
