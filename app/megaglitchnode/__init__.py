"""MegaGlitch GPU package without pre-bootstrap OpenCV side effects."""

from __future__ import annotations

from .opencv_cuda_bootstrap import (
    bootstrap_opencv_cuda,
    bootstrap_for_dev_defaults,
    is_cuda_available,
)


_GPU_EXPORTS = {
    "Frame": ("megaglitchnode.gpu", "Frame"),
    "Context": ("megaglitchnode.gpu", "Context"),
    "Pipeline": ("megaglitchnode.gpu", "Pipeline"),
    "create_default_pipeline": ("megaglitchnode.gpu", "create_default_pipeline"),
    "NodeBase": ("megaglitchnode.gpu.nodes", "NodeBase"),
    "GpuUpload": ("megaglitchnode.gpu.nodes", "GpuUpload"),
    "GpuWarpRemap": ("megaglitchnode.gpu.nodes", "GpuWarpRemap"),
    "GpuFeedbackMixer": ("megaglitchnode.gpu.nodes", "GpuFeedbackMixer"),
    "GpuBloomGhost": ("megaglitchnode.gpu.nodes", "GpuBloomGhost"),
}


def __getattr__(name):
    """Load GPU symbols only after callers have had a chance to bootstrap cv2."""
    target = _GPU_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value


__all__ = [
    "bootstrap_opencv_cuda",
    "bootstrap_for_dev_defaults",
    "is_cuda_available",
    *_GPU_EXPORTS,
]

__version__ = "0.6.0"
