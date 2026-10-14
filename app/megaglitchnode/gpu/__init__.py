"""
MegaGlitchNode GPU Package

GPU-accelerated image processing using OpenCV CUDA.
"""

from .frame import Frame
from .context import Context
from .pipeline import Pipeline, create_default_pipeline

__all__ = [
    'Frame',
    'Context',
    'Pipeline',
    'create_default_pipeline'
]



