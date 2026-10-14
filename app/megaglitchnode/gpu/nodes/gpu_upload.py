"""
GPU Upload Node

Transfers frame from CPU to GPU memory.
"""

import cv2
import logging

from .base import NodeBase
from ..frame import Frame
from ..context import Context

logger = logging.getLogger(__name__)


class GpuUpload(NodeBase):
    """
    Upload node: Transfers CPU frame data to GPU.
    
    If frame already has GPU data, returns as-is.
    """
    
    def __init__(self, node_id: str = "gpu_upload"):
        """Initialize upload node."""
        super().__init__(node_id, prefers_gpu=True)
    
    def process(self, frame: Frame, ctx: Context) -> Frame:
        """
        Ensure frame is on GPU.
        
        Args:
            frame: Input frame
            ctx: GPU context
        
        Returns:
            Frame with GPU data
        """
        # If already on GPU, nothing to do
        if frame.gpu is not None:
            return frame
        
        # Upload to GPU
        if frame.cpu is None:
            raise RuntimeError(f"{self.id}: Cannot upload - no CPU data available")
        
        frame.ensure_gpu(ctx)
        
        return frame

