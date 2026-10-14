"""
GPU Feedback Mixer Node

Creates persistent feedback trails by blending current frame with previous frame.
"""

import cv2
import numpy as np
import logging

from .base import NodeBase
from ..frame import Frame
from ..context import Context

logger = logging.getLogger(__name__)


class GpuFeedbackMixer(NodeBase):
    """
    Feedback Mixer node: Blends current frame with persistent feedback buffer.
    
    Parameters:
        decay: float [0,1], feedback persistence (default: 0.85, higher = longer trails)
        mix: float [0,1], current frame weight (default: 0.25)
        warp_prev: bool, apply warp to feedback before mixing (default: False)
        warp_strength: float, warp intensity if warp_prev=True (default: 0.0005)
    """
    
    def __init__(self, node_id: str = "gpu_feedback_mixer"):
        """Initialize feedback mixer node."""
        super().__init__(node_id, prefers_gpu=True)
        
        # Default parameters
        self.set_params(
            decay=0.85,
            mix=0.25,
            warp_prev=False,
            warp_strength=0.0005
        )
    
    def _build_light_swirl_map(self, w: int, h: int, strength: float) -> tuple[np.ndarray, np.ndarray]:
        """Build a light swirl map for feedback warping."""
        cx, cy = w / 2, h / 2
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx*dx + dy*dy)
        
        angle = strength * r
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        map_x = cx + dx * cos_a - dy * sin_a
        map_y = cy + dx * sin_a + dy * cos_a
        
        return map_x, map_y
    
    def process(self, frame: Frame, ctx: Context) -> Frame:
        """
        Mix current frame with persistent feedback.
        
        Args:
            frame: Input frame
            ctx: GPU context
        
        Returns:
            Frame with feedback trails
        """
        # Ensure frame is on GPU
        frame.ensure_gpu(ctx)
        
        # Get parameters
        decay = self.get_param("decay", 0.85)
        mix = self.get_param("mix", 0.25)
        warp_prev = self.get_param("warp_prev", False)
        warp_strength = self.get_param("warp_strength", 0.0005)
        
        # Get cache
        cache = ctx.get_cache(self.id)
        
        # Initialize feedback buffer if needed
        if "prev" not in cache or cache["prev"] is None:
            # First frame - clone current as feedback
            cache["prev"] = frame.gpu.clone()
            return frame
        
        prev_gpu = cache["prev"]
        
        # Check if dimensions match
        prev_size = prev_gpu.size()
        curr_size = frame.gpu.size()
        if prev_size != curr_size:
            # Resolution changed - reinitialize
            cache["prev"] = frame.gpu.clone()
            return frame
        
        # Optionally warp previous frame
        if warp_prev:
            # Check if warp maps are cached
            if "warp_maps" not in cache:
                # Build and upload warp maps
                map_x, map_y = self._build_light_swirl_map(frame.width, frame.height, warp_strength)
                map_x_gpu = cv2.cuda_GpuMat()
                map_y_gpu = cv2.cuda_GpuMat()
                map_x_gpu.upload(map_x)
                map_y_gpu.upload(map_y)
                cache["warp_maps"] = (map_x_gpu, map_y_gpu)
            
            map_x_gpu, map_y_gpu = cache["warp_maps"]
            prev_gpu = cv2.cuda.remap(prev_gpu, map_x_gpu, map_y_gpu, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        
        # Blend: prev * decay + current * (1 - decay)
        # Then mix with original current frame
        blended = cv2.cuda.addWeighted(prev_gpu, decay, frame.gpu, 1.0 - decay, 0.0)
        
        # Mix blended result with current frame
        output = cv2.cuda.addWeighted(blended, 1.0 - mix, frame.gpu, mix, 0.0)
        
        # Store blended as new prev
        cache["prev"] = blended
        
        # Return mixed output
        return Frame.from_gpu_mat(output, frame.fmt)

