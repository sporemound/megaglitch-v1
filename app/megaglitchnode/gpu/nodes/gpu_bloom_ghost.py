"""
GPU Bloom Ghost Node

Applies Gaussian bloom effect using GPU-accelerated filtering.
"""

import cv2
import logging

from .base import NodeBase
from ..frame import Frame
from ..context import Context

logger = logging.getLogger(__name__)


class GpuBloomGhost(NodeBase):
    """
    Bloom Ghost node: Applies Gaussian blur and blends with original (bloom effect).
    
    Parameters:
        ksize: int, kernel size for Gaussian blur (must be odd, default: 21)
        sigma: float, Gaussian sigma (default: 5.0)
        amount: float, bloom intensity [0,1] (default: 0.35)
    """
    
    def __init__(self, node_id: str = "gpu_bloom_ghost"):
        """Initialize bloom ghost node."""
        super().__init__(node_id, prefers_gpu=True)
        
        # Default parameters
        self.set_params(
            ksize=21,
            sigma=5.0,
            amount=0.35
        )
    
    def process(self, frame: Frame, ctx: Context) -> Frame:
        """
        Apply bloom effect to frame.
        
        Args:
            frame: Input frame
            ctx: GPU context
        
        Returns:
            Frame with bloom effect
        """
        # Ensure frame is on GPU
        frame.ensure_gpu(ctx)
        
        # Get parameters
        ksize = self.get_param("ksize", 21)
        sigma = self.get_param("sigma", 5.0)
        amount = self.get_param("amount", 0.35)
        
        # Ensure ksize is odd and reasonable
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, min(ksize, 31))  # Clamp to reasonable range
        
        # Get cache
        cache = ctx.get_cache(self.id)
        
        # Create cache key for filter
        filter_key = f"filter_{ksize}_{sigma:.2f}_{frame.gpu.type()}"
        
        # Check if filter is cached
        if filter_key not in cache:
            # Create Gaussian filter
            src_type = frame.gpu.type()
            
            try:
                gauss_filter = cv2.cuda.createGaussianFilter(
                    src_type, src_type,
                    (ksize, ksize),
                    sigma
                )
                cache[filter_key] = gauss_filter
            except Exception as e:
                logger.error(f"{self.id}: Failed to create Gaussian filter: {e}")
                # Return original frame on error
                return frame
        
        gauss_filter = cache[filter_key]
        
        # Apply Gaussian blur on GPU
        try:
            blurred = gauss_filter.apply(frame.gpu)
            
            # Check if blurred is valid
            if blurred is None or blurred.empty():
                logger.error(f"{self.id}: Gaussian filter returned empty result")
                return frame
            
        except Exception as e:
            logger.error(f"{self.id}: Failed to apply Gaussian filter: {e}")
            return frame
        
        # Blend original with blurred (bloom effect)
        # output = original * 1.0 + blurred * amount
        try:
            bloomed = cv2.cuda.addWeighted(frame.gpu, 1.0, blurred, amount, 0.0)
            
            # Check if blend succeeded
            if bloomed is None or bloomed.empty():
                logger.error(f"{self.id}: addWeighted returned empty result")
                return frame
                
        except Exception as e:
            logger.error(f"{self.id}: Failed to blend bloom: {e}")
            return frame
        
        # Return bloomed frame
        return Frame.from_gpu_mat(bloomed, frame.fmt)

