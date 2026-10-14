"""
GPU Warp Remap Node

Applies geometric warping effects using cv2.cuda.remap.
"""

import cv2
import numpy as np
import logging
import hashlib

from .base import NodeBase
from ..frame import Frame
from ..context import Context

logger = logging.getLogger(__name__)


class GpuWarpRemap(NodeBase):
    """
    Warp Remap node: Applies geometric distortions (swirl, ripple, blockshift).
    
    Parameters:
        mode: "swirl" | "ripple" | "blockshift" (default: "swirl")
        strength: float, distortion intensity (default: 0.001)
        block_size: int, for blockshift mode (default: 32)
        border_mode: "reflect" | "replicate" | "constant" (default: "reflect")
        interpolation: "linear" | "nearest" (default: "linear")
        animate: bool, vary effect over time (default: True)
    """
    
    def __init__(self, node_id: str = "gpu_warp_remap"):
        """Initialize warp remap node."""
        super().__init__(node_id, prefers_gpu=True)
        
        # Default parameters
        self.set_params(
            mode="swirl",
            strength=0.001,
            block_size=32,
            border_mode="reflect",
            interpolation="linear",
            animate=True
        )
    
    def _build_swirl_map(self, w: int, h: int, strength: float, phase: float) -> tuple[np.ndarray, np.ndarray]:
        """Build swirl distortion maps."""
        cx, cy = w / 2, h / 2
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Distance from center
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx*dx + dy*dy)
        
        # Swirl angle based on distance
        angle = strength * r + phase
        
        # Apply rotation
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        map_x = cx + dx * cos_a - dy * sin_a
        map_y = cy + dx * sin_a + dy * cos_a
        
        return map_x, map_y
    
    def _build_ripple_map(self, w: int, h: int, strength: float, phase: float) -> tuple[np.ndarray, np.ndarray]:
        """Build ripple distortion maps."""
        cx, cy = w / 2, h / 2
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Distance from center
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx*dx + dy*dy)
        
        # Ripple displacement
        ripple = np.sin(r * 0.05 + phase) * strength * 50
        
        # Apply displacement along radius
        factor = 1.0 + ripple / (r + 1)
        map_x = cx + dx * factor
        map_y = cy + dy * factor
        
        return map_x, map_y
    
    def _build_blockshift_map(self, w: int, h: int, block_size: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
        """Build block shift distortion maps."""
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Divide into blocks
        block_y = (y // block_size).astype(int)
        block_x = (x // block_size).astype(int)
        
        # Generate pseudo-random offsets per block (deterministic based on integer phase)
        np.random.seed(int(phase) & 0x7FFFFFFF)
        max_offset = block_size // 4
        
        # Create unique offsets for each block
        offset_x = np.zeros_like(x)
        offset_y = np.zeros_like(y)
        
        for by in range(0, h, block_size):
            for bx in range(0, w, block_size):
                ox = np.random.randint(-max_offset, max_offset + 1)
                oy = np.random.randint(-max_offset, max_offset + 1)
                
                mask = (y >= by) & (y < by + block_size) & (x >= bx) & (x < bx + block_size)
                offset_x[mask] = ox
                offset_y[mask] = oy
        
        map_x = x + offset_x
        map_y = y + offset_y
        
        return map_x, map_y
    
    def _get_cache_key(self, w: int, h: int, mode: str, strength: float, block_size: int, phase: float) -> str:
        """Generate cache key for maps. Round phase to 2 decimal places for cache efficiency."""
        # Round phase to reduce unique cache entries while maintaining smooth animation
        phase_rounded = round(phase, 2)
        key_str = f"{w}x{h}_{mode}_{strength}_{block_size}_{phase_rounded:.2f}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def process(self, frame: Frame, ctx: Context) -> Frame:
        """
        Apply warp effect to frame.
        
        Args:
            frame: Input frame
            ctx: GPU context
        
        Returns:
            Warped frame on GPU
        """
        # Ensure frame is on GPU
        frame.ensure_gpu(ctx)
        
        # Get parameters
        mode = self.get_param("mode", "swirl")
        strength = self.get_param("strength", 0.001)
        block_size = self.get_param("block_size", 32)
        border_mode = self.get_param("border_mode", "reflect")
        interpolation = self.get_param("interpolation", "linear")
        animate = self.get_param("animate", True)
        
        # Calculate phase for animation
        if animate:
            # Use frame index for deterministic, smooth animation
            # Scale by 0.05 to create smooth but visible motion (full cycle every ~125 frames or ~2 seconds at 60fps)
            phase = float(ctx.frame_index) * 0.05
        else:
            phase = 0.0
        
        # Get cache
        cache = ctx.get_cache(self.id)
        cache_key = self._get_cache_key(frame.width, frame.height, mode, strength, block_size, phase)
        
        # Check if maps are cached
        if cache_key in cache:
            map_x_gpu, map_y_gpu = cache[cache_key]
        else:
            # Build maps on CPU
            if mode == "swirl":
                map_x, map_y = self._build_swirl_map(frame.width, frame.height, strength, phase)
            elif mode == "ripple":
                map_x, map_y = self._build_ripple_map(frame.width, frame.height, strength, phase)
            elif mode == "blockshift":
                map_x, map_y = self._build_blockshift_map(frame.width, frame.height, block_size, phase)
            else:
                logger.warning(f"{self.id}: Unknown mode '{mode}', using identity map")
                y, x = np.mgrid[0:frame.height, 0:frame.width].astype(np.float32)
                map_x, map_y = x, y
            
            # Upload maps to GPU
            map_x_gpu = cv2.cuda_GpuMat()
            map_y_gpu = cv2.cuda_GpuMat()
            map_x_gpu.upload(map_x)
            map_y_gpu.upload(map_y)
            
            # Cache for reuse (increased cache size for smoother playback)
            if len(cache) > 100:
                # Clear oldest entries (keep last 50)
                keys_to_remove = list(cache.keys())[:-50]
                for key in keys_to_remove:
                    del cache[key]
            
            cache[cache_key] = (map_x_gpu, map_y_gpu)
        
        # Map border mode
        border_dict = {
            "reflect": cv2.BORDER_REFLECT_101,
            "replicate": cv2.BORDER_REPLICATE,
            "constant": cv2.BORDER_CONSTANT
        }
        border_flag = border_dict.get(border_mode, cv2.BORDER_REFLECT_101)
        
        # Map interpolation
        interp_dict = {
            "linear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST
        }
        interp_flag = interp_dict.get(interpolation, cv2.INTER_LINEAR)
        
        # Apply remap on GPU
        # cv2.cuda.remap signature: remap(src, xmap, ymap, interpolation[, dst[, borderMode[, borderValue[, stream]]]]) -> dst
        warped_gpu = cv2.cuda.remap(frame.gpu, map_x_gpu, map_y_gpu, interp_flag, borderMode=border_flag)
        
        # Return new frame with warped data
        return Frame.from_gpu_mat(warped_gpu, frame.fmt)

