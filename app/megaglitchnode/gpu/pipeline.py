"""
GPU Pipeline

Orchestrates execution of GPU node chains.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional

from .frame import Frame
from .context import Context
from .nodes.base import NodeBase
from .nodes.gpu_upload import GpuUpload
from .nodes.gpu_warp_remap import GpuWarpRemap
from .nodes.gpu_feedback_mixer import GpuFeedbackMixer
from .nodes.gpu_bloom_ghost import GpuBloomGhost

logger = logging.getLogger(__name__)


class Pipeline:
    """
    GPU processing pipeline that chains nodes together.
    
    Attributes:
        nodes: List of nodes to execute in order
        context: GPU execution context
    """
    
    def __init__(self, nodes: Optional[list[NodeBase]] = None, use_stream: bool = False):
        """
        Initialize pipeline.
        
        Args:
            nodes: List of nodes (or None for empty pipeline)
            use_stream: Whether to use CUDA stream
        """
        self.nodes = nodes if nodes is not None else []
        self.context = Context(use_stream=use_stream)
    
    def add_node(self, node: NodeBase):
        """Add a node to the pipeline."""
        self.nodes.append(node)
    
    def clear(self):
        """Remove all nodes."""
        self.nodes.clear()
    
    def process(self, frame: Frame) -> Frame:
        """
        Process a frame through the pipeline.
        
        Args:
            frame: Input frame
        
        Returns:
            Processed frame
        """
        for node in self.nodes:
            try:
                frame = node.process(frame, self.context)
            except Exception as e:
                logger.error(f"Node {node.id} failed: {e}", exc_info=True)
                # Continue with unprocessed frame
        
        # Advance frame counter
        self.context.advance_frame()
        
        return frame
    
    def process_numpy(self, img_array: np.ndarray) -> np.ndarray:
        """
        Convenience method: process numpy array and return numpy array.
        
        Args:
            img_array: Input image as numpy array (H, W, 3) uint8
        
        Returns:
            Processed image as numpy array
        """
        # Convert to Frame
        frame = Frame.from_numpy(img_array, fmt="bgr8")
        
        # Process
        frame = self.process(frame)
        
        # Convert back to numpy
        return frame.ensure_cpu(self.context)
    
    def benchmark(self, num_frames: int = 100, width: int = 1280, height: int = 720) -> dict:
        """
        Benchmark pipeline performance.
        
        Args:
            num_frames: Number of frames to process
            width: Frame width
            height: Frame height
        
        Returns:
            Dict with performance stats
        """
        logger.info(f"Starting benchmark: {num_frames} frames at {width}x{height}")
        
        # Create test frame
        test_frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(5):
            _ = self.process_numpy(test_frame)
        
        # Benchmark
        start_time = time.perf_counter()
        
        for i in range(num_frames):
            _ = self.process_numpy(test_frame)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # Calculate stats
        fps = num_frames / elapsed
        ms_per_frame = (elapsed / num_frames) * 1000
        
        stats = {
            "num_frames": num_frames,
            "resolution": f"{width}x{height}",
            "elapsed_sec": elapsed,
            "fps": fps,
            "ms_per_frame": ms_per_frame,
            "nodes": [node.id for node in self.nodes]
        }
        
        logger.info(f"Benchmark complete: {fps:.1f} FPS ({ms_per_frame:.2f} ms/frame)")
        
        return stats
    
    def reset(self):
        """Reset pipeline state (clear caches, reset frame counter)."""
        self.context.clear_cache()
        self.context.reset_frame_index()
    
    def __repr__(self) -> str:
        return f"Pipeline(nodes={len(self.nodes)}, frame={self.context.frame_index})"


def create_default_pipeline(use_stream: bool = False, include_bloom: bool = False) -> Pipeline:
    """
    Create default GPU pipeline: Upload -> Warp -> Feedback (-> Bloom).
    
    Args:
        use_stream: Whether to use CUDA stream
        include_bloom: Whether to include bloom effect (requires cudafilters module)
    
    Returns:
        Configured pipeline
    """
    pipeline = Pipeline(use_stream=use_stream)
    
    # Add nodes in order
    pipeline.add_node(GpuUpload("upload"))
    
    warp = GpuWarpRemap("warp")
    warp.set_params(mode="swirl", strength=0.002, animate=True)
    pipeline.add_node(warp)
    
    feedback = GpuFeedbackMixer("feedback")
    feedback.set_params(decay=0.85, mix=0.3, warp_prev=False)
    pipeline.add_node(feedback)
    
    if include_bloom:
        bloom = GpuBloomGhost("bloom")
        bloom.set_params(ksize=15, sigma=4.0, amount=0.35)
        pipeline.add_node(bloom)
    
    return pipeline

