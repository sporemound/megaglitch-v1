"""
GPU Context

Holds GPU execution context including stream, node caches, and deterministic RNG.
"""

import cv2
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class Context:
    """
    GPU execution context for node processing.
    
    Attributes:
        stream: cv2.cuda_Stream for async operations (or None)
        cache: Dict for node-specific persistent state
        frame_index: Current frame number (for temporal effects)
        rng_seed: Base random seed (for deterministic behavior)
        cache_epoch: Current cache epoch (for invalidation detection)
    """
    
    def __init__(self, use_stream: bool = False, frame_index: int = 0, rng_seed: int = 42):
        """
        Initialize GPU context.
        
        Args:
            use_stream: Whether to use CUDA stream for async operations
            frame_index: Starting frame index
            rng_seed: Base random seed for determinism
        """
        self.stream: Optional[cv2.cuda_Stream] = None
        self.cache: dict[str, dict[str, Any]] = {}
        self.frame_index = frame_index
        self.rng_seed = rng_seed
        self._cache_epoch = 0
        
        # Get initial cache epoch from bootstrap
        try:
            from megaglitchnode.opencv_cuda_bootstrap import get_cache_epoch
            self._cache_epoch = get_cache_epoch()
        except ImportError:
            pass
        
        # Create stream if requested and CUDA is available
        if use_stream:
            try:
                self.stream = cv2.cuda_Stream()
            except Exception as e:
                logger.debug(f"Failed to create CUDA stream: {e}")
                self.stream = None
    
    def get_cache(self, node_id: str) -> dict[str, Any]:
        """
        Get or create cache dictionary for a specific node.
        Checks cache epoch to detect invalidation from CUDA/OpenCV updates.
        
        Args:
            node_id: Unique node identifier
        
        Returns:
            Dict for storing node-specific state
        """
        # Check if cache epoch changed (CUDA was updated)
        try:
            from megaglitchnode.opencv_cuda_bootstrap import get_cache_epoch
            current_epoch = get_cache_epoch()
            if current_epoch != self._cache_epoch:
                logger.info(f"Cache epoch changed ({self._cache_epoch} -> {current_epoch}), clearing all GPU caches")
                self.cache.clear()
                self._cache_epoch = current_epoch
        except ImportError:
            pass
        
        if node_id not in self.cache:
            self.cache[node_id] = {}
        return self.cache[node_id]
    
    def seed_for(self, node_id: str, extra: int = 0) -> int:
        """
        Generate deterministic seed for a node.
        
        Combines base seed, node ID hash, frame index, and extra value
        to produce reproducible random numbers.
        
        Args:
            node_id: Unique node identifier
            extra: Additional value to mix into seed
        
        Returns:
            Deterministic seed value
        """
        # Hash node_id to get consistent value
        node_hash = hash(node_id) & 0x7FFFFFFF  # Keep positive
        
        # Combine all factors
        seed = (self.rng_seed + node_hash + self.frame_index + extra) & 0x7FFFFFFF
        return seed
    
    def advance_frame(self):
        """Increment frame index (call after processing each frame)."""
        self.frame_index += 1
    
    def reset_frame_index(self, index: int = 0):
        """Reset frame index to specific value."""
        self.frame_index = index
    
    def clear_cache(self, node_id: Optional[str] = None):
        """
        Clear node cache.
        
        Args:
            node_id: If provided, clear only this node's cache. Otherwise clear all.
        """
        if node_id:
            if node_id in self.cache:
                self.cache[node_id].clear()
        else:
            self.cache.clear()
    
    def synchronize(self):
        """Synchronize stream (wait for all GPU operations to complete)."""
        if self.stream:
            self.stream.waitForCompletion()
    
    def clear_all_caches(self):
        """Clear all node caches and reset context state.
        
        Called on CUDA DLL update or when GPU state needs to be invalidated.
        """
        self.cache.clear()
        self.frame_index = 0
        logger.info("Cleared all GPU context caches")
    
    def __repr__(self) -> str:
        return f"Context(stream={self.stream is not None}, frame_index={self.frame_index}, cached_nodes={len(self.cache)})"


# Global context for sharing across modules
_global_context: Optional['Context'] = None


def get_global_context() -> 'Context':
    """Get or create global GPU context."""
    global _global_context
    if _global_context is None:
        _global_context = Context(use_stream=False, frame_index=0)
    return _global_context


def reset_global_context():
    """Reset global GPU context (clears all caches)."""
    global _global_context
    if _global_context is not None:
        _global_context.clear_all_caches()
    _global_context = None
    logger.info("Reset global GPU context")

