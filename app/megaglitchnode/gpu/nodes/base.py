"""
GPU Node Base Class

Abstract base for GPU-accelerated nodes that process frames.
"""

from abc import ABC, abstractmethod
from typing import Any
import logging

from ..frame import Frame
from ..context import Context

logger = logging.getLogger(__name__)


class NodeBase(ABC):
    """
    Abstract base class for GPU nodes.
    
    All GPU nodes inherit from this and implement the process() method.
    """
    
    def __init__(self, node_id: str, prefers_gpu: bool = True):
        """
        Initialize node.
        
        Args:
            node_id: Unique identifier for this node instance
            prefers_gpu: Whether node prefers GPU processing
        """
        self.id = node_id
        self.prefers_gpu = prefers_gpu
        self.params: dict[str, Any] = {}
    
    def set_param(self, name: str, value: Any):
        """
        Set a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
        """
        self.params[name] = value
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value with optional default.
        
        Args:
            name: Parameter name
            default: Default value if parameter not set
        
        Returns:
            Parameter value or default
        """
        return self.params.get(name, default)
    
    def set_params(self, **kwargs):
        """
        Set multiple parameters at once.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        self.params.update(kwargs)
    
    @abstractmethod
    def process(self, frame: Frame, ctx: Context) -> Frame:
        """
        Process a frame.
        
        Args:
            frame: Input frame (may be on CPU or GPU)
            ctx: GPU context with stream, cache, etc.
        
        Returns:
            Processed frame
        
        Notes:
            - Implementations can access persistent state via ctx.cache[self.id]
            - Use ctx.frame_index for temporal effects
            - Use ctx.seed_for(self.id) for deterministic randomness
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, prefers_gpu={self.prefers_gpu})"



