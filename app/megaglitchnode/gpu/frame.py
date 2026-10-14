"""
GPU Frame Container

Lightweight container that holds either CPU (numpy) or GPU (cv2.cuda_GpuMat) image data.
Manages transfers between CPU and GPU to avoid redundant operations.
"""

import cv2
import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context


class Frame:
    """
    Container for image data that can reside on CPU or GPU.
    
    Attributes:
        cpu: numpy.ndarray (H, W, C) or None
        gpu: cv2.cuda_GpuMat or None
        fmt: Format string (e.g., "bgr8", "rgb8")
        width: Image width in pixels
        height: Image height in pixels
    """
    
    def __init__(self, cpu: Optional[np.ndarray] = None, gpu: Optional[cv2.cuda_GpuMat] = None, fmt: str = "bgr8"):
        """
        Initialize frame with either CPU or GPU data.
        
        Args:
            cpu: Numpy array (H, W, C) uint8
            gpu: cv2.cuda_GpuMat
            fmt: Format string ("bgr8" or "rgb8")
        
        Raises:
            ValueError: If both cpu and gpu are None
        """
        if cpu is None and gpu is None:
            raise ValueError("Frame must have either cpu or gpu data")
        
        self.cpu = cpu
        self.gpu = gpu
        self.fmt = fmt
        
        # Determine dimensions
        if cpu is not None:
            self.height, self.width = cpu.shape[:2]
        elif gpu is not None:
            self.width, self.height = gpu.size()
        else:
            self.height = 0
            self.width = 0
    
    @classmethod
    def from_numpy(cls, arr: np.ndarray, fmt: str = "bgr8") -> "Frame":
        """Create frame from numpy array."""
        return cls(cpu=arr, gpu=None, fmt=fmt)
    
    @classmethod
    def from_gpu_mat(cls, gpu_mat: cv2.cuda_GpuMat, fmt: str = "bgr8") -> "Frame":
        """Create frame from GpuMat."""
        return cls(cpu=None, gpu=gpu_mat, fmt=fmt)
    
    def ensure_gpu(self, ctx: "Context") -> cv2.cuda_GpuMat:
        """
        Ensure frame data is on GPU, uploading if necessary.
        
        Args:
            ctx: GPU context (for stream support)
        
        Returns:
            cv2.cuda_GpuMat
        """
        if self.gpu is not None:
            return self.gpu
        
        if self.cpu is None:
            raise RuntimeError("Cannot upload to GPU: no CPU data available")
        
        # Upload to GPU
        self.gpu = cv2.cuda_GpuMat()
        if ctx.stream:
            self.gpu.upload(self.cpu, ctx.stream)
        else:
            self.gpu.upload(self.cpu)
        import gpu_ops

        gpu_ops.record_external_transfer("cpu_to_gpu", int(self.cpu.nbytes))
        
        return self.gpu
    
    def ensure_cpu(self, ctx: "Context") -> np.ndarray:
        """
        Ensure frame data is on CPU, downloading if necessary.
        
        Args:
            ctx: GPU context (for stream support)
        
        Returns:
            numpy.ndarray
        """
        if self.cpu is not None:
            return self.cpu
        
        if self.gpu is None:
            raise RuntimeError("Cannot download from GPU: no GPU data available")
        
        # Download from GPU
        if ctx.stream:
            self.cpu = self.gpu.download(ctx.stream)
        else:
            self.cpu = self.gpu.download()
        import gpu_ops

        gpu_ops.record_external_transfer("gpu_to_cpu", int(self.cpu.nbytes))
        
        return self.cpu
    
    def clone(self) -> "Frame":
        """Create a copy of this frame."""
        cpu_copy = self.cpu.copy() if self.cpu is not None else None
        gpu_copy = self.gpu.clone() if self.gpu is not None else None
        return Frame(cpu=cpu_copy, gpu=gpu_copy, fmt=self.fmt)
    
    def release_gpu(self):
        """Release GPU memory."""
        if self.gpu is not None:
            # GpuMat doesn't have explicit release in Python, just dereference
            self.gpu = None
    
    def release_cpu(self):
        """Release CPU memory."""
        self.cpu = None
    
    def __repr__(self) -> str:
        cpu_str = f"cpu={self.cpu.shape if self.cpu is not None else None}"
        gpu_str = f"gpu={self.gpu.size() if self.gpu is not None else None}"
        return f"Frame({cpu_str}, {gpu_str}, fmt={self.fmt})"



