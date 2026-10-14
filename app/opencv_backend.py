"""
OpenCV Backend for MegaGlitch Node0.6
======================================

Optional, modular backend for frame loading/saving, video decode/encode,
and utility operations (resize/colorspace/warp).

Design Principles:
- Safe import: Never crash if cv2 unavailable
- Color consistency: Internal format is RGB uint8 (H,W,3)
- BGR only at boundaries: Convert to/from BGR only when calling cv2 functions
- Fallback gracefully: Return None or raise clear errors with install instructions
- No GUI dependencies: Never require cv2.imshow()

Environment:
- Windows 11 + Miniforge conda env 'cv311'
- Python 3.11 + OpenCV 4.12
- conda-forge packages preferred
"""

import sys
import numpy as np
import logging
import threading
from typing import Optional, Tuple, Generator
from pathlib import Path

# =============================================================================
# Safe Import
# =============================================================================

_cv2 = None
_cv2_error = None
_bootstrap_attempted = False

def safe_import_cv2() -> Tuple[Optional[object], Optional[str]]:
    """
    Safely import cv2 without crashing if unavailable.
    
    Returns:
        (cv2_module_or_None, error_string_or_None)
    """
    global _cv2, _cv2_error
    
    if _cv2 is not None:
        return _cv2, None
    if _cv2_error is not None:
        return None, _cv2_error
    
    global _bootstrap_attempted

    if not _bootstrap_attempted:
        _bootstrap_attempted = True
        try:
            # Ensure CUDA/OpenCV DLL paths are configured for custom builds.
            from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
            bootstrap_for_dev_defaults()
        except Exception as e:
            logging.warning(f"OpenCV bootstrap skipped: {e}")

    try:
        import cv2
        _cv2 = cv2
        logging.info(f"OpenCV loaded: {cv2.__version__}")
        return _cv2, None
    except ImportError as e:
        _cv2_error = f"OpenCV not available: {e}\n" + \
                     "Install instructions:\n" + \
                     "  1. Open PowerShell\n" + \
                     "  2. conda activate cv311\n" + \
                     "  3. conda install -c conda-forge opencv -y\n"
        logging.warning(_cv2_error)
        return None, _cv2_error

def get_cv2():
    """Get cv2 module or raise with helpful error."""
    cv2, error = safe_import_cv2()
    if cv2 is None:
        raise ImportError(error)
    return cv2

def is_cv2_available() -> bool:
    """Check if cv2 is available without raising."""
    cv2, _ = safe_import_cv2()
    return cv2 is not None

def check_environment():
    """
    Check if running in correct conda environment.
    Returns (is_correct_env, message)
    """
    executable = sys.executable.lower()
    in_cv311 = "envs\\cv311" in executable or "envs/cv311" in executable
    
    if not in_cv311:
        msg = f"Warning: Not running in cv311 environment!\n" + \
              f"Current Python: {sys.executable}\n" + \
              f"Expected path to contain: envs\\cv311\n" + \
              f"Run: conda activate cv311"
        return False, msg
    return True, "Running in cv311 environment [OK]"


# =============================================================================
# Color Space Conversions
# =============================================================================

def as_rgb_u8(frame: np.ndarray, source_format: str = "auto") -> np.ndarray:
    """
    Convert input frame to canonical RGB uint8 (H,W,3) format.
    
    Args:
        frame: Input frame (RGB/BGR/RGBA/float32/float64)
        source_format: "auto", "rgb", "bgr", "rgba", "bgra"
    
    Returns:
        np.ndarray: RGB uint8 (H,W,3)
    """
    if frame is None:
        raise ValueError("Frame is None")
    
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {frame.shape}")
    
    h, w, c = frame.shape
    
    # Convert float to uint8
    if frame.dtype in (np.float32, np.float64):
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    
    # Handle alpha channel
    if c == 4:
        if source_format == "bgra":
            # BGRA -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB) if is_cv2_available() else frame[:, :, [2, 1, 0, 3]][:, :, :3]
        else:
            # RGBA -> RGB (drop alpha)
            frame = frame[:, :, :3]
        c = 3
    
    # BGR -> RGB conversion
    if source_format == "bgr" or (source_format == "auto" and c == 3):
        # If from OpenCV, assume BGR
        cv2 = get_cv2()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame

def as_bgr_u8(frame: np.ndarray) -> np.ndarray:
    """
    Convert RGB uint8 to BGR uint8 for cv2 calls.
    
    Args:
        frame: RGB uint8 (H,W,3)
    
    Returns:
        np.ndarray: BGR uint8 (H,W,3)
    """
    if frame is None:
        raise ValueError("Frame is None")
    
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB (H,W,3), got shape {frame.shape}")
    
    cv2 = get_cv2()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def as_rgba_u8(frame: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Add alpha channel to RGB frame.
    
    Args:
        frame: RGB uint8 (H,W,3)
        alpha: Optional alpha channel (H,W) or scalar value (0-255)
    
    Returns:
        np.ndarray: RGBA uint8 (H,W,4)
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB (H,W,3), got shape {frame.shape}")
    
    h, w = frame.shape[:2]
    
    if alpha is None:
        alpha = np.full((h, w), 255, dtype=np.uint8)
    elif isinstance(alpha, (int, float)):
        alpha = np.full((h, w), int(alpha), dtype=np.uint8)
    
    return np.dstack([frame, alpha])


# =============================================================================
# Image I/O
# =============================================================================

def read_image(path: str, as_format: str = "rgb") -> Optional[np.ndarray]:
    """
    Load image from file using OpenCV.
    
    Args:
        path: File path
        as_format: "rgb" (default) or "bgr" or "rgba"
    
    Returns:
        np.ndarray: RGB uint8 (H,W,3) or None if failed
    """
    cv2 = get_cv2()
    
    try:
        # Read with cv2 (returns BGR by default)
        flags = cv2.IMREAD_COLOR  # Default: BGR
        if as_format == "rgba":
            flags = cv2.IMREAD_UNCHANGED  # Load with alpha if present
        
        img = cv2.imread(str(path), flags)
        
        if img is None:
            logging.error(f"Failed to load image: {path}")
            return None
        
        # Convert to RGB if needed
        if as_format == "rgb":
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)[:, :, :3]
        elif as_format == "rgba":
            if img.ndim == 3 and img.shape[2] == 3:
                # Add alpha channel
                img = as_rgba_u8(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        # else: as_format == "bgr", keep as-is
        
        logging.debug(f"Loaded image: {path} -> shape={img.shape}, dtype={img.dtype}")
        return img
    
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}")
        return None

def write_image(frame: np.ndarray, path: str, source_format: str = "rgb", quality: int = 95) -> bool:
    """
    Write image to file using OpenCV.
    
    Args:
        frame: Image array
        path: Output file path
        source_format: "rgb" (default) or "bgr" or "rgba"
        quality: JPEG quality (0-100), PNG compression (0-9)
    
    Returns:
        bool: True if successful
    """
    cv2 = get_cv2()
    
    try:
        # Convert to BGR for cv2
        if source_format == "rgb":
            frame = as_bgr_u8(frame)
        elif source_format == "rgba":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        # else: already BGR
        
        # Choose codec params by extension
        ext = Path(path).suffix.lower()
        params = []
        
        if ext in ['.jpg', '.jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == '.png':
            # PNG compression: 0-9 (map quality 0-100 to 0-9 inverted)
            compression = int((100 - quality) / 100 * 9)
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif ext == '.webp':
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        
        success = cv2.imwrite(str(path), frame, params)
        
        if success:
            logging.debug(f"Wrote image: {path}")
        else:
            logging.error(f"Failed to write image: {path}")
        
        return success
    
    except Exception as e:
        logging.error(f"Error writing image {path}: {e}")
        return False


# =============================================================================
# Video I/O
# =============================================================================

class VideoReader:
    """
    Video reader wrapper yielding RGB frames.
    
    Usage:
        reader = VideoReader("video.mp4")
        for frame in reader:
            # frame is RGB uint8 (H,W,3)
            process(frame)
        reader.close()
    
    Or as context manager:
        with VideoReader("video.mp4") as reader:
            for frame in reader:
                process(frame)
    """
    
    def __init__(self, path: str):
        self.path = str(path)
        self.cap = None
        self.fps = 30.0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self._opened = False
        self._next_index = 0
        self._cap_lock = threading.RLock()
        
        self._open()
    
    def _open(self):
        """Open video file."""
        cv2 = get_cv2()
        
        try:
            self.cap = cv2.VideoCapture(self.path)
            
            if not self.cap.isOpened():
                raise IOError(f"Failed to open video: {self.path}")
            
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._next_index = 0
            self._opened = True
            
            logging.info(f"Opened video: {self.path} ({self.width}x{self.height}, {self.fps} fps, {self.frame_count} frames)")
        
        except Exception as e:
            logging.error(f"Error opening video {self.path}: {e}")
            raise
    
    def __iter__(self):
        """Iterate over frames."""
        return self
    
    def __next__(self) -> np.ndarray:
        """Get next frame (RGB uint8)."""
        with self._cap_lock:
            if not self._opened or self.cap is None:
                raise StopIteration
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                raise StopIteration
            
            # Convert BGR to RGB
            cv2 = get_cv2()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._next_index += 1
            
            return frame
    
    def read(self) -> Optional[np.ndarray]:
        """Read single frame (returns None if end of video)."""
        try:
            return next(self)
        except StopIteration:
            return None
    
    def set_position(self, frame_num: int):
        """Seek to specific frame."""
        with self._cap_lock:
            if self.cap:
                cv2 = get_cv2()
                frame_num = max(0, int(frame_num))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                self._next_index = frame_num

    def set_image_index(self, frame_num: int):
        """imageio-compatible seek API."""
        self.set_position(frame_num)

    def get_next_index(self) -> int:
        """imageio-compatible next frame index."""
        with self._cap_lock:
            return int(self._next_index)

    def get_next_data(self) -> np.ndarray:
        """imageio-compatible frame read API."""
        return next(self)

    def get_meta_data(self) -> dict:
        """imageio-compatible metadata API."""
        return {
            "fps": float(self.fps) if self.fps else 30.0,
            "nframes": int(self.frame_count),
            "size": (int(self.width), int(self.height)),
        }
    
    def close(self):
        """Release video capture."""
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
                self._opened = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()


class VideoWriter:
    """
    Video writer wrapper accepting RGB frames.
    
    Usage:
        with VideoWriter("output.mp4", fps=30, size=(1920, 1080)) as writer:
            for frame in frames:
                writer.write(frame)  # frame is RGB uint8 (H,W,3)
    """
    
    def __init__(self, path: str, fps: float = 30.0, size: Tuple[int, int] = None,
                 codec: str = "mp4v", fallback_to_existing: bool = True):
        """
        Args:
            path: Output file path
            fps: Frames per second
            size: (width, height) - required
            codec: FourCC codec string ("mp4v", "avc1", "H264", "XVID", etc.)
            fallback_to_existing: If codec fails, don't crash (for compatibility)
        """
        if size is None:
            raise ValueError("VideoWriter requires size=(width, height)")
        
        self.path = str(path)
        self.fps = fps
        self.size = size
        self.codec_str = codec
        self.fallback = fallback_to_existing
        self.writer = None
        self._opened = False
        
        self._open()
    
    def _open(self):
        """Open video writer."""
        cv2 = get_cv2()
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*self.codec_str)
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
            
            if not self.writer.isOpened():
                if self.fallback:
                    logging.warning(f"Failed to open VideoWriter with codec {self.codec_str}, will skip video write")
                    self.writer = None
                    return
                else:
                    raise IOError(f"Failed to open VideoWriter: {self.path}")
            
            self._opened = True
            logging.info(f"Opened video writer: {self.path} ({self.size[0]}x{self.size[1]}, {self.fps} fps, codec={self.codec_str})")
        
        except Exception as e:
            logging.error(f"Error opening video writer {self.path}: {e}")
            if not self.fallback:
                raise
    
    def write(self, frame: np.ndarray):
        """Write RGB frame to video."""
        if not self._opened or self.writer is None:
            return
        
        cv2 = get_cv2()
        
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Ensure correct size
        if frame_bgr.shape[:2][::-1] != self.size:
            frame_bgr = cv2.resize(frame_bgr, self.size)
        
        self.writer.write(frame_bgr)
    
    def close(self):
        """Release video writer."""
        if self.writer:
            self.writer.release()
            self.writer = None
            self._opened = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()


# =============================================================================
# Utility Operations
# =============================================================================

def resize(frame: np.ndarray, size: Tuple[int, int], 
           interpolation: str = "lanczos") -> np.ndarray:
    """
    Resize frame using OpenCV.
    
    Args:
        frame: Input frame
        size: (width, height)
        interpolation: "nearest", "linear", "cubic", "lanczos", "area"
    
    Returns:
        Resized frame
    """
    cv2 = get_cv2()
    
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    
    interp = interp_map.get(interpolation, cv2.INTER_LANCZOS4)
    
    return cv2.resize(frame, size, interpolation=interp)

def rotate(frame: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None,
           scale: float = 1.0, border_mode: str = "constant", border_value: int = 0) -> np.ndarray:
    """
    Rotate frame using OpenCV warpAffine.
    
    Args:
        frame: Input frame
        angle: Rotation angle in degrees (counter-clockwise)
        center: Rotation center (x, y), default is image center
        scale: Scaling factor
        border_mode: "constant", "replicate", "reflect", "wrap"
        border_value: Fill value for constant border
    
    Returns:
        Rotated frame
    """
    cv2 = get_cv2()
    
    h, w = frame.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Border mode mapping
    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT,
        "wrap": cv2.BORDER_WRAP,
    }
    border = border_map.get(border_mode, cv2.BORDER_CONSTANT)
    
    # Apply rotation
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=border, borderValue=border_value)
    
    return rotated

def gaussian_blur(frame: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur using OpenCV.
    
    Args:
        frame: Input frame
        kernel_size: Kernel size (must be odd)
        sigma: Gaussian kernel standard deviation (0 = auto)
    
    Returns:
        Blurred frame
    """
    cv2 = get_cv2()
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)


# =============================================================================
# CUDA-Accelerated Operations
# =============================================================================

_cuda_available = None
_cuda_check_message = None

def is_cuda_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    More robust check that verifies OpenCV build configuration.
    """
    global _cuda_available, _cuda_check_message
    
    if _cuda_available is not None:
        return _cuda_available
    
    cv2 = get_cv2()
    
    try:
        # Get build information
        info = cv2.getBuildInformation()
        
        # 1) Check for explicit CUDA build
        has_cuda_build = "NVIDIA CUDA:" in info and "YES" in info
        
        # 2) Try device count
        try:
            try:
                from megaglitchnode.opencv_cuda_bootstrap import _get_cuda_device_count
                n = _get_cuda_device_count(cv2)
            except Exception:
                n = cv2.cuda.getCudaEnabledDeviceCount()
        except Exception as e:
            _cuda_available = False
            _cuda_check_message = f"cv2.cuda present but unusable: {e}"
            logging.warning(_cuda_check_message)
            return False
        
        if n <= 0:
            if not has_cuda_build:
                _cuda_available = False
                _cuda_check_message = "OpenCV was built without CUDA modules."
            else:
                _cuda_available = False
                _cuda_check_message = "No CUDA devices detected by OpenCV (driver/runtime issue or no NVIDIA GPU)."
            logging.info(_cuda_check_message)
            return False
        
        # Success!
        _cuda_available = True
        _cuda_check_message = f"CUDA devices detected: {n}"
        logging.info(_cuda_check_message)
        return True
        
    except Exception as e:
        _cuda_available = False
        _cuda_check_message = f"CUDA check failed: {e}"
        logging.error(_cuda_check_message)
        return False

def get_cuda_info() -> tuple:
    """
    Get CUDA availability status and detailed message.
    Returns: (is_available: bool, message: str)
    """
    available = is_cuda_available()
    return available, _cuda_check_message or "CUDA status unknown"


def resize_cuda(frame: np.ndarray, size: Tuple[int, int], 
                interpolation: str = "lanczos") -> np.ndarray:
    """
    Resize frame using CUDA GPU acceleration.
    
    Args:
        frame: Input frame (RGB uint8)
        size: (width, height)
        interpolation: "nearest", "linear", "cubic", "lanczos"
    
    Returns:
        Resized frame (RGB uint8)
    """
    if not is_cuda_available():
        # Fallback to CPU
        return resize(frame, size, interpolation)
    
    cv2 = get_cv2()
    
    try:
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Map interpolation
        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
        
        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_frame, size, interpolation=interp)
        
        # Download from GPU
        result = gpu_resized.download()
        
        return result
    
    except Exception as e:
        logging.warning(f"CUDA resize failed, falling back to CPU: {e}")
        return resize(frame, size, interpolation)


def rotate_cuda(frame: np.ndarray, angle: float, center: Optional[Tuple[int, int]] = None,
                scale: float = 1.0, border_mode: str = "constant", border_value: int = 0) -> np.ndarray:
    """
    Rotate frame using CUDA GPU acceleration.
    
    Args:
        frame: Input frame (RGB uint8)
        angle: Rotation angle in degrees
        center: Rotation center (x, y)
        scale: Scaling factor
        border_mode: "constant", "replicate", "reflect", "wrap"
        border_value: Fill value
    
    Returns:
        Rotated frame (RGB uint8)
    """
    if not is_cuda_available():
        # Fallback to CPU
        return rotate(frame, angle, center, scale, border_mode, border_value)
    
    cv2 = get_cv2()
    
    try:
        h, w = frame.shape[:2]
        
        if center is None:
            center = (w // 2, h // 2)
        
        # Get rotation matrix (CPU operation)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Border mode mapping
        border_map = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "wrap": cv2.BORDER_WRAP,
        }
        border = border_map.get(border_mode, cv2.BORDER_CONSTANT)
        
        # Warp on GPU
        gpu_rotated = cv2.cuda.warpAffine(
            gpu_frame, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=border,
            borderValue=border_value
        )
        
        # Download from GPU
        result = gpu_rotated.download()
        
        return result
    
    except Exception as e:
        logging.warning(f"CUDA rotate failed, falling back to CPU: {e}")
        return rotate(frame, angle, center, scale, border_mode, border_value)


def gaussian_blur_cuda(frame: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur using CUDA GPU acceleration.
    
    Args:
        frame: Input frame (RGB uint8)
        kernel_size: Kernel size (must be odd)
        sigma: Gaussian standard deviation
    
    Returns:
        Blurred frame (RGB uint8)
    """
    if not is_cuda_available():
        # Fallback to CPU
        return gaussian_blur(frame, kernel_size, sigma)
    
    cv2 = get_cv2()
    
    try:
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Create Gaussian filter
        gaussian_filter = cv2.cuda.createGaussianFilter(
            gpu_frame.type(), -1,
            (kernel_size, kernel_size),
            sigma
        )
        
        # Apply on GPU
        gpu_blurred = gaussian_filter.apply(gpu_frame)
        
        # Download from GPU
        result = gpu_blurred.download()
        
        return result
    
    except Exception as e:
        logging.warning(f"CUDA blur failed, falling back to CPU: {e}")
        return gaussian_blur(frame, kernel_size, sigma)


def bilateral_filter_cuda(frame: np.ndarray, d: int = 9, 
                          sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter using CUDA (edge-preserving blur).
    
    Args:
        frame: Input frame (RGB uint8)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
    
    Returns:
        Filtered frame (RGB uint8)
    """
    if not is_cuda_available():
        # Fallback to CPU bilateral filter
        cv2 = get_cv2()
        return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    
    cv2 = get_cv2()
    
    try:
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Apply bilateral filter on GPU (per-channel)
        channels = cv2.split(frame)
        gpu_channels = []
        
        for ch in channels:
            gpu_ch = cv2.cuda_GpuMat()
            gpu_ch.upload(ch)
            
            # Create bilateral filter
            bilateral = cv2.cuda.createBilateralFilter(
                gpu_ch.type(), d, sigma_color, sigma_space
            )
            
            # Apply filter
            gpu_filtered = bilateral.apply(gpu_ch)
            gpu_channels.append(gpu_filtered.download())
        
        # Merge channels
        result = cv2.merge(gpu_channels)
        
        return result
    
    except Exception as e:
        logging.warning(f"CUDA bilateral filter failed, falling back to CPU: {e}")
        cv2 = get_cv2()
        return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)


def kaleidoscope_cuda(frame: np.ndarray, segments: int = 6, rotation: float = 0) -> np.ndarray:
    """
    Apply kaleidoscope effect using CUDA GPU acceleration.
    
    Args:
        frame: Input frame (RGB uint8)
        segments: Number of kaleidoscope segments
        rotation: Rotation angle in degrees
    
    Returns:
        Kaleidoscope frame (RGB uint8)
    """
    if not is_cuda_available():
        # Fallback to CPU kaleidoscope (would need to implement)
        logging.warning("CUDA kaleidoscope not available, no CPU fallback")
        return frame
    
    cv2 = get_cv2()
    
    try:
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        
        # Create output frame
        result = np.zeros_like(frame)
        
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Apply each segment
        angle_per_segment = 360.0 / segments
        
        for i in range(segments):
            angle = rotation + i * angle_per_segment
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Warp on GPU
            gpu_warped = cv2.cuda.warpAffine(
                gpu_frame, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )
            
            # Download segment
            segment = gpu_warped.download()
            
            # Blend with result
            result = cv2.addWeighted(result, 1.0, segment, 1.0 / segments, 0)
        
        return result.astype(np.uint8)
    
    except Exception as e:
        logging.warning(f"CUDA kaleidoscope failed: {e}")
        return frame


# =============================================================================
# Module Info
# =============================================================================

def get_info() -> dict:
    """Get OpenCV backend info."""
    cv2, error = safe_import_cv2()
    env_ok, env_msg = check_environment()
    
    info = {
        "available": cv2 is not None,
        "version": cv2.__version__ if cv2 else None,
        "error": error,
        "python": sys.version,
        "executable": sys.executable,
        "env_check": env_msg,
        "env_ok": env_ok,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_check_message": None,
    }
    
    if cv2:
        # Check for CUDA support with detailed message
        cuda_available, cuda_message = get_cuda_info()
        info["cuda_available"] = cuda_available
        info["cuda_check_message"] = cuda_message
        
        if cuda_available:
            # Get device count
            try:
                try:
                    from megaglitchnode.opencv_cuda_bootstrap import _get_cuda_device_count
                    cuda_count = _get_cuda_device_count(cv2)
                except Exception:
                    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
                info["cuda_device_count"] = cuda_count
                
                # Get device info
                try:
                    device_info = cv2.cuda.getDevice()
                    info["cuda_device_name"] = f"CUDA Device {device_info}"
                except:
                    info["cuda_device_name"] = "CUDA Device (info unavailable)"
            except:
                info["cuda_device_count"] = 0
    
    return info

def print_info():
    """Print OpenCV backend info (for debugging)."""
    info = get_info()
    
    print("=" * 70)
    print("OpenCV Backend Info")
    print("=" * 70)
    print(f"Available: {info['available']}")
    print(f"Version: {info['version']}")
    print(f"Python: {info['python']}")
    print(f"Executable: {info['executable']}")
    print(f"Environment: {info['env_check']}")
    print(f"CUDA: {info.get('cuda_available', False)}")
    
    if info.get('cuda_check_message'):
        print(f"CUDA Status: {info['cuda_check_message']}")
    
    if info.get('cuda_available'):
        print(f"CUDA Devices: {info.get('cuda_device_count', 0)}")
        if info.get('cuda_device_name'):
            print(f"Device Name: {info['cuda_device_name']}")
    
    if info['error']:
        print(f"\nError:\n{info['error']}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Self-test when run directly
    print_info()
