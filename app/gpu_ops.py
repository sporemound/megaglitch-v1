"""
GPU Operations Module - CUDA-accelerated image processing with GpuMat support.

This module provides GPU-first operations using cv2.cuda with automatic CPU fallback.
Operations prefer GpuMat input/output to minimize CPU↔GPU transfers.

Usage:
    import gpu_ops
    gpu_ops.set_use_cuda(True)  # Enable CUDA
    
    # CPU input (will upload internally)
    result = gpu_ops.gaussian_blur(numpy_array, ksize=5, sigmaX=1.0)
    
    # GPU input (stays on GPU)
    gpu_result = gpu_ops.gaussian_blur_gpu(gpu_mat, ksize=5, sigmaX=1.0)
"""

import cv2
import numpy as np
import logging
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# ============================================================================
# Global State
# ============================================================================
USE_CUDA = False
_cached_filters = {}  # Cache for CUDA filter objects
_cached_gpu_mats = {}  # Cache for persistent GpuMat objects
_current_cache_epoch = 0  # Track cache epoch for invalidation
_transfer_observer = ContextVar("megaglitch_gpu_transfer_observer", default=None)


def _record_transfer(direction: str, byte_count: int) -> None:
    observer = _transfer_observer.get()
    if observer is not None:
        observer(str(direction), int(byte_count))


def record_external_transfer(direction: str, byte_count: int) -> None:
    """Report a real provider boundary implemented outside gpu_ops kernels."""
    _record_transfer(direction, byte_count)


@contextmanager
def observe_transfers(observer):
    """Measure uploads/downloads made through the central GPU boundary helpers."""
    token = _transfer_observer.set(observer)
    try:
        yield
    finally:
        _transfer_observer.reset(token)


def _clear_cached_state():
    """Clear all cached GPU state. Called on CUDA DLL update or context reset."""
    global _cached_filters, _cached_gpu_mats, _current_cache_epoch
    _cached_filters.clear()
    _cached_gpu_mats.clear()
    _current_cache_epoch = 0  # Reset epoch to force rebuild
    logger.info("Cleared gpu_ops cached state")


def _check_cache_epoch():
    """Check if cache epoch changed and clear caches if needed."""
    global _current_cache_epoch, _cached_filters, _cached_gpu_mats
    try:
        from megaglitchnode.opencv_cuda_bootstrap import get_cache_epoch
        current_epoch = get_cache_epoch()
        if current_epoch != _current_cache_epoch:
            logger.info(f"Cache epoch changed ({_current_cache_epoch} -> {current_epoch}), clearing GPU caches")
            _cached_filters.clear()
            _cached_gpu_mats.clear()
            _current_cache_epoch = current_epoch
    except ImportError:
        pass


# ============================================================================
# CUDA Detection and Configuration
# ============================================================================
def get_cuda_device_count() -> int:
    """Get number of CUDA-enabled devices."""
    try:
        try:
            from megaglitchnode.opencv_cuda_bootstrap import _get_cuda_device_count
            return int(_get_cuda_device_count(cv2))
        except Exception:
            pass

        if hasattr(cv2, "cuda"):
            return int(cv2.cuda.getCudaEnabledDeviceCount())
        return 0
    except Exception as e:
        logger.warning("Error querying CUDA device count: %s", e)
        return 0


def cuda_available() -> bool:
    """Check if CUDA is available and functional."""
    try:
        return hasattr(cv2, "cuda") and get_cuda_device_count() > 0
    except Exception:
        return False


def is_cuda_available() -> bool:
    """Public API to check CUDA availability."""
    return bool(cuda_available())


def set_use_cuda(flag: bool):
    """Enable or disable CUDA usage."""
    global USE_CUDA
    avail = cuda_available()
    USE_CUDA = bool(flag) and avail
    if flag and not avail:
        logger.warning("CUDA requested but not available in this OpenCV/runtime environment.")
    elif flag and avail:
        logger.debug("CUDA enabled")


def get_use_cuda() -> bool:
    """Check if CUDA is currently enabled."""
    return USE_CUDA


def get_cuda_debug_info() -> dict:
    """Return diagnostic info for UI/logs."""
    info = {
        "cv2_path": getattr(cv2, "__file__", None),
        "hasattr_cuda": hasattr(cv2, "cuda"),
        "device_count": get_cuda_device_count(),
        "use_cuda": USE_CUDA,
    }
    try:
        build = cv2.getBuildInformation()
        info["build_has_cuda"] = ("CUDA:                    YES" in build) or ("CUDA:                    Yes" in build)
    except Exception:
        info["build_has_cuda"] = None
    return info


# ============================================================================
# GpuMat Utilities
# ============================================================================
def upload(arr: np.ndarray) -> cv2.cuda_GpuMat:
    """Upload numpy array to GPU."""
    g = cv2.cuda_GpuMat()
    g.upload(arr)
    _record_transfer("cpu_to_gpu", int(arr.nbytes))
    return g


def download(gpu_mat) -> np.ndarray:
    """Download GpuMat to numpy array."""
    if isinstance(gpu_mat, np.ndarray):
        return gpu_mat
    result = gpu_mat.download()
    _record_transfer("gpu_to_cpu", int(result.nbytes))
    return result


def ensure_gpu(data):
    """Ensure data is on GPU. Uploads if needed."""
    if not USE_CUDA:
        return data
        
    try:
        from megaglitchnode.gpu import Frame
        if isinstance(data, Frame):
            if getattr(data, 'gpu', None) is not None:
                return data.gpu
            if getattr(data, 'cpu', None) is not None:
                return upload(data.cpu)
            raise RuntimeError("Frame has no CPU or GPU data")
    except ImportError:
        pass
        
    if isinstance(data, cv2.cuda_GpuMat):
        return data
    elif isinstance(data, np.ndarray):
        return upload(data)
    else:
        raise TypeError(f"Cannot upload {type(data)} to GPU")


def ensure_cpu(data):
    """Ensure data is on CPU. Downloads if needed."""
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(cv2, 'cuda_GpuMat') and isinstance(data, cv2.cuda_GpuMat):
        return download(data)
    
    try:
        from megaglitchnode.gpu import Frame
        if isinstance(data, Frame):
            if getattr(data, 'cpu', None) is not None:
                return data.cpu
            if getattr(data, 'gpu', None) is not None:
                return download(data.gpu)
            raise RuntimeError("Frame has no CPU or GPU data")
    except ImportError:
        pass
        
    raise TypeError(f"Cannot convert {type(data)} to numpy array")


def _to_type_code(arr: np.ndarray):
    """Get OpenCV type code from numpy array."""
    if arr.dtype == np.uint8:
        chans = 1 if arr.ndim == 2 else arr.shape[2]
        return {1: cv2.CV_8UC1, 3: cv2.CV_8UC3, 4: cv2.CV_8UC4}.get(chans, cv2.CV_8UC3)
    if arr.dtype == np.float32:
        chans = 1 if arr.ndim == 2 else arr.shape[2]
        return {1: cv2.CV_32FC1, 3: cv2.CV_32FC3, 4: cv2.CV_32FC4}.get(chans, cv2.CV_32FC3)
    return cv2.CV_8UC3


def _get_type_code_from_gpumat(gpu_mat: cv2.cuda_GpuMat):
    """Get OpenCV type code from GpuMat."""
    return gpu_mat.type()


# ============================================================================
# GPU-First Operations (prefer GpuMat in/out)
# ============================================================================
def remap_gpu(src: cv2.cuda_GpuMat, map_x: cv2.cuda_GpuMat, map_y: cv2.cuda_GpuMat,
              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) -> cv2.cuda_GpuMat:
    """Remap on GPU. Input and output are GpuMat."""
    if not USE_CUDA:
        raise RuntimeError("CUDA not enabled")
    return cv2.cuda.remap(src, map_x, map_y, interpolation, borderMode=borderMode)


def gaussian_blur_gpu(src: cv2.cuda_GpuMat, ksize: int, sigmaX: float, 
                      sigmaY: float = 0) -> cv2.cuda_GpuMat:
    """Gaussian blur on GPU. Input and output are GpuMat."""
    if not USE_CUDA:
        raise RuntimeError("CUDA not enabled")
    
    # Check cache epoch for invalidation
    _check_cache_epoch()
    
    cache_key = f"gauss_{ksize}_{sigmaX}_{sigmaY}_{src.type()}"
    if cache_key not in _cached_filters:
        src_type = src.type()
        _cached_filters[cache_key] = cv2.cuda.createGaussianFilter(
            src_type, src_type, (ksize, ksize), sigmaX, sigmaY if sigmaY else sigmaX
        )
    
    return _cached_filters[cache_key].apply(src)


def add_weighted_gpu(src1: cv2.cuda_GpuMat, alpha: float, src2: cv2.cuda_GpuMat, 
                     beta: float, gamma: float) -> cv2.cuda_GpuMat:
    """Weighted blend on GPU. Input and output are GpuMat."""
    if not USE_CUDA:
        raise RuntimeError("CUDA not enabled")
    return cv2.cuda.addWeighted(src1, alpha, src2, beta, gamma)


def resize_gpu(src: cv2.cuda_GpuMat, size: tuple, 
               interpolation=cv2.INTER_LINEAR) -> cv2.cuda_GpuMat:
    """Resize on GPU. Input and output are GpuMat."""
    if not USE_CUDA:
        raise RuntimeError("CUDA not enabled")
    return cv2.cuda.resize(src, size, interpolation=interpolation)


def cvt_color_gpu(src: cv2.cuda_GpuMat, code: int) -> cv2.cuda_GpuMat:
    """Color conversion on GPU. Input and output are GpuMat."""
    if not USE_CUDA:
        raise RuntimeError("CUDA not enabled")
    return cv2.cuda.cvtColor(src, code)


# ============================================================================
# Hybrid Operations (accept numpy or GpuMat, return numpy)
# These are for backward compatibility with existing code
# ============================================================================
def remap(arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, 
          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) -> np.ndarray:
    """Remap with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_mx = upload(map_x.astype(np.float32))
            g_my = upload(map_y.astype(np.float32))
            g_out = cv2.cuda.remap(g_img, g_mx, g_my, interpolation, borderMode=borderMode)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA remap fallback to CPU: {e}")
    return cv2.remap(arr, map_x, map_y, interpolation=interpolation, borderMode=borderMode)


def gaussian_blur(arr: np.ndarray, ksize: int, sigmaX: float, sigmaY: float = 0) -> np.ndarray:
    """Gaussian blur with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            src_type = _to_type_code(arr)
            dst_type = src_type
            
            cache_key = f"gauss_{ksize}_{sigmaX}_{sigmaY}_{src_type}"
            if cache_key not in _cached_filters:
                _cached_filters[cache_key] = cv2.cuda.createGaussianFilter(
                    src_type, dst_type, (ksize, ksize), sigmaX, sigmaY if sigmaY else sigmaX
                )
            
            g_img = upload(arr)
            g_out = _cached_filters[cache_key].apply(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA gaussian_blur fallback to CPU: {e}")
    return cv2.GaussianBlur(arr, (ksize, ksize), sigmaX, sigmaY)


def bilateral_filter(arr: np.ndarray, d: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
    """Bilateral filter with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            src_type = _to_type_code(arr)
            
            cache_key = f"bilateral_{d}_{sigmaColor}_{sigmaSpace}_{src_type}"
            if cache_key not in _cached_filters:
                _cached_filters[cache_key] = cv2.cuda.createBilateralFilter(
                    src_type, -1, sigmaColor, sigmaSpace, d if d > 0 else 0
                )
            
            g_img = upload(arr)
            g_out = _cached_filters[cache_key].apply(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA bilateral_filter fallback to CPU: {e}")
    return cv2.bilateralFilter(arr, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


def resize(arr: np.ndarray, size: tuple, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_out = cv2.cuda.resize(g_img, size, interpolation=interpolation)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA resize fallback to CPU: {e}")
    return cv2.resize(arr, size, interpolation=interpolation)


def cvtColor(arr: np.ndarray, code: int) -> np.ndarray:
    """Color conversion with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_out = cv2.cuda.cvtColor(g_img, code)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA cvtColor fallback to CPU: {e}")
    return cv2.cvtColor(arr, code)


def warpAffine(arr: np.ndarray, M: np.ndarray, dsize: tuple, 
               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) -> np.ndarray:
    """Affine warp with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_out = cv2.cuda.warpAffine(g_img, M.astype(np.float32), dsize, 
                                        flags=flags, borderMode=borderMode)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA warpAffine fallback to CPU: {e}")
    return cv2.warpAffine(arr, M, dsize, flags=flags, borderMode=borderMode)


def warpPerspective(arr: np.ndarray, M: np.ndarray, dsize: tuple, 
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) -> np.ndarray:
    """Perspective warp with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_out = cv2.cuda.warpPerspective(g_img, M.astype(np.float32), dsize, 
                                             flags=flags, borderMode=borderMode)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA warpPerspective fallback to CPU: {e}")
    return cv2.warpPerspective(arr, M, dsize, flags=flags, borderMode=borderMode)


def flip(arr: np.ndarray, flipCode: int) -> np.ndarray:
    """Flip with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_out = cv2.cuda.flip(g_img, flipCode)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA flip fallback to CPU: {e}")
    return cv2.flip(arr, flipCode)


def rotate(arr: np.ndarray, angle: float, center=None, scale=1.0, 
           borderMode=cv2.BORDER_REFLECT) -> np.ndarray:
    """Rotate with CUDA acceleration. Returns numpy array."""
    h, w = arr.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return warpAffine(arr, M, (w, h), borderMode=borderMode)


def addWeighted(src1: np.ndarray, alpha: float, src2: np.ndarray, beta: float, 
                gamma: float, dst=None) -> np.ndarray:
    """Weighted blend with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.addWeighted(g_src1, alpha, g_src2, beta, gamma)
            result = download(g_out)
            if dst is not None:
                np.copyto(dst, result)
                return dst
            return result
        except Exception as e:
            logger.debug(f"CUDA addWeighted fallback to CPU: {e}")
    
    result = cv2.addWeighted(src1, alpha, src2, beta, gamma)
    if dst is not None:
        np.copyto(dst, result)
        return dst
    return result


def add(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Add with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.add(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA add fallback to CPU: {e}")
    return cv2.add(src1, src2)


def subtract(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Subtract with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.subtract(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA subtract fallback to CPU: {e}")
    return cv2.subtract(src1, src2)


def multiply(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Multiply with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.multiply(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA multiply fallback to CPU: {e}")
    return cv2.multiply(src1, src2)


def divide(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Divide with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.divide(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA divide fallback to CPU: {e}")
    return cv2.divide(src1, src2)


def threshold(arr: np.ndarray, thresh: float, maxval: float, type: int) -> tuple:
    """Threshold with CUDA acceleration. Returns (thresh, result)."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            ret, g_out = cv2.cuda.threshold(g_img, thresh, maxval, type)
            return ret, download(g_out)
        except Exception as e:
            logger.debug(f"CUDA threshold fallback to CPU: {e}")
    return cv2.threshold(arr, thresh, maxval, type)


def medianBlur(arr: np.ndarray, ksize: int) -> np.ndarray:
    """Median blur (CPU only - not available in CUDA)."""
    return cv2.medianBlur(arr, ksize)


def boxFilter(arr: np.ndarray, ddepth: int, ksize: tuple, normalize: bool = True) -> np.ndarray:
    """Box filter with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            src_type = _to_type_code(arr)
            dst_type = src_type if ddepth == -1 else ddepth
            
            cache_key = f"box_{ksize}_{normalize}_{src_type}"
            if cache_key not in _cached_filters:
                _cached_filters[cache_key] = cv2.cuda.createBoxFilter(
                    src_type, dst_type, ksize, normalize=normalize
                )
            
            g_img = upload(arr)
            g_out = _cached_filters[cache_key].apply(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA boxFilter fallback to CPU: {e}")
    return cv2.boxFilter(arr, ddepth, ksize, normalize=normalize)


def morphologyEx(arr: np.ndarray, op: int, kernel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphological operations with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            src_type = _to_type_code(arr)
            
            cache_key = f"morph_{op}_{kernel.tobytes()}_{iterations}_{src_type}"
            if cache_key not in _cached_filters:
                _cached_filters[cache_key] = cv2.cuda.createMorphologyFilter(
                    op, src_type, kernel, iterations=iterations
                )
            
            g_img = upload(arr)
            g_out = _cached_filters[cache_key].apply(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA morphologyEx fallback to CPU: {e}")
    return cv2.morphologyEx(arr, op, kernel, iterations=iterations)


def Sobel(arr: np.ndarray, ddepth: int, dx: int, dy: int, ksize: int = 3) -> np.ndarray:
    """Sobel edge detection with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            src_type = _to_type_code(arr)
            dst_type = src_type if ddepth == -1 else ddepth
            
            cache_key = f"sobel_{dx}_{dy}_{ksize}_{src_type}"
            if cache_key not in _cached_filters:
                _cached_filters[cache_key] = cv2.cuda.createSobelFilter(
                    src_type, dst_type, dx, dy, ksize
                )
            
            g_img = upload(arr)
            g_out = _cached_filters[cache_key].apply(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA Sobel fallback to CPU: {e}")
    return cv2.Sobel(arr, ddepth, dx, dy, ksize)


def Laplacian(arr: np.ndarray, ddepth: int, ksize: int = 1) -> np.ndarray:
    """Laplacian edge detection with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            src_type = _to_type_code(arr)
            dst_type = src_type if ddepth == -1 else ddepth
            
            cache_key = f"laplacian_{ksize}_{src_type}"
            if cache_key not in _cached_filters:
                _cached_filters[cache_key] = cv2.cuda.createLaplacianFilter(
                    src_type, dst_type, ksize
                )
            
            g_img = upload(arr)
            g_out = _cached_filters[cache_key].apply(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA Laplacian fallback to CPU: {e}")
    return cv2.Laplacian(arr, ddepth, ksize)


def Canny(arr: np.ndarray, threshold1: float, threshold2: float) -> np.ndarray:
    """Canny edge detection with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            cache_key = f"canny_{threshold1}_{threshold2}"
            if cache_key not in _cached_filters:
                _cached_filters[cache_key] = cv2.cuda.createCannyEdgeDetector(threshold1, threshold2)
            
            g_img = upload(arr)
            g_out = _cached_filters[cache_key].detect(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA Canny fallback to CPU: {e}")
    return cv2.Canny(arr, threshold1, threshold2)


def equalizeHist(arr: np.ndarray) -> np.ndarray:
    """Histogram equalization with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_out = cv2.cuda.equalizeHist(g_img)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA equalizeHist fallback to CPU: {e}")
    return cv2.equalizeHist(arr)


def calcHist(arr: np.ndarray, channels: list, mask, histSize: list, ranges: list) -> np.ndarray:
    """Calculate histogram (CPU only - CUDA version more complex)."""
    return cv2.calcHist([arr], channels, mask, histSize, ranges)


def split(arr: np.ndarray) -> tuple:
    """Split multi-channel image with CUDA acceleration. Returns tuple of numpy arrays."""
    if USE_CUDA:
        try:
            g_img = upload(arr)
            g_channels = cv2.cuda.split(g_img)
            return tuple(download(ch) for ch in g_channels)
        except Exception as e:
            logger.debug(f"CUDA split fallback to CPU: {e}")
    return cv2.split(arr)


def merge(channels: list) -> np.ndarray:
    """Merge channels with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_channels = [upload(ch) for ch in channels]
            g_out = cv2.cuda.merge(g_channels)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA merge fallback to CPU: {e}")
    return cv2.merge(channels)


def absdiff(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Absolute difference with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.absdiff(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA absdiff fallback to CPU: {e}")
    return cv2.absdiff(src1, src2)


def bitwise_and(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Bitwise AND with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.bitwise_and(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA bitwise_and fallback to CPU: {e}")
    return cv2.bitwise_and(src1, src2)


def bitwise_or(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Bitwise OR with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.bitwise_or(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA bitwise_or fallback to CPU: {e}")
    return cv2.bitwise_or(src1, src2)


def bitwise_xor(src1: np.ndarray, src2: np.ndarray) -> np.ndarray:
    """Bitwise XOR with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src1 = upload(src1)
            g_src2 = upload(src2)
            g_out = cv2.cuda.bitwise_xor(g_src1, g_src2)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA bitwise_xor fallback to CPU: {e}")
    return cv2.bitwise_xor(src1, src2)


def bitwise_not(src: np.ndarray) -> np.ndarray:
    """Bitwise NOT with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src = upload(src)
            g_out = cv2.cuda.bitwise_not(g_src)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA bitwise_not fallback to CPU: {e}")
    return cv2.bitwise_not(src)


def min_max(src: np.ndarray) -> tuple:
    """Find min/max values with CUDA acceleration. Returns (min, max)."""
    if USE_CUDA:
        try:
            g_src = upload(src)
            minVal, maxVal = cv2.cuda.minMax(g_src)
            return minVal, maxVal
        except Exception as e:
            logger.debug(f"CUDA min_max fallback to CPU: {e}")
    return cv2.minMaxLoc(src)[:2]


def normalize(src: np.ndarray, alpha: float = 1.0, beta: float = 0.0, 
              norm_type: int = cv2.NORM_MINMAX) -> np.ndarray:
    """Normalize with CUDA acceleration. Returns numpy array."""
    if USE_CUDA:
        try:
            g_src = upload(src)
            g_out = cv2.cuda.normalize(g_src, alpha, beta, norm_type, -1)
            return download(g_out)
        except Exception as e:
            logger.debug(f"CUDA normalize fallback to CPU: {e}")
    return cv2.normalize(src, None, alpha, beta, norm_type)
