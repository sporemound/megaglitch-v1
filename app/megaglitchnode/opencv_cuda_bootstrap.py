"""
OpenCV CUDA Bootstrap Module

Provides reliable CUDA-enabled OpenCV import on Windows by managing DLL paths
and the custom OpenCV Python bindings path.
Must be called BEFORE any cv2 import.

CUDA Configuration (v12.6):
    - OpenCV bin: C:\\src\\opencv-cuda\\build_py312\\bin\\Release
    - OpenCV Python: C:\\src\\opencv-cuda\\build_py312\\lib\\python3\\Release
    - CUDA Toolkit: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin
    
Usage (Development):
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_for_dev_defaults
    bootstrap_for_dev_defaults()
    import cv2  # Now safe

Usage (Custom paths):
    from megaglitchnode.opencv_cuda_bootstrap import bootstrap_opencv_cuda
    bootstrap_opencv_cuda(
        opencv_bin=r"C:\\path\\to\\opencv\\bin",
        cuda_bin=r"C:\\path\\to\\cuda\\bin"
    )
    import cv2
"""

import os
import sys
import logging
import time
import re

logger = logging.getLogger(__name__)

# Global state to track bootstrap status
_bootstrap_complete = False
_cuda_verified = False
_cuda_status_message = ""
_cache_epoch = 0  # Increment to invalidate all GPU caches
_dll_directory_handles = []  # Keep Windows DLL search registrations alive.

# ============================================================================
# AUTHORITATIVE CUDA PATHS - Update these when CUDA version changes
# ============================================================================
DEFAULT_OPENCV_BIN = r"C:\src\opencv-cuda\build_py312\bin\Release"
DEFAULT_OPENCV_PYTHON = r"C:\src\opencv-cuda\build_py312\lib\python3\Release"
DEFAULT_CUDA_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
DEFAULT_CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
CUDA_VERSION = "12.6"


def _ensure_sys_path(path: str):
    """Prepend a path to sys.path once so the custom cv2 module wins import resolution."""
    if not path:
        return

    normalized = os.path.normcase(os.path.abspath(path))
    existing = {
        os.path.normcase(os.path.abspath(entry))
        for entry in sys.path
        if entry
    }
    if normalized not in existing:
        sys.path.insert(0, path)
        logger.info(f"Prepended OpenCV Python path: {path}")


def _get_cuda_device_count(cv2_module=None, retries: int = 3, delay: float = 0.25) -> int:
    """
    Query CUDA device count with a short retry window.

    Some Windows/OpenCV-CUDA setups transiently report 0 devices on the first
    query in a fresh process even though the runtime becomes usable moments later.
    """
    if cv2_module is None:
        import cv2 as cv2_module

    if not hasattr(cv2_module, "cuda"):
        return 0

    last_error = None
    for attempt in range(retries):
        try:
            count = int(cv2_module.cuda.getCudaEnabledDeviceCount())
            if count > 0 or attempt == retries - 1:
                return count
        except Exception as e:
            last_error = e
            if attempt == retries - 1:
                raise

        if delay > 0:
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    return 0


def _has_cuda_build(cv2_module=None) -> bool:
    """Check whether the imported OpenCV build includes CUDA modules."""
    if cv2_module is None:
        import cv2 as cv2_module

    if not hasattr(cv2_module, "cuda"):
        return False

    cuda_api_present = hasattr(cv2_module.cuda, "getCudaEnabledDeviceCount")
    cv2_path = str(getattr(cv2_module, "__file__", "") or "").lower()

    try:
        info = cv2_module.getBuildInformation()
    except Exception:
        return cuda_api_present

    if re.search(r"(?im)^\s*(?:NVIDIA\s+)?CUDA\s*:\s*YES\b", info):
        return True

    # Custom local OpenCV-CUDA builds do not always report CUDA cleanly in
    # getBuildInformation(), but the Python extension and cuda namespace are real.
    if cuda_api_present and ("opencv-cuda" in cv2_path or "build_py312" in cv2_path):
        return True

    return False


def bootstrap_opencv_cuda(opencv_bin: str, cuda_bin: str = None, opencv_python: str = None,
                          cuda_path: str = None):
    """
    Bootstrap OpenCV CUDA by adding required DLL directories.
    
    CRITICAL: Call this BEFORE importing cv2.
    Do NOT rely on global PATH - always use explicit os.add_dll_directory().
    
    Args:
        opencv_bin: Path to OpenCV DLLs (opencv_*.dll)
        cuda_bin: Path to CUDA Toolkit bin (cudart64_*.dll, etc.)
        opencv_python: Path to custom OpenCV Python bindings
        cuda_path: Path to CUDA Toolkit root
    
    Raises:
        RuntimeError: If paths don't exist or DLL directories can't be added
    """
    global _bootstrap_complete
    
    if _bootstrap_complete:
        logger.debug("Bootstrap already complete, skipping")
        return

    if cuda_path:
        os.environ["CUDA_PATH"] = cuda_path
        logger.info(f"Set CUDA_PATH={cuda_path}")

    if opencv_python:
        if not os.path.isdir(opencv_python):
            raise RuntimeError(f"OpenCV Python directory not found: {opencv_python}")
        _ensure_sys_path(opencv_python)
    
    # Validate and add OpenCV DLL directory (REQUIRED)
    if not os.path.isdir(opencv_bin):
        raise RuntimeError(f"OpenCV bin directory not found: {opencv_bin}")
    
    try:
        _dll_directory_handles.append(os.add_dll_directory(opencv_bin))
        logger.info(f"Added OpenCV DLL directory: {opencv_bin}")
    except Exception as e:
        raise RuntimeError(f"Failed to add OpenCV DLL directory: {e}")
    
    # Validate and add CUDA bin directory (REQUIRED for GPU effects)
    if cuda_bin:
        if not os.path.isdir(cuda_bin):
            logger.warning(f"CUDA bin directory not found: {cuda_bin}")
            logger.warning("GPU effects will be disabled")
        else:
            try:
                _dll_directory_handles.append(os.add_dll_directory(cuda_bin))
                logger.info(f"Added CUDA {CUDA_VERSION} DLL directory: {cuda_bin}")
            except Exception as e:
                logger.warning(f"Failed to add CUDA DLL directory: {e}")
    
    _bootstrap_complete = True
    logger.info("OpenCV CUDA bootstrap complete")


def bootstrap_for_dev_defaults():
    """
    Bootstrap using authoritative development paths.
    
    Environment variables (optional overrides):
        MEGAGLITCH_OPENCV_BIN: Path to OpenCV bin directory
        MEGAGLITCH_OPENCV_PYTHON: Path to OpenCV Python bindings directory
        MEGAGLITCH_CUDA_PATH: Path to CUDA Toolkit root
        MEGAGLITCH_CUDA_BIN: Path to CUDA Toolkit bin directory
    
    Default paths (CUDA 12.6):
        OpenCV: C:\\src\\opencv-cuda\\build_py312\\bin\\Release
        OpenCV Python: C:\\src\\opencv-cuda\\build_py312\\lib\\python3\\Release
        CUDA Root: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6
        CUDA: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin
    """
    global _bootstrap_complete
    opencv_python = None
    cuda_path = os.environ.get('MEGAGLITCH_CUDA_PATH', DEFAULT_CUDA_PATH)

    # Check if running as frozen executable (PyInstaller)
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        bundled_opencv = os.path.join(bundle_dir, "bin", "opencv")
        adjacent_opencv = os.path.join(os.path.dirname(sys.executable), "bin", "opencv")
        opencv_bin = (
            bundled_opencv if os.path.isdir(bundled_opencv) else adjacent_opencv
        )
        configured_cuda_bin = os.environ.get('MEGAGLITCH_CUDA_BIN', DEFAULT_CUDA_BIN)
        cuda_bin = configured_cuda_bin if os.path.isdir(configured_cuda_bin) else None
        
        logger.info("Running as frozen executable, using bundled OpenCV and system CUDA")
    else:
        # Running as normal Python script (development)
        opencv_bin = os.environ.get('MEGAGLITCH_OPENCV_BIN', DEFAULT_OPENCV_BIN)
        opencv_python = os.environ.get('MEGAGLITCH_OPENCV_PYTHON', DEFAULT_OPENCV_PYTHON)
        cuda_bin = os.environ.get('MEGAGLITCH_CUDA_BIN', DEFAULT_CUDA_BIN)

        if not os.path.isdir(opencv_bin) or not os.path.isdir(opencv_python):
            logger.warning(
                "Custom OpenCV paths are unavailable; using the installed CPU/system cv2: "
                "bin=%s python=%s",
                opencv_bin,
                opencv_python,
            )
            _bootstrap_complete = True
            return

    # Call main bootstrap
    bootstrap_opencv_cuda(opencv_bin, cuda_bin, opencv_python=opencv_python, cuda_path=cuda_path)


def get_cache_epoch() -> int:
    """Get current cache epoch. Nodes should check this to detect cache invalidation."""
    return _cache_epoch


def increment_cache_epoch():
    """Increment cache epoch to force all GPU nodes to rebuild caches."""
    global _cache_epoch
    _cache_epoch += 1
    logger.info(f"Cache epoch incremented to {_cache_epoch}")
    return _cache_epoch


def invalidate_gpu_caches():
    """
    Invalidate all GPU-side caches to ensure clean state after DLL updates.
    
    This clears:
    - CUDA effect states (feedback buffers, flow accumulators)
    - GPU node caches (filters, remap maps)
    - Any persistent GpuMat references
    - Increments cache epoch to force rebuilds
    """
    global _cache_epoch
    _cache_epoch += 1
    logger.info(f"Invalidating GPU caches (epoch={_cache_epoch})...")
    
    # 1. Clear CUDA effect states
    try:
        import cuda_effects
        cuda_effects.reset_cuda_states()
        logger.debug("Cleared cuda_effects states")
    except Exception as e:
        logger.debug(f"Could not clear cuda_effects states: {e}")
    
    # 2. Clear GPU node caches
    try:
        from megaglitchnode.gpu.context import reset_global_context
        reset_global_context()
        logger.debug("Cleared GPU context caches")
    except Exception as e:
        logger.debug(f"Could not clear GPU context caches: {e}")
    
    # 3. Clear gpu_ops cached state
    try:
        import gpu_ops
        gpu_ops._clear_cached_state()
        logger.debug("Cleared gpu_ops cached state")
    except Exception as e:
        logger.debug(f"Could not clear gpu_ops state: {e}")
    
    logger.info("GPU cache invalidation complete")


def run_cuda_selftest() -> dict:
    """
    Run comprehensive CUDA self-test to verify GPU functionality.
    
    Tests:
    1. CUDA device detection
    2. cv2.cuda.addWeighted
    3. cv2.cuda.createGaussianFilter
    4. cv2.cuda.remap
    
    Returns:
        dict with keys:
            'success': bool - All tests passed
            'device_count': int - Number of CUDA devices
            'device_name': str - Primary GPU name
            'tests': dict - Individual test results
            'errors': list - Error messages for failed tests
    """
    import cv2
    import numpy as np
    
    results = {
        'success': False,
        'device_count': 0,
        'device_name': 'Unknown',
        'tests': {},
        'errors': []
    }
    
    # Test 1: Device detection
    try:
        device_count = _get_cuda_device_count(cv2)
        results['device_count'] = device_count
        results['tests']['device_detection'] = device_count > 0
        
        if device_count > 0:
            cv2.cuda.setDevice(0)
            results['device_name'] = "CUDA Device 0"
        else:
            results['errors'].append("No CUDA devices detected")
            return results
    except Exception as e:
        results['errors'].append(f"Device detection failed: {e}")
        results['tests']['device_detection'] = False
        return results
    
    # Create test images
    test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test 2: addWeighted
    try:
        gpu_img1 = cv2.cuda_GpuMat()
        gpu_img2 = cv2.cuda_GpuMat()
        gpu_img1.upload(test_img)
        gpu_img2.upload(test_img)
        
        gpu_result = cv2.cuda.addWeighted(gpu_img1, 0.5, gpu_img2, 0.5, 0.0)
        result = gpu_result.download()
        
        results['tests']['addWeighted'] = result.shape == test_img.shape
    except Exception as e:
        results['errors'].append(f"addWeighted failed: {e}")
        results['tests']['addWeighted'] = False
    
    # Test 3: GaussianFilter
    try:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        
        gauss_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, cv2.CV_8UC3, (5, 5), 1.0)
        gpu_result = gauss_filter.apply(gpu_img)
        result = gpu_result.download()
        
        results['tests']['gaussianFilter'] = result.shape == test_img.shape
    except Exception as e:
        results['errors'].append(f"GaussianFilter failed: {e}")
        results['tests']['gaussianFilter'] = False
    
    # Test 4: remap
    try:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(test_img)
        
        h, w = test_img.shape[:2]
        map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        gpu_map_x = cv2.cuda_GpuMat()
        gpu_map_y = cv2.cuda_GpuMat()
        gpu_map_x.upload(map_x)
        gpu_map_y.upload(map_y)
        
        gpu_result = cv2.cuda.remap(gpu_img, gpu_map_x, gpu_map_y, cv2.INTER_LINEAR)
        result = gpu_result.download()
        
        results['tests']['remap'] = result.shape == test_img.shape
    except Exception as e:
        results['errors'].append(f"remap failed: {e}")
        results['tests']['remap'] = False
    
    # Overall success
    results['success'] = all(results['tests'].values())
    
    return results


def verify_cuda_on_startup() -> bool:
    """
    Verify CUDA functionality on startup.
    
    Returns:
        bool: True if CUDA is fully functional, False otherwise
    """
    global _cuda_verified, _cuda_status_message
    
    if _cuda_verified:
        return True
    
    try:
        results = run_cuda_selftest()
        
        if results['success']:
            _cuda_verified = True
            _cuda_status_message = f"CUDA {CUDA_VERSION} verified: {results['device_count']} device(s)"
            logger.info(_cuda_status_message)
            print(f"[OK] {_cuda_status_message}")
            return True
        else:
            _cuda_status_message = f"CUDA verification failed: {', '.join(results['errors'])}"
            logger.warning(_cuda_status_message)
            print(f"[FAIL] {_cuda_status_message}")
            return False
    
    except Exception as e:
        _cuda_status_message = f"CUDA verification error: {e}"
        logger.error(_cuda_status_message)
        print(f"[FAIL] {_cuda_status_message}")
        return False


def is_cuda_available() -> tuple:
    """
    Check if CUDA is available in OpenCV (safe to call after bootstrap).
    
    Returns:
        (is_available, message): Boolean and descriptive message
    """
    global _cuda_verified, _cuda_status_message
    
    if _cuda_verified:
        return True, _cuda_status_message
    
    try:
        import cv2
        
        # Check for CUDA module
        if not hasattr(cv2, 'cuda'):
            return False, "cv2.cuda module not found (CPU-only build)"
        
        # Check device count
        try:
            device_count = _get_cuda_device_count(cv2)
            if device_count <= 0:
                return False, "No CUDA-enabled devices detected"
            
            return True, f"CUDA available with {device_count} device(s)"
        except Exception as e:
            return False, f"CUDA module present but unusable: {e}"
    
    except ImportError:
        return False, "cv2 not installed"


def has_cuda_build() -> tuple:
    """
    Check whether the loaded OpenCV build includes CUDA support.

    This is intentionally weaker than runtime device verification and is useful
    for UI gating when the worker process performs the authoritative runtime
    validation.
    """
    configured_opencv_bin = os.environ.get('MEGAGLITCH_OPENCV_BIN', DEFAULT_OPENCV_BIN)
    configured_opencv_python = os.environ.get('MEGAGLITCH_OPENCV_PYTHON', DEFAULT_OPENCV_PYTHON)

    # For this app, the worker process is the authoritative runtime verifier.
    # If the configured custom OpenCV-CUDA build is present on disk, expose CUDA
    # controls and let the worker confirm device access.
    if os.path.isdir(configured_opencv_bin) and os.path.isdir(configured_opencv_python):
        return True, "Configured OpenCV-CUDA build found"

    try:
        import cv2
    except ImportError:
        return False, "cv2 not installed"

    if not hasattr(cv2, 'cuda'):
        return False, "cv2.cuda module not found"

    if _has_cuda_build(cv2):
        cv2_path = str(getattr(cv2, "__file__", "") or "")
        if "opencv-cuda" in cv2_path.lower() or "build_py312" in cv2_path.lower():
            return True, "Custom OpenCV-CUDA build loaded"
        return True, "CUDA-capable OpenCV build loaded"

    return False, "OpenCV build does not report CUDA support"


def get_cuda_info() -> dict:
    """
    Get detailed CUDA/OpenCV information for diagnostics.
    
    Returns:
        dict with CUDA configuration details
    """
    info = {
        'bootstrap_complete': _bootstrap_complete,
        'cuda_verified': _cuda_verified,
        'cuda_version': CUDA_VERSION,
        'opencv_bin': DEFAULT_OPENCV_BIN,
        'opencv_python': DEFAULT_OPENCV_PYTHON,
        'cuda_path': DEFAULT_CUDA_PATH,
        'cuda_bin': DEFAULT_CUDA_BIN,
        'status_message': _cuda_status_message
    }
    
    try:
        import cv2
        info['opencv_version'] = cv2.__version__
        info['has_cuda_module'] = hasattr(cv2, 'cuda')
        
        if hasattr(cv2, 'cuda'):
            info['cuda_device_count'] = _get_cuda_device_count(cv2)
    except:
        pass
    
    return info


if __name__ == "__main__":
    # Test bootstrap and run self-test
    print("=" * 60)
    print("OpenCV CUDA Bootstrap Self-Test")
    print("=" * 60)
    print(f"CUDA Version: {CUDA_VERSION}")
    print(f"OpenCV Bin: {DEFAULT_OPENCV_BIN}")
    print(f"OpenCV Python: {DEFAULT_OPENCV_PYTHON}")
    print(f"CUDA Path: {DEFAULT_CUDA_PATH}")
    print(f"CUDA Bin: {DEFAULT_CUDA_BIN}")
    print()
    
    try:
        print("1. Running bootstrap...")
        bootstrap_for_dev_defaults()
        print("   [OK] Bootstrap successful")
        
        print("\n2. Running CUDA self-test...")
        results = run_cuda_selftest()
        
        print(f"   Device count: {results['device_count']}")
        for test_name, passed in results['tests'].items():
            status = "[OK]" if passed else "[FAIL]"
            print(f"   {status} {test_name}")
        
        if results['errors']:
            print("\n   Errors:")
            for error in results['errors']:
                print(f"   - {error}")
        
        print(f"\n3. Overall: {'[OK] PASS' if results['success'] else '[FAIL]'}")
        
        if results['success']:
            print("\n4. GPU effects are ready to use!")
        else:
            print("\n4. GPU effects will be disabled - falling back to CPU")
            sys.exit(1)
    
    except Exception as e:
        print(f"[FAIL] Bootstrap failed: {e}")
        sys.exit(1)
