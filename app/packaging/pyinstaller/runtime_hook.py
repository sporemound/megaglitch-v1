"""Register isolated OpenCV/CUDA DLL directories before application imports."""

import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


# The bridge owns stdout as a strict JSON-lines protocol.  Some legacy audio
# imports reach pygame while the frozen module graph initializes; suppress its
# support banner before any application modules can be imported.
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def _write_frozen_exception(exc_type, exc_value, exc_traceback):
    """Persist child bootstrap failures that have no usable console handle."""
    try:
        destination = Path(tempfile.gettempdir()) / f"megaglitch-frozen-{os.getpid()}.log"
        destination.write_text(
            f"argv={sys.argv!r}\n"
            + "".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
            encoding="utf-8",
        )
    finally:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


sys.excepthook = _write_frozen_exception


bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
opencv_bin = bundle_dir / "bin" / "opencv"
cuda_bin = bundle_dir / "bin" / "cuda"
portaudio_bin = bundle_dir / "_sounddevice_data" / "portaudio-binaries"
scipy_bin = bundle_dir / "scipy.libs"

# Frozen processes must not re-register or advertise a development Toolkit
# when main.py later calls the shared bootstrap. Keep both bootstrap overrides
# pinned to the already validated bundle directory.
os.environ["MEGAGLITCH_CUDA_BIN"] = str(cuda_bin)
os.environ["MEGAGLITCH_CUDA_PATH"] = str(cuda_bin)

handles = []
# PyInstaller places wheel-owned native dependencies beside their packages.
# Register these directories before importing cv2: cv2 imports NumPy, whose
# ``_multiarray_umath`` extension depends on DLLs under ``numpy.libs``.  The
# frozen interpreter does not inherit wheel installers' normal DLL bootstrap
# behavior early enough for this runtime hook.
native_dependency_dirs = (
    bundle_dir / "numpy.libs",
    scipy_bin,
    bundle_dir / "torch" / "lib",
    # sounddevice's CFFI loader opens this bundled DLL by absolute path, but
    # Windows still needs its directory registered under the frozen loader's
    # restricted DLL search policy.
    portaudio_bin,
    cuda_bin,
    opencv_bin,
)
for directory in native_dependency_dirs:
    if directory.is_dir():
        handles.append(os.add_dll_directory(str(directory)))

# SciPy's first native dependency load can fail with ERROR_INVALID_HANDLE when
# the frozen worker is created with Windows CREATE_NO_WINDOW, just like CFFI's
# first PortAudio load below.  _fblas directly imports the wheel-owned OpenBLAS
# DLL, so preload that exact, uniquely versioned library and retain its handle
# before application imports can reach scipy.signal/scipy.linalg.
scipy_openblas_candidates = tuple(sorted(scipy_bin.glob("libscipy_openblas*.dll")))
if len(scipy_openblas_candidates) != 1:
    raise RuntimeError(
        "Expected exactly one bundled SciPy OpenBLAS runtime in "
        f"{scipy_bin}, found {len(scipy_openblas_candidates)}"
    )
scipy_openblas_path = scipy_openblas_candidates[0].resolve(strict=True)
try:
    scipy_openblas_module_handle = ctypes.WinDLL(str(scipy_openblas_path))
except OSError as error:
    raise RuntimeError(
        f"Could not preload bundled SciPy OpenBLAS runtime {scipy_openblas_path}: {error}"
    ) from error

# CFFI's first LoadLibrary call for sounddevice can fail with ERROR_INVALID_HANDLE
# when the frozen worker is created with Windows CREATE_NO_WINDOW, despite the
# absolute wheel-owned DLL path being valid. Load the exact variant selected by
# sounddevice before application imports and retain it for process lifetime.
portaudio_variant = "libportaudio64bit"
if "SD_ENABLE_ASIO" in os.environ:
    portaudio_variant += "-asio"
portaudio_path = (portaudio_bin / f"{portaudio_variant}.dll").resolve(strict=True)
try:
    portaudio_module_handle = ctypes.WinDLL(str(portaudio_path))
except OSError as error:
    raise RuntimeError(
        f"Could not preload bundled PortAudio runtime {portaudio_path}: {error}"
    ) from error


def _preload_manifest(
    directory,
    manifest_name,
    label,
    allowed_prefixes,
    forbidden_names=(),
):
    """Load a validated dependency-first DLL manifest from one bundle directory."""

    resolved_directory = directory.resolve(strict=True)
    manifest = resolved_directory / manifest_name
    with manifest.open("r", encoding="utf-8") as manifest_file:
        dependency_names = json.load(manifest_file)
    if (
        not isinstance(dependency_names, list)
        or not dependency_names
        or len({str(name).lower() for name in dependency_names}) != len(dependency_names)
    ):
        raise RuntimeError(f"Invalid {label} preload manifest: {manifest}")

    module_handles = []
    for dependency_name in dependency_names:
        lowered_name = dependency_name.lower() if isinstance(dependency_name, str) else ""
        if (
            not isinstance(dependency_name, str)
            or Path(dependency_name).name != dependency_name
            or not lowered_name.endswith(".dll")
            or not lowered_name.startswith(tuple(allowed_prefixes))
            or lowered_name in forbidden_names
        ):
            raise RuntimeError(
                f"Unsafe {label} preload entry {dependency_name!r} in {manifest}"
            )
        dependency = (resolved_directory / dependency_name).resolve(strict=True)
        if dependency.parent != resolved_directory or not dependency.is_file():
            raise RuntimeError(f"Missing required {label} preload dependency: {dependency}")
        try:
            module_handles.append(
                ctypes.WinDLL(
                    str(dependency),
                    winmode=0x00000100 | 0x00001000,
                )
            )
        except OSError as error:
            raise RuntimeError(
                f"Could not preload {label} dependency {dependency}: {error}"
            ) from error
    return module_handles


# CUDA Toolkit redistributables are part of the bundle and are loaded before
# OpenCV. Driver-owned APIs (notably nvcuda.dll) remain display-driver
# prerequisites and are intentionally absent from this directory and manifest.
cuda_module_handles = _preload_manifest(
    cuda_bin,
    "cuda-runtime-manifest.json",
    "CUDA runtime",
    (
        "cublas",
        "cudart",
        "cudnn",
        "cufft",
        "cufile",
        "cupti",
        "curand",
        "cusolver",
        "cusparse",
        "npp",
        "nvblas",
        "nvfatbin",
        "nvjitlink",
        "nvjpeg",
        "nvrtc",
    ),
    {
        "nvcuda.dll",
        "nvapi64.dll",
        "nvcuvid.dll",
        "nvencodeapi64.dll",
        "nvml.dll",
        "nvoptix.dll",
    },
)

# The custom cv2 extension directly imports an unusually wide set of modular
# OpenCV DLLs. Registering the directory alone is not sufficient on every NTFS
# volume: the Windows loader can reject the extension with
# ERROR_BAD_EXE_FORMAT even though each dependency is a valid x64 binary. The
# build validates this manifest against cv2's PE import table and orders it with
# dependencies first. Do not broaden it to every opencv_*.dll: the bundle also
# contains optional modules (notably the FFmpeg plugin) whose platform
# dependencies must not become engine-start requirements.
opencv_module_handles = _preload_manifest(
    opencv_bin,
    "opencv-preload-manifest.json",
    "OpenCV",
    ("opencv_",),
)

# Import the extension completely while its isolated DLL search paths are
# authoritative. A raw LoadLibrary/ctypes preload is insufficient because it
# leaves Python/NumPy initialization until after other runtime hooks. Dependency
# preloading above is separate; importing cv2 here still completes its Python
# and NumPy initialization before the application or a multiprocessing child
# can unpickle frame-path modules.
import cv2 as bundled_cv2  # noqa: E402 - DLL bootstrap must run first

# Runtime-hook globals are not a public lifetime contract. Retain the DLL
# directory cookies and module explicitly for the process lifetime.
sys._megaglitch_dll_directory_handles = handles
sys._megaglitch_scipy_openblas_module_handle = scipy_openblas_module_handle
sys._megaglitch_portaudio_module_handle = portaudio_module_handle
sys._megaglitch_cuda_module_handles = cuda_module_handles
sys._megaglitch_opencv_module_handles = opencv_module_handles
sys._megaglitch_bundled_cv2 = bundled_cv2
