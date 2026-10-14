# -*- mode: python ; coding: utf-8 -*-
"""Environment-configurable full Python/Dear PyGui Windows bundle."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pefile
import PyInstaller
from PyInstaller.utils.hooks import collect_submodules


ROOT = Path(SPECPATH).resolve().parents[1]
OPENCV_BIN = Path(
    os.getenv(
        "MEGAGLITCH_OPENCV_BIN",
        r"C:\src\opencv-cuda\build_py312\bin\Release",
    )
)
OPENCV_PYTHON = Path(
    os.getenv(
        "MEGAGLITCH_OPENCV_PYTHON",
        r"C:\src\opencv-cuda\build_py312\lib\python3\Release",
    )
)
CUDA_BIN = Path(
    os.getenv(
        "MEGAGLITCH_CUDA_BIN",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    )
)
FFMPEG_BIN_VALUE = os.getenv("MEGAGLITCH_FFMPEG_BIN", "").strip()
if not FFMPEG_BIN_VALUE:
    raise SystemExit(
        "MEGAGLITCH_FFMPEG_BIN must name an audited x64 FFmpeg bin directory "
        "containing ffmpeg.exe and ffprobe.exe"
    )
FFMPEG_BIN = Path(FFMPEG_BIN_VALUE)
FFMPEG_DISTRIBUTION_ROOT = FFMPEG_BIN.parent
FFMPEG_SOURCE_ARCHIVE_SHA256 = os.getenv(
    "MEGAGLITCH_FFMPEG_SOURCE_ARCHIVE_SHA256", ""
).strip().lower()
if not re.fullmatch(r"[0-9a-f]{64}", FFMPEG_SOURCE_ARCHIVE_SHA256):
    raise SystemExit(
        "MEGAGLITCH_FFMPEG_SOURCE_ARCHIVE_SHA256 must record the 64-character "
        "SHA-256 of the audited FFmpeg distribution archive"
    )
BUNDLE_NAME = os.getenv("MEGAGLITCH_BUNDLE_NAME", "MegaGlitchNode").strip()
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", BUNDLE_NAME):
    raise SystemExit(f"Unsafe MEGAGLITCH_BUNDLE_NAME: {BUNDLE_NAME!r}")

CUDA_RUNTIME_PREFIXES = (
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
)
CUDA_DRIVER_DLL_NAMES = {
    "nvcuda.dll",
    "nvapi64.dll",
    "nvcuvid.dll",
    "nvencodeapi64.dll",
    "nvml.dll",
    "nvoptix.dll",
}
QT_BINDING_COMPONENTS = {"pyqt5", "pyqt6", "pyside2", "pyside6"}
QT_RUNTIME_DLL_PATTERN = re.compile(r"^qt[56].*\.dll$", re.IGNORECASE)

for label, directory in (
    ("OpenCV DLL", OPENCV_BIN),
    ("OpenCV Python", OPENCV_PYTHON),
    ("CUDA Toolkit", CUDA_BIN),
    ("FFmpeg", FFMPEG_BIN),
):
    if not directory.is_dir():
        raise SystemExit(f"{label} directory not found: {directory}")

opencv_dlls = sorted(OPENCV_BIN.glob("opencv_*.dll"))
cv2_extensions = sorted(OPENCV_PYTHON.glob("cv2*.pyd"))
if not opencv_dlls:
    raise SystemExit(f"No opencv_*.dll files found in {OPENCV_BIN}")
if len(cv2_extensions) != 1:
    raise SystemExit(
        f"Expected one cv2 extension in {OPENCV_PYTHON}, found {len(cv2_extensions)}"
    )

ffmpeg_executable = FFMPEG_BIN / "ffmpeg.exe"
ffprobe_executable = FFMPEG_BIN / "ffprobe.exe"
for executable in (ffmpeg_executable, ffprobe_executable):
    if not executable.is_file():
        raise SystemExit(f"Required FFmpeg executable not found: {executable}")
ffmpeg_license = FFMPEG_DISTRIBUTION_ROOT / "LICENSE"
ffmpeg_readme = FFMPEG_DISTRIBUTION_ROOT / "README.txt"
for document in (ffmpeg_license, ffmpeg_readme):
    if not document.is_file():
        raise SystemExit(
            "The audited FFmpeg distribution must retain its root LICENSE and "
            f"README.txt: missing {document}"
        )

preload_manifest = ROOT / "packaging" / "pyinstaller" / "opencv-preload-manifest.json"
try:
    preload_names = json.loads(preload_manifest.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"Could not read OpenCV preload manifest {preload_manifest}: {error}")
if (
    not isinstance(preload_names, list)
    or not preload_names
    or any(not isinstance(name, str) or Path(name).name != name for name in preload_names)
    or len({name.lower() for name in preload_names}) != len(preload_names)
):
    raise SystemExit(f"Invalid OpenCV preload manifest: {preload_manifest}")

cuda_runtime_manifest = ROOT / "packaging" / "pyinstaller" / "cuda-runtime-manifest.json"
try:
    cuda_runtime_names = json.loads(cuda_runtime_manifest.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(
        f"Could not read CUDA runtime manifest {cuda_runtime_manifest}: {error}"
    )
if (
    not isinstance(cuda_runtime_names, list)
    or not cuda_runtime_names
    or any(
        not isinstance(name, str)
        or Path(name).name != name
        or not name.lower().endswith(".dll")
        for name in cuda_runtime_names
    )
    or len({name.lower() for name in cuda_runtime_names}) != len(cuda_runtime_names)
    or any(name.lower() in CUDA_DRIVER_DLL_NAMES for name in cuda_runtime_names)
):
    raise SystemExit(f"Invalid CUDA runtime manifest: {cuda_runtime_manifest}")


def pe_imports(path):
    image = pefile.PE(str(path), fast_load=True)
    try:
        if image.FILE_HEADER.Machine != pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
            raise SystemExit(f"Expected an x64 PE image: {path}")
        image.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
            ]
        )
        imports = []
        for directory_name in (
            "DIRECTORY_ENTRY_IMPORT",
            "DIRECTORY_ENTRY_DELAY_IMPORT",
        ):
            imports.extend(
                entry.dll.decode("ascii")
                for entry in getattr(image, directory_name, ())
            )
        return imports
    finally:
        image.close()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments):
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="strict",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise SystemExit(
            f"Git metadata command failed ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout.strip()


def is_cuda_runtime_dependency(name):
    lowered = name.lower()
    return lowered.endswith(".dll") and lowered.startswith(CUDA_RUNTIME_PREFIXES)


# Validate the checked-in dependency-first OpenCV manifest against this exact
# custom cv2 build. The runtime hook imports only this validated list.
opencv_by_name = {path.name.lower(): path for path in opencv_dlls}
opencv_imports_by_name = {
    name: pe_imports(path) for name, path in opencv_by_name.items()
}
cv2_imports = pe_imports(cv2_extensions[0])
cv2_direct_imports = [
    name for name in cv2_imports if name.lower().startswith("opencv_")
]
if {name.lower() for name in preload_names} != {
    name.lower() for name in cv2_direct_imports
}:
    missing = sorted(
        {name.lower() for name in cv2_direct_imports}
        - {name.lower() for name in preload_names}
    )
    stale = sorted(
        {name.lower() for name in preload_names}
        - {name.lower() for name in cv2_direct_imports}
    )
    raise SystemExit(
        f"OpenCV preload manifest does not match {cv2_extensions[0].name}; "
        f"missing={missing}, stale={stale}"
    )

preload_positions = {name.lower(): index for index, name in enumerate(preload_names)}
for name in preload_names:
    dependency = opencv_by_name.get(name.lower())
    if dependency is None:
        raise SystemExit(f"OpenCV preload dependency is absent from {OPENCV_BIN}: {name}")
    for imported_name in opencv_imports_by_name[name.lower()]:
        imported_position = preload_positions.get(imported_name.lower())
        if imported_position is not None and imported_position >= preload_positions[name.lower()]:
            raise SystemExit(
                "OpenCV preload manifest is not dependency-first: "
                f"{name} imports {imported_name}"
            )

# Resolve and verify the complete CUDA Toolkit closure used by cv2 and the
# modular OpenCV DLLs. Driver-owned DLLs remain NVIDIA-driver prerequisites.
cuda_by_name = {
    path.name.lower(): path for path in CUDA_BIN.glob("*.dll") if path.is_file()
}
required_cuda_names = {}
all_opencv_imports = list(cv2_imports)
for imports in opencv_imports_by_name.values():
    all_opencv_imports.extend(imports)
pending_cuda_names = [
    name for name in all_opencv_imports if is_cuda_runtime_dependency(name)
]
cuda_imports_by_name = {}
while pending_cuda_names:
    imported_name = pending_cuda_names.pop()
    lowered_name = imported_name.lower()
    if lowered_name in required_cuda_names:
        continue
    dependency = cuda_by_name.get(lowered_name)
    if dependency is None:
        raise SystemExit(
            f"Required CUDA runtime dependency is absent from {CUDA_BIN}: {imported_name}"
        )
    imports = pe_imports(dependency)
    required_cuda_names[lowered_name] = dependency.name
    cuda_imports_by_name[lowered_name] = imports
    pending_cuda_names.extend(
        name for name in imports if is_cuda_runtime_dependency(name)
    )

manifest_cuda_names = {name.lower() for name in cuda_runtime_names}
if manifest_cuda_names != set(required_cuda_names):
    missing = sorted(set(required_cuda_names) - manifest_cuda_names)
    stale = sorted(manifest_cuda_names - set(required_cuda_names))
    raise SystemExit(
        "CUDA runtime manifest does not match the OpenCV dependency closure; "
        f"missing={missing}, stale={stale}"
    )

cuda_preload_positions = {
    name.lower(): index for index, name in enumerate(cuda_runtime_names)
}
for name in cuda_runtime_names:
    for imported_name in cuda_imports_by_name[name.lower()]:
        imported_position = cuda_preload_positions.get(imported_name.lower())
        if (
            imported_position is not None
            and imported_position >= cuda_preload_positions[name.lower()]
        ):
            raise SystemExit(
                "CUDA runtime manifest is not dependency-first: "
                f"{name} imports {imported_name}"
            )

# Reject 32-bit or otherwise mismatched FFmpeg tools at analysis time.
pe_imports(ffmpeg_executable)
pe_imports(ffprobe_executable)
try:
    ffmpeg_identity = subprocess.run(
        [str(ffmpeg_executable), "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    ffmpeg_capabilities = subprocess.run(
        [str(ffmpeg_executable), "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    ffprobe_identity = subprocess.run(
        [str(ffprobe_executable), "-version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
except (OSError, subprocess.SubprocessError) as error:
    raise SystemExit(f"Could not validate configured FFmpeg tools: {error}")
if (
    ffmpeg_identity.returncode != 0
    or ffmpeg_capabilities.returncode != 0
    or ffprobe_identity.returncode != 0
):
    raise SystemExit("Configured FFmpeg or FFprobe executable failed its build-time probe")
ffmpeg_version_lines = [
    line.strip() for line in ffmpeg_identity.stdout.splitlines() if line.strip()
]
ffprobe_version_lines = [
    line.strip() for line in ffprobe_identity.stdout.splitlines() if line.strip()
]
if not ffmpeg_version_lines or not ffmpeg_version_lines[0].lower().startswith(
    "ffmpeg version "
):
    raise SystemExit("Configured FFmpeg returned no recognizable version banner")
if not ffprobe_version_lines or not ffprobe_version_lines[0].lower().startswith(
    "ffprobe version "
):
    raise SystemExit("Configured FFprobe returned no recognizable version banner")
ffmpeg_version_match = re.match(r"(?i)^ffmpeg version (\S+)", ffmpeg_version_lines[0])
ffprobe_version_match = re.match(r"(?i)^ffprobe version (\S+)", ffprobe_version_lines[0])
if (
    ffmpeg_version_match is None
    or ffprobe_version_match is None
    or ffmpeg_version_match.group(1) != ffprobe_version_match.group(1)
):
    raise SystemExit("Configured FFmpeg and FFprobe do not report the same build version")
encoder_listing = ffmpeg_capabilities.stdout + ffmpeg_capabilities.stderr
missing_encoders = sorted(
    encoder
    for encoder in ("libx264", "libx265", "mpeg4", "ffv1")
    if not re.search(rf"(?m)^\s*[A-Z.]+\s+{re.escape(encoder)}\s", encoder_listing)
)
if missing_encoders:
    raise SystemExit(
        "Configured FFmpeg build lacks required encoders: "
        + ", ".join(missing_encoders)
    )

binaries = [(str(path), "bin/opencv") for path in opencv_dlls]
binaries.extend(
    (str(cuda_by_name[name.lower()]), "bin/cuda") for name in cuda_runtime_names
)
binaries.extend(
    (
        (str(ffmpeg_executable), "bin/ffmpeg"),
        (str(ffprobe_executable), "bin/ffmpeg"),
    )
)
binaries.append((str(cv2_extensions[0]), "."))

# Conda NumPy links through versioned MKL dispatch modules that PyInstaller's
# dependency scan can miss. Wheel environments simply have no matching files.
mkl_runtime_dir = Path(sys.prefix) / "Library" / "bin"
for mkl_runtime in sorted(mkl_runtime_dir.glob("mkl*.dll")):
    binaries.append((str(mkl_runtime), "."))

temporal_extensions = sorted((ROOT / "megaglitch").glob("_native_temporal*.pyd"))
if len(temporal_extensions) != 1:
    raise SystemExit("Build native/temporal_ring before packaging")
binaries.append((str(temporal_extensions[0]), "megaglitch"))

rust_runtime = ROOT / "megaglitch" / "_native" / "megaglitch_runtime.dll"
if rust_runtime.exists():
    binaries.append((str(rust_runtime), "megaglitch/_native"))

source_commit = git_output("rev-parse", "--verify", "HEAD").lower()
if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_commit):
    raise SystemExit(f"Git returned an invalid full commit id: {source_commit!r}")
source_dirty = bool(git_output("status", "--porcelain=v1", "--untracked-files=all"))
build_metadata_path = Path(workpath) / "python-host-build.json"
build_metadata = {
    "schema": "megaglitch.python-host-build/v1",
    "artifact": "megaglitch-python-host",
    "bundle_name": BUNDLE_NAME,
    "source_commit": source_commit,
    "source_dirty": source_dirty,
    "python_cache_tag": sys.implementation.cache_tag,
    "python_version": sys.version.split()[0],
    "pyinstaller_version": PyInstaller.__version__,
    "cv2_extension": cv2_extensions[0].name,
    "cv2_sha256": sha256_file(cv2_extensions[0]),
    "opencv_dll_count": len(opencv_dlls),
    "opencv_preload_manifest_sha256": sha256_file(preload_manifest),
    "cuda_runtime_dll_count": len(cuda_runtime_names),
    "cuda_runtime_manifest_sha256": sha256_file(cuda_runtime_manifest),
    "ffmpeg_distribution": FFMPEG_DISTRIBUTION_ROOT.name,
    "ffmpeg_source_archive_sha256": FFMPEG_SOURCE_ARCHIVE_SHA256,
    "ffmpeg_sha256": sha256_file(ffmpeg_executable),
    "ffmpeg_version": ffmpeg_version_lines[0],
    "ffprobe_sha256": sha256_file(ffprobe_executable),
    "ffprobe_version": ffprobe_version_lines[0],
    "ffmpeg_license_sha256": sha256_file(ffmpeg_license),
    "ffmpeg_readme_sha256": sha256_file(ffmpeg_readme),
    "spec_sha256": sha256_file(ROOT / "packaging" / "pyinstaller" / "megaglitch.spec"),
    "runtime_hook_sha256": sha256_file(
        ROOT / "packaging" / "pyinstaller" / "runtime_hook.py"
    ),
    "external_tools_sha256": sha256_file(ROOT / "megaglitch" / "external_tools.py"),
}
build_metadata_path.write_text(
    json.dumps(build_metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

datas = [
    (str(preload_manifest), "bin/opencv"),
    (str(cuda_runtime_manifest), "bin/cuda"),
    (str(build_metadata_path), "."),
    (str(ffmpeg_license), "licenses/ffmpeg"),
    (str(ffmpeg_readme), "licenses/ffmpeg"),
]
for relative in (
    "presets",
    "node_presets",
    "shaders",
    "lua_glsl_demo/shaders",
    "lua_glsl_demo/scripts",
    "svg_templates",
):
    source = ROOT / relative
    if source.exists():
        datas.append((str(source), relative))

quantum_model = ROOT / "quantum_gestalt" / "quantum_gestalt_model.pt"
if quantum_model.is_file():
    datas.append((str(quantum_model), "quantum_gestalt"))

hiddenimports = [
    "numpy",
    "dearpygui",
    "PIL",
    "mss",
    "pygetwindow",
    "pyvirtualcam",
    "imageio",
    "moderngl",
    "glfw",
    "lupa",
    "sounddevice",
    "mido",
    "rtmidi",
    "cairosvg",
    "noise",
    "ffmpeg",
    # Lazy root-level effect imports used by the established Python machine.
    "faultline_possession",
    "quantum_gestalt",
    "svg_generator",
    "svg_particle_engine",
    "genesis_vdp_renderer",
]
hiddenimports.extend(collect_submodules("megaglitch"))
hiddenimports.extend(collect_submodules("megaglitchnode"))
hiddenimports.extend(collect_submodules("lupa"))
hiddenimports.extend(collect_submodules("quantum_gestalt"))
# NumPy 2.3 imports private pure modules dynamically during initialization.
hiddenimports.extend(collect_submodules("numpy._core"))

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT), str(OPENCV_PYTHON)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "pyinstaller" / "runtime_hook.py")],
    # MegaGlitch uses Dear PyGui. Development environments may also contain
    # several unrelated Qt stacks; collecting more than one makes PyInstaller
    # abort and none is needed by the supported host.
    excludes=["cv2", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
)

# Remove dependency-scan collisions so only the explicitly isolated custom
# OpenCV and CUDA closure can satisfy those names at runtime.
opencv_names = {path.name.lower() for path in opencv_dlls}
cuda_runtime_name_set = {name.lower() for name in cuda_runtime_names}


def isolated_binary(entry_name, directory, expected_names):
    normalized = entry_name.replace("\\", "/").lower()
    return normalized.startswith(f"{directory}/") and Path(normalized).name in expected_names


def is_unrelated_qt_binary(entry_name):
    """Reject Qt runtimes pulled from a development environment by hook scans."""

    normalized = entry_name.replace("\\", "/")
    components = {component.lower() for component in normalized.split("/")}
    return bool(QT_RUNTIME_DLL_PATTERN.fullmatch(Path(normalized).name)) or bool(
        components & QT_BINDING_COMPONENTS
    )


a.binaries = type(a.binaries)(
    entry
    for entry in a.binaries
    if not (
        (
            Path(entry[0].replace("\\", "/")).name.lower() in opencv_names
            and not isolated_binary(entry[0], "bin/opencv", opencv_names)
        )
        or (
            Path(entry[0].replace("\\", "/")).name.lower()
            in cuda_runtime_name_set
            and not isolated_binary(entry[0], "bin/cuda", cuda_runtime_name_set)
        )
        or Path(entry[0].replace("\\", "/")).name.lower()
        in CUDA_DRIVER_DLL_NAMES
        or entry[0].replace("\\", "/").lower().startswith("cv2/")
        or is_unrelated_qt_binary(entry[0])
    )
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MegaGlitchNode",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=BUNDLE_NAME,
)
