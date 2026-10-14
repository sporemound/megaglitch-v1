# -*- mode: python ; coding: utf-8 -*-
"""Frozen Engine V2 sidecar for the Rust Mycotroph functional preview."""

import ast
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
BRIDGE_SOURCE = ROOT / "tools" / "rust_ui_bridge.py"
PREVIEW_EXECUTOR_SOURCE = ROOT / "megaglitch" / "preview_executor.py"
OPENCV_BIN = Path(
    os.getenv("MEGAGLITCH_OPENCV_BIN", r"C:\src\opencv-cuda\build_py312\bin\Release")
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
        raise SystemExit(f"Git metadata command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout.strip()


def source_constant(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError) as error:
                raise SystemExit(f"{path} constant {name} is not a literal: {error}")
    raise SystemExit(f"{path} does not define required build constant {name}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


source_commit = git_output("rev-parse", "--verify", "HEAD").lower()
if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_commit):
    raise SystemExit(f"Git returned an invalid full commit id: {source_commit!r}")
source_status = git_output("status", "--porcelain=v1", "--untracked-files=all")
source_dirty = bool(source_status)
bridge_protocol_version = source_constant(BRIDGE_SOURCE, "PROTOCOL_VERSION")
bridge_build_id = source_constant(BRIDGE_SOURCE, "BUILD_ID")
if not PREVIEW_EXECUTOR_SOURCE.is_file():
    raise SystemExit(f"Preview executor source not found: {PREVIEW_EXECUTOR_SOURCE}")
if not isinstance(bridge_protocol_version, int) or isinstance(bridge_protocol_version, bool):
    raise SystemExit("Engine V2 PROTOCOL_VERSION must be an integer literal")
if not isinstance(bridge_build_id, str) or not bridge_build_id:
    raise SystemExit("Engine V2 BUILD_ID must be a non-empty string literal")

if not OPENCV_BIN.is_dir():
    raise SystemExit(f"OpenCV DLL directory not found: {OPENCV_BIN}")
if not OPENCV_PYTHON.is_dir():
    raise SystemExit(f"OpenCV Python directory not found: {OPENCV_PYTHON}")
if not CUDA_BIN.is_dir():
    raise SystemExit(
        "CUDA runtime DLL directory not found: "
        f"{CUDA_BIN}. Set MEGAGLITCH_CUDA_BIN to the matching Toolkit bin directory."
    )

opencv_dlls = sorted(OPENCV_BIN.glob("opencv_*.dll"))
cv2_extensions = sorted(OPENCV_PYTHON.glob("cv2*.pyd"))
if not opencv_dlls:
    raise SystemExit(f"No opencv_*.dll files found in {OPENCV_BIN}")
if len(cv2_extensions) != 1:
    raise SystemExit(
        f"Expected one cv2 extension in {OPENCV_PYTHON}, found {len(cv2_extensions)}"
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


def is_cuda_runtime_dependency(name):
    lowered = name.lower()
    return lowered.endswith(".dll") and lowered.startswith(CUDA_RUNTIME_PREFIXES)


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

# Resolve the complete CUDA Toolkit dependency closure used by cv2 and every
# bundled OpenCV module. Driver-owned DLLs such as nvcuda.dll are deliberately
# absent: the installed NVIDIA display driver remains a runtime prerequisite.
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

binaries = [(str(path), "bin/opencv") for path in opencv_dlls]
binaries.extend(
    (str(cuda_by_name[name.lower()]), "bin/cuda") for name in cuda_runtime_names
)
binaries.append((str(cv2_extensions[0]), "."))

# The development interpreter's Conda NumPy extension links through
# ``libcblas.dll``, whose forwarders load the versioned MKL dispatcher at
# runtime. PyInstaller sees an unversioned ``mkl_rt.dll`` reference, then misses
# versioned modules that the dispatcher loads by name (for example
# ``mkl_core.2.dll``). The resulting process aborts with the native delay-load
# exception 0xC06D007E when launched without the Conda Library/bin directory on
# PATH. Include the complete installed MKL runtime family; wheel-based
# environments simply have no matching files.
mkl_runtime_dir = Path(sys.prefix) / "Library" / "bin"
for mkl_runtime in sorted(mkl_runtime_dir.glob("mkl*.dll")):
    binaries.append((str(mkl_runtime), "."))

temporal_extensions = sorted((ROOT / "megaglitch").glob("_native_temporal*.pyd"))
if len(temporal_extensions) != 1:
    raise SystemExit("Build native/temporal_ring before packaging Engine V2")
binaries.append((str(temporal_extensions[0]), "megaglitch"))

rust_runtime = ROOT / "megaglitch" / "_native" / "megaglitch_runtime.dll"
if rust_runtime.exists():
    binaries.append((str(rust_runtime), "megaglitch/_native"))

engine_build_metadata_path = Path(workpath) / "engine-v2-build.json"
engine_build_metadata = {
    "schema": "megaglitch.engine-v2-build/v1",
    "artifact": "megaglitch-engine-v2",
    "source_commit": source_commit,
    "source_dirty": source_dirty,
    "bridge_build_id": bridge_build_id,
    "protocol_version": bridge_protocol_version,
    "bridge_source_sha256": sha256_file(BRIDGE_SOURCE),
    "preview_executor_source_sha256": sha256_file(PREVIEW_EXECUTOR_SOURCE),
    "python_cache_tag": sys.implementation.cache_tag,
    "python_version": sys.version.split()[0],
    "pyinstaller_version": PyInstaller.__version__,
    "cv2_extension": cv2_extensions[0].name,
    "cv2_sha256": sha256_file(cv2_extensions[0]),
    "opencv_dll_count": len(opencv_dlls),
    "opencv_preload_manifest_sha256": sha256_file(preload_manifest),
    "cuda_runtime_dll_count": len(cuda_runtime_names),
    "cuda_runtime_manifest_sha256": sha256_file(cuda_runtime_manifest),
}
engine_build_metadata_path.write_text(
    json.dumps(engine_build_metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

datas = [
    (str(preload_manifest), "bin/opencv"),
    (str(cuda_runtime_manifest), "bin/cuda"),
    (str(engine_build_metadata_path), "."),
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

# The current flat-pipeline registry contains deliberately lazy imports for
# CUDA, ModernGL/Lua, generators, SVG, and optional backends. Preserve the
# validated production closure until capability-gated package profiles exist.
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
    "faultline_possession",
    "quantum_gestalt",
    "svg_generator",
    "svg_particle_engine",
    "genesis_vdp_renderer",
]
# Do not collect the entire ``megaglitch`` package here. Engine V2 reaches a
# deliberately small subset, while blanket collection also pulls in benchmark,
# worker, and audio modules that are not sidecar entry points. Keep the lazy
# frame-path modules explicit so additions remain reviewable at this boundary.
hiddenimports.extend(
    [
        "megaglitch.bubble_sim2_cpu",
        "megaglitch.native_temporal",
        "megaglitch.nodes.bubble_sim_generator_node",
        "megaglitch.preview_executor",
        "megaglitch.session_graph_compiler",
    ]
)
hiddenimports.extend(collect_submodules("megaglitchnode"))
hiddenimports.extend(collect_submodules("lupa"))
hiddenimports.extend(collect_submodules("numpy._core"))

# The development interpreter contains desktop/notebook stacks discovered by
# PyInstaller's Torch/Matplotlib hooks even though the JSONL sidecar does not
# import them. Collecting both PyQt5 and PySide6 is a hard PyInstaller error.
# SciPy remains because the current generator boundary reaches media-audio
# imports in ``megaglitchfm_app_new``. Torch/torchvision remain because Quantum
# Gestalt effects are still supported by the registry.
engine_excludes = [
    "cv2",
    "IPython",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "matplotlib",
    "notebook",
    "pandas",
    "sklearn",
]

a = Analysis(
    [str(ROOT / "tools" / "rust_ui_bridge.py")],
    pathex=[str(ROOT), str(OPENCV_PYTHON)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "pyinstaller" / "runtime_hook.py")],
    # Supply the CUDA extension explicitly and prevent PyInstaller from
    # collecting an installed CPU cv2 package beside it.
    excludes=engine_excludes,
    noarchive=False,
)

# Dependency analysis may copy alternate OpenCV/CUDA files beside the
# explicitly isolated custom build. Remove those collisions, and never collect
# display-driver-owned DLLs, so runtime resolution stays deterministic.
opencv_names = {path.name.lower() for path in opencv_dlls}
cuda_runtime_name_set = {name.lower() for name in cuda_runtime_names}


def isolated_binary(entry_name, directory, expected_names):
    normalized = entry_name.replace("\\", "/").lower()
    return normalized.startswith(f"{directory}/") and Path(normalized).name in expected_names


a.binaries = type(a.binaries)(
    entry
    for entry in a.binaries
    if not (
        (
            Path(entry[0].replace("\\", "/")).name.lower() in opencv_names
            and not isolated_binary(entry[0], "bin/opencv", opencv_names)
        )
        or (
            Path(entry[0].replace("\\", "/")).name.lower() in cuda_runtime_name_set
            and not isolated_binary(entry[0], "bin/cuda", cuda_runtime_name_set)
        )
        or Path(entry[0].replace("\\", "/")).name.lower() in CUDA_DRIVER_DLL_NAMES
        or entry[0].replace("\\", "/").lower().startswith("cv2/")
    )
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="megaglitch-engine-v2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Engine V2 communicates over stdin/stdout JSONL. The Rust host suppresses
    # the child window; a windowed PyInstaller build may null these streams.
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="megaglitch-engine-v2",
)

final_source_commit = git_output("rev-parse", "--verify", "HEAD").lower()
final_source_status = git_output("status", "--porcelain=v1", "--untracked-files=all")
if final_source_commit != source_commit or final_source_status != source_status:
    raise SystemExit(
        "Source commit or working-tree state changed during the Engine V2 build; "
        "discard this bundle and rebuild from a stable tree."
    )

built_bundle = Path(DISTPATH) / "megaglitch-engine-v2"
built_executable = built_bundle / "megaglitch-engine-v2.exe"
built_metadata = built_bundle / "_internal" / "engine-v2-build.json"
if not built_executable.is_file() or not built_metadata.is_file():
    raise SystemExit(
        "PyInstaller output is missing the Engine V2 executable or source metadata; "
        f"bundle={built_bundle}"
    )
engine_build_metadata["engine_executable_sha256"] = sha256_file(built_executable)
serialized_engine_metadata = (
    json.dumps(engine_build_metadata, indent=2, sort_keys=True) + "\n"
)
engine_build_metadata_path.write_text(serialized_engine_metadata, encoding="utf-8")
built_metadata.write_text(serialized_engine_metadata, encoding="utf-8")
