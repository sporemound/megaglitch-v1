import json
from pathlib import Path, PurePath


ROOT = Path(__file__).resolve().parents[1]
PYINSTALLER = ROOT / "packaging" / "pyinstaller"


def test_full_python_spec_matches_runtime_hook_native_layout():
    spec = PYINSTALLER / "megaglitch.spec"
    source = spec.read_text(encoding="utf-8")
    compile(source, str(spec), "exec")

    for required in (
        "MEGAGLITCH_CUDA_BIN",
        "opencv-preload-manifest.json",
        "cuda-runtime-manifest.json",
        '"bin/opencv"',
        '"bin/cuda"',
        "CUDA_DRIVER_DLL_NAMES",
        "required_cuda_names",
        "cuda_preload_positions",
        'glob("mkl*.dll")',
        "megaglitch.python-host-build/v1",
        "source_commit",
        "source_dirty",
        "python-host-build.json",
    ):
        assert required in source

    hook = (PYINSTALLER / "runtime_hook.py").read_text(encoding="utf-8")
    assert 'bundle_dir / "bin" / "opencv"' in hook
    assert 'bundle_dir / "bin" / "cuda"' in hook
    assert 'bundle_dir / "_sounddevice_data" / "portaudio-binaries"' in hook
    assert 'scipy_bin = bundle_dir / "scipy.libs"' in hook
    assert 'scipy_bin.glob("libscipy_openblas*.dll")' in hook
    assert 'ctypes.WinDLL(str(scipy_openblas_path))' in hook
    assert (
        "sys._megaglitch_scipy_openblas_module_handle = scipy_openblas_module_handle"
        in hook
    )
    assert 'ctypes.WinDLL(str(portaudio_path))' in hook
    assert 'sys._megaglitch_portaudio_module_handle = portaudio_module_handle' in hook
    assert 'if "SD_ENABLE_ASIO" in os.environ' in hook
    assert 'os.environ["MEGAGLITCH_CUDA_BIN"] = str(cuda_bin)' in hook
    assert 'os.environ["MEGAGLITCH_CUDA_PATH"] = str(cuda_bin)' in hook


def test_full_python_spec_bundles_media_tools_and_lazy_assets():
    source = (PYINSTALLER / "megaglitch.spec").read_text(encoding="utf-8")
    for required in (
        "MEGAGLITCH_FFMPEG_BIN",
        "MEGAGLITCH_FFMPEG_SOURCE_ARCHIVE_SHA256",
        "ffmpeg.exe",
        "ffprobe.exe",
        '"bin/ffmpeg"',
        '"licenses/ffmpeg"',
        'FFMPEG_DISTRIBUTION_ROOT / "LICENSE"',
        'FFMPEG_DISTRIBUTION_ROOT / "README.txt"',
        '"ffmpeg_version"',
        '"ffprobe_version"',
        '"ffmpeg_sha256"',
        '"ffprobe_sha256"',
        '"ffmpeg_license_sha256"',
        '"ffmpeg_readme_sha256"',
        '"libx264"',
        '"libx265"',
        '"mpeg4"',
        '"ffv1"',
        "quantum_gestalt_model.pt",
        '"faultline_possession"',
        '"quantum_gestalt"',
        '"svg_generator"',
        '"svg_particle_engine"',
        '"genesis_vdp_renderer"',
        "MEGAGLITCH_BUNDLE_NAME",
        "name=BUNDLE_NAME",
    ):
        assert required in source


def test_full_python_spec_excludes_unrelated_qt_bindings():
    source = (PYINSTALLER / "megaglitch.spec").read_text(encoding="utf-8")
    assert 'excludes=["cv2", "PyQt5", "PyQt6", "PySide2", "PySide6"]' in source
    assert 'QT_BINDING_COMPONENTS = {"pyqt5", "pyqt6", "pyside2", "pyside6"}' in source
    assert 'QT_RUNTIME_DLL_PATTERN = re.compile(r"^qt[56].*\\.dll$", re.IGNORECASE)' in source
    assert "def is_unrelated_qt_binary(entry_name):" in source
    assert "or is_unrelated_qt_binary(entry[0])" in source


def test_full_python_spec_qt_binary_filter_covers_orphan_and_binding_destinations():
    source = (PYINSTALLER / "megaglitch.spec").read_text(encoding="utf-8")
    function_start = source.index("def is_unrelated_qt_binary(entry_name):")
    function_end = source.index("\n\na.binaries =", function_start)
    namespace = {
        "Path": Path,
        "QT_BINDING_COMPONENTS": {"pyqt5", "pyqt6", "pyside2", "pyside6"},
        "QT_RUNTIME_DLL_PATTERN": __import__("re").compile(
            r"^qt[56].*\.dll$", __import__("re").IGNORECASE
        ),
    }
    exec(source[function_start:function_end], namespace)
    is_unrelated_qt_binary = namespace["is_unrelated_qt_binary"]

    for rejected in (
        "Qt6Core.dll",
        "plugins/Qt5Gui.dll",
        "PyQt5/Qt5/bin/qwindows.dll",
        r"PySide6\plugins\platforms\qwindows.dll",
    ):
        assert is_unrelated_qt_binary(rejected), rejected

    for retained in (
        "bin/opencv/opencv_core4130.dll",
        "bin/cuda/cudart64_12.dll",
        "dearpygui/dearpygui.pyd",
        "assets/qt6-reference.png",
    ):
        assert not is_unrelated_qt_binary(retained), retained


def test_native_manifests_are_nonempty_unique_safe_leaf_names():
    driver_owned = {
        "nvcuda.dll",
        "nvapi64.dll",
        "nvcuvid.dll",
        "nvencodeapi64.dll",
        "nvml.dll",
        "nvoptix.dll",
    }
    for filename, prefix in (
        ("opencv-preload-manifest.json", "opencv_"),
        ("cuda-runtime-manifest.json", None),
    ):
        values = json.loads((PYINSTALLER / filename).read_text(encoding="utf-8"))
        assert isinstance(values, list) and values
        assert len({value.lower() for value in values}) == len(values)
        assert all(PurePath(value).name == value for value in values)
        assert all(value.lower().endswith(".dll") for value in values)
        if prefix:
            assert all(value.lower().startswith(prefix) for value in values)
        else:
            assert not ({value.lower() for value in values} & driver_owned)


def test_build_scripts_support_external_artifact_dist_and_work_roots():
    package_build = (PYINSTALLER / "build.ps1").read_text(encoding="utf-8")
    production_build = (ROOT / "tools" / "build_and_run.ps1").read_text(
        encoding="utf-8"
    )
    for required in (
        "ArtifactRoot",
        "DistRoot",
        "WorkRoot",
        "--distpath",
        "--workpath",
    ):
        assert required in package_build
    assert "ArtifactRoot" in production_build
    assert "releaseArtifactBase" in production_build
    assert "run_render_quality_path.py" in production_build
    assert "true_accelerated" in production_build
