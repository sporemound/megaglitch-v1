"""Regression checks for source discovery and the known NVENC probe defect."""
from pathlib import Path
import importlib.util
import os
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_auto_encoder_probes_the_actual_4k_export_size(monkeypatch):
    renderer = load(ROOT / 'preset/render.py', 'renderer_4k_probe_test')
    seen = []

    def encode(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(renderer, 'run_hidden', encode)
    assert renderer.choose_encoder(Path('ffmpeg.exe'), 'auto', (2160, 3840)) == 'h264_nvenc'
    assert 'color=size=2160x3840:rate=1' in seen[0]


def test_auto_encoder_falls_back_after_a_rejected_encode(monkeypatch):
    renderer = load(ROOT / 'preset/render.py', 'renderer_cpu_fallback_test')
    monkeypatch.setattr(renderer, 'run_hidden', lambda *args, **kwargs: SimpleNamespace(returncode=1))
    assert renderer.choose_encoder(Path('ffmpeg.exe'), 'auto', (32, 32)) == 'libx264'


def test_source_launcher_preserves_ffmpeg_discovery_and_selects_cpu(monkeypatch, tmp_path):
    # Execute the real launcher from a source-shaped folder, without opening a UI.
    source = tmp_path / 'source copy'
    source.mkdir()
    launcher = source / 'launch.py'
    launcher.write_bytes((ROOT / 'launch.py').read_bytes())
    explicit = str(tmp_path / 'chosen-ffmpeg.exe')
    monkeypatch.setenv('IMAGEIO_FFMPEG_EXE', explicit)
    monkeypatch.setattr(os, 'chdir', lambda path: None)
    monkeypatch.setattr(sys, 'argv', [str(launcher)])
    monkeypatch.setattr(sys, 'path', sys.path[:])
    for name in ('MEGAGLITCH_RUNTIME_BACKEND', 'MEGAGLITCH_FRAME_TRANSPORT', 'MEGAGLITCH_OPENCV_BIN', 'MEGAGLITCH_OPENCV_PYTHON', 'MEGAGLITCH_CUDA_PATH', 'MEGAGLITCH_CUDA_BIN'):
        monkeypatch.setenv(name, 'old-development-value')
    load(launcher, 'source_launcher_test')
    assert os.environ['IMAGEIO_FFMPEG_EXE'] == explicit
    assert os.environ['MEGAGLITCH_OPENCV_BIN'] == str(source / 'unavailable-optional-gpu')
    assert os.environ['MEGAGLITCH_OPENCV_PYTHON'] == str(source / 'unavailable-optional-gpu')


def test_portable_launcher_uses_its_existing_bundled_ffmpeg(monkeypatch, tmp_path):
    package = tmp_path / 'portable copy'
    (package / 'tools').mkdir(parents=True)
    bundled = package / 'tools/ffmpeg.exe'
    bundled.write_bytes(b'fixture')
    launcher = package / 'launch.py'
    launcher.write_bytes((ROOT / 'launch.py').read_bytes())
    monkeypatch.setenv('IMAGEIO_FFMPEG_EXE', 'old-development-value')
    monkeypatch.setattr(os, 'chdir', lambda path: None)
    monkeypatch.setattr(sys, 'argv', [str(launcher)])
    monkeypatch.setattr(sys, 'path', sys.path[:])
    for name in ('MEGAGLITCH_RUNTIME_BACKEND', 'MEGAGLITCH_FRAME_TRANSPORT', 'MEGAGLITCH_OPENCV_BIN', 'MEGAGLITCH_OPENCV_PYTHON', 'MEGAGLITCH_CUDA_PATH', 'MEGAGLITCH_CUDA_BIN'):
        monkeypatch.setenv(name, 'old-development-value')
    load(launcher, 'portable_launcher_test')
    assert os.environ['IMAGEIO_FFMPEG_EXE'] == str(bundled)
