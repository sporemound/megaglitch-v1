"""Use the bundled interpreter, assets and optional GPU runtime."""
from pathlib import Path
import ctypes
import os
import runpy
import sys

ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app'
sys.dont_write_bytecode = True
os.environ['MEGAGLITCH_RUNTIME_BACKEND'] = 'python'
os.environ['MEGAGLITCH_FRAME_TRANSPORT'] = 'shared'
bundled_ffmpeg = ROOT / 'tools/ffmpeg.exe'
if bundled_ffmpeg.is_file():
    os.environ['IMAGEIO_FFMPEG_EXE'] = str(bundled_ffmpeg)
gpu = ROOT / 'gpu'
cpu_requested = '--cpu' in sys.argv
if cpu_requested:
    sys.argv.remove('--cpu')
gpu_requested = '--gpu' in sys.argv
if gpu_requested:
    sys.argv.remove('--gpu')
driver_available = False
if gpu_requested and not cpu_requested:
    try:
        ctypes.WinDLL('nvcuda.dll')
        driver_available = True
    except OSError:
        pass
if gpu_requested and (gpu / 'python').is_dir() and driver_available:
    os.environ['MEGAGLITCH_OPENCV_BIN'] = str(gpu / 'opencv')
    os.environ['MEGAGLITCH_OPENCV_PYTHON'] = str(gpu / 'python')
    os.environ['MEGAGLITCH_CUDA_PATH'] = str(gpu)
    os.environ['MEGAGLITCH_CUDA_BIN'] = str(gpu / 'cuda')
else:
    for name in ('MEGAGLITCH_OPENCV_BIN', 'MEGAGLITCH_OPENCV_PYTHON', 'MEGAGLITCH_CUDA_PATH', 'MEGAGLITCH_CUDA_BIN'):
        os.environ[name] = str(ROOT / 'unavailable-optional-gpu')
sys.path.insert(0, str(APP))
os.chdir(APP)
sys.argv = [str(APP / 'main.py')] + sys.argv[1:]
if __name__ == '__main__':
    runpy.run_path(str(APP / 'main.py'), run_name='__main__')
