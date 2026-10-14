from pathlib import Path
import importlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
modules = ['numpy', 'cv2', 'scipy.signal', 'PIL.Image', 'dearpygui.dearpygui', 'mido', 'rtmidi', 'sounddevice', 'lupa', 'moderngl', 'glfw', 'cairosvg', 'imageio', 'pyvirtualcam', 'mss', 'psutil', 'noise']
results = {'python': sys.version, 'executable': sys.executable, 'imports': {}, 'tools': {}}
ok = True
for name in modules:
    try:
        module = importlib.import_module(name)
        results['imports'][name] = {'ok': True, 'path': getattr(module, '__file__', None)}
    except Exception as error:
        ok = False
        results['imports'][name] = {'ok': False, 'error': str(error)}
for name in ('ffmpeg', 'ffprobe'):
    result = subprocess.run([str(ROOT / 'tools' / (name + '.exe')), '-version'], capture_output=True, text=True)
    results['tools'][name] = {'ok': result.returncode == 0, 'version': result.stdout.splitlines()[0] if result.stdout else result.stderr}
    ok = ok and result.returncode == 0
result = subprocess.run([sys.executable, '-I', str(ROOT / 'preset/render.py'), '--output', str(ROOT / 'outputs/installation-check-example.mp4'), '--check'], capture_output=True, text=True)
results['preset'] = {'ok': result.returncode == 0, 'details': result.stdout, 'errors': result.stderr}
ok = ok and result.returncode == 0
results['ok'] = ok
(ROOT / 'installation-check.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
print('\nInstallation check ' + ('PASSED.' if ok else 'FAILED. See installation-check.json.'))
sys.exit(0 if ok else 1)
