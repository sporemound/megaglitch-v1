# Native temporal ring prototype

Build from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File native\temporal_ring\build.ps1
```

This requires the active environment's NumPy, Cython, and setuptools packages plus Visual Studio C++ Build Tools. The script puts compiler intermediates under `%TEMP%` to avoid the repository's long path exceeding Windows linker limits. The extension intentionally has no OpenCV or CUDA dependency.

Select it with `MEGAGLITCH_HISTORY_BACKEND=native`; select the rollback implementation with `MEGAGLITCH_HISTORY_BACKEND=python`. The default `auto` setting uses the extension when it is built and otherwise falls back to Python.
