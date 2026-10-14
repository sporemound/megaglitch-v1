@echo off
title MegaGlitchNode 0.9.4
cd /d "%~dp0"
set PATH=C:\src\opencv-cuda\build_py312\bin\Release;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%
set PYTHONPATH=%~dp0;C:\src\opencv-cuda\build_py312\lib\python3\Release;%PYTHONPATH%
"C:\Users\alexw\miniforge3\python.exe" main.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
