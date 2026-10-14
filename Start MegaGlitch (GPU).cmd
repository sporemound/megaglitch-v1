@echo off
setlocal
title MegaGlitch GPU Mode
cd /d "%~dp0"
call "%~dp0runtime-env.cmd"
"%~dp0runtime\python.exe" -I "%~dp0launch.py" --gpu %*
if errorlevel 1 (
  echo.
  echo GPU startup failed. Try Start MegaGlitch.cmd for CPU mode.
  pause
)
