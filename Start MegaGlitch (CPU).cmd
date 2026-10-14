@echo off
setlocal
title MegaGlitch CPU Mode
cd /d "%~dp0"
call "%~dp0runtime-env.cmd"
"%~dp0runtime\python.exe" -I "%~dp0launch.py" --cpu %*
if errorlevel 1 (
  echo.
  echo MegaGlitch could not start. Run Check Installation.cmd for details.
  pause
)
