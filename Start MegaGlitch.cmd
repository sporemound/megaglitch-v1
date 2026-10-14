@echo off
setlocal
title MegaGlitch Full-Frame
cd /d "%~dp0"
call "%~dp0runtime-env.cmd"
"%~dp0runtime\python.exe" -I "%~dp0launch.py" %*
if errorlevel 1 (
  echo.
  echo MegaGlitch could not start. Run Check Installation.cmd for details.
  pause
)
