@echo off
setlocal
title MegaGlitch Full-Frame Export
cd /d "%~dp0"
call "%~dp0runtime-env.cmd"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0render-picker.ps1" %*
if errorlevel 1 (
  echo.
  echo Export failed. The original image and completed exports are preserved.
  pause
)
