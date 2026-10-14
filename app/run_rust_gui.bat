@echo off
title MegaGlitchNode - Rust UI Shell
cd /d "%~dp0"
if exist "%~dp0.tmp" (
    set "TEMP=%~dp0.tmp"
    set "TMP=%~dp0.tmp"
) else if exist "v:\.tmp" (
    set "TEMP=v:\.tmp"
    set "TMP=v:\.tmp"
)
set MEGAGLITCH_PYTHON=C:\Users\alexw\miniforge3\python.exe
start "" "%~dp0MegaGlitchNode-Workstation.exe"
