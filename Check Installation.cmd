@echo off
setlocal
title MegaGlitch Installation Check
cd /d "%~dp0"
call "%~dp0runtime-env.cmd"
"%~dp0runtime\python.exe" -I "%~dp0check.py"
pause
