"""Retain Windows dependency search handles inside this portable runtime."""
from pathlib import Path
import os
import sys

RUNTIME = Path(sys.executable).resolve().parent
_dll_handles = []
if hasattr(os, 'add_dll_directory'):
    for directory in (RUNTIME, RUNTIME / 'DLLs', RUNTIME / 'Library/bin'):
        if directory.is_dir():
            _dll_handles.append(os.add_dll_directory(str(directory)))
os.environ.setdefault('TCL_LIBRARY', str(RUNTIME / 'Library/lib/tcl8.6'))
os.environ.setdefault('TK_LIBRARY', str(RUNTIME / 'Library/lib/tk8.6'))
sys.dont_write_bytecode = True
