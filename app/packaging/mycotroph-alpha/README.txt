MEGAGLITCHNODE 00.9.4 — MYCOTROPH FUNCTIONAL PREVIEW
======================================================

ENGINE V2 CHECKPOINT — WINDOWS 10/11 x64

This build is a functional-preview artifact, not a replacement for the complete
Python/Dear PyGui MegaGlitch machine. It includes the matching Engine V2 sidecar
under `engine-v2`; keep that entire directory beside the application.

Launch
------
Double-click `MegaGlitchNode-Mycotroph-Functional-Preview.exe`. External Python
and a MegaGlitch source checkout are not required by a complete package.

Functional in Engine V2
-----------------------
- Real processed RGB program pixels in the LIVE OUTPUT pane.
- Test-pattern, still-image, and video sources.
- Image/video selection by full path or file drop.
- Run/stop and video seek-to-start.
- General primary-source graphs, including branches, generators, selected
  outputs, and supported multi-input/video-mixer/Lua socket semantics.
- Existing Python effect execution and temporal state.
- Transactional graph/source/transport commands with visible acceptance,
  rejection, effective frame, and actionable engine errors.
- Fixed-capacity mmap preview slots with bounded drop and shutdown behavior.
- Searchable NodeSpec catalog, editable graph, named sockets, and typed controls.

Not yet connected
-----------------
- Source B, Deck B, and the master transition/crossfader.
- Session/preset files beyond the in-memory legacy graph buffer.
- Automation, modulation, audio/MIDI, media-audio, recording/export, and virtual
  camera.
- Webcam/window capture and the complete backend/hardware diagnostics surface.
- Broader whole-machine parity with the Python/Dear PyGui application.

Those systems remain available in `python main.py` while their established
owners are exposed through the versioned engine service. Reserved controls do
not simulate activity.

Runtime boundary
----------------
The native executable requires `engine-v2\megaglitch-engine-v2.exe` and its
PyInstaller `_internal` runtime. A package that contains only the Rust EXE is not
functional and must not be distributed as this checkpoint. The matching CUDA
Toolkit redistributable DLL closure is isolated under
`engine-v2\_internal\bin\cuda`; the loader does not consult an external Toolkit
directory. A compatible NVIDIA display driver (which supplies driver-owned APIs
such as `nvcuda.dll`) and the Microsoft Visual C++ 2015-2022 x64 runtime remain
explicit prerequisites. Do not add a second cv2 package to the bundled engine.

Use the package build only after its bundled-engine validation succeeds on a
freshly extracted copy launched without development bridge environment variables.
The package build runs that validation automatically before staging. It can also
be run directly from the repository root:

  powershell -ExecutionPolicy Bypass -File tools\verify_engine_v2_bundle.ps1 `
    -EngineBundle build\mycotroph-engine-dist\megaglitch-engine-v2

This gate requires a protocol-v2 ready event and a valid RGB8 mmap frame, hashes
the mapped slot, releases it, performs a revisioned shutdown, and requires exit
code 0, silent stderr, JSON-only stdout, and engine-owned surface removal. It
runs the bundle with a Windows-system-only PATH and clears Python, Conda,
development bridge, and CUDA Toolkit directory overrides.

For package creation, pass `-VerificationRoot 'V:\some-directory'` to
`packaging\mycotroph-alpha\build.ps1` to force the fresh-extraction gate onto a
target or secondary volume. The builder creates and removes only a marked,
uniquely named child directory. Omitting the option uses the user temp directory.

Distribution
------------
Private functional preview. Keep this README and BUILD-INFO.txt with the build.
