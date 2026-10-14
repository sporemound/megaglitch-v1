# Portable Windows build

This recipe assembles the existing application and full-frame preset without changing the engine. The normal editor launcher uses CPU effects. The preset's `auto` encoder selects working NVIDIA NVENC at the requested output dimensions, with CPU `libx264` as the fallback. Optional GPU editor files are included only when a GPU plan is explicitly supplied; they are activated only by the GPU launcher.

## Inputs

- Windows x64 CPython **3.12**, including its standard library, DLLs, Tcl/Tk and the installed versions in `requirements-runtime.lock`. Run the builder using this same interpreter. `pefile` and `packaging` are included in the lock.
- A Windows static FFmpeg ZIP containing `ffmpeg.exe`, `ffprobe.exe` and its licenses/notices.
- An offline wheelhouse with exactly `numpy-2.2.6-cp312-cp312-win_amd64.whl`, `noise-1.2.2-cp312-cp312-win_amd64.whl`, and `opencv_python-4.12.0.88-cp37-abi3-win_amd64.whl`. These supply standalone NumPy, the compiled noise extension, and one CPU OpenCV build. The OpenCV wheel's `abi3` extension is compatible with CPython 3.12. Installed OpenCV trees and metadata are skipped entirely, so duplicate or environment-specific builds and their original DLL paths cannot enter the package. The builder checks wheel identity and ABI tags and records SHA256 values; it does not download or compile them.
- This checkout's `app/`, `preset/`, root launchers, `check.py`, `README.md` and `LICENSE`. The preset includes its source image and existing soundtrack WAV, so building does not require MegaGlitchFM. Additional root notices and source provenance are also retained when present.
- The Python input's native-library package licenses. For a conda-based input, retain its `conda-meta` records and linked package caches, including `info/licenses` and `info/about.json`. Wheel licenses remain inside their copied `.dist-info` directories. The split FreeType notices are taken from the installed same-version `freetype` package; SQLite's public-domain blessing is retained as its package declaration and `about.json`.

The lock pins the runtime dependency closure used for the verified portable editor and export. Installed dependencies are copied from the declared Python prefix instead of reinstalled from the network. Their active metadata requirements must all be present and satisfied by the lock. The resulting package records wheel/archive input hashes and every output file hash.

## Build

From the checkout root in PowerShell:

```powershell
$pythonRoot = (Resolve-Path '.\build-inputs\python312').Path
& (Join-Path $pythonRoot 'python.exe') packaging/portable/build.py `
  --python-root $pythonRoot `
  --ffmpeg-archive .\build-inputs\ffmpeg-static.zip `
  --wheelhouse .\build-inputs\wheels `
  --output-root ..\MegaGlitch-V1-Windows
```

Use `--validate-inputs` first for lightweight input and installed-version checks. It creates no package. `--source-root` and `--lock` override the default checkout and lock paths.

The full build:

1. Copies the application, preset, launchers and Python core into a new output directory.
2. Copies pinned dependency inventories and installs the three supplied standalone wheels into the new runtime.
3. Closes native dependencies through PE imports, delay imports and exported forwarders, considering bundled vendor `.libs` directories. Each package-local native binary is inspected independently. Unresolved non-driver dependencies fail the build. Windows system DLLs stay system-owned. Cairo, which is loaded dynamically, is included explicitly; runtime DLL search handles remain alive.
4. Retains Python, package, native-library and FFmpeg notices. Optional GPU files must match their manifest hashes and include their notices.
5. Runs the portable dependency check with a Windows-only search path, then renders a real **two-second 2160×3840, 24 fps H.264 video with stereo 48 kHz AAC** using the default 720×1280 effect canvas. Export dimensions, frame count, codecs and duration are checked with bundled FFprobe.
6. Writes `BUILD-INFO.json`, `BUNDLE-SHA256.txt`, verification records, a ZIP, a ZIP SHA256 sidecar and a build-result JSON. ZIP CRC and complete archive inventory are checked.

Existing package directories, downloads and checksum files are preserved. A failed build leaves its newly created directory and logs for inspection; choose a new output name or remove only that failed build directory before retrying. No application source is edited. Intermediate video frames are streamed rather than saved individually.

This is a declared-input offline recipe, not a promise that arbitrary Python prefixes produce identical binaries. Use the pinned prefix and the recorded wheels/archive to reproduce its runtime. Verification timing and hardware encoding output can vary. The checks confirm the built package on the build machine; they do not certify a clean machine or test an optional GPU editor without its compatible driver.

The two-second verification still uses the full 4K encoder and default effect canvas. It streams frames, but OpenCV's feedback buffers and the encoder need available RAM/pagefile headroom; allow several GB of free memory and avoid parallel exports. The renderer requires at least 32 MB free on the output drive for this short run; the package itself and ZIP need additional disk space. These are practical preparation guidelines, not a measured maximum-memory guarantee. See `preset/README.md` for a smaller diagnostic command.

## Optional GPU plan

Pass `--gpu-plan path/to/gpu-plan.json` to bundle an already built, compatible GPU runtime. A plan declares the input root and every included file's SHA256:

```json
{
  "source_root": "gpu-input",
  "files": {
    "python/cv2/__init__.py": "<sha256>",
    "opencv/<opencv-dll>": "<sha256>",
    "cuda/<cuda-runtime-dll>": "<sha256>",
    "licenses/<notice-file>": "<sha256>"
  }
}
```

`source_root` is absolute or relative to the plan file. The example is schematic: enumerate all actual Python/OpenCV/CUDA dependencies and notices needed by that compatible runtime. Paths must stay inside the declared input root; each hash must match. The builder copies only declared files into `gpu/` and stores the manifest without its machine-specific source path. It does not discover, download or silently enable GPU components. NVIDIA driver DLLs are supplied by the user's installed driver.
