# Mycotroph render-quality Python release

This package path publishes the full Python/Dear PyGui MegaGlitchNode host. It
is intentionally separate from `packaging/mycotroph-alpha`, which packages the
limited Rust functional-preview host.

The wrapper consumes an already-built PyInstaller **onedir** whose root contains
`MegaGlitchNode.exe` and `_internal`. It does not build the application and it
does not make a second full staging copy. 7-Zip reads the onedir directly; the
work root holds only a candidate ZIP, 7-Zip scratch data, and a small metadata
overlay.

Example using roomy non-system volumes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging\mycotroph-python-release\build.ps1 `
  -ArtifactRoot 'V:\MegaGlitch\dist\MegaGlitchNode' `
  -OutputRoot 'V:\MegaGlitch\release' `
  -WorkRoot 'V:\MegaGlitch\release-work' `
  -VerificationRoot 'E:\MegaGlitch\release-verify'
```

The source tree must be clean and committed. Existing archives and checksum
files are never overwritten. The artifact must include its frozen CPython and
cv2 runtimes, `_internal\bin\ffmpeg\ffmpeg.exe` and `ffprobe.exe`, and
`_internal\licenses\ffmpeg\LICENSE` and `README.txt`. It must
also include `_internal\python-host-build.json` with schema
`megaglitch.python-host-build/v1`; bundles without exact build provenance are
not releasable.

The wrapper binds that metadata to the current clean commit and current hashes
of the PyInstaller spec, runtime hook, external-tool resolver, OpenCV preload
manifest, and CUDA runtime manifest. It also verifies the frozen cv2 hash,
OpenCV/CUDA manifest identities and DLL counts, and FFmpeg/FFprobe hashes.
Metadata that reports a dirty build, a different commit, stale source inputs,
or changed bundle binaries is rejected before archiving. The same checks run
against the fresh extraction.

Before archiving and again from a fresh extraction, the release gate runs:

```text
MegaGlitchNode.exe --diagnose-only
MegaGlitchNode.exe --diagnose-worker
```

Both run with `MEGAGLITCH_RUNTIME_BACKEND=python` and
`MEGAGLITCH_FRAME_TRANSPORT=shared` in an environment stripped of development
Python, Conda, virtual-environment, OpenCV, CUDA, native-runtime, and UI-bridge
overrides. The gate also executes the bundled `ffmpeg.exe -hide_banner
-encoders` and `ffprobe.exe -version` under a Windows-system-only `PATH`, and
requires `libx264`, `libx265`, `mpeg4`, and `ffv1`. The extracted bundle is also
checked against `BUNDLE-SHA256.txt`.

Frozen UI and worker processes write PID-qualified logs beneath
`%LOCALAPPDATA%\MegaGlitchNode\logs`, with a process-specific temporary path as
fallback. Diagnostics therefore cannot mutate the bundle or make concurrent
processes compete for a bundle-root `glitch_app.log`. Source launches retain
the established repository-relative log.

The result is a 7-Zip-written ZIP (with automatic ZIP64 extensions whenever
classic ZIP limits are crossed), a `.sha256` sidecar, and these files inside the
single onedir archive root:

- `BUILD-INFO.txt`
- `BUNDLE-SHA256.txt`

`BUILD-INFO.txt` records exact FFmpeg/FFprobe version banners and SHA-256
identities, license/README hashes, and the verified source-archive SHA-256.

The verifier can also be run independently against an onedir:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging\mycotroph-python-release\verify.ps1 `
  -PackageRoot 'V:\MegaGlitch\dist\MegaGlitchNode' `
  -RequireBundledFfmpeg
```

An independent invocation always validates the metadata schema, clean-build
claim, and bundled binary identities. The release wrapper additionally supplies
the expected clean commit and source-file hashes, so only the wrapper performs
the complete checkout-to-artifact provenance comparison used for publication.

Temporary work and extraction directories use per-run GUID names and ownership
markers. Recursive cleanup is refused unless the resolved directory is a strict
child of the requested root, is not a reparse point, and has the exact matching
marker. The artifact itself is read-only to this workflow.
