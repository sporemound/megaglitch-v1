# MegaGlitch V1

MegaGlitch is a Windows image/video glitch workstation with a Python/Dear PyGui editor and a deterministic full-frame export preset. The preset applies liquid warping, recursive feedback, RGB separation and broken strips/tiles across the entire image, including lettering and edges.

## Portable Windows build

Follow [the portable build instructions](packaging/portable/README.md) to assemble a self-contained download. The completed portable folder includes Python, dependencies and FFmpeg; it does not require a separate Python installation.

After extracting the portable ZIP:

- **Start MegaGlitch.cmd** opens the editor with CPU effects.
- **Render Full-Frame.cmd** lets you choose an image and save a new MP4. You can also drag an image onto it.
- **Start MegaGlitch (GPU).cmd** selects the optional custom CUDA editor libraries, when included in the build.
- **Check Installation.cmd** checks the packaged dependencies, media tools and preset.

The full-frame preset defaults to 60 seconds, 24 fps and vertical 2160 × 3840 output, with the supplied stereo soundtrack. Effects run at 720 × 1280 and are enlarged with Lanczos, matching the reference video's processing/export dimensions. NVIDIA video encoding is selected when a real encode probe succeeds; CPU H.264 is the fallback. Existing exports are preserved.

The default editor uses the supported CPU fallback because loading the optional CUDA libraries can exceed available Windows virtual memory. This does not change the full-frame preset's effects or prevent NVIDIA video encoding.

## Run from source

Use 64-bit Python 3.12. Install the application's CPU dependencies in an isolated environment, then run the root launcher:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r app\requirements-cpu-opencv.txt
.\.venv\Scripts\python.exe launch.py
```

Some integrations require external devices, drivers or native libraries. In particular, CairoSVG requires a Cairo DLL on Windows; the portable builder retains the native dependency closure from its explicit Python runtime input. CUDA, virtual cameras, CDP and experimental native/Quantum Gestalt components remain optional. See [the original application README](app/README.md) and the portable build guide for those boundaries.

To render without opening the editor, make FFmpeg available on PATH or provide `--ffmpeg`:

```powershell
.\.venv\Scripts\python.exe preset\render.py --image "your-image.png" --output "new-video.mp4"
.\.venv\Scripts\python.exe preset\render.py --help
```

The poster, editable SVG and original 60-second soundtrack are in `preset/`. The default timeline records 47 control gestures and 14 history flushes. `--seconds`, `--fps`, `--size`, `--native-size`, `--seed`, `--audio`, `--silent` and `--encoder` allow explicit export choices.

## Verification

[Verification notes](VERIFICATION.md) distinguish source tests, local portable-package tests and remaining release limits. The upstream application's release-convergence documents remain in `app/`; this import does not mark every upstream release gate certified.

## Source and license

The application source was imported from [sporemound/megaglitch](https://github.com/sporemound/megaglitch), inspected parent commit `ff8b53c9376c271c1b7ce8ca95f38065f3073603`. The engine files retain their original behavior. The root launchers, full-frame preset and portable packaging helpers are the additions in this repository.

This repository retains its existing [GPLv3 license](LICENSE). Third-party media, style references and dependency notices retain their separate terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the preset provenance files.
