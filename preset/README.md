# Full-frame chaotic preset

The default export is **60 seconds at 24 fps, 2160×3840**, using a **720×1280** effect canvas, seed 1961 and the included soundtrack. Every part of the source image passes through the original Node CPU stages. The `auto` encoder checks the requested export size, uses working NVIDIA NVENC and falls back to CPU `libx264`.

From the portable package root:

```powershell
& .\runtime\python.exe -I .\preset\render.py --image .\your-image.png --output .\outputs\your-new-video.mp4
```

Use `--audio` for another soundtrack, `--silent` for no audio, and `--seconds`, `--fps`, `--size` or `--native-size` for a different export. Existing MP4 files and render-asset folders are preserved. `--check` verifies inputs without exporting.

The portable builder's **two-second check is still a full 4K export** with 48 frames and the default effect canvas. It needs a 64-bit runtime and available RAM/pagefile headroom for feedback buffers and the video encoder. Allow several GB of free memory and avoid simultaneous exports. The renderer checks for at least **32 MB of free output space** for that short check; preview files, the portable runtime and ZIP need additional space. Memory use varies with the image and encoder, so this is preparation guidance rather than a strict maximum.

For a smaller diagnostic on a memory-constrained machine:

```powershell
& .\runtime\python.exe -I .\preset\render.py `
  --seconds 2 --native-size 360x640 --size 720x1280 --encoder libx264 `
  --output .\outputs\small-diagnostic.mp4
```

This exercises the same effect stages at reduced dimensions. It does not replace the builder's full 4K verification. Progress appears about every five seconds, and the finished MP4 is published only after encoding succeeds. A source checkout can use an installed Python with the runtime dependencies, FFmpeg on PATH, and `--node-root .\app`.
