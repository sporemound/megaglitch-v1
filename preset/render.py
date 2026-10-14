"""Relocatable full-frame MegaGlitch preset; streams frames straight to FFmpeg.

Default effects, seed, control gestures, feedback flushes, audio and output
dimensions reproduce the supplied 60-second Red Corona render recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import wave

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent


def dimensions(value):
    try:
        width, height = map(int, value.lower().split("x"))
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 2160x3840.") from None
    if min(width, height) < 32 or width % 2 or height % 2:
        raise argparse.ArgumentTypeError("Dimensions must be even integers of at least 32.")
    return width, height


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=Path, default=ROOT / "source-poster.png", help="Input image; any Pillow-readable format.")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "MegaGlitch-Full-Frame-Chaotic-4K-60s.mp4", help="New MP4 file; existing exports are always preserved.")
    parser.add_argument("--audio", type=Path, default=ROOT / "soundtrack.wav", help="Audio file; short inputs are padded, long inputs trimmed.")
    parser.add_argument("--silent", action="store_true", help="Export without audio; disables the audio-level modulation.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Video duration; the control timeline scales to this duration.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second.")
    parser.add_argument("--size", type=dimensions, default=(2160, 3840), help="Export WIDTHxHEIGHT; native effects are upscaled with Lanczos.")
    parser.add_argument("--native-size", type=dimensions, default=(720, 1280), help="Effect-processing WIDTHxHEIGHT; source image is resized to this canvas.")
    parser.add_argument("--seed", type=int, default=1961, help="Deterministic random seed.")
    parser.add_argument("--encoder", choices=("auto", "h264_nvenc", "libx264"), default="auto", help="Auto uses working NVIDIA encoding, otherwise CPU H.264.")
    parser.add_argument("--threads", type=int, default=4, help="OpenCV worker threads.")
    parser.add_argument("--node-root", type=Path, default=Path(os.environ.get("MEGAGLITCH_NODE_ROOT", str(PACKAGE / "app"))), help="Original MegaGlitch Node source directory.")
    parser.add_argument("--ffmpeg", type=Path, default=None, help="FFmpeg executable; default: package tools, then PATH.")
    parser.add_argument("--check", action="store_true", help="Validate dependencies and settings without creating an export.")
    result = parser.parse_args()
    if not math.isfinite(result.seconds) or result.seconds <= 0:
        parser.error("--seconds must be finite and greater than zero.")
    if not 1 <= result.fps <= 120:
        parser.error("--fps must be between 1 and 120.")
    if result.threads < 1:
        parser.error("--threads must be at least 1.")
    if round(result.fps * result.seconds) < 1:
        parser.error("The selected duration and frame rate must produce at least one frame.")
    if result.output.suffix.lower() != ".mp4":
        parser.error("--output must end with .mp4.")
    for name in ("image", "output", "audio", "node_root"):
        setattr(result, name, getattr(result, name).expanduser().resolve())
    return result


def find_ffmpeg(explicit):
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
    elif (PACKAGE / "tools" / "ffmpeg.exe").is_file():
        candidate = PACKAGE / "tools" / "ffmpeg.exe"
    elif (PACKAGE / "tools" / "ffmpeg").is_file():
        candidate = PACKAGE / "tools" / "ffmpeg"
    else:
        located = shutil.which("ffmpeg")
        if not located:
            raise FileNotFoundError("FFmpeg was not found. Bundle tools/ffmpeg.exe or supply --ffmpeg.")
        candidate = Path(located)
    if not candidate.is_file():
        raise FileNotFoundError(f"FFmpeg was not found: {candidate}")
    return candidate.resolve()


def run_hidden(command, **kwargs):
    return subprocess.run(command, creationflags=0x08000000 if sys.platform == "win32" else 0, **kwargs)


def choose_encoder(ffmpeg, requested, output_size):
    if requested != "auto":
        return requested
    try:
        probe = run_hidden([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
                            "-f", "lavfi", "-i", f"color=size={output_size[0]}x{output_size[1]}:rate=1", "-frames:v", "1",
                            "-c:v", "h264_nvenc", "-f", "null", "-"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=15)
    except subprocess.TimeoutExpired:
        return "libx264"
    return "h264_nvenc" if probe.returncode == 0 else "libx264"


def load_adapter(node_root):
    os.environ["MEGAGLITCH_NODE_ROOT"] = str(node_root)
    spec = importlib.util.spec_from_file_location("megaglitch_preset_adapter", ROOT / "node-adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_audio(ffmpeg, audio, assets, seconds):
    # Preserve the supplied PCM score sample-for-sample at the default duration.
    try:
        with wave.open(str(audio), "rb") as handle:
            if (handle.getnchannels() == 2 and handle.getsampwidth() == 2
                    and abs(handle.getnframes() / handle.getframerate() - seconds) < 1 / handle.getframerate()):
                return audio
    except (wave.Error, EOFError):
        pass
    prepared = assets / "prepared-audio.wav"
    print("Preparing soundtrack for the selected duration...", flush=True)
    result = run_hidden([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
                         "-i", str(audio), "-vn", "-af", "apad", "-t", str(seconds),
                         "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(prepared)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError("Audio preparation failed:\n" + result.stderr)
    return prepared


def soundtrack_envelope(audio, fps, count):
    import numpy as np
    with wave.open(str(audio), "rb") as handle:
        rate = handle.getframerate()
        if handle.getnchannels() != 2 or handle.getsampwidth() != 2:
            raise ValueError("Prepared audio must be stereo 16-bit PCM.")
        signal = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32)
    signal = signal.reshape(-1, 2) / 32768.0
    rms = []
    for i in range(count):
        window = signal[int(i * rate / fps):int((i + 1) * rate / fps)]
        rms.append(float(np.sqrt(np.mean(window ** 2))) if window.size else 0.0)
    rms = np.array(rms, dtype=np.float32)
    low, high = np.percentile(rms, [5, 96])
    return np.clip((rms - low) / max(high - low, 1e-6), 0, 1), rate


def control_timeline(module, seed, seconds, fps):
    import numpy as np
    rng = np.random.default_rng(seed)
    ranges = dict(module.CONTROL_RANGES)
    ranges.update({"edge_mix": (1.0, 1.0), "feedback_gain": (1.035, 1.072),
                   "feedback": (0.77, 0.875), "tile_strength": (0.004, 0.065),
                   "contour_gain": (1.8, 3.1)})
    current = dict(module.default_controls)
    current.update({"edge_mix": 1.0, "feedback_gain": 1.057, "feedback": 0.845,
                    "tile_strength": 0.014, "rgb_split": 27.0, "warp_amplitude": 43.0,
                    "noise_warp": 0.21, "strip_shift": 10.0, "line_shift": 4.0})
    initial = dict(current)
    keys = [key for key in ranges if key not in ("edge_mix", "contour_gain")]
    scale = seconds / 60.0
    times = (np.linspace(0.45, 58.65, 47) + rng.uniform(-0.27, 0.27, 47)) * scale
    events = []
    for index, t in enumerate(times):
        before = dict(current)
        names = list(rng.choice(keys, size=int(rng.integers(2, 5)), replace=False))
        updates = {}
        for name in names:
            minimum, maximum = ranges[name]
            target = float(rng.uniform(minimum, maximum))
            current[name] = target
            updates[name] = target
        events.append({"index": index, "seconds": float(t), "duration": float(rng.uniform(0.32, 0.88)) * scale,
                       "before": before, "after": dict(current), "changes": updates})
    flush_seconds = (np.linspace(3.6, 57.0, 14) + rng.uniform(-0.45, 0.45, 14)) * scale
    flush_frames = sorted({int(round(float(t) * fps)) for t in flush_seconds})
    return initial, events, flush_frames


def controls_at(t, initial, events):
    current = dict(initial)
    for event in events:
        if t < event["seconds"]:
            break
        u = min(1.0, max(0.0, (t - event["seconds"]) / event["duration"]))
        smooth = u * u * (3.0 - 2.0 * u)
        current = {key: event["before"][key] * (1 - smooth) + event["after"][key] * smooth
                   for key in event["after"]}
    return current


def finish_contours(module, frame):
    import cv2
    import numpy as np
    frame = np.clip(module.ge.apply_brightness_contrast(frame, brightness=-3.0, contrast=42.0), 0, 255).astype(np.uint8)
    frame = np.clip(module.ge.apply_hsv_adjust(frame, hue_shift=0.0, saturation=1.95), 0, 255).astype(np.uint8)
    table = np.clip(module.ge.apply_gamma(np.arange(256, dtype=np.uint8), gamma=1.65), 0, 255).astype(np.uint8)
    frame = cv2.LUT(frame, table)
    frame = np.clip(module.ge.apply_sharpen(frame.astype(np.float32), amount=0.8, radius=1), 0, 255).astype(np.uint8)
    frame = np.clip(frame.astype(np.float32) * 2.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def encoding_options(encoder, fps):
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
                "-cq", "15", "-b:v", "42M", "-maxrate", "48M", "-bufsize", "96M",
                "-rc-lookahead", "8", "-surfaces", "12", "-multipass", "qres", "-bf", "3", "-g", str(fps * 2)]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-maxrate", "48M", "-bufsize", "96M", "-g", str(fps * 2)]


def publish_without_overwrite(partial, output):
    # Both paths share a volume. Linking creates the final name exclusively.
    # If linking is unavailable, Windows rename also refuses an existing name.
    try:
        os.link(partial, output)
    except FileExistsError:
        raise FileExistsError(f"Existing export preserved: {output}") from None
    except OSError:
        if sys.platform == "win32":
            os.rename(partial, output)
            return
        with output.open("xb") as destination, partial.open("rb") as source:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    partial.unlink()


def main():
    args = arguments()
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image was not found: {args.image}")
    if not args.silent and not args.audio.is_file():
        raise FileNotFoundError(f"Audio was not found: {args.audio}. Supply --audio or use --silent.")
    if args.output.exists():
        raise FileExistsError(f"Existing export preserved: {args.output}")
    ffmpeg = find_ffmpeg(args.ffmpeg)
    module = load_adapter(args.node_root)
    encoder_name = choose_encoder(ffmpeg, args.encoder, args.size)
    count = int(round(args.fps * args.seconds))
    assets = args.output.parent / (args.output.stem + "-assets")
    if assets.exists():
        raise FileExistsError(f"Existing render assets preserved: {assets}; choose a new --output name.")
    with Image.open(args.image) as opened:
        source = np.array(ImageOps.exif_transpose(opened).convert("RGB").resize(args.native_size, Image.Resampling.LANCZOS))
    plan = {"image": str(args.image), "output": str(args.output), "audio": None if args.silent else str(args.audio),
            "seconds": args.seconds, "fps": args.fps, "frames": count, "native_size": args.native_size,
            "export_size": args.size, "encoder": encoder_name, "node_root": str(args.node_root), "ffmpeg": str(ffmpeg)}
    print(json.dumps(plan, indent=2), flush=True)
    if args.check:
        print("Dependency and input checks passed. No output files created.", flush=True)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The streamed render stores previews, not thousands of intermediate frames.
    minimum_free = max(32_000_000, int(700_000_000 * args.seconds / 60 * args.size[0] * args.size[1] / (2160 * 3840)))
    if shutil.disk_usage(args.output.parent).free < minimum_free:
        raise OSError(f"The output drive needs at least {minimum_free / 1_000_000:.0f} MB free for this render.")
    assets.mkdir(exist_ok=False)
    cv2.setNumThreads(args.threads)
    Image.fromarray(source).save(assets / "native-source.png")
    adapter = module.NodeAdapter(source, seed=args.seed)
    if args.silent:
        envelope, audio_rate, audio = np.zeros(count, dtype=np.float32), None, None
    else:
        audio = prepare_audio(ffmpeg, args.audio, assets, args.seconds)
        envelope, audio_rate = soundtrack_envelope(audio, args.fps, count)
    initial, gestures, flushes = control_timeline(module, args.seed, args.seconds, args.fps)
    (assets / "control-timeline.json").write_text(json.dumps({"seed": args.seed, "initial": initial,
        "gestures": gestures, "buffer_flush_frames": flushes, "full_canvas": True,
        "protected_text_or_borders": False}, indent=2), encoding="utf-8")
    partial = assets / "render-in-progress.mp4"
    width, height = args.native_size
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin", "-n", "-threads", "2", "-filter_threads", "2",
        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
        "-framerate", str(args.fps), "-i", "pipe:0"]
    if audio is not None:
        command += ["-i", str(audio)]
    command += ["-map", "0:v:0"]
    if audio is not None:
        command += ["-map", "1:a:0"]
    command += ["-vf", f"scale={args.size[0]}:{args.size[1]}:flags=lanczos"]
    command += encoding_options(encoder_name, args.fps)
    command += ["-pix_fmt", "yuv420p"]
    if audio is not None:
        command += ["-c:a", "aac", "-b:a", "320k"]
    command += ["-movflags", "+faststart", "-metadata", "title=MegaGlitch Full-frame Chaotic",
        "-metadata", "comment=Original MegaGlitchNode CPU stages; full-canvas destructive effects; recorded randomized control timeline.",
        "-t", str(args.seconds), str(partial)]
    previews, diagnostics = [], []
    selected = sorted({min(count - 1, int(round(i * count / 1440))) for i in (24, 168, 360, 648, 984, 1320)})
    start = time.perf_counter()
    last_progress = start
    log_path = assets / "encode-log.txt"
    with log_path.open("w", encoding="utf-8") as log:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log,
            creationflags=0x08000000 if sys.platform == "win32" else 0)
        (assets / "active-render.json").write_text(json.dumps({"renderer_pid": os.getpid(), "encoder_pid": encoder.pid}), encoding="utf-8")
        try:
            for i in range(count):
                t = i / args.fps
                if i in flushes:
                    adapter.reset()
                raw = adapter.process(t, controls_at(t, initial, gestures), float(envelope[i]))
                frame = finish_contours(module, raw)
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise AssertionError("Native processing dimensions changed.")
                encoder.stdin.write(frame.tobytes())
                if i in selected:
                    preview = assets / f"full-frame-preview-{i:04d}.png"
                    Image.fromarray(frame).save(preview)
                    previews.append(preview)
                    diagnostics.append({"frame": i, "seconds": t, "audio_level": float(envelope[i]),
                        "average_brightness": float(frame.mean()),
                        "full_canvas_change": float(np.mean(np.abs(frame.astype(float) - source)))})
                now = time.perf_counter()
                if now - last_progress >= 5 or i + 1 == count:
                    elapsed = now - start
                    eta = elapsed * (count - i - 1) / (i + 1)
                    print(f"{i + 1}/{count} frames ({100 * (i + 1) / count:.1f}%) | {elapsed:.1f}s elapsed | about {eta:.1f}s remaining", flush=True)
                    last_progress = now
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(log_path.read_text(encoding="utf-8"))
        except BaseException as error:
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()
            if isinstance(error, BrokenPipeError):
                raise RuntimeError("FFmpeg stopped:\n" + log_path.read_text(encoding="utf-8")) from error
            raise
    sheet = Image.new("RGB", (1080, 640 * math.ceil(len(previews) / 3)), "#000000")
    for i, preview in enumerate(previews):
        with Image.open(preview) as image:
            thumb = image.resize((360, 640), Image.Resampling.LANCZOS)
            sheet.paste(thumb, ((i % 3) * 360, (i // 3) * 640))
    sheet.save(assets / "full-frame-contact-sheet.png")
    recipe = dict(plan)
    recipe.update({"preset": "Full-frame chaotic MegaGlitch", "seed": args.seed,
        "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(), "node_stages": module.STAGE_PROVENANCE,
        "audio_sample_rate": audio_rate, "gesture_count": len(gestures), "history_flush_count": len(flushes),
        "protected_regions": False, "diagnostics": diagnostics, "encoder_command": command,
        "render_seconds": time.perf_counter() - start})
    (assets / "render-recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    publish_without_overwrite(partial, args.output)
    (assets / "active-render.json").unlink()
    print(f"Finished: {args.output}\nRender details: {assets}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Render cancelled. Completed exports and existing files were preserved.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Render failed: {error}", file=sys.stderr)
        sys.exit(1)
