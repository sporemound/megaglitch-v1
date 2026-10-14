"""
Post-Network Codec-Domain Glitch Pipeline.
Stages:
MegaGlitchNode deterministic frame output -> MPEG-4 long-GOP preparation ->
Bitstream P-frame/packet/datamosh processing -> Error-tolerant decode -> Final master encode.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from megaglitch.mastering import find_ffmpeg_binary


@dataclass
class CodecDomainConfig:
    """Configuration for traditional codec-domain glitching."""
    gop_size: int = 120
    qscale: int = 2
    datamosh_mode: str = "strip_iframe"  # "strip_iframe", "repeat_pframe", "bit_corruption", "packet_loss"
    corruption_intensity: float = 0.5
    pframe_repeat_count: int = 4
    packet_loss_rate: float = 0.02
    random_seed: Optional[int] = None
    final_codec: str = "libx264"
    final_crf: int = 16


class CodecDomainPipeline:
    """
    Executes traditional codec-domain bitstream corruption as a post-network stage.
    """

    def __init__(self, config: Optional[CodecDomainConfig] = None):
        self.config = config or CodecDomainConfig()
        self.ffmpeg_bin = find_ffmpeg_binary()

    def encode_intermediate_mpeg4(
        self,
        frames: List[np.ndarray],
        output_avi: str,
        fps: Union[int, float, str, Fraction] = 30,
    ) -> str:
        """Encode raw master frames into an intermediate long-GOP MPEG-4 AVI."""
        if not frames:
            raise ValueError("No frames provided for MPEG-4 encoding")

        h, w = frames[0].shape[:2]
        fps_frac = Fraction(str(fps))
        fps_str = f"{fps_frac.numerator}/{fps_frac.denominator}"

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", fps_str,
            "-i", "pipe:0",
            "-c:v", "mpeg4",
            "-vtag", "XVID",
            "-qscale:v", str(int(self.config.qscale)),
            "-g", str(int(self.config.gop_size)),
            "-bf", "0",  # No B-frames for clean P-frame datamoshing
            str(output_avi),
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                if frame.ndim == 2:
                    frame = np.stack([frame, frame, frame], axis=-1)
                elif frame.shape[2] == 4:
                    frame = frame[..., :3]
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            stdout, stderr = proc.communicate(timeout=60.0)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg intermediate encode failed: {stderr.decode('utf-8', errors='ignore')}")
        except Exception as exc:
            proc.kill()
            raise exc

        return output_avi

    def corrupt_bitstream(
        self,
        input_avi: str,
        output_avi: str,
    ) -> str:
        """
        Apply bitstream / P-frame / chunk-level corruption to the AVI file.
        """
        with open(input_avi, "rb") as f:
            data = bytearray(f.read())

        rng = random.Random(self.config.random_seed)
        mode = self.config.datamosh_mode

        # AVI chunk parsing: Find video chunks ('00dc' or '00db')
        chunk_indices = []
        pos = 0
        while pos < len(data) - 8:
            fourcc = data[pos:pos+4]
            if fourcc in (b"00dc", b"00db"):
                chunk_len = struct.unpack("<I", data[pos+4:pos+8])[0]
                chunk_indices.append((pos, chunk_len))
                pos += 8 + chunk_len + (chunk_len % 2)
            else:
                pos += 1

        if not chunk_indices:
            # Fallback byte corruption if not in chunk structure
            with open(output_avi, "wb") as f:
                f.write(data)
            return output_avi

        if mode == "strip_iframe":
            # Retain the 1st keyframe chunk, then strip subsequent I-frames by replacing with previous chunk
            last_valid_pframe = None
            for idx, (c_pos, c_len) in enumerate(chunk_indices):
                if idx == 0:
                    continue  # Keep header/first frame
                chunk_data = bytes(data[c_pos+8:c_pos+8+c_len])
                # Check for MPEG-4 VOP header (0x00 0x00 0x01 0xB6)
                vop_pos = chunk_data.find(b"\x00\x00\x01\xb6")
                if vop_pos != -1 and vop_pos + 4 < len(chunk_data):
                    vop_type = (chunk_data[vop_pos + 4] & 0xC0) >> 6
                    if vop_type == 0:  # 00 = I-frame
                        if last_valid_pframe is not None and rng.random() < self.config.corruption_intensity:
                            # Replace I-frame with previous P-frame payload
                            replace_len = min(c_len, len(last_valid_pframe))
                            data[c_pos+8:c_pos+8+replace_len] = last_valid_pframe[:replace_len]
                    else:
                        last_valid_pframe = chunk_data

        elif mode == "repeat_pframe":
            # Duplicates P-frame vectors across multiple frame intervals
            repeat_k = max(1, int(self.config.pframe_repeat_count))
            last_pframe = None
            repeats_left = 0
            for idx, (c_pos, c_len) in enumerate(chunk_indices):
                if idx == 0:
                    continue
                chunk_data = bytes(data[c_pos+8:c_pos+8+c_len])
                if rng.random() < self.config.corruption_intensity and repeats_left == 0:
                    last_pframe = chunk_data
                    repeats_left = repeat_k
                elif repeats_left > 0 and last_pframe is not None:
                    replace_len = min(c_len, len(last_pframe))
                    data[c_pos+8:c_pos+8+replace_len] = last_pframe[:replace_len]
                    repeats_left -= 1

        elif mode == "bit_corruption":
            # Flip random bits in non-keyframe chunk payloads
            for idx, (c_pos, c_len) in enumerate(chunk_indices):
                if idx < 2:
                    continue
                payload_start = c_pos + 8
                payload_end = payload_start + c_len
                num_corruptions = int(c_len * self.config.corruption_intensity * 0.005)
                for _ in range(num_corruptions):
                    offset = rng.randint(payload_start, max(payload_start, payload_end - 1))
                    data[offset] ^= (1 << rng.randint(0, 7))

        with open(output_avi, "wb") as f:
            f.write(data)

        return output_avi

    def decode_error_tolerant(
        self,
        corrupted_avi: str,
        width: int,
        height: int,
        expected_frames: int,
    ) -> List[np.ndarray]:
        """Decode corrupted AVI using error-tolerant FFmpeg parameters."""
        cmd = [
            self.ffmpeg_bin,
            "-err_detect", "ignore_err",
            "-flags2", "+showall",
            "-i", str(corrupted_avi),
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = width * height * 3
        decoded_frames: List[np.ndarray] = []

        last_valid_frame = np.zeros((height, width, 3), dtype=np.uint8)

        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            decoded_frames.append(arr.copy())
            last_valid_frame = arr

        try:
            proc.communicate(timeout=10.0)
        except Exception:
            proc.kill()

        # Pad with last valid frame if decoder dropped any corrupted frames
        while len(decoded_frames) < expected_frames:
            decoded_frames.append(last_valid_frame.copy())

        return decoded_frames[:expected_frames]

    def process(
        self,
        frames: List[np.ndarray],
        work_dir: str,
        fps: Union[int, float, str, Fraction] = 30,
    ) -> List[np.ndarray]:
        """Execute the full codec-domain glitch pipeline."""
        if not frames:
            return []

        h, w = frames[0].shape[:2]
        os.makedirs(work_dir, exist_ok=True)
        intermediate_avi = os.path.join(work_dir, "intermediate_long_gop.avi")
        corrupted_avi = os.path.join(work_dir, "corrupted_stream.avi")

        self.encode_intermediate_mpeg4(frames, intermediate_avi, fps=fps)
        self.corrupt_bitstream(intermediate_avi, corrupted_avi)
        decoded = self.decode_error_tolerant(corrupted_avi, width=w, height=h, expected_frames=len(frames))

        return decoded
