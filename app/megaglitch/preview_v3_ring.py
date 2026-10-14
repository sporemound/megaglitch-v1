"""Protocol V3 CPU Preview Ring transport using Windows named shared memory."""

from __future__ import annotations

import mmap
import struct
import time
from typing import Any
import numpy as np

# Format Enums
FMT_UNKNOWN = 0
FMT_RGB8_PACKED = 1
FMT_RGBA8_UNORM = 2
FMT_BGRA8_UNORM = 3
FMT_NV12 = 4
FMT_P010 = 5

MAGIC = b"MGNPVW03"
PROTOCOL_VERSION_V3 = 3
GLOBAL_HEADER_SIZE = 128
SLOT_HEADER_SIZE = 128
DEFAULT_SLOTS = 3
DEFAULT_MAX_WIDTH = 1280
DEFAULT_MAX_HEIGHT = 720
DEFAULT_CHANNELS = 3
DEFAULT_PAYLOAD_BYTES = DEFAULT_MAX_WIDTH * DEFAULT_MAX_HEIGHT * DEFAULT_CHANNELS
DEFAULT_SLOT_BYTES = SLOT_HEADER_SIZE + DEFAULT_PAYLOAD_BYTES

# Struct formats (Little-endian, aligned to 128 bytes)
GLOBAL_HEADER_FORMAT = "<8sIIQIIIIIIQQQQ48s"
SLOT_HEADER_FORMAT = "<QQIIIIIIQQ72s"


class PreviewV3Producer:
    """Zero-allocation producer for the Protocol V3 Windows named shared memory ring."""

    def __init__(
        self,
        generation: int | None = None,
        max_width: int = DEFAULT_MAX_WIDTH,
        max_height: int = DEFAULT_MAX_HEIGHT,
        slot_count: int = DEFAULT_SLOTS,
        tag_name: str | None = None,
    ) -> None:
        self.generation = int(generation if generation is not None else time.time_ns() & 0x7FFF_FFFF_FFFF_FFFF)
        self.max_width = max_width
        self.max_height = max_height
        self.slot_count = slot_count
        self.payload_bytes = max_width * max_height * DEFAULT_CHANNELS
        self.slot_bytes = SLOT_HEADER_SIZE + self.payload_bytes
        self.total_bytes = GLOBAL_HEADER_SIZE + self.slot_count * self.slot_bytes

        self.tag_name = tag_name or f"megaglitch_preview_v3_{self.generation}"
        # On Windows, mmap(-1, size, tagname) creates a pagefile-backed named shared memory section
        self._mmap = mmap.mmap(-1, self.total_bytes, tagname=self.tag_name, access=mmap.ACCESS_WRITE)
        self.sequence = 0
        self.published_frames = 0
        self.overwritten_frames = 0
        self._init_global_header()

    def _init_global_header(self) -> None:
        header = struct.pack(
            GLOBAL_HEADER_FORMAT,
            MAGIC,
            PROTOCOL_VERSION_V3,
            GLOBAL_HEADER_SIZE,
            self.generation,
            FMT_RGB8_PACKED,
            0,
            self.max_width,
            self.max_height,
            self.slot_count,
            SLOT_HEADER_SIZE,
            self.slot_bytes,
            0,  # latest_sequence
            0,  # published_frames
            0,  # overwritten_frames
            b"\x00" * 48,
        )
        self._mmap[0:GLOBAL_HEADER_SIZE] = header

    def descriptor(self) -> dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION_V3,
            "type": "preview_v3_ring",
            "tag_name": self.tag_name,
            "generation": self.generation,
            "slots": self.slot_count,
            "slot_bytes": self.slot_bytes,
            "pixel_format": "RGB8_PACKED",
            "max_width": self.max_width,
            "max_height": self.max_height,
            "total_bytes": self.total_bytes,
        }

    def write_frame(
        self,
        rgb_frame: np.ndarray,
        sequence: int,
        timestamp_ns: int | None = None,
    ) -> int:
        """Write frame directly into ring slot without intermediate copies or ACK requirement."""
        ts_ns = timestamp_ns if timestamp_ns is not None else time.monotonic_ns()
        height, width = rgb_frame.shape[:2]
        stride = width * DEFAULT_CHANNELS
        nbytes = height * stride

        slot_idx = sequence % self.slot_count
        slot_offset = GLOBAL_HEADER_SIZE + slot_idx * self.slot_bytes
        payload_offset = slot_offset + SLOT_HEADER_SIZE

        # 1. Mark in-progress (commit_seq = 0)
        slot_hdr = struct.pack(
            SLOT_HEADER_FORMAT,
            sequence,
            ts_ns,
            width,
            height,
            stride,
            nbytes,
            FMT_RGB8_PACKED,
            0,
            self.generation,
            0,  # in progress
            b"\x00" * 72,
        )
        self._mmap[slot_offset : slot_offset + SLOT_HEADER_SIZE] = slot_hdr

        # 2. Write payload directly via memoryview
        view = memoryview(np.ascontiguousarray(rgb_frame)).cast("B")
        self._mmap[payload_offset : payload_offset + nbytes] = view[:nbytes]

        # 3. Commit slot (commit_seq = sequence at byte offset 48)
        struct.pack_into("<Q", self._mmap, slot_offset + 48, sequence)

        # 4. Update global header latest_sequence & published_frames (offset 56)
        self.published_frames += 1
        self.sequence = sequence
        struct.pack_into("<QQ", self._mmap, 56, sequence, self.published_frames)

        return slot_idx

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None


class PreviewV3Consumer:
    """Zero-ACK consumer for the Protocol V3 Windows named shared memory ring."""

    def __init__(self, tag_name: str, total_bytes: int, generation: int) -> None:
        self.tag_name = tag_name
        self.total_bytes = total_bytes
        self.generation = generation
        self._mmap = mmap.mmap(-1, total_bytes, tagname=tag_name, access=mmap.ACCESS_READ)
        self.last_consumed_seq = 0

    def acquire_latest_frame(self) -> dict[str, Any] | None:
        """Acquire the newest complete frame without acquiring or releasing explicit slot locks."""
        # Read latest sequence from global header (byte offset 56)
        latest_seq = struct.unpack_from("<Q", self._mmap, 56)[0]
        if latest_seq == 0 or latest_seq <= self.last_consumed_seq:
            return None

        slot_count = struct.unpack_from("<I", self._mmap, 40)[0]
        slot_bytes = struct.unpack_from("<Q", self._mmap, 48)[0]
        slot_idx = latest_seq % slot_count

        slot_offset = GLOBAL_HEADER_SIZE + slot_idx * slot_bytes
        payload_offset = slot_offset + SLOT_HEADER_SIZE

        # Read slot header
        (
            seq,
            ts_ns,
            width,
            height,
            stride,
            nbytes,
            pix_fmt,
            flags,
            gen,
            commit_seq,
            _,
        ) = struct.unpack_from(SLOT_HEADER_FORMAT, self._mmap, slot_offset)

        # Verify seqlock integrity
        if gen != self.generation or seq != latest_seq or commit_seq != latest_seq:
            return None  # Incomplete write or stale generation

        # Extract contiguous bytes or view
        raw_payload = bytes(self._mmap[payload_offset : payload_offset + nbytes])

        # Re-verify commit sequence after read
        final_commit = struct.unpack_from("<Q", self._mmap, slot_offset + 48)[0]
        if final_commit != latest_seq:
            return None

        self.last_consumed_seq = latest_seq
        return {
            "sequence": latest_seq,
            "timestamp_ns": ts_ns,
            "width": width,
            "height": height,
            "stride": stride,
            "nbytes": nbytes,
            "pixel_format": pix_fmt,
            "payload": raw_payload,
        }

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None
