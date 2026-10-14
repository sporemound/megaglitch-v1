"""
Unit tests for codec-domain post-network glitch pipeline.
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

from megaglitch.codec_domain import CodecDomainConfig, CodecDomainPipeline


def test_codec_domain_strip_iframe():
    """Verify intermediate long-GOP encoding, iframe stripping, and error-tolerant decode."""
    w, h = 320, 240
    frames = []
    for i in range(12):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[i*10:(i+1)*10, i*10:(i+1)*10] = 255
        frames.append(arr)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CodecDomainConfig(
            gop_size=30,
            datamosh_mode="strip_iframe",
            corruption_intensity=0.5,
            random_seed=42,
        )
        pipeline = CodecDomainPipeline(config)
        out_frames = pipeline.process(frames, work_dir=tmpdir, fps=30)

        assert len(out_frames) == 12
        for f in out_frames:
            assert f.shape == (h, w, 3)
            assert f.dtype == np.uint8


def test_codec_domain_repeat_pframe():
    """Verify P-frame repetition mode."""
    w, h = 320, 240
    frames = []
    for i in range(10):
        arr = np.full((h, w, 3), i * 20, dtype=np.uint8)
        frames.append(arr)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CodecDomainConfig(
            gop_size=30,
            datamosh_mode="repeat_pframe",
            pframe_repeat_count=2,
            corruption_intensity=1.0,
            random_seed=42,
        )
        pipeline = CodecDomainPipeline(config)
        out_frames = pipeline.process(frames, work_dir=tmpdir, fps=30)

        assert len(out_frames) == 10
        assert out_frames[0].shape == (h, w, 3)
