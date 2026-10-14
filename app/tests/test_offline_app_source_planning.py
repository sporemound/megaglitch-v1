from __future__ import annotations

import numpy as np

from megaglitchfm_app_new import MegaGlitchFMApp


class _Reader:
    def __init__(self, count=7, fps=30.0):
        self.frames = [np.full((4, 6, 3), index, np.uint8) for index in range(count)]
        self.index = 0
        self.fps = fps

    def get_meta_data(self):
        return {"nframes": len(self.frames), "fps": self.fps}

    def get_next_index(self):
        return self.index

    def set_image_index(self, index):
        self.index = int(index)

    def get_next_data(self):
        if self.index >= len(self.frames):
            raise StopIteration
        frame = self.frames[self.index]
        self.index += 1
        return frame.copy()


def _app(reader):
    app = MegaGlitchFMApp.__new__(MegaGlitchFMApp)
    app.offline_source_range = "entire_source"
    app.offline_source_fps_auto = True
    app.offline_source_fps = 30.0
    app.export_duration = 1.0
    app.framerate = 30
    app.video_reader = reader
    app.image = None
    app.original_image_with_alpha = None
    app.gif_frames = []
    app.gif_durations = []
    app.bg_video_reader = None
    app.bg_gif_frames = []
    app.bg_media = None
    app.image_bg_preset = "Black"
    return app


def test_app_video_plan_uses_real_frame_metadata_not_export_duration():
    app = _app(_Reader(count=7, fps=24.0))
    plan, slot = app._build_offline_source_plan(
        {"primary_video_index": 4},
        [{"node_type": "source"}],
        [],
    )
    assert slot == "primary_video"
    assert plan.source_indices == tuple(range(7))
    assert plan.frame_count == 7


def test_app_from_playhead_enumerates_once_to_eof_and_exact_reads():
    reader = _Reader(count=7, fps=24.0)
    app = _app(reader)
    app.offline_source_range = "from_playhead"
    plan, _ = app._build_offline_source_plan(
        {"primary_video_index": 4},
        [{"node_type": "source"}],
        [],
    )
    assert plan.source_indices == (4, 5, 6)
    values = [int(app._read_offline_reader_frame(reader, index)[0, 0, 0]) for index in plan.source_indices]
    assert values == [4, 5, 6]
