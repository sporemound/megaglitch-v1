from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import megaglitchfm_app_new as app_module
from megaglitch.codec_domain_post import build_codec_domain_commands
from megaglitch.offline_master import MasterCodec
from megaglitchfm_app_new import MegaGlitchFMApp


MASTER_TAGS = {
    "offline_source_fps_auto",
    "offline_source_fps",
    "offline_source_range",
    "offline_output_fps",
    "offline_fps_semantics",
    "offline_master_codec",
    "offline_master_crf",
    "offline_master_preset",
    "offline_random_seed",
    "offline_ffmpeg_executable",
    "offline_ffprobe_executable",
    "offline_analyze_source_audio",
    "offline_audio_sample_rate",
}
CODEC_TAGS = {
    "codec_post_enabled",
    "codec_post_command",
    "codec_post_work_directory",
    "codec_post_seed",
    "codec_post_long_gop",
    "codec_post_preparation_quality",
    "codec_post_final_codec",
    "codec_post_final_crf",
    "codec_post_final_preset",
}
SETTING_TAGS = {
    "master": {
        "source_fps_auto": "offline_source_fps_auto",
        "source_fps": "offline_source_fps",
        "source_range": "offline_source_range",
        "output_fps": "offline_output_fps",
        "fps_semantics": "offline_fps_semantics",
        "codec": "offline_master_codec",
        "crf": "offline_master_crf",
        "preset": "offline_master_preset",
        "random_seed": "offline_random_seed",
        "ffmpeg_executable": "offline_ffmpeg_executable",
        "ffprobe_executable": "offline_ffprobe_executable",
        "analyze_source_audio": "offline_analyze_source_audio",
        "audio_sample_rate": "offline_audio_sample_rate",
    },
    "codec_post": {
        "enabled": "codec_post_enabled",
        "command": "codec_post_command",
        "work_directory": "codec_post_work_directory",
        "seed": "codec_post_seed",
        "long_gop": "codec_post_long_gop",
        "preparation_quality": "codec_post_preparation_quality",
        "final_codec": "codec_post_final_codec",
        "final_crf": "codec_post_final_crf",
        "final_preset": "codec_post_final_preset",
    },
}


def _app() -> MegaGlitchFMApp:
    app = MegaGlitchFMApp.__new__(MegaGlitchFMApp)
    app.offline_source_fps_auto = True
    app.offline_source_fps = 30.0
    app.offline_source_range = "entire_source"
    app.offline_output_fps = 55.0
    app.offline_fps_semantics = "true_accelerated"
    app.offline_master_codec = "h264"
    app.offline_master_crf = 16
    app.offline_master_preset = "slow"
    app.offline_random_seed = 0
    app.offline_ffmpeg_executable = "ffmpeg"
    app.offline_ffprobe_executable = "ffprobe"
    app.offline_analyze_source_audio = True
    app.offline_audio_sample_rate = 48000
    app.codec_post_enabled = False
    app.codec_post_command = (
        "pframe-mosh --input {prepared} --output {corrupted} --seed {seed}"
    )
    app.codec_post_work_directory = "build/codec-domain"
    app.codec_post_seed = 0
    app.codec_post_long_gop = 120
    app.codec_post_preparation_quality = 2
    app.codec_post_final_codec = "h264"
    app.codec_post_final_crf = 16
    app.codec_post_final_preset = "slow"
    return app


def _configured_settings():
    return {
        "version": 1,
        "master": {
            "source_fps_auto": False,
            "source_fps": 29.97,
            "source_range": "from_playhead",
            "output_fps": 60.0,
            "fps_semantics": "duration_preserving",
            "codec": "h265",
            "crf": 18,
            "preset": "fast",
            "random_seed": 1234,
            "ffmpeg_executable": "custom-ffmpeg",
            "ffprobe_executable": "custom-ffprobe",
            "analyze_source_audio": False,
            "audio_sample_rate": 44100,
        },
        "codec_post": {
            "enabled": True,
            "command": (
                "packet-mosh --source {prepared} --target {corrupted} "
                "--rng {seed}"
            ),
            "work_directory": "build/custom-codec-work",
            "seed": 9876,
            "long_gop": 48,
            "preparation_quality": 7,
            "final_codec": "h265",
            "final_crf": 22,
            "final_preset": "faster",
        },
    }


class _Context:
    def __init__(self, owner, tag):
        self.owner = owner
        self.tag = tag

    def __enter__(self):
        self.owner.stack.append(self.tag)
        return self.tag

    def __exit__(self, exc_type, exc, traceback):
        self.owner.stack.pop()
        return False


class _FakeDPG:
    def __init__(self):
        self.values = {}
        self.controls = {}
        self.contexts = {}
        self.config = {}
        self.stack = []
        self._counter = 0

    def _tag(self, kind, kwargs):
        self._counter += 1
        return kwargs.get("tag") or f"{kind}-{self._counter}"

    def _context(self, kind, **kwargs):
        tag = self._tag(kind, kwargs)
        self.contexts[tag] = {
            "kind": kind,
            "kwargs": dict(kwargs),
            "parent": self.stack[-1] if self.stack else None,
        }
        return _Context(self, tag)

    def window(self, **kwargs):
        return self._context("window", **kwargs)

    def child_window(self, **kwargs):
        return self._context("child_window", **kwargs)

    def group(self, **kwargs):
        return self._context("group", **kwargs)

    def _control(self, kind, **kwargs):
        tag = self._tag(kind, kwargs)
        self.controls[tag] = {
            "kind": kind,
            "kwargs": dict(kwargs),
            "parent": self.stack[-1] if self.stack else None,
        }
        if "default_value" in kwargs:
            self.values[tag] = kwargs["default_value"]
        return tag

    def add_checkbox(self, **kwargs):
        return self._control("checkbox", **kwargs)

    def add_input_float(self, **kwargs):
        return self._control("input_float", **kwargs)

    def add_input_int(self, **kwargs):
        return self._control("input_int", **kwargs)

    def add_input_text(self, **kwargs):
        return self._control("input_text", **kwargs)

    def add_combo(self, **kwargs):
        return self._control("combo", **kwargs)

    def add_button(self, **kwargs):
        return self._control("button", **kwargs)

    def add_text(self, value="", **kwargs):
        tag = self._control("text", **kwargs)
        self.values[tag] = value
        return tag

    def add_separator(self, **kwargs):
        return self._control("separator", **kwargs)

    def does_item_exist(self, tag):
        return tag in self.controls or tag in self.contexts

    def delete_item(self, tag):
        self.controls.pop(tag, None)
        self.contexts.pop(tag, None)
        self.values.pop(tag, None)

    def get_value(self, tag):
        return self.values[tag]

    def set_value(self, tag, value):
        self.values[tag] = value

    def configure_item(self, tag, **kwargs):
        self.config.setdefault(tag, {}).update(kwargs)


def test_all_master_and_codec_settings_round_trip_and_feed_runtime_configs():
    app = _app()
    configured = _configured_settings()

    assert app._restore_offline_render_settings(configured) is True
    assert app._get_offline_render_settings() == configured

    encoder = app._build_offline_master_encoder_config(
        "deterministic.mp4", 1920, 1080, 60.0
    )
    assert encoder.output_path == "deterministic.mp4"
    assert (encoder.width, encoder.height, float(encoder.fps)) == (1920, 1080, 60.0)
    assert encoder.codec is MasterCodec.H265
    assert encoder.crf == 18
    assert encoder.preset == "fast"
    assert encoder.ffmpeg_executable == "custom-ffmpeg"
    assert encoder.ffprobe_executable == "custom-ffprobe"

    post = app._build_codec_domain_post_config(
        "deterministic.mp4", "final.mp4"
    )
    assert post.deterministic_input == "deterministic.mp4"
    assert post.final_output == "final.mp4"
    assert post.work_directory == "build/custom-codec-work"
    assert post.corruption_seed == 9876
    assert post.long_gop_size == 48
    assert post.preparation_quality == 7
    assert post.final_codec is MasterCodec.H265
    assert post.final_crf == 22
    assert post.final_preset == "faster"
    assert post.ffmpeg_executable == "custom-ffmpeg"
    assert post.corruption_command == (
        "packet-mosh",
        "--source",
        "{prepared}",
        "--target",
        "{corrupted}",
        "--rng",
        "{seed}",
    )

    commands = build_codec_domain_commands(post)
    assert commands.preparation[0] == "custom-ffmpeg"
    assert commands.preparation[commands.preparation.index("-g") + 1] == "48"
    assert commands.preparation[commands.preparation.index("-q:v") + 1] == "7"
    assert commands.corruption[commands.corruption.index("--rng") + 1] == "9876"
    assert commands.final_encode[commands.final_encode.index("-crf") + 1] == "22"
    assert commands.final_encode[commands.final_encode.index("-preset") + 1] == "faster"


def test_invalid_restore_is_atomic_instead_of_partially_mutating_state():
    app = _app()
    baseline = deepcopy(app._get_offline_render_settings())
    invalid = _configured_settings()
    invalid["codec_post"]["command"] = "packet-mosh --source {prepared}"

    assert app._restore_offline_render_settings(invalid) is False
    assert app._get_offline_render_settings() == baseline


def test_disabled_legacy_codec_pass_may_restore_with_an_empty_command():
    app = _app()
    settings = _configured_settings()
    settings["codec_post"]["enabled"] = False
    settings["codec_post"]["command"] = ""

    assert app._restore_offline_render_settings(settings) is True
    assert app.codec_post_enabled is False
    assert app.codec_post_command == ""


def test_settings_window_is_scrollable_with_sticky_actions_and_live_enable_callbacks(
    monkeypatch,
):
    app = _app()
    fake = _FakeDPG()
    monkeypatch.setattr(app_module, "dpg", fake)

    app.show_offline_master_settings()

    assert MASTER_TAGS | CODEC_TAGS <= fake.controls.keys()
    assert fake.contexts["offline_master_settings_window"]["kwargs"][
        "no_scrollbar"
    ] is True
    assert fake.contexts["offline_master_settings_scroll"]["kind"] == "child_window"
    assert fake.contexts["offline_master_settings_scroll"]["kwargs"]["height"] < 0
    assert fake.controls["offline_master_settings_status"]["parent"] == (
        "offline_master_settings_window"
    )
    assert all(
        fake.controls[tag]["parent"] == "offline_master_settings_scroll"
        or fake.controls[tag]["parent"].startswith("group-")
        for tag in MASTER_TAGS | CODEC_TAGS
    )
    visible_text = "\n".join(
        str(fake.values[tag])
        for tag, control in fake.controls.items()
        if control["kind"] == "text"
    )
    assert "DIRECT IMAGE DOMAIN" in visible_text
    assert "TEMPORAL NETWORK DOMAIN" in visible_text
    assert "CODEC DOMAIN" in visible_text

    assert fake.config["offline_source_fps"]["enabled"] is False
    assert all(fake.config[tag]["enabled"] is False for tag in CODEC_TAGS - {"codec_post_enabled"})

    fake.set_value("offline_source_fps_auto", False)
    fake.controls["offline_source_fps_auto"]["kwargs"]["callback"]()
    assert fake.config["offline_source_fps"]["enabled"] is True
    fake.set_value("codec_post_enabled", True)
    fake.controls["codec_post_enabled"]["kwargs"]["callback"]()
    assert all(fake.config[tag]["enabled"] is True for tag in CODEC_TAGS - {"codec_post_enabled"})


def test_valid_gui_values_are_applied_and_become_the_persisted_snapshot(monkeypatch):
    app = _app()
    fake = _FakeDPG()
    monkeypatch.setattr(app_module, "dpg", fake)
    app.show_offline_master_settings()
    configured = _configured_settings()
    for section, mapping in SETTING_TAGS.items():
        for field, tag in mapping.items():
            fake.set_value(tag, configured[section][field])

    assert app._apply_offline_master_settings() is True

    assert app._get_offline_render_settings() == configured
    assert fake.values["offline_master_settings_status"].startswith(
        "Validated and applied."
    )


def test_preset_restore_refreshes_every_open_settings_control(monkeypatch):
    app = _app()
    fake = _FakeDPG()
    monkeypatch.setattr(app_module, "dpg", fake)
    app.show_offline_master_settings()
    configured = _configured_settings()

    assert app._restore_offline_render_settings(configured) is True

    for section, mapping in SETTING_TAGS.items():
        for field, tag in mapping.items():
            assert fake.values[tag] == configured[section][field]
    assert all(
        fake.config[tag]["enabled"] is True
        for tag in CODEC_TAGS - {"codec_post_enabled"}
    )


def test_apply_callback_validates_transactionally_and_reports_errors(monkeypatch):
    app = _app()
    fake = _FakeDPG()
    monkeypatch.setattr(app_module, "dpg", fake)
    app.show_offline_master_settings()
    baseline = deepcopy(app._get_offline_render_settings())

    fake.set_value("offline_output_fps", 72.0)
    fake.set_value("codec_post_enabled", True)
    fake.set_value("codec_post_command", "packet-mosh --source {prepared}")

    assert app._apply_offline_master_settings() is False
    assert app._get_offline_render_settings() == baseline
    assert fake.values["offline_master_settings_status"].startswith(
        "Invalid settings:"
    )


def test_validation_rejects_each_invalid_codec_boundary_setting():
    app = _app()
    mutations = (
        ("enabled", "sometimes"),
        ("command", ""),
        ("work_directory", ""),
        ("seed", "not-an-integer"),
        ("long_gop", 1),
        ("preparation_quality", 0),
        ("final_codec", "av1"),
        ("final_crf", 52),
        ("final_preset", "not-a-preset"),
    )
    for field, value in mutations:
        settings = _configured_settings()
        settings["codec_post"][field] = value
        try:
            app._validate_offline_render_settings(settings)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid codec setting was accepted: {field}")

    invalid_executable = _configured_settings()
    invalid_executable["codec_post"]["command"] = (
        "{prepared} --output {corrupted}"
    )
    try:
        app._validate_offline_render_settings(invalid_executable)
    except ValueError:
        pass
    else:
        raise AssertionError("placeholder was accepted as the external executable")


def test_work_directory_paths_are_not_created_during_gui_validation(tmp_path):
    app = _app()
    settings = _configured_settings()
    work = tmp_path / "not-created-by-validation"
    settings["codec_post"]["work_directory"] = str(work)

    validated = app._validate_offline_render_settings(settings)

    assert Path(validated["codec_post"]["work_directory"]) == work
    assert not work.exists()


def test_real_dearpygui_context_accepts_the_scroll_and_callback_layout():
    dpg = app_module.dpg
    app = _app()
    dpg.create_context()
    try:
        app.show_offline_master_settings()
        assert dpg.does_item_exist("offline_master_settings_scroll")
        assert dpg.does_item_exist("offline_master_settings_status")
        assert all(dpg.does_item_exist(tag) for tag in MASTER_TAGS | CODEC_TAGS)
        assert dpg.get_item_configuration("codec_post_command")["enabled"] is False

        dpg.set_value("codec_post_enabled", True)
        app._update_offline_master_control_states()
        assert dpg.get_item_configuration("codec_post_command")["enabled"] is True
    finally:
        app._close_offline_master_settings()
        dpg.destroy_context()
