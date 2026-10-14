from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_TIMECODE_RE = re.compile(
    r"^\s*(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?P<ms>[,.]\d{1,3})?\s*$"
)
_KEY_VALUE_RE = re.compile(r"^\s*(?P<key>[A-Za-z_][\w .-]*)\s*[:=]\s*(?P<value>.*?)\s*$")
_LUA_SHADER_TYPES = {"Lua Shader", "Kaleido Feedback", "Abyssal Signal Feedback"}
_CONTROL_KEYS = {
    "preset",
    "shader_preset",
    "shader",
    "node",
    "node_id",
    "target",
    "deck",
    "pipeline",
}


@dataclass(frozen=True)
class SrtShaderCue:
    start: float
    end: float
    preset: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    node: str | None = None
    node_id: str | None = None
    deck: str = "both"
    text: str = ""


def parse_srt_shader_cues(path: str | Path, default_preset: str | None = None) -> list[SrtShaderCue]:
    """Parse SRT cues into shader preset/parameter commands.

    Cue text supports lines such as:
        preset: enhance
        deck: a
        node_id: LuaShaderNode:1
        uZoom=2.4
    """
    srt_path = Path(path)
    try:
        content = srt_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = srt_path.read_text(encoding="cp1252")

    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())
    cues: list[SrtShaderCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if lines[0].isdigit():
            lines = lines[1:]
        if not lines or "-->" not in lines[0]:
            continue
        start_text, end_text = [part.strip() for part in lines[0].split("-->", 1)]
        start = _parse_timecode(start_text)
        end = _parse_timecode(end_text)
        if start is None or end is None or end <= start:
            continue
        payload = lines[1:]
        cue = _parse_cue_payload(start, end, payload, default_preset=default_preset)
        if cue.preset or cue.params:
            cues.append(cue)
    return sorted(cues, key=lambda cue: (cue.start, cue.end))


def find_video_srt_sidecar(video_path: str | Path) -> Path | None:
    video = Path(video_path)
    candidates = [video.with_suffix(".srt")]
    try:
        candidates.extend(sorted(video.parent.glob(f"{video.stem}.*.srt")))
    except Exception:
        pass
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def apply_srt_shader_cues(
    pipeline: list[dict[str, Any]],
    cues: list[SrtShaderCue],
    seconds: float,
    base_dir: str | Path,
    deck: str = "a",
) -> dict[str, Any]:
    if not pipeline or not cues:
        return {}
    active = [
        cue
        for cue in cues
        if cue.start <= seconds < cue.end and _deck_matches(cue.deck, deck)
    ]
    if not active:
        return {}

    changes: dict[str, Any] = {}
    for cue in active:
        changed_indices = _apply_cue_to_pipeline(pipeline, cue, base_dir)
        for index in changed_indices:
            changes[f"{deck}:{index}"] = {
                "preset": cue.preset,
                "params": dict(cue.params),
            }
    return changes


def _parse_timecode(text: str) -> float | None:
    match = _TIMECODE_RE.match(text)
    if not match:
        return None
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    ms_text = match.group("ms") or ",0"
    millis = int(ms_text[1:].ljust(3, "0")[:3])
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


def _parse_cue_payload(
    start: float,
    end: float,
    lines: list[str],
    default_preset: str | None = None,
) -> SrtShaderCue:
    preset = str(default_preset).strip() if default_preset else None
    node = None
    node_id = None
    deck = "both"
    params: dict[str, Any] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("param "):
            line = line[6:].strip()
        match = _KEY_VALUE_RE.match(line)
        if not match:
            continue
        key = match.group("key").strip()
        key_norm = key.lower().replace(" ", "_").replace("-", "_")
        value = _parse_value(match.group("value").strip())
        if key_norm in {"preset", "shader_preset", "shader"}:
            preset = str(value).strip()
        elif key_norm in {"node", "target"}:
            node = str(value).strip()
        elif key_norm == "node_id":
            node_id = str(value).strip()
        elif key_norm in {"deck", "pipeline"}:
            deck = str(value).strip().lower()
        elif key_norm not in _CONTROL_KEYS:
            params[key] = value

    return SrtShaderCue(
        start=start,
        end=end,
        preset=preset,
        params=params,
        node=node,
        node_id=node_id,
        deck=deck,
        text="\n".join(lines),
    )


def _parse_value(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        if re.match(r"^[+-]?\d+$", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _deck_matches(cue_deck: str, active_deck: str) -> bool:
    normalized = str(cue_deck or "both").strip().lower()
    if normalized in {"both", "all", "*"}:
        return True
    if active_deck == "a":
        return normalized in {"a", "deck_a", "primary", "main"}
    if active_deck == "b":
        return normalized in {"b", "deck_b", "secondary"}
    return False


def _apply_cue_to_pipeline(
    pipeline: list[dict[str, Any]], cue: SrtShaderCue, base_dir: str | Path
) -> list[int]:
    paths = _shader_preset_paths(cue.preset, base_dir) if cue.preset else {}
    changed: list[int] = []
    for index, effect_config in _iter_effect_configs(pipeline):
        if not _cue_targets_effect(effect_config, cue):
            continue
        params = effect_config.setdefault("params", {})
        if not isinstance(params, dict):
            params = {}
            effect_config["params"] = params
        if cue.preset:
            params["shader_preset"] = cue.preset
            params.update(paths)
        params.update(copy.deepcopy(cue.params))
        changed.append(index)
    if cue.preset and not changed:
        logging.debug("SRT shader cue target not found for preset %s", cue.preset)
    return changed


def _iter_effect_configs(pipeline: list[dict[str, Any]]):
    stack: list[tuple[int, dict[str, Any]]] = [
        (index, effect_config)
        for index, effect_config in enumerate(pipeline or [])
        if isinstance(effect_config, dict)
    ]
    seen: set[int] = set()
    while stack:
        index, effect_config = stack.pop()
        object_id = id(effect_config)
        if object_id in seen:
            continue
        seen.add(object_id)
        yield index, effect_config
        for key in ("input", "input_a", "input_b"):
            child = effect_config.get(key)
            if isinstance(child, dict):
                stack.append((index, child))
        inputs = effect_config.get("inputs")
        if isinstance(inputs, dict):
            for child in inputs.values():
                if isinstance(child, dict):
                    stack.append((index, child))
        elif isinstance(inputs, list):
            for child in inputs:
                if isinstance(child, dict):
                    stack.append((index, child))


def _cue_targets_effect(effect_config: dict[str, Any], cue: SrtShaderCue) -> bool:
    params = effect_config.get("params", {}) or {}
    effect_type = str(effect_config.get("effect_type") or "")
    node_id = str(effect_config.get("id") or effect_config.get("node_id") or params.get("_node_id") or "")
    if cue.node_id:
        return node_id == cue.node_id
    if cue.node:
        target = str(cue.node).strip().lower()
        if target and (
            target == effect_type.lower()
            or target == str(params.get("shader_preset") or "").lower()
            or target == node_id.lower()
        ):
            return True
        return False
    return effect_type in _LUA_SHADER_TYPES or "shader_preset" in params


def _shader_preset_paths(preset: str | None, base_dir: str | Path) -> dict[str, str]:
    stem = str(preset or "").strip()
    if not stem:
        return {}
    root = Path(base_dir)
    scripts_dir = root / "lua_glsl_demo" / "scripts"
    shaders_dir = root / "lua_glsl_demo" / "shaders"
    paths = {
        "shader_path": shaders_dir / f"{stem}.frag",
        "lua_path": scripts_dir / f"{stem}.lua",
        "metadata_path": scripts_dir / f"{stem}.effect",
    }
    result: dict[str, str] = {}
    for key, path in paths.items():
        if path.exists():
            try:
                result[key] = str(path.relative_to(root)).replace("\\", "/")
            except ValueError:
                result[key] = str(path)
    return result
