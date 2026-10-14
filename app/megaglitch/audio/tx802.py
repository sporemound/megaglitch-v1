"""TX802/DX7-style six-operator patch data and SysEx import helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os


OPERATOR_COUNT = 6
DX7_PACKED_VOICE_SIZE = 128
DX7_UNPACKED_VOICE_SIZE = 155

TX802_ALGORITHM_TARGETS = (
    (-1, 0, 1, 2, 3, 4),
    (-1, 0, 1, 2, -1, 4),
    (-1, 0, 1, -1, 3, 4),
    (-1, 0, -1, 2, 3, 4),
    (-1, 0, 1, -1, 3, -1),
    (-1, 0, -1, 2, -1, 4),
    (-1, 0, -1, -1, 3, 4),
    (-1, 0, 1, -1, -1, 4),
    (-1, 0, -1, 2, 3, -1),
    (-1, 0, -1, 2, -1, -1),
    (-1, 0, -1, -1, 3, -1),
    (-1, 0, 1, -1, -1, -1),
    (-1, 0, -1, 0, 3, 4),
    (-1, 0, -1, 0, -1, 4),
    (-1, 0, -1, 0, -1, -1),
    (-1, 0, 1, 0, 3, -1),
    (-1, 0, 1, 0, -1, -1),
    (-1, 0, -1, 2, 2, -1),
    (-1, 0, 0, -1, 3, 3),
    (-1, 0, 0, -1, -1, 4),
    (-1, 0, 0, -1, -1, -1),
    (-1, 0, 0, 2, 2, -1),
    (-1, 0, 0, 2, -1, -1),
    (-1, 0, 0, 0, 3, -1),
    (-1, 0, 0, 0, -1, -1),
    (-1, 0, -1, 2, 2, 2),
    (-1, 0, 0, 0, 0, -1),
    (-1, 0, 0, 0, 0, 0),
    (-1, -1, 1, 1, 3, 3),
    (-1, -1, 1, -1, 3, -1),
    (-1, -1, -1, 2, -1, 4),
    (-1, -1, -1, -1, -1, -1),
)

TX802_FEEDBACK_OPERATORS = (
    5, 5, 5, 5, 2, 5, 5, 2,
    5, 3, 5, 2, 5, 5, 3, 2,
    2, 5, 5, 5, 3, 5, 3, 4,
    4, 5, 5, 5, 5, 3, 5, 0,
)


@dataclass(frozen=True)
class TX802EnvelopeSpec:
    rates: tuple[float, float, float, float] = (86.0, 62.0, 36.0, 50.0)
    levels: tuple[float, float, float, float] = (1.0, 0.65, 0.5, 0.0)


@dataclass(frozen=True)
class TX802Operator:
    ratio: float = 1.0
    fixed: bool = False
    fixed_freq: float = 440.0
    detune: float = 0.0
    output_level: float = 0.75
    pan: float = 0.0
    velocity: float = 0.0
    key_level: float = 0.0
    key_rate: float = 0.0
    enabled: bool = True
    envelope: TX802EnvelopeSpec = field(default_factory=TX802EnvelopeSpec)


@dataclass(frozen=True)
class TX802Patch:
    name: str = "INIT"
    algorithm: int = 32
    feedback: float = 0.0
    master_gain: float = 0.25
    transpose: int = 0
    lfo_rate: float = 0.0
    lfo_depth: float = 0.0
    lfo_delay: float = 0.0
    lfo_wave: str = "sine"
    pitch_envelope: TX802EnvelopeSpec = field(
        default_factory=lambda: TX802EnvelopeSpec(
            rates=(50.0, 50.0, 50.0, 50.0),
            levels=(0.5, 0.5, 0.5, 0.5),
        )
    )
    operators: tuple[TX802Operator, ...] = field(
        default_factory=lambda: tuple(TX802Operator() for _ in range(OPERATOR_COUNT))
    )


@dataclass
class TX802RenderState:
    envelope_levels: object
    envelope_stages: object
    pitch_envelope_levels: object | None = None
    pitch_envelope_stages: object | None = None


def tx802_operator_count() -> int:
    return OPERATOR_COUNT


def tx802_carrier_count(algorithm: int) -> int:
    algorithm = min(32, max(1, int(algorithm)))
    return sum(1 for target in TX802_ALGORITHM_TARGETS[algorithm - 1] if target < 0)


def dx7_checksum(data: bytes | bytearray | list[int]) -> int:
    return (-sum(int(value) & 0x7F for value in data)) & 0x7F


def _clean_sysex_bytes(raw: bytes) -> list[list[int]]:
    values = list(raw)
    messages: list[list[int]] = []
    start = None
    for index, value in enumerate(values):
        if value == 0xF0:
            start = index
        elif value == 0xF7 and start is not None:
            messages.append(values[start:index + 1])
            start = None
    if not messages and values:
        messages.append(values)
    return messages


def parse_tx802_sysex(path: str) -> list[TX802Patch]:
    """Parse DX7/TX802-compatible single voice or 32-voice bank SysEx."""
    with open(path, "rb") as handle:
        raw = handle.read()
    patches: list[TX802Patch] = []
    for message in _clean_sysex_bytes(raw):
        if len(message) < 8:
            continue
        body = message[1:-1] if message[0] == 0xF0 and message[-1] == 0xF7 else message
        if len(body) >= 6 and body[0] == 0x43 and body[2] == 0x09:
            data = body[5:-1] if len(body) >= 5 + 4096 + 1 else body[5:]
            if len(data) < 4096:
                raise ValueError(f"DX7 bank SysEx is too short in {os.path.basename(path)}.")
            _validate_optional_checksum(data[:4096], body[-1] if len(body) >= 5 + 4096 + 1 else None)
            for index in range(32):
                patches.append(_decode_packed_voice(data[index * 128:(index + 1) * 128], index))
        elif len(body) >= 6 and body[0] == 0x43 and body[2] == 0x00:
            data = body[5:-1] if len(body) >= 5 + DX7_UNPACKED_VOICE_SIZE + 1 else body[5:]
            if len(data) < DX7_UNPACKED_VOICE_SIZE:
                raise ValueError(f"DX7 single-voice SysEx is too short in {os.path.basename(path)}.")
            _validate_optional_checksum(data[:DX7_UNPACKED_VOICE_SIZE], body[-1] if len(body) >= 5 + DX7_UNPACKED_VOICE_SIZE + 1 else None)
            patches.append(_decode_unpacked_voice(data[:DX7_UNPACKED_VOICE_SIZE], 0))
    if not patches:
        raise ValueError(f"No DX7/TX802 voice data found in {os.path.basename(path)}.")
    return patches


def _validate_optional_checksum(data, checksum):
    if checksum is None:
        return
    if dx7_checksum(data) != (int(checksum) & 0x7F):
        raise ValueError("DX7/TX802 SysEx checksum mismatch.")


def _level(value: int) -> float:
    return min(1.0, max(0.0, float(value) / 99.0))


def _rate(value: int) -> float:
    return min(99.0, max(0.0, float(value)))


def _signed_detune(value: int) -> float:
    return (min(14, max(0, int(value))) - 7) * (100.0 / 7.0)


def _ratio_from_coarse_fine(coarse: int, fine: int) -> float:
    coarse = max(0, int(coarse))
    fine = min(99, max(0, int(fine)))
    base = 0.5 if coarse == 0 else float(coarse)
    return max(0.125, base * (1.0 + fine / 100.0))


def _fixed_frequency_from_coarse_fine(coarse: int, fine: int) -> float:
    coarse = min(31, max(0, int(coarse)))
    fine = min(99, max(0, int(fine)))
    return max(0.1, min(12000.0, 10.0 ** (coarse / 8.0) * (1.0 + fine / 100.0)))


def _operator_from_packed(data: list[int]) -> TX802Operator:
    rates = tuple(_rate(data[i]) for i in range(4))
    levels = tuple(_level(data[4 + i]) for i in range(4))
    rate_scale_detune = int(data[12])
    velocity_amp = int(data[13])
    osc_mode_coarse = int(data[15])
    coarse = max(0, osc_mode_coarse & 0x1F)
    fixed = bool((osc_mode_coarse >> 5) & 0x01)
    fine = int(data[16])
    ratio = _ratio_from_coarse_fine(coarse, fine)
    fixed_freq = _fixed_frequency_from_coarse_fine(coarse, fine)
    return TX802Operator(
        ratio=ratio,
        fixed=fixed,
        fixed_freq=fixed_freq,
        detune=_signed_detune(rate_scale_detune & 0x0F),
        output_level=_level(data[14]),
        velocity=(velocity_amp & 0x07) / 7.0,
        key_rate=((rate_scale_detune >> 3) & 0x07) / 7.0,
        key_level=(int(data[10]) + int(data[11])) / 198.0,
        envelope=TX802EnvelopeSpec(rates=rates, levels=levels),
    )


def _decode_packed_voice(data, index: int) -> TX802Patch:
    data = [int(value) & 0x7F for value in data]
    if len(data) != DX7_PACKED_VOICE_SIZE:
        raise ValueError("Packed DX7 voice must be 128 bytes.")
    operators = [_operator_from_packed(data[start:start + 17]) for start in range(0, 102, 17)]
    operators = tuple(reversed(operators))
    pitch = TX802EnvelopeSpec(
        rates=tuple(_rate(data[102 + i]) for i in range(4)),
        levels=tuple(_level(data[106 + i]) for i in range(4)),
    )
    lfo_pmd = data[114] / 99.0
    lfo_pitch_sensitivity = (data[116] & 0x07) / 7.0
    lfo_wave_code = (data[116] >> 3) & 0x07
    name = bytes(max(32, min(126, data[118 + i])) for i in range(10)).decode("ascii", "ignore").strip() or f"DX7 {index + 1:02d}"
    return TX802Patch(
        name=name,
        algorithm=min(32, max(1, data[110] + 1)),
        feedback=(data[111] & 0x07) / 7.0,
        transpose=int(data[117]) - 24,
        lfo_rate=data[112] / 99.0 * 20.0,
        lfo_delay=data[113] / 99.0 * 2.0,
        lfo_depth=lfo_pmd * lfo_pitch_sensitivity,
        lfo_wave=("triangle", "saw_down", "saw_up", "square", "sine", "sample_hold")[min(5, lfo_wave_code)],
        pitch_envelope=pitch,
        operators=operators,
    )


def _operator_from_unpacked(data: list[int]) -> TX802Operator:
    rates = tuple(_rate(data[i]) for i in range(4))
    levels = tuple(_level(data[4 + i]) for i in range(4))
    fixed = bool(data[17])
    coarse = max(0, int(data[18]))
    fine = int(data[19])
    return TX802Operator(
        ratio=_ratio_from_coarse_fine(coarse, fine),
        fixed=fixed,
        fixed_freq=_fixed_frequency_from_coarse_fine(coarse, fine),
        detune=_signed_detune(data[20]),
        output_level=_level(data[16]),
        velocity=int(data[15]) / 7.0,
        key_rate=int(data[13]) / 7.0,
        key_level=(int(data[10]) + int(data[11])) / 198.0,
        envelope=TX802EnvelopeSpec(rates=rates, levels=levels),
    )


def _decode_unpacked_voice(data, index: int) -> TX802Patch:
    data = [int(value) & 0x7F for value in data]
    if len(data) != DX7_UNPACKED_VOICE_SIZE:
        raise ValueError("Unpacked DX7 voice must be 155 bytes.")
    operators = [_operator_from_unpacked(data[start:start + 21]) for start in range(0, 126, 21)]
    operators = tuple(reversed(operators))
    pitch = TX802EnvelopeSpec(
        rates=tuple(_rate(data[126 + i]) for i in range(4)),
        levels=tuple(_level(data[130 + i]) for i in range(4)),
    )
    name = bytes(max(32, min(126, data[145 + i])) for i in range(10)).decode("ascii", "ignore").strip() or f"DX7 {index + 1:02d}"
    lfo_pmd = data[139] / 99.0
    lfo_pitch_sensitivity = min(7, data[143]) / 7.0
    return TX802Patch(
        name=name,
        algorithm=min(32, max(1, data[134] + 1)),
        feedback=min(7, data[135]) / 7.0,
        transpose=int(data[144]) - 24,
        lfo_rate=data[137] / 99.0 * 20.0,
        lfo_delay=data[138] / 99.0 * 2.0,
        lfo_depth=lfo_pmd * lfo_pitch_sensitivity,
        lfo_wave=("triangle", "saw_down", "saw_up", "square", "sine", "sample_hold")[min(5, data[142] & 0x07)],
        pitch_envelope=pitch,
        operators=operators,
    )


def tx802_patch_to_media_params(patch: TX802Patch) -> dict:
    base_frequency = 261.6255653005986 * (2.0 ** (patch.transpose / 12.0))
    params = {
        "tx802_patch_name": patch.name,
        "algorithm": patch.algorithm,
        "feedback": patch.feedback,
        "master_gain": patch.master_gain,
        "carrier_freq": base_frequency,
        "voice_count": 1,
        "poly_spread": 0.0,
        "chord_depth": 0.0,
        "timbre_spread": 0.0,
        "modulation_index": 1.0,
        "modulator_ratio": 1.0,
        "visual_drive": 1.0,
        "visual_influence": 0.0,
        "visual_evolution": 0.0,
        "visual_pitch_motion": 0.0,
        "feedback_color": 0.0,
        "motion_gating": 0.0,
        "motion_response": 0.0,
        "color_stereo": 0.0,
        "visual_morph_depth": 0.0,
        "midi_transpose": float(patch.transpose),
        "lfo_rate": patch.lfo_rate,
        "lfo_depth": patch.lfo_depth,
        "lfo_delay": patch.lfo_delay,
        "lfo_wave": patch.lfo_wave,
    }
    for stage, value in enumerate(patch.pitch_envelope.rates, 1):
        params[f"pitch_eg_rate_{stage}"] = value
    for stage, value in enumerate(patch.pitch_envelope.levels, 1):
        params[f"pitch_eg_level_{stage}"] = value
    for index, operator in enumerate(patch.operators, 1):
        params.update({
            f"operator_ratio_{index}": operator.ratio,
            f"operator_fixed_{index}": operator.fixed,
            f"operator_fixed_freq_{index}": operator.fixed_freq,
            f"operator_detune_{index}": operator.detune,
            f"operator_output_level_{index}": operator.output_level,
            f"operator_level_{index}": operator.output_level,
            f"operator_pan_{index}": operator.pan,
            f"operator_velocity_{index}": operator.velocity,
            f"operator_key_level_{index}": operator.key_level,
            f"operator_key_rate_{index}": operator.key_rate,
            f"operator_enabled_{index}": operator.enabled,
        })
        for stage, value in enumerate(operator.envelope.rates, 1):
            params[f"operator_eg_rate_{index}_{stage}"] = value
        for stage, value in enumerate(operator.envelope.levels, 1):
            params[f"operator_eg_level_{index}_{stage}"] = value
    return params
