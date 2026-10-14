import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
from zlib import crc32

import cv2
import numpy as np


BAYER4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float32,
) / 16.0

CHECKER4 = np.array(
    [
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

DIAGONAL4 = np.array(
    [
        [0.0, 0.2, 0.5, 0.8],
        [0.2, 0.5, 0.8, 1.0],
        [0.5, 0.8, 1.0, 0.2],
        [0.8, 1.0, 0.2, 0.5],
    ],
    dtype=np.float32,
)


GENESIS_VDP_PRESETS: Dict[str, Dict[str, object]] = {
    "toxic biotech caverns": {
        "palette_bias": 2,
        "motifs": ["cellular", "biomech", "industrial", "crystalline"],
        "sprite_mode": "organism",
        "scroll_bias": 0.7,
    },
    "chrome industrial skyline": {
        "palette_bias": 5,
        "motifs": ["industrial", "storm", "bitmask", "xor"],
        "sprite_mode": "fighter",
        "scroll_bias": 0.45,
    },
    "alien ocean sunset": {
        "palette_bias": 1,
        "motifs": ["ocean", "sine", "crystalline", "cellular"],
        "sprite_mode": "orb",
        "scroll_bias": 0.55,
    },
    "lava temple": {
        "palette_bias": 3,
        "motifs": ["lava", "industrial", "bitmask", "sine"],
        "sprite_mode": "shard",
        "scroll_bias": 0.85,
    },
    "crystalline data ruins": {
        "palette_bias": 4,
        "motifs": ["crystalline", "xor", "bayer", "storm"],
        "sprite_mode": "glyph",
        "scroll_bias": 0.5,
    },
    "storm grid horizon": {
        "palette_bias": 6,
        "motifs": ["storm", "industrial", "xor", "sine"],
        "sprite_mode": "serpent",
        "scroll_bias": 0.95,
    },
    "FM dream city": {
        "palette_bias": 0,
        "motifs": ["sine", "xor", "crystalline", "ocean"],
        "sprite_mode": "fighter",
        "scroll_bias": 0.6,
    },
    "broken title-screen abstraction": {
        "palette_bias": 7,
        "motifs": ["xor", "bitmask", "bayer", "industrial"],
        "sprite_mode": "glyph",
        "scroll_bias": 0.35,
    },
}


PALETTE_FAMILIES: List[Dict[str, object]] = [
    {
        "name": "fm_dream",
        "banks": [
            [(12, 20, 44), (28, 76, 172), (64, 210, 255), (220, 248, 255)],
            [(18, 12, 36), (122, 40, 158), (255, 120, 214), (255, 224, 250)],
            [(10, 20, 18), (22, 110, 74), (98, 255, 150), (222, 255, 236)],
            [(22, 22, 24), (88, 88, 106), (180, 190, 220), (255, 208, 108)],
        ],
    },
    {
        "name": "ocean_sunset",
        "banks": [
            [(10, 12, 42), (28, 72, 148), (72, 182, 255), (224, 246, 255)],
            [(20, 12, 30), (112, 36, 86), (255, 122, 84), (255, 216, 164)],
            [(10, 26, 26), (8, 112, 128), (36, 232, 202), (212, 255, 236)],
            [(20, 22, 28), (84, 94, 116), (156, 176, 208), (246, 198, 126)],
        ],
    },
    {
        "name": "toxic_bio",
        "banks": [
            [(8, 14, 22), (18, 84, 64), (86, 236, 124), (224, 255, 218)],
            [(18, 12, 30), (78, 28, 126), (186, 96, 252), (255, 212, 255)],
            [(12, 20, 18), (48, 92, 22), (200, 255, 72), (244, 255, 208)],
            [(20, 22, 24), (92, 94, 88), (172, 186, 176), (196, 255, 240)],
        ],
    },
    {
        "name": "lava_temple",
        "banks": [
            [(24, 8, 8), (124, 24, 18), (255, 92, 28), (255, 228, 168)],
            [(20, 10, 24), (126, 34, 96), (214, 88, 212), (255, 220, 248)],
            [(18, 16, 10), (132, 72, 18), (255, 176, 28), (255, 244, 196)],
            [(20, 18, 18), (92, 88, 76), (176, 164, 144), (220, 210, 188)],
        ],
    },
    {
        "name": "data_ruins",
        "banks": [
            [(10, 14, 34), (44, 52, 138), (120, 158, 255), (226, 242, 255)],
            [(18, 12, 34), (98, 44, 150), (192, 118, 255), (250, 228, 255)],
            [(8, 18, 26), (16, 112, 102), (120, 250, 230), (224, 255, 248)],
            [(20, 22, 28), (76, 88, 112), (182, 194, 224), (255, 255, 255)],
        ],
    },
    {
        "name": "chrome_industry",
        "banks": [
            [(12, 16, 26), (56, 76, 126), (142, 170, 224), (240, 248, 255)],
            [(22, 18, 20), (118, 62, 34), (214, 126, 78), (255, 224, 180)],
            [(14, 18, 14), (86, 102, 86), (164, 186, 170), (240, 255, 246)],
            [(20, 20, 24), (92, 96, 104), (182, 188, 200), (255, 255, 255)],
        ],
    },
    {
        "name": "storm_grid",
        "banks": [
            [(12, 16, 34), (28, 62, 170), (90, 196, 255), (232, 250, 255)],
            [(16, 12, 26), (86, 34, 140), (160, 94, 255), (244, 224, 255)],
            [(8, 20, 24), (18, 118, 142), (60, 244, 255), (224, 255, 255)],
            [(22, 24, 28), (86, 94, 110), (180, 192, 210), (255, 220, 124)],
        ],
    },
    {
        "name": "broken_title",
        "banks": [
            [(10, 10, 10), (72, 72, 84), (210, 212, 224), (255, 255, 255)],
            [(18, 0, 24), (132, 18, 88), (255, 88, 164), (255, 220, 244)],
            [(0, 18, 20), (0, 106, 124), (0, 240, 216), (216, 255, 248)],
            [(24, 10, 8), (160, 40, 10), (255, 150, 42), (255, 246, 178)],
        ],
    },
]


@dataclass(frozen=True)
class GenesisVDPConfig:
    output_width: int
    output_height: int
    internal_width: int
    internal_height: int
    tile_size: int
    sprite_scale: int
    pixel_aspect_bias: float
    aesthetic_preset: str
    palette_bank: int
    palette_shift: float
    palette_cycle_speed: float
    quantize_bits: int
    dither_amount: float
    shadow_highlight_strength: float
    tile_variation: float
    tile_mutation_rate: float
    pattern_mode: str
    noise_seed: int
    edge_density: float
    symmetry: float
    scroll_x_a: float
    scroll_y_a: float
    scroll_x_b: float
    scroll_y_b: float
    parallax_depth: float
    line_scroll_depth: float
    raster_wave_freq: float
    raster_wave_amp: float
    split_line_count: int
    sprite_count: int
    sprite_speed: float
    sprite_jitter: float
    sprite_pattern_mode: str
    sprite_priority_chaos: float
    vram_scramble: float
    tile_index_corruption: float
    palette_corruption: float
    sprite_attribute_glitch: float
    nametable_tear: float
    register_instability: float
    scanline_strength: float
    composite_bleed: float
    phosphor_blur: float
    chroma_offset: float
    crt_curve: float
    feedback_gain: float
    feedback_corruption_depth: float
    feedback_displacement_depth: float
    feedback_palette_depth: float
    feedback_sprite_mask_depth: float
    feedback_scanline_depth: float
    mask_invert: bool
    mask_plane_a: float
    mask_plane_b: float
    mask_sprites: float
    mask_corruption: float
    mask_palette_corruption: float
    mask_crt_finish: float
    mask_scanline_mod: float


def get_genesis_vdp_param_defs() -> List[Dict[str, object]]:
    preset_names = list(GENESIS_VDP_PRESETS.keys())
    defs: List[Dict[str, object]] = [
        {"name": "aesthetic_preset", "type": "combo", "label": "Preset", "items": preset_names, "default": preset_names[0]},
        {"name": "internal_width", "type": "int", "label": "Internal W", "min": 128, "max": 512, "default": 320},
        {"name": "internal_height", "type": "int", "label": "Internal H", "min": 112, "max": 320, "default": 224},
        {"name": "tile_size", "type": "int", "label": "Tile Size", "min": 4, "max": 16, "default": 8},
        {"name": "sprite_scale", "type": "int", "label": "Sprite Scale", "min": 1, "max": 4, "default": 2},
        {"name": "pixel_aspect_bias", "type": "float", "label": "Pixel Aspect", "min": 0.75, "max": 1.5, "default": 1.0},
        {"name": "palette_bank", "type": "int", "label": "Palette Bank", "min": 0, "max": 7, "default": 0},
        {"name": "palette_shift", "type": "float", "label": "Palette Shift", "min": 0.0, "max": 15.0, "default": 0.0},
        {"name": "palette_cycle_speed", "type": "float", "label": "Palette Cycle", "min": 0.0, "max": 6.0, "default": 0.75},
        {"name": "quantize_bits", "type": "int", "label": "Quantize Bits", "min": 2, "max": 6, "default": 4},
        {"name": "dither_amount", "type": "float", "label": "Dither", "min": 0.0, "max": 1.0, "default": 0.65},
        {"name": "shadow_highlight_strength", "type": "float", "label": "Shadow/HL", "min": 0.0, "max": 1.0, "default": 0.45},
        {"name": "tile_variation", "type": "float", "label": "Tile Variation", "min": 0.0, "max": 1.0, "default": 0.7},
        {"name": "tile_mutation_rate", "type": "float", "label": "Tile Mutation", "min": 0.0, "max": 4.0, "default": 0.55},
        {
            "name": "pattern_mode",
            "type": "combo",
            "label": "Pattern Mode",
            "items": ["preset", "xor", "sine", "cellular", "bayer", "bitmask", "crystalline", "industrial", "biomech", "lava", "ocean", "storm"],
            "default": "preset",
        },
        {"name": "noise_seed", "type": "int", "label": "Noise Seed", "min": 0, "max": 99999, "default": 1337},
        {"name": "edge_density", "type": "float", "label": "Edge Density", "min": 0.0, "max": 1.0, "default": 0.5},
        {"name": "symmetry", "type": "float", "label": "Symmetry", "min": 0.0, "max": 1.0, "default": 0.35},
        {"name": "scroll_x_a", "type": "float", "label": "Scroll X A", "min": -6.0, "max": 6.0, "default": 0.8},
        {"name": "scroll_y_a", "type": "float", "label": "Scroll Y A", "min": -6.0, "max": 6.0, "default": 0.2},
        {"name": "scroll_x_b", "type": "float", "label": "Scroll X B", "min": -6.0, "max": 6.0, "default": 0.35},
        {"name": "scroll_y_b", "type": "float", "label": "Scroll Y B", "min": -6.0, "max": 6.0, "default": 0.05},
        {"name": "parallax_depth", "type": "float", "label": "Parallax", "min": 0.0, "max": 2.0, "default": 0.75},
        {"name": "line_scroll_depth", "type": "float", "label": "Line Scroll", "min": 0.0, "max": 48.0, "default": 8.0},
        {"name": "raster_wave_freq", "type": "float", "label": "Raster Freq", "min": 0.0, "max": 12.0, "default": 3.5},
        {"name": "raster_wave_amp", "type": "float", "label": "Raster Amp", "min": 0.0, "max": 32.0, "default": 6.0},
        {"name": "split_line_count", "type": "int", "label": "Split Lines", "min": 1, "max": 16, "default": 4},
        {"name": "sprite_count", "type": "int", "label": "Sprite Count", "min": 0, "max": 64, "default": 18},
        {"name": "sprite_speed", "type": "float", "label": "Sprite Speed", "min": 0.0, "max": 6.0, "default": 1.2},
        {"name": "sprite_jitter", "type": "float", "label": "Sprite Jitter", "min": 0.0, "max": 12.0, "default": 0.8},
        {
            "name": "sprite_pattern_mode",
            "type": "combo",
            "label": "Sprite Mode",
            "items": ["preset", "orb", "fighter", "glyph", "serpent", "shard", "organism"],
            "default": "preset",
        },
        {"name": "sprite_priority_chaos", "type": "float", "label": "Sprite Priority", "min": 0.0, "max": 1.0, "default": 0.25},
        {"name": "vram_scramble", "type": "float", "label": "VRAM Scramble", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "tile_index_corruption", "type": "float", "label": "Tile Corrupt", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "palette_corruption", "type": "float", "label": "Palette Corrupt", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "sprite_attribute_glitch", "type": "float", "label": "Sprite Glitch", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "nametable_tear", "type": "float", "label": "NameTable Tear", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "register_instability", "type": "float", "label": "Reg Instability", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "scanline_strength", "type": "float", "label": "Scanlines", "min": 0.0, "max": 1.0, "default": 0.18},
        {"name": "composite_bleed", "type": "float", "label": "Composite", "min": 0.0, "max": 1.0, "default": 0.12},
        {"name": "phosphor_blur", "type": "float", "label": "Phosphor", "min": 0.0, "max": 1.0, "default": 0.08},
        {"name": "chroma_offset", "type": "float", "label": "Chroma Offset", "min": 0.0, "max": 4.0, "default": 0.5},
        {"name": "crt_curve", "type": "float", "label": "CRT Curve", "min": 0.0, "max": 1.0, "default": 0.08},
        {"name": "feedback_gain", "type": "float", "label": "Feedback Gain", "min": 0.0, "max": 2.0, "default": 1.0},
        {"name": "feedback_corruption_depth", "type": "float", "label": "FB Corrupt", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "feedback_displacement_depth", "type": "float", "label": "FB Displace", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "feedback_palette_depth", "type": "float", "label": "FB Palette", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "feedback_sprite_mask_depth", "type": "float", "label": "FB Sprite Mask", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "feedback_scanline_depth", "type": "float", "label": "FB Scanline", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "mask_invert", "type": "checkbox", "label": "Invert Mask", "default": False},
        {"name": "mask_plane_a", "type": "float", "label": "Mask Plane A", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "mask_plane_b", "type": "float", "label": "Mask Plane B", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "mask_sprites", "type": "float", "label": "Mask Sprites", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "mask_corruption", "type": "float", "label": "Mask Corrupt", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "mask_palette_corruption", "type": "float", "label": "Mask Palette", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "mask_crt_finish", "type": "float", "label": "Mask CRT", "min": 0.0, "max": 1.0, "default": 0.0},
        {"name": "mask_scanline_mod", "type": "float", "label": "Mask Scanline", "min": 0.0, "max": 1.0, "default": 0.0},
    ]

    group_map = {
        "aesthetic_preset": "Scene",
        "internal_width": "Scene",
        "internal_height": "Scene",
        "tile_size": "Scene",
        "sprite_scale": "Scene",
        "pixel_aspect_bias": "Scene",
        "palette_bank": "Palette",
        "palette_shift": "Palette",
        "palette_cycle_speed": "Palette",
        "quantize_bits": "Palette",
        "dither_amount": "Palette",
        "shadow_highlight_strength": "Palette",
        "tile_variation": "Pattern",
        "tile_mutation_rate": "Pattern",
        "pattern_mode": "Pattern",
        "noise_seed": "Pattern",
        "edge_density": "Pattern",
        "symmetry": "Pattern",
        "scroll_x_a": "Motion",
        "scroll_y_a": "Motion",
        "scroll_x_b": "Motion",
        "scroll_y_b": "Motion",
        "parallax_depth": "Motion",
        "line_scroll_depth": "Motion",
        "raster_wave_freq": "Motion",
        "raster_wave_amp": "Motion",
        "split_line_count": "Motion",
        "sprite_count": "Sprites",
        "sprite_speed": "Sprites",
        "sprite_jitter": "Sprites",
        "sprite_pattern_mode": "Sprites",
        "sprite_priority_chaos": "Sprites",
        "vram_scramble": "Corruption",
        "tile_index_corruption": "Corruption",
        "palette_corruption": "Corruption",
        "sprite_attribute_glitch": "Corruption",
        "nametable_tear": "Corruption",
        "register_instability": "Corruption",
        "scanline_strength": "CRT",
        "composite_bleed": "CRT",
        "phosphor_blur": "CRT",
        "chroma_offset": "CRT",
        "crt_curve": "CRT",
        "feedback_gain": "Patch / Feedback",
        "feedback_corruption_depth": "Patch / Feedback",
        "feedback_displacement_depth": "Patch / Feedback",
        "feedback_palette_depth": "Patch / Feedback",
        "feedback_sprite_mask_depth": "Patch / Feedback",
        "feedback_scanline_depth": "Patch / Feedback",
        "mask_invert": "Patch / Mask",
        "mask_plane_a": "Patch / Mask",
        "mask_plane_b": "Patch / Mask",
        "mask_sprites": "Patch / Mask",
        "mask_corruption": "Patch / Mask",
        "mask_palette_corruption": "Patch / Mask",
        "mask_crt_finish": "Patch / Mask",
        "mask_scanline_mod": "Patch / Mask",
    }
    tooltips = {
        "feedback_corruption_depth": "Uses the feedback input as a spatial corruption driver without rebuilding the cached world state.",
        "feedback_displacement_depth": "Uses feedback luminance as a row and field displacement source.",
        "feedback_palette_depth": "Lets the feedback input push palette cycling and local palette drift.",
        "feedback_sprite_mask_depth": "Uses feedback luminance as a sprite visibility field.",
        "feedback_scanline_depth": "Injects feedback into scanline and raster offset motion.",
        "mask_plane_a": "Gates foreground plane visibility with the external mask input.",
        "mask_plane_b": "Gates background plane visibility with the external mask input.",
        "mask_sprites": "Gates sprite visibility with the external mask input.",
        "mask_corruption": "Concentrates corruption into bright regions of the external mask input.",
        "mask_palette_corruption": "Applies palette-region corruption inside masked areas.",
        "mask_crt_finish": "Limits CRT finishing to masked regions, useful for keyed compositing.",
        "mask_scanline_mod": "Gates scanline and raster modulation with the external mask input.",
    }
    mod_overrides = {
        "palette_shift": {"mod_mode": "offset", "mod_span": 4.0},
        "palette_cycle_speed": {"mod_mode": "offset", "mod_span": 1.5},
        "line_scroll_depth": {"mod_mode": "offset", "mod_span": 10.0},
        "raster_wave_amp": {"mod_mode": "offset", "mod_span": 8.0},
        "raster_wave_freq": {"mod_mode": "offset", "mod_span": 2.5},
        "parallax_depth": {"mod_mode": "offset", "mod_span": 0.45},
        "sprite_speed": {"mod_mode": "offset", "mod_span": 1.6},
        "sprite_jitter": {"mod_mode": "offset", "mod_span": 2.5},
        "sprite_priority_chaos": {"mod_mode": "offset", "mod_span": 0.25},
        "vram_scramble": {"mod_mode": "offset", "mod_span": 0.25},
        "tile_index_corruption": {"mod_mode": "offset", "mod_span": 0.25},
        "palette_corruption": {"mod_mode": "offset", "mod_span": 0.25},
        "sprite_attribute_glitch": {"mod_mode": "offset", "mod_span": 0.25},
        "nametable_tear": {"mod_mode": "offset", "mod_span": 0.25},
        "register_instability": {"mod_mode": "offset", "mod_span": 0.2},
        "scanline_strength": {"mod_mode": "offset", "mod_span": 0.2},
        "composite_bleed": {"mod_mode": "offset", "mod_span": 0.2},
    }
    for param in defs:
        name = str(param.get("name") or "")
        group = group_map.get(name)
        if group:
            param["group"] = group
        tooltip = tooltips.get(name)
        if tooltip:
            param["tooltip"] = tooltip
        if name in mod_overrides:
            param.update(mod_overrides[name])
    return defs


def get_genesis_vdp_io_defs() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    input_defs = [
        {"name": "feedback_in", "attr_label": "Feedback In", "text": "Feedback / Buffer"},
        {"name": "mask_in", "attr_label": "Mask In", "text": "Mask / Key"},
    ]
    output_defs = [
        {"name": "image_out", "attr_label": "Output", "text": "Final Frame", "primary": True},
        {"name": "final_frame", "attr_label": "Aux Final", "text": "Final Frame"},
        {"name": "plane_a", "attr_label": "Aux A", "text": "Plane A"},
        {"name": "plane_b", "attr_label": "Aux B", "text": "Plane B"},
        {"name": "sprite_layer", "attr_label": "Aux Sprite", "text": "Sprite Layer"},
        {"name": "priority_mask", "attr_label": "Aux Mask", "text": "Priority Mask"},
        {"name": "palette_index_buffer", "attr_label": "Aux Index", "text": "Palette Index"},
        {"name": "luma_mask", "attr_label": "Aux Luma", "text": "Luma Mask"},
        {"name": "corruption_mask", "attr_label": "Aux Corrupt", "text": "Corruption Mask"},
        {"name": "scanline_mod_buffer", "attr_label": "Aux Scan", "text": "Scanline Mod"},
        {"name": "crt_prepass", "attr_label": "Aux CRT", "text": "CRT Prepass"},
    ]
    return input_defs, output_defs


def render_genesis_vdp_buffers(
    width: int,
    height: int,
    t: float,
    params: Dict[str, object],
    worker_state=None,
    node_key=None,
    input_frames: Dict[str, np.ndarray] | None = None,
) -> Dict[str, np.ndarray]:
    cfg = _build_config(width, height, params)
    preset = GENESIS_VDP_PRESETS.get(cfg.aesthetic_preset, GENESIS_VDP_PRESETS["toxic biotech caverns"])
    state = _get_or_build_state(cfg, preset, worker_state=worker_state, node_key=node_key)
    external = _prepare_external_inputs(cfg, input_frames or {})

    palette = _build_palette(cfg, preset, t, palette_driver=external["feedback_mean"] * cfg.feedback_palette_depth * 4.0)
    palette = _apply_palette_corruption(palette, cfg, t)

    atlas = _mutate_tile_atlas(state["tile_atlas"], cfg, preset, t)
    sprite_patterns = _mutate_sprite_patterns(state["sprite_patterns"], cfg, preset, t)
    atlas = _apply_vram_scramble(atlas, cfg, t)

    plane_a_map = state["plane_a_map"].copy()
    plane_b_map = state["plane_b_map"].copy()
    plane_a_bank = state["plane_a_bank"].copy()
    plane_b_bank = state["plane_b_bank"].copy()
    tile_corruption_map = _apply_tile_index_corruption(
        plane_a_map,
        plane_b_map,
        cfg,
        t,
        state["tile_count"],
        feedback_field=external["feedback_luma"],
        mask_field=external["mask_luma"],
    )

    # The renderer stays low-res until the very end; all palette, tile, sprite,
    # and scanline decisions happen against the internal framebuffer.
    scan = _build_scanline_tables(
        cfg,
        preset,
        t,
        feedback_row=external["feedback_row"],
        mask_row=external["mask_row"],
    )
    plane_b_idx, plane_b_bank = _render_plane(
        atlas=atlas,
        plane_map=plane_b_map,
        bank_map=plane_b_bank,
        internal_width=cfg.internal_width,
        internal_height=cfg.internal_height,
        tile_size=cfg.tile_size,
        scroll_x=cfg.scroll_x_b * t * cfg.tile_size * (1.6 + preset["scroll_bias"]),
        scroll_y=cfg.scroll_y_b * t * cfg.tile_size * 1.2,
        line_scroll=scan["line_scroll_b"],
    )
    plane_a_idx, plane_a_bank = _render_plane(
        atlas=atlas,
        plane_map=plane_a_map,
        bank_map=plane_a_bank,
        internal_width=cfg.internal_width,
        internal_height=cfg.internal_height,
        tile_size=cfg.tile_size,
        scroll_x=cfg.scroll_x_a * t * cfg.tile_size * (2.2 + cfg.parallax_depth),
        scroll_y=cfg.scroll_y_a * t * cfg.tile_size * 1.6,
        line_scroll=scan["line_scroll_a"],
    )

    tear_mask_a = _apply_nametable_tear(
        plane_a_idx,
        plane_a_bank,
        cfg,
        t,
        foreground=True,
        feedback_row=external["feedback_row"],
    )
    tear_mask_b = _apply_nametable_tear(
        plane_b_idx,
        plane_b_bank,
        cfg,
        t,
        foreground=False,
        feedback_row=external["feedback_row"],
    )

    plane_a_idx, plane_a_bank = _apply_layer_visibility(plane_a_idx, plane_a_bank, external["mask_luma"], cfg.mask_plane_a)
    plane_b_idx, plane_b_bank = _apply_layer_visibility(plane_b_idx, plane_b_bank, external["mask_luma"], cfg.mask_plane_b)

    composed_idx = plane_b_idx.copy()
    composed_bank = plane_b_bank.copy()

    sprites_under_idx = np.zeros_like(composed_idx, dtype=np.uint8)
    sprites_under_bank = np.zeros_like(composed_bank, dtype=np.uint8)
    sprites_over_idx = np.zeros_like(composed_idx, dtype=np.uint8)
    sprites_over_bank = np.zeros_like(composed_bank, dtype=np.uint8)

    sprites = _build_sprite_instances(cfg, preset, state["sprite_seeds"], t)
    _render_sprites(
        sprite_patterns=sprite_patterns,
        sprites=sprites,
        out_under_idx=sprites_under_idx,
        out_under_bank=sprites_under_bank,
        out_over_idx=sprites_over_idx,
        out_over_bank=sprites_over_bank,
        cfg=cfg,
        t=t,
        visibility_field=_combine_sprite_visibility(external["mask_luma"], external["feedback_luma"], cfg),
    )

    under_mask = sprites_under_idx > 0
    composed_idx[under_mask] = sprites_under_idx[under_mask]
    composed_bank[under_mask] = sprites_under_bank[under_mask]

    fg_mask = plane_a_idx > 0
    composed_idx[fg_mask] = plane_a_idx[fg_mask]
    composed_bank[fg_mask] = plane_a_bank[fg_mask]

    over_mask = sprites_over_idx > 0
    composed_idx[over_mask] = sprites_over_idx[over_mask]
    composed_bank[over_mask] = sprites_over_bank[over_mask]

    shade_delta = scan["shade_delta"][:, None] + scan["palette_shift_delta"][:, None]
    shaded_idx = np.clip(composed_idx.astype(np.int16) + shade_delta.astype(np.int16), 0, 15).astype(np.uint8)
    rgb_low = palette[composed_bank, shaded_idx]
    rgb_low = _apply_raster_bars(rgb_low, scan["raster_bar"], cfg)
    rgb_low = _cram_palette_quantize(rgb_low, palette, cfg)
    rgb_low = _apply_region_palette_glitch(rgb_low, cfg, t, external["mask_luma"], external["feedback_luma"])
    rgb_low = _apply_pixel_aspect_bias(rgb_low, cfg.pixel_aspect_bias)

    plane_a_rgb = _finalize_low_res_rgb(
        _indices_to_rgb(plane_a_idx, plane_a_bank, palette, shade_delta, scan["raster_bar"], cfg),
        cfg,
    )
    plane_b_rgb = _finalize_low_res_rgb(
        _indices_to_rgb(plane_b_idx, plane_b_bank, palette, shade_delta, scan["raster_bar"], cfg),
        cfg,
    )
    sprite_idx = sprites_under_idx.copy()
    sprite_bank = sprites_under_bank.copy()
    sprite_idx[over_mask] = sprites_over_idx[over_mask]
    sprite_bank[over_mask] = sprites_over_bank[over_mask]
    sprite_rgb = _finalize_low_res_rgb(
        _indices_to_rgb(sprite_idx, sprite_bank, palette, shade_delta, scan["raster_bar"], cfg),
        cfg,
    )

    crt_prepass = cv2.resize(rgb_low, (cfg.output_width, cfg.output_height), interpolation=cv2.INTER_NEAREST)
    finished_full = _apply_crt_finish(crt_prepass, cfg)
    finished = _blend_masked_region(crt_prepass, finished_full, external["mask_full"], cfg.mask_crt_finish)

    priority_mask = np.zeros_like(composed_idx, dtype=np.uint8)
    priority_mask[under_mask] = 96
    priority_mask[fg_mask] = 160
    priority_mask[over_mask] = 255
    palette_index_buffer = np.clip(shaded_idx.astype(np.float32) / 15.0 * 255.0, 0, 255).astype(np.uint8)
    luma_mask = _rgb_to_gray(crt_prepass)
    corruption_mask = _build_corruption_mask(
        cfg,
        tile_corruption_map,
        tear_mask_a,
        tear_mask_b,
        external["feedback_luma"],
        external["mask_luma"],
        scan,
    )
    scanline_mod = _build_scanline_mod_buffer(cfg, scan, external["mask_row"])

    outputs = {
        "final_frame": np.clip(finished, 0, 255).astype(np.uint8),
        "plane_a": plane_a_rgb,
        "plane_b": plane_b_rgb,
        "sprite_layer": sprite_rgb,
        "priority_mask": _finalize_mask_output(priority_mask, cfg),
        "palette_index_buffer": _finalize_mask_output(palette_index_buffer, cfg),
        "luma_mask": _ensure_rgb_output(luma_mask),
        "corruption_mask": _finalize_mask_output(corruption_mask, cfg),
        "scanline_mod_buffer": _finalize_mask_output(scanline_mod, cfg),
        "crt_prepass": np.clip(crt_prepass, 0, 255).astype(np.uint8),
    }
    outputs["luma_mask"] = _ensure_rgb_output(cv2.resize(outputs["luma_mask"][..., 0], (cfg.output_width, cfg.output_height), interpolation=cv2.INTER_LINEAR))
    return outputs


def render_genesis_vdp_frame(
    width: int,
    height: int,
    t: float,
    params: Dict[str, object],
    worker_state=None,
    node_key=None,
    input_frames: Dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    outputs = render_genesis_vdp_buffers(
        width,
        height,
        t,
        params,
        worker_state=worker_state,
        node_key=node_key,
        input_frames=input_frames,
    )
    return outputs["final_frame"]


def _build_config(output_width: int, output_height: int, params: Dict[str, object]) -> GenesisVDPConfig:
    def _get_int(name: str, default: int, lo: int, hi: int) -> int:
        try:
            value = int(round(float(params.get(name, default))))
        except Exception:
            value = default
        return int(np.clip(value, lo, hi))

    def _get_float(name: str, default: float, lo: float, hi: float) -> float:
        try:
            value = float(params.get(name, default))
        except Exception:
            value = default
        return float(np.clip(value, lo, hi))

    def _get_str(name: str, default: str) -> str:
        value = params.get(name, default)
        return str(value if value not in (None, "") else default)

    def _get_bool(name: str, default: bool) -> bool:
        value = params.get(name, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    preset = _get_str("aesthetic_preset", "toxic biotech caverns")
    if preset not in GENESIS_VDP_PRESETS:
        preset = "toxic biotech caverns"

    return GenesisVDPConfig(
        output_width=max(1, int(output_width)),
        output_height=max(1, int(output_height)),
        internal_width=_get_int("internal_width", 320, 128, 512),
        internal_height=_get_int("internal_height", 224, 112, 320),
        tile_size=_get_int("tile_size", 8, 4, 16),
        sprite_scale=_get_int("sprite_scale", 2, 1, 4),
        pixel_aspect_bias=_get_float("pixel_aspect_bias", 1.0, 0.75, 1.5),
        aesthetic_preset=preset,
        palette_bank=_get_int("palette_bank", 0, 0, 7),
        palette_shift=_get_float("palette_shift", 0.0, 0.0, 15.0),
        palette_cycle_speed=_get_float("palette_cycle_speed", 0.75, 0.0, 6.0),
        quantize_bits=_get_int("quantize_bits", 4, 2, 6),
        dither_amount=_get_float("dither_amount", 0.65, 0.0, 1.0),
        shadow_highlight_strength=_get_float("shadow_highlight_strength", 0.45, 0.0, 1.0),
        tile_variation=_get_float("tile_variation", 0.7, 0.0, 1.0),
        tile_mutation_rate=_get_float("tile_mutation_rate", 0.55, 0.0, 4.0),
        pattern_mode=_get_str("pattern_mode", "preset"),
        noise_seed=_get_int("noise_seed", 1337, 0, 99999),
        edge_density=_get_float("edge_density", 0.5, 0.0, 1.0),
        symmetry=_get_float("symmetry", 0.35, 0.0, 1.0),
        scroll_x_a=_get_float("scroll_x_a", 0.8, -6.0, 6.0),
        scroll_y_a=_get_float("scroll_y_a", 0.2, -6.0, 6.0),
        scroll_x_b=_get_float("scroll_x_b", 0.35, -6.0, 6.0),
        scroll_y_b=_get_float("scroll_y_b", 0.05, -6.0, 6.0),
        parallax_depth=_get_float("parallax_depth", 0.75, 0.0, 2.0),
        line_scroll_depth=_get_float("line_scroll_depth", 8.0, 0.0, 48.0),
        raster_wave_freq=_get_float("raster_wave_freq", 3.5, 0.0, 12.0),
        raster_wave_amp=_get_float("raster_wave_amp", 6.0, 0.0, 32.0),
        split_line_count=_get_int("split_line_count", 4, 1, 16),
        sprite_count=_get_int("sprite_count", 18, 0, 64),
        sprite_speed=_get_float("sprite_speed", 1.2, 0.0, 6.0),
        sprite_jitter=_get_float("sprite_jitter", 0.8, 0.0, 12.0),
        sprite_pattern_mode=_get_str("sprite_pattern_mode", "preset"),
        sprite_priority_chaos=_get_float("sprite_priority_chaos", 0.25, 0.0, 1.0),
        vram_scramble=_get_float("vram_scramble", 0.0, 0.0, 1.0),
        tile_index_corruption=_get_float("tile_index_corruption", 0.0, 0.0, 1.0),
        palette_corruption=_get_float("palette_corruption", 0.0, 0.0, 1.0),
        sprite_attribute_glitch=_get_float("sprite_attribute_glitch", 0.0, 0.0, 1.0),
        nametable_tear=_get_float("nametable_tear", 0.0, 0.0, 1.0),
        register_instability=_get_float("register_instability", 0.0, 0.0, 1.0),
        scanline_strength=_get_float("scanline_strength", 0.18, 0.0, 1.0),
        composite_bleed=_get_float("composite_bleed", 0.12, 0.0, 1.0),
        phosphor_blur=_get_float("phosphor_blur", 0.08, 0.0, 1.0),
        chroma_offset=_get_float("chroma_offset", 0.5, 0.0, 4.0),
        crt_curve=_get_float("crt_curve", 0.08, 0.0, 1.0),
        feedback_gain=_get_float("feedback_gain", 1.0, 0.0, 2.0),
        feedback_corruption_depth=_get_float("feedback_corruption_depth", 0.0, 0.0, 1.0),
        feedback_displacement_depth=_get_float("feedback_displacement_depth", 0.0, 0.0, 1.0),
        feedback_palette_depth=_get_float("feedback_palette_depth", 0.0, 0.0, 1.0),
        feedback_sprite_mask_depth=_get_float("feedback_sprite_mask_depth", 0.0, 0.0, 1.0),
        feedback_scanline_depth=_get_float("feedback_scanline_depth", 0.0, 0.0, 1.0),
        mask_invert=_get_bool("mask_invert", False),
        mask_plane_a=_get_float("mask_plane_a", 0.0, 0.0, 1.0),
        mask_plane_b=_get_float("mask_plane_b", 0.0, 0.0, 1.0),
        mask_sprites=_get_float("mask_sprites", 0.0, 0.0, 1.0),
        mask_corruption=_get_float("mask_corruption", 0.0, 0.0, 1.0),
        mask_palette_corruption=_get_float("mask_palette_corruption", 0.0, 0.0, 1.0),
        mask_crt_finish=_get_float("mask_crt_finish", 0.0, 0.0, 1.0),
        mask_scanline_mod=_get_float("mask_scanline_mod", 0.0, 0.0, 1.0),
    )


def _ensure_rgb_input(frame) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        return None
    return arr


def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return np.clip(arr, 0, 255).astype(np.uint8)
    gray = arr.astype(np.float32)[..., 0] * 0.2126 + arr.astype(np.float32)[..., 1] * 0.7152 + arr.astype(np.float32)[..., 2] * 0.0722
    return np.clip(gray, 0, 255).astype(np.uint8)


def _ensure_rgb_output(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _prepare_external_inputs(cfg: GenesisVDPConfig, input_frames: Dict[str, np.ndarray]) -> Dict[str, np.ndarray | float]:
    feedback = _ensure_rgb_input(input_frames.get("feedback_in"))
    mask = _ensure_rgb_input(input_frames.get("mask_in"))

    if feedback is not None:
        feedback_small = cv2.resize(feedback, (cfg.internal_width, cfg.internal_height), interpolation=cv2.INTER_LINEAR)
        feedback_luma = _rgb_to_gray(feedback_small).astype(np.float32) / 255.0
    else:
        feedback_small = None
        feedback_luma = np.zeros((cfg.internal_height, cfg.internal_width), dtype=np.float32)

    if mask is not None:
        mask_small = cv2.resize(mask, (cfg.internal_width, cfg.internal_height), interpolation=cv2.INTER_LINEAR)
        mask_luma = _rgb_to_gray(mask_small).astype(np.float32) / 255.0
        if cfg.mask_invert:
            mask_luma = 1.0 - mask_luma
    else:
        mask_luma = np.ones((cfg.internal_height, cfg.internal_width), dtype=np.float32)

    mask_row = mask_luma.mean(axis=1).astype(np.float32)
    mask_full = cv2.resize(mask_luma, (cfg.output_width, cfg.output_height), interpolation=cv2.INTER_LINEAR)
    return {
        "feedback_luma": feedback_luma * cfg.feedback_gain,
        "feedback_row": feedback_luma.mean(axis=1).astype(np.float32) * cfg.feedback_gain,
        "feedback_mean": float(feedback_luma.mean()) * cfg.feedback_gain,
        "feedback_small": feedback_small,
        "mask_luma": mask_luma.astype(np.float32),
        "mask_row": mask_row,
        "mask_full": mask_full.astype(np.float32),
    }


def _indices_to_rgb(idx: np.ndarray, bank: np.ndarray, palette: np.ndarray, shade_delta: np.ndarray, raster_bar: np.ndarray, cfg: GenesisVDPConfig) -> np.ndarray:
    shaded_idx = np.clip(idx.astype(np.int16) + shade_delta.astype(np.int16), 0, 15).astype(np.uint8)
    rgb = palette[bank, shaded_idx]
    rgb = _apply_raster_bars(rgb, raster_bar, cfg)
    rgb = _cram_palette_quantize(rgb, palette, cfg)
    rgb[idx <= 0] = 0
    return rgb


def _finalize_low_res_rgb(rgb_low: np.ndarray, cfg: GenesisVDPConfig) -> np.ndarray:
    aspect_rgb = _apply_pixel_aspect_bias(rgb_low, cfg.pixel_aspect_bias)
    return cv2.resize(aspect_rgb, (cfg.output_width, cfg.output_height), interpolation=cv2.INTER_NEAREST)


def _finalize_mask_output(mask_low: np.ndarray, cfg: GenesisVDPConfig) -> np.ndarray:
    mask_rgb = _ensure_rgb_output(mask_low)
    aspect_rgb = _apply_pixel_aspect_bias(mask_rgb, cfg.pixel_aspect_bias)
    return cv2.resize(aspect_rgb, (cfg.output_width, cfg.output_height), interpolation=cv2.INTER_NEAREST)


def _apply_layer_visibility(idx: np.ndarray, bank: np.ndarray, field: np.ndarray | None, amount: float) -> Tuple[np.ndarray, np.ndarray]:
    if field is None or amount <= 0.0:
        return idx, bank
    gate = np.clip((1.0 - amount) + np.clip(field, 0.0, 1.0) * amount, 0.0, 1.0)
    out_idx = np.clip(np.round(idx.astype(np.float32) * gate), 0, 15).astype(np.uint8)
    out_bank = bank.copy()
    out_bank[out_idx == 0] = 0
    return out_idx, out_bank


def _combine_sprite_visibility(mask_field: np.ndarray, feedback_field: np.ndarray, cfg: GenesisVDPConfig) -> np.ndarray | None:
    use_mask = cfg.mask_sprites > 0.0
    use_feedback = cfg.feedback_sprite_mask_depth > 0.0
    if not use_mask and not use_feedback:
        return None
    vis = np.ones_like(mask_field, dtype=np.float32)
    if use_mask:
        vis *= np.clip((1.0 - cfg.mask_sprites) + mask_field * cfg.mask_sprites, 0.0, 1.0)
    if use_feedback:
        vis *= np.clip((1.0 - cfg.feedback_sprite_mask_depth) + feedback_field * cfg.feedback_sprite_mask_depth, 0.0, 1.0)
    return np.clip(vis, 0.0, 1.0)


def _apply_region_palette_glitch(rgb: np.ndarray, cfg: GenesisVDPConfig, t: float, mask_field: np.ndarray, feedback_field: np.ndarray) -> np.ndarray:
    region = np.zeros(mask_field.shape, dtype=np.float32)
    if cfg.mask_palette_corruption > 0.0:
        region += mask_field * cfg.mask_palette_corruption
    if cfg.feedback_palette_depth > 0.0:
        region += feedback_field * (cfg.feedback_palette_depth * 0.35)
    region = np.clip(region, 0.0, 1.0)
    if float(region.max()) <= 1e-4:
        return rgb

    yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]].astype(np.float32)
    pulse = np.sin((xx * 0.03 + yy * 0.05 + t * 2.9) * math.tau)
    out = rgb.astype(np.float32).copy()
    out[..., 0] += pulse * region * (cfg.palette_corruption * 16.0 + cfg.feedback_palette_depth * 10.0)
    out[..., 2] -= pulse * region * (cfg.palette_corruption * 12.0 + cfg.feedback_palette_depth * 8.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def _blend_masked_region(base: np.ndarray, effected: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0 or mask is None:
        return effected
    blend = np.clip(mask.astype(np.float32) * amount, 0.0, 1.0)[..., None]
    out = base.astype(np.float32) * (1.0 - blend) + effected.astype(np.float32) * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_corruption_mask(
    cfg: GenesisVDPConfig,
    tile_corruption_map: np.ndarray,
    tear_mask_a: np.ndarray,
    tear_mask_b: np.ndarray,
    feedback_field: np.ndarray,
    mask_field: np.ndarray,
    scan: Dict[str, np.ndarray],
) -> np.ndarray:
    tile_mask = cv2.resize(tile_corruption_map.astype(np.float32), (cfg.internal_width, cfg.internal_height), interpolation=cv2.INTER_NEAREST)
    scan_energy = np.abs(scan["line_scroll_a"]) + np.abs(scan["line_scroll_b"])
    scan_norm = scan_energy / max(1.0, float(scan_energy.max()))
    scan_mask = np.repeat(scan_norm[:, None], cfg.internal_width, axis=1)
    combined = tile_mask * 0.6 + tear_mask_a * 0.8 + tear_mask_b * 0.55 + scan_mask * (cfg.register_instability * 0.35)
    if cfg.feedback_corruption_depth > 0.0:
        combined += feedback_field * cfg.feedback_corruption_depth * 0.75
    if cfg.mask_corruption > 0.0:
        combined *= np.clip((1.0 - cfg.mask_corruption) + mask_field * cfg.mask_corruption, 0.0, 1.0)
    return np.clip(combined * 255.0, 0, 255).astype(np.uint8)


def _build_scanline_mod_buffer(cfg: GenesisVDPConfig, scan: Dict[str, np.ndarray], mask_row: np.ndarray) -> np.ndarray:
    scroll_energy = np.abs(scan["line_scroll_a"]) + np.abs(scan["line_scroll_b"])
    scroll_energy += np.abs(scan["shade_delta"].astype(np.float32)) * 1.8
    scroll_energy += np.abs(scan["palette_shift_delta"].astype(np.float32)) * 2.4
    if cfg.mask_scanline_mod > 0.0:
        scroll_energy *= np.clip((1.0 - cfg.mask_scanline_mod) + mask_row * cfg.mask_scanline_mod, 0.0, 1.0)
    if float(scroll_energy.max()) > 1e-5:
        scroll_energy = scroll_energy / float(scroll_energy.max())
    out = np.repeat(scroll_energy[:, None], cfg.internal_width, axis=1)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _get_or_build_state(cfg: GenesisVDPConfig, preset: Dict[str, object], worker_state=None, node_key=None) -> Dict[str, object]:
    cache = getattr(worker_state, "generator_states", None)
    if cache is None:
        cache = {}
        if worker_state is not None:
            worker_state.generator_states = cache

    state_key = f"genesis_vdp::{node_key if node_key is not None else 'default'}"
    signature = (
        cfg.internal_width,
        cfg.internal_height,
        cfg.tile_size,
        cfg.noise_seed,
        cfg.pattern_mode,
        cfg.sprite_pattern_mode,
        cfg.aesthetic_preset,
    )
    state = cache.get(state_key)
    if state is not None and state.get("signature") == signature:
        return state

    # Cache the static machine state so scrolling/corruption animate on top of a
    # stable retro scene instead of regenerating an unrelated world every frame.
    tile_count = 96
    map_w = max(64, int(math.ceil(cfg.internal_width / cfg.tile_size)) + 24)
    map_h = max(48, int(math.ceil(cfg.internal_height / cfg.tile_size)) + 20)
    tile_atlas = _generate_tile_atlas(cfg, preset, tile_count)
    sprite_patterns = _generate_sprite_patterns(cfg, preset, 28)
    plane_a_map, plane_a_bank = _generate_plane_map(cfg, preset, map_w, map_h, tile_count, foreground=True)
    plane_b_map, plane_b_bank = _generate_plane_map(cfg, preset, map_w, map_h, tile_count, foreground=False)
    sprite_seeds = _generate_sprite_seed_table(cfg, count=64)

    state = {
        "signature": signature,
        "tile_count": tile_count,
        "tile_atlas": tile_atlas,
        "sprite_patterns": sprite_patterns,
        "plane_a_map": plane_a_map,
        "plane_a_bank": plane_a_bank,
        "plane_b_map": plane_b_map,
        "plane_b_bank": plane_b_bank,
        "sprite_seeds": sprite_seeds,
    }
    cache[state_key] = state
    return state


def _generate_tile_atlas(cfg: GenesisVDPConfig, preset: Dict[str, object], tile_count: int) -> np.ndarray:
    atlas = np.zeros((tile_count, cfg.tile_size, cfg.tile_size), dtype=np.uint8)
    motif_list = list(preset["motifs"]) if cfg.pattern_mode == "preset" else [cfg.pattern_mode]
    motif_cycle = motif_list or ["xor"]
    for tile_id in range(1, tile_count):
        mode = motif_cycle[(tile_id - 1) % len(motif_cycle)]
        atlas[tile_id] = _make_tile_pattern(cfg, mode, tile_id)
    return atlas


def _generate_sprite_patterns(cfg: GenesisVDPConfig, preset: Dict[str, object], count: int) -> np.ndarray:
    sprite_mode = preset["sprite_mode"] if cfg.sprite_pattern_mode == "preset" else cfg.sprite_pattern_mode
    patterns = np.zeros((count, cfg.tile_size, cfg.tile_size), dtype=np.uint8)
    for sprite_id in range(count):
        patterns[sprite_id] = _make_sprite_pattern(cfg, sprite_mode, sprite_id)
    return patterns


def _generate_plane_map(
    cfg: GenesisVDPConfig,
    preset: Dict[str, object],
    map_w: int,
    map_h: int,
    tile_count: int,
    foreground: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    gx, gy = np.meshgrid(
        np.arange(map_w, dtype=np.float32),
        np.arange(map_h, dtype=np.float32),
        indexing="xy",
    )
    xn = gx / max(1.0, float(map_w - 1))
    yn = gy / max(1.0, float(map_h - 1))
    seedf = float((_stable_hash(f"{cfg.noise_seed}:{cfg.aesthetic_preset}:{'fg' if foreground else 'bg'}") % 9973) / 9973.0)

    if foreground:
        field = (
            np.sin((xn * 11.0 + seedf * 3.0) * math.tau)
            + np.cos((yn * 7.5 - seedf * 1.8) * math.tau)
            + np.sin(((xn + yn) * 6.0 + seedf * 5.0) * math.tau)
        )
        detail = 0.5 + 0.5 * np.sin((xn * 19.0 - yn * 13.0 + seedf * 7.0) * math.tau)
        sparse = 0.5 + 0.5 * np.cos((xn * 9.0 + yn * 15.0 + seedf * 11.0) * math.tau)
        norm = _normalize_field(field * 0.7 + detail * 0.6)
        indices = 1 + np.floor(norm * float(tile_count - 2)).astype(np.int32)
        indices[sparse < (0.34 + (1.0 - cfg.tile_variation) * 0.26)] = 0
        bank = ((np.floor(detail * 3.99).astype(np.int32) + int(preset["palette_bias"])) % 4).astype(np.uint8)
    else:
        field = (
            np.sin((xn * (3.0 + preset["scroll_bias"] * 2.0) + seedf * 2.5) * math.tau)
            + np.cos((yn * 4.2 + seedf * 0.8) * math.tau)
            + np.sin(((xn * 0.7 + yn * 1.2) * 3.5 - seedf * 6.0) * math.tau)
        )
        horizon = np.clip((yn - 0.45) * 1.8, 0.0, 1.0)
        norm = _normalize_field(field * 0.85 + horizon * 0.55)
        indices = 1 + np.floor(norm * float(tile_count - 2)).astype(np.int32)
        bank = ((np.floor((xn * 2.0 + norm * 2.0) * 1.2).astype(np.int32) + int(preset["palette_bias"])) % 4).astype(np.uint8)

    return indices.astype(np.int32), bank


def _generate_sprite_seed_table(cfg: GenesisVDPConfig, count: int) -> np.ndarray:
    rng = np.random.default_rng(cfg.noise_seed + 9049)
    return rng.random((count, 8), dtype=np.float32)


def _build_palette(cfg: GenesisVDPConfig, preset: Dict[str, object], t: float, palette_driver: float = 0.0) -> np.ndarray:
    family_idx = (cfg.palette_bank + int(preset["palette_bias"])) % len(PALETTE_FAMILIES)
    family = PALETTE_FAMILIES[family_idx]
    shift_amount = cfg.palette_shift + cfg.palette_cycle_speed * t + palette_driver
    palette = np.zeros((4, 16, 3), dtype=np.uint8)

    for bank_idx, bank_anchors in enumerate(family["banks"]):
        bank = _build_subpalette(bank_anchors, cfg.shadow_highlight_strength)
        if shift_amount != 0.0:
            roll = int(math.floor(shift_amount * (0.8 + bank_idx * 0.18))) % 15
            bank[1:] = np.roll(bank[1:], roll, axis=0)
        palette[bank_idx] = _quantize_palette_channels(bank, cfg.quantize_bits)

    return palette


def _build_subpalette(anchor_colors: List[Tuple[int, int, int]], shadow_highlight_strength: float) -> np.ndarray:
    anchors = np.asarray(anchor_colors, dtype=np.float32)
    dark = anchors[0] * (0.45 + shadow_highlight_strength * 0.18)
    mid_a = anchors[1]
    mid_b = anchors[2]
    highlight = np.clip(anchors[3] * (1.0 + shadow_highlight_strength * 0.12), 0, 255)
    out = np.zeros((16, 3), dtype=np.float32)
    out[0] = dark * 0.6
    for idx in range(1, 16):
        if idx <= 5:
            alpha = idx / 5.0
            out[idx] = dark * (1.0 - alpha) + mid_a * alpha
        elif idx <= 10:
            alpha = (idx - 5) / 5.0
            out[idx] = mid_a * (1.0 - alpha) + mid_b * alpha
        else:
            alpha = (idx - 10) / 5.0
            out[idx] = mid_b * (1.0 - alpha) + highlight * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _quantize_palette_channels(palette: np.ndarray, bits: int) -> np.ndarray:
    levels = max(2, min(64, 2 ** bits))
    scale = 255.0 / float(levels - 1)
    return np.clip(np.round(palette.astype(np.float32) / scale) * scale, 0, 255).astype(np.uint8)


def _make_tile_pattern(cfg: GenesisVDPConfig, mode: str, tile_id: int) -> np.ndarray:
    ts = cfg.tile_size
    yy, xx = np.mgrid[0:ts, 0:ts].astype(np.float32)
    xf = (xx + 0.5) / float(ts)
    yf = (yy + 0.5) / float(ts)
    cx = xf * 2.0 - 1.0
    cy = yf * 2.0 - 1.0
    phase = cfg.tile_mutation_rate * 0.11 * tile_id + cfg.tile_variation * 0.17
    seed = _stable_hash(f"{cfg.noise_seed}:{mode}:{tile_id}")
    rng = np.random.default_rng(seed)

    if mode == "xor":
        ix = xx.astype(np.int32) * (1 + (tile_id % 5))
        iy = yy.astype(np.int32) * (2 + (tile_id % 7))
        field = ((np.bitwise_xor(ix, iy + int(phase * 19.0))) & 31).astype(np.float32) / 31.0
    elif mode == "sine":
        field = 0.5 + 0.5 * np.sin((xf * (3.0 + tile_id % 4) + yf * (2.0 + tile_id % 3) + phase) * math.tau)
        field *= 0.7 + 0.3 * (0.5 + 0.5 * np.cos((yf * (5.0 + tile_id % 5) - phase) * math.tau))
    elif mode == "cellular":
        points = rng.uniform(-1.0, 1.0, size=(3, 2)).astype(np.float32)
        dist = np.min(np.sqrt((cx[..., None] - points[:, 0]) ** 2 + (cy[..., None] - points[:, 1]) ** 2), axis=2)
        field = 1.0 - np.clip(dist / 1.5, 0.0, 1.0)
    elif mode == "bayer":
        field = np.tile(BAYER4, (max(1, ts // 4 + 1), max(1, ts // 4 + 1)))[:ts, :ts]
        field = np.mod(field + (0.5 + 0.5 * np.sin((xf + yf + phase) * math.tau)) * 0.65, 1.0)
    elif mode == "bitmask":
        ix = xx.astype(np.int32) + tile_id * 3
        iy = yy.astype(np.int32) + tile_id * 5
        field = (((ix & 3) * 3 + (iy & 5) * 2 + ((ix ^ iy) & 7)) % 16).astype(np.float32) / 15.0
    elif mode == "crystalline":
        ang = np.arctan2(cy, cx)
        rad = np.sqrt(cx * cx + cy * cy)
        field = 0.5 + 0.5 * np.cos((ang * (3.0 + tile_id % 4) + rad * 6.0 + phase * math.tau) * 1.3)
        field *= 1.0 - np.clip(rad * 0.7, 0.0, 1.0)
    elif mode == "industrial":
        girders = np.maximum((np.sin((xf * (4.0 + tile_id % 4) + phase) * math.tau) > 0.55).astype(np.float32), (np.cos((yf * (3.0 + tile_id % 5) - phase) * math.tau) > 0.65).astype(np.float32))
        rivets = (0.5 + 0.5 * np.sin((xf * 12.0 + yf * 7.0 + phase) * math.tau)) * 0.35
        field = np.clip(girders * 0.85 + rivets, 0.0, 1.0)
    elif mode == "biomech":
        ang = np.arctan2(cy, cx)
        rad = np.sqrt(cx * cx + cy * cy)
        tendons = 0.5 + 0.5 * np.sin((ang * 3.0 + phase * math.tau) * 2.0)
        vertebrae = 0.5 + 0.5 * np.cos((cy * 5.0 - phase * 1.8) * math.tau)
        field = np.clip((1.0 - rad) * 0.8 + tendons * 0.35 + vertebrae * 0.25, 0.0, 1.0)
    elif mode == "lava":
        field = 0.5 + 0.5 * np.sin((xf * 3.0 + phase) * math.tau)
        field += 0.5 + 0.5 * np.cos((yf * 4.0 - phase * 1.7) * math.tau)
        field += 0.5 + 0.5 * np.sin(((xf + yf) * 2.0 + phase * 2.3) * math.tau)
        field = _normalize_field(field)
    elif mode == "ocean":
        waves = 0.5 + 0.5 * np.sin((yf * (4.0 + tile_id % 3) + phase) * math.tau)
        ripple = 0.5 + 0.5 * np.cos((xf * 6.0 - yf * 2.0 - phase) * math.tau)
        field = np.clip(waves * 0.65 + ripple * 0.45, 0.0, 1.0)
    else:
        grid = np.maximum((np.sin((xf * 6.0 + phase) * math.tau) > 0.72).astype(np.float32), (np.cos((yf * 8.0 - phase) * math.tau) > 0.76).astype(np.float32))
        lightning = 0.5 + 0.5 * np.sin(((xf - yf) * 5.5 + phase * 1.8) * math.tau)
        field = np.clip(grid * 0.7 + lightning * 0.5, 0.0, 1.0)

    if cfg.symmetry > 0.0:
        field = field * (1.0 - cfg.symmetry * 0.7) + field[:, ::-1] * (cfg.symmetry * 0.35) + field[::-1, :] * (cfg.symmetry * 0.35)

    edges = (
        np.abs(field - np.roll(field, 1, axis=0))
        + np.abs(field - np.roll(field, -1, axis=0))
        + np.abs(field - np.roll(field, 1, axis=1))
        + np.abs(field - np.roll(field, -1, axis=1))
    ) * 0.25
    field = np.clip(field + edges * cfg.edge_density * 0.9, 0.0, 1.0)

    sparse_cut = 0.08 + (1.0 - cfg.tile_variation) * 0.22 + ((tile_id % 5) * 0.02)
    field = np.clip((field - sparse_cut) / max(1e-4, 1.0 - sparse_cut), 0.0, 1.0)
    return _quantize_field_to_indices(field, levels=16, amount=cfg.dither_amount)


def _make_sprite_pattern(cfg: GenesisVDPConfig, mode: str, sprite_id: int) -> np.ndarray:
    ts = cfg.tile_size
    yy, xx = np.mgrid[0:ts, 0:ts].astype(np.float32)
    xf = (xx + 0.5) / float(ts)
    yf = (yy + 0.5) / float(ts)
    cx = xf * 2.0 - 1.0
    cy = yf * 2.0 - 1.0
    rad = np.sqrt(cx * cx + cy * cy)
    phase = sprite_id * 0.37

    if mode == "orb":
        field = 1.0 - np.clip(rad * 1.3, 0.0, 1.0)
        field += (0.5 + 0.5 * np.cos((np.arctan2(cy, cx) * 3.0 + phase) * 1.4)) * 0.25
    elif mode == "fighter":
        hull = (np.abs(cx) * 0.7 + np.abs(cy) * 1.4) < 0.85
        wing = (np.abs(cy + 0.18) < (0.18 + (0.55 - np.abs(cx)) * 0.5))
        cockpit = (rad < 0.22).astype(np.float32)
        field = hull.astype(np.float32) * 0.6 + wing.astype(np.float32) * 0.45 + cockpit * 0.7
    elif mode == "glyph":
        vertical = (np.abs(cx) < 0.16).astype(np.float32)
        horizontal = (np.abs(cy) < 0.16).astype(np.float32)
        diagonal = (np.abs(cx + cy * (1.0 + (sprite_id % 2))) < 0.18).astype(np.float32)
        field = np.clip(vertical + horizontal * 0.8 + diagonal * 0.6, 0.0, 1.0)
    elif mode == "serpent":
        ribbon = np.abs(cy - np.sin((xf * 2.0 + phase) * math.tau) * 0.25) < 0.18
        head = ((cx - 0.45) ** 2 + cy ** 2) < 0.08
        field = ribbon.astype(np.float32) * 0.65 + head.astype(np.float32) * 0.85
    elif mode == "shard":
        field = ((cx + cy * 0.6) > -0.4).astype(np.float32) * ((cx - cy * 0.9) < 0.45).astype(np.float32)
        field *= 1.0 - np.clip(np.abs(cy) * 0.8, 0.0, 1.0)
    else:
        shell = 1.0 - np.clip(rad * 1.1, 0.0, 1.0)
        veins = 0.5 + 0.5 * np.sin((cx * 2.5 - cy * 3.0 + phase) * math.tau)
        core = np.exp(-(cx * cx + cy * cy) / 0.18)
        field = np.clip(shell * 0.45 + veins * 0.35 + core * 0.45, 0.0, 1.0)

    field *= (rad < 0.95).astype(np.float32)
    indices = _quantize_field_to_indices(np.clip(field, 0.0, 1.0), levels=16, amount=cfg.dither_amount * 0.7)
    indices[indices < 4] = 0
    return indices


def _quantize_field_to_indices(field: np.ndarray, levels: int, amount: float) -> np.ndarray:
    h, w = field.shape
    dither = _make_dither_map(h, w)
    biased = np.clip(field + (dither - 0.5) * (amount * 0.22), 0.0, 1.0)
    return np.clip(np.round(biased * float(levels - 1)), 0, levels - 1).astype(np.uint8)


def _make_dither_map(height: int, width: int) -> np.ndarray:
    tile_h = max(1, int(math.ceil(height / 4.0)))
    tile_w = max(1, int(math.ceil(width / 4.0)))
    bayer = np.tile(BAYER4, (tile_h, tile_w))[:height, :width]
    checker = np.tile(CHECKER4, (tile_h, tile_w))[:height, :width]
    diagonal = np.tile(DIAGONAL4, (tile_h, tile_w))[:height, :width]
    return (bayer * 0.6) + (checker * 0.2) + (diagonal * 0.2)


def _mutate_tile_atlas(base_atlas: np.ndarray, cfg: GenesisVDPConfig, preset: Dict[str, object], t: float) -> np.ndarray:
    atlas = base_atlas.copy()
    if cfg.tile_mutation_rate <= 0.0:
        return atlas

    motif_list = list(preset["motifs"]) if cfg.pattern_mode == "preset" else [cfg.pattern_mode]
    pulse = 0.5 + 0.5 * math.sin(t * max(0.1, cfg.tile_mutation_rate) * 2.1)
    mutate_count = max(1, int(round(cfg.tile_mutation_rate * 6.0)))
    for offset in range(mutate_count):
        tile_id = 1 + ((int(t * (4.0 + cfg.tile_mutation_rate * 6.0)) + offset * 13) % max(1, atlas.shape[0] - 1))
        mode = motif_list[(tile_id + offset) % len(motif_list)]
        mutated = _make_tile_pattern(cfg, mode, tile_id + int(t * 3.0))
        blend_mask = (mutated.astype(np.float32) * pulse + atlas[tile_id].astype(np.float32) * (1.0 - pulse))
        atlas[tile_id] = np.clip(np.round(blend_mask), 0, 15).astype(np.uint8)
    return atlas


def _mutate_sprite_patterns(base_patterns: np.ndarray, cfg: GenesisVDPConfig, preset: Dict[str, object], t: float) -> np.ndarray:
    patterns = base_patterns.copy()
    if cfg.sprite_attribute_glitch <= 0.01 and cfg.tile_mutation_rate <= 0.01:
        return patterns

    sprite_mode = preset["sprite_mode"] if cfg.sprite_pattern_mode == "preset" else cfg.sprite_pattern_mode
    pulse = 0.5 + 0.5 * math.sin(t * (2.6 + cfg.sprite_attribute_glitch * 5.0))
    mutate_count = max(1, int(round((cfg.tile_mutation_rate + cfg.sprite_attribute_glitch) * 4.0)))
    for offset in range(mutate_count):
        sprite_id = (int(t * 7.0) + offset * 5) % patterns.shape[0]
        updated = _make_sprite_pattern(cfg, sprite_mode, sprite_id + int(t * 5.0))
        mix = np.clip(0.35 + pulse * 0.5, 0.0, 1.0)
        patterns[sprite_id] = np.clip(np.round(patterns[sprite_id].astype(np.float32) * (1.0 - mix) + updated.astype(np.float32) * mix), 0, 15).astype(np.uint8)
    return patterns


def _apply_vram_scramble(atlas: np.ndarray, cfg: GenesisVDPConfig, t: float) -> np.ndarray:
    if cfg.vram_scramble <= 0.0:
        return atlas

    out = atlas.copy()
    ts = cfg.tile_size
    scramble_count = max(1, int(round(cfg.vram_scramble * 8.0)))
    for idx in range(scramble_count):
        tile_id = 1 + ((int(t * 11.0) + idx * 17) % max(1, out.shape[0] - 1))
        shift_x = int(round(math.sin(t * 3.7 + idx) * cfg.vram_scramble * ts * 0.7))
        shift_y = int(round(math.cos(t * 2.9 + idx * 0.8) * cfg.vram_scramble * ts * 0.7))
        out[tile_id] = np.roll(out[tile_id], shift_x, axis=1)
        out[tile_id] = np.roll(out[tile_id], shift_y, axis=0)
        if (idx + int(t * 3.0)) % 3 == 0:
            out[tile_id] = out[tile_id].T
    return out


def _apply_tile_index_corruption(
    plane_a_map: np.ndarray,
    plane_b_map: np.ndarray,
    cfg: GenesisVDPConfig,
    t: float,
    tile_count: int,
    feedback_field: np.ndarray | None = None,
    mask_field: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.tile_index_corruption <= 0.0 and cfg.feedback_corruption_depth <= 0.0:
        return np.zeros_like(plane_a_map, dtype=np.float32)

    h, w = plane_a_map.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = t * (1.0 + cfg.tile_index_corruption * 5.0)
    mask_a = (0.5 + 0.5 * np.sin((xx * 0.31 + yy * 0.23 + phase) * math.tau)) > (0.985 - cfg.tile_index_corruption * 0.25)
    mask_b = (0.5 + 0.5 * np.cos((xx * 0.17 - yy * 0.19 + phase * 0.7) * math.tau)) > (0.99 - cfg.tile_index_corruption * 0.18)

    local_drive = np.zeros((h, w), dtype=np.float32)
    if feedback_field is not None and cfg.feedback_corruption_depth > 0.0:
        feedback_small = cv2.resize(feedback_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        local_drive += np.clip(feedback_small, 0.0, 1.0) * cfg.feedback_corruption_depth
    if mask_field is not None and cfg.mask_corruption > 0.0:
        mask_small = cv2.resize(mask_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        local_drive += np.clip(mask_small, 0.0, 1.0) * (cfg.mask_corruption * 0.65)

    if float(local_drive.max()) > 0.0:
        extra_a = local_drive > (0.78 - cfg.feedback_corruption_depth * 0.35)
        extra_b = np.roll(local_drive, 1, axis=1) > (0.82 - cfg.feedback_corruption_depth * 0.28)
        mask_a = np.logical_or(mask_a, extra_a)
        mask_b = np.logical_or(mask_b, extra_b)

    offset_a = 1 + int(round(cfg.tile_index_corruption * 11.0))
    offset_b = 1 + int(round(cfg.tile_index_corruption * 7.0))
    plane_a_map[mask_a] = (plane_a_map[mask_a] + offset_a + ((xx[mask_a].astype(np.int32) + yy[mask_a].astype(np.int32)) % 9)) % max(2, tile_count - 1)
    plane_b_map[mask_b] = (plane_b_map[mask_b] + offset_b + ((xx[mask_b].astype(np.int32) * 3 + yy[mask_b].astype(np.int32)) % 7)) % max(2, tile_count - 1)
    return np.logical_or(mask_a, mask_b).astype(np.float32)


def _build_scanline_tables(
    cfg: GenesisVDPConfig,
    preset: Dict[str, object],
    t: float,
    feedback_row: np.ndarray | None = None,
    mask_row: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    y = np.arange(cfg.internal_height, dtype=np.float32)
    yn = y / max(1.0, float(cfg.internal_height - 1))
    splits = max(1, cfg.split_line_count)
    split_size = max(1.0, float(cfg.internal_height) / float(splits))
    split_id = np.floor(y / split_size)

    wave = np.sin((yn * max(0.0, cfg.raster_wave_freq) + t * 0.22) * math.tau)
    wobble = np.sin((yn * 14.0 - t * (2.8 + cfg.register_instability * 6.0)) * math.tau)
    split_wave = np.sin((split_id * 0.37 + t * 0.55) * math.tau)
    shimmer = np.cos((yn * 23.0 + t * (3.5 + preset["scroll_bias"] * 2.5)) * math.tau)
    perspective = np.square(np.clip(yn - 0.55, 0.0, 1.0))

    line_scroll_a = (
        wave * (cfg.raster_wave_amp * 0.6)
        + wobble * (cfg.line_scroll_depth * 0.35)
        + split_wave * (cfg.line_scroll_depth * 0.25)
        + shimmer * (cfg.line_scroll_depth * 0.18)
        + perspective * (cfg.line_scroll_depth * (0.4 + cfg.parallax_depth * 0.9))
    )
    line_scroll_b = (
        wave * (cfg.raster_wave_amp * 0.32)
        + split_wave * (cfg.line_scroll_depth * 0.45)
        - perspective * (cfg.line_scroll_depth * (0.7 + preset["scroll_bias"]))
        + wobble * (cfg.line_scroll_depth * 0.08)
    )

    if cfg.register_instability > 0.0:
        jitter = np.sin((y * 0.17 + t * 9.0) * math.tau) * cfg.register_instability * 4.0
        line_scroll_a = line_scroll_a + jitter
        line_scroll_b = line_scroll_b - jitter * 0.6

    if feedback_row is not None and (cfg.feedback_displacement_depth > 0.0 or cfg.feedback_scanline_depth > 0.0):
        row_drive = (np.clip(feedback_row.astype(np.float32), 0.0, 1.0) - 0.5)
        displacement = row_drive * (cfg.feedback_displacement_depth * (cfg.tile_size * 4.0 + cfg.raster_wave_amp))
        scanline_drive = row_drive * (cfg.feedback_scanline_depth * (6.0 + cfg.line_scroll_depth * 0.55))
        line_scroll_a = line_scroll_a + displacement + scanline_drive
        line_scroll_b = line_scroll_b - displacement * 0.65 + scanline_drive * 0.35
    shade_delta = np.sin((yn * 6.0 - t * 0.7) * math.tau) * cfg.shadow_highlight_strength * 2.2
    palette_shift_delta = np.sin((yn * 10.0 + t * (0.35 + cfg.palette_cycle_speed * 0.15)) * math.tau) * min(2.0, cfg.palette_shift * 0.15 + cfg.palette_corruption)
    raster_bar = 0.75 + 0.25 * np.sin((yn * 4.0 - t * 0.55) * math.tau)
    if feedback_row is not None and cfg.feedback_palette_depth > 0.0:
        palette_shift_delta = palette_shift_delta + (np.clip(feedback_row.astype(np.float32), 0.0, 1.0) - 0.5) * (cfg.feedback_palette_depth * 3.0)
    if feedback_row is not None and cfg.feedback_scanline_depth > 0.0:
        raster_bar = np.clip(raster_bar + (feedback_row.astype(np.float32) - 0.5) * (cfg.feedback_scanline_depth * 0.22), 0.45, 1.25)
    if mask_row is not None and cfg.mask_scanline_mod > 0.0:
        gate = np.clip((1.0 - cfg.mask_scanline_mod) + mask_row.astype(np.float32) * cfg.mask_scanline_mod, 0.0, 1.0)
        line_scroll_a = line_scroll_a * gate
        line_scroll_b = line_scroll_b * gate
        shade_delta = shade_delta * gate
        palette_shift_delta = palette_shift_delta * gate
        raster_bar = 1.0 + (raster_bar - 1.0) * gate

    return {
        "line_scroll_a": line_scroll_a.astype(np.float32),
        "line_scroll_b": line_scroll_b.astype(np.float32),
        "shade_delta": np.round(shade_delta).astype(np.int16),
        "palette_shift_delta": np.round(palette_shift_delta).astype(np.int16),
        "raster_bar": raster_bar.astype(np.float32),
    }


def _render_plane(
    atlas: np.ndarray,
    plane_map: np.ndarray,
    bank_map: np.ndarray,
    internal_width: int,
    internal_height: int,
    tile_size: int,
    scroll_x: float,
    scroll_y: float,
    line_scroll: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    plane_pixels = atlas[plane_map]
    plane_indices = plane_pixels.transpose(0, 2, 1, 3).reshape(plane_map.shape[0] * tile_size, plane_map.shape[1] * tile_size)
    plane_banks = np.repeat(np.repeat(bank_map, tile_size, axis=0), tile_size, axis=1)

    tex_h, tex_w = plane_indices.shape
    y_base = (np.arange(internal_height, dtype=np.int32) + int(round(scroll_y))) % tex_h
    sampled_idx = plane_indices[y_base]
    sampled_bank = plane_banks[y_base]
    out_idx = np.zeros((internal_height, internal_width), dtype=np.uint8)
    out_bank = np.zeros((internal_height, internal_width), dtype=np.uint8)

    for row in range(internal_height):
        x_shift = int(round(scroll_x + float(line_scroll[row])))
        row_idx = np.roll(sampled_idx[row], -x_shift)[:internal_width]
        row_bank = np.roll(sampled_bank[row], -x_shift)[:internal_width]
        out_idx[row] = row_idx
        out_bank[row] = row_bank

    return out_idx, out_bank


def _apply_nametable_tear(
    idx_buf: np.ndarray,
    bank_buf: np.ndarray,
    cfg: GenesisVDPConfig,
    t: float,
    foreground: bool,
    feedback_row: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.nametable_tear <= 0.0:
        return np.zeros_like(idx_buf, dtype=np.float32)

    h, w = idx_buf.shape
    mask = np.zeros((h, w), dtype=np.float32)
    band_count = max(1, int(round(cfg.nametable_tear * 4.0)))
    for band in range(band_count):
        band_h = max(4, int(round(h * (0.04 + cfg.nametable_tear * 0.07))))
        start = (int(t * (17 if foreground else 11)) * (band + 1) * 3 + band * 29) % max(1, h - band_h)
        shift = math.sin(t * (3.7 + band * 0.6) + band) * cfg.nametable_tear * (18 if foreground else 12)
        if feedback_row is not None and cfg.feedback_corruption_depth > 0.0:
            local_feedback = float(np.clip(feedback_row[start:start + band_h].mean() if start + band_h <= feedback_row.shape[0] else feedback_row.mean(), 0.0, 1.0))
            shift += (local_feedback - 0.5) * cfg.feedback_corruption_depth * (16 if foreground else 10)
        shift = int(round(shift))
        idx_buf[start:start + band_h] = np.roll(idx_buf[start:start + band_h], shift, axis=1)
        bank_buf[start:start + band_h] = np.roll(bank_buf[start:start + band_h], shift, axis=1)
        mask[start:start + band_h] = np.maximum(mask[start:start + band_h], 0.5 + 0.5 * min(1.0, abs(shift) / max(1.0, 18.0 if foreground else 12.0)))
        if foreground and cfg.nametable_tear > 0.55 and band == band_count - 1:
            hole_mask = (np.sin((np.arange(band_h)[:, None] * 0.7 + np.arange(w)[None, :] * 0.11 + t * 8.0) * math.tau) > 0.82)
            idx_buf[start:start + band_h][hole_mask] = 0
            mask[start:start + band_h][hole_mask] = 1.0
    return mask

def _build_sprite_instances(cfg: GenesisVDPConfig, preset: Dict[str, object], sprite_seeds: np.ndarray, t: float) -> List[Dict[str, int]]:
    count = min(cfg.sprite_count, sprite_seeds.shape[0])
    sprites: List[Dict[str, int]] = []
    for idx in range(count):
        seed = sprite_seeds[idx]
        phase = t * (0.7 + seed[0] * cfg.sprite_speed * 1.8) + seed[1] * math.tau
        drift = 0.5 + 0.5 * math.sin(t * (1.7 + seed[2] * 2.6) + idx * 0.4)
        x = (0.5 + 0.42 * math.sin(phase) + 0.12 * math.sin(phase * 2.3 + seed[3] * math.tau)) * (cfg.internal_width - 1)
        y = (0.5 + 0.28 * math.cos(phase * 0.73 + seed[4] * math.tau) + 0.18 * math.sin(phase * 1.37 + seed[5] * math.tau)) * (cfg.internal_height - 1)
        x += math.sin(t * (7.0 + seed[6] * 3.0) + idx) * cfg.sprite_jitter
        y += math.cos(t * (6.0 + seed[7] * 3.0) + idx * 0.7) * cfg.sprite_jitter * 0.7

        if cfg.sprite_attribute_glitch > 0.0:
            glitch = cfg.sprite_attribute_glitch * (0.5 + 0.5 * math.sin(t * 11.0 + idx * 1.2))
            x += round(math.sin(t * 19.0 + idx) * glitch * 8.0)
            y += round(math.cos(t * 15.0 + idx * 0.8) * glitch * 6.0)

        priority = 1 if ((seed[2] + drift * cfg.sprite_priority_chaos) % 1.0) > (0.48 - cfg.sprite_priority_chaos * 0.22) else 0
        sprites.append(
            {
                "x": int(round(x)),
                "y": int(round(y)),
                "pattern_id": int((seed[0] * 1000 + idx * 7 + int(t * 3.0)) % 28),
                "bank": int((seed[3] * 4 + preset["palette_bias"]) % 4),
                "priority": priority,
                "flip_x": int((seed[4] * 10 + idx) % 2),
                "flip_y": int((seed[5] * 10 + idx * 2) % 2),
                "scale_mul": 1 + int(seed[6] > 0.68),
            }
        )
    return sprites


def _render_sprites(
    sprite_patterns: np.ndarray,
    sprites: List[Dict[str, int]],
    out_under_idx: np.ndarray,
    out_under_bank: np.ndarray,
    out_over_idx: np.ndarray,
    out_over_bank: np.ndarray,
    cfg: GenesisVDPConfig,
    t: float,
    visibility_field: np.ndarray | None = None,
) -> None:
    h, w = out_under_idx.shape
    for idx, sprite in enumerate(sprites):
        pattern = sprite_patterns[sprite["pattern_id"] % sprite_patterns.shape[0]]
        if sprite["flip_x"]:
            pattern = pattern[:, ::-1]
        if sprite["flip_y"]:
            pattern = pattern[::-1, :]

        scale = max(1, cfg.sprite_scale * sprite["scale_mul"])
        if scale > 1:
            pattern = np.repeat(np.repeat(pattern, scale, axis=0), scale, axis=1)

        sh, sw = pattern.shape
        x0 = sprite["x"] - sw // 2
        y0 = sprite["y"] - sh // 2
        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(w, x0 + sw)
        y2 = min(h, y0 + sh)
        if x1 >= x2 or y1 >= y2:
            continue

        src = pattern[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
        if cfg.sprite_attribute_glitch > 0.0 and ((idx + int(t * 5.0)) % 5 == 0):
            src = np.roll(src, int(round(math.sin(t * 13.0 + idx) * cfg.sprite_attribute_glitch * 2.0)), axis=1)
        if visibility_field is not None:
            vis = np.clip(visibility_field[y1:y2, x1:x2], 0.0, 1.0)
            src = np.clip(np.round(src.astype(np.float32) * vis), 0, 15).astype(np.uint8)
        mask = src > 0
        if not np.any(mask):
            continue

        if sprite["priority"] > 0:
            out_over_idx[y1:y2, x1:x2][mask] = src[mask]
            out_over_bank[y1:y2, x1:x2][mask] = sprite["bank"]
        else:
            out_under_idx[y1:y2, x1:x2][mask] = src[mask]
            out_under_bank[y1:y2, x1:x2][mask] = sprite["bank"]


def _apply_palette_corruption(palette: np.ndarray, cfg: GenesisVDPConfig, t: float) -> np.ndarray:
    if cfg.palette_corruption <= 0.0:
        return palette

    out = palette.astype(np.int16).copy()
    pulse = 0.5 + 0.5 * math.sin(t * (2.3 + cfg.palette_corruption * 6.0))
    for bank in range(out.shape[0]):
        color_shift = int(round(math.sin(t * (3.1 + bank * 0.8) + bank) * cfg.palette_corruption * 18.0))
        out[bank, 1:] = np.roll(out[bank, 1:], color_shift // 6, axis=0)
        out[bank] += int(round((bank - 1.5) * pulse * cfg.palette_corruption * 5.0))
    channel_offsets = np.array(
        [
            math.sin(t * 4.1),
            math.cos(t * 3.7 + 1.1),
            math.sin(t * 5.3 + 2.2),
        ],
        dtype=np.float32,
    ) * (cfg.palette_corruption * 18.0)
    out = out + channel_offsets.reshape(1, 1, 3).astype(np.int16)
    return np.clip(out, 0, 255).astype(np.uint8)


def _cram_palette_quantize(rgb: np.ndarray, palette: np.ndarray, cfg: GenesisVDPConfig) -> np.ndarray:
    palette_flat = palette.reshape(-1, 3).astype(np.int16)
    work = rgb.astype(np.float32)
    if cfg.dither_amount > 0.0:
        dither = _make_dither_map(rgb.shape[0], rgb.shape[1])
        work = work + (dither[..., None] - 0.5) * (cfg.dither_amount * 18.0)
    flat = np.clip(work, 0, 255).reshape(-1, 3).astype(np.int16)
    diff = flat[:, None, :] - palette_flat[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    idx = np.argmin(dist, axis=1)
    return palette_flat[idx].reshape(rgb.shape).astype(np.uint8)


def _apply_raster_bars(rgb: np.ndarray, raster_bar: np.ndarray, cfg: GenesisVDPConfig) -> np.ndarray:
    if cfg.shadow_highlight_strength <= 0.0 and cfg.register_instability <= 0.0:
        return rgb
    gain = (0.92 + raster_bar * 0.14 + cfg.register_instability * 0.04).astype(np.float32)
    return np.clip(rgb.astype(np.float32) * gain[:, None, None], 0, 255).astype(np.uint8)


def _apply_pixel_aspect_bias(rgb: np.ndarray, bias: float) -> np.ndarray:
    if abs(bias - 1.0) < 0.01:
        return rgb

    h, w = rgb.shape[:2]
    stretch_w = max(1, int(round(w * bias)))
    stretched = cv2.resize(rgb, (stretch_w, h), interpolation=cv2.INTER_NEAREST)
    if stretch_w > w:
        start = (stretch_w - w) // 2
        return stretched[:, start:start + w]
    pad_left = (w - stretch_w) // 2
    pad_right = w - stretch_w - pad_left
    return cv2.copyMakeBorder(stretched, 0, 0, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)


def _apply_crt_finish(frame: np.ndarray, cfg: GenesisVDPConfig) -> np.ndarray:
    out = frame.astype(np.float32)
    h, w = out.shape[:2]

    if cfg.composite_bleed > 0.0:
        bleed_px = max(1, int(round(1 + cfg.composite_bleed * 4.0)))
        ksize = max(3, bleed_px * 2 + 1)
        kernel = cv2.getGaussianKernel(ksize, max(0.5, cfg.composite_bleed * 1.7))
        kernel = (kernel @ kernel.T).astype(np.float32)
        kernel /= max(1e-6, float(kernel.sum()))
        blurred = cv2.filter2D(out, -1, kernel)
        out = out * (1.0 - cfg.composite_bleed * 0.38) + blurred * (cfg.composite_bleed * 0.38)

    if cfg.chroma_offset > 0.0:
        shift = max(1, int(round(cfg.chroma_offset)))
        red = np.roll(out[..., 0], shift, axis=1)
        blue = np.roll(out[..., 2], -shift, axis=1)
        out[..., 0] = red
        out[..., 2] = blue

    if cfg.phosphor_blur > 0.0:
        sigma = max(0.1, cfg.phosphor_blur * 0.9)
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma * 0.55)
        out = out * (1.0 - cfg.phosphor_blur * 0.24) + blur * (cfg.phosphor_blur * 0.24)

    if cfg.scanline_strength > 0.0:
        line_gain = np.ones((h, 1, 1), dtype=np.float32)
        line_gain[1::2] -= cfg.scanline_strength * 0.28
        line_gain[::2] += cfg.scanline_strength * 0.04
        out *= line_gain

    if cfg.crt_curve > 0.0:
        out = _apply_crt_curve(out, cfg.crt_curve)

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_crt_curve(frame: np.ndarray, amount: float) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / max(1.0, float(w - 1))) * 2.0 - 1.0
    yn = (yy / max(1.0, float(h - 1))) * 2.0 - 1.0
    r2 = xn * xn + yn * yn
    warp = 1.0 + r2 * amount * 0.12
    map_x = ((xn / warp) * 0.5 + 0.5) * (w - 1)
    map_y = ((yn / warp) * 0.5 + 0.5) * (h - 1)
    curved = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    vignette = np.clip(1.0 - r2 * amount * 0.18, 0.84, 1.0)
    return np.clip(curved.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)


def _normalize_field(field: np.ndarray) -> np.ndarray:
    fmin = float(field.min())
    fmax = float(field.max())
    if abs(fmax - fmin) < 1e-6:
        return np.zeros_like(field, dtype=np.float32)
    return ((field - fmin) / (fmax - fmin)).astype(np.float32)


def _stable_hash(text: str) -> int:
    return int(crc32(text.encode("utf-8")) & 0xFFFFFFFF)
