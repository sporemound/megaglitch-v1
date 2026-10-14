"""Stateful ModernGL bubble simulation generator node.

The node owns ping-pong floating point simulation textures and renders named
bubble source textures that downstream Lua/GLSL effects can sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from array import array

import moderngl
import numpy as np


_ROOT = Path(__file__).resolve().parents[2]
_SHADER_DIR = _ROOT / "shaders" / "bubble_sim"

_FULLSCREEN_VERT = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_POINT_VERT = """
#version 330 core

uniform sampler2D u_state;
uniform sampler2D u_velocity;
uniform float u_particle_texture_size;
uniform vec2 u_resolution;
uniform float u_show_energy;
uniform float u_perspective;
uniform float u_time;
uniform float u_motion_speed;
uniform float u_orbit_strength;
uniform float u_turbulence;
uniform float u_flow_speed;
uniform float u_depth_motion;
uniform float u_show_chaos;
uniform float u_buoyancy;
uniform float u_motion_style;

out float v_depth;
out float v_age;
out float v_seed;
out float v_radius;

void main() {
    float size = u_particle_texture_size;
    float id = float(gl_VertexID);
    float x = mod(id, size);
    float y = floor(id / size);
    vec2 uv = (vec2(x, y) + 0.5) / size;

    vec4 state = texture(u_state, uv);
    vec4 velocity = texture(u_velocity, uv);
    vec2 pos = state.xy * 2.0 - 1.0;
    float phase = id * 0.754877666;
    float drift_time = u_time * max(u_motion_speed, 0.0);
    float orbit_amount = 0.012 + u_orbit_strength * 0.040 + u_turbulence * 0.018;
    float drift_rate = 0.35 + u_flow_speed * 0.65;
    vec2 projected_drift = vec2(
        sin(drift_time * drift_rate * (0.75 + fract(phase) * 0.45) + phase * 6.2831853),
        cos(drift_time * drift_rate * (0.52 + fract(phase * 1.7) * 0.32) + phase * 4.3982297)
    );
    pos += projected_drift * orbit_amount * mix(1.0, 0.45, clamp(state.z, 0.0, 1.0));
    pos.x += sin(drift_time * 3.7 + phase * 11.0) * u_show_chaos * 0.025;
    pos.y += sin(drift_time * 0.9 + phase * 2.3) * u_depth_motion * 0.018;
    pos.y += u_buoyancy * (state.a - 0.5) * 0.10;
    if (u_motion_style > 1.5 && u_motion_style < 2.5) {
        float radius = length(pos);
        float angle = drift_time * (0.12 + u_orbit_strength * 0.18) * smoothstep(1.2, 0.0, radius);
        mat2 rotation = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        pos = rotation * pos;
    } else if (u_motion_style > 2.5) {
        vec2 radial = normalize(pos + vec2(0.0001));
        pos += radial * sin(drift_time * 1.7 + phase * 9.0) * (0.015 + u_show_chaos * 0.04);
    } else if (u_motion_style > 0.5) {
        pos.y += fract(state.a + drift_time * 0.08) * 0.08 - 0.04;
    }

    float depth_scale = mix(1.0 + u_perspective * 0.45, max(0.2, 1.0 - u_perspective * 0.55), clamp(state.z, 0.0, 1.0));
    float energy_scale = 1.0 + u_show_energy * 0.35;
    gl_Position = vec4(pos, 0.0, 1.0);
    gl_PointSize = max(1.0, velocity.a * u_resolution.y * depth_scale * energy_scale);

    v_depth = state.z;
    v_age = state.a;
    v_seed = id;
    v_radius = velocity.a;
}
"""

_SINGLE_OUTPUT_BUBBLES_FRAG = """
#version 330 core

in float v_depth;
in float v_age;
in float v_seed;
in float v_radius;
out vec4 fragColor;

uniform float u_density;
uniform float u_highlight_strength;
uniform float u_sparkle;
uniform float u_time;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

void main() {
    if (hash11(v_seed) > u_density) discard;
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) discard;
    float edge = smoothstep(1.0, 0.62, r2);
    float rim = smoothstep(0.45, 0.98, r2) * edge;
    float pop = smoothstep(0.78, 1.0, v_age);
    float glint = step(0.86, hash11(v_seed * 19.7 + floor(u_time * 18.0))) * pop * u_sparkle;
    float depth_gain = mix(1.0, 0.35, clamp(v_depth, 0.0, 1.0));
    vec3 highlight = vec3(0.72, 0.95, 1.0) * (rim * 0.35 + glint) * u_highlight_strength;
    fragColor = vec4(vec3(edge * depth_gain) + highlight, edge);
}
"""

_TRIANGLE = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]


@dataclass(frozen=True)
class ParamSpec:
    default: Any
    minimum: float | int | None = None
    maximum: float | int | None = None
    kind: type = float


class BubbleSimGeneratorNode:
    """Multipass GPU source node for real-time bubble physics."""

    PARAMS: dict[str, ParamSpec] = {
        "enabled": ParamSpec(True, kind=bool),
        "seed": ParamSpec(1977, 0, 999999, int),
        "particle_texture_size": ParamSpec(128, 16, 512, int),
        "density": ParamSpec(0.55, 0.0, 1.0, float),
        "base_radius": ParamSpec(0.018, 0.001, 0.08, float),
        "radius_variation": ParamSpec(0.65, 0.0, 1.0, float),
        "buoyancy": ParamSpec(0.16, 0.0, 2.0, float),
        "drag": ParamSpec(0.92, 0.0, 1.0, float),
        "turbulence": ParamSpec(0.35, 0.0, 2.0, float),
        "flow_speed": ParamSpec(0.25, 0.0, 4.0, float),
        "motion_speed": ParamSpec(1.0, 0.0, 4.0, float),
        "motion_style": ParamSpec(2.0, 0.0, 3.0, float),
        "depth_spread": ParamSpec(0.8, 0.0, 1.0, float),
        "depth_motion": ParamSpec(0.7, 0.0, 2.0, float),
        "orbit_strength": ParamSpec(0.8, 0.0, 2.0, float),
        "perspective": ParamSpec(1.0, 0.0, 2.0, float),
        "refraction_strength": ParamSpec(0.025, 0.0, 0.15, float),
        "highlight_strength": ParamSpec(0.45, 0.0, 2.0, float),
        "sparkle": ParamSpec(0.25, 0.0, 1.0, float),
        "show_energy": ParamSpec(0.0, 0.0, 1.0, float),
        "show_chaos": ParamSpec(0.0, 0.0, 1.0, float),
        "mutation_strength": ParamSpec(0.0, 0.0, 1.0, float),
        "mutation_rate": ParamSpec(1.0, 0.0, 5.0, float),
        "mutation_warp": ParamSpec(0.5, 0.0, 2.0, float),
        "mutation_palette": ParamSpec(0.4, 0.0, 2.0, float),
        "mutation_fracture": ParamSpec(0.4, 0.0, 1.0, float),
        "mutation_echo": ParamSpec(0.35, 0.0, 1.0, float),
        "debug_mode": ParamSpec(0.0, 0.0, 3.0, float),
    }

    OUTPUT_NAMES = (
        "bubble_mask",
        "bubble_normal",
        "bubble_depth",
        "bubble_refraction",
        "bubble_highlight",
        "bubble_debug",
    )

    def __init__(self, gl_context: moderngl.Context, width: int, height: int, particle_texture_size: int = 128):
        self.ctx = gl_context
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.params = {name: spec.default for name, spec in self.PARAMS.items()}
        self._programs: dict[str, moderngl.Program] = {}
        self._vaos: dict[str, moderngl.VertexArray] = {}
        self._buffers: dict[str, moderngl.Buffer] = {}
        self._state_textures: list[moderngl.Texture] = []
        self._velocity_textures: list[moderngl.Texture] = []
        self._state_fbos: list[moderngl.Framebuffer] = []
        self._state_index = 0
        self._outputs: dict[str, moderngl.Texture] = {}
        self._render_fbo: moderngl.Framebuffer | None = None
        self._debug_fbo: moderngl.Framebuffer | None = None
        self._mrt_render = True
        self._initialized = False
        self._last_time = 0.0

        self.set_param("particle_texture_size", particle_texture_size)
        self.load_shaders()
        self._allocate_simulation_textures()
        self._allocate_output_textures()
        self.initialize_state(self.get_param("seed"))

    @property
    def particle_texture_size(self) -> int:
        return int(self.get_param("particle_texture_size"))

    @property
    def particle_count(self) -> int:
        s = self.particle_texture_size
        return s * s

    def load_shaders(self):
        init_frag = self._read_shader("init_state.frag")
        update_frag = self._read_shader("update_state.frag")
        render_frag = self._read_shader("render_bubbles.frag")
        debug_frag = self._read_shader("composite_debug.frag")

        self._programs["init"] = self._compile("init_state", _FULLSCREEN_VERT, init_frag)
        self._programs["update"] = self._compile("update_state", _FULLSCREEN_VERT, update_frag)
        self._programs["debug"] = self._compile("composite_debug", _FULLSCREEN_VERT, debug_frag)

        max_attachments = self._max_color_attachments()
        self._mrt_render = max_attachments >= 5
        render_source = render_frag if self._mrt_render else _SINGLE_OUTPUT_BUBBLES_FRAG
        self._programs["render"] = self._compile("render_bubbles", _POINT_VERT, render_source)

        self._buffers["fullscreen"] = self.ctx.buffer(array("f", _TRIANGLE).tobytes())
        self._vaos["init"] = self.ctx.simple_vertex_array(self._programs["init"], self._buffers["fullscreen"], "in_pos")
        self._vaos["update"] = self.ctx.simple_vertex_array(self._programs["update"], self._buffers["fullscreen"], "in_pos")
        self._vaos["debug"] = self.ctx.simple_vertex_array(self._programs["debug"], self._buffers["fullscreen"], "in_pos")
        self._vaos["render"] = self.ctx.vertex_array(self._programs["render"], [])

    def initialize_state(self, seed):
        self.set_param("seed", seed)
        s = self.particle_texture_size
        fbo = self._state_fbos[self._state_index]
        fbo.use()
        self.ctx.viewport = (0, 0, s, s)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        prog = self._programs["init"]
        self._set_uniform(prog, "u_seed", int(self.get_param("seed")))
        self._set_uniform(prog, "u_particle_texture_size", float(s))
        self._set_uniform(prog, "u_base_radius", float(self.get_param("base_radius")))
        self._set_uniform(prog, "u_radius_variation", float(self.get_param("radius_variation")))
        self._set_uniform(prog, "u_depth_spread", float(self.get_param("depth_spread")))
        self._vaos["init"].render(mode=moderngl.TRIANGLES)
        self._initialized = True

    def reset(self, seed=None):
        if seed is not None:
            self.set_param("seed", seed)
        self._state_index = 0
        self.initialize_state(self.get_param("seed"))
        self._clear_outputs()

    def resize(self, width: int, height: int):
        width = max(1, int(width))
        height = max(1, int(height))
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        self._release_outputs()
        self._allocate_output_textures()

    def set_param(self, name: str, value):
        if name not in self.PARAMS:
            raise KeyError(f"Unknown BubbleSimGeneratorNode parameter: {name}")
        spec = self.PARAMS[name]
        if spec.kind is bool:
            coerced = bool(value)
        else:
            coerced = spec.kind(value)
            if spec.minimum is not None:
                coerced = max(spec.minimum, coerced)
            if spec.maximum is not None:
                coerced = min(spec.maximum, coerced)
        self.params[name] = coerced

        if name == "particle_texture_size" and self._state_textures:
            self._release_simulation()
            self._allocate_simulation_textures()
            self.initialize_state(self.get_param("seed"))

    def get_param(self, name: str):
        if name not in self.PARAMS:
            raise KeyError(f"Unknown BubbleSimGeneratorNode parameter: {name}")
        return self.params[name]

    def update(self, time: float, dt: float, slot_time: float):
        self._last_time = float(time)
        if not self.get_param("enabled"):
            return
        if not self._initialized:
            self.initialize_state(self.get_param("seed"))

        src = self._state_index
        dst = 1 - self._state_index
        s = self.particle_texture_size

        self._state_fbos[dst].use()
        self.ctx.viewport = (0, 0, s, s)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        self._state_textures[src].use(location=0)
        self._velocity_textures[src].use(location=1)
        prog = self._programs["update"]
        self._set_uniform(prog, "u_state_prev", 0)
        self._set_uniform(prog, "u_velocity_prev", 1)
        self._set_uniform(prog, "u_time", float(time))
        self._set_uniform(prog, "u_dt", float(dt))
        self._set_uniform(prog, "u_seed", int(self.get_param("seed")))
        self._set_uniform(prog, "u_particle_texture_size", float(s))
        self._set_uniform(prog, "u_buoyancy", float(self.get_param("buoyancy")))
        self._set_uniform(prog, "u_motion_style", float(self.get_param("motion_style")))
        self._set_uniform(prog, "u_drag", float(self.get_param("drag")))
        self._set_uniform(prog, "u_turbulence", float(self.get_param("turbulence")))
        self._set_uniform(prog, "u_flow_speed", float(self.get_param("flow_speed")))
        self._set_uniform(prog, "u_motion_speed", float(self.get_param("motion_speed")))
        self._set_uniform(prog, "u_base_radius", float(self.get_param("base_radius")))
        self._set_uniform(prog, "u_radius_variation", float(self.get_param("radius_variation")))
        self._set_uniform(prog, "u_depth_spread", float(self.get_param("depth_spread")))
        self._set_uniform(prog, "u_depth_motion", float(self.get_param("depth_motion")))
        self._set_uniform(prog, "u_orbit_strength", float(self.get_param("orbit_strength")))
        self._set_uniform(prog, "u_show_energy", float(self.get_param("show_energy")))
        self._set_uniform(prog, "u_show_chaos", float(self.get_param("show_chaos")))
        self._set_uniform(prog, "u_mutation_strength", float(self.get_param("mutation_strength")))
        self._set_uniform(prog, "u_mutation_rate", float(self.get_param("mutation_rate")))
        self._set_uniform(prog, "u_mutation_warp", float(self.get_param("mutation_warp")))
        self._set_uniform(prog, "u_mutation_fracture", float(self.get_param("mutation_fracture")))
        self._vaos["update"].render(mode=moderngl.TRIANGLES)
        self._state_index = dst

    def render(self):
        self._clear_outputs()
        fbo = self._render_fbo
        if fbo is None:
            return

        fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)

        self._state_textures[self._state_index].use(location=0)
        self._velocity_textures[self._state_index].use(location=1)
        prog = self._programs["render"]
        self._set_uniform(prog, "u_state", 0)
        self._set_uniform(prog, "u_velocity", 1)
        self._set_uniform(prog, "u_particle_texture_size", float(self.particle_texture_size))
        self._set_uniform(prog, "u_resolution", (float(self.width), float(self.height)))
        self._set_uniform(prog, "u_density", float(self.get_param("density")))
        self._set_uniform(prog, "u_refraction_strength", float(self.get_param("refraction_strength")))
        self._set_uniform(prog, "u_highlight_strength", float(self.get_param("highlight_strength")))
        self._set_uniform(prog, "u_sparkle", float(self.get_param("sparkle")))
        self._set_uniform(prog, "u_show_energy", float(self.get_param("show_energy")))
        self._set_uniform(prog, "u_show_chaos", float(self.get_param("show_chaos")))
        self._set_uniform(prog, "u_perspective", float(self.get_param("perspective")))
        self._set_uniform(prog, "u_motion_speed", float(self.get_param("motion_speed")))
        self._set_uniform(prog, "u_orbit_strength", float(self.get_param("orbit_strength")))
        self._set_uniform(prog, "u_turbulence", float(self.get_param("turbulence")))
        self._set_uniform(prog, "u_flow_speed", float(self.get_param("flow_speed")))
        self._set_uniform(prog, "u_depth_motion", float(self.get_param("depth_motion")))
        self._set_uniform(prog, "u_buoyancy", float(self.get_param("buoyancy")))
        self._set_uniform(prog, "u_motion_style", float(self.get_param("motion_style")))
        self._set_uniform(prog, "u_mutation_strength", float(self.get_param("mutation_strength")))
        self._set_uniform(prog, "u_mutation_rate", float(self.get_param("mutation_rate")))
        self._set_uniform(prog, "u_mutation_palette", float(self.get_param("mutation_palette")))
        self._set_uniform(prog, "u_time", self._last_time)

        old_blend = getattr(self.ctx, "enable_flags", moderngl.NOTHING)
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        # MRT values already encode their own coverage. Add them directly
        # instead of multiplying every buffer by alpha a second time.
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._vaos["render"].render(mode=moderngl.POINTS, vertices=self.particle_count)
        try:
            self.ctx.enable_only(old_blend)
        except Exception:
            self.ctx.disable(moderngl.BLEND)
        self._render_debug()

    def get_output_texture(self, name: str):
        if name not in self._outputs:
            raise KeyError(f"Unknown bubble output texture: {name}")
        return self._outputs[name]

    def get_output_textures(self):
        return dict(self._outputs)

    def destroy(self):
        for vao in self._vaos.values():
            vao.release()
        for buf in self._buffers.values():
            buf.release()
        for prog in self._programs.values():
            prog.release()
        self._release_outputs()
        self._release_simulation()
        self._vaos.clear()
        self._buffers.clear()
        self._programs.clear()

    def _read_shader(self, name: str) -> str:
        path = _SHADER_DIR / name
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Unable to read bubble simulation shader {path}: {exc}") from exc

    def _compile(self, label: str, vertex_shader: str, fragment_shader: str):
        try:
            return self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        except Exception as exc:
            raise RuntimeError(f"BubbleSimGeneratorNode shader compile/link failed for {label}: {exc}") from exc

    def _make_texture(self, size: tuple[int, int], dtype: str = "f4"):
        tex = self.ctx.texture(size, components=4, dtype=dtype)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def _allocate_simulation_textures(self):
        max_attachments = self._max_color_attachments()
        if max_attachments < 2:
            raise RuntimeError("BubbleSimGeneratorNode requires at least 2 color attachments for state ping-pong simulation")
        s = self.particle_texture_size
        self._state_textures = [self._make_texture((s, s), "f4"), self._make_texture((s, s), "f4")]
        self._velocity_textures = [self._make_texture((s, s), "f4"), self._make_texture((s, s), "f4")]
        self._state_fbos = [
            self.ctx.framebuffer(color_attachments=[self._state_textures[0], self._velocity_textures[0]]),
            self.ctx.framebuffer(color_attachments=[self._state_textures[1], self._velocity_textures[1]]),
        ]
        self._state_index = 0

    def _allocate_output_textures(self):
        # Render outputs are normalized visual buffers. RGBA8 reduces readback
        # bandwidth by 4x versus float32 while simulation state remains float32.
        self._outputs = {name: self._make_texture((self.width, self.height), "f1") for name in self.OUTPUT_NAMES}
        if self._mrt_render:
            self._render_fbo = self.ctx.framebuffer(
                color_attachments=[
                    self._outputs["bubble_mask"],
                    self._outputs["bubble_normal"],
                    self._outputs["bubble_depth"],
                    self._outputs["bubble_refraction"],
                    self._outputs["bubble_highlight"],
                ]
            )
        else:
            self._render_fbo = self.ctx.framebuffer(color_attachments=[self._outputs["bubble_mask"]])
            for alias in ("bubble_normal", "bubble_depth", "bubble_refraction", "bubble_highlight"):
                self._outputs[alias] = self._outputs["bubble_mask"]
        self._debug_fbo = self.ctx.framebuffer(color_attachments=[self._outputs["bubble_debug"]])
        self._clear_outputs()

    def _release_simulation(self):
        for fbo in self._state_fbos:
            fbo.release()
        for tex in self._state_textures + self._velocity_textures:
            tex.release()
        self._state_fbos = []
        self._state_textures = []
        self._velocity_textures = []
        self._initialized = False

    def _release_outputs(self):
        if self._render_fbo is not None:
            self._render_fbo.release()
            self._render_fbo = None
        if self._debug_fbo is not None:
            self._debug_fbo.release()
            self._debug_fbo = None
        released = set()
        for tex in self._outputs.values():
            if id(tex) not in released:
                tex.release()
                released.add(id(tex))
        self._outputs = {}

    def _clear_outputs(self):
        if self._render_fbo is not None:
            self._render_fbo.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        if self._debug_fbo is not None:
            self._debug_fbo.use()
            self.ctx.viewport = (0, 0, self.width, self.height)
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)

    def _render_debug(self):
        if self._debug_fbo is None:
            return
        self._debug_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self._outputs["bubble_mask"].use(location=0)
        self._outputs["bubble_normal"].use(location=1)
        self._outputs["bubble_depth"].use(location=2)
        self._outputs["bubble_refraction"].use(location=3)
        self._outputs["bubble_highlight"].use(location=4)
        prog = self._programs["debug"]
        self._set_uniform(prog, "u_bubble_mask", 0)
        self._set_uniform(prog, "u_bubble_normal", 1)
        self._set_uniform(prog, "u_bubble_depth", 2)
        self._set_uniform(prog, "u_bubble_refraction", 3)
        self._set_uniform(prog, "u_bubble_highlight", 4)
        self._set_uniform(prog, "u_debug_mode", float(self.get_param("debug_mode")))
        self._vaos["debug"].render(mode=moderngl.TRIANGLES)

    @staticmethod
    def _set_uniform(program: moderngl.Program, name: str, value):
        try:
            uniform = program[name]
        except Exception:
            return
        uniform.value = tuple(value) if isinstance(value, tuple) else value

    def _max_color_attachments(self) -> int:
        direct = getattr(self.ctx, "max_color_attachments", None)
        if direct is not None:
            return int(direct)
        info = getattr(self.ctx, "info", {}) or {}
        return int(info.get("GL_MAX_COLOR_ATTACHMENTS", 1))


class BubbleSimGenerator2Runtime(BubbleSimGeneratorNode):
    """V2 runtime with refractive defaults and a three-stage readback ring."""

    DEFAULT_OVERRIDES = {
        "density": 0.12,
        "base_radius": 0.021,
        "radius_variation": 0.8,
        "buoyancy": 0.02,
        "drag": 0.94,
        "turbulence": 0.30,
        "flow_speed": 0.22,
        "depth_spread": 0.95,
        "depth_motion": 0.35,
        "orbit_strength": 0.70,
        "perspective": 1.2,
        "refraction_strength": 0.040,
        "highlight_strength": 0.34,
        "sparkle": 0.04,
    }

    def __init__(self, gl_context, width, height, particle_texture_size=64):
        self._readback_buffers = []
        self._readback_index = 0
        self._readback_frames = 0
        super().__init__(gl_context, width, height, particle_texture_size)
        for name, value in self.DEFAULT_OVERRIDES.items():
            self.set_param(name, value)
        if not self._mrt_render:
            raise RuntimeError(
                "Bubble Sim Gen (2) requires at least 5 color attachments for distinct GPU outputs"
            )
        self._allocate_readback_ring()
        self.reset(self.get_param("seed"))

    @property
    def runtime_mode(self):
        return "GPU"

    def resize(self, width, height):
        old_size = (self.width, self.height)
        super().resize(width, height)
        if old_size != (self.width, self.height):
            self._release_readback_ring()
            self._allocate_readback_ring()

    def apply_params(self, params):
        for name in self.PARAMS:
            if name not in params:
                continue
            value = params[name]
            if name == "debug_mode" and isinstance(value, str):
                value = {
                    "composite": 0.0,
                    "mask": 0.0,
                    "normal": 1.0,
                    "depth": 2.0,
                    "refraction": 3.0,
                    "highlight": 0.0,
                }.get(value, 0.0)
            if name == "motion_style" and isinstance(value, str):
                value = {
                    "Fluid Drift": 0.0,
                    "Rising": 1.0,
                    "Vortex": 2.0,
                    "Burst": 3.0,
                }.get(value, 2.0)
            self.set_param(name, value)

    def render_and_readback(self):
        self.render()
        output_names = (
            "bubble_mask",
            "bubble_normal",
            "bubble_depth",
            "bubble_refraction",
            "bubble_highlight",
        )
        write_slot = self._readback_buffers[self._readback_index]
        for name in output_names:
            self._outputs[name].read_into(write_slot[name], alignment=1)

        ready = None
        if self._readback_frames >= 2:
            ready_index = (self._readback_index + 1) % 3
            ready = {}
            for name in output_names:
                raw = self._readback_buffers[ready_index][name].read()
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 4))
                ready[name] = np.flipud(arr[..., :3]).copy()

        self._readback_index = (self._readback_index + 1) % 3
        self._readback_frames += 1
        return ready

    def destroy(self):
        self._release_readback_ring()
        super().destroy()

    def _allocate_readback_ring(self):
        byte_count = self.width * self.height * 4
        names = (
            "bubble_mask",
            "bubble_normal",
            "bubble_depth",
            "bubble_refraction",
            "bubble_highlight",
        )
        self._readback_buffers = [
            {name: self.ctx.buffer(reserve=byte_count) for name in names}
            for _ in range(3)
        ]
        self._readback_index = 0
        self._readback_frames = 0

    def _release_readback_ring(self):
        for slot in self._readback_buffers:
            for buffer in slot.values():
                try:
                    buffer.release()
                except Exception:
                    pass
        self._readback_buffers = []
