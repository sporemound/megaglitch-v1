# lua_shader_node.py
import os, time, threading
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import moderngl
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from lupa import LuaRuntime

# ============ Utilities ============

class _ReloadFlag:
    def __init__(self): self._flag=False; self._lock=threading.Lock()
    def set(self):       self._lock.acquire(); self._flag=True; self._lock.release()
    def take(self)->bool:
        self._lock.acquire(); v=self._flag; self._flag=False; self._lock.release(); return v

class _WatchHandler(FileSystemEventHandler):
    def __init__(self, paths, flag:_ReloadFlag):
        self.paths = [os.path.normpath(p) for p in paths]
        self.flag  = flag
    def on_modified(self, e):
        if e.is_directory: return
        p = os.path.normpath(e.src_path)
        if any(p.endswith(x) or x in p for x in self.paths):
            self.flag.set()

def _make_lua_runtime():
    lua = LuaRuntime(unpack_returned_tuples=True, register_eval=False)
    import math
    allowed_math = {
        "abs":abs,"min":min,"max":max,"sin":math.sin,"cos":math.cos,"tan":math.tan,
        "asin":math.asin,"acos":math.acos,"atan":math.atan,"sqrt":math.sqrt,"pi":math.pi,
        "floor":math.floor,"ceil":math.ceil,"pow":math.pow,"exp":math.exp,"log":math.log,
    }
    class RNG:
        def __init__(self, seed=1337): self.state=seed&0xFFFFFFFF
        def seed(self, s): self.state=int(s)&0xFFFFFFFF
        def next(self):
            self.state=(1664525*self.state+1013904223)&0xFFFFFFFF
            return self.state/0xFFFFFFFF
    rng = RNG()

    api = {"version":"1.0","math":allowed_math,"rng_seed":rng.seed,"rng_next":rng.next}
    lua.globals()["_PY_API"]=api
    return lua

def _load_lua_module(lua:LuaRuntime, path:str):
    code = open(path,"r",encoding="utf-8").read()
    wrapper = f"""
        local mod=(function()
{code}
        end)()
        return mod
    """
    mod = lua.execute(wrapper)
    per_frame = mod["per_frame"] if mod else None
    update = mod["update"] if mod else None
    if mod and not per_frame and update:
        mod["per_frame"] = update
        per_frame = update
    if not mod or not per_frame:
        raise RuntimeError("Lua script must return a table with per_frame(state, api)")
    return mod

VERT_SRC = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main(){
  v_uv = in_pos*0.5 + 0.5;
  gl_Position = vec4(in_pos,0,1);
}
"""
FS_TRI = np.array([-1.0,-1.0, 3.0,-1.0, -1.0,3.0], dtype="f4")

DEFAULT_FRAG = """// fallback fragment so node still runs if your file is empty
#version 330
uniform sampler2D uPrev;
uniform sampler2D uInput0;
uniform vec2  uResolution;
uniform float uTime;
uniform int   uFrame;
// overridable from Lua:
uniform float uDomainWarp;
uniform float uFeedbackMix;
uniform float uHueShift;

in vec2 v_uv;
out vec4 fragColor;

vec2 hash2(vec2 p){
    p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
    return fract(sin(p)*43758.5453123);
}
vec3 hsv2rgb(vec3 c){
    vec3 p = abs(fract(c.xxx+vec3(0,2./3.,1./3.))*6.-3.);
    return c.z * mix(vec3(1.), clamp(p-1.,0.,1.), c.y);
}

void main(){
    vec2 uv = v_uv;
    float warp = uDomainWarp;
    vec2 q = uv + (hash2(uv+uTime*0.11)-0.5)*0.15*warp;
    q     += (hash2(uv*2.3+uTime*0.07)-0.5)*0.10*warp;
    vec4 prev = texture(uPrev,q);

    float hue = fract(uHueShift*0.5 + sin(uTime*0.1 + uv.x*3.14)*0.05 + prev.r*0.1);
    vec3 tint = hsv2rgb(vec3(hue,0.6,0.8));
    vec3 base = mix(prev.rgb,tint,0.25);
    vec3 col  = mix(prev.rgb,base,clamp(uFeedbackMix,0.0,1.0));
    fragColor = vec4(col,1.0);
}
"""

@dataclass
class _GLKit:
    prog: moderngl.Program
    vao: moderngl.VertexArray
    vbo: moderngl.Buffer
    fb: list
    tex: list
    prev_idx: int = 0

def _build_gl(ctx:moderngl.Context, frag_src:str, size)->_GLKit:
    prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=frag_src)
    vbo  = ctx.buffer(FS_TRI.tobytes())
    vao  = ctx.simple_vertex_array(prog, vbo, "in_pos")

    w,h = size
    # Use RGBA8 normalized for speed; sufficient for glitch feedback
    texA = ctx.texture((w,h), components=4, dtype="f1")
    texB = ctx.texture((w,h), components=4, dtype="f1")
    for t in (texA, texB):
        t.filter=(moderngl.LINEAR,moderngl.LINEAR)
        try:
            t.repeat_x = True; t.repeat_y = True
        except Exception:
            pass
    fboA = ctx.framebuffer(color_attachments=[texA])
    fboB = ctx.framebuffer(color_attachments=[texB])
    for fb in (fboA,fboB):
        fb.use(); ctx.clear(0,0,0,1)

    return _GLKit(prog=prog, vao=vao, vbo=vbo, fb=[fboA,fboB], tex=[texA,texB], prev_idx=0)

# ============ LuaShaderNode ============

class LuaShaderNode:
    """
    Texture-in/texture-out node.
    - Inputs: optional previous stage texture (sampled as uInput0 at unit 1)
    - Internal feedback: ping-pong texture bound as uPrev (unit 0)
    - Params: dict[str,float] merged with Lua outputs -> uniforms
    """
    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        shader_path: str,
        lua_path: str,
        auto_reload: bool = True,
        static_uniforms: Optional[Dict[str, float]] = None,
    ):
        self.ctx = ctx
        self.size = (int(width), int(height))
        self.shader_path = shader_path
        self.lua_path = lua_path
        self.auto_reload = auto_reload
        self._reload_flag = _ReloadFlag()
        self._observer: Optional[Observer] = None

        # time/frame counters owned by node (host can override via set_time)
        self._t0 = time.perf_counter()
        self._time = 0.0
        self._frame = 0
        self._frozen = False
        self._frozen_at = 0.0

        # boot resources
        self._lua = _make_lua_runtime()
        self._lua_mod = _load_lua_module(self._lua, self.lua_path)
        self._frag_src = self._read_shader_or_default(self.shader_path)
        self._gl = _build_gl(self.ctx, self._frag_src, self.size)
        # Minimize GL state overhead
        try:
            self.ctx.enable_only(moderngl.NOTHING)
        except Exception:
            pass
        # Seed feedback so first frame is not black
        try:
            w,h = self.size
            seed = (np.random.rand(h, w, 4) * 255).astype('u1')
            self._gl.tex[self._gl.prev_idx].write(seed.tobytes())
        except Exception:
            pass

        # track delta-time between renders
        self._last_render_time = float(self.time)

        # static uniforms (e.g., resolution)
        self._bind_static_uniforms(static_uniforms or {})

        if self.auto_reload:
            self._observer = Observer()
            self._observer.schedule(
                _WatchHandler([self.shader_path, self.lua_path], self._reload_flag),
                path=os.path.dirname(os.path.abspath(self.shader_path)) or ".",
                recursive=False
            )
            if os.path.dirname(self.lua_path) != os.path.dirname(self.shader_path):
                self._observer.schedule(
                    _WatchHandler([self.shader_path, self.lua_path], self._reload_flag),
                    path=os.path.dirname(os.path.abspath(self.lua_path)) or ".",
                    recursive=False
                )
            self._observer.start()

    # ---------- lifecycle ----------
    def dispose(self):
        if self._observer:
            self._observer.stop(); self._observer.join(timeout=1.0)
        try:
            for fb in self._gl.fb: fb.release()
            for t in self._gl.tex: t.release()
            self._gl.vbo.release()
            self._gl.vao.release()
            self._gl.prog.release()
        except Exception:
            pass

    # ---------- time control ----------
    def set_time(self, t_seconds: float): self._time = float(t_seconds)
    def step_frame(self): self._frame += 1
    def freeze(self, on: bool):
        if on and not self._frozen:
            self._frozen_at = self.time
        if not on and self._frozen:
            # resume continuous time
            self._t0 = time.perf_counter() - self._frozen_at
        self._frozen = on

    @property
    def time(self)->float:
        if self._time:  # host-driven time
            return self._time if self._frozen else self._time + 0.0
        if self._frozen: return self._frozen_at
        return time.perf_counter() - self._t0

    # ---------- I/O ----------
    def output_texture(self) -> moderngl.Texture:
        """Texture produced by the last render() call."""
        return self._gl.tex[self._gl.prev_idx]

    # ---------- main render ----------
    def render(
        self,
        external_params: Optional[Dict[str, float]] = None,
        input_tex: Optional[moderngl.Texture] = None,
        named_input_textures: Optional[Dict[str, moderngl.Texture]] = None,
    ):
        """Run one frame. Optionally pass params and an input texture (uInput0)."""
        if self._reload_flag.take():
            self._reload()

        # resize if backend changed
        if self._gl.tex[0].size != self.size:
            self._gl = _build_gl(self.ctx, self._frag_src, self.size)
            self._bind_static_uniforms({})

        # gather params from Lua
        ui_params = {}
        if external_params:
            for k, v in external_params.items():
                try:
                    ui_params[str(k)] = float(v)
                except (TypeError, ValueError):
                    try:
                        ui_params[str(k)] = float(bool(v))
                    except Exception:
                        pass
        state = {"time": float(self.time), "frame": int(self._frame), "knobs": {}, "params": dict(ui_params)}
        lua_params: Dict[str,float] = {}
        try:
            out = self._lua_mod["per_frame"](state, self._lua.globals()._PY_API)
            if out and hasattr(out,"items"):
                for k,v in out.items():
                    try: lua_params[str(k)] = float(v)
                    except: pass
        except Exception as e:
            # keep running on Lua errors
            print("[LuaShaderNode][Lua] runtime error:", e)

        # merge order: external overrides Lua if same key
        params = dict(ui_params)
        params.update(lua_params)

        # ping-pong
        prev, cur = self._gl.prev_idx, 1 - self._gl.prev_idx
        self._gl.fb[cur].use()
        w,h = self._gl.tex[cur].size
        self.ctx.viewport = (0,0,w,h)
        self.ctx.clear(0,0,0,1)

        # bind textures
        self._gl.tex[prev].use(location=0)  # uPrev
        if input_tex is not None:
            input_tex.use(location=1)       # uInput0 (optional)
        named_units = {}
        for unit, (uniform_name, texture) in enumerate(
            (named_input_textures or {}).items(),
            start=2,
        ):
            texture.use(location=unit)
            named_units[str(uniform_name)] = unit

        # bind uniforms and draw
        now = float(self.time)
        delta = now - getattr(self, '_last_render_time', now)
        self._bind_dynamic_uniforms(
            params,
            input_bound=(input_tex is not None),
            delta=delta,
            named_units=named_units,
        )
        self._gl.vao.render(mode=moderngl.TRIANGLES)

        # swap
        self._gl.prev_idx = cur
        self._frame += 1
        self._last_render_time = now

    # ---------- internals ----------
    def _read_shader_or_default(self, path)->str:
        try:
            return open(path,"r",encoding="utf-8").read()
        except Exception:
            print("[LuaShaderNode] Using DEFAULT_FRAG (couldn't read shader).")
            return DEFAULT_FRAG

    def _reload(self):
        print("[LuaShaderNode] Hot-reload...")
        # Shader
        try:
            self._frag_src = self._read_shader_or_default(self.shader_path)
            self._gl = _build_gl(self.ctx, self._frag_src, self.size)
            self._bind_static_uniforms({})
        except Exception as e:
            print("[LuaShaderNode][Shader] ERROR:", e)
        # Lua
        try:
            self._lua = _make_lua_runtime()
            self._lua_mod = _load_lua_module(self._lua, self.lua_path)
        except Exception as e:
            print("[LuaShaderNode][Lua] ERROR:", e)

    def _bind_static_uniforms(self, extra: Dict[str,float]):
        def setu(name, *vals):
            try:
                u = self._gl.prog[name]
                u.value = vals[0] if len(vals)==1 else tuple(vals)
            except Exception:
                pass
        w,h = self.size
        setu("uResolution", float(w), float(h))
        setu("u_resolution", float(w), float(h))
        for k,v in extra.items(): setu(k, float(v))

    def _bind_dynamic_uniforms(
        self,
        params: Dict[str, float],
        input_bound: bool,
        delta: float,
        named_units: Optional[Dict[str, int]] = None,
    ):
        prog = self._gl.prog
        def setu(name, *vals):
            try:
                u = prog[name]
                u.value = vals[0] if len(vals)==1 else tuple(vals)
            except Exception:
                pass
        setu("uTime", float(self.time))
        setu("u_time", float(self.time))
        setu("uDelta", float(delta))
        setu("u_delta", float(delta))
        setu("uFrame", int(self._frame))
        setu("u_frame", int(self._frame))
        setu("uPrev", 0)
        setu("u_prev", 0)
        if input_bound:
            setu("uInput0", 1)
            setu("u_input0", 1)
        for name, unit in (named_units or {}).items():
            setu(name, int(unit))
        for k,v in params.items(): setu(k, float(v))
