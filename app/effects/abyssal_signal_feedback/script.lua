local M = {}

local SLOT_LENGTH = 3600.0

local defaults = {
  master = 0.8,
  bpm = 92.0,
  manual_energy = 0.0,
  manual_chaos = 0.0,
  phase_override = -1,
  bubble_mix = 0.0,
  bubble_refraction_mix = 0.0,
  bubble_highlight_mix = 0.0
}

local phase_starts = { 0.0, 480.0, 1080.0, 1680.0, 2280.0, 2880.0, 3420.0 }
local phase_ends = { 480.0, 1080.0, 1680.0, 2280.0, 2880.0, 3420.0, 3600.0 }

local phase_energy = { 0.10, 0.24, 0.42, 0.58, 0.74, 0.95, 0.18 }
local phase_chaos = { 0.04, 0.14, 0.44, 0.52, 0.68, 0.96, 0.10 }

local function clamp(v, lo, hi)
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

local function saturate(v)
  return clamp(v, 0.0, 1.0)
end

local function smoothstep(edge0, edge1, x)
  local t = saturate((x - edge0) / (edge1 - edge0))
  return t * t * (3.0 - 2.0 * t)
end

local function param(ctx, name)
  local value = ctx:param(name)
  if value == nil then
    return defaults[name]
  end
  return value
end

local function find_phase(slot_time)
  for i = 1, 7 do
    if slot_time < phase_ends[i] then
      return i - 1
    end
  end
  return 6
end

local function section_progress(slot_time, phase)
  local index = phase + 1
  local start_time = phase_starts[index]
  local end_time = phase_ends[index]
  return saturate((slot_time - start_time) / (end_time - start_time))
end

function M.init(ctx)
  ctx:set_param("master", defaults.master)
  ctx:set_param("bpm", defaults.bpm)
  ctx:set_param("manual_energy", defaults.manual_energy)
  ctx:set_param("manual_chaos", defaults.manual_chaos)
  ctx:set_param("phase_override", defaults.phase_override)
  ctx:set_param("bubble_mix", defaults.bubble_mix)
  ctx:set_param("bubble_refraction_mix", defaults.bubble_refraction_mix)
  ctx:set_param("bubble_highlight_mix", defaults.bubble_highlight_mix)
end

function M.update(ctx)
  local time = ctx:time()
  local slot_time = time % SLOT_LENGTH

  local master = saturate(param(ctx, "master"))
  local bpm = clamp(param(ctx, "bpm"), 40.0, 180.0)
  local manual_energy = saturate(param(ctx, "manual_energy"))
  local manual_chaos = saturate(param(ctx, "manual_chaos"))
  local bubble_mix = saturate(param(ctx, "bubble_mix"))
  local bubble_refraction_mix = saturate(param(ctx, "bubble_refraction_mix"))
  local bubble_highlight_mix = clamp(param(ctx, "bubble_highlight_mix"), 0.0, 2.0)
  local override = math.floor(param(ctx, "phase_override") + 0.5)

  local phase = find_phase(slot_time)
  if override >= 0 and override <= 6 then
    phase = override
  end

  local progress = section_progress(slot_time, phase)
  local phase_index = phase + 1
  local rise = smoothstep(0.0, 1.0, progress)
  local breathe = 0.5 + 0.5 * math.sin(time * 0.035 + phase * 0.71)
  local slow_tide = 0.5 + 0.5 * math.sin(time * 0.011)

  local beat_phase = (time * bpm / 60.0) % 1.0
  local beat = math.exp(-beat_phase * 9.0)
  beat = saturate(beat + 0.35 * math.exp(-(((beat_phase - 0.5) * 12.0) * ((beat_phase - 0.5) * 12.0))))

  local energy = saturate(phase_energy[phase_index] * (0.72 + 0.28 * rise) + manual_energy + beat * 0.10)
  local chaos = saturate(phase_chaos[phase_index] * (0.65 + 0.35 * rise) + manual_chaos + beat * 0.08)

  if phase == 6 then
    local fade = 1.0 - rise
    energy = saturate((0.35 + manual_energy * 0.35) * fade)
    chaos = saturate((0.22 + manual_chaos * 0.25) * fade)
  end

  local caustics = saturate((0.40 + energy * 0.45 + slow_tide * 0.12) * master)
  local sonar = saturate((phase >= 1 and (0.18 + energy * 0.72 + beat * 0.20) or 0.04) * master)
  local chroma = saturate((0.05 + chaos * 0.78 + beat * 0.12) * master)
  local tear = saturate((phase >= 2 and chaos * 0.92 or chaos * 0.24) * master)
  local feedback_mix = saturate((phase >= 3 and (0.18 + energy * 0.55) or 0.05 + energy * 0.12) * master)
  local scanlines = saturate((phase >= 4 and (0.42 + energy * 0.46) or 0.10 + energy * 0.16) * master)
  local spiral = saturate((phase >= 3 and (0.12 + energy * 0.70) or energy * 0.08) * master)
  local vignette = saturate((0.24 + energy * 0.50 + breathe * 0.12) * master)

  ctx:set_uniform("u_time", time)
  ctx:set_uniform("u_slot_time", slot_time)
  ctx:set_uniform("u_resolution", { 1280.0, 720.0 })
  ctx:set_uniform("u_master", master)
  ctx:set_uniform("u_phase", phase)
  ctx:set_uniform("u_section_progress", progress)
  ctx:set_uniform("u_energy", energy)
  ctx:set_uniform("u_chaos", chaos)
  ctx:set_uniform("u_beat", beat)
  ctx:set_uniform("u_caustics", caustics)
  ctx:set_uniform("u_sonar", sonar)
  ctx:set_uniform("u_chroma", chroma)
  ctx:set_uniform("u_tear", tear)
  ctx:set_uniform("u_feedback_mix", feedback_mix)
  ctx:set_uniform("u_scanlines", scanlines)
  ctx:set_uniform("u_spiral", spiral)
  ctx:set_uniform("u_vignette", vignette)
  ctx:set_uniform("u_bubble_mix", bubble_mix)
  ctx:set_uniform("u_bubble_refraction_mix", bubble_refraction_mix)
  ctx:set_uniform("u_bubble_highlight_mix", bubble_highlight_mix)
end

return M
