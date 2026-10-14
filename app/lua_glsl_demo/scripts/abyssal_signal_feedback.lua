local SLOT_LENGTH = 3600.0

local defaults = {
  master = 0.8,
  phase_override = -1,
  timeline_speed = 1.0,
  bpm = 92.0,
  beat_gain = 1.0,
  manual_energy = 0.0,
  manual_chaos = 0.0,
  caustics_gain = 0.55,
  caustics_scale = 1.0,
  caustics_speed = 1.0,
  sonar_gain = 1.25,
  sonar_speed = 1.0,
  sonar_width = 0.03,
  sonar_beam = 1.0,
  chroma_gain = 0.45,
  chroma_pixels = 14.0,
  tear_gain = 0.55,
  tear_density = 1.0,
  tear_speed = 1.0,
  tear_displace = 0.75,
  feedback_gain = 0.65,
  feedback_decay = 0.93,
  feedback_zoom = 0.035,
  spiral_gain = 1.45,
  spiral_twist = 1.0,
  spiral_radius = 1.0,
  scanline_gain = 0.35,
  grille_gain = 0.65,
  vignette_gain = 0.55,
  deep_mix = 0.55,
  teal_gain = 0.75,
  amber_gain = 0.85,
  brightness = 1.0,
  contrast = 1.0,
  grain_gain = 0.45,
  bloom_gain = 0.65
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

local function read(tbl, key, fallback)
  if tbl == nil then return fallback end
  local ok, value = pcall(function() return tbl[key] end)
  if not ok or value == nil then return fallback end
  return value
end

local function get_param(params, name)
  return read(params, name, defaults[name])
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
  return saturate((slot_time - phase_starts[index]) / (phase_ends[index] - phase_starts[index]))
end

return {
  params = {
    master = { value = defaults.master, min = 0.0, max = 1.0, step = 0.001 },
    phase_override = { value = defaults.phase_override, min = -1, max = 6, step = 1 },
    timeline_speed = { value = defaults.timeline_speed, min = 0.0, max = 4.0, step = 0.001 },
    bpm = { value = defaults.bpm, min = 40.0, max = 180.0, step = 0.1 },
    beat_gain = { value = defaults.beat_gain, min = 0.0, max = 2.0, step = 0.001 },
    manual_energy = { value = defaults.manual_energy, min = 0.0, max = 1.0, step = 0.001 },
    manual_chaos = { value = defaults.manual_chaos, min = 0.0, max = 1.0, step = 0.001 }
  },

  per_frame = function(state, api)
    local P = read(state, "params", {})
    local time = read(state, "time", 0.0)
    local timeline_speed = clamp(get_param(P, "timeline_speed"), 0.0, 4.0)
    local slot_time = (time * timeline_speed) % SLOT_LENGTH

    local master = saturate(get_param(P, "master"))
    local bpm = clamp(get_param(P, "bpm"), 40.0, 180.0)
    local beat_gain = clamp(get_param(P, "beat_gain"), 0.0, 2.0)
    local manual_energy = saturate(get_param(P, "manual_energy"))
    local manual_chaos = saturate(get_param(P, "manual_chaos"))
    local override = math.floor(get_param(P, "phase_override") + 0.5)

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
    beat = saturate((beat + 0.35 * math.exp(-(((beat_phase - 0.5) * 12.0) * ((beat_phase - 0.5) * 12.0)))) * beat_gain)

    local energy = saturate(phase_energy[phase_index] * (0.72 + 0.28 * rise) + manual_energy + beat * 0.10)
    local chaos = saturate(phase_chaos[phase_index] * (0.65 + 0.35 * rise) + manual_chaos + beat * 0.08)

    if phase == 6 then
      local fade = 1.0 - rise
      energy = saturate((0.35 + manual_energy * 0.35) * fade)
      chaos = saturate((0.22 + manual_chaos * 0.25) * fade)
    end

    local auto_caustics = 0.40 + energy * 0.45 + slow_tide * 0.12
    local auto_sonar = phase >= 1 and (0.18 + energy * 0.72 + beat * 0.20) or 0.04
    local auto_chroma = 0.05 + chaos * 0.78 + beat * 0.12
    local auto_tear = phase >= 2 and chaos * 0.92 or chaos * 0.24
    local auto_feedback = phase >= 3 and (0.18 + energy * 0.55) or 0.05 + energy * 0.12
    local auto_scanlines = phase >= 4 and (0.42 + energy * 0.46) or 0.10 + energy * 0.16
    local auto_spiral = phase >= 3 and (0.12 + energy * 0.70) or energy * 0.08
    local auto_vignette = 0.24 + energy * 0.50 + breathe * 0.12

    return {
      u_slot_time = slot_time,
      u_phase = phase,
      u_section_progress = progress,
      u_master = master,
      u_energy = energy,
      u_chaos = chaos,
      u_beat = beat,
      u_caustics = saturate(auto_caustics * get_param(P, "caustics_gain") * master),
      u_sonar = saturate(auto_sonar * get_param(P, "sonar_gain") * master),
      u_chroma = saturate(auto_chroma * get_param(P, "chroma_gain") * master),
      u_tear = saturate(auto_tear * get_param(P, "tear_gain") * master),
      u_feedback_mix = saturate(auto_feedback * get_param(P, "feedback_gain") * master),
      u_scanlines = saturate(auto_scanlines * get_param(P, "scanline_gain") * master),
      u_spiral = saturate(auto_spiral * get_param(P, "spiral_gain") * master),
      u_vignette = saturate(auto_vignette * get_param(P, "vignette_gain") * master),
      u_caustics_scale = clamp(get_param(P, "caustics_scale"), 0.25, 4.0),
      u_caustics_speed = clamp(get_param(P, "caustics_speed"), 0.0, 3.0),
      u_sonar_speed = clamp(get_param(P, "sonar_speed"), 0.0, 3.0),
      u_sonar_width = clamp(get_param(P, "sonar_width"), 0.004, 0.12),
      u_sonar_beam = clamp(get_param(P, "sonar_beam"), 0.0, 2.0),
      u_chroma_pixels = clamp(get_param(P, "chroma_pixels"), 0.0, 64.0),
      u_tear_density = clamp(get_param(P, "tear_density"), 0.25, 3.0),
      u_tear_speed = clamp(get_param(P, "tear_speed"), 0.0, 4.0),
      u_tear_displace = clamp(get_param(P, "tear_displace"), 0.0, 2.0),
      u_feedback_decay = clamp(get_param(P, "feedback_decay"), 0.75, 1.02),
      u_feedback_zoom = clamp(get_param(P, "feedback_zoom"), -0.15, 0.15),
      u_spiral_twist = clamp(get_param(P, "spiral_twist"), -2.0, 3.0),
      u_spiral_radius = clamp(get_param(P, "spiral_radius"), 0.2, 2.0),
      u_grille_gain = clamp(get_param(P, "grille_gain"), 0.0, 2.0),
      u_deep_mix = saturate(get_param(P, "deep_mix")),
      u_teal_gain = clamp(get_param(P, "teal_gain"), 0.0, 2.0),
      u_amber_gain = clamp(get_param(P, "amber_gain"), 0.0, 2.0),
      u_brightness = clamp(get_param(P, "brightness"), 0.0, 2.5),
      u_contrast = clamp(get_param(P, "contrast"), 0.0, 2.5),
      u_grain_gain = clamp(get_param(P, "grain_gain"), 0.0, 2.0),
      u_bloom_gain = clamp(get_param(P, "bloom_gain"), 0.0, 2.0)
    }
  end
}
