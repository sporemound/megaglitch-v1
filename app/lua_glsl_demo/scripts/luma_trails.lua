local function clamp(v, lo, hi)
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

local function read(tbl, key, fallback)
  if tbl == nil then return fallback end
  local ok, value = pcall(function() return tbl[key] end)
  if not ok or value == nil then return fallback end
  return value
end

local defaults = {
  uThreshold = 0.46,
  uSoftness = 0.13,
  uTrailMix = 0.72,
  uTrailDecay = 0.93,
  uEdgeGlow = 0.75,
  uWarp = 0.65,
  uTintHue = 0.44,
  uUseInput = 1.0,
  uAudioDrive = 0.30,
}

return {
  params = {
    uThreshold = { value = defaults.uThreshold, min = 0.0, max = 1.0, step = 0.001 },
    uSoftness = { value = defaults.uSoftness, min = 0.001, max = 0.5, step = 0.001 },
    uTrailMix = { value = defaults.uTrailMix, min = 0.0, max = 0.98, step = 0.001 },
    uTrailDecay = { value = defaults.uTrailDecay, min = 0.0, max = 1.0, step = 0.001 },
    uEdgeGlow = { value = defaults.uEdgeGlow, min = 0.0, max = 2.0, step = 0.001 },
    uWarp = { value = defaults.uWarp, min = 0.0, max = 2.0, step = 0.001 },
    uTintHue = { value = defaults.uTintHue, min = 0.0, max = 1.0, step = 0.001 },
    uUseInput = { value = defaults.uUseInput, min = 0.0, max = 1.0, step = 1.0 },
    uAudioDrive = { value = defaults.uAudioDrive, min = 0.0, max = 1.0, step = 0.001 },
  },
  per_frame = function(state)
    local P = read(state, "params", {})
    local audio = clamp(read(state, "audio_rms", 0.0), 0.0, 1.0)
    local function get(name) return read(P, name, defaults[name]) end
    local hit = audio * clamp(get("uAudioDrive"), 0.0, 1.0)
    return {
      uThreshold = clamp(get("uThreshold") - hit * 0.10, 0.0, 1.0),
      uSoftness = clamp(get("uSoftness") + hit * 0.08, 0.001, 0.5),
      uTrailMix = clamp(get("uTrailMix"), 0.0, 0.98),
      uTrailDecay = clamp(get("uTrailDecay"), 0.0, 1.0),
      uEdgeGlow = clamp(get("uEdgeGlow") + hit * 0.7, 0.0, 2.0),
      uWarp = clamp(get("uWarp") + hit * 0.5, 0.0, 2.0),
      uTintHue = get("uTintHue"),
      uUseInput = get("uUseInput") > 0.5 and 1.0 or 0.0,
      uAudioDrive = clamp(get("uAudioDrive"), 0.0, 1.0),
    }
  end
}
