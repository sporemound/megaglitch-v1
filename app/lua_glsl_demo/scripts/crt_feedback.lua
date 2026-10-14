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
  uFeedbackMix = 0.22,
  uCurvature = 0.55,
  uScanline = 0.65,
  uMask = 0.55,
  uGlow = 0.55,
  uJitter = 0.18,
  uChromatic = 0.42,
  uNoise = 0.16,
  uSaturation = 1.25,
  uUseInput = 1.0,
  uAudioDrive = 0.35,
}

return {
  params = {
    uFeedbackMix = { value = defaults.uFeedbackMix, min = 0.0, max = 0.97, step = 0.001 },
    uCurvature = { value = defaults.uCurvature, min = 0.0, max = 1.5, step = 0.001 },
    uScanline = { value = defaults.uScanline, min = 0.0, max = 1.0, step = 0.001 },
    uMask = { value = defaults.uMask, min = 0.0, max = 1.0, step = 0.001 },
    uGlow = { value = defaults.uGlow, min = 0.0, max = 2.0, step = 0.001 },
    uJitter = { value = defaults.uJitter, min = 0.0, max = 1.0, step = 0.001 },
    uChromatic = { value = defaults.uChromatic, min = 0.0, max = 2.0, step = 0.001 },
    uNoise = { value = defaults.uNoise, min = 0.0, max = 1.0, step = 0.001 },
    uSaturation = { value = defaults.uSaturation, min = 0.0, max = 3.0, step = 0.001 },
    uUseInput = { value = defaults.uUseInput, min = 0.0, max = 1.0, step = 1.0 },
    uAudioDrive = { value = defaults.uAudioDrive, min = 0.0, max = 1.0, step = 0.001 },
  },
  per_frame = function(state)
    local P = read(state, "params", {})
    local t = read(state, "time", 0.0)
    local audio = clamp(read(state, "audio_rms", 0.0), 0.0, 1.0)
    local function get(name) return read(P, name, defaults[name]) end
    local drive = clamp(get("uAudioDrive"), 0.0, 1.0)
    local pulse = audio * drive
    return {
      uFeedbackMix = clamp(get("uFeedbackMix") * (1.0 - pulse * 0.25), 0.0, 0.97),
      uCurvature = clamp(get("uCurvature"), 0.0, 1.5),
      uScanline = clamp(get("uScanline"), 0.0, 1.0),
      uMask = clamp(get("uMask"), 0.0, 1.0),
      uGlow = clamp(get("uGlow") + pulse * 0.8, 0.0, 2.0),
      uJitter = clamp(get("uJitter") * (0.8 + 0.2 * math.sin(t * 2.7)) + pulse * 0.4, 0.0, 1.0),
      uChromatic = clamp(get("uChromatic") + pulse * 0.5, 0.0, 2.0),
      uNoise = clamp(get("uNoise") + pulse * 0.25, 0.0, 1.0),
      uSaturation = clamp(get("uSaturation"), 0.0, 3.0),
      uUseInput = get("uUseInput") > 0.5 and 1.0 or 0.0,
      uAudioDrive = drive,
    }
  end
}
