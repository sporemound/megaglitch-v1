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
  uInputMix = 0.15,
  uFeedbackMix = 0.08,
  uScale = 5.5,
  uSpeed = 0.75,
  uRings = 0.55,
  uSymmetry = 6,
  uScopeGain = 1.0,
  uHue = 0.58,
  uContrast = 1.2,
  uAudioDrive = 0.30,
}

return {
  params = {
    uInputMix = { value = defaults.uInputMix, min = 0.0, max = 1.0, step = 0.001 },
    uFeedbackMix = { value = defaults.uFeedbackMix, min = 0.0, max = 0.98, step = 0.001 },
    uScale = { value = defaults.uScale, min = 1.0, max = 24.0, step = 0.001 },
    uSpeed = { value = defaults.uSpeed, min = -4.0, max = 4.0, step = 0.001 },
    uRings = { value = defaults.uRings, min = 0.0, max = 2.0, step = 0.001 },
    uSymmetry = { value = defaults.uSymmetry, min = 1, max = 16, step = 1 },
    uScopeGain = { value = defaults.uScopeGain, min = 0.2, max = 4.0, step = 0.001 },
    uHue = { value = defaults.uHue, min = 0.0, max = 1.0, step = 0.001 },
    uContrast = { value = defaults.uContrast, min = 0.0, max = 3.0, step = 0.001 },
    uAudioDrive = { value = defaults.uAudioDrive, min = 0.0, max = 1.0, step = 0.001 },
  },
  per_frame = function(state)
    local P = read(state, "params", {})
    local t = read(state, "time", 0.0)
    local audio = clamp(read(state, "audio_rms", 0.0), 0.0, 1.0)
    local function get(name) return read(P, name, defaults[name]) end
    local drive = clamp(get("uAudioDrive"), 0.0, 1.0)
    local hit = audio * drive
    return {
      uInputMix = clamp(get("uInputMix"), 0.0, 1.0),
      uFeedbackMix = clamp(get("uFeedbackMix") + hit * 0.08, 0.0, 0.98),
      uScale = clamp(get("uScale") * (1.0 + hit * 0.25), 1.0, 24.0),
      uSpeed = clamp(get("uSpeed"), -4.0, 4.0),
      uRings = clamp(get("uRings") + math.sin(t * 0.23) * 0.08, 0.0, 2.0),
      uSymmetry = clamp(math.floor(get("uSymmetry") + 0.5), 1, 16),
      uScopeGain = clamp(get("uScopeGain") + hit * 1.2, 0.2, 4.0),
      uHue = get("uHue"),
      uContrast = clamp(get("uContrast"), 0.0, 3.0),
      uAudioDrive = drive,
    }
  end
}
