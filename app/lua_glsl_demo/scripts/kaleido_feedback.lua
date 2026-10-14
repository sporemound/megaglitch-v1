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
  uSegments = 8,
  uRotation = 0.0,
  uZoom = 1.15,
  uFeedbackMix = 0.18,
  uBloom = 0.45,
  uColorCycle = 0.58,
  uChromatic = 0.35,
  uUseInput = 1.0,
  uExternalFeedbackMix = 0.35,
  uExternalFeedbackZoom = 0.04,
  uExternalFeedbackRotate = 0.0,
  uExternalFeedbackWarp = 0.25,
  uFeedbackDecay = 0.96,
  uFeedbackSpin = 0.0,
  uMirrorOffset = 0.0,
  uCenterX = 0.5,
  uCenterY = 0.5,
  uSourceMix = 0.0,
  uFeedbackColorize = 0.0,
  uAudioDrive = 0.35,
}

return {
  params = {
    uSegments = { value = defaults.uSegments, min = 1, max = 24, step = 1 },
    uRotation = { value = defaults.uRotation, min = -3.14159, max = 3.14159, step = 0.001 },
    uZoom = { value = defaults.uZoom, min = 0.2, max = 4.0, step = 0.001 },
    uFeedbackMix = { value = defaults.uFeedbackMix, min = 0.0, max = 0.98, step = 0.001 },
    uBloom = { value = defaults.uBloom, min = 0.0, max = 2.0, step = 0.001 },
    uColorCycle = { value = defaults.uColorCycle, min = 0.0, max = 1.0, step = 0.001 },
    uChromatic = { value = defaults.uChromatic, min = 0.0, max = 2.0, step = 0.001 },
    uUseInput = { value = defaults.uUseInput, min = 0.0, max = 1.0, step = 1.0 },
    uExternalFeedbackMix = { value = defaults.uExternalFeedbackMix, min = 0.0, max = 1.0, step = 0.001 },
    uExternalFeedbackZoom = { value = defaults.uExternalFeedbackZoom, min = -0.5, max = 0.5, step = 0.001 },
    uExternalFeedbackRotate = { value = defaults.uExternalFeedbackRotate, min = -3.14159, max = 3.14159, step = 0.001 },
    uExternalFeedbackWarp = { value = defaults.uExternalFeedbackWarp, min = 0.0, max = 2.0, step = 0.001 },
    uFeedbackDecay = { value = defaults.uFeedbackDecay, min = 0.75, max = 1.02, step = 0.001 },
    uFeedbackSpin = { value = defaults.uFeedbackSpin, min = -4.0, max = 4.0, step = 0.001 },
    uMirrorOffset = { value = defaults.uMirrorOffset, min = -0.5, max = 0.5, step = 0.001 },
    uCenterX = { value = defaults.uCenterX, min = 0.0, max = 1.0, step = 0.001 },
    uCenterY = { value = defaults.uCenterY, min = 0.0, max = 1.0, step = 0.001 },
    uSourceMix = { value = defaults.uSourceMix, min = 0.0, max = 1.0, step = 0.001 },
    uFeedbackColorize = { value = defaults.uFeedbackColorize, min = 0.0, max = 1.0, step = 0.001 },
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
      uSegments = clamp(math.floor(get("uSegments") + 0.5), 1, 24),
      uRotation = clamp(get("uRotation") + math.sin(t * 0.23) * 0.25 + hit * 0.4, -3.14159, 3.14159),
      uZoom = clamp(get("uZoom") * (1.0 + hit * 0.18), 0.2, 4.0),
      uFeedbackMix = clamp(get("uFeedbackMix"), 0.0, 0.98),
      uBloom = clamp(get("uBloom") + hit * 0.8, 0.0, 2.0),
      uColorCycle = get("uColorCycle"),
      uChromatic = clamp(get("uChromatic") + hit * 0.45, 0.0, 2.0),
      uUseInput = get("uUseInput") > 0.5 and 1.0 or 0.0,
      uExternalFeedbackMix = clamp(get("uExternalFeedbackMix") + hit * 0.12, 0.0, 1.0),
      uExternalFeedbackZoom = clamp(get("uExternalFeedbackZoom"), -0.5, 0.5),
      uExternalFeedbackRotate = clamp(get("uExternalFeedbackRotate"), -3.14159, 3.14159),
      uExternalFeedbackWarp = clamp(get("uExternalFeedbackWarp") + hit * 0.25, 0.0, 2.0),
      uFeedbackDecay = clamp(get("uFeedbackDecay"), 0.75, 1.02),
      uFeedbackSpin = clamp(get("uFeedbackSpin"), -4.0, 4.0),
      uMirrorOffset = clamp(get("uMirrorOffset"), -0.5, 0.5),
      uCenterX = clamp(get("uCenterX"), 0.0, 1.0),
      uCenterY = clamp(get("uCenterY"), 0.0, 1.0),
      uSourceMix = clamp(get("uSourceMix"), 0.0, 1.0),
      uFeedbackColorize = clamp(get("uFeedbackColorize") + hit * 0.15, 0.0, 1.0),
      uAudioDrive = drive,
    }
  end
}
