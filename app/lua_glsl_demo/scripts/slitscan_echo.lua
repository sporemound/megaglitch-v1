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
  uFeedbackMix = 0.30,
  uSliceCount = 72,
  uSliceDrift = 0.20,
  uOffset = 0.45,
  uSmear = 0.22,
  uColorSplit = 0.55,
  uStrobe = 0.0,
  uUseInput = 1.0,
  uAudioDrive = 0.35,
}

return {
  params = {
    uFeedbackMix = { value = defaults.uFeedbackMix, min = 0.0, max = 0.98, step = 0.001 },
    uSliceCount = { value = defaults.uSliceCount, min = 2, max = 240, step = 1 },
    uSliceDrift = { value = defaults.uSliceDrift, min = -2.0, max = 2.0, step = 0.001 },
    uOffset = { value = defaults.uOffset, min = 0.0, max = 2.0, step = 0.001 },
    uSmear = { value = defaults.uSmear, min = 0.0, max = 1.0, step = 0.001 },
    uColorSplit = { value = defaults.uColorSplit, min = 0.0, max = 2.0, step = 0.001 },
    uStrobe = { value = defaults.uStrobe, min = 0.0, max = 1.0, step = 0.001 },
    uUseInput = { value = defaults.uUseInput, min = 0.0, max = 1.0, step = 1.0 },
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
      uFeedbackMix = clamp(get("uFeedbackMix"), 0.0, 0.98),
      uSliceCount = clamp(math.floor(get("uSliceCount") + 0.5), 2, 240),
      uSliceDrift = clamp(get("uSliceDrift") + math.sin(t * 0.19) * 0.08, -2.0, 2.0),
      uOffset = clamp(get("uOffset") * (0.85 + hit * 1.2), 0.0, 2.0),
      uSmear = clamp(get("uSmear") + hit * 0.35, 0.0, 1.0),
      uColorSplit = clamp(get("uColorSplit") + hit * 0.45, 0.0, 2.0),
      uStrobe = clamp(get("uStrobe"), 0.0, 1.0),
      uUseInput = get("uUseInput") > 0.5 and 1.0 or 0.0,
      uAudioDrive = drive,
    }
  end
}
