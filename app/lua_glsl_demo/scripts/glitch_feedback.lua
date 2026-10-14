local function clamp(v, lo, hi)
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

local function saturate(v)
  return clamp(v, 0.0, 1.0)
end

local function read(tbl, key, fallback)
  if tbl == nil then return fallback end
  local ok, value = pcall(function() return tbl[key] end)
  if not ok or value == nil then return fallback end
  return value
end

local defaults = {
  uUseInput = 1.0,
  uFeedbackMix = 0.18,
  uFeedbackZoom = 0.65,
  uWarp = 0.38,
  uGlitch = 0.28,
  uScanline = 0.55,
  uChromatic = 0.55,
  uPosterize = 0.0,
  uColorDrift = 0.06,
  uBrightness = 1.05,
  uContrast = 1.12,
  uSaturation = 1.24,
  uInvert = 0.0,
  uAudioDrive = 0.35,
}

return {
  params = {
    uUseInput = { value = defaults.uUseInput, min = 0.0, max = 1.0, step = 1.0 },
    uFeedbackMix = { value = defaults.uFeedbackMix, min = 0.0, max = 0.98, step = 0.001 },
    uFeedbackZoom = { value = defaults.uFeedbackZoom, min = -2.0, max = 2.0, step = 0.001 },
    uWarp = { value = defaults.uWarp, min = 0.0, max = 2.0, step = 0.001 },
    uGlitch = { value = defaults.uGlitch, min = 0.0, max = 1.0, step = 0.001 },
    uScanline = { value = defaults.uScanline, min = 0.0, max = 1.0, step = 0.001 },
    uChromatic = { value = defaults.uChromatic, min = 0.0, max = 2.0, step = 0.001 },
    uPosterize = { value = defaults.uPosterize, min = 0.0, max = 32.0, step = 1.0 },
    uColorDrift = { value = defaults.uColorDrift, min = -1.0, max = 1.0, step = 0.001 },
    uBrightness = { value = defaults.uBrightness, min = 0.0, max = 3.0, step = 0.001 },
    uContrast = { value = defaults.uContrast, min = 0.0, max = 3.0, step = 0.001 },
    uSaturation = { value = defaults.uSaturation, min = 0.0, max = 3.0, step = 0.001 },
    uInvert = { value = defaults.uInvert, min = 0.0, max = 1.0, step = 1.0 },
    uAudioDrive = { value = defaults.uAudioDrive, min = 0.0, max = 1.0, step = 0.001 },
  },

  per_frame = function(state, api)
    local P = read(state, "params", {})
    local time = read(state, "time", 0.0)
    local audio = saturate(read(state, "audio_rms", read(state, "rms", 0.0)))

    local function get(name)
      local v = read(P, name, nil)
      if v == nil then v = defaults[name] end
      return v
    end

    local drive = saturate(get("uAudioDrive"))
    local hit = audio * drive

    local U = {
      uUseInput = get("uUseInput") > 0.5 and 1.0 or 0.0,
      uFeedbackMix = clamp(get("uFeedbackMix") * (1.0 - hit * 0.35), 0.0, 0.98),
      uFeedbackZoom = clamp(get("uFeedbackZoom") + hit * 0.35, -2.0, 2.0),
      uWarp = clamp(get("uWarp") * (1.0 + hit * 0.75), 0.0, 2.0),
      uGlitch = clamp(get("uGlitch") + hit * 0.35, 0.0, 1.0),
      uScanline = saturate(get("uScanline")),
      uChromatic = clamp(get("uChromatic") * (1.0 + hit * 0.65), 0.0, 2.0),
      uPosterize = clamp(get("uPosterize"), 0.0, 32.0),
      uColorDrift = clamp(get("uColorDrift") + hit * 0.08, -1.0, 1.0),
      uBrightness = clamp(get("uBrightness") + hit * 0.12, 0.0, 3.0),
      uContrast = clamp(get("uContrast") + hit * 0.20, 0.0, 3.0),
      uSaturation = clamp(get("uSaturation"), 0.0, 3.0),
      uInvert = get("uInvert") > 0.5 and 1.0 or 0.0,
      uAudioDrive = drive,
    }

    local ext = read(state, "ext", nil)
    if ext then
      for k, v in pairs(ext) do
        U[k] = v
      end
    end

    return U
  end
}
