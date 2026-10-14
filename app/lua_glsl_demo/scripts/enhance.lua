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
  uZoom = 2.35,
  uPanX = 0.0,
  uPanY = 0.0,
  uSharpen = 0.22,
  uEdgeTrace = 0.28,
  uScanner = 0.42,
  uGrid = 0.18,
  uReadout = 0.58,
  uNoise = 0.08,
  uChromatic = 0.10,
  uFeedbackMix = 0.08,
  uAmber = 0.58,
  uCyan = 0.32,
  uContrast = 1.18,
  uExposure = 1.08,
  uFreezeBlend = 0.04,
  uPixelLock = 0.12,
  uVignette = 0.55,
  uAutoScan = 1.0,
}

return {
  params = {
    uZoom = { value = defaults.uZoom, min = 0.5, max = 8.0, step = 0.001 },
    uPanX = { value = defaults.uPanX, min = -0.5, max = 0.5, step = 0.001 },
    uPanY = { value = defaults.uPanY, min = -0.5, max = 0.5, step = 0.001 },
    uSharpen = { value = defaults.uSharpen, min = 0.0, max = 1.5, step = 0.001 },
    uEdgeTrace = { value = defaults.uEdgeTrace, min = 0.0, max = 1.5, step = 0.001 },
    uScanner = { value = defaults.uScanner, min = 0.0, max = 1.0, step = 0.001 },
    uGrid = { value = defaults.uGrid, min = 0.0, max = 1.0, step = 0.001 },
    uReadout = { value = defaults.uReadout, min = 0.0, max = 1.0, step = 0.001 },
    uNoise = { value = defaults.uNoise, min = 0.0, max = 1.0, step = 0.001 },
    uChromatic = { value = defaults.uChromatic, min = 0.0, max = 2.0, step = 0.001 },
    uFeedbackMix = { value = defaults.uFeedbackMix, min = 0.0, max = 0.98, step = 0.001 },
    uAmber = { value = defaults.uAmber, min = 0.0, max = 1.0, step = 0.001 },
    uCyan = { value = defaults.uCyan, min = 0.0, max = 1.0, step = 0.001 },
    uContrast = { value = defaults.uContrast, min = 0.0, max = 3.0, step = 0.001 },
    uExposure = { value = defaults.uExposure, min = 0.0, max = 3.0, step = 0.001 },
    uFreezeBlend = { value = defaults.uFreezeBlend, min = 0.0, max = 1.0, step = 0.001 },
    uPixelLock = { value = defaults.uPixelLock, min = 0.0, max = 1.0, step = 0.001 },
    uVignette = { value = defaults.uVignette, min = 0.0, max = 1.0, step = 0.001 },
    uAutoScan = { value = defaults.uAutoScan, min = 0.0, max = 1.0, step = 1.0 },
  },

  per_frame = function(state)
    local t = read(state, "time", 0.0)
    local P = read(state, "params", {})
    local function get(name) return read(P, name, defaults[name]) end

    local auto = get("uAutoScan") > 0.5 and 1.0 or 0.0
    local drift_x = math.sin(t * 0.11) * 0.026 * auto
    local drift_y = math.cos(t * 0.087 + 1.7) * 0.018 * auto
    local focus_pulse = 0.5 + 0.5 * math.sin(t * 0.31)

    return {
      uZoom = clamp(get("uZoom") * (1.0 + focus_pulse * 0.018 * auto), 0.5, 8.0),
      uPanX = clamp(get("uPanX") + drift_x, -0.5, 0.5),
      uPanY = clamp(get("uPanY") + drift_y, -0.5, 0.5),
      uSharpen = clamp(get("uSharpen"), 0.0, 1.5),
      uEdgeTrace = clamp(get("uEdgeTrace") * (0.88 + focus_pulse * 0.18), 0.0, 1.5),
      uScanner = clamp(get("uScanner"), 0.0, 1.0),
      uGrid = clamp(get("uGrid"), 0.0, 1.0),
      uReadout = clamp(get("uReadout"), 0.0, 1.0),
      uNoise = clamp(get("uNoise"), 0.0, 1.0),
      uChromatic = clamp(get("uChromatic"), 0.0, 2.0),
      uFeedbackMix = clamp(get("uFeedbackMix"), 0.0, 0.98),
      uAmber = clamp(get("uAmber"), 0.0, 1.0),
      uCyan = clamp(get("uCyan"), 0.0, 1.0),
      uContrast = clamp(get("uContrast"), 0.0, 3.0),
      uExposure = clamp(get("uExposure"), 0.0, 3.0),
      uFreezeBlend = clamp(get("uFreezeBlend"), 0.0, 1.0),
      uPixelLock = clamp(get("uPixelLock"), 0.0, 1.0),
      uVignette = clamp(get("uVignette"), 0.0, 1.0),
      uAutoScan = auto,
    }
  end
}
