local function clamp(v, lo, hi)
  if v < lo then return lo end
  if v > hi then return hi end
  return v
end

local function smoothstep(edge0, edge1, x)
  local t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)
end

local function read(tbl, key, fallback)
  if tbl == nil then return fallback end
  local ok, value = pcall(function() return tbl[key] end)
  if not ok or value == nil then return fallback end
  return value
end

local defaults = {
  uFeedbackMix = 0.25,
  uAberration = 0.003,
  uWarp = 0.08,
  uNoise = 0.10,
  uExposure = 1.00,
  uContrast = 1.00,
  uSaturation = 1.00,
  uHue = 0.00,
  uYFlip = 0.00,
}

return {
  params = {
    uFeedbackMix = { value = defaults.uFeedbackMix, min = 0.0, max = 1.0, step = 0.001 },
    uAberration = { value = defaults.uAberration, min = 0.0, max = 0.03, step = 0.0001 },
    uWarp = { value = defaults.uWarp, min = 0.0, max = 0.50, step = 0.0005 },
    uNoise = { value = defaults.uNoise, min = 0.0, max = 1.00, step = 0.001 },
    uExposure = { value = defaults.uExposure, min = 0.0, max = 3.00, step = 0.01 },
    uContrast = { value = defaults.uContrast, min = 0.0, max = 2.00, step = 0.01 },
    uSaturation = { value = defaults.uSaturation, min = 0.0, max = 2.00, step = 0.01 },
    uHue = { value = defaults.uHue, min = -math.pi, max = math.pi, step = 0.001 },
    uYFlip = { value = defaults.uYFlip, min = 0.0, max = 1.0, step = 1.0 },
  },

  per_frame = function(state)
    local t = read(state, "time", 0.0)
    local rms = read(state, "audio_rms", 0.0)
    local P = read(state, "params", {})

    local function get(name)
      return read(P, name, defaults[name])
    end

    local lfo_slow = 0.5 + 0.5 * math.sin(t * 0.63)
    local lfo_fast = 0.5 + 0.5 * math.sin(t * 2.41 + 1.1)
    local audio_env = smoothstep(0.02, 0.20, rms)

    local U = {
      uFeedbackMix = clamp(get("uFeedbackMix") * (1.0 - 0.25 * audio_env), 0.0, 1.0),
      uAberration = get("uAberration") * (1.0 + 0.7 * audio_env),
      uWarp = get("uWarp") * (0.85 + 0.30 * lfo_slow),
      uNoise = clamp(get("uNoise") * (0.75 + 0.50 * lfo_fast) + 0.10 * audio_env, 0.0, 1.0),
      uExposure = get("uExposure"),
      uContrast = get("uContrast"),
      uSaturation = get("uSaturation"),
      uHue = get("uHue"),
      uYFlip = get("uYFlip") > 0.5 and 1.0 or 0.0,
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
