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
  uBlockSize = 28,
  uDisplace = 1.15,
  uFreezeMix = 0.32,
  uCorrupt = 0.28,
  uColorBleed = 0.50,
  uQuantize = 0,
  uUseInput = 1.0,
  uAudioDrive = 0.40,
}

return {
  params = {
    uBlockSize = { value = defaults.uBlockSize, min = 4, max = 160, step = 1 },
    uDisplace = { value = defaults.uDisplace, min = 0.0, max = 4.0, step = 0.001 },
    uFreezeMix = { value = defaults.uFreezeMix, min = 0.0, max = 0.98, step = 0.001 },
    uCorrupt = { value = defaults.uCorrupt, min = 0.0, max = 1.0, step = 0.001 },
    uColorBleed = { value = defaults.uColorBleed, min = 0.0, max = 2.0, step = 0.001 },
    uQuantize = { value = defaults.uQuantize, min = 0, max = 32, step = 1 },
    uUseInput = { value = defaults.uUseInput, min = 0.0, max = 1.0, step = 1.0 },
    uAudioDrive = { value = defaults.uAudioDrive, min = 0.0, max = 1.0, step = 0.001 },
  },
  per_frame = function(state)
    local P = read(state, "params", {})
    local audio = clamp(read(state, "audio_rms", 0.0), 0.0, 1.0)
    local function get(name) return read(P, name, defaults[name]) end
    local hit = audio * clamp(get("uAudioDrive"), 0.0, 1.0)
    return {
      uBlockSize = clamp(math.floor(get("uBlockSize") + 0.5), 4, 160),
      uDisplace = clamp(get("uDisplace") * (1.0 + hit * 1.2), 0.0, 4.0),
      uFreezeMix = clamp(get("uFreezeMix"), 0.0, 0.98),
      uCorrupt = clamp(get("uCorrupt") + hit * 0.35, 0.0, 1.0),
      uColorBleed = clamp(get("uColorBleed") + hit * 0.5, 0.0, 2.0),
      uQuantize = clamp(get("uQuantize"), 0, 32),
      uUseInput = get("uUseInput") > 0.5 and 1.0 or 0.0,
      uAudioDrive = clamp(get("uAudioDrive"), 0.0, 1.0),
    }
  end
}
