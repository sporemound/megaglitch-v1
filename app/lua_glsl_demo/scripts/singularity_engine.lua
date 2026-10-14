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
  pixel_size = 5.0,
  palette_levels = 14.0,
  dither_strength = 0.14,
  posterize_amount = 1.0,
  iso_slant = 0.45,
  iso_depth = 0.25,
  cascade_angle = -0.28,
  perspective_compression = 0.18,
  swell_strength = 0.045,
  chop_strength = 0.022,
  wave_speed = 0.85,
  cross_current = 0.035,
  turbulence_amount = 0.045,
  surge_amount = 0.18,
  fall_line_position = 0.52,
  fall_line_curve = 0.15,
  fall_strength = 0.24,
  cascade_speed = 0.75,
  cascade_stretch = 0.18,
  sheet_breakup = 0.34,
  foam_threshold = 0.34,
  foam_edge_strength = 1.0,
  foam_opacity = 0.72,
  foam_breakup = 0.38,
  foam_scroll_speed = 0.85,
  band_count = 4.0,
  band_thickness = 0.46,
  band_spacing = 8.0,
  band_speed_variation = 0.62,
  band_breakup = 0.28,
  band_source_offset = 0.075,
  undertow_strength = 0.08,
  undertow_noise = 0.55,
  edge_pull_amount = 0.07,
  current_direction = -0.22,
  feedback_wave_echo = 0.18,
  old_wave_opacity = 0.22,
  echo_decay = 0.72,
  cascade_feedback_strength = 0.24,
  debris_amount = 0.58,
  tile_dropout = 0.16,
  streak_length = 0.12,
  collapse_darkening = 0.55,
  collapse_pixel_size = 2.4,
}

local ranges = {
  pixel_size = {1.0, 12.0}, palette_levels = {4.0, 32.0}, dither_strength = {0.0, 0.35}, posterize_amount = {0.0, 1.0},
  iso_slant = {-1.0, 1.0}, iso_depth = {-0.6, 0.8}, cascade_angle = {-1.0, 1.0}, perspective_compression = {0.0, 0.6},
  swell_strength = {0.0, 0.12}, chop_strength = {0.0, 0.08}, wave_speed = {0.0, 3.0}, cross_current = {-0.15, 0.15}, turbulence_amount = {0.0, 0.14}, surge_amount = {0.0, 0.6},
  fall_line_position = {0.05, 0.95}, fall_line_curve = {0.0, 0.45}, fall_strength = {0.0, 0.65}, cascade_speed = {0.0, 2.5}, cascade_stretch = {0.0, 0.6}, sheet_breakup = {0.0, 1.0},
  foam_threshold = {0.0, 1.0}, foam_edge_strength = {0.0, 3.0}, foam_opacity = {0.0, 1.0}, foam_breakup = {0.0, 1.0}, foam_scroll_speed = {0.0, 3.0},
  band_count = {1.0, 6.0}, band_thickness = {0.1, 1.2}, band_spacing = {2.0, 24.0}, band_speed_variation = {0.0, 2.0}, band_breakup = {0.0, 1.0}, band_source_offset = {0.0, 0.24},
  undertow_strength = {0.0, 0.24}, undertow_noise = {0.0, 1.0}, edge_pull_amount = {0.0, 0.22}, current_direction = {-1.0, 1.0},
  feedback_wave_echo = {0.0, 0.6}, old_wave_opacity = {0.0, 0.8}, echo_decay = {0.0, 1.0}, cascade_feedback_strength = {0.0, 0.8},
  debris_amount = {0.0, 1.0}, tile_dropout = {0.0, 0.6}, streak_length = {0.0, 0.5}, collapse_darkening = {0.0, 1.0}, collapse_pixel_size = {1.0, 8.0},
}

return {
  params = (function()
    local out = {}
    for name, value in pairs(defaults) do
      local r = ranges[name] or {0.0, 1.0}
      out[name] = { value = value, min = r[1], max = r[2], step = (name == "pixel_size" or name == "palette_levels" or name == "band_count") and 1.0 or 0.001 }
    end
    return out
  end)(),

  per_frame = function(state)
    local P = read(state, "params", {})
    local out = {}
    for name, value in pairs(defaults) do
      local r = ranges[name] or {0.0, 1.0}
      out[name] = clamp(read(P, name, value), r[1], r[2])
    end
    return out
  end
}
