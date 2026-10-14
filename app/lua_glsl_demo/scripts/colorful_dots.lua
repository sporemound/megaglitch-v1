local function clamp(v, min_v, max_v)
    if v < min_v then return min_v end
    if v > max_v then return max_v end
    return v
end

local function saturate(v)
    return clamp(v, 0.0, 1.0)
end

local defaults = {
    u_cluster_count = 10,
    u_dot_count = 200,
    u_cluster_speed = 0.2,
    u_movement_range = 0.3,
    u_cluster_ttl_min = 1.0,
    u_cluster_ttl_max = 3.0,
    u_cluster_start_scatter_min = 0.2,
    u_cluster_start_scatter_max = 1.0,
    u_cluster_end_scatter_min = 0.05,
    u_cluster_end_scatter_max = 0.5,
    u_dot_size_min = 0.002,
    u_dot_size_max = 0.002,
    u_cluster_center_acceleration = 0.05,
    u_base_color_r = 0.0,
    u_base_color_g = 1.0,
    u_base_color_b = 1.0,
    u_cluster_color_offset = 0.3,
    u_dot_color_offset = 0.3,
    u_audio_drive = 0.2,
}

return {
    params = defaults,

    per_frame = function(state, api)
        local P = state.params or {}
        local time = state.time or 0.0
        local audio = saturate(state.audio_rms or 0.0)

        local function get(name)
            local v = P[name]
            if v == nil then
                v = defaults[name]
            end
            return v
        end

        local ttl_min = clamp(get('u_cluster_ttl_min'), 0.1, 10.0)
        local ttl_max = clamp(get('u_cluster_ttl_max'), ttl_min + 0.01, 15.0)
        local start_min = clamp(get('u_cluster_start_scatter_min'), 0.0, 2.0)
        local start_max = clamp(get('u_cluster_start_scatter_max'), start_min, 2.5)
        local end_min = clamp(get('u_cluster_end_scatter_min'), 0.0, 2.0)
        local end_max = clamp(get('u_cluster_end_scatter_max'), end_min, 2.5)
        local dot_size_min = clamp(get('u_dot_size_min'), 0.0005, 0.05)
        local dot_size_max = clamp(get('u_dot_size_max'), dot_size_min, 0.05)

        local drive = saturate(get('u_audio_drive'))
        local lfo = 0.5 + 0.5 * math.sin(time * 0.45)

        local uniforms = {
            u_cluster_count = clamp(math.floor(get('u_cluster_count') + 0.5), 1, 32),
            u_dot_count = clamp(math.floor(get('u_dot_count') + 0.5), 1, 512),
            u_cluster_speed = clamp(get('u_cluster_speed'), 0.0, 5.0) * (1.0 + audio * drive * 0.75),
            u_movement_range = clamp(get('u_movement_range'), 0.0, 2.0) * (0.85 + 0.3 * lfo),
            u_cluster_ttl_min = ttl_min,
            u_cluster_ttl_max = ttl_max,
            u_cluster_start_scatter_min = start_min,
            u_cluster_start_scatter_max = start_max,
            u_cluster_end_scatter_min = end_min,
            u_cluster_end_scatter_max = end_max,
            u_dot_size_min = dot_size_min,
            u_dot_size_max = dot_size_max,
            u_cluster_center_acceleration = clamp(get('u_cluster_center_acceleration'), 0.0, 1.0),
            u_base_color_r = saturate(get('u_base_color_r')),
            u_base_color_g = saturate(get('u_base_color_g')),
            u_base_color_b = saturate(get('u_base_color_b')),
            u_cluster_color_offset = clamp(get('u_cluster_color_offset'), 0.0, 1.0),
            u_dot_color_offset = clamp(get('u_dot_color_offset'), 0.0, 1.0) * (0.7 + 0.3 * (audio * drive + lfo * (1.0 - drive))),
        }

        uniforms.u_audio_drive = drive

        if state.ext then
            for k, v in pairs(state.ext) do
                uniforms[k] = v
            end
        end

        return uniforms
    end
}
