#version 330 core

in vec2 v_uv;
layout(location = 0) out vec4 out_state;
layout(location = 1) out vec4 out_velocity;

uniform sampler2D u_state_prev;
uniform sampler2D u_velocity_prev;

uniform float u_time;
uniform float u_dt;
uniform int u_seed;
uniform float u_particle_texture_size;
uniform float u_buoyancy;
uniform float u_drag;
uniform float u_turbulence;
uniform float u_flow_speed;
uniform float u_motion_speed;
uniform float u_base_radius;
uniform float u_radius_variation;
uniform float u_depth_spread;
uniform float u_depth_motion;
uniform float u_orbit_strength;
uniform float u_show_energy;
uniform float u_show_chaos;
uniform float u_mutation_strength;
uniform float u_mutation_rate;
uniform float u_mutation_warp;
uniform float u_mutation_fracture;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash11(dot(i, vec2(127.1, 311.7)) + float(u_seed));
    float b = hash11(dot(i + vec2(1.0, 0.0), vec2(127.1, 311.7)) + float(u_seed));
    float c = hash11(dot(i + vec2(0.0, 1.0), vec2(127.1, 311.7)) + float(u_seed));
    float d = hash11(dot(i + vec2(1.0, 1.0), vec2(127.1, 311.7)) + float(u_seed));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

vec2 flow_field(vec2 p, float depth) {
    float t = u_time * u_flow_speed;
    float a = noise(p * 5.0 + vec2(t, depth * 3.1));
    float b = noise(p * 5.0 + vec2(-t * 0.7, 9.7 + depth * 2.3));
    return vec2(a - 0.5, b - 0.5);
}

void respawn(float id, float cycle, out vec4 state, out vec4 velocity) {
    float base = id + cycle * 97.0 + float(u_seed) * 13.0;
    float rx = hash11(base + 1.0);
    float rz = hash11(base + 2.0);
    float rr = hash11(base + 3.0);
    float rv = hash11(base + 4.0);

    float depth = mix(0.08, 1.0, rz) * max(u_depth_spread, 0.001);
    float radius = u_base_radius * mix(0.35, 1.0 + u_radius_variation, rr);
    state = vec4(rx, -0.08 - hash11(base + 5.0) * 0.25, clamp(depth, 0.0, 1.0), 0.0);
    velocity = vec4(
        (rv - 0.5) * 0.018,
        (hash11(base + 6.0) - 0.5) * 0.014,
        (hash11(base + 7.0) - 0.5) * 0.004,
        radius
    );
}

void main() {
    vec4 state = texture(u_state_prev, v_uv);
    vec4 velocity = texture(u_velocity_prev, v_uv);

    vec2 texel = floor(v_uv * u_particle_texture_size);
    float id = texel.x + texel.y * u_particle_texture_size;
    float dt = clamp(u_dt, 0.0, 1.0 / 15.0);
    float depth = clamp(state.z, 0.0, 1.0);
    float radius = max(velocity.a, 0.0001);

    float motion_speed = max(u_motion_speed, 0.0);
    vec2 flow = flow_field(state.xy, depth);
    vec2 curl = vec2(-flow.y, flow.x) * u_orbit_strength;
    flow = flow * (1.0 + u_mutation_strength * u_mutation_warp) + curl;
    float depth_speed = mix(1.35, 0.55, depth);
    float radius_speed = mix(0.65, 1.45, clamp(radius / max(u_base_radius, 0.0001), 0.0, 2.0));

    velocity.xy += flow * u_turbulence * (0.22 + u_show_chaos * 0.28) * dt;
    float target_rise = u_buoyancy * depth_speed * radius_speed * (0.10 + u_show_energy * 0.08);
    velocity.y = mix(velocity.y, target_rise, clamp(dt * (1.5 + u_buoyancy * 2.0), 0.0, 1.0));
    velocity.z += (noise(state.xy * 4.0 + u_time * 0.12 * max(u_flow_speed, 0.05)) - 0.5)
        * u_turbulence * 0.035 * dt;
    velocity.z += sin(u_time * (0.35 + u_flow_speed * 0.3) + id * 0.017)
        * u_depth_motion * 0.055 * dt;
    velocity.xy += vec2(
        sin(u_time * 8.0 + id * 0.37),
        cos(u_time * 6.3 + id * 0.29)
    ) * u_show_chaos * 0.028 * dt;
    float fracture_gate = step(
        0.96 - u_mutation_strength * u_mutation_fracture * 0.22,
        noise(vec2(id * 0.017, floor(u_time * 3.0 * max(u_mutation_rate, 0.01))))
    );
    velocity.xy += vec2(hash11(id + floor(u_time * 5.0)) - 0.5, 0.0)
        * fracture_gate * u_mutation_strength * 0.08;
    velocity.xyz *= pow(clamp(u_drag, 0.0, 1.0), dt * 60.0);

    state.xyz += velocity.xyz * dt * motion_speed;
    state.x = fract(state.x);
    state.z = clamp(state.z, 0.0, 1.0);
    state.a += dt * mix(0.035, 0.12, radius_speed + u_show_energy * 0.5);

    float top = 1.08 + radius * 2.0;
    bool expired = state.y > top || state.a > 1.0;
    if (expired) {
        float cycle = floor(u_time * 0.25 + state.a * 17.0 + hash11(id));
        respawn(id, cycle, state, velocity);
    }

    float swell = 0.85 + 0.25 * sin(clamp(state.a, 0.0, 1.0) * 3.14159265);
    float target_radius = u_base_radius * mix(0.35, 1.0 + u_radius_variation, hash11(id + float(u_seed) * 0.17 + 5.0)) * swell;
    velocity.a = max(0.0001, mix(radius, target_radius, clamp(dt * 8.0, 0.0, 1.0)));

    out_state = state;
    out_velocity = velocity;
}
