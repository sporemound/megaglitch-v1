#version 330 core

in vec2 v_uv;
layout(location = 0) out vec4 out_state;
layout(location = 1) out vec4 out_velocity;

uniform int u_seed;
uniform float u_particle_texture_size;
uniform float u_base_radius;
uniform float u_radius_variation;
uniform float u_depth_spread;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33 + float(u_seed) * 0.00017);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec2 texel = floor(v_uv * u_particle_texture_size);
    float id = texel.x + texel.y * u_particle_texture_size + float(u_seed) * 13.0;

    float rx = hash11(id + 1.0);
    float ry = hash11(id + 2.0);
    float rz = hash11(id + 3.0);
    float ra = hash11(id + 4.0);
    float rr = hash11(id + 5.0);

    vec2 pos = vec2(rx, ry);
    float depth = mix(0.08, 1.0, rz) * max(u_depth_spread, 0.001);
    float age = ra;
    float radius = u_base_radius * mix(0.35, 1.0 + u_radius_variation, rr);

    float wobble = hash12(texel + 17.0) - 0.5;
    vec3 velocity = vec3(
        wobble * 0.010,
        (hash11(id + 6.0) - 0.5) * 0.012,
        (hash11(id + 7.0) - 0.5) * 0.003
    );

    out_state = vec4(pos, clamp(depth, 0.0, 1.0), age);
    out_velocity = vec4(velocity, radius);
}
