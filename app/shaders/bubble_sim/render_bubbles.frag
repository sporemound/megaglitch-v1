#version 330 core

in float v_depth;
in float v_age;
in float v_seed;
in float v_radius;

layout(location = 0) out vec4 out_mask;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_depth;
layout(location = 3) out vec4 out_refraction;
layout(location = 4) out vec4 out_highlight;

uniform float u_density;
uniform float u_refraction_strength;
uniform float u_highlight_strength;
uniform float u_sparkle;
uniform float u_show_energy;
uniform float u_show_chaos;
uniform float u_mutation_strength;
uniform float u_mutation_rate;
uniform float u_mutation_palette;
uniform float u_time;

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

void main() {
    if (hash11(v_seed) > u_density) {
        discard;
    }

    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) {
        discard;
    }

    float z = sqrt(max(0.0, 1.0 - r2));
    vec3 normal = normalize(vec3(p, z));
    float edge = smoothstep(1.0, 0.62, r2);
    float rim = smoothstep(0.45, 0.98, r2) * edge;
    float depth_gain = mix(1.0, 0.38, clamp(v_depth, 0.0, 1.0));
    float pop = smoothstep(0.78, 1.0, v_age);
    float sparkle_seed = hash11(v_seed * 19.7 + floor(u_time * 18.0));
    float glint = pow(max(dot(normal, normalize(vec3(-0.45, 0.55, 0.70))), 0.0), 26.0);
    glint += step(0.70, sparkle_seed) * (0.25 + pop * 0.75) * u_sparkle * 1.8;

    float alpha = edge * depth_gain * mix(0.55, 1.0, u_show_energy + 0.2);
    vec2 refract = normal.xy * u_refraction_strength * mix(1.25, 0.45, v_depth);
    float palette_shift = sin(u_time * 0.25 * max(u_mutation_rate, 0.01) + v_seed * 0.13) * u_mutation_strength * u_mutation_palette;
    vec3 tint = vec3(0.72 + palette_shift * 0.16, 0.92, 1.0 - palette_shift * 0.12);
    vec3 highlight = tint * (rim * 0.22 + glint * 0.65) * u_highlight_strength;

    out_mask = vec4(alpha, alpha, alpha, alpha);
    out_normal = vec4(normal * 0.5 + 0.5, alpha);
    out_depth = vec4(v_depth, v_age, v_radius, alpha);
    out_refraction = vec4(refract * 0.5 + 0.5, alpha, rim);
    out_highlight = vec4(highlight, alpha);
}
