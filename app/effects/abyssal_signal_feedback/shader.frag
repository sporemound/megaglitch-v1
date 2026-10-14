#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_slot_time;
uniform vec2 u_resolution;
uniform sampler2D u_image;
uniform sampler2D u_feedback;
uniform sampler2D u_bubble_mask;
uniform sampler2D u_bubble_normal;
uniform sampler2D u_bubble_depth;
uniform sampler2D u_bubble_refraction;
uniform sampler2D u_bubble_highlight;
uniform float u_master;
uniform int u_phase;
uniform float u_section_progress;
uniform float u_energy;
uniform float u_chaos;
uniform float u_beat;
uniform float u_caustics;
uniform float u_sonar;
uniform float u_chroma;
uniform float u_tear;
uniform float u_feedback_mix;
uniform float u_scanlines;
uniform float u_spiral;
uniform float u_vignette;
uniform float u_bubble_mix;
uniform float u_bubble_refraction_mix;
uniform float u_bubble_highlight_mix;

const float PI = 3.14159265359;

float hash(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 34.79);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amp = 0.5;
    mat2 rot = mat2(0.80, -0.60, 0.60, 0.80);
    for (int i = 0; i < 4; i++) {
        value += amp * noise(p);
        p = rot * p * 2.03 + 17.13;
        amp *= 0.5;
    }
    return value;
}

vec2 aspect_position(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    p.x *= u_resolution.x / max(u_resolution.y, 1.0);
    return p;
}

float caustic_pattern(vec2 uv) {
    vec2 p = uv * vec2(2.1, 1.3);
    float t = u_time * 0.08;
    float a = sin((p.x + fbm(p * 2.0 + t)) * 18.0 + u_time * 0.7);
    float b = sin((p.y + fbm(p * 2.7 - t)) * 15.0 - u_time * 0.5);
    float c = sin((p.x + p.y) * 12.0 + fbm(p * 4.0 + u_time * 0.05) * 7.0);
    float lines = abs(a + b + c) / 3.0;
    return smoothstep(0.50, 0.98, lines);
}

float sonar_ring(vec2 uv) {
    vec2 p = aspect_position(uv);
    float d = length(p);
    float sweep = fract(u_slot_time * 0.075 + float(u_phase) * 0.137);
    float ring = 1.0 - smoothstep(0.0, 0.030, abs(d - sweep * 1.65));
    float second = 1.0 - smoothstep(0.0, 0.020, abs(d - fract(sweep + 0.33) * 1.65));
    float radial = atan(p.y, p.x);
    float beam = smoothstep(0.965, 1.0, cos(radial - u_time * 0.42));
    float grid = smoothstep(0.985, 1.0, cos(d * 52.0 - u_time * 1.7));
    return max(ring, second * 0.55) * (0.45 + beam * 0.75) + grid * 0.18;
}

vec2 spiral_warp(vec2 uv, float amount) {
    vec2 p = aspect_position(uv);
    float r = length(p);
    float a = atan(p.y, p.x);
    float pull = amount * smoothstep(1.10, 0.05, r);
    a += pull * (2.4 + sin(u_time * 0.09) * 0.6) + r * amount * 1.8;
    r *= 1.0 - amount * 0.055 * (0.5 + 0.5 * sin(u_time * 0.21 + r * 9.0));
    vec2 warped = vec2(cos(a), sin(a)) * r;
    warped.x /= u_resolution.x / max(u_resolution.y, 1.0);
    return warped * 0.5 + 0.5;
}

vec2 block_tear(vec2 uv, float amount) {
    float coarse = floor(uv.y * mix(18.0, 92.0, amount));
    float fine = floor(uv.y * mix(90.0, 260.0, amount));
    float tick = floor(u_time * mix(7.0, 26.0, amount));
    float coarse_hash = hash(vec2(coarse, tick));
    float fine_hash = hash(vec2(fine, tick + 19.0));
    float gate = smoothstep(0.58, 0.98, coarse_hash + amount * 0.22);
    float rip = sin(uv.y * 120.0 + u_time * 9.0 + coarse_hash * 6.0) * 0.006;
    float offset = (coarse_hash - 0.5) * amount * 0.16 * gate;
    offset += (fine_hash - 0.5) * amount * 0.030 * step(0.78, fine_hash);
    return uv + vec2(offset + rip * amount, 0.0);
}

vec3 chromatic_sample(sampler2D tex, vec2 uv, float amount) {
    vec2 p = normalize(aspect_position(uv) + vec2(0.0001));
    vec2 px = 1.0 / max(u_resolution, vec2(1.0));
    vec2 off = p * px * (2.0 + amount * 22.0);
    float r = texture(tex, uv + off).r;
    float g = texture(tex, uv).g;
    float b = texture(tex, uv - off).b;
    return vec3(r, g, b);
}

float crt_scanline(vec2 uv) {
    float line = sin(uv.y * u_resolution.y * PI);
    float grille = sin(uv.x * u_resolution.x * PI * 0.333);
    float scan = 1.0 - u_scanlines * (0.055 + 0.065 * (0.5 + 0.5 * line));
    float mask = 1.0 - u_scanlines * 0.035 * (0.5 + 0.5 * grille);
    return scan * mask;
}

float vignette(vec2 uv) {
    vec2 p = aspect_position(uv);
    float d = dot(p, p);
    return mix(1.0, smoothstep(1.65, 0.18, d), u_vignette);
}

vec3 grade(vec3 col, vec2 uv) {
    vec3 deep = vec3(0.010, 0.035, 0.055);
    vec3 teal = vec3(0.12, 0.70, 0.74);
    vec3 amber = vec3(1.00, 0.62, 0.24);
    float c = caustic_pattern(uv) * u_caustics;
    float s = sonar_ring(uv) * u_sonar;
    col = mix(col, deep + col * 0.45, 0.22 + u_energy * 0.22);
    col += teal * c * (0.18 + u_master * 0.22);
    col += mix(teal, amber, u_beat) * s * (0.15 + u_energy * 0.28);
    col += amber * u_beat * u_energy * 0.08;
    col = (col - 0.5) * (1.0 + u_chaos * 0.34) + 0.5;
    col *= 0.92 + u_energy * 0.25;
    return col;
}

void main() {
    vec2 uv = v_uv;
    float master = clamp(u_master, 0.0, 1.0);
    float chaos = clamp(u_chaos, 0.0, 1.0);
    float decompression = (u_phase == 6) ? (1.0 - u_section_progress) : 1.0;

    vec2 drift = vec2(
        fbm(uv * 3.1 + vec2(u_time * 0.030, -u_time * 0.018)) - 0.5,
        fbm(uv * 3.5 + vec2(-u_time * 0.022, u_time * 0.027)) - 0.5
    );

    vec2 warped_uv = uv + drift * (0.012 + u_caustics * 0.030);
    warped_uv = spiral_warp(warped_uv, u_spiral * decompression);
    warped_uv = block_tear(warped_uv, u_tear * decompression);

    vec4 bubble_refraction = texture(u_bubble_refraction, uv);
    vec4 bubble_mask_sample = texture(u_bubble_mask, uv);
    vec4 bubble_depth = texture(u_bubble_depth, uv);
    vec3 bubble_normal = texture(u_bubble_normal, uv).rgb * 2.0 - 1.0;
    vec3 bubble_highlight = texture(u_bubble_highlight, uv).rgb;
    float bubble_mask = clamp(bubble_mask_sample.r * u_bubble_mix, 0.0, 1.0);
    vec2 bubble_offset = ((bubble_refraction.rg * 2.0 - 1.0) + bubble_normal.xy * 0.25);
    bubble_offset *= u_bubble_refraction_mix * bubble_mask * mix(1.25, 0.45, bubble_depth.r);
    warped_uv += bubble_offset;

    vec2 feedback_uv = spiral_warp(uv, u_spiral * 1.25 + u_feedback_mix * 0.20);
    feedback_uv = block_tear(feedback_uv + drift * 0.018, u_tear * 0.55);
    feedback_uv = (feedback_uv - 0.5) * (1.0 - 0.035 * u_feedback_mix - 0.018 * u_beat) + 0.5;

    vec3 src = chromatic_sample(u_image, clamp(warped_uv, 0.001, 0.999), u_chroma);
    vec3 prev = chromatic_sample(u_feedback, clamp(feedback_uv, 0.001, 0.999), u_chroma * 0.65);
    prev *= 0.985 - u_feedback_mix * 0.070;

    float smear = smoothstep(0.30, 1.0, u_feedback_mix);
    vec3 col = mix(src, prev, clamp(u_feedback_mix, 0.0, 0.96));
    col = mix(col, max(col, prev + src * 0.12), smear * 0.45);

    float tear_shadow = abs(warped_uv.x - uv.x) * 6.0;
    col += vec3(0.02, 0.16, 0.15) * tear_shadow * u_tear;
    col = grade(col, uv);
    col = mix(col, col + bubble_highlight * u_bubble_highlight_mix, bubble_mask);
    col = mix(col, col * (0.86 + bubble_depth.r * 0.18), bubble_mask * 0.20);

    float scan = crt_scanline(uv);
    float vig = vignette(uv);
    float grain = hash(uv * u_resolution + vec2(floor(u_time * 60.0), u_slot_time));
    col *= scan * vig;
    col += (grain - 0.5) * (0.025 + chaos * 0.075);

    float bloom = smoothstep(0.70, 1.45, dot(col, vec3(0.299, 0.587, 0.114)));
    col += bloom * vec3(0.08, 0.18, 0.20) * (u_energy + u_beat);

    if (u_phase == 6) {
        vec3 calm = vec3(dot(col, vec3(0.299, 0.587, 0.114))) * vec3(0.70, 0.88, 0.92);
        col = mix(calm * (0.45 + decompression * 0.55), col, decompression);
    }

    fragColor = vec4(clamp(col * master, 0.0, 1.0), 1.0);
}
