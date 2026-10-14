#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uInput0;
uniform sampler2D uPrev;
uniform sampler2D u_bubble_mask;
uniform sampler2D u_bubble_normal;
uniform sampler2D u_bubble_depth;
uniform sampler2D u_bubble_refraction;
uniform sampler2D u_bubble_highlight;
uniform vec2 uResolution;
uniform float uTime;

uniform float u_slot_time;
uniform float u_master;
uniform float u_phase;
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
uniform float u_caustics_scale;
uniform float u_caustics_speed;
uniform float u_sonar_speed;
uniform float u_sonar_width;
uniform float u_sonar_beam;
uniform float u_chroma_pixels;
uniform float u_tear_density;
uniform float u_tear_speed;
uniform float u_tear_displace;
uniform float u_feedback_decay;
uniform float u_feedback_zoom;
uniform float u_spiral_twist;
uniform float u_spiral_radius;
uniform float u_grille_gain;
uniform float u_deep_mix;
uniform float u_teal_gain;
uniform float u_amber_gain;
uniform float u_brightness;
uniform float u_contrast;
uniform float u_grain_gain;
uniform float u_bloom_gain;

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
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
               u.y);
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
    p.x *= uResolution.x / max(uResolution.y, 1.0);
    return p;
}

float caustic_pattern(vec2 uv) {
    vec2 p = uv * vec2(2.1, 1.3) * max(u_caustics_scale, 0.001);
    float t = uTime * 0.08 * u_caustics_speed;
    float a = sin((p.x + fbm(p * 2.0 + t)) * 18.0 + uTime * 0.7 * u_caustics_speed);
    float b = sin((p.y + fbm(p * 2.7 - t)) * 15.0 - uTime * 0.5 * u_caustics_speed);
    float c = sin((p.x + p.y) * 12.0 + fbm(p * 4.0 + uTime * 0.05 * u_caustics_speed) * 7.0);
    return smoothstep(0.50, 0.98, abs(a + b + c) / 3.0);
}

float sonar_ring(vec2 uv) {
    vec2 p = aspect_position(uv);
    float d = length(p);
    float width = max(u_sonar_width, 0.001);
    float sweep = fract(u_slot_time * 0.075 * u_sonar_speed + u_phase * 0.137);
    float ring = 1.0 - smoothstep(0.0, width, abs(d - sweep * 1.65));
    float second = 1.0 - smoothstep(0.0, width * 0.66, abs(d - fract(sweep + 0.33) * 1.65));
    float radial = atan(p.y, p.x);
    float beam = smoothstep(1.0 - 0.035 * max(u_sonar_beam, 0.0), 1.0, cos(radial - uTime * 0.42 * u_sonar_speed));
    float grid = smoothstep(0.985, 1.0, cos(d * 52.0 - uTime * 1.7));
    return max(ring, second * 0.55) * (0.45 + beam * 0.75) + grid * 0.18;
}

vec2 spiral_warp(vec2 uv, float amount) {
    vec2 p = aspect_position(uv);
    float r = length(p);
    float a = atan(p.y, p.x);
    float pull = amount * smoothstep(1.10 * u_spiral_radius, 0.05, r);
    a += pull * u_spiral_twist * (2.4 + sin(uTime * 0.09) * 0.6) + r * amount * u_spiral_twist * 1.8;
    r *= 1.0 - amount * 0.055 * (0.5 + 0.5 * sin(uTime * 0.21 + r * 9.0));
    vec2 warped = vec2(cos(a), sin(a)) * r;
    warped.x /= uResolution.x / max(uResolution.y, 1.0);
    return warped * 0.5 + 0.5;
}

vec2 block_tear(vec2 uv, float amount) {
    float density = max(u_tear_density, 0.001);
    float coarse = floor(uv.y * mix(18.0, 92.0, amount) * density);
    float fine = floor(uv.y * mix(90.0, 260.0, amount) * density);
    float tick = floor(uTime * mix(7.0, 26.0, amount) * u_tear_speed);
    float coarse_hash = hash(vec2(coarse, tick));
    float fine_hash = hash(vec2(fine, tick + 19.0));
    float gate = smoothstep(0.58, 0.98, coarse_hash + amount * 0.22);
    float rip = sin(uv.y * 120.0 + uTime * 9.0 + coarse_hash * 6.0) * 0.006;
    float offset = (coarse_hash - 0.5) * amount * 0.16 * gate * u_tear_displace;
    offset += (fine_hash - 0.5) * amount * 0.030 * step(0.78, fine_hash) * u_tear_displace;
    return uv + vec2(offset + rip * amount, 0.0);
}

vec3 chromatic_sample(sampler2D tex, vec2 uv, float amount) {
    vec2 p = normalize(aspect_position(uv) + vec2(0.0001));
    vec2 px = 1.0 / max(uResolution, vec2(1.0));
    vec2 off = p * px * (2.0 + amount * max(u_chroma_pixels, 0.0));
    return vec3(texture(tex, uv + off).r, texture(tex, uv).g, texture(tex, uv - off).b);
}

float crt_scanline(vec2 uv) {
    float line = sin(uv.y * uResolution.y * PI);
    float grille = sin(uv.x * uResolution.x * PI * 0.333);
    return (1.0 - u_scanlines * (0.055 + 0.065 * (0.5 + 0.5 * line))) *
           (1.0 - u_scanlines * u_grille_gain * 0.035 * (0.5 + 0.5 * grille));
}

float vignette(vec2 uv) {
    vec2 p = aspect_position(uv);
    return mix(1.0, smoothstep(1.65, 0.18, dot(p, p)), u_vignette);
}

void main() {
    vec2 uv = v_uv;
    float master = clamp(u_master, 0.0, 1.0);
    float chaos = clamp(u_chaos, 0.0, 1.0);
    float decompression = (u_phase > 5.5) ? (1.0 - u_section_progress) : 1.0;

    vec2 drift = vec2(
        fbm(uv * 3.1 + vec2(uTime * 0.030, -uTime * 0.018)) - 0.5,
        fbm(uv * 3.5 + vec2(-uTime * 0.022, uTime * 0.027)) - 0.5
    );

    vec2 warped_uv = uv + drift * (0.012 + u_caustics * 0.030);
    warped_uv = spiral_warp(warped_uv, u_spiral * decompression);
    warped_uv = block_tear(warped_uv, u_tear * decompression);

    vec4 bubble_mask_sample = texture(u_bubble_mask, uv);
    vec3 bubble_normal = texture(u_bubble_normal, uv).rgb * 2.0 - 1.0;
    vec3 bubble_depth = texture(u_bubble_depth, uv).rgb;
    vec3 bubble_refraction = texture(u_bubble_refraction, uv).rgb;
    vec3 bubble_highlight = texture(u_bubble_highlight, uv).rgb;
    float bubble_mask = clamp(bubble_mask_sample.r, 0.0, 1.0);
    vec2 bubble_offset = (bubble_refraction.rg - 0.5) * 0.09;
    bubble_offset += bubble_normal.xy * 0.006;
    bubble_offset *= bubble_mask * mix(1.2, 0.45, bubble_depth.r);
    warped_uv += bubble_offset;

    vec2 feedback_uv = spiral_warp(uv, u_spiral * 1.25 + u_feedback_mix * 0.20);
    feedback_uv = block_tear(feedback_uv + drift * 0.018, u_tear * 0.55);
    feedback_uv = (feedback_uv - 0.5) * (1.0 - u_feedback_zoom * u_feedback_mix - 0.018 * u_beat) + 0.5;

    vec3 src = chromatic_sample(uInput0, clamp(warped_uv, 0.001, 0.999), u_chroma);
    vec3 prev = chromatic_sample(uPrev, clamp(feedback_uv, 0.001, 0.999), u_chroma * 0.65);
    prev *= u_feedback_decay - u_feedback_mix * 0.040;

    vec3 deep = vec3(0.010, 0.035, 0.055);
    vec3 teal = vec3(0.12, 0.70, 0.74);
    vec3 amber = vec3(1.00, 0.62, 0.24);

    vec3 col = mix(src, prev, clamp(u_feedback_mix, 0.0, 0.96));
    col = mix(col, max(col, prev + src * 0.12), smoothstep(0.30, 1.0, u_feedback_mix) * 0.45);
    col += vec3(0.02, 0.16, 0.15) * abs(warped_uv.x - uv.x) * 6.0 * u_tear;

    float caustic = caustic_pattern(uv) * u_caustics;
    float sonar = sonar_ring(uv) * u_sonar;
    col = mix(col, deep + col * 0.45, (0.22 + u_energy * 0.22) * u_deep_mix);
    col += teal * u_teal_gain * caustic * (0.18 + master * 0.22);
    col += mix(teal * u_teal_gain, amber * u_amber_gain, u_beat) * sonar * (0.15 + u_energy * 0.28);
    col += amber * u_amber_gain * u_beat * u_energy * 0.08;
    float bubble_atmosphere = bubble_mask * mix(0.03, 0.16, bubble_depth.r);
    col = mix(col, col * vec3(0.78, 0.90, 0.94), bubble_atmosphere);
    col += bubble_highlight * bubble_mask * 0.32;
    col += bubble_refraction.b * bubble_mask * vec3(0.015, 0.035, 0.045);

    col *= u_brightness;
    col = (col - 0.5) * u_contrast * (1.0 + chaos * 0.34) + 0.5;
    col *= (0.92 + u_energy * 0.25) * crt_scanline(uv) * vignette(uv);

    float grain = hash(uv * uResolution + vec2(floor(uTime * 60.0), u_slot_time));
    col += (grain - 0.5) * (0.025 + chaos * 0.075) * u_grain_gain;

    float bloom = smoothstep(0.70, 1.45, dot(col, vec3(0.299, 0.587, 0.114)));
    col += bloom * vec3(0.08, 0.18, 0.20) * (u_energy + u_beat) * u_bloom_gain;

    if (u_phase > 5.5) {
        vec3 calm = vec3(dot(col, vec3(0.299, 0.587, 0.114))) * vec3(0.70, 0.88, 0.92);
        col = mix(calm * (0.45 + decompression * 0.55), col, decompression);
    }

    fragColor = vec4(clamp(col * master, 0.0, 1.0), 1.0);
}
