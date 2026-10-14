#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uPrev;
uniform sampler2D uInput0;

uniform vec2 uResolution;
uniform float uTime;
uniform float uDelta;
uniform int uFrame;

uniform float uUseInput;
uniform float uFeedbackMix;
uniform float uFeedbackZoom;
uniform float uWarp;
uniform float uGlitch;
uniform float uScanline;
uniform float uChromatic;
uniform float uPosterize;
uniform float uColorDrift;
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;
uniform float uInvert;

float hash(vec2 p) {
    p = fract(p * vec2(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

float temporal_hash(float row, float time_value) {
    float cell = floor(time_value);
    float phase = fract(time_value);
    phase = phase * phase * (3.0 - 2.0 * phase);
    float current = hash(vec2(row, cell));
    float next = hash(vec2(row, cell + 1.0));
    return mix(current, next, phase);
}

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec2 aspect_uv(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    p.x *= uResolution.x / max(uResolution.y, 1.0);
    return p;
}

vec2 warp_uv(vec2 uv, float amount) {
    vec2 p = aspect_uv(uv);
    float n1 = noise(p * 2.7 + vec2(uTime * 0.17, -uTime * 0.11));
    float n2 = noise(p * 6.0 - vec2(uTime * 0.09, uTime * 0.13));
    vec2 flow = vec2(n1 - 0.5, n2 - 0.5);
    float ripple = sin((length(p) * 10.0) - uTime * 2.0) * 0.5 + 0.5;
    return uv + (flow * 0.08 + normalize(p + 1.0e-4) * ripple * 0.015) * amount;
}

vec3 sample_rgb(sampler2D tex, vec2 uv, float chroma) {
    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 1.0 / max(uResolution.y, 1.0));
    vec2 dir = normalize(uv - 0.5 + vec2(1.0e-4));
    vec2 off = dir * px * chroma * 8.0;
    float r = texture(tex, uv + off).r;
    float g = texture(tex, uv).g;
    float b = texture(tex, uv - off).b;
    return vec3(r, g, b);
}

vec3 apply_tone(vec3 col) {
    col *= max(uBrightness, 0.0);
    col = (col - 0.5) * max(uContrast, 0.0) + 0.5;

    vec3 hsv = rgb2hsv(clamp(col, 0.0, 1.0));
    hsv.x = fract(hsv.x + uColorDrift);
    hsv.y = clamp(hsv.y * max(uSaturation, 0.0), 0.0, 1.0);
    col = hsv2rgb(hsv);

    if (uPosterize > 1.0) {
        float steps = floor(uPosterize);
        col = floor(col * steps) / max(steps - 1.0, 1.0);
    }

    return mix(col, 1.0 - col, clamp(uInvert, 0.0, 1.0));
}

void main() {
    vec2 uv = v_uv;
    float glitch = clamp(uGlitch, 0.0, 1.0);

    float band = floor(uv.y * mix(24.0, 180.0, glitch));
    // Interpolate between tear states instead of replacing the pattern every
    // frame. This keeps the damaged-sync motion without full-frame strobing.
    float band_rand = temporal_hash(band, uTime * 2.25);
    float tear = (band_rand - 0.5) * glitch * 0.08;
    tear *= step(0.72, band_rand);

    vec2 source_uv = warp_uv(uv + vec2(tear, 0.0), uWarp);
    vec2 feedback_uv = (uv - 0.5) * (1.0 - uFeedbackZoom * 0.02) + 0.5;
    feedback_uv = warp_uv(feedback_uv - vec2(tear * 0.35, 0.0), uWarp * 0.65);

    vec3 src = sample_rgb(uInput0, source_uv, uChromatic);
    vec3 prev = sample_rgb(uPrev, feedback_uv, uChromatic * 0.5);
    vec3 col = mix(prev, src, clamp(uUseInput, 0.0, 1.0));
    col = mix(col, prev, clamp(uFeedbackMix, 0.0, 0.98));

    float scan = sin((uv.y * uResolution.y) * 3.14159265);
    col *= 1.0 - (0.04 + 0.08 * glitch) * uScanline * (0.5 + 0.5 * scan);

    // Stable sensor grain prevents high-feedback frames from flashing as a
    // completely unrelated noise field on every render.
    float grain_static = hash(floor(uv * uResolution));
    float grain_drift = noise(uv * uResolution * 0.035 + vec2(uTime * 0.7, -uTime * 0.43));
    float grain = mix(grain_static, grain_drift, 0.28);
    col += (grain - 0.5) * (0.025 + glitch * 0.065);

    vec2 p = aspect_uv(uv);
    float vignette = smoothstep(1.55, 0.25, dot(p, p));
    col *= mix(0.78, 1.08, vignette);

    fragColor = vec4(clamp(apply_tone(col), 0.0, 1.0), 1.0);
}
