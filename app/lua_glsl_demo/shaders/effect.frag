#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uInput0;
uniform sampler2D uPrev;

uniform vec2 uResolution;
uniform float uTime;
uniform float uDelta;

uniform float uFeedbackMix;
uniform float uAberration;
uniform float uWarp;
uniform float uNoise;
uniform float uExposure;
uniform float uContrast;
uniform float uSaturation;
uniform float uHue;
uniform float uYFlip;

const float PI = 3.14159265359;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 1.0 / 3.0, 2.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec2 aspect_correct(vec2 uv) {
    vec2 st = uv * 2.0 - 1.0;
    st.x *= uResolution.x / max(uResolution.y, 1.0);
    return st;
}

vec2 barrel(vec2 uv, float k) {
    if (abs(k) < 0.000001) {
        return uv;
    }
    vec2 st = aspect_correct(uv);
    float r2 = dot(st, st);
    vec2 warped = st * (1.0 + k * r2);
    vec2 out_uv = vec2(warped.x / (uResolution.x / max(uResolution.y, 1.0)), warped.y);
    return (out_uv + 1.0) * 0.5;
}

float hash(vec2 p) {
    p = fract(p * vec2(443.8975, 441.423));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

vec3 sample_aberrated(sampler2D tex, vec2 uv, float px) {
    vec2 off = vec2(px * uAberration);
    float t = uTime * 0.3;
    vec2 wiggle = off * vec2(sin(t), cos(t)) * 0.5;
    float r = texture(tex, uv + off + wiggle).r;
    float g = texture(tex, uv).g;
    float b = texture(tex, uv - off - wiggle).b;
    return vec3(r, g, b);
}

vec3 apply_tone(vec3 col, float exposure, float contrast, float saturation, float hue) {
    col *= max(exposure, 0.0);
    vec3 hsv = rgb2hsv(clamp(col, 0.0, 1.0));
    hsv.x = fract(hsv.x + hue / (2.0 * PI));
    hsv.y = clamp(hsv.y * saturation, 0.0, 1.0);
    col = hsv2rgb(hsv);
    col = (col - 0.5) * contrast + 0.5;
    return clamp(col, 0.0, 1.0);
}

void main() {
    vec2 uv = v_uv;
    if (uYFlip > 0.5) {
        uv.y = 1.0 - uv.y;
    }

    float warp_anim = uWarp * (0.8 + 0.2 * sin(uTime * 0.7));
    vec2 wuv = barrel(uv, warp_anim);
    float px = 1.0 / max(uResolution.y, 1.0);

    vec3 src = sample_aberrated(uInput0, wuv, px);
    vec3 out_col = src;

    if (uFeedbackMix > 0.001) {
        float angle = 0.01 * sin(uTime * 0.6);
        float s = sin(angle);
        float c = cos(angle);
        vec2 centered = uv - 0.5;
        vec2 rot = vec2(centered.x * c - centered.y * s, centered.x * s + centered.y * c);
        vec2 fuv = rot * (1.0 - 0.006) + 0.5;
        vec3 fb = texture(uPrev, fuv).rgb;
        out_col = mix(out_col, fb, clamp(uFeedbackMix, 0.0, 1.0));
    }

    float grain = hash(uv * uResolution + vec2(uTime * 37.7, uTime * 15.3));
    out_col += (grain - 0.5) * (uNoise * 0.12);

    vec2 ac = aspect_correct(uv);
    float radius = dot(ac, ac);
    float vignette = smoothstep(1.15, 0.35, radius);
    out_col *= mix(0.85, 1.05, vignette);

    out_col = apply_tone(out_col, uExposure, uContrast, uSaturation, uHue);
    fragColor = vec4(out_col, 1.0);
}
