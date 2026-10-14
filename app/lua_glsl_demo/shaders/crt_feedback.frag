#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uPrev;
uniform sampler2D uInput0;
uniform vec2 uResolution;
uniform float uTime;
uniform int uFrame;

uniform float uFeedbackMix;
uniform float uCurvature;
uniform float uScanline;
uniform float uMask;
uniform float uGlow;
uniform float uJitter;
uniform float uChromatic;
uniform float uNoise;
uniform float uSaturation;
uniform float uUseInput;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + 1.0e-10)), d / (q.x + 1.0e-10), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec2 crt_warp(vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    float r2 = dot(p, p);
    p *= 1.0 + uCurvature * r2 * 0.18;
    return p * 0.5 + 0.5;
}

vec3 sample_chroma(sampler2D tex, vec2 uv, float amount) {
    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 0.0) * amount * 5.0;
    return vec3(texture(tex, uv + px).r, texture(tex, uv).g, texture(tex, uv - px).b);
}

void main() {
    vec2 uv = crt_warp(v_uv);
    float roll = sin(uTime * 8.0 + v_uv.y * 80.0) * uJitter * 0.002;
    uv.x += roll + (hash(vec2(floor(v_uv.y * 96.0), floor(uTime * 24.0))) - 0.5) * uJitter * 0.012;

    vec3 src = sample_chroma(uInput0, uv, uChromatic);
    vec2 fuv = (v_uv - 0.5) * 0.992 + 0.5 + vec2(0.0, -0.0015);
    vec3 prev = texture(uPrev, fuv).rgb;
    vec3 col = mix(prev, src, clamp(uUseInput, 0.0, 1.0));
    col = mix(col, prev, clamp(uFeedbackMix, 0.0, 0.97));

    vec3 bloom = (
        texture(uInput0, uv + vec2(2.0 / uResolution.x, 0.0)).rgb +
        texture(uInput0, uv - vec2(2.0 / uResolution.x, 0.0)).rgb +
        texture(uInput0, uv + vec2(0.0, 2.0 / uResolution.y)).rgb +
        texture(uInput0, uv - vec2(0.0, 2.0 / uResolution.y)).rgb
    ) * 0.25;
    col += max(bloom - 0.45, 0.0) * uGlow;

    float scan = 0.5 + 0.5 * sin(v_uv.y * uResolution.y * 3.14159265);
    col *= 1.0 - scan * uScanline * 0.18;
    vec3 mask = mix(vec3(1.0), vec3(
        0.75 + 0.25 * sin(v_uv.x * uResolution.x * 2.094),
        0.75 + 0.25 * sin(v_uv.x * uResolution.x * 2.094 + 2.094),
        0.75 + 0.25 * sin(v_uv.x * uResolution.x * 2.094 + 4.188)
    ), clamp(uMask, 0.0, 1.0));
    col *= mask;
    col += (hash(v_uv * uResolution + vec2(uFrame)) - 0.5) * uNoise * 0.12;

    vec3 hsv = rgb2hsv(clamp(col, 0.0, 1.0));
    hsv.y = clamp(hsv.y * uSaturation, 0.0, 1.0);
    fragColor = vec4(hsv2rgb(hsv), 1.0);
}
