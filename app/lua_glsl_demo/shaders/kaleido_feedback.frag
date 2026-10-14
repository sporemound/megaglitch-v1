#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uInput0;
uniform sampler2D uFeedbackInput;
uniform sampler2D uPrev;
uniform vec2 uResolution;
uniform float uTime;

uniform float uSegments;
uniform float uRotation;
uniform float uZoom;
uniform float uFeedbackMix;
uniform float uBloom;
uniform float uColorCycle;
uniform float uChromatic;
uniform float uUseInput;
uniform float uExternalFeedbackMix;
uniform float uExternalFeedbackZoom;
uniform float uExternalFeedbackRotate;
uniform float uExternalFeedbackWarp;
uniform float uFeedbackDecay;
uniform float uFeedbackSpin;
uniform float uMirrorOffset;
uniform float uCenterX;
uniform float uCenterY;
uniform float uSourceMix;
uniform float uFeedbackColorize;

const float PI = 3.14159265359;

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

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

vec2 rotate2(vec2 p, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c) * p;
}

vec2 kaleido(vec2 uv) {
    vec2 center = vec2(uCenterX, uCenterY);
    vec2 p = (uv - center) * 2.0;
    p.x *= uResolution.x / max(uResolution.y, 1.0);
    p *= max(uZoom, 0.01);
    float seg = max(1.0, floor(uSegments));
    float a = atan(p.y, p.x) + uRotation + uTime * 0.08;
    float r = length(p);
    float slice = 2.0 * PI / seg;
    a = abs(mod(a + slice * (0.5 + uMirrorOffset), slice) - slice * 0.5);
    vec2 q = vec2(cos(a), sin(a)) * r;
    q.x /= uResolution.x / max(uResolution.y, 1.0);
    return q * 0.5 + center;
}

vec3 sample_chroma(sampler2D tex, vec2 uv, float amount) {
    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 1.0 / max(uResolution.y, 1.0));
    vec2 d = normalize(uv - 0.5 + vec2(0.0001)) * px * amount * 8.0;
    return vec3(texture(tex, uv + d).r, texture(tex, uv).g, texture(tex, uv - d).b);
}

vec2 feedback_uv(vec2 uv, float zoom, float rotation, float warp_amount) {
    vec2 center = vec2(uCenterX, uCenterY);
    vec2 p = uv - center;
    p.x *= uResolution.x / max(uResolution.y, 1.0);
    p = rotate2(p, rotation);
    p *= max(0.01, 1.0 - zoom);
    p.x /= uResolution.x / max(uResolution.y, 1.0);
    vec2 out_uv = p + center;
    float n1 = noise(out_uv * 3.5 + vec2(uTime * 0.11, -uTime * 0.09));
    float n2 = noise(out_uv * 6.0 - vec2(uTime * 0.07, uTime * 0.13));
    return out_uv + (vec2(n1, n2) - 0.5) * warp_amount * 0.08;
}

void main() {
    vec2 kuv = kaleido(v_uv);
    vec3 src = sample_chroma(uInput0, kuv, uChromatic);
    vec2 prev_uv = feedback_uv(kuv, 0.015, uFeedbackSpin * uTime * 0.05, 0.0);
    vec3 prev = texture(uPrev, prev_uv).rgb * uFeedbackDecay;
    vec2 ext_uv = feedback_uv(
        kuv,
        uExternalFeedbackZoom,
        uExternalFeedbackRotate + uFeedbackSpin * uTime * 0.08,
        uExternalFeedbackWarp
    );
    vec3 external_feedback = sample_chroma(uFeedbackInput, ext_uv, uChromatic * 0.65);
    external_feedback = mix(
        external_feedback,
        hsv2rgb(vec3(fract(uColorCycle + uTime * 0.06 + external_feedback.r * 0.2), 0.82, max(max(external_feedback.r, external_feedback.g), external_feedback.b))),
        clamp(uFeedbackColorize, 0.0, 1.0)
    );
    vec3 col = mix(prev, src, clamp(uUseInput, 0.0, 1.0));
    col = mix(col, src, clamp(uSourceMix, 0.0, 1.0));
    col = mix(col, external_feedback, clamp(uExternalFeedbackMix, 0.0, 1.0));
    col = mix(col, prev, clamp(uFeedbackMix, 0.0, 0.98));

    float hot = max(max(col.r, col.g), col.b);
    col += hsv2rgb(vec3(fract(uColorCycle + uTime * 0.035 + hot * 0.12), 0.65, hot)) * max(hot - 0.45, 0.0) * uBloom;

    vec2 p = v_uv * 2.0 - 1.0;
    col *= smoothstep(1.45, 0.2, dot(p, p));
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
