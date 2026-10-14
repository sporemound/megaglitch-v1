#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uPrev;
uniform sampler2D uInput0;
uniform vec2 uResolution;
uniform float uTime;

uniform float uInputMix;
uniform float uFeedbackMix;
uniform float uScale;
uniform float uSpeed;
uniform float uRings;
uniform float uSymmetry;
uniform float uScopeGain;
uniform float uHue;
uniform float uContrast;

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    p.x *= uResolution.x / max(uResolution.y, 1.0);
    float sides = max(1.0, floor(uSymmetry));
    float a = atan(p.y, p.x);
    float r = length(p);
    a = mod(a, 6.2831853 / sides) - 3.14159265 / sides;
    vec2 k = vec2(cos(a), sin(a)) * r;

    float t = uTime * uSpeed;
    float field = sin(k.x * uScale + t) + sin(k.y * (uScale * 1.37) - t * 0.8);
    field += sin((k.x + k.y) * uScale * 0.65 + t * 1.3);
    float ring = sin(r * uRings * 18.0 - t * 2.0);
    float scope = smoothstep(0.045 * uScopeGain, 0.0, abs(field * 0.18 + ring * 0.08));

    vec3 plasma = hsv2rgb(vec3(fract(uHue + field * 0.08 + r * 0.18), 0.78, 0.35 + scope * 0.75));
    vec3 input_col = texture(uInput0, v_uv).rgb;
    vec3 prev = texture(uPrev, (v_uv - 0.5) * 0.992 + 0.5).rgb;
    vec3 col = mix(plasma, input_col, clamp(uInputMix, 0.0, 1.0));
    col = mix(col, prev, clamp(uFeedbackMix, 0.0, 0.98));
    col = (col - 0.5) * max(uContrast, 0.0) + 0.5;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
