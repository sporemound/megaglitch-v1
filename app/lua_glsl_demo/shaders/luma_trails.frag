#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uInput0;
uniform sampler2D uPrev;
uniform vec2 uResolution;
uniform float uTime;

uniform float uThreshold;
uniform float uSoftness;
uniform float uTrailMix;
uniform float uTrailDecay;
uniform float uEdgeGlow;
uniform float uWarp;
uniform float uTintHue;
uniform float uUseInput;

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    vec2 p = v_uv * 2.0 - 1.0;
    p.x *= uResolution.x / max(uResolution.y, 1.0);
    vec2 flow = vec2(sin(p.y * 6.0 + uTime), cos(p.x * 5.0 - uTime * 0.7)) * uWarp * 0.006;
    vec2 uv = v_uv + flow;

    vec3 src = texture(uInput0, uv).rgb;
    float y = luma(src);
    float matte = smoothstep(uThreshold - uSoftness, uThreshold + uSoftness, y);
    vec3 prev = texture(uPrev, v_uv - flow * 1.8).rgb * uTrailDecay;

    vec3 tint = hsv2rgb(vec3(fract(uTintHue + y * 0.15), 0.65, 1.0));
    vec3 keyed = mix(src * 0.35, tint * y + src * 0.45, matte);
    float edge = matte * (1.0 - smoothstep(uThreshold + uSoftness, uThreshold + uSoftness * 2.5, y));
    keyed += tint * edge * uEdgeGlow;

    vec3 base = mix(prev, keyed, clamp(uUseInput, 0.0, 1.0));
    vec3 col = mix(keyed, max(base, prev), clamp(uTrailMix, 0.0, 0.98));
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
