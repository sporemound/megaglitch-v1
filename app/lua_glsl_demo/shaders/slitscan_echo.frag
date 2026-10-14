#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uPrev;
uniform sampler2D uInput0;
uniform vec2 uResolution;
uniform float uTime;
uniform int uFrame;

uniform float uFeedbackMix;
uniform float uSliceCount;
uniform float uSliceDrift;
uniform float uOffset;
uniform float uSmear;
uniform float uColorSplit;
uniform float uStrobe;
uniform float uUseInput;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 sample_split(sampler2D tex, vec2 uv, vec2 off) {
    return vec3(texture(tex, uv + off).r, texture(tex, uv).g, texture(tex, uv - off).b);
}

void main() {
    vec2 uv = v_uv;
    float slices = max(1.0, floor(uSliceCount));
    float id = floor(uv.y * slices);
    float rnd = hash(vec2(id, floor(uTime * 8.0)));
    float phase = id / slices + uTime * uSliceDrift;
    float wave = sin(phase * 6.2831853) * 0.5 + 0.5;
    float shift = (rnd - 0.5) * uOffset * 0.18 + (wave - 0.5) * uOffset * 0.10;

    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 0.0);
    vec2 src_uv = uv + vec2(shift, 0.0);
    vec2 echo_uv = uv - vec2(shift * (0.35 + uSmear), 0.0);
    echo_uv.y += sin(uv.x * 18.0 + uTime) * uSmear * 0.01;

    vec3 src = sample_split(uInput0, src_uv, px * uColorSplit * 8.0);
    vec3 prev = sample_split(uPrev, echo_uv, px * uColorSplit * 4.0);
    vec3 col = mix(prev, src, clamp(uUseInput, 0.0, 1.0));
    col = mix(col, prev, clamp(uFeedbackMix + uSmear * 0.35, 0.0, 0.98));

    float edge = step(0.965, fract(uv.y * slices));
    col += edge * vec3(0.08, 0.12, 0.16);
    float gate = mix(1.0, step(0.18, hash(vec2(float(uFrame), id))), clamp(uStrobe, 0.0, 1.0));
    fragColor = vec4(clamp(col * gate, 0.0, 1.0), 1.0);
}
