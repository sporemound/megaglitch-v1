#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uInput0;
uniform sampler2D uPrev;
uniform vec2 uResolution;
uniform float uTime;
uniform int uFrame;

uniform float uBlockSize;
uniform float uDisplace;
uniform float uFreezeMix;
uniform float uCorrupt;
uniform float uColorBleed;
uniform float uQuantize;
uniform float uUseInput;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 sample_bleed(sampler2D tex, vec2 uv, float amount) {
    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 0.0) * amount * 12.0;
    return vec3(texture(tex, uv + px).r, texture(tex, uv).g, texture(tex, uv - px).b);
}

void main() {
    float bs = max(2.0, floor(uBlockSize));
    vec2 block = floor(v_uv * uResolution / bs);
    vec2 cell_uv = (block * bs + bs * 0.5) / uResolution;
    float rnd = hash(block + floor(uTime * 8.0));
    float rnd2 = hash(block.yx + vec2(float(uFrame) * 0.07, 19.0));
    vec2 shift = vec2(rnd - 0.5, rnd2 - 0.5) * uDisplace * bs / uResolution;
    shift *= step(1.0 - clamp(uCorrupt, 0.0, 1.0), rnd);

    vec2 uv = v_uv + shift;
    vec3 src = sample_bleed(uInput0, uv, uColorBleed);
    vec3 prev = texture(uPrev, uv - shift * 0.35).rgb;
    vec3 col = mix(prev, src, clamp(uUseInput, 0.0, 1.0));
    col = mix(col, prev, clamp(uFreezeMix + step(0.92, rnd) * uCorrupt * 0.45, 0.0, 0.98));

    if (uQuantize > 1.0) {
        float q = floor(uQuantize);
        col = floor(col * q) / max(q - 1.0, 1.0);
    }
    float grid = step(0.985, fract(v_uv.x * uResolution.x / bs)) + step(0.985, fract(v_uv.y * uResolution.y / bs));
    col += grid * uCorrupt * 0.04;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
