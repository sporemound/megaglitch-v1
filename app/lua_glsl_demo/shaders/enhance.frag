#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uInput0;
uniform sampler2D uPrev;
uniform vec2 uResolution;
uniform float uTime;

uniform float uZoom;
uniform float uPanX;
uniform float uPanY;
uniform float uSharpen;
uniform float uEdgeTrace;
uniform float uScanner;
uniform float uGrid;
uniform float uReadout;
uniform float uNoise;
uniform float uChromatic;
uniform float uFeedbackMix;
uniform float uAmber;
uniform float uCyan;
uniform float uContrast;
uniform float uExposure;
uniform float uFreezeBlend;
uniform float uPixelLock;
uniform float uVignette;

float hash(vec2 p) {
    p = fract(p * vec2(443.8975, 397.2973));
    p += dot(p, p + 19.19);
    return fract(p.x * p.y);
}

vec3 sample_chroma(sampler2D tex, vec2 uv, float amount) {
    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 1.0 / max(uResolution.y, 1.0));
    vec2 off = px * amount * vec2(5.0, 1.7);
    float r = texture(tex, uv + off).r;
    float g = texture(tex, uv).g;
    float b = texture(tex, uv - off).b;
    return vec3(r, g, b);
}

vec3 sharpen_sample(vec2 uv) {
    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 1.0 / max(uResolution.y, 1.0));
    vec3 c = sample_chroma(uInput0, uv, uChromatic);
    vec3 n = sample_chroma(uInput0, uv + vec2(0.0, px.y), uChromatic);
    vec3 s = sample_chroma(uInput0, uv - vec2(0.0, px.y), uChromatic);
    vec3 e = sample_chroma(uInput0, uv + vec2(px.x, 0.0), uChromatic);
    vec3 w = sample_chroma(uInput0, uv - vec2(px.x, 0.0), uChromatic);
    return c + (c * 4.0 - n - s - e - w) * uSharpen;
}

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float edge_at(vec2 uv) {
    vec2 px = vec2(1.0 / max(uResolution.x, 1.0), 1.0 / max(uResolution.y, 1.0));
    float l = luminance(texture(uInput0, uv - vec2(px.x, 0.0)).rgb);
    float r = luminance(texture(uInput0, uv + vec2(px.x, 0.0)).rgb);
    float d = luminance(texture(uInput0, uv - vec2(0.0, px.y)).rgb);
    float u = luminance(texture(uInput0, uv + vec2(0.0, px.y)).rgb);
    return clamp(length(vec2(r - l, u - d)) * 3.5, 0.0, 1.0);
}

vec2 enhance_uv(vec2 uv) {
    vec2 center = vec2(0.5 + uPanX, 0.5 + uPanY);
    vec2 p = uv - 0.5;
    float zoom = max(0.05, uZoom);
    return center + p / zoom;
}

float box_mask(vec2 uv, vec2 center, vec2 half_size, float feather) {
    vec2 d = abs(uv - center) - half_size;
    float outside = length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    return 1.0 - smoothstep(0.0, feather, outside);
}

void main() {
    vec2 uv = v_uv;
    vec2 euv = enhance_uv(uv);
    vec3 src = sharpen_sample(euv);
    vec3 prev = texture(uPrev, uv).rgb;
    vec3 col = mix(src, prev, clamp(uFeedbackMix, 0.0, 0.98));

    float edge = edge_at(euv);
    vec3 trace_color = mix(vec3(0.1, 0.9, 1.0), vec3(1.0, 0.62, 0.16), clamp(uAmber, 0.0, 1.0));
    col += trace_color * edge * uEdgeTrace;

    float scan_y = abs(fract((uv.y + uTime * 0.065) * 42.0) - 0.5);
    float scanline = 1.0 - smoothstep(0.0, 0.085, scan_y);
    col += vec3(0.2, 0.85, 1.0) * scanline * uScanner * 0.18;
    col *= 1.0 - uScanner * 0.10 * (0.5 + 0.5 * sin(uv.y * uResolution.y * 3.14159));

    vec2 grid_uv = fract((uv - 0.5) * vec2(28.0, 18.0));
    float grid_line = 1.0 - min(
        smoothstep(0.0, 0.035, min(grid_uv.x, 1.0 - grid_uv.x)),
        smoothstep(0.0, 0.035, min(grid_uv.y, 1.0 - grid_uv.y))
    );
    col += vec3(0.08, 0.55, 0.72) * grid_line * uGrid;

    float reticle = box_mask(uv, vec2(0.5), vec2(0.265, 0.18), 0.006);
    float reticle_edge = reticle - box_mask(uv, vec2(0.5), vec2(0.248, 0.163), 0.006);
    col += mix(vec3(0.0, 0.72, 1.0), vec3(1.0, 0.58, 0.10), uAmber) * reticle_edge * uReadout;

    if (uPixelLock > 0.001) {
        vec2 block_count = vec2(96.0, 54.0);
        vec2 cell = floor(uv * block_count);
        float pulse = step(0.985 - uPixelLock * 0.08, hash(cell + floor(uTime * 8.0)));
        col = mix(col, vec3(0.98, 0.68, 0.20), pulse * uPixelLock * 0.45);
    }

    float grain = hash(uv * uResolution + vec2(uTime * 23.7, -uTime * 11.1));
    col += (grain - 0.5) * uNoise * 0.18;

    vec3 amber_tint = vec3(1.0, 0.58, 0.18);
    vec3 cyan_tint = vec3(0.18, 0.78, 1.0);
    col = mix(col, col * amber_tint, clamp(uAmber, 0.0, 1.0) * 0.55);
    col = mix(col, max(col, cyan_tint * luminance(col)), clamp(uCyan, 0.0, 1.0) * 0.38);
    col *= max(0.0, uExposure);
    col = (col - 0.5) * max(0.0, uContrast) + 0.5;

    vec2 p = uv * 2.0 - 1.0;
    p.x *= uResolution.x / max(uResolution.y, 1.0);
    float vig = smoothstep(1.45, 0.24, dot(p, p));
    col *= mix(1.0 - uVignette * 0.65, 1.0, vig);

    col = mix(col, prev, clamp(uFreezeBlend, 0.0, 1.0) * 0.5);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
