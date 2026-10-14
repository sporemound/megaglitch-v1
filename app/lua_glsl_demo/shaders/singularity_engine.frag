#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D uInput0;
uniform sampler2D uPrev;
// Future host extension points for true delayed wave buffers:
// uniform sampler2D uDelay1;
// uniform sampler2D uDelay2;
// uniform sampler2D uDelay3;
// uniform sampler2D uDelay4;

uniform vec2 uResolution;
uniform float uTime;
uniform int uFrame;

uniform float pixel_size;
uniform float palette_levels;
uniform float dither_strength;
uniform float posterize_amount;
uniform float iso_slant;
uniform float iso_depth;
uniform float cascade_angle;
uniform float perspective_compression;
uniform float swell_strength;
uniform float chop_strength;
uniform float wave_speed;
uniform float cross_current;
uniform float turbulence_amount;
uniform float surge_amount;
uniform float fall_line_position;
uniform float fall_line_curve;
uniform float fall_strength;
uniform float cascade_speed;
uniform float cascade_stretch;
uniform float sheet_breakup;
uniform float foam_threshold;
uniform float foam_edge_strength;
uniform float foam_opacity;
uniform float foam_breakup;
uniform float foam_scroll_speed;
uniform float band_count;
uniform float band_thickness;
uniform float band_spacing;
uniform float band_speed_variation;
uniform float band_breakup;
uniform float band_source_offset;
uniform float undertow_strength;
uniform float undertow_noise;
uniform float edge_pull_amount;
uniform float current_direction;
uniform float feedback_wave_echo;
uniform float old_wave_opacity;
uniform float echo_decay;
uniform float cascade_feedback_strength;
uniform float debris_amount;
uniform float tile_dropout;
uniform float streak_length;
uniform float collapse_darkening;
uniform float collapse_pixel_size;

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float luminance(vec3 c) {
    return dot(c, vec3(0.299, 0.587, 0.114));
}

vec3 video(vec2 uv) {
    return texture(uInput0, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
}

vec3 previous(vec2 uv) {
    return texture(uPrev, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
}

vec2 pixelSnapUV(vec2 uv) {
    float ps = max(1.0, pixel_size);
    vec2 p = uv * uResolution;
    p = floor(p / ps) * ps + ps * 0.5;
    return p / uResolution;
}

float orderedDither(vec2 uv) {
    vec2 p = floor(uv * uResolution / max(pixel_size, 1.0));
    float x = mod(p.x, 4.0);
    float y = mod(p.y, 4.0);
    float index = x + y * 4.0;
    float threshold = 0.0;

    if (index < 0.5) threshold = 0.00;
    else if (index < 1.5) threshold = 0.50;
    else if (index < 2.5) threshold = 0.125;
    else if (index < 3.5) threshold = 0.625;
    else if (index < 4.5) threshold = 0.75;
    else if (index < 5.5) threshold = 0.25;
    else if (index < 6.5) threshold = 0.875;
    else if (index < 7.5) threshold = 0.375;
    else if (index < 8.5) threshold = 0.1875;
    else if (index < 9.5) threshold = 0.6875;
    else if (index < 10.5) threshold = 0.0625;
    else if (index < 11.5) threshold = 0.5625;
    else if (index < 12.5) threshold = 0.9375;
    else if (index < 13.5) threshold = 0.4375;
    else if (index < 14.5) threshold = 0.8125;
    else threshold = 0.3125;

    return threshold - 0.5;
}

vec3 posterize16bit(vec3 color) {
    vec3 dithered = color + orderedDither(v_uv) * dither_strength;
    float levels = max(2.0, palette_levels);
    vec3 quantized = floor(clamp(dithered, 0.0, 1.0) * levels) / max(1.0, levels - 1.0);
    return mix(color, quantized, clamp(posterize_amount, 0.0, 1.0));
}

vec2 isoCoords(vec2 uv) {
    vec2 centered = uv - 0.5;
    centered.y *= mix(1.0, 0.62, perspective_compression);
    float isoX = centered.x + centered.y * iso_slant;
    float isoY = centered.y - centered.x * iso_depth;
    return vec2(isoX, isoY);
}

float waveField(vec2 iso, float time) {
    float t = time * wave_speed;
    float swell =
        sin(iso.x * 11.0 + t * 0.80) * 0.46 +
        sin(iso.x * 6.5 + iso.y * 4.0 - t * 0.55) * 0.34 +
        sin(iso.x * 19.0 - iso.y * 8.0 + t * 1.25) * 0.20;

    float chop =
        sin(iso.x * 53.0 + iso.y * 13.0 + t * 3.1) *
        sin(iso.y * 39.0 - t * 2.2);

    vec2 cell = floor((iso + vec2(t * 0.025, -t * 0.018)) * 18.0);
    float blockNoise = hash12(cell) * 2.0 - 1.0;
    float surge = floor((sin(t * 0.75) * 0.5 + 0.5) * 5.0) / 5.0;

    return swell * swell_strength +
           chop * chop_strength +
           blockNoise * turbulence_amount +
           (surge - 0.5) * surge_amount * 0.05;
}

float fallMask(vec2 iso) {
    float curve =
        sin(iso.x * 5.0 + uTime * 0.35 * cascade_speed) * fall_line_curve +
        sin(iso.x * 13.0 - uTime * 0.60 * cascade_speed) * fall_line_curve * 0.35;
    float lip = fall_line_position - 0.5 + curve;
    return smoothstep(-0.08, 0.18, iso.y - lip);
}

float lipMask(vec2 iso) {
    float curve =
        sin(iso.x * 5.0 + uTime * 0.35 * cascade_speed) * fall_line_curve +
        sin(iso.x * 13.0 - uTime * 0.60 * cascade_speed) * fall_line_curve * 0.35;
    float lip = fall_line_position - 0.5 + curve;
    return 1.0 - smoothstep(0.006, 0.060, abs(iso.y - lip));
}

vec2 cascadeUV(vec2 uv, vec2 iso, float fallAmount) {
    float waves = waveField(iso, uTime);
    vec2 fallDir = normalize(vec2(cascade_angle, 1.0));
    float steppedFall = floor(fallAmount * 8.0) / 8.0;
    float sheet = hash12(floor(vec2(iso.x * 28.0, iso.y * 8.0)));
    float cascade =
        steppedFall * fall_strength +
        waves * 0.45 +
        sin(iso.x * 40.0 + uTime * cascade_speed * 5.0) * cascade_stretch * 0.05 * fallAmount;

    vec2 outUV = uv - fallDir * cascade;
    outUV.y -= sheet * sheet_breakup * 0.08 * fallAmount;
    outUV.x += cross_current * fallAmount;
    return outUV;
}

float edgeStrength(vec2 uv) {
    vec2 texel = 1.0 / max(uResolution, vec2(1.0));
    float left = luminance(video(uv - vec2(texel.x, 0.0)));
    float right = luminance(video(uv + vec2(texel.x, 0.0)));
    float up = luminance(video(uv - vec2(0.0, texel.y)));
    float down = luminance(video(uv + vec2(0.0, texel.y)));
    return clamp(length(vec2(right - left, down - up)) * 3.0, 0.0, 1.0);
}

float localContrast(vec2 uv) {
    vec2 texel = 1.0 / max(uResolution, vec2(1.0));
    float c = luminance(video(uv));
    float avg = (
        luminance(video(uv - vec2(texel.x, 0.0))) +
        luminance(video(uv + vec2(texel.x, 0.0))) +
        luminance(video(uv - vec2(0.0, texel.y))) +
        luminance(video(uv + vec2(0.0, texel.y)))
    ) * 0.25;
    return clamp(abs(c - avg) * 3.5, 0.0, 1.0);
}

float foamMask(vec2 uv, vec2 iso, float fallAmount) {
    float lip = lipMask(iso);
    float edge = edgeStrength(uv);
    float contrast = localContrast(uv);
    float lum = luminance(video(uv));
    float crest = smoothstep(0.015, 0.07, abs(waveField(iso, uTime)));
    vec2 tile = floor(uv * uResolution / max(pixel_size, 1.0));
    float breakup = hash12(tile + floor(uTime * foam_scroll_speed * 8.0));

    float featureFoam = edge * foam_edge_strength * 0.65 + contrast * 1.45 + lum * 0.12;
    float mask = featureFoam + lip * 0.75 + crest * 0.22 + fallAmount * 0.12;
    mask = smoothstep(foam_threshold, foam_threshold + 0.25, mask);
    mask *= smoothstep(foam_breakup, 1.0, breakup + 0.35);
    return clamp(mask * foam_opacity, 0.0, 1.0);
}

vec3 sampleWaveBand(vec2 uv, vec2 iso, int bandIndex) {
    float fi = float(bandIndex);
    float speed = 0.14 + fi * 0.09 * max(0.05, band_speed_variation);
    float spacing = band_spacing + fi * 2.8;
    float phase = iso.y * spacing + iso.x * (2.0 + fi) + uTime * speed * 5.0;
    float band = abs(fract(phase) - 0.5);
    float bandMask = 1.0 - smoothstep(0.0, max(0.02, band_thickness * 0.5), band);

    vec2 block = floor((uv * uResolution) / (max(pixel_size, 1.0) * (2.0 + fi)));
    float gate = hash12(block + vec2(fi * 17.0, floor(uTime * (2.0 + fi))));
    bandMask *= step(band_breakup, gate);

    vec2 carry = vec2(-0.05 + current_direction * 0.04, 0.09) * fract(uTime * speed + fi * 0.13);
    vec2 bandOffset = vec2(
        sin(uTime * speed + fi * 2.1) * band_source_offset,
        cos(uTime * speed * 0.7 + fi * 1.7) * band_source_offset * 0.55
    ) + carry;
    vec2 sampleUV = uv + bandOffset;

    float split = (0.0025 + fi * 0.0013) * max(1.0, pixel_size * 0.35);
    vec3 c = vec3(
        video(sampleUV + vec2(split, -split)).r,
        video(sampleUV).g,
        video(sampleUV + vec2(-split, split)).b
    );
    float feature = clamp(edgeStrength(sampleUV) * 0.55 + localContrast(sampleUV) * 0.70 + luminance(c) * 0.20, 0.0, 1.0);
    return posterize16bit(c) * bandMask * feature;
}

vec3 applyUndertow(vec2 uv, vec2 iso) {
    float fallAmount = fallMask(iso);
    float w = waveField(iso, uTime);
    vec2 tile = floor(uv * uResolution / max(pixel_size * 3.0, 1.0));
    float n = hash12(tile + floor(uTime * 6.0)) * 2.0 - 1.0;
    vec2 current = vec2(
        sin(iso.y * 9.0 + uTime * 0.60),
        cos(iso.x * 11.0 - uTime * 0.50)
    );
    current += vec2(w * 4.0 + n * undertow_noise, -abs(w) * 2.0);
    current += normalize(vec2(current_direction, 1.0)) * edgeStrength(uv) * edge_pull_amount;
    vec2 undertowUV = uv + current * undertow_strength * (0.35 + fallAmount);
    return video(pixelSnapUV(undertowUV));
}

vec3 applyPixelDebris(vec2 uv, vec2 iso, float fallAmount) {
    vec2 tile = floor(uv * uResolution / max(pixel_size * collapse_pixel_size, 1.0));
    float n = hash12(tile + floor(uTime * 10.0));
    float dropout = step(n, tile_dropout * fallAmount * debris_amount);

    vec2 chunkUV = uv;
    chunkUV.y -= (n * streak_length + 0.025) * fallAmount;
    chunkUV.x += (hash12(tile + 7.0) - 0.5) * 0.08 * fallAmount;

    vec3 chunk = video(pixelSnapUV(chunkUV));
    chunk = posterize16bit(chunk);
    chunk *= mix(1.0, 1.0 - collapse_darkening, fallAmount);
    return chunk * dropout;
}

vec3 finalRetroPass(vec3 color, vec2 uv) {
    vec3 outColor = posterize16bit(color);
    float row = floor(uv.y * uResolution.y / max(pixel_size, 1.0));
    float scan = mix(0.92, 1.05, mod(row, 2.0));
    outColor *= scan;
    float checker = orderedDither(uv) * dither_strength * 0.35;
    outColor += checker;
    return clamp(outColor, 0.0, 1.0);
}

void main() {
    vec2 uv = v_uv;
    vec2 snappedUV = pixelSnapUV(uv);
    vec2 iso = isoCoords(snappedUV);
    float fallAmount = fallMask(iso);

    float waves = waveField(iso, uTime);
    vec2 waveUV = snappedUV;
    waveUV.x += waves * 0.75;
    waveUV.y += sin(iso.x * 18.0 + uTime * 1.1 * wave_speed) * swell_strength * 0.35;

    vec3 undertowColor = applyUndertow(waveUV, iso);
    vec2 cascadedUV = cascadeUV(waveUV, iso, fallAmount);
    vec3 base = video(pixelSnapUV(cascadedUV));
    base = mix(base, undertowColor, undertow_strength * (0.4 + fallAmount * 0.5));
    base = posterize16bit(base);

    vec3 debris = applyPixelDebris(snappedUV, iso, fallAmount);
    float debrisMask = clamp(max(max(debris.r, debris.g), debris.b) * debris_amount, 0.0, 1.0);
    base = mix(base, debris, debrisMask);

    vec3 bands = vec3(0.0);
    float activeBands = clamp(floor(band_count + 0.5), 1.0, 6.0);
    for (int i = 0; i < 6; ++i) {
        if (float(i) >= activeBands) {
            continue;
        }
        bands += sampleWaveBand(snappedUV, iso, i) * (0.58 - float(i) * 0.055);
    }
    float bandZone = smoothstep(0.05, 0.78, fallAmount);
    base = mix(base, base * 0.55 + bands, bandZone * debris_amount * 0.55);

    vec3 prev = previous(pixelSnapUV(cascadedUV + vec2(waves * 0.3, -fallAmount * 0.04)));
    prev = posterize16bit(prev) * mix(1.0, echo_decay, fallAmount);
    base = mix(base, prev, feedback_wave_echo * old_wave_opacity * fallAmount + cascade_feedback_strength * fallAmount * 0.12);

    float foam = foamMask(snappedUV, iso, fallAmount);
    vec3 foamColor = vec3(0.92, 0.96, 0.88) * (0.82 + 0.18 * hash12(floor(snappedUV * uResolution / max(pixel_size, 1.0))));
    foamColor = posterize16bit(foamColor);
    base = mix(base, foamColor, foam);

    float collapse = smoothstep(0.55, 1.0, fallAmount);
    base *= mix(1.0, 1.0 - collapse_darkening, collapse);

    float shadowBand = sin(iso.y * 45.0 + iso.x * 8.0 + uTime * 2.2 * cascade_speed);
    shadowBand = smoothstep(0.35, 0.9, shadowBand) * fallAmount;
    base *= 1.0 - shadowBand * 0.18;

    fragColor = vec4(finalRetroPass(base, uv), 1.0);
}
