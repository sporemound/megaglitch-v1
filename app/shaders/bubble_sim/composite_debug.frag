#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_bubble_mask;
uniform sampler2D u_bubble_normal;
uniform sampler2D u_bubble_depth;
uniform sampler2D u_bubble_refraction;
uniform sampler2D u_bubble_highlight;
uniform float u_debug_mode;

void main() {
    vec4 mask = texture(u_bubble_mask, v_uv);
    vec4 normal = texture(u_bubble_normal, v_uv);
    vec4 depth = texture(u_bubble_depth, v_uv);
    vec4 refract = texture(u_bubble_refraction, v_uv);
    vec4 highlight = texture(u_bubble_highlight, v_uv);

    if (u_debug_mode < 0.5) {
        fragColor = vec4(mask.rrr + highlight.rgb, 1.0);
    } else if (u_debug_mode < 1.5) {
        fragColor = vec4(normal.rgb, 1.0);
    } else if (u_debug_mode < 2.5) {
        fragColor = vec4(depth.r, depth.g, depth.b, 1.0);
    } else {
        fragColor = vec4(refract.rg, mask.r, 1.0);
    }
}

