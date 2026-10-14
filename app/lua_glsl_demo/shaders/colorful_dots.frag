#version 330 core

uniform vec2 uResolution;
uniform float uTime;

uniform float u_cluster_count;
uniform float u_dot_count;
uniform float u_cluster_start_scatter_min;
uniform float u_cluster_start_scatter_max;
uniform float u_cluster_end_scatter_min;
uniform float u_cluster_end_scatter_max;
uniform float u_cluster_speed;
uniform float u_cluster_ttl_min;
uniform float u_cluster_ttl_max;
uniform float u_dot_size_min;
uniform float u_dot_size_max;
uniform float u_movement_range;
uniform float u_cluster_center_acceleration;
uniform float u_base_color_r;
uniform float u_base_color_g;
uniform float u_base_color_b;
uniform float u_cluster_color_offset;
uniform float u_dot_color_offset;

out vec4 f_color;

const int MAX_CLUSTERS = 32;
const int MAX_DOTS = 512;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec2 hash2(vec2 p) {
    return fract(sin(vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}

vec3 hash3(vec2 p) {
    return fract(sin(vec3(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)), dot(p, vec2(419.2, 371.9)))) * 43758.5453);
}

int getClusterCount() {
    return int(clamp(u_cluster_count, 1.0, float(MAX_CLUSTERS)));
}

int getDotCount() {
    return int(clamp(u_dot_count, 1.0, float(MAX_DOTS)));
}

vec2 getStartScatterRange() {
    float minVal = max(0.0, u_cluster_start_scatter_min);
    float maxVal = max(minVal + 1e-4, u_cluster_start_scatter_max);
    return vec2(minVal, maxVal);
}

vec2 getEndScatterRange() {
    float minVal = max(0.0, u_cluster_end_scatter_min);
    float maxVal = max(minVal + 1e-4, u_cluster_end_scatter_max);
    return vec2(minVal, maxVal);
}

vec2 getTTLRange() {
    float minVal = max(0.05, u_cluster_ttl_min);
    float maxVal = max(minVal + 0.01, u_cluster_ttl_max);
    return vec2(minVal, maxVal);
}

float getClusterSpeed() {
    return max(u_cluster_speed, 0.0);
}

float getMovementRange() {
    return max(u_movement_range, 0.0);
}

float getCenterAcceleration() {
    return max(u_cluster_center_acceleration, 0.0);
}

vec3 getBaseColor() {
    return clamp(vec3(u_base_color_r, u_base_color_g, u_base_color_b), 0.0, 1.0);
}

float getClusterColorOffset() {
    return max(u_cluster_color_offset, 0.0);
}

float getDotColorOffset() {
    return max(u_dot_color_offset, 0.0);
}

float getDotSizeMin() {
    return max(0.0, u_dot_size_min);
}

float getDotSizeMax() {
    float minVal = getDotSizeMin();
    return max(minVal + 1e-4, u_dot_size_max);
}

float getMaxScatter() {
    vec2 startRange = getStartScatterRange();
    vec2 endRange = getEndScatterRange();
    return max(startRange.y, endRange.y);
}

struct Cluster {
    vec2 pos;
    float weight;
    float scatterRange;
    vec3 color;
    float ttl;
};

Cluster getCluster(float id) {
    Cluster c;

    vec2 seed = vec2(id, id * 1.7);
    vec2 basePos = hash2(seed);
    float timeOffset = hash(seed) * 6.28318;
    float clusterSpeed = getClusterSpeed();
    float movementRange = getMovementRange();
    vec2 randomDrift = vec2(sin(uTime * clusterSpeed + timeOffset), cos(uTime * clusterSpeed * 0.8 + timeOffset)) * movementRange;

    vec2 toCenter = vec2(0.5) - basePos;
    float centerForce = uTime * getCenterAcceleration();
    vec2 centerPull = toCenter * centerForce * 0.01;

    c.pos = basePos + randomDrift + centerPull;

    vec2 ttlRange = getTTLRange();
    float clusterLifetime = ttlRange.x + hash(seed + vec2(10.0)) * (ttlRange.y - ttlRange.x);
    float birthTime = hash(seed + vec2(11.0)) * 100.0;
    float age = mod(uTime - birthTime, clusterLifetime + 1.0);

    float ttlBase = 0.0;
    float fadeTime = min(0.3, clusterLifetime * 0.5);
    if (age <= clusterLifetime) {
        if (age < fadeTime) {
            ttlBase = age / max(fadeTime, 1e-4);
        } else if (age > clusterLifetime - fadeTime) {
            ttlBase = (clusterLifetime - age) / max(fadeTime, 1e-4);
        } else {
            ttlBase = 1.0;
        }
    }

    float scatterT = hash(seed + vec2(2.0));
    vec2 startRange = getStartScatterRange();
    vec2 endRange = getEndScatterRange();
    float lifeProgress = 1.0 - ttlBase;
    vec2 currentRange = mix(startRange, endRange, clamp(lifeProgress, 0.0, 1.0));
    c.scatterRange = currentRange.x + scatterT * (currentRange.y - currentRange.x);

    float maxScatter = getMaxScatter();
    vec2 minBounds = vec2(-maxScatter);
    vec2 maxBounds = vec2(1.0 + maxScatter);
    if (c.pos.x < minBounds.x || c.pos.x > maxBounds.x || c.pos.y < minBounds.y || c.pos.y > maxBounds.y) {
        ttlBase = 0.0;
    }

    c.ttl = ttlBase;
    c.weight = (0.1 + hash(seed + vec2(1.0)) * 0.9) * c.ttl;

    vec3 clusterOffset = (hash3(seed + vec2(3.0)) - 0.5) * 2.0 * getClusterColorOffset();
    c.color = getBaseColor() + clusterOffset;

    return c;
}

int selectCluster(float dotId, int clusterCount) {
    clusterCount = max(clusterCount, 1);
    float totalWeight = 0.0;
    for (int i = 0; i < MAX_CLUSTERS; ++i) {
        if (i >= clusterCount) break;
        totalWeight += getCluster(float(i)).weight;
    }
    if (totalWeight <= 0.0) {
        return max(clusterCount - 1, 0);
    }

    float threshold = hash(vec2(dotId, uTime * 0.01)) * totalWeight;
    float cumulative = 0.0;
    for (int i = 0; i < MAX_CLUSTERS; ++i) {
        if (i >= clusterCount) break;
        cumulative += getCluster(float(i)).weight;
        if (threshold <= cumulative) {
            return i;
        }
    }
    return max(clusterCount - 1, 0);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution.xy;
    vec3 color = vec3(0.0);

    int clusterCount = getClusterCount();
    int dotCount = getDotCount();
    float dotSizeMin = getDotSizeMin();
    float dotSizeMax = getDotSizeMax();
    float dotColorOffset = getDotColorOffset();

    for (int dotId = 0; dotId < MAX_DOTS; ++dotId) {
        if (dotId >= dotCount) break;

        int clusterId = selectCluster(float(dotId), clusterCount);
        Cluster cluster = getCluster(float(clusterId));

        if (cluster.ttl < 0.01) continue;

        vec2 dotSeed = vec2(float(dotId), uTime * 0.1);
        float angle = hash(dotSeed) * 6.28318;
        float radius = sqrt(hash(dotSeed + vec2(100.0))) * cluster.scatterRange;
        vec2 offset = vec2(cos(angle), sin(angle)) * radius;
        vec2 dotPos = cluster.pos + offset;

        if (length(offset) > cluster.scatterRange * 0.9) {
            angle = hash(dotSeed + vec2(200.0)) * 6.28318;
            radius = sqrt(hash(dotSeed + vec2(300.0))) * cluster.scatterRange * 0.7;
            offset = vec2(cos(angle), sin(angle)) * radius;
            dotPos = cluster.pos + offset;
        }

        float dist = length(uv - dotPos);
        float dotSize = dotSizeMin + hash(dotSeed + vec2(200.0)) * (dotSizeMax - dotSizeMin);

        if (dist < dotSize) {
            vec3 dotOffset = (hash3(dotSeed + vec2(300.0)) - 0.5) * 2.0 * dotColorOffset;
            vec3 dotColor = cluster.color + dotOffset;
            color += dotColor * cluster.ttl;
        }
    }

    f_color = vec4(color, 1.0);
}
