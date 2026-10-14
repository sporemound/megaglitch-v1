# MegaGlitch Production Quality Contract

**Status**: Frozen Architecture Contract  
**Version**: 1.0  
**Date**: August 2026  
**Target Engine**: MegaGlitch Engine V1 / V2 / V3  

---

## 1. Scope & Intent

MegaGlitch generates intentional, structured analog and digital glitch art. Traditional video metrics (such as MSE or PSNR compared to an uncorrupted source) are inappropriate for measuring glitch art because they penalize the very aesthetic transformations the user intends to create.

This Quality Contract establishes the **objective aesthetic and structural rules** that distinguish authentic MegaGlitch output from corrupted implementations (e.g. low-resolution upscaling blur, blown-out recursive feedback, or unintended frame blend smoothing).

---

## 2. Invariant Classification

### 2.1 MUST Preserve (Non-Negotiable Production Invariants)
1. **Master-Resolution Ingress/Egress**: All processing, temporal history, and final render output must occur at full master resolution (e.g. 1920x1080). Intermediate frame dimensions must never be silently downscaled to preview dimensions and upscaled.
2. **Spatial Contrast & Edge Preservation**: Source structures outside glitch-modulated regions must retain full native sharpness and high-frequency edge energy.
3. **Structured Directional Corruption**: Glitch transformations (P-frame macroblock displacements, spiral/wave motion vector corruption, mesh tearing) must maintain crisp geometric block/grid alignment without diffuse blurring.
4. **Bounded Recursive Feedback**: Temporal feedback must remain strictly bounded:
   - Must implement exponential decay ($\le 0.95$).
   - Must enforce clean source floor ($\ge 0.10$).
   - Must trigger automatic feedback reset on scene cuts to prevent persistent ghost smears.
5. **Ordered Pipeline Restoration**: Structural restoration and edge reconstruction must execute *after* destructive glitch operations, restoring contrast headroom without introducing clipping halos.
6. **Temporal Discontinuity**: Intentional glitch transitions and temporal cuts must not be softened by accidental frame interpolation or inter-frame blend smoothing.

### 2.2 SHOULD Preserve (Aesthetic Fidelity Targets)
1. **Multimodal Spatial Selection**: When active, modulation masks should localize glitch intensity according to motion vectors, edge contours, and local entropy rather than acting as a flat global opacity multiplier.
2. **Luminance Channel Headroom**: Color grading and luma lift must preserve headroom without blowing out white levels ($> 255$) or crushing shadow details.
3. **Chromatic Channel Independence**: Channel flow offsets and chromatic aberrations should cleanly separate RGB color planes along defined directional axes.

### 2.3 AESTHETICALLY VARIABLE (Permitted Provider Divergence)
The following numerical variations are explicitly permitted between the CPU Reference implementation and Accelerated Production (CUDA / Native) implementations:
1. **Sub-Pixel Optical Flow Intermediates**: Floating-point motion estimation vectors produced by CUDA Farneback or GPU pyramids may differ from single-thread CPU Farneback.
2. **PRNG Sequence Initialization**: Parallel GPU pseudorandom noise generation (e.g. CUDA cuRAND) produces different spatial distributions than single-threaded CPU MT19937, provided noise variance and distribution type match.
3. **1-LSB Color Rounding**: Single least-significant-bit rounding variations resulting from fused multiply-add (FMA) instructions or hardware texture sampling.

### 2.4 QA-ONLY / Implementation Artifacts (Do NOT Charge Against Production)
1. Single-threaded CPU execution pinning.
2. Per-frame synchronous SHA-256 cryptographic hashing.
3. Strict bit-identical IEEE-754 floating-point order-of-operations matching against Python/NumPy loops.

---

## 3. Perceptual Quality Verification Metrics

To validate accelerated production pipelines against the Headless Quality Reference (Benchmark B), the test harness evaluates:

| Metric | Target / Range | Purpose |
| :--- | :--- | :--- |
| **SSIM (Structural Similarity)** | $\ge 0.85$ (across non-stochastic nodes) | Verifies geometric and luminance structure |
| **High-Frequency Spectral Energy** | Retention within $\pm 5\%$ of reference | Proves absence of unintended low-pass blurring |
| **Edge Magnitude Distribution** | Histogram correlation $\ge 0.90$ | Validates edge reconstruction and sharpness |
| **Temporal Delta Kurtosis** | Matches within $\pm 8\%$ | Proves preservation of sharp glitch transitions |
| **Dynamic Range Utilization** | Full $[0, 255]$ without hard clipping | Verifies color quantization and headroom |
