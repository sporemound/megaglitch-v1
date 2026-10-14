# MegaGlitch Benchmark Contract & Measurement Taxonomy

**Status**: Frozen Architecture Contract  
**Version**: 1.0 (Post-Drift Reconciliation)  
**Date**: August 2026  
**Target Engine**: MegaGlitch Engine V1 / V2 / V3  

---

## 1. Executive Summary: The Three-Contract Framework

To prevent benchmark drift and optimization confusion, MegaGlitch permanently divides all performance and correctness measurement into **three distinct, isolated benchmark classes**:

`	ext
+-------------------------------------------------------------------------------+
|                       MEGAGLITCH BENCHMARK TAXONOMY                           |
+-------------------------------------------------------------------------------+
|                                                                               |
|  [ BENCHMARK A: REFERENCE QA ]                                                |
|  - Role: Correctness Oracle & Regression Gate                                 |
|  - Execution: Deterministic Python/OpenCV CPU, 10-node full coverage graph    |
|  - Pass Criteria: Exact bit-for-bit SHA-256 digest match (43db5c51...)        |
|  - Performance Target: NONE (Diagnostic wall-clock tracking only)            |
|                                                                               |
|  [ BENCHMARK B: HEADLESS QUALITY REFERENCE ]                                  |
|  - Role: Aesthetic Oracle & Raw Engine Throughput                             |
|  - Execution: Headless Master Renderer -> Direct FFmpeg / Media Sink          |
|  - Isolation: NO UI, NO bridge, NO preview, NO IPC, NO live checksums         |
|  - Pass Criteria: Visual quality contract, perceptual fidelity, raw speed     |
|  - Performance Target: Offline export throughput & acceleration ratio         |
|                                                                               |
|  [ BENCHMARK C: PRODUCTION LIVE WORKSTATION ]                                 |
|  - Role: Real-Time User Experience & Interactive Performance                  |
|  - Execution: Native/CUDA accelerated graph + Downstream Preview Surface      |
|  - Isolation: Bounded queues, lossy preview, zero-copy GPU display            |
|  - Pass Criteria: Interactive responsiveness, temporal continuity, no tearing |
|  - Performance Target: >= 30-60 FPS live presentation (< 33 ms latency)      |
|                                                                               |
+-------------------------------------------------------------------------------+
`

> [!IMPORTANT]
> **Cardinal Invariant**: Benchmark A (Reference QA) is strictly a correctness instrument. Its execution time must **NEVER** be used as the performance baseline against which live interactive throughput targets (e.g. 55 FPS) are evaluated.

---

## 2. Benchmark A: Reference QA (Stress & Correctness Oracle)

### 2.1 Purpose
Benchmark A guarantees architectural integrity, master-resolution boundary enforcement, temporal buffer safety, frame scheduling determinism, and algorithm stability across releases.

### 2.2 Workload Definition
- **Resolution**: 1920x1080 RGB8 (Strictly master resolution; zero scaling).
- **Graph**: 10-node Quality-First Reference Preset (megaglitch.quality_graph.build_quality_first_graph).
- **Input Source**: 30 deterministic synthetic RGB8 frames (seed=917).
- **OpenCV Environment**: Single-thread deterministic CPU pinning (cv2.setNumThreads(1), cv2.ocl.setUseOpenCL(False)).
- **Temporal Depth**: 12-frame master-resolution history buffer.
- **Egress Sink**: Memory Hash Sink (measures raw master output stream).

### 2.3 Correctness Gates
1. **Master Integrity**: Every ingress frame, intermediate tensor, temporal history slot, and output buffer must be strictly (1080, 1920, 3) uint8 / float32.
2. **Deterministic Digest**: Output stream must match the canonical 30-frame SHA-256:
   `	ext
   43db5c512a3c44ba01fd127e0cca50376cc734d44d46303c894c6b948d010507
   `
3. **Zero Drops / Duplications**: In 	rue_accelerated mode, exactly 30 unique source frames are processed and emitted.

### 2.4 Performance Policy
- **Target**: **NONE** (Diagnostic only).
- **Current Baseline**: ~2,689 ms/frame (~0.372 FPS).
- **Optimization Rule**: No optimization may be applied to Benchmark A if it alters float32/uint8 rounding ties, changes RNG stream sequence, or weakens the 43db5c51... hash lock.

---

## 3. Benchmark B: Headless Quality Reference (Aesthetic Oracle)

### 3.1 Purpose
Benchmark B measures the pure rendering throughput and aesthetic fidelity of the visual engine in headless mode (e.g. CLI offline mastering, batch export, background rendering) without workstation overhead.

### 3.2 Workload Definition
- **Execution Path**: Source Ingress -> Master Graph Execution -> Direct Codec / Memory Sink.
- **Complete Subsystem Isolation**:
  - **NO** Dear PyGui / Rust UI presentation.
  - **NO** Python <-> Rust bridge IPC or JSON serialization.
  - **NO** Downstream preview generation or slot leasing.
  - **NO** OBS Studio or Virtual Camera frame submission.
  - **NO** Per-frame live SHA-256 calculation.
- **Input Sources**:
  1. Deterministic Synthetic Ingress (for reproducible profiling).
  2. Representative Real Video Footage (e.g. 1080p music video footage).
- **Accelerators**: CUDA enabled with multi-threaded OpenCV (cv2.setNumThreads(2) or GPU-resident pipelines).

### 3.3 Recorded Metrics
To prevent metric conflation, all reports must separate:
1. **processing_fps_unique**: Unique Master Graph Evaluations / Wall-Clock Render Seconds (Real compute throughput).
2. **output_fps**: The timebase frame rate declared in the output container (e.g. 55.0 or 30.0).
3. **source_fps**: The native frame rate of the input media.
4. **wall_time_seconds**: Total elapsed wall-clock rendering duration.
5. **speed_multiplier**: Source Duration / Wall-Clock Duration (>1.0x indicates faster-than-realtime).

---

## 4. Benchmark C: Production Live (Workstation Runtime Target)

### 4.1 Purpose
Benchmark C measures the real-time interactive performance and stability experienced by end users operating the workstation application.

### 4.2 Workload Definition
- **Resolution**: User-configured master resolution (e.g. 1080p, 720p, or 1200x800).
- **Graph**: Production Quality Graph V1 (active aesthetic preset).
- **Display Pipeline**: Master Graph -> Downstream Preview Surface (V2 packed mmap / V3 shared ring) -> UI Viewport Texture Upload -> Screen Present.
- **Scheduling**: Real-time media clock with bounded queues and lossy preview drops (preview stalls never backpressure the master engine).

### 4.3 Correctness & Acceptance Standard
- **Aesthetic Fidelity**: Visual output is evaluated against the Benchmark B Quality Contract using perceptual metrics (PSNR, SSIM, edge retention, motion coherence), **not** byte-for-byte SHA matching against CPU reference code.
- **Interactive Throughput Target**:
  - **Minimum**: >= 30 FPS live presentation (<= 33.3 ms/frame).
  - **Standard**: >= 55-60 FPS live presentation (<= 18.18 ms/frame).
  - **Interactive Latency**: <= 2 frames end-to-end input-to-display delay.

---

## 5. Summary Matrix of Benchmark Responsibilities

| Attribute | Benchmark A (Reference QA) | Benchmark B (Headless Reference) | Benchmark C (Production Live) |
| :--- | :--- | :--- | :--- |
| **Primary Goal** | Architectural Correctness | Aesthetic Fidelity & Export Speed | Interactive Real-Time UX |
| **Execution Mode** | Headless Offline | Headless CLI / Batch | Interactive Workstation UI |
| **Graph** | 10-node Quality Reference | Production / Offline Preset | Production / Interactive Preset |
| **Provider** | Deterministic CPU (Pinned) | Multi-Thread CPU / CUDA | CUDA / Native Accelerated |
| **Preview Active** | **NO** | **NO** | **YES** (Downstream Lossy) |
| **UI Active** | **NO** | **NO** | **YES** (wgpu / Native) |
| **Correctness Gate** | Bit-Exact SHA-256 | Perceptual Quality Contract | Perceptual Quality & Smoothness |
| **Throughput Target** | None (Diagnostic) | Maximal Export Multiplier | >= 55 FPS (<= 18.18 ms) |
