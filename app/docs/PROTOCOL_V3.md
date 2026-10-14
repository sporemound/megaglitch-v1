# MegaGlitch Protocol V3 Specification & Architecture Contract

**Status**: Approved Specification / Active Prototype  
**Date**: August 2026  
**Target Engine**: MegaGlitch Engine V2 / V3  
**Compatibility Layer**: Protocol V2 JSONL (Supported for Python development)

---

## 1. Executive Summary & Design Principles

MegaGlitch Protocol V3 modernizes the communication, frame publication, and resource sharing subsystems across UI, engine sidecar, and external sinks.

### Core Architectural Invariants
1. **Separation of Verification from Transport**: Cryptographic validation (SHA-256) is strictly a QA/benchmark gate and is never executed in the live frame hot path.
2. **Zero-Copy Hot Paths**: Frame data is transferred across process and API boundaries via lock-free shared memory rings (CPU) and shared NT resource handles (GPU), eliminating intermediate byte allocations, disk I/O, and per-frame control serialization.
3. **Lossy Preview / Lossless Master**: Preview transports must never stall, throttle, or synchronize master graph computation. If a consumer falls behind, intermediate preview slots are overwritten, and the consumer acquires the latest committed sequence.
4. **Adapter Identity Enforcement**: Multi-GPU systems must explicitly match the CUDA compute device and Direct3D 12 display device by physical Adapter LUID before enabling GPU-direct zero-copy paths.

---

## 2. Format Matrix & Plane Division

| Subsystem | Canonical Role | In-Memory / Wire Format | Transport Medium | Synchronization Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Master Compute** | Primary Render Graph | `RGB8` (packed `[H, W, 3]`) | GPU Device Memory / Host RAM | Stream Sequential / CUDA Events |
| **CPU Preview (V3)** | Low-overhead UI Fallback | `RGB8_PACKED` | Win32 Pagefile Named Shared Memory | 128-byte Seqlock Ring |
| **GPU UI Presentation** | Native Zero-Copy Viewport | `BGRA8_UNORM` / `RGBA8_UNORM` | D3D12 Committed Textures | D3D12 Shared Monitored Fence |
| **Virtual Camera** | DirectShow / Windows Media | `NV12` (Rec. 709) | DXGI / Media Foundation IMFMediaBuffer | Media Foundation Sample Allocator |
| **HDR Video Sink (Future)**| Wide-Gamut Master Egress | `P010` (10-bit YUV 4:2:0) | Direct3D 12 HDR Swapchain | DXGI Flip Model HDR Metadata |
| **Dev Control Plane** | Python Engine Sidecar | JSON-Lines (v2 schema) | Standard IO Pipes (Piped Stdio) | Monotonic Request/Ack Revisions |
| **Native Control Plane** | Packaged Engine Binary | Length-prefixed FlatBuffers (v3) | Named Pipes (`\\.\pipe\megaglitch-*`) | Atomic Message Framing |
| **Project & Presets** | Serialization / Storage | Canonical JSON | Local Filesystem / SQLite DB | Atomic File Replacement |
| **QA / Benchmark Determinism**| Regression Verification | Full-frame SHA-256 | Test Telemetry JSON / CSV | Explicit Diagnostic Opt-in |
| **Live Frame Checksum** | Production Runtime | **`NONE`** | Zero Overhead | N/A |

---

## 3. Control Plane V3 (FlatBuffers Binary Framing)

While Protocol V2 JSON-lines remains fully supported for Python-hosted sidecars, the native packaged runtime uses a binary control plane to eliminate text parsing overhead on transaction-heavy control paths.

### 3.1 Binary Frame Header
All control plane messages across the native pipe are framed with an 8-byte fixed header:

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       payload_length (u32)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      protocol_major (u16)     |       message_type (u16)      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     FlatBuffer Payload Bytes...               |
+---------------------------------------------------------------+
```

- `payload_length`: Little-endian `u32` byte count of following payload.
- `protocol_major`: Must equal `3`.
- `message_type`:
  - `0x0001`: `HandshakeReady`
  - `0x0002`: `CommandRequest` (`replace_graph`, `set_source`, `set_transport`, `seek`, `shutdown`)
  - `0x0003`: `CommandAck`
  - `0x0004`: `RuntimeSnapshot`
  - `0x0005`: `RuntimeError`

### 3.2 Transactional State Revisions
Every modifying command increments `state_revision`. If a command arrives with `expected_state_revision != current_state_revision`, the command is rejected with `state_revision_conflict`, preserving atomic deterministic state across the bridge.

---

## 4. Protocol V3 CPU Preview Ring Specification

**Status**: Experimental Prototype (Not yet production qualified for lock-free IPC)

The V3 CPU Preview Ring replaces filesystem-backed temporary files (`%TEMP%\megaglitch-preview-*.rgb8`) and per-frame JSON ACK exchanges with a zero-allocation Win32 named shared memory ring.

> [!NOTE]
> The current prototype proves binary layout, pagefile named shared memory, engine generation tracking, overwrite/drop semantics, and elimination of per-frame JSON ACKs. However, full production qualification for lock-free multi-process operation requires native C++/Rust 64-bit atomic publication/acquisition with explicit memory barriers (`acquire`/`release`) or a documented Windows synchronization primitive (e.g. named event / waitable timer).

### 4.1 Memory Mapping Backing
- **Windows Implementation**: `CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, total_bytes, tag_name)` (exposed in Python via `mmap.mmap(-1, total_bytes, tagname=tag_name)`).
- **Naming Scope**: `megaglitch_preview_v3_<generation>`.
- **Lifetime**: Created on engine generation startup; destroyed on engine shutdown.

### 4.2 Numeric Pixel Format Registry
```c
enum PixelFormat : uint32_t {
    PIXEL_FORMAT_UNKNOWN      = 0,
    PIXEL_FORMAT_RGB8_PACKED  = 1,
    PIXEL_FORMAT_RGBA8_UNORM  = 2,
    PIXEL_FORMAT_BGRA8_UNORM  = 3,
    PIXEL_FORMAT_NV12         = 4,
    PIXEL_FORMAT_P010         = 5
};
```

### 4.3 Binary Header Layouts (128-Byte Aligned)

#### Global Header (Offset `0x0000`, 128 Bytes)
| Byte Offset | Type | Field Name | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `char[8]` | `magic` | Magic identifier: `"MGNPVW03"` |
| `0x08` | `u32` | `protocol_version` | Must equal `3` |
| `0x0C` | `u32` | `header_size` | Size of global header (`128` bytes) |
| `0x10` | `u64` | `generation` | Unique engine lifecycle generation timestamp |
| `0x18` | `u32` | `pixel_format` | Canonical format (`PIXEL_FORMAT_RGB8_PACKED`) |
| `0x1C` | `u32` | `flags` | Bitmask flags (reserved) |
| `0x20` | `u32` | `max_width` | Maximum width capacity (`1280`) |
| `0x24` | `u32` | `max_height` | Maximum height capacity (`720`) |
| `0x28` | `u32` | `slot_count` | Number of ring slots (`3`) |
| `0x2C` | `u32` | `slot_header_size` | Size of slot header (`128` bytes) |
| `0x30` | `u64` | `slot_bytes` | Byte size per slot (header + max payload) |
| `0x38` | `u64` | `latest_sequence` | Atomic sequence of latest committed slot |
| `0x40` | `u64` | `published_frames`| Monotonic counter of published frames |
| `0x48` | `u64` | `overwritten_frames`| Monotonic counter of dropped frames |
| `0x50` | `u8[48]`| `reserved` | Cache-line padding |

#### Slot Header (Offset `128 + slot_idx * slot_bytes`, 128 Bytes)
| Byte Offset | Type | Field Name | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `u64` | `sequence` | Frame sequence number |
| `0x08` | `u64` | `timestamp_ns` | Monotonic clock timestamp (`time.monotonic_ns()`) |
| `0x10` | `u32` | `width` | Frame pixel width |
| `0x14` | `u32` | `height` | Frame pixel height |
| `0x18` | `u32` | `stride` | Row stride in bytes (`width * 3`) |
| `0x1C` | `u32` | `nbytes` | Total payload byte size (`height * stride`) |
| `0x20` | `u32` | `pixel_format` | Pixel format enum value |
| `0x24` | `u32` | `flags` | Frame flags |
| `0x28` | `u64` | `generation` | Generation matching global header |
| `0x30` | `u64` | `commit_seq` | Atomic commit sequence (equals `sequence` when ready) |
| `0x38` | `u8[72]`| `reserved` | Cache-line padding |

### 4.4 Lock-Free Publication & Consumption Protocol

```text
PRODUCER (Engine / Sidecar)                     CONSUMER (UI / Presenter)
--------------------------                     -------------------------
1. slot_idx = sequence % slot_count
2. Write slot header (commit_seq = 0)
3. Copy RGB8 buffer into payload offset
4. [Memory Barrier Release]
5. Write commit_seq = sequence
6. Write global latest_sequence = sequence
7. Increment published_frames
                                               1. Read global latest_sequence
                                               2. [Memory Barrier Acquire]
                                               3. Read slot header at (latest_sequence % slot_count)
                                               4. Verify gen == current_gen AND commit_seq == latest_sequence
                                               5. Copy / Upload payload bytes
                                               6. Re-verify commit_seq == latest_sequence
                                               7. (Frame accepted; no ACK packet sent)
```

---

## 5. GPU Zero-Copy V3 Architecture (Direct3D 12 / CUDA / wgpu)

For live workstation display at 55+ FPS without CPU readback, Protocol V3 specifies a direct GPU texture sharing model.

```text
+------------------------+
|  CUDA Pipeline Kernel  |
+------------------------+
            | (CUDA External Memory Access)
            v
+--------------------------------------------------------+
|  Triple-Buffered D3D12 Committed Resources (Shared NT) |
|  - Format: DXGI_FORMAT_B8G8R8A8_UNORM                  |
|  - Synchronization: ID3D12Fence (Shared NT Handle)     |
+--------------------------------------------------------+
            |
            v
+------------------------+
|   wgpu-hal DX12 Import |
+------------------------+
            |
            v
+------------------------+
|   eframe / UI Display  |
+------------------------+
```

### 5.1 Direct3D 12 Shared Resource Contract
1. **Resource Creation**:
   - Heap: `D3D12_HEAP_TYPE_DEFAULT`
   - Flags: `D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS`
   - Heap Flags: `D3D12_HEAP_FLAG_SHARED`
   - Dimensions: Master width x height (or downscaled preview target).
2. **NT Handle Sharing**:
   - Created via `ID3D12Device::CreateSharedHandle` with `GENERIC_ALL` security attributes.
   - Handled uniquely per slot generation.
3. **Synchronization**:
   - `ID3D12Fence` created with `D3D12_FENCE_FLAG_SHARED`.
   - Producer signals fence value `N` via CUDA stream upon kernel completion.
   - UI consumer waits on fence value `N` before sampling texture in `wgpu` render pass.

### 5.2 CUDA External Memory Interop
1. Import D3D12 texture NT handle via `cudaImportExternalMemory(&extMem, &memDesc)`.
2. Map mipmapped array via `cudaExternalMemoryGetMappedMipmappedArray(&mipArray, extMem, &arrayDesc)`.
3. Import fence NT handle via `cudaImportExternalSemaphore(&extSem, &semDesc)`.
4. Signal semaphore via `cudaSignalExternalSemaphoresAsync(&extSem, &params, 1, stream)`.

### 5.3 Adapter Matching Invariant
Zero-copy texture sharing requires physical GPU parity:
- The UI queries `DXGI_ADAPTER_DESC::AdapterLuid`.
- The engine queries `cudaDeviceProp::luid`.
- **Validation Rule**: If `memcmp(dxgi_luid, cuda_luid, sizeof(LUID)) != 0`, zero-copy initialization fails immediately with structured error `adapter_luid_mismatch`. System fallback to Protocol V3 CPU ring is initiated automatically.

---

## 6. External Interop (D3D11 / Spout / OBS)

External broadcast applications (e.g. OBS Studio, Spout2, TouchDesigner, Resolume) expect Direct3D 11 shared handles.

### 6.1 D3D11 Shared Texture Sink
- Direct3D 11 shared resources are exposed via an independent output sink plugin.
- Direct3D 11 textures are created with `D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX`.
- Engine uses `IDXGIKeyedMutex::AcquireSync` / `ReleaseSync` for safe multi-process lockstep frame delivery.
- D3D11 interop is completely decoupled from the internal UI renderer.

---

## 7. Windows Virtual Camera (Media Foundation NV12 Sink)

For live webcam emulation in conferencing and streaming software (Zoom, Teams, OBS, Discord):

### 7.1 Media Pipeline Contract
1. **Colorspace Conversion**: The authoritative RGB8 master frame is converted exactly **once** on the GPU to `NV12` (Rec. 709 narrow range with standard BT.709 matrix coefficients).
2. **Media Foundation Buffer Allocation**: Samples are packaged directly into `IMFMediaBuffer` instances managed by a DXGI-backed sample allocator pool (`MFCreateDXGISurfaceBuffer`).
3. **Virtual Camera Registration**: Registered via Windows 11 `IMFVirtualCamera` APIs (`MFCreateVirtualCamera`).
4. **Invariant**: Master render graph pipeline remains 100% RGB8; NV12 format conversion is restricted solely to the virtual camera sink boundary.

---

## 8. Verification & Determinism Summary

- **Release QA & Master Determinism Gate**: All automated regression suites continue to verify exact master output frames using full-frame SHA-256 digests against the Gate 4 correctness lock:
  ```text
  43db5c512a3c44ba01fd127e0cca50376cc734d44d46303c894c6b948d010507
  ```
- **Live Runtime**: Live rendering and UI bridge paths operate with `hash_mode = OFF`, guaranteeing maximum frame throughput and zero per-frame CPU checksum overhead.
