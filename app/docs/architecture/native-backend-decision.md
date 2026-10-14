# Native temporal-backend decision

## Decision

The first temporal prototype is a small C++ CPU ring exposed to Python through Cython. It owns one fixed contiguous allocation, performs indexing and temporal mixing natively, and crosses into native code once for each coarse operation. It does not use OpenCV or CUDA and does not migrate graph scheduling.

This is an intermediate measurement boundary, not a commitment to Cython or C++ as the final runtime host. The public Python facade deliberately does not expose Cython types, so a later Rust or C++/CUDA implementation can replace the storage without changing callers.

## Options evaluated

### Rust with OpenCV bindings

Deferred for this prototype. At the time the prototype boundary was selected, the repository had no Cargo workspace or Rust boundary and the toolchain was unavailable to the active shell. Rust was subsequently installed and verified as Rust/Cargo 1.97.0 with the `stable-x86_64-pc-windows-msvc` toolchain. The current Codex shell still needs `%USERPROFILE%\.cargo\bin` added to `PATH`, or the tools must be invoked by absolute path.

Toolchain availability removes the installation blocker, but it does not change the measured gate: adding Rust OpenCV bindings, OpenCV discovery, and temporal semantics would make a second ring prototype difficult to attribute, while direct Rust bindings still do not resolve the custom OpenCV CUDA ABI and ownership questions established by the pipeline audit. A future Rust scheduler or resource owner can consume a narrow C ABI/C++ CUDA backend without rewriting this already-measured CPU experiment.

### Rust controlling a narrow C++ OpenCV CUDA backend

Deferred. It remains the likely shape for device-resident graph work, but the baseline showed that the immediate temporal question is fixed storage and copy cost on CPU. A CUDA bridge would introduce stream, DLL, and `GpuMat` lifetime work that is not needed to test ring indexing. The existing custom CUDA OpenCV module also differs between the parent and worker processes.

### C++ exposed to Python

Selected for the completed experiment. Visual Studio 2022 Professional and Build Tools are installed, and the active Python environment already contains Cython, NumPy, and setuptools. A Cython C++ extension gives native allocation and loops with a small amount of binding code. It can be built without OpenCV or CUDA linkage and leaves the Python engine untouched as the default fallback. Reimplementing the same CPU ring in Rust after the toolchain installation would not answer a new performance question: the benchmark already showed that preallocation, rather than implementation language, explains the write-path gain.

### Existing native boundary

None was found. The application uses Python modules and binary dependencies, but the repository contains no native project, extension manifest, or application-owned native frame API that could be extended safely.

## Boundary and ownership

`TemporalFrameRing` exposes `configure`, `write_frame`, `read_frame`, `process_temporal_mix`, `set_delay`, `set_mix`, `freeze`, `clear`, `get_statistics`, and `shutdown`. The ring owns its fixed allocation. Writes copy one complete contiguous RGB8 frame into that allocation. Native temporal mixing reads history and writes a caller-provided or newly allocated output in one native call. `read_frame` intentionally returns a copy so Python cannot retain a pointer that becomes invalid after resize or shutdown.

The prototype supports RGB8 only because that is the only end-to-end format supported by the measured pipeline. The API carries channel count and keeps history storage separate from transient output so RGBA8, RGBA16F, luma-only, reduced-resolution, shared histories, and device-resident implementations remain possible. Those tiers are not implemented without evidence.

## Safety and fallback

- Configuration validates positive dimensions and capacity before allocating.
- Delay must refer to a written frame; invalid values fail explicitly.
- Reconfiguration replaces the allocation atomically from the Python caller's perspective and clears temporal contents.
- Freeze prevents writes without changing indices.
- Clear zeroes storage and resets indices.
- Shutdown releases the vector and is idempotent; the object can be configured again.
- `MEGAGLITCH_HISTORY_BACKEND=python` selects the always-available NumPy implementation. `native` requires the compiled extension. `auto` uses native when available and otherwise falls back to Python.

The live graph continues to use its existing Python list unless a caller explicitly requests the new `FrameState.get_temporal_history(...)` prototype boundary. This preserves visual behavior and makes rollback the removal of one environment setting or explicit backend argument.

## Advancement gate

This prototype justifies more migration only if it reduces steady-state write/mix time or allocation variance at practical history sizes without output differences, leaks, lifecycle failures, or a new per-frame history export to Python. A result limited to faster indexing is insufficient: baseline lookup was already measured in microseconds.
