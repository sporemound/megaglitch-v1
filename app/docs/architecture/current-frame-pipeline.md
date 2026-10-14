# Current frame pipeline audit

## Scope and confidence

This is a static, evidence-based audit of the live preview path. It does not change processing behavior and it does not yet contain runtime measurements. Statements marked **observed** follow directly from code; statements marked **hypothesis** require the profiling baseline before they can be called bottlenecks.

The primary path inspected is `python main.py` into `MegaGlitchFMApp.run()`. Batch export has separate code and is noted only where it shares live-path helpers.

## Quality-first master-path update (2026-08-27)

The detailed sections below preserve the original pre-overhaul audit as baseline evidence. The
current quality contract removes adaptive/preview scaling before worker submission: primary ingress
defines one master resolution, Deck B may be deliberately conformed once at ingress, and the worker
rejects non-master input, output, or temporal state rather than repairing it with an implicit resize.
Preview/UI scaling now occurs only after the worker returns the validated master frame; recording and
virtual-camera submission consume the master frame, not the preview texture.

Offline Master is a separate serial path. `OfflineMasterRenderer` advances a fixed
`MasterGraphExecutor` once per source frame, keeps temporal state at master resolution, applies FPS
placement only at encoder egress, and bypasses preview, Dear PyGui presentation, OBS capture, and
virtual-camera capture. Direct production output uses an FFmpeg stdin pipe; optional codec-domain
corruption remains a downstream stage rather than an image-domain node.

Run `tools/benchmark/run_render_quality_path.py` to verify resolution/history integrity, fixed graph
hash and seed, FPS semantics, provider assignments, observed transfers, copies, and drop/duplicate
counts. Its memory sink deliberately reports null bitrate, so it is a correctness harness and not an
encoder-performance result. Gates, limitations, and rollback are documented in
`docs/profiling/render-quality-overhaul.md`.

## Pipeline diagram

```text
MAIN PYTHON PROCESS / Dear PyGui thread

image/GIF/video/webcam/window/generator
    |  PIL RGBA and/or CPU NumPy RGB/BGR/BGRA uint8
    v
MegaGlitchFMApp.run()
    capture/decode, graph snapshots, timeline, audio-node and modulation updates
    optional CPU resize for adaptive processing scale
    |
    | mp.Queue(maxsize=2): frame(s) + both Python pipeline lists + parameters
    | Windows multiprocessing serialization/copy boundary
    v

WORKER PYTHON PROCESS / one processing thread

glitch_process_worker()
    PIL -> CPU NumPy RGB uint8 when needed
    two FrameState instances (Deck A and Deck B)
    |
    +--> execute_flat_node_pipeline(Deck A)
    |       Python list iteration and per-node apply_node_effect()
    |       CPU NumPy <-> cv::cuda::GpuMat/Frame <-> ModernGL texture
    |       persistent CPU/GPU/GL temporal state depending on node
    |
    +--> execute_flat_node_pipeline(Deck B)
    |
    +--> blend_decks() in CPU float32/uint8
    |
    +--> forced CPU RGB uint8 NumPy output
    |
    | mp.Queue(maxsize=2): master frame and Deck A/Deck B frames
    | Windows multiprocessing serialization/copy boundary
    v

MAIN PYTHON PROCESS / Dear PyGui thread

latest result drain
    +--> visual analyzer queue (full-frame copy) -> analyzer thread
    +--> two Dear PyGui dynamic textures (CPU resize, RGBA8 canvas, float32 canvas)
    +--> optional recording_frames list (full-frame copy)
    +--> optional virtual-camera queue -> sender thread -> pyvirtualcam
    +--> Dear PyGui render
```

There is a separate passthrough fast path in `MegaGlitchFMApp.run()` that bypasses both multiprocessing queues when Deck A is a source-to-display graph with no generator, Deck B, recording, or secondary input. It still performs visual analysis, two preview uploads, optional virtual-camera submission, and Dear PyGui rendering (`megaglitchfm_app_new.py:6101-6136`).

## Process, thread, and resource ownership

| Resource/work | Current owner | Evidence and implications |
|---|---|---|
| Dear PyGui context, UI callbacks, UI render loop | Main process, main thread | `MegaGlitchFMApp.setup_ui()` creates the context and viewport (`megaglitchfm_app_new.py:2151-2167`); `MegaGlitchFMApp.run()` calls `dpg.render_dearpygui_frame()` (`megaglitchfm_app_new.py:5842-6529`). |
| Image/GIF/video/webcam/window capture | Main process, main/UI loop thread | Source reads happen inline in `MegaGlitchFMApp.run()` (`megaglitchfm_app_new.py:5937-6094`). There is no dedicated live capture thread. |
| Graph snapshot, timeline, modulation, audio-node UI updates | Main process, main/UI loop thread | `NodeEditor.get_processing_pipeline()`, timeline and modulation traversal execute before queue submission (`megaglitchfm_app_new.py:5900-5902`, `5915-5924`, `6150-6249`). |
| Graph processing | Child Python process, its main thread | `MegaGlitchFMApp.run()` starts one `multiprocessing.Process` running `glitch_process_worker()` (`megaglitchfm_app_new.py:5842-5844`). |
| Per-deck legacy temporal state | Worker process, `FrameState` for each deck | `state` and `state_b` are created by `glitch_process_worker()` (`megaglitchfm_app_new.py:1449-1455`). |
| CUDA device objects and caches | Worker process, but split among module globals, a global `Context`, global GPU-node instances, and effect dictionaries | `gpu_ops.py:27-30`, `cuda_effects.py:18-24`, and `cuda_effects.get_effect_state()`/`get_gpu_node()` own state independently of `FrameState`. Ownership is not unified. |
| ModernGL context and Lua shader resources | Worker process, processing thread | One context is stored on each deck's `FrameState`; shader nodes and auxiliary textures are keyed per graph node (`effects_registry.py:567-622`). |
| Visual analysis | Main process, dedicated latest-frame thread | `VisualAnalyzerWorker.submit_frame()` copies a frame into a size-one queue and `_run()` analyzes it (`megaglitch_visual_analyzer.py:145-205`). |
| Virtual-camera output | Main process, dedicated sender thread | `_start_virtual_cam_sender()` starts `_virtual_cam_sender_loop()`; a size-one queue drops stale frames (`megaglitchfm_app_new.py:7809-7886`). |
| Audio, MIDI, live sequencer, media-audio analysis | Main process, several callback/worker threads | Threads are created in `audio_midi_manager.py`, `megaglitch_live_sequencer.py`, `megaglitch_visual_analyzer.py`, and `megaglitch/audio/media_audio_engine.py`. They do not own graph processing but can contend for CPU and the main-process GIL. |

The separate worker process prevents its Python graph loop from sharing the main process's GIL. Within each process, however, Python traversal and callbacks remain serialized by that process's GIL except where native libraries release it.

## Source paths and representations

### Still images and GIFs

- `media_handler.open_image_file()` reads through OpenCV or PIL but returns a PIL RGBA image (`media_handler.py:90-119`). `load_image_callback()` keeps an original copy and a composited PIL image (`megaglitchfm_app_new.py:4533-4546`). The live loop usually calls `self.image.copy()` (`megaglitchfm_app_new.py:5937-5948`).
- `media_handler.open_gif_file()` eagerly copies every GIF frame into PIL RGBA storage (`media_handler.py:121-128`). The live loop copies the selected frame (`megaglitchfm_app_new.py:5955-5983`).

### Video

- `media_handler.open_video_file()` prefers `opencv_backend.VideoReader`, falling back to imageio (`media_handler.py:134-170`).
- `opencv_backend.VideoReader.__next__()` synchronously calls `cv2.VideoCapture.read()` and converts BGR to a newly allocated CPU RGB NumPy array (`opencv_backend.py:355-371`). The live loop reads/seeks on the UI thread and caches the last NumPy frame (`megaglitchfm_app_new.py:5985-6062`).
- Channel B reads similarly, but converts NumPy RGB to a PIL RGBA image and also keeps a PIL copy (`megaglitchfm_app_new.py:5624-5665`).

### Webcam and window capture

- Webcam capture uses `cv2.VideoCapture` and calls `read()` inline in the main loop, then converts BGR to PIL RGBA (`megaglitchfm_app_new.py:4889-4924`, `6069-6079`).
- Window capture reuses an `mss` object but grabs inline in the main loop. It creates a BGRA NumPy view, then an explicit contiguous RGB copy, and may resize on CPU (`megaglitchfm_app_new.py:5044-5107`).

### Generator-only graphs

The main process creates a full-size black PIL image as a dummy source, which is serialized to the worker even though the generator replaces it (`megaglitchfm_app_new.py:6080-6083`).

## Main-process scheduling and queue boundary

`NodeEditor.get_processing_pipeline()` caches graph topology but returns a new list of dictionaries with copied parameter dictionaries (`node_editor.py:74-86`, `850-865`). It is called for both decks every eligible frame. The main loop then:

1. iterates audio nodes and publishes/reroutes modulation (`node_editor.py:892-930`);
2. normalizes Deck B source entries (`megaglitchfm_app_new.py:5918-5924`);
3. applies timeline and SRT changes (`megaglitchfm_app_new.py:6150-6174`);
4. collects and updates parameter dictionaries and potentially Dear PyGui controls (`megaglitchfm_app_new.py:6176-6249`);
5. optionally resizes both sources on CPU (`megaglitchfm_app_new.py:6251-6257`);
6. packages full source frames, both pipeline lists, parameters, and timing data into a dictionary (`megaglitchfm_app_new.py:6272-6296`);
7. sends that dictionary through `multiprocessing.Queue` (`megaglitchfm_app_new.py:6298-6321`).

Both queues have a nominal maximum size of two (`megaglitchfm_app_new.py:1827-1831`). Live preview drains/drops stale input and output; recording uses a blocking input put with a one-second timeout. The queues use ordinary multiprocessing serialization, not shared memory, so CPU frames and graph/parameter objects cross a serialization boundary. The exact number of kernel/user-space copies is an implementation detail to measure, but a receive-side NumPy allocation is unavoidable with this design.

## Worker graph execution

`glitch_process_worker()` receives one item at a time (`megaglitchfm_app_new.py:1464-1488`), converts PIL sources to NumPy (`1490-1501`), ensures three full-size CPU state arrays per deck as needed (`1506-1536`, `1608-1615`), and calls `execute_flat_node_pipeline()` for each connected deck (`1583-1642`).

`execute_flat_node_pipeline()` is a Python interpreter loop over every flattened entry (`node_pipeline_runtime.py:675-697`). It uses a Python list as a frame stack and dispatches each regular node through `effects_registry.apply_node_effect()` (`node_pipeline_runtime.py:697-856`). Source nodes copy their full input (`node_pipeline_runtime.py:709-715`). Generator, mixer, and multi-input nodes add further Python dictionary/list traversal and frame conversions.

Therefore, **observed:** Python dispatches every graph node individually in the worker. Whether this dispatch time is material relative to image work is not yet measured.

The final pipeline value is always passed through `gpu_ops.ensure_cpu()` and normalized to CPU RGB `uint8` (`node_pipeline_runtime.py:858-863`). Deck mixing is then CPU-based: `blend_decks()` converts each available deck to float32, creates a black float32 peer for a missing deck, computes masks/blends, and converts back to `uint8` (`node_pipeline_runtime.py:866-965` and following mode branches). Even a one-deck processed graph goes through this function.

The worker returns a dictionary containing the master frame plus Deck A and Deck B frames (`megaglitchfm_app_new.py:1694-1724`). With two decks, this can serialize three full-resolution arrays per result. The main process currently uses the deck arrays only to enforce live crossfader endpoints (`megaglitchfm_app_new.py:6353-6381`).

## CPU, CUDA, and ModernGL boundaries

### OpenCV CUDA

There are two CUDA API styles:

- GPU-first helpers/nodes accept and return `cv2.cuda_GpuMat` or `megaglitchnode.gpu.Frame` (`gpu_ops.py:203-254`, `megaglitchnode/gpu/`). Adjacent compatible CUDA nodes can keep the primary image resident.
- Hybrid compatibility helpers accept NumPy, allocate/upload one or more `GpuMat`s, execute a CUDA operation, and immediately download NumPy (`gpu_ops.py:257-702`). Each such helper creates a transfer and a synchronization boundary.

`effects_registry.apply_node_effect()` preserves the primary `GpuMat` for a CUDA-labeled node, but if `previous_frame` exists it unconditionally calls `gpu_ops.ensure_cpu(previous_frame)` before dispatch (`effects_registry.py:521-540`). After the first CUDA node, `previous_frame` is often the prior CUDA result, so a chain of CUDA nodes can download a full frame before each node even when that handler does not use it. This is a concrete residency break to instrument.

`cuda_effects.apply_cuda_effect()` contains both genuinely GPU-resident handlers and handlers that explicitly call `gpu_ops.ensure_cpu()` (`cuda_effects.py:1795-1984`). Examples:

- persistent feedback keeps a GPU accumulator and returns `GpuMat` (`cuda_effects.py:486-528`);
- edge slicer/channel drift return `GpuMat` (`cuda_effects.py:790-896`);
- digital melt downloads its mask, performs a CPU bitwise operation, and uploads the result again (`cuda_effects.py:898-946`);
- temporal median and several other CUDA-labeled nodes are CPU NumPy implementations (`cuda_effects.py:556-675`, `1919-1960`);
- the “CUDA Temporal Slit-Scan” uses a preallocated CPU NumPy ring and begins by forcing CPU residency (`cuda_effects.py:1012-1068`).

The active global GPU `Context` is created with `use_stream=False` (`cuda_effects.py:40-49`), and most raw `cv2.cuda` calls omit a stream. No active multiple-stream scheduling was found. `GpuMat.download()`, GL texture reads, and return-to-host conversion are synchronization points; other implicit default-stream ordering must be confirmed with CUDA events/Nsight.

Filter reuse is mixed. `gpu_ops` caches many filters, and GPU node classes cache filters/maps. In contrast, `opencv_backend` CUDA convenience functions create an input `GpuMat` each call and some recreate filters each call (`opencv_backend.py:681-877`). Most CUDA calls omit a reusable destination, so output `GpuMat` allocation/reuse behavior is left to OpenCV. Animated `GpuWarpRemap` builds CPU maps, uploads them, and retains up to roughly 100 phase-keyed map pairs (`megaglitchnode/gpu/nodes/gpu_warp_remap.py:125-190`), which is a likely allocation/VRAM pressure point.

### ModernGL/Lua

Lua shader nodes receive CPU NumPy input because all non-CUDA nodes first call `gpu_ops.ensure_cpu()` (`effects_registry.py:506-514`). Each node then:

- constructs an RGBA NumPy array, including a full alpha allocation for RGB input;
- converts it to bytes and uploads it with `texture.write()`;
- similarly converts/uploads every named auxiliary input;
- renders into persistent GL resources;
- synchronously reads the full output texture to CPU bytes;
- creates a NumPy RGBA view, a PIL RGBA image, and then a new NumPy RGB output.

These boundaries are at `effects_registry.py:635-745`. Adjacent Lua shaders do not remain GPU-resident between nodes. The GL context and node textures are persistent and rebuilt on size/path changes, but frame payloads cross host/GL memory for every shader node.

No CUDA/ModernGL interoperability path was found.

## Temporal state and history ownership

Temporal storage is not one subsystem:

1. Each deck's `FrameState` owns `previous_frame`, `feedback_image`, `trail_image`, and a Python `list` of CPU NumPy history frames (`frame_state.py:6-35`).
2. After a flat pipeline runs, the worker deep-copies its deck output into the list and removes index zero while over the selected limit (`megaglitchfm_app_new.py:1571-1606`, `1617-1648`). `pop(0)` shifts Python references, not pixel buffers, but the new full-frame `copy()` is unavoidable.
3. History sampling builds Python lists and may allocate float32 blends (`effects_registry.py:136-160`, `173-341`). Row stutter converts the selected history window again and loops over image rows in Python (`glitch_effects.py:815-828`). P-frame temporal bleeding can copy every history frame before processing (`effects_registry.py:1333-1336`).
4. The fallback `glitch_effects.apply_glitch_effect()` also appends to the supplied history (`glitch_effects.py:1775-1789`). Because `apply_node_effect()` may invoke that fallback for an individual unmapped node (`effects_registry.py:1885-1897`) and the worker appends again after the complete graph, history can advance more than once per displayed frame. Ownership/semantics are ambiguous and should be tested with a graph containing multiple fallback nodes.
5. CUDA effects independently own deques, NumPy rings, GPU accumulators, prior-frame clones, and maps in module-global dictionaries (`cuda_effects.py:18-24`, `70-88`). `get_effect_state()` keys by effect name and frame shape, not graph-node or deck identity; multiple same-type nodes and both decks can therefore share temporal state. The inline GPU nodes also use fixed IDs such as `feedback_mixer_inline` (`cuda_effects.py:1844-1869`). This is an unsafe/ambiguous ownership boundary.
6. Lua shader state is better isolated: it is held in the deck's `FrameState.effect_states` and keyed by graph node ID (`effects_registry.py:567-622`).

The main `FrameState` history is CPU `uint8`. No 16-bit-float or 32-bit-float long history was found. Some operations create transient float32 arrays and some CUDA filters/maps use float formats. Separate temporal nodes can retain independent copies of related frames.

## Resolution, format, and memory behavior

- The common worker contract is RGB `uint8` NumPy, but source code also uses PIL RGBA, OpenCV BGR/BGRA, `GpuMat`, the `Frame` wrapper, GL RGBA8 textures, and float32 intermediates.
- Adaptive processing uses quantized scales to reduce resize churn (`megaglitchfm_app_new.py:4072-4123`), but each new shape can create new worker state.
- `previous_frame`, feedback, and trail arrays are replaced on dimension changes, but `FrameState.frame_buffer` is not explicitly cleared in that size-change block (`megaglitchfm_app_new.py:1506-1536`). Mixed-size history can therefore persist until effects resize it or the list ages out.
- CUDA effect state is keyed by shape, so a resolution change creates another state entry and retains prior CPU/GPU resources until a module-level reset. A normal worker `RESET` calls only `FrameState.reset()` and does not call `cuda_effects.reset_cuda_states()` (`megaglitchfm_app_new.py:1452-1462`).
- ModernGL textures are resized/rebuilt explicitly. Dear PyGui preview buffers are cached by texture tag and display dimensions (`media_handler.py:262-279`).
- The main history list and node-local deques/rings are bounded by parameters, but limits are fragmented. The worker scans generic `buffer_size`/`frame_buffer_size` parameters from every node (`megaglitchfm_app_new.py:1571-1581`), so unrelated parameter names may affect the deck-wide history limit. Recording accumulates copied full frames in `recording_frames` until completion (`megaglitchfm_app_new.py:6461-6466`).

Static evidence points to system RAM pressure from multiprocessing payloads, CPU histories, preview canvases, analysis, and recording; and VRAM pressure from CUDA state/maps plus GL textures. No pinned-host-memory allocation was found. Which capacity is reached first is unknown.

## Output, preview, and recording

The main process drains the entire output queue and keeps only the newest result (`megaglitchfm_app_new.py:6323-6335`). It then performs these operations on the UI thread:

- endpoint crossfader correction and latency bookkeeping (`megaglitchfm_app_new.py:6353-6396`);
- optional retro processing and a full-frame visual-analyzer submission copy (`megaglitchfm_app_new.py:6397-6415`, `megaglitch_visual_analyzer.py:173-190`);
- two calls to `media_handler.update_display_texture()`, one for each deck display, even though both monitor the same master output (`megaglitchfm_app_new.py:6416-6459`);
- optional full-frame recording copy (`megaglitchfm_app_new.py:6461-6466`);
- optional virtual-camera conversion/enqueue (`megaglitchfm_app_new.py:6511-6521`).

`update_display_texture()` may resize, writes an RGBA8 display canvas, converts the entire canvas to float32 normalized data, and calls `dpg.set_value()` (`media_handler.py:219-280`). Buffers are reused, but every pixel is rewritten twice per displayed result and the Dear PyGui internal copy/upload cost is unknown.

The virtual-camera path drains to the latest frame, makes input contiguous if needed, optionally resizes, prepares contiguous `uint8`, then `cam.send()`/paces on its own thread (`megaglitchfm_app_new.py:7839-7886`). OBS-specific processing is not present; OBS consumes the virtual camera or window capture externally.

## Full-frame copy and allocation accounting

There is no single correct copy count because it depends on source type, one versus two decks, node types, history, preview, recording, and virtual-camera state. Static code nevertheless establishes the following per-frame materializations for a typical non-passthrough, single-deck graph:

| Phase | Explicit full-frame work visible in Python code |
|---|---|
| Source | Usually 0-2 copies/conversions; static PIL and GIF paths call `copy()`, webcam/video/window create converted arrays. |
| Main -> worker | At least one serialized frame payload and one receive-side object/array materialization; internal pipe copy count is unmeasured. |
| Flat pipeline | One source-stack `copy()` plus node-specific outputs/temporaries. |
| Deck history | At least one output `copy()` per active deck; fallback nodes can add more. |
| Deck blend | For one deck, at least a float32 deck conversion, a float32 black peer, and a uint8 output; transition modes add masks and float temporaries. |
| Worker -> main | Master plus active deck-monitor arrays are serialized; up to three full frames for two decks. Receive-side arrays are materialized in the main process. |
| Visual analyzer | One full RGB copy when enabled/running. |
| Preview | Two display-sized RGBA8 writes and two float32 conversions; resize allocations when source size differs. DPG's internal texture copy is unknown. |
| Recording | One full output `copy()` per recorded frame. |
| Virtual camera | A contiguous copy only when needed, plus a resize allocation if camera dimensions differ. |

Thus the ordinary processed path is unequivocally multi-copy, but an exact “copies per frame” number must be produced by instrumentation rather than inferred from source. The baseline should count bytes and distinguish allocations, memory-to-memory copies, serialization, GPU transfers, and repeated writes into reusable buffers.

## Current instrumentation gaps

`performance_profiler.py` provides CPU `perf_counter` averages/maxima for named stages and `glitch_effects.apply_glitch_effect()` uses it internally. The main loop also logs end-to-end queue latency and processed FPS. It does not currently provide:

- per-node timing for the flat dispatcher;
- queue serialization/deserialization time or bytes;
- allocation/copy counts;
- CUDA-event timing or explicit synchronization attribution;
- upload/download counts and sizes;
- preview, analyzer, recording, and virtual-camera phase timings;
- p95/p99 or 1% low frame statistics;
- RAM, pinned RAM, VRAM, GPU utilization, or dropped-frame totals;
- a repeatable machine-readable benchmark result.

Consequently none of the suspected bottlenecks below is yet confirmed.

## Prioritized hypotheses to test

1. **Host/device and GL transfer/synchronization dominates mixed GPU graphs.** Measure adjacent CUDA nodes, hybrid CUDA nodes, and adjacent Lua shaders separately. Instrument `effects_registry.apply_node_effect()` (`effects_registry.py:477-751`), `gpu_ops.ensure_cpu()`/hybrid helpers (`gpu_ops.py:125-181`, `257-702`), the CUDA dispatcher (`cuda_effects.py:1795-1984`), and final CPU conversion (`node_pipeline_runtime.py:858-863`). Compare with final download disabled.
2. **Multiprocessing serialization and excess return payloads materially inflate latency and RAM bandwidth.** Time/size both queue boundaries (`megaglitchfm_app_new.py:6272-6335`, `1694-1724`) and compare ordinary arrays against a shared/preallocated transport. Specifically test returning only the master frame versus master plus both deck frames.
3. **Per-frame full-frame allocations/copies, CPU float deck blending, and fragmented histories dominate before Python call overhead.** Count allocations/bytes in source-stack copies, `blend_decks()` (`node_pipeline_runtime.py:866` onward), worker history writes (`megaglitchfm_app_new.py:1571-1648`), fallback history (`glitch_effects.py:1740-1789`), and temporal effects that stack/copy history.
4. **Synchronous preview/output fan-out stalls the UI/capture loop.** Compare preview on/off, one versus two monitor uploads, visual analysis on/off, virtual camera on/off, and recording on/off around `media_handler.update_display_texture()` (`media_handler.py:219-280`) and the result consumer (`megaglitchfm_app_new.py:6397-6521`).
5. **Python per-frame graph/control traversal and per-node dispatch leave measurable CPU gaps between native operations.** Measure `NodeEditor.get_processing_pipeline()`/modulation work in the main process and each iteration/call in `execute_flat_node_pipeline()` (`node_pipeline_runtime.py:675-863`). Compare a representative graph with the same operations grouped behind fewer Python calls; do not assume Rust is required unless this gap remains significant after transfer/allocation issues are controlled.
6. **Temporal state is both semantically unsafe and memory-inefficient.** Test multiple same-type CUDA temporal nodes across both decks, multiple fallback nodes, resolution changes, and reset behavior. Track whether history advances more than once per displayed frame and whether old-shape CUDA/GL resources remain allocated.
7. **Default-stream serialization and per-call output allocation prevent useful GPU overlap.** Add CUDA events around uploads, kernels, downloads, and implicit waits. Compare the current `Context(use_stream=False)` with carefully controlled streams only after correctness is established.
8. **The practical ceiling is capacity rather than orchestration.** Sweep supported resolution/history/format combinations while recording system RAM and VRAM. Distinguish CPU history, recording accumulation, CUDA maps/state, and GL textures; no evidence currently selects RAM or VRAM as the dominant limit.

## Audit conclusion

The current architecture provides a genuine process boundary between UI and processing, and some CUDA nodes can remain device-resident across compatible neighbors. It does not provide end-to-end device residency or one clear temporal/GPU owner. The most concrete static concerns are repeated CPU/GPU/GL crossings, serialized full frames in both directions, full-frame allocation/copy fan-out, and fragmented state ownership. Python per-node dispatch is real but must be measured after those costs are isolated. This audit does not establish Rust—or any native rewrite—as the solution.
