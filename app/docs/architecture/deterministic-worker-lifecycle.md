# Deterministic processing-worker lifecycle

## Scope

This is the adapted Prompt 7 lifecycle for MegaGlitch's production architecture: Python/Dear PyGui host, spawned Python processing worker, fixed shared-memory transport, and optional coarse native effect segments. It does not promote the experimental Rust selector or a CUDA graph to ownership of arbitrary production graphs. Prompt 6 remains closed.

`WorkerProcessLifecycle` owns the child process and status channel. Lifecycle
traffic uses a semaphore-free `multiprocessing.SimpleQueue`: these few status
records are written synchronously, avoiding the bounded-queue capacity
semaphore and feeder-thread teardown race seen in detached frozen Windows
children. Its states are:

```text
uninitialized -> initializing -> warming_up -> ready -> running
                                                    |
                                                    v
                                             stopping -> stopped

initializing / warming_up / running -> failed -> stopping -> stopped
```

Each transition is timestamped, retained in a snapshot, and logged to the console. Child failures include the exception type, message, traceback, PID, and timestamp. `MEGAGLITCH_WORKER_STARTUP_TIMEOUT` controls the readiness deadline and defaults to 30 seconds.

## Actual startup order

1. `main.py` configures OpenCV DLL discovery, invalidates cached GPU state, and presents the existing splash.
2. `MegaGlitchFMApp` creates the Dear PyGui application, graph editors, queues, configuration objects, and current audio/MIDI UI services.
3. `run()` selects the explicit runtime backend and frame-transport backend.
4. Shared mode allocates fixed input/output regions and ownership semaphores. Allocation failure logs the error and safely retains legacy queue transport.
5. The parent enters `initializing`, spawns the managed worker, and waits; no application frame is submitted before readiness.
6. The Python worker reports `warming_up`, bootstraps the configured OpenCV build, runs the CUDA upload/filter/download self-test, runs the broader CUDA self-test, and selects CUDA or CPU effect fallback.
7. The worker attaches the required shared regions. Attachment failure is fatal because accepting descriptors as legacy arrays would violate the boundary contract.
8. The worker creates empty Deck A/B `FrameState` objects after the synthetic CUDA tests, then reports `ready` with backend, transport, and CUDA status.
9. The parent enters `running` and begins the Dear PyGui frame loop. Capture and processing work can now enter the bounded queue.

Synthetic CUDA data is created before temporal state exists and is never submitted to preview, history, recording, or virtual camera. The current application still initializes some audio/MIDI services while constructing the UI, before the processing worker; moving all integrations behind processing readiness is deferred because it is a broader UI/service lifecycle change and is not required for frame-state safety.

When the experimental Rust backend is explicitly selected, its selector validates that the runtime DLL loads before reporting ready. Arbitrary unsupported graphs retain the existing permanent Python-worker fallback. The Rust path still uses legacy frame transport.

## Runtime failure behavior

The parent polls child status once per UI iteration. A managed child exception after readiness changes the state to `failed`, records the structured error, stops frame submission, exits the render loop, and enters ordered cleanup. An unexpected normal worker return is also a failure unless shutdown was already requested.

CUDA self-test failure is not fatal: the worker reports ready with CPU fallback. Explicit native-Rust selection with a missing DLL is fatal and reports the build/load error. Shared-owner creation failure falls back to legacy transport before spawning. Required shared attachment failure, worker startup exception, premature exit, and startup timeout are fatal and trigger partial-startup cleanup.

## Partial-startup cleanup

If readiness is never reached, MegaGlitch terminates/reaps the child within a bounded timeout, stops capture and virtual-camera resources, closes media readers, stops sequencer/audio/MIDI services, destroys audio nodes, unlinks shared regions, closes processing queues, and destroys the Dear PyGui context. It does not depend on garbage collection for worker or shared-memory cleanup.

## Ordered shutdown

Normal UI-loop exit performs:

1. Set processing paused so no new frame work is accepted.
2. Stop/release window and webcam capture.
3. Close primary, secondary, and background media readers.
4. Drain pending input queue entries and release their shared slots.
5. Set the worker stop event and join for up to five seconds.
6. If necessary, terminate and join for two seconds, recording forced termination.
7. Drain pending output descriptors after the worker has stopped.
8. Stop/close virtual-camera output.
9. Close and unlink shared-memory regions.
10. Stop live sequencer, audio/MIDI, media-audio, and per-node audio resources.
11. Close multiprocessing queues.
12. Destroy the Dear PyGui context and log the final lifecycle snapshot.

`stop()` is idempotent. The shared owner is never unlinked before the worker has detached.

## Validation

Automated lifecycle tests cover:

- ordered warm-up/readiness/running/stopping/stopped transitions;
- startup exception and structured traceback;
- bounded startup timeout and forced termination;
- exception after readiness detected through parent polling;
- three repeated synthetic start/stop cycles;
- a real OpenCV/CUDA worker readiness cycle with required shared attachment;
- rejection and cleanup of a deliberately broken shared attachment;
- existing shared-worker dual-deck soak, graph replacement, reset, and repeated lifecycle tests.

The repeatable benchmark command is:

```powershell
python tools\benchmark\run_worker_lifecycle.py --cycles 3
```

At 1200x800 with shared transport:

| Cycle | Ready ms | First frame ms | Shutdown ms | Forced | Exit code | Slots retained |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 1571.3 | 10.3 | 225.2 | no | 0 | 0 |
| 2 | 1599.8 | 9.9 | 227.9 | no | 0 | 0 |
| 3 | 1662.4 | 12.9 | 247.1 | no | 0 | 0 |

All three outputs have the same deterministic SHA-256. Raw results are `tools/benchmark/results/worker-lifecycle.json` and `.csv`.

## Remaining manual/future cases

Automated tests do not close an actual Dear PyGui window during initialization, disconnect a real camera, activate a real virtual-camera driver, force real VRAM exhaustion, or inject a native access violation/allocator abort. CUDA-active effect shutdown is limited to the worker process boundary because no resident production CUDA graph exists. Those cases belong in production hardening and hardware-in-the-loop acceptance.

The lifecycle is sufficiently deterministic to proceed to original Prompt 8 production hardening and host-decision work. That task should improve build/package reproducibility and diagnostics; it must not reopen the scheduler migration without new benchmark evidence.
