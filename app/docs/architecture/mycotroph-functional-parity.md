# Mycotroph functional parity contract

## Product boundary

Mycotroph is a new control and presentation surface for MegaGlitch, not a second visual engine.
During the migration, the existing Python application remains the behavioral specification and
the supported rollback. Python continues to own sources, media clocks, graph compilation,
processing-worker lifecycle, automation, audio/MIDI, recording, and virtual-camera delivery.
Rust owns workspace presentation, editing gestures, preview texture upload, and presentation of
authoritative engine state.

The overhaul may reorganize the interface, but it must preserve this signal model:

```text
source/media transport
        |
        +--> Deck A graph --+
        |                   +--> master transition/crossfader --> program preview
        +--> Deck B graph --+                                  +--> recorder/export
                                                               +--> virtual camera

timeline + audio/MIDI/BPM/onset + modulation + hardware mappings
        `-----------------------------> deck and master parameters
```

A node catalog, editable-looking graph, connector styling, synthetic animation, or telemetry is
not evidence that this model is operational. A feature is live only when the Python engine has
accepted the relevant state and the resulting program frame or service state is observable.

## Ownership and compatibility matrix

| Surface | Authoritative owner during migration | Clock / queue policy | Compatibility gate |
| --- | --- | --- | --- |
| Image, GIF, video, webcam, window, generator, background and secondary sources | Python engine service | Media/capture clock; latest source frame for live work | Source pixels and transport behavior match the Python UI, including disconnect and resolution changes |
| Deck A and Deck B graphs | Python graph compiler and processing worker | Immutable graph generations applied at frame boundaries | Existing single-deck and dual-deck presets compile without field or routing loss |
| Program preview | Python program output; Rust texture consumer | Fixed leased preview slots; drop preview when the consumer stalls | Displayed RGB pixels match the canonical worker result; preview stalls cannot stall processing |
| Master mix and transitions | Python processing worker | Program-frame clock | Endpoints and every transition match the existing implementation |
| Parameters and timeline | Python timeline and graph state | Media/global-time clock; last-value-wins control patches | Fixed-clock parameter traces match and preset state round-trips |
| Audio analysis, MIDI, BPM/onset and modulation | Existing Python managers | Independent audio/MIDI clocks; bounded event/state snapshots | Recorded input traces produce matching parameter values; panic and shutdown are deterministic |
| Recording and batch export | Existing Python recording/render paths | Ordered, no-drop/backpressure path | Frame count, timing, cancellation and cleanup match |
| Virtual camera | Existing Python sender | Latest-program-frame queue | Camera format, pacing, failure reporting and teardown match |
| Session/preset files | Python engine service | Transactional file operation | Legacy v1 graphs and dual-deck sessions load; save/readback loses no fields |
| Runtime telemetry | Component that performs the work | Measured, never fabricated | Timings, drops, copies and lifecycle state are attributable and reproducible |

There is no separate project format in the prior application. Its dual-deck session preset is the
effective project document. Mycotroph must use honest session/preset terminology until a versioned
project wrapper and migration path exist.

## Control and frame boundary

Low-frequency control uses a versioned request/acknowledgement protocol. Every mutating command
contains a command ID and the state revision it expects. Acceptance returns the new authoritative
revision and the frame sequence where the change becomes effective. Rejection returns a structured,
recoverable error and leaves the last accepted state running. The interface distinguishes draft,
pending, accepted and rejected graph generations; process liveness alone is never labelled `LIVE`.

Program preview uses a fixed-capacity shared surface, independent of JSON control traffic:

1. Python claims a preview slot and copies a downscaled canonical RGB program frame.
2. Python publishes the slot, generation, sequence, dimensions, stride and timestamp.
3. Rust copies the slot into its presentation buffer and releases it.
4. With no free slot, Python drops only that preview publication and increments a measured counter.

Recording and virtual camera consume the canonical program output through their existing policies.
They never depend on the Rust repaint rate.

## Functional stages and release gates

### Gate 1: one real visual path

- Real test, still-image and video sources with run/pause/seek and actionable errors.
- One accepted Deck A graph executed through the existing Python frame path.
- Actual processed RGB pixels rendered in the Rust preview.
- Transactional graph/source/transport state and bounded preview slots.
- Clean repeated start/stop with no remaining child, mapped surface or device handle.
- Byte parity for pass-through and Invert, and sequence parity for Feedback/Trail.

Unsupported systems are visibly unavailable at this gate; they are not represented by animated
meters, fake costs, hard-coded device lamps, or enabled buttons.

### Gate 2: graph and two-deck session parity

- Extract the existing graph compiler behind a UI-independent Python contract.
- Support stable node IDs, branches, generators and selected outputs, multi-input nodes, primary
  and secondary sources, Lua/GLSL metadata, bypass, Deck A/B, master mix and all transitions.
- Round-trip the preset corpus through Rust -> Python -> disk without field loss.

### Gate 3: source and performance-control parity

- Connect image, GIF, video, webcam, window, generator, background and channel-B sources.
- Connect media transport, shader cues, resolution/FPS/processing scale, adaptive scaling,
  backend selection, diagnostics, and Genesis auxiliary inspection.

### Gate 4: automation, audio and MIDI parity

- Expose the existing modulation matrix, parameter timeline, time macros, AudioMidiManager,
  media-audio controller, visual analyzer, live sequencer, MIDI routing/learn and MPD26 controls.
- Keep their clocks and device ownership in Python; Rust edits configuration and renders snapshots.

### Gate 5: output parity

- Connect real-time GIF/MP4 recording, batch rendering/progress/cancellation and virtual camera.
- Verify that preview stalls do not alter recorded frames or virtual-camera pacing.

### Gate 6: release hardening

- Bundle and test the required Python runtime, custom CUDA OpenCV build, effect modules, shaders,
  Lua assets, presets and native libraries with the Rust frontend.
- Pass clean-machine, camera, audio, MIDI, CUDA, GL/Lua, virtual-camera, overnight resource-soak,
  compatibility and rollback tests.

Until these gates are green, builds are functional-preview artifacts, not production replacements.
`python main.py` remains the supported full machine and immediate rollback.

## Workspace logic

The new spatial design follows responsibilities instead of copying the old widget arrangement:

- **PATCH**: Deck A/B graphs, sources, sockets, routing and node parameters.
- **PERFORM**: source transport, real program preview, A/B crossfader, transitions and time macros.
- **DESIGN**: catalog, session/preset management, display/background and generator configuration.
- **AUTOMATE**: modulation matrix, parameter timeline, live sequencer and MIDI learn.
- **MONITOR**: measured processing, capture, CUDA/GL, audio/MIDI, recording, virtual-camera and
  lifecycle health.

A workspace becomes enabled only when its negotiated engine capabilities exist. Switching a tab
must change the operational surface, not only highlighted text.
