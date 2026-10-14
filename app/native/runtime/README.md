# Rust runtime vertical slice

Build with no third-party Rust crates:

```powershell
powershell -ExecutionPolicy Bypass -File native\runtime\build.ps1
```

The build script uses `%USERPROFILE%\.cargo\bin\cargo.exe`, places Cargo intermediates under `%TEMP%`, and copies the release DLL to `megaglitch\_native\megaglitch_runtime.dll`.

The application worker compiles supported primary single-deck pipelines to versioned graph IR
instead of selecting a mode per effect. IR v1 supports Input, Invert, Brightness/Contrast,
Temporal Mix, history Blend, and Output. Secondary sources, Deck B, active temporal macros, and
arbitrary effects remain on the Python fallback. The old `passthrough`, `temporal_mix`, and
`invert` constructors remain only as compatibility adapters that compile to IR v1. Select the
facade backend with `MEGAGLITCH_RUNTIME_BACKEND=native`, `python`, or `auto`.

Run the comparison benchmark with:

```powershell
python tools\benchmark\run_native_runtime.py --label native-runtime --warmup 5 --iterations 30
```

See `docs/architecture/python-hosted-native-runtime.md` for the lifecycle, compatibility boundary, and measured results.
The first graph-node gate is recorded in `docs/profiling/rust-graph-v1-invert.md`.
The versioned IR contract is in `docs/architecture/rust-graph-ir-v1.md`, with measurements in
`docs/profiling/rust-graph-ir-v1.md`.
