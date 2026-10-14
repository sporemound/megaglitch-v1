use memmap2::{Mmap, MmapOptions};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{
    Arc, Mutex, RwLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
    mpsc,
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const BRIDGE_PROTOCOL_VERSION: u32 = 2;
const BRIDGE_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(1);
const DEFAULT_BRIDGE_STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const STDERR_TAIL_LINES: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EngineLifecycleState {
    Stopped,
    Starting,
    Bootstrapping,
    Negotiating,
    Ready,
    GraphSync,
    SourceSync,
    Running,
    Faulted,
    RestartPending,
}

impl EngineLifecycleState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Stopped => "STOPPED",
            Self::Starting => "STARTING",
            Self::Bootstrapping => "BOOTSTRAPPING",
            Self::Negotiating => "NEGOTIATING",
            Self::Ready => "READY",
            Self::GraphSync => "GRAPH_SYNC",
            Self::SourceSync => "SOURCE_SYNC",
            Self::Running => "RUNNING",
            Self::Faulted => "FAULTED",
            Self::RestartPending => "RESTART_PENDING",
        }
    }
}

impl std::fmt::Display for EngineLifecycleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeSnapshot {
    /// Existing Engine V2 authoritative/current-path completion rate.
    pub authoritative_fps: f32,
    /// Existing Engine V2 authoritative/current-path processing time.
    pub authoritative_frame_ms: f32,
    pub authoritative_sequence: u64,
    pub graph_fps: f32,
    pub preview_fps: f32,
    pub frame_ms: f32,
    pub preview_render_fps: f32,
    pub preview_frame_ms: f32,
    pub preview_p95_ms: f32,
    pub preview_width: usize,
    pub preview_height: usize,
    pub preview_requests: u64,
    pub preview_rendered: u64,
    pub preview_superseded: u64,
    pub preview_dropped: u64,
    pub preview_rendered_sequence: u64,
    pub preview_received_sequence: u64,
    pub dropped_frames: u64,
    pub preview_dropped_frames: u64,
    pub graph_sequence: u64,
    pub preview_sequence: u64,
    pub bridge_online: bool,
    /// Last graph generation confirmed by the engine. Kept under its original name so the
    /// existing monitor remains source-compatible while protocol v2 calls it graph_generation.
    pub bridge_generation: u64,
    pub bridge_acks: u64,
    pub bridge_rejections: u64,
    pub state_revision: u64,
    pub transport_running: bool,
    pub source_ready: bool,
    pub lifecycle: EngineLifecycleState,
}

impl Default for RuntimeSnapshot {
    fn default() -> Self {
        Self {
            authoritative_fps: 0.0,
            authoritative_frame_ms: 0.0,
            authoritative_sequence: 0,
            graph_fps: 0.0,
            preview_fps: 0.0,
            frame_ms: 0.0,
            preview_render_fps: 0.0,
            preview_frame_ms: 0.0,
            preview_p95_ms: 0.0,
            preview_width: 0,
            preview_height: 0,
            preview_requests: 0,
            preview_rendered: 0,
            preview_superseded: 0,
            preview_dropped: 0,
            preview_rendered_sequence: 0,
            preview_received_sequence: 0,
            dropped_frames: 0,
            preview_dropped_frames: 0,
            graph_sequence: 0,
            preview_sequence: 0,
            bridge_online: false,
            bridge_generation: 0,
            bridge_acks: 0,
            bridge_rejections: 0,
            state_revision: 0,
            transport_running: false,
            source_ready: false,
            lifecycle: EngineLifecycleState::Stopped,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreviewFrame {
    /// Monotonic timestamp after copying shared-memory pixels into Rust ownership.
    pub received_at: Instant,
    pub surface_generation: u64,
    pub sequence: u64,
    pub graph_revision: u64,
    pub resolution_generation: u64,
    pub source_timestamp: Option<f64>,
    pub submitted_at: Option<f64>,
    pub rendered_at: Option<f64>,
    pub render_ms: f32,
    pub width: usize,
    pub height: usize,
    pub rgb: Arc<[u8]>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct RuntimeProblem {
    pub code: String,
    #[serde(default)]
    pub path: Option<String>,
    pub message: String,
    #[serde(default)]
    pub recoverable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct PreviewNodeStatus {
    pub node_id: String,
    pub label: String,
    pub status: String,
    pub provider: String,
}

#[derive(Clone, Debug)]
pub struct RuntimeDetails {
    pub state_revision: u64,
    pub transport_running: bool,
    pub source_ready: bool,
    pub source_kind: String,
    pub source_label: String,
    pub source_position_seconds: Option<f64>,
    pub source_duration_seconds: Option<f64>,
    pub current_graph_accepted: bool,
    pub pending_command_id: Option<u64>,
    pub pending_command_type: Option<String>,
    pub last_ack_command_id: u64,
    pub last_effective_frame_sequence: u64,
    pub master_width: usize,
    pub master_height: usize,
    pub preview_graph_revision: u64,
    pub preview_resolution_generation: u64,
    pub preview_node_statuses: Vec<PreviewNodeStatus>,
    pub last_error: Option<RuntimeProblem>,
    pub backend: String,
    pub capabilities: Vec<String>,
    pub lifecycle: EngineLifecycleState,
    pub lifecycle_reason: String,
}

impl Default for RuntimeDetails {
    fn default() -> Self {
        Self {
            state_revision: 0,
            transport_running: false,
            source_ready: false,
            source_kind: "none".to_owned(),
            source_label: "NO SOURCE".to_owned(),
            source_position_seconds: None,
            source_duration_seconds: None,
            current_graph_accepted: false,
            pending_command_id: None,
            pending_command_type: None,
            last_ack_command_id: 0,
            last_effective_frame_sequence: 0,
            master_width: 0,
            master_height: 0,
            preview_graph_revision: 0,
            preview_resolution_generation: 0,
            preview_node_statuses: Vec::new(),
            last_error: None,
            backend: "offline".to_owned(),
            capabilities: Vec::new(),
            lifecycle: EngineLifecycleState::Stopped,
            lifecycle_reason: "Engine offline".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RuntimeView {
    pub snapshot: RuntimeSnapshot,
    pub details: RuntimeDetails,
    pub preview: Option<Arc<PreviewFrame>>,
}

#[derive(Clone, Debug)]
enum CommandIntent {
    ReplaceGraph,
    SetTransport { running: bool },
    SetSource { kind: String, label: String },
    Seek,
}

#[derive(Debug)]
struct QueuedCommand {
    id: u64,
    command_type: String,
    line: String,
    intent: CommandIntent,
}

#[derive(Clone, Debug)]
struct PendingCommand {
    command_type: String,
    intent: CommandIntent,
}

/// Owns the Python frame-path sidecar and exposes immutable, cheap-to-clone UI state.
///
/// Protocol failure is deliberately visible. Production never substitutes animated demo data for
/// an unavailable engine, because doing so makes a disconnected machine look operational.
pub struct RuntimeStateSource {
    state: Arc<RwLock<RuntimeView>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    commands: Option<mpsc::Sender<QueuedCommand>>,
    next_command_id: AtomicU64,
}

impl RuntimeStateSource {
    #[cfg(test)]
    pub fn demo() -> Self {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_state = Arc::clone(&state);
        let worker_stop = Arc::clone(&stop);
        let worker = thread::Builder::new()
            .name("mg-ui-state-source".into())
            .spawn(move || run_demo_source(worker_state, worker_stop))
            .expect("failed to start UI state source");
        Self {
            state,
            stop,
            worker: Some(worker),
            commands: None,
            next_command_id: AtomicU64::new(1),
        }
    }

    /// The historical method name is retained for callers, but there is no production demo
    /// fallback. A failed bridge remains offline with an actionable RuntimeProblem.
    pub fn bridge_or_demo() -> Self {
        Self::bridge_with_optional_repaint(None)
    }

    /// Starts the production bridge and wakes egui immediately whenever a new preview arrives.
    pub fn bridge_with_repaint(repaint_context: eframe::egui::Context) -> Self {
        Self::bridge_with_optional_repaint(Some(repaint_context))
    }

    fn bridge_with_optional_repaint(repaint_context: Option<eframe::egui::Context>) -> Self {
        let state = Arc::new(RwLock::new(RuntimeView {
            snapshot: RuntimeSnapshot {
                lifecycle: EngineLifecycleState::Starting,
                ..RuntimeSnapshot::default()
            },
            details: RuntimeDetails {
                lifecycle: EngineLifecycleState::Starting,
                lifecycle_reason: "Initializing engine process".to_owned(),
                ..RuntimeDetails::default()
            },
            preview: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let worker_state = Arc::clone(&state);
        let worker_stop = Arc::clone(&stop);
        let (commands, command_rx) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("mg-ui-python-bridge".into())
            .spawn(move || {
                run_bridge_source(worker_state, worker_stop, command_rx, repaint_context)
            })
            .expect("failed to start UI bridge owner");
        Self {
            state,
            stop,
            worker: Some(worker),
            commands: Some(commands),
            next_command_id: AtomicU64::new(1),
        }
    }

    pub fn replace_graph(&self, graph: &crate::preset::PresetGraph) -> Result<u64, String> {
        let graph = serde_json::to_value(graph)
            .map_err(|error| format!("could not serialize graph command: {error}"))?;
        self.enqueue_command(
            "replace_graph",
            CommandIntent::ReplaceGraph,
            [("graph", graph)],
        )
    }

    pub fn set_transport(&self, running: bool) -> Result<u64, String> {
        self.enqueue_command(
            "set_transport",
            CommandIntent::SetTransport { running },
            [("running", Value::Bool(running))],
        )
    }

    pub fn set_source(&self, kind: &str, path: Option<&str>) -> Result<u64, String> {
        let kind = kind.trim();
        if kind.is_empty() {
            return Err("source kind cannot be empty".to_owned());
        }
        let label = path
            .filter(|value| !value.trim().is_empty())
            .map(source_label)
            .unwrap_or_else(|| kind.to_ascii_uppercase());
        self.enqueue_command(
            "set_source",
            CommandIntent::SetSource {
                kind: kind.to_owned(),
                label,
            },
            [
                ("kind", Value::String(kind.to_owned())),
                (
                    "path",
                    path.map_or(Value::Null, |value| Value::String(value.to_owned())),
                ),
            ],
        )
    }

    pub fn seek(&self, seconds: f64) -> Result<u64, String> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err("seek position must be a finite, non-negative number".to_owned());
        }
        self.enqueue_command("seek", CommandIntent::Seek, [("seconds", json!(seconds))])
    }

    #[cfg(test)]
    pub fn read(&self) -> RuntimeSnapshot {
        self.state
            .read()
            .map(|value| value.snapshot)
            .unwrap_or_default()
    }

    pub fn read_view(&self) -> RuntimeView {
        self.state
            .read()
            .map(|value| value.clone())
            .unwrap_or_default()
    }

    fn enqueue_command<const N: usize>(
        &self,
        command_type: &str,
        intent: CommandIntent,
        payload: [(&str, Value); N],
    ) -> Result<u64, String> {
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| "Python bridge is not configured".to_owned())?;
        let (id, expected_state_revision) = {
            let mut state = self
                .state
                .write()
                .map_err(|_| "Python bridge state lock is poisoned".to_owned())?;
            if let Some(pending_id) = state.details.pending_command_id {
                return Err(format!(
                    "Python bridge is busy waiting for command {pending_id}; retry {command_type} after its acknowledgement"
                ));
            }
            let id = self.next_command_id.fetch_add(1, Ordering::Relaxed);
            state.details.pending_command_id = Some(id);
            state.details.pending_command_type = Some(command_type.to_owned());
            if matches!(intent, CommandIntent::ReplaceGraph) {
                state.details.current_graph_accepted = false;
            }
            state.details.last_error = None;
            (id, state.details.state_revision)
        };

        let mut object = Map::new();
        object.insert("version".to_owned(), json!(BRIDGE_PROTOCOL_VERSION));
        object.insert("id".to_owned(), json!(id));
        object.insert("type".to_owned(), Value::String(command_type.to_owned()));
        object.insert(
            "expected_state_revision".to_owned(),
            json!(expected_state_revision),
        );
        for (key, value) in payload {
            object.insert(key.to_owned(), value);
        }
        let queued = QueuedCommand {
            id,
            command_type: command_type.to_owned(),
            line: Value::Object(object).to_string(),
            intent,
        };
        let result = commands
            .send(queued)
            .map_err(|_| "Python bridge command channel is closed".to_owned());
        if let Err(message) = &result
            && let Ok(mut state) = self.state.write()
        {
            if state.details.pending_command_id == Some(id) {
                state.details.pending_command_id = None;
                state.details.pending_command_type = None;
            }
            state.details.last_error = Some(RuntimeProblem {
                code: "bridge.command_channel_closed".to_owned(),
                path: None,
                message: message.clone(),
                recoverable: false,
            });
        }
        result.map(|_| id)
    }
}

fn source_label(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(path)
        .to_owned()
}

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type")]
enum BridgeEvent {
    #[serde(rename = "ready")]
    Ready {
        version: u32,
        backend: String,
        state_revision: u64,
        #[serde(default)]
        capabilities: Value,
        #[serde(default)]
        graph_generation: u64,
        #[serde(default)]
        transport_running: bool,
        #[serde(default)]
        source_kind: String,
        #[serde(default)]
        source_label: String,
        #[serde(default)]
        source_ready: bool,
        #[serde(default, alias = "position_seconds", alias = "media_position_seconds")]
        source_position_seconds: Option<f64>,
        #[serde(default, alias = "duration_seconds", alias = "media_duration_seconds")]
        source_duration_seconds: Option<f64>,
    },
    #[serde(rename = "ack")]
    Ack {
        version: u32,
        id: u64,
        command_type: String,
        accepted: bool,
        state_revision: u64,
        #[serde(default)]
        effective_frame_sequence: u64,
        #[serde(default)]
        graph_generation: u64,
        #[serde(default)]
        error: Option<RuntimeProblem>,
    },
    #[serde(rename = "preview_surface")]
    PreviewSurface {
        version: u32,
        path: String,
        generation: u64,
        slots: usize,
        slot_bytes: usize,
        max_width: usize,
        max_height: usize,
        pixel_format: String,
    },
    #[serde(rename = "preview_frame")]
    PreviewFrame {
        version: u32,
        slot: usize,
        generation: u64,
        sequence: u64,
        width: usize,
        height: usize,
        stride: usize,
        nbytes: usize,
        #[serde(default)]
        timestamp: f64,
        #[serde(default)]
        graph_revision: u64,
        #[serde(default)]
        resolution_generation: u64,
        #[serde(default)]
        source_timestamp: Option<f64>,
        #[serde(default)]
        submitted_at: Option<f64>,
        #[serde(default)]
        rendered_at: Option<f64>,
        #[serde(default)]
        render_ms: f32,
    },
    #[serde(rename = "snapshot")]
    Snapshot {
        version: u32,
        state_revision: u64,
        graph_generation: u64,
        #[serde(default, alias = "frame_sequence", alias = "output_sequence")]
        graph_sequence: u64,
        #[serde(default)]
        preview_sequence: u64,
        #[serde(default)]
        preview_rendered_sequence: Option<u64>,
        #[serde(default, alias = "measured_graph_fps")]
        graph_fps: f32,
        #[serde(default, alias = "measured_preview_fps")]
        preview_fps: f32,
        #[serde(default)]
        preview_render_fps: Option<f32>,
        #[serde(default, alias = "measured_frame_ms")]
        frame_ms: f32,
        #[serde(default)]
        preview_frame_ms: f32,
        #[serde(default, alias = "preview_p95_frame_ms")]
        preview_p95_ms: f32,
        #[serde(default)]
        preview_width: Option<usize>,
        #[serde(default)]
        preview_height: Option<usize>,
        #[serde(default)]
        preview_requests: u64,
        #[serde(default)]
        preview_rendered: u64,
        #[serde(default)]
        preview_superseded: u64,
        #[serde(default)]
        preview_dropped: u64,
        #[serde(default)]
        preview_graph_revision: u64,
        #[serde(default)]
        preview_resolution_generation: u64,
        #[serde(default)]
        preview_node_statuses: Vec<PreviewNodeStatus>,
        #[serde(default)]
        dropped_frames: u64,
        #[serde(default, alias = "preview_drops")]
        preview_dropped_frames: u64,
        transport_running: bool,
        source_ready: bool,
        source_kind: String,
        #[serde(default)]
        source_label: Option<String>,
        #[serde(default, alias = "position_seconds", alias = "media_position_seconds")]
        source_position_seconds: Option<f64>,
        #[serde(default, alias = "duration_seconds", alias = "media_duration_seconds")]
        source_duration_seconds: Option<f64>,
        #[serde(default)]
        width: Option<usize>,
        #[serde(default)]
        height: Option<usize>,
        #[serde(default)]
        last_error: Option<RuntimeProblem>,
    },
    #[serde(rename = "runtime_error")]
    RuntimeError {
        version: u32,
        state_revision: u64,
        graph_generation: u64,
        #[serde(default)]
        effective_frame_sequence: u64,
        error: RuntimeProblem,
    },
}

struct PreviewSurfaceMapping {
    _path: PathBuf,
    generation: u64,
    slots: usize,
    slot_bytes: usize,
    max_width: usize,
    max_height: usize,
    mmap: Mmap,
}

fn run_bridge_source(
    state: Arc<RwLock<RuntimeView>>,
    stop: Arc<AtomicBool>,
    commands: mpsc::Receiver<QueuedCommand>,
    repaint_context: Option<eframe::egui::Context>,
) {
    if let Err(message) = run_bridge_process(&state, &stop, &commands, repaint_context.as_ref())
        && !stop.load(Ordering::Acquire)
    {
        set_offline_problem(&state, "bridge.process_failed", message, false);
    }
}

fn bundled_bridge_executable() -> Result<Option<PathBuf>, String> {
    if let Some(configured) = std::env::var_os("MEGAGLITCH_UI_BRIDGE_EXECUTABLE") {
        let path = PathBuf::from(configured);
        if path.is_file() {
            return Ok(Some(path));
        }
        return Err(format!(
            "MEGAGLITCH_UI_BRIDGE_EXECUTABLE does not name a file: {}",
            path.display()
        ));
    }
    let executable = std::env::current_exe()
        .map_err(|error| format!("could not locate the Rust UI executable: {error}"))?;
    let Some(parent) = executable.parent() else {
        return Ok(None);
    };
    for candidate in [
        parent.join("engine-v2").join("megaglitch-engine-v2.exe"),
        parent.join("megaglitch-engine-v2.exe"),
    ] {
        if candidate.is_file() {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

pub fn resolve_repository_root() -> Result<PathBuf, String> {
    if let Some(configured) = std::env::var_os("MEGAGLITCH_REPOSITORY_ROOT")
        .or_else(|| std::env::var_os("MEGAGLITCH_ROOT"))
    {
        let path = PathBuf::from(configured);
        if path.is_dir() {
            return Ok(path);
        }
        return Err(format!(
            "configured repository root does not exist: {}",
            path.display()
        ));
    }
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join("tools").join("rust_ui_bridge.py").is_file() {
            return Ok(cwd);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            if parent.join("tools").join("rust_ui_bridge.py").is_file() {
                return Ok(parent.to_path_buf());
            }
            if let Some(grandparent) = parent.parent() {
                if grandparent
                    .join("tools")
                    .join("rust_ui_bridge.py")
                    .is_file()
                {
                    return Ok(grandparent.to_path_buf());
                }
            }
        }
    }
    let manifest_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    if manifest_root
        .join("tools")
        .join("rust_ui_bridge.py")
        .is_file()
    {
        if let Ok(canonical) = manifest_root.canonicalize() {
            return Ok(canonical);
        }
        return Ok(manifest_root);
    }
    Err("could not locate MegaGlitch repository root containing tools/rust_ui_bridge.py".to_owned())
}

pub fn configure_bridge_environment(command: &mut Command, repo_root: &Path) {
    command.current_dir(repo_root);
    command.env("MEGAGLITCH_REPOSITORY_ROOT", repo_root);

    // Ensure PYTHONPATH prioritizes the repository root and development OpenCV Python path
    let mut python_paths = vec![repo_root.to_path_buf()];
    if let Some(configured_py) = std::env::var_os("MEGAGLITCH_OPENCV_PYTHON") {
        let p = PathBuf::from(configured_py);
        if p.is_dir() {
            python_paths.push(p);
        }
    } else {
        let default_opencv_py =
            PathBuf::from(r"C:\src\opencv-cuda\build_py312\lib\python3\Release");
        if default_opencv_py.is_dir() {
            python_paths.push(default_opencv_py);
        }
    }
    if let Some(existing_pythonpath) = std::env::var_os("PYTHONPATH") {
        for entry in std::env::split_paths(&existing_pythonpath) {
            if !python_paths.contains(&entry) {
                python_paths.push(entry);
            }
        }
    }
    if let Ok(joined_pythonpath) = std::env::join_paths(&python_paths) {
        command.env("PYTHONPATH", joined_pythonpath);
    }

    // Ensure PATH prioritizes OpenCV DLL and CUDA bin directories without destroying user PATH
    let mut bin_paths = Vec::new();
    if let Some(configured_bin) = std::env::var_os("MEGAGLITCH_OPENCV_BIN") {
        let p = PathBuf::from(configured_bin);
        if p.is_dir() {
            bin_paths.push(p);
        }
    } else {
        let default_opencv_bin = PathBuf::from(r"C:\src\opencv-cuda\build_py312\bin\Release");
        if default_opencv_bin.is_dir() {
            bin_paths.push(default_opencv_bin);
        }
    }
    if let Some(configured_cuda) = std::env::var_os("MEGAGLITCH_CUDA_BIN") {
        let p = PathBuf::from(configured_cuda);
        if p.is_dir() {
            bin_paths.push(p);
        }
    } else {
        let default_cuda_bin =
            PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin");
        if default_cuda_bin.is_dir() {
            bin_paths.push(default_cuda_bin);
        }
    }
    if let Some(existing_path) = std::env::var_os("PATH") {
        for entry in std::env::split_paths(&existing_path) {
            if !bin_paths.contains(&entry) {
                bin_paths.push(entry);
            }
        }
    }
    if let Ok(joined_path) = std::env::join_paths(&bin_paths) {
        command.env("PATH", joined_path);
    }

    // Ensure TEMP and TMP point to a valid writable temporary directory on V: if system C: is full
    let repo_tmp = repo_root.join(".tmp");
    if repo_tmp.is_dir() {
        command.env("TEMP", &repo_tmp);
        command.env("TMP", &repo_tmp);
    } else {
        let v_tmp = PathBuf::from(r"V:\.tmp");
        if v_tmp.is_dir() {
            command.env("TEMP", &v_tmp);
            command.env("TMP", &v_tmp);
        }
    }
}

fn set_lifecycle(
    state: &Arc<RwLock<RuntimeView>>,
    lifecycle: EngineLifecycleState,
    reason: impl Into<String>,
) {
    if let Ok(mut value) = state.write() {
        value.snapshot.lifecycle = lifecycle;
        value.details.lifecycle = lifecycle;
        value.details.lifecycle_reason = reason.into();
    }
}

fn run_bridge_process(
    state: &Arc<RwLock<RuntimeView>>,
    stop: &Arc<AtomicBool>,
    commands: &mpsc::Receiver<QueuedCommand>,
    repaint_context: Option<&eframe::egui::Context>,
) -> Result<(), String> {
    set_lifecycle(
        state,
        EngineLifecycleState::Bootstrapping,
        "Spawning engine process",
    );
    let (mut bridge_command, launch_description) =
        if let Some(executable) = bundled_bridge_executable()? {
            let mut command = Command::new(&executable);
            if let Some(engine_directory) = executable.parent() {
                command.current_dir(engine_directory);
            }
            (
                command,
                format!("bundled Engine V2 {}", executable.display()),
            )
        } else {
            let repo_root = resolve_repository_root()?;
            let python = std::env::var("MEGAGLITCH_PYTHON").unwrap_or_else(|_| "python".to_owned());
            let script = std::env::var("MEGAGLITCH_UI_BRIDGE_SCRIPT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| repo_root.join("tools").join("rust_ui_bridge.py"));
            let mut command = Command::new(&python);
            command.arg("-u").arg(&script);
            configure_bridge_environment(&mut command, &repo_root);
            (
                command,
                format!("Python {python:?} with {}", script.display()),
            )
        };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        bridge_command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = bridge_command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start {launch_description}: {error}"))?;
    set_lifecycle(
        state,
        EngineLifecycleState::Negotiating,
        "Awaiting protocol handshake",
    );
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Python bridge stdin unavailable".to_owned())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Python bridge stdout unavailable".to_owned())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Python bridge stderr unavailable".to_owned())?;
    let (events_tx, events_rx) = mpsc::channel();
    let reader = thread::Builder::new()
        .name("mg-ui-python-bridge-reader".into())
        .spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else {
                    break;
                };
                if events_tx.send(line).is_err() {
                    break;
                }
            }
        })
        .map_err(|error| format!("could not start Python bridge reader: {error}"))?;
    let stderr_tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
    let worker_stderr_tail = Arc::clone(&stderr_tail);
    let stderr_reader = thread::Builder::new()
        .name("mg-ui-python-bridge-stderr".into())
        .spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else {
                    break;
                };
                if let Ok(mut tail) = worker_stderr_tail.lock() {
                    if tail.len() == STDERR_TAIL_LINES {
                        tail.pop_front();
                    }
                    tail.push_back(line);
                }
            }
        })
        .map_err(|error| format!("could not start Python bridge stderr reader: {error}"))?;

    let mut pending = HashMap::<u64, PendingCommand>::new();
    let mut preview_surface = None;
    let startup_deadline = Instant::now() + bridge_startup_timeout();
    let mut handshake_complete = false;
    let result = (|| {
        while !stop.load(Ordering::Acquire) {
            while let Ok(command) = commands.try_recv() {
                writeln!(stdin, "{}", command.line)
                    .and_then(|_| stdin.flush())
                    .map_err(|error| format!("Python bridge command write failed: {error}"))?;
                pending.insert(
                    command.id,
                    PendingCommand {
                        command_type: command.command_type,
                        intent: command.intent,
                    },
                );
            }
            match events_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(line) => {
                    let event: BridgeEvent = serde_json::from_str(&line).map_err(|error| {
                        format!("invalid Python bridge event ({error}): {line}")
                    })?;
                    match event {
                        BridgeEvent::PreviewFrame {
                            version,
                            slot,
                            generation,
                            sequence,
                            width,
                            height,
                            stride,
                            nbytes,
                            timestamp,
                            graph_revision,
                            resolution_generation,
                            source_timestamp,
                            submitted_at,
                            rendered_at,
                            render_ms,
                        } => {
                            check_version(version, "preview frame")?;
                            let copy_result = copy_preview_frame(
                                state,
                                preview_surface.as_ref(),
                                PreviewFrameDescriptor {
                                    slot,
                                    generation,
                                    sequence,
                                    width,
                                    height,
                                    stride,
                                    nbytes,
                                    timestamp,
                                    graph_revision,
                                    resolution_generation,
                                    source_timestamp,
                                    submitted_at,
                                    rendered_at,
                                    render_ms,
                                },
                            );
                            let release = json!({
                                "version": BRIDGE_PROTOCOL_VERSION,
                                "type": "release_preview_slot",
                                "slot": slot,
                                "generation": generation,
                                "sequence": sequence,
                            });
                            writeln!(stdin, "{release}")
                                .and_then(|_| stdin.flush())
                                .map_err(|error| {
                                    format!("Python preview-slot release failed: {error}")
                                })?;
                            match copy_result {
                                Ok(()) => {
                                    if let Some(context) = repaint_context {
                                        context.request_repaint();
                                    }
                                }
                                Err(message) => record_preview_problem(state, message),
                            }
                        }
                        BridgeEvent::PreviewSurface {
                            version,
                            path,
                            generation,
                            slots,
                            slot_bytes,
                            max_width,
                            max_height,
                            pixel_format,
                        } => {
                            check_version(version, "preview surface")?;
                            preview_surface = Some(map_preview_surface(
                                path,
                                generation,
                                slots,
                                slot_bytes,
                                max_width,
                                max_height,
                                pixel_format,
                            )?);
                        }
                        event => {
                            let ready = matches!(&event, BridgeEvent::Ready { .. });
                            apply_control_event(state, event, &mut pending)?;
                            handshake_complete |= ready;
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("Python bridge output closed".to_owned());
                }
            }
            if let Some(status) = child
                .try_wait()
                .map_err(|error| format!("Python bridge status failed: {error}"))?
            {
                return Err(format!("Python bridge exited with {status}"));
            }
            if !handshake_complete && Instant::now() >= startup_deadline {
                return Err(format!(
                    "Engine V2 did not complete protocol handshake within {} ms",
                    bridge_startup_timeout().as_millis()
                ));
            }
        }
        Ok(())
    })();

    // The Python owner unlinks the mmap backing file during shutdown. Windows cannot complete
    // that unlink while the Rust consumer still has the surface mapped, so release the consumer
    // mapping before asking the owner to tear the surface down.
    let preview_surface_path = preview_surface
        .take()
        .map(|surface: PreviewSurfaceMapping| {
            let path = surface._path.clone();
            drop(surface);
            path
        });
    let expected_state_revision = state
        .read()
        .map(|value| value.details.state_revision)
        .unwrap_or_default();
    let _ = writeln!(
        stdin,
        "{}",
        json!({
            "version": BRIDGE_PROTOCOL_VERSION,
            "id": u64::MAX,
            "type": "shutdown",
            "expected_state_revision": expected_state_revision,
        })
    );
    let _ = stdin.flush();
    let deadline = Instant::now() + BRIDGE_SHUTDOWN_TIMEOUT;
    while Instant::now() < deadline {
        if child.try_wait().ok().flatten().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    if child.try_wait().ok().flatten().is_none() {
        let _ = child.kill();
    }
    let _ = child.wait();
    let _ = reader.join();
    let _ = stderr_reader.join();
    if let Some(path) = preview_surface_path {
        // Python normally owns removal. This is a bounded best-effort cleanup for forced child
        // termination or a platform-specific delayed unlink after the mapping is gone.
        if path.is_file() {
            let _ = std::fs::remove_file(path);
        }
    }
    if let Ok(mut value) = state.write() {
        value.snapshot.bridge_online = false;
        value.details.pending_command_id = None;
        value.details.pending_command_type = None;
        value.details.backend = "offline".to_owned();
    }

    result.map_err(|message| append_stderr_tail(message, &stderr_tail))
}

fn append_stderr_tail(message: String, stderr_tail: &Arc<Mutex<VecDeque<String>>>) -> String {
    let Ok(tail) = stderr_tail.lock() else {
        return message;
    };
    if tail.is_empty() {
        message
    } else {
        format!(
            "{message}\nPython bridge stderr:\n{}",
            tail.iter().cloned().collect::<Vec<_>>().join("\n")
        )
    }
}

fn check_version(version: u32, event_name: &str) -> Result<(), String> {
    if version == BRIDGE_PROTOCOL_VERSION {
        Ok(())
    } else {
        Err(format!(
            "Python bridge {event_name} version mismatch: expected {BRIDGE_PROTOCOL_VERSION}, got {version}"
        ))
    }
}

fn bridge_startup_timeout() -> Duration {
    std::env::var("MEGAGLITCH_UI_BRIDGE_STARTUP_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|milliseconds| *milliseconds >= 1_000)
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_BRIDGE_STARTUP_TIMEOUT)
}

fn apply_control_event(
    state: &Arc<RwLock<RuntimeView>>,
    event: BridgeEvent,
    pending: &mut HashMap<u64, PendingCommand>,
) -> Result<(), String> {
    let mut value = state
        .write()
        .map_err(|_| "Python bridge state lock poisoned".to_owned())?;
    match event {
        BridgeEvent::Ready {
            version,
            backend,
            state_revision,
            capabilities,
            graph_generation,
            transport_running,
            source_kind,
            source_label,
            source_ready,
            source_position_seconds,
            source_duration_seconds,
        } => {
            check_version(version, "handshake")?;
            if backend.trim().is_empty() {
                return Err("Python bridge handshake omitted backend identity".to_owned());
            }
            value.snapshot.bridge_online = true;
            value.snapshot.state_revision = state_revision;
            value.snapshot.bridge_generation = graph_generation;
            value.snapshot.transport_running = transport_running;
            value.snapshot.source_ready = source_ready;
            value.snapshot.lifecycle = EngineLifecycleState::Ready;
            value.details.state_revision = state_revision;
            value.details.transport_running = transport_running;
            value.details.source_ready = source_ready;
            value.details.lifecycle = EngineLifecycleState::Ready;
            value.details.lifecycle_reason = format!("Engine ready (backend: {backend})");
            value.details.source_kind = if source_kind.is_empty() {
                "none".to_owned()
            } else {
                source_kind
            };
            value.details.source_label = if source_label.is_empty() {
                "NO SOURCE".to_owned()
            } else {
                source_label
            };
            value.details.source_position_seconds = valid_media_position(source_position_seconds);
            value.details.source_duration_seconds = valid_media_duration(source_duration_seconds);
            value.details.backend = backend;
            value.details.capabilities = capability_labels(&capabilities);
            value.details.last_error = None;
        }
        BridgeEvent::Ack {
            version,
            id,
            command_type,
            accepted,
            state_revision,
            effective_frame_sequence,
            graph_generation,
            error,
        } => {
            check_version(version, "acknowledgement")?;
            let queued = pending
                .remove(&id)
                .ok_or_else(|| format!("Python bridge acknowledged unknown command id {id}"))?;
            if queued.command_type != command_type {
                return Err(format!(
                    "Python bridge acknowledgement type mismatch for command {id}: sent {}, received {command_type}",
                    queued.command_type
                ));
            }
            if state_revision < value.details.state_revision {
                return Err(format!(
                    "Python bridge state revision regressed from {} to {state_revision}",
                    value.details.state_revision
                ));
            }
            value.snapshot.bridge_online = true;
            value.snapshot.state_revision = state_revision;
            value.details.state_revision = state_revision;
            value.details.last_ack_command_id = id;
            value.details.last_effective_frame_sequence = effective_frame_sequence;
            if value.details.pending_command_id == Some(id) {
                value.details.pending_command_id = None;
                value.details.pending_command_type = None;
            }
            if accepted {
                value.snapshot.bridge_acks = value.snapshot.bridge_acks.saturating_add(1);
                value.snapshot.bridge_generation = graph_generation;
                value.details.last_error = None;
                match queued.intent {
                    CommandIntent::ReplaceGraph => {
                        value.details.current_graph_accepted = true;
                    }
                    CommandIntent::SetTransport { running } => {
                        value.snapshot.transport_running = running;
                        value.details.transport_running = running;
                    }
                    CommandIntent::SetSource { kind, label } => {
                        value.snapshot.source_ready = true;
                        value.details.source_ready = true;
                        value.details.source_kind = kind;
                        value.details.source_label = label;
                        value.details.source_position_seconds = None;
                        value.details.source_duration_seconds = None;
                    }
                    CommandIntent::Seek => {}
                }
                if value.details.transport_running
                    && value.details.source_ready
                    && value.details.current_graph_accepted
                {
                    value.snapshot.lifecycle = EngineLifecycleState::Running;
                    value.details.lifecycle = EngineLifecycleState::Running;
                    value.details.lifecycle_reason = "Processing frames".to_owned();
                } else {
                    value.snapshot.lifecycle = EngineLifecycleState::Ready;
                    value.details.lifecycle = EngineLifecycleState::Ready;
                    value.details.lifecycle_reason = "Engine ready".to_owned();
                }
            } else {
                value.snapshot.bridge_rejections =
                    value.snapshot.bridge_rejections.saturating_add(1);
                if matches!(queued.intent, CommandIntent::ReplaceGraph) {
                    value.details.current_graph_accepted = false;
                }
                value.details.last_error = Some(error.unwrap_or_else(|| RuntimeProblem {
                    code: "bridge.command_rejected".to_owned(),
                    path: Some(command_type),
                    message: format!("Python bridge rejected command {id}"),
                    recoverable: true,
                }));
            }
        }
        BridgeEvent::Snapshot {
            version,
            state_revision,
            graph_generation,
            graph_sequence,
            preview_sequence,
            preview_rendered_sequence,
            graph_fps,
            preview_fps,
            preview_render_fps,
            frame_ms,
            preview_frame_ms,
            preview_p95_ms,
            preview_width,
            preview_height,
            preview_requests,
            preview_rendered,
            preview_superseded,
            preview_dropped,
            preview_graph_revision,
            preview_resolution_generation,
            preview_node_statuses,
            dropped_frames,
            preview_dropped_frames,
            transport_running,
            source_ready,
            source_kind,
            source_label,
            source_position_seconds,
            source_duration_seconds,
            width,
            height,
            last_error,
        } => {
            check_version(version, "snapshot")?;
            if state_revision < value.details.state_revision {
                return Err(format!(
                    "Python bridge snapshot state revision regressed from {} to {state_revision}",
                    value.details.state_revision
                ));
            }
            value.snapshot.bridge_online = true;
            value.snapshot.state_revision = state_revision;
            value.snapshot.bridge_generation = graph_generation;
            value.snapshot.graph_sequence = graph_sequence;
            value.snapshot.preview_sequence = preview_sequence;
            value.snapshot.authoritative_sequence = graph_sequence;
            value.snapshot.preview_rendered_sequence =
                preview_rendered_sequence.unwrap_or(preview_sequence);
            value.snapshot.authoritative_fps = graph_fps;
            value.snapshot.authoritative_frame_ms = frame_ms;
            value.snapshot.graph_fps = graph_fps;
            value.snapshot.frame_ms = frame_ms;
            let has_explicit_preview_telemetry = preview_render_fps.is_some();
            let render_fps = preview_render_fps.unwrap_or(preview_fps);
            value.snapshot.preview_fps = render_fps;
            value.snapshot.preview_render_fps = render_fps;
            value.snapshot.preview_frame_ms = if has_explicit_preview_telemetry {
                preview_frame_ms
            } else {
                frame_ms
            };
            value.snapshot.preview_p95_ms = preview_p95_ms;
            if let Some(width) = preview_width {
                value.snapshot.preview_width = width;
            }
            if let Some(height) = preview_height {
                value.snapshot.preview_height = height;
            }
            value.snapshot.preview_requests = preview_requests;
            value.snapshot.preview_rendered = if has_explicit_preview_telemetry {
                preview_rendered
            } else {
                preview_sequence
            };
            value.snapshot.preview_superseded = preview_superseded;
            value.snapshot.preview_dropped = preview_dropped.max(preview_dropped_frames);
            value.snapshot.dropped_frames = dropped_frames;
            value.snapshot.preview_dropped_frames = value.snapshot.preview_dropped;
            value.snapshot.transport_running = transport_running;
            value.snapshot.source_ready = source_ready;
            if transport_running && source_ready {
                value.snapshot.lifecycle = EngineLifecycleState::Running;
                value.details.lifecycle = EngineLifecycleState::Running;
            } else {
                value.snapshot.lifecycle = EngineLifecycleState::Ready;
                value.details.lifecycle = EngineLifecycleState::Ready;
            }
            value.details.state_revision = state_revision;
            value.details.transport_running = transport_running;
            value.details.source_ready = source_ready;
            value.details.source_kind = source_kind.clone();
            value.details.preview_graph_revision = preview_graph_revision;
            value.details.preview_resolution_generation = preview_resolution_generation;
            value.details.preview_node_statuses = preview_node_statuses;
            if let Some(w) = width {
                value.details.master_width = w;
            }
            if let Some(h) = height {
                value.details.master_height = h;
            }
            if let Some(label) = source_label.filter(|label| !label.is_empty()) {
                value.details.source_label = label;
            } else if !source_ready {
                value.details.source_label = "NO SOURCE".to_owned();
            } else if value.details.source_label == "NO SOURCE" {
                value.details.source_label = source_kind.to_ascii_uppercase();
            }
            value.details.source_position_seconds = valid_media_position(source_position_seconds);
            value.details.source_duration_seconds = valid_media_duration(source_duration_seconds);
            if let Some(error) = last_error {
                value.details.last_error = Some(error);
            }
        }
        BridgeEvent::RuntimeError {
            version,
            state_revision,
            graph_generation,
            effective_frame_sequence,
            error,
        } => {
            check_version(version, "runtime error")?;
            if state_revision < value.details.state_revision {
                return Err(format!(
                    "Python bridge runtime-error state revision regressed from {} to {state_revision}",
                    value.details.state_revision
                ));
            }
            value.snapshot.bridge_online = true;
            value.snapshot.state_revision = state_revision;
            value.snapshot.bridge_generation = graph_generation;
            value.snapshot.dropped_frames = value.snapshot.dropped_frames.saturating_add(1);
            value.details.state_revision = state_revision;
            value.details.last_effective_frame_sequence = effective_frame_sequence;
            value.details.last_error = Some(error);
        }
        BridgeEvent::PreviewSurface { .. } | BridgeEvent::PreviewFrame { .. } => {
            return Err("preview event reached control-event handler".to_owned());
        }
    }
    Ok(())
}

fn valid_media_position(value: Option<f64>) -> Option<f64> {
    value.filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
}

fn valid_wall_timestamp(value: Option<f64>) -> Option<f64> {
    value.filter(|seconds| seconds.is_finite() && *seconds >= 0.0)
}

fn finite_non_negative(value: f32) -> f32 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    }
}

fn valid_media_duration(value: Option<f64>) -> Option<f64> {
    value.filter(|seconds| seconds.is_finite() && *seconds > 0.0)
}

fn capability_labels(value: &Value) -> Vec<String> {
    fn visit(prefix: &str, value: &Value, labels: &mut Vec<String>) {
        match value {
            Value::Object(object) => {
                for (key, child) in object {
                    let next = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    visit(&next, child, labels);
                }
            }
            Value::Array(items) => {
                for item in items {
                    visit(prefix, item, labels);
                }
            }
            Value::String(text) => labels.push(if prefix.is_empty() {
                text.clone()
            } else {
                format!("{prefix}:{text}")
            }),
            Value::Bool(true) => labels.push(prefix.to_owned()),
            Value::Number(number) => labels.push(format!("{prefix}:{number}")),
            Value::Null | Value::Bool(false) => {}
        }
    }

    let mut labels = Vec::new();
    visit("", value, &mut labels);
    labels
}

fn map_preview_surface(
    path: String,
    generation: u64,
    slots: usize,
    slot_bytes: usize,
    max_width: usize,
    max_height: usize,
    pixel_format: String,
) -> Result<PreviewSurfaceMapping, String> {
    let normalized = pixel_format.to_ascii_lowercase().replace(['-', '_'], "");
    if !matches!(normalized.as_str(), "rgb" | "rgb8" | "rgb24") {
        return Err(format!(
            "unsupported Python preview pixel format {pixel_format:?}; expected packed RGB8"
        ));
    }
    if slots == 0 || slot_bytes == 0 || max_width == 0 || max_height == 0 {
        return Err("Python preview surface has a zero-sized dimension".to_owned());
    }
    let mapped_bytes = slots
        .checked_mul(slot_bytes)
        .ok_or_else(|| "Python preview surface byte size overflowed usize".to_owned())?;
    let path = PathBuf::from(path);
    let file = File::open(&path).map_err(|error| {
        format!(
            "could not open Python preview surface {}: {error}",
            path.display()
        )
    })?;
    let file_bytes = file
        .metadata()
        .map_err(|error| {
            format!(
                "could not inspect Python preview surface {}: {error}",
                path.display()
            )
        })?
        .len();
    if file_bytes < mapped_bytes as u64 {
        return Err(format!(
            "Python preview surface {} is {file_bytes} bytes; protocol announced {mapped_bytes}",
            path.display()
        ));
    }
    // SAFETY: the bridge announces a fixed-size file and keeps it alive for the surface
    // generation. Rust maps only the validated byte range and copies every announced frame into
    // owned memory before releasing the producer slot.
    let mmap = unsafe { MmapOptions::new().len(mapped_bytes).map(&file) }.map_err(|error| {
        format!(
            "could not map Python preview surface {}: {error}",
            path.display()
        )
    })?;
    Ok(PreviewSurfaceMapping {
        _path: path,
        generation,
        slots,
        slot_bytes,
        max_width,
        max_height,
        mmap,
    })
}

#[derive(Default)]
struct PreviewFrameDescriptor {
    slot: usize,
    generation: u64,
    sequence: u64,
    width: usize,
    height: usize,
    stride: usize,
    nbytes: usize,
    timestamp: f64,
    graph_revision: u64,
    resolution_generation: u64,
    source_timestamp: Option<f64>,
    submitted_at: Option<f64>,
    rendered_at: Option<f64>,
    render_ms: f32,
}

fn copy_preview_frame(
    state: &Arc<RwLock<RuntimeView>>,
    surface: Option<&PreviewSurfaceMapping>,
    frame: PreviewFrameDescriptor,
) -> Result<(), String> {
    let surface = surface
        .ok_or_else(|| "Python announced a preview frame before its preview surface".to_owned())?;
    if frame.generation != surface.generation {
        return Err(format!(
            "stale preview generation {}; current surface generation is {}",
            frame.generation, surface.generation
        ));
    }
    if frame.slot >= surface.slots {
        return Err(format!(
            "preview slot {} is outside the announced {} slots",
            frame.slot, surface.slots
        ));
    }
    if frame.width == 0
        || frame.height == 0
        || frame.width > surface.max_width
        || frame.height > surface.max_height
    {
        return Err(format!(
            "preview frame {}x{} is outside surface maximum {}x{}",
            frame.width, frame.height, surface.max_width, surface.max_height
        ));
    }
    let row_bytes = frame
        .width
        .checked_mul(3)
        .ok_or_else(|| "preview row byte size overflowed usize".to_owned())?;
    if frame.stride < row_bytes {
        return Err(format!(
            "preview stride {} is smaller than packed RGB row size {row_bytes}",
            frame.stride
        ));
    }
    let required_bytes = frame
        .height
        .saturating_sub(1)
        .checked_mul(frame.stride)
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or_else(|| "preview frame byte size overflowed usize".to_owned())?;
    if frame.nbytes < required_bytes || frame.nbytes > surface.slot_bytes {
        return Err(format!(
            "preview nbytes {} cannot contain {required_bytes} bytes and fit slot size {}",
            frame.nbytes, surface.slot_bytes
        ));
    }
    let slot_offset = frame
        .slot
        .checked_mul(surface.slot_bytes)
        .ok_or_else(|| "preview slot offset overflowed usize".to_owned())?;
    let frame_end = slot_offset
        .checked_add(required_bytes)
        .ok_or_else(|| "preview frame mapping range overflowed usize".to_owned())?;
    if frame_end > surface.mmap.len() {
        return Err("preview frame range is outside mapped surface".to_owned());
    }
    let compact_bytes = row_bytes
        .checked_mul(frame.height)
        .ok_or_else(|| "compact preview byte size overflowed usize".to_owned())?;
    let rgb = if frame.stride == row_bytes {
        surface.mmap[slot_offset..slot_offset + compact_bytes].to_vec()
    } else {
        let mut rgb = Vec::with_capacity(compact_bytes);
        for row in 0..frame.height {
            let start = slot_offset + row * frame.stride;
            rgb.extend_from_slice(&surface.mmap[start..start + row_bytes]);
        }
        rgb
    };
    let preview = Arc::new(PreviewFrame {
        received_at: Instant::now(),
        surface_generation: frame.generation,
        sequence: frame.sequence,
        graph_revision: frame.graph_revision,
        resolution_generation: frame.resolution_generation,
        source_timestamp: valid_wall_timestamp(frame.source_timestamp),
        submitted_at: valid_wall_timestamp(frame.submitted_at),
        rendered_at: valid_wall_timestamp(frame.rendered_at)
            .or_else(|| valid_wall_timestamp(Some(frame.timestamp))),
        render_ms: finite_non_negative(frame.render_ms),
        width: frame.width,
        height: frame.height,
        rgb: Arc::from(rgb),
    });
    let mut value = state
        .write()
        .map_err(|_| "Python bridge preview state lock poisoned".to_owned())?;
    value.snapshot.bridge_online = true;
    value.snapshot.preview_sequence = frame.sequence;
    value.snapshot.preview_received_sequence = frame.sequence;
    value.snapshot.preview_width = frame.width;
    value.snapshot.preview_height = frame.height;
    if frame.render_ms.is_finite() && frame.render_ms >= 0.0 {
        value.snapshot.preview_frame_ms = frame.render_ms;
    }
    value.details.preview_graph_revision = frame.graph_revision;
    value.details.preview_resolution_generation = frame.resolution_generation;
    value.preview = Some(preview);
    Ok(())
}

fn record_preview_problem(state: &Arc<RwLock<RuntimeView>>, message: String) {
    if let Ok(mut value) = state.write() {
        value.snapshot.preview_dropped_frames =
            value.snapshot.preview_dropped_frames.saturating_add(1);
        value.snapshot.preview_dropped = value.snapshot.preview_dropped.saturating_add(1);
        value.details.last_error = Some(RuntimeProblem {
            code: "bridge.preview_frame_invalid".to_owned(),
            path: Some("preview_frame".to_owned()),
            message,
            recoverable: true,
        });
    }
}

fn set_offline_problem(
    state: &Arc<RwLock<RuntimeView>>,
    code: &str,
    message: String,
    recoverable: bool,
) {
    if let Ok(mut value) = state.write() {
        value.snapshot.bridge_online = false;
        value.snapshot.transport_running = false;
        value.snapshot.lifecycle = EngineLifecycleState::Faulted;
        value.details.transport_running = false;
        value.details.pending_command_id = None;
        value.details.pending_command_type = None;
        value.details.backend = "offline".to_owned();
        value.details.lifecycle = EngineLifecycleState::Faulted;
        value.details.lifecycle_reason = format!("{code}: {message}");
        value.details.last_error = Some(RuntimeProblem {
            code: code.to_owned(),
            path: None,
            message,
            recoverable,
        });
    }
}

impl Drop for RuntimeStateSource {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
fn run_demo_source(state: Arc<RwLock<RuntimeView>>, stop: Arc<AtomicBool>) {
    let start = Instant::now();
    let mut graph_sequence = 0_u64;
    let mut preview_sequence = 0_u64;
    let mut next_graph = start;
    let mut next_preview = start;
    let graph_period = Duration::from_micros(8_333);
    let preview_period = Duration::from_micros(16_667);

    while !stop.load(Ordering::Acquire) {
        let now = Instant::now();
        if now >= next_graph {
            graph_sequence = graph_sequence.saturating_add(1);
            next_graph += graph_period;
        }
        if now >= next_preview {
            preview_sequence = preview_sequence.saturating_add(1);
            next_preview += preview_period;
        }

        let seconds = start.elapsed().as_secs_f32();
        if let Ok(mut value) = state.write() {
            value.snapshot.graph_fps = 120.0;
            value.snapshot.preview_fps = 60.0;
            value.snapshot.frame_ms = 8.33 + (seconds * 1.7).sin() * 0.22;
            value.snapshot.authoritative_fps = value.snapshot.graph_fps;
            value.snapshot.authoritative_frame_ms = value.snapshot.frame_ms;
            value.snapshot.preview_render_fps = value.snapshot.preview_fps;
            value.snapshot.preview_frame_ms = value.snapshot.frame_ms;
            value.snapshot.graph_sequence = graph_sequence;
            value.snapshot.preview_sequence = preview_sequence;
            value.snapshot.authoritative_sequence = graph_sequence;
            value.snapshot.preview_rendered_sequence = preview_sequence;
            value.snapshot.transport_running = true;
            value.snapshot.source_ready = true;
            value.details.transport_running = true;
            value.details.source_ready = true;
            value.details.source_kind = "demo".to_owned();
            value.details.source_label = "TEST DEMO".to_owned();
            value.details.backend = "test-demo".to_owned();
        }
        thread::sleep(Duration::from_millis(2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn producer_advances_without_ui_frames() {
        let source = RuntimeStateSource::demo();
        let first = source.read();
        thread::sleep(Duration::from_millis(35));
        let second = source.read();
        assert!(second.graph_sequence > first.graph_sequence);
        assert!(second.preview_sequence > first.preview_sequence);
        assert!(second.graph_sequence > second.preview_sequence);
    }

    #[test]
    fn state_commands_include_v2_id_and_authoritative_revision() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        state.write().unwrap().details.state_revision = 7;
        let (sender, receiver) = mpsc::channel();
        let source = RuntimeStateSource {
            state,
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            commands: Some(sender),
            next_command_id: AtomicU64::new(31),
        };
        assert_eq!(source.set_transport(true).unwrap(), 31);
        let command = receiver.recv().unwrap();
        let value: Value = serde_json::from_str(&command.line).unwrap();
        assert_eq!(value["version"], 2);
        assert_eq!(value["id"], 31);
        assert_eq!(value["type"], "set_transport");
        assert_eq!(value["expected_state_revision"], 7);
        assert_eq!(value["running"], true);
        assert_eq!(source.read_view().details.pending_command_id, Some(31));
        assert_eq!(
            source.read_view().details.pending_command_type.as_deref(),
            Some("set_transport")
        );
    }

    #[test]
    fn transactional_commands_are_serialized_behind_the_authoritative_revision() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        let (sender, _receiver) = mpsc::channel();
        let source = RuntimeStateSource {
            state,
            stop: Arc::new(AtomicBool::new(false)),
            worker: None,
            commands: Some(sender),
            next_command_id: AtomicU64::new(1),
        };
        assert_eq!(source.set_transport(true).unwrap(), 1);
        let error = source.seek(2.0).unwrap_err();
        assert!(error.contains("busy waiting for command 1"));
        assert_eq!(source.read_view().details.pending_command_id, Some(1));
    }

    #[test]
    fn accepted_graph_ack_updates_only_confirmed_graph_truth() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        let mut pending = HashMap::from([(
            4,
            PendingCommand {
                command_type: "replace_graph".to_owned(),
                intent: CommandIntent::ReplaceGraph,
            },
        )]);
        {
            let mut value = state.write().unwrap();
            value.details.pending_command_id = Some(4);
            value.details.current_graph_accepted = false;
        }
        apply_control_event(
            &state,
            BridgeEvent::Ack {
                version: 2,
                id: 4,
                command_type: "replace_graph".to_owned(),
                accepted: true,
                state_revision: 1,
                effective_frame_sequence: 8,
                graph_generation: 3,
                error: None,
            },
            &mut pending,
        )
        .unwrap();
        let value = state.read().unwrap();
        assert!(value.details.current_graph_accepted);
        assert_eq!(value.details.pending_command_id, None);
        assert_eq!(value.details.last_ack_command_id, 4);
        assert_eq!(value.snapshot.bridge_generation, 3);
        assert_eq!(value.snapshot.bridge_acks, 1);
    }

    #[test]
    fn rejected_graph_ack_keeps_rejection_and_problem_visible() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        let mut pending = HashMap::from([(
            9,
            PendingCommand {
                command_type: "replace_graph".to_owned(),
                intent: CommandIntent::ReplaceGraph,
            },
        )]);
        state.write().unwrap().details.pending_command_id = Some(9);
        apply_control_event(
            &state,
            BridgeEvent::Ack {
                version: 2,
                id: 9,
                command_type: "replace_graph".to_owned(),
                accepted: false,
                state_revision: 0,
                effective_frame_sequence: 0,
                graph_generation: 0,
                error: Some(RuntimeProblem {
                    code: "graph.unsupported_branch".to_owned(),
                    path: Some("graph.links".to_owned()),
                    message: "branching graph is not supported by this engine".to_owned(),
                    recoverable: true,
                }),
            },
            &mut pending,
        )
        .unwrap();
        let value = state.read().unwrap();
        assert!(!value.details.current_graph_accepted);
        assert_eq!(value.snapshot.bridge_rejections, 1);
        assert_eq!(
            value
                .details
                .last_error
                .as_ref()
                .map(|error| error.code.as_str()),
            Some("graph.unsupported_branch")
        );
    }

    #[test]
    fn python_bridge_accepts_and_processes_demo_graph_when_enabled() {
        if std::env::var("MEGAGLITCH_BRIDGE_INTEGRATION_TEST").as_deref() != Ok("1") {
            return;
        }
        let source = RuntimeStateSource::bridge_or_demo();
        let (graph, _) = crate::graph::demo_graph();
        source
            .replace_graph(&crate::preset::export_graph(&graph))
            .expect("bridge command should be queued");
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            let view = source.read_view();
            if view.snapshot.bridge_online
                && view.snapshot.bridge_acks == 1
                && view.snapshot.bridge_generation == 1
                && view.snapshot.graph_sequence > 0
                && view.details.current_graph_accepted
                && view.preview.as_ref().is_some_and(|frame| {
                    frame.sequence > 0
                        && frame.width > 0
                        && frame.height > 0
                        && frame.rgb.len() == frame.width * frame.height * 3
                })
            {
                return;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!(
            "Python bridge did not process the demo graph: {:?}",
            source.read_view()
        );
    }

    #[test]
    fn repository_root_resolution_finds_tools_directory() {
        let root = resolve_repository_root().expect("repository root must resolve");
        assert!(root.is_dir());
        assert!(root.join("tools").join("rust_ui_bridge.py").is_file());
    }

    #[test]
    fn configure_bridge_environment_sets_repo_root_and_pythonpath() {
        let repo_root = resolve_repository_root().expect("repository root must resolve");
        let mut cmd = Command::new("python");
        configure_bridge_environment(&mut cmd, &repo_root);
        // Verify current directory is repo root and MEGAGLITCH_REPOSITORY_ROOT is set
        assert_eq!(cmd.get_current_dir(), Some(repo_root.as_path()));
    }

    #[test]
    fn offline_failure_sets_faulted_lifecycle() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        set_offline_problem(
            &state,
            "bridge.process_failed",
            "Process terminated unexpectedly".to_owned(),
            false,
        );
        let view = state.read().unwrap();
        assert!(!view.snapshot.bridge_online);
        assert_eq!(view.snapshot.lifecycle, EngineLifecycleState::Faulted);
        assert_eq!(view.details.lifecycle, EngineLifecycleState::Faulted);
        assert_eq!(
            view.details.last_error.as_ref().map(|p| p.code.as_str()),
            Some("bridge.process_failed")
        );
    }

    #[test]
    fn handshake_and_ack_drive_finite_lifecycle_to_ready_and_running() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        let mut pending = HashMap::new();

        // 1. Ready event -> Ready state
        apply_control_event(
            &state,
            BridgeEvent::Ready {
                version: 2,
                backend: "python_flat_pipeline".to_owned(),
                state_revision: 0,
                capabilities: json!({"transactional_commands": ["replace_graph"]}),
                graph_generation: 0,
                transport_running: false,
                source_kind: "test_pattern".to_owned(),
                source_label: "TEST PATTERN".to_owned(),
                source_ready: true,
                source_position_seconds: None,
                source_duration_seconds: None,
            },
            &mut pending,
        )
        .unwrap();

        {
            let view = state.read().unwrap();
            assert!(view.snapshot.bridge_online);
            assert_eq!(view.snapshot.lifecycle, EngineLifecycleState::Ready);
            assert_eq!(view.details.lifecycle, EngineLifecycleState::Ready);
        }

        // 2. Ack replace_graph -> current_graph_accepted = true
        pending.insert(
            1,
            PendingCommand {
                command_type: "replace_graph".to_owned(),
                intent: CommandIntent::ReplaceGraph,
            },
        );
        apply_control_event(
            &state,
            BridgeEvent::Ack {
                version: 2,
                id: 1,
                command_type: "replace_graph".to_owned(),
                accepted: true,
                state_revision: 1,
                effective_frame_sequence: 1,
                graph_generation: 1,
                error: None,
            },
            &mut pending,
        )
        .unwrap();

        // 3. Ack set_transport(true) -> Running state
        pending.insert(
            2,
            PendingCommand {
                command_type: "set_transport".to_owned(),
                intent: CommandIntent::SetTransport { running: true },
            },
        );
        apply_control_event(
            &state,
            BridgeEvent::Ack {
                version: 2,
                id: 2,
                command_type: "set_transport".to_owned(),
                accepted: true,
                state_revision: 2,
                effective_frame_sequence: 1,
                graph_generation: 1,
                error: None,
            },
            &mut pending,
        )
        .unwrap();

        {
            let view = state.read().unwrap();
            assert_eq!(view.snapshot.lifecycle, EngineLifecycleState::Running);
            assert_eq!(view.details.lifecycle, EngineLifecycleState::Running);
        }
    }

    #[test]
    fn snapshot_event_establishes_authoritative_master_dimensions() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        let mut pending = HashMap::new();

        apply_control_event(
            &state,
            BridgeEvent::Snapshot {
                version: 2,
                state_revision: 3,
                graph_generation: 1,
                graph_sequence: 15,
                preview_sequence: 15,
                preview_rendered_sequence: Some(19),
                graph_fps: 60.0,
                preview_fps: 60.0,
                preview_render_fps: Some(58.0),
                frame_ms: 16.6,
                preview_frame_ms: 12.0,
                preview_p95_ms: 15.5,
                preview_width: Some(960),
                preview_height: Some(540),
                preview_requests: 20,
                preview_rendered: 18,
                preview_superseded: 2,
                preview_dropped: 1,
                preview_graph_revision: 7,
                preview_resolution_generation: 3,
                preview_node_statuses: vec![PreviewNodeStatus {
                    node_id: "node-1".to_owned(),
                    label: "P-Frame Glitches".to_owned(),
                    status: "implemented".to_owned(),
                    provider: "preview_scaled_existing".to_owned(),
                }],
                dropped_frames: 0,
                preview_dropped_frames: 0,
                transport_running: true,
                source_ready: true,
                source_kind: "image".to_owned(),
                source_label: Some("test_fixture.png".to_owned()),
                source_position_seconds: None,
                source_duration_seconds: None,
                width: Some(1920),
                height: Some(1080),
                last_error: None,
            },
            &mut pending,
        )
        .unwrap();

        let view = state.read().unwrap();
        assert_eq!(view.details.master_width, 1920);
        assert_eq!(view.details.master_height, 1080);
        assert_eq!(view.details.source_kind, "image");
        assert_eq!(view.details.source_label, "test_fixture.png");
        assert_eq!(view.snapshot.authoritative_fps, 60.0);
        assert_eq!(view.snapshot.preview_render_fps, 58.0);
        assert_eq!(view.snapshot.preview_frame_ms, 12.0);
        assert_eq!(view.snapshot.preview_p95_ms, 15.5);
        assert_eq!(view.snapshot.preview_rendered_sequence, 19);
        assert_eq!(
            (view.snapshot.preview_width, view.snapshot.preview_height),
            (960, 540)
        );
        assert_eq!(view.snapshot.preview_requests, 20);
        assert_eq!(view.snapshot.preview_rendered, 18);
        assert_eq!(view.snapshot.preview_superseded, 2);
        assert_eq!(view.snapshot.preview_dropped, 1);
        assert_eq!(view.details.preview_graph_revision, 7);
        assert_eq!(view.details.preview_resolution_generation, 3);
        assert_eq!(view.details.preview_node_statuses.len(), 1);
    }

    #[test]
    fn legacy_protocol_v2_snapshot_maps_to_explicit_render_metrics() {
        let event: BridgeEvent = serde_json::from_value(json!({
            "version": 2,
            "type": "snapshot",
            "state_revision": 1,
            "graph_generation": 0,
            "graph_sequence": 4,
            "preview_sequence": 4,
            "graph_fps": 31.0,
            "preview_fps": 30.0,
            "frame_ms": 22.5,
            "dropped_frames": 0,
            "preview_drops": 0,
            "transport_running": true,
            "source_ready": true,
            "source_kind": "test_pattern"
        }))
        .unwrap();
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        apply_control_event(&state, event, &mut HashMap::new()).unwrap();

        let view = state.read().unwrap();
        assert_eq!(view.snapshot.authoritative_fps, 31.0);
        assert_eq!(view.snapshot.authoritative_frame_ms, 22.5);
        assert_eq!(view.snapshot.preview_render_fps, 30.0);
        assert_eq!(view.snapshot.preview_frame_ms, 22.5);
        assert_eq!(view.snapshot.preview_rendered_sequence, 4);
    }

    #[test]
    fn stale_ack_rejected_on_state_revision_regression() {
        let state = Arc::new(RwLock::new(RuntimeView::default()));
        state.write().unwrap().details.state_revision = 5;
        let mut pending = HashMap::from([(
            1,
            PendingCommand {
                command_type: "replace_graph".to_owned(),
                intent: CommandIntent::ReplaceGraph,
            },
        )]);

        // An ACK claiming state_revision 2 when state is already at 5 must be rejected
        let result = apply_control_event(
            &state,
            BridgeEvent::Ack {
                version: 2,
                id: 1,
                command_type: "replace_graph".to_owned(),
                accepted: true,
                state_revision: 2, // regressed!
                effective_frame_sequence: 1,
                graph_generation: 1,
                error: None,
            },
            &mut pending,
        );

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("regressed"));
    }

    #[test]
    fn preview_copy_packed_and_padded_stride_parity_and_bounds() {
        use std::io::Write;

        let file_path = std::env::temp_dir().join(format!(
            "mg_test_preview_{}.rgb8",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let width = 4;
        let height = 3;
        let packed_stride = width * 3; // 12 bytes
        let padded_stride = 16; // 16 bytes (4 bytes padding per row)
        let slot_bytes = 256;
        let slots = 2;
        let total_bytes = slots * slot_bytes;

        // Write test pattern into temporary surface file
        // Slot 0: packed stride
        // Slot 1: padded stride with identical RGB content
        let mut buffer = vec![0u8; total_bytes];

        // Fill slot 0 (packed)
        for y in 0..height {
            for x in 0..width {
                let idx = y * packed_stride + x * 3;
                buffer[idx] = (x * 20 + y * 50) as u8;
                buffer[idx + 1] = (x * 30 + y * 60) as u8;
                buffer[idx + 2] = (x * 40 + y * 70) as u8;
            }
        }

        // Fill slot 1 (padded at slot_bytes offset)
        for y in 0..height {
            for x in 0..width {
                let idx = slot_bytes + y * padded_stride + x * 3;
                buffer[idx] = (x * 20 + y * 50) as u8;
                buffer[idx + 1] = (x * 30 + y * 60) as u8;
                buffer[idx + 2] = (x * 40 + y * 70) as u8;
            }
            // Add padding bytes
            for p in (width * 3)..padded_stride {
                buffer[slot_bytes + y * padded_stride + p] = 0xFF;
            }
        }

        {
            let mut file = File::create(&file_path).unwrap();
            file.write_all(&buffer).unwrap();
            file.flush().unwrap();
        }

        let surface = map_preview_surface(
            file_path.to_str().unwrap().to_owned(),
            100,
            slots,
            slot_bytes,
            64,
            64,
            "RGB8".to_owned(),
        )
        .unwrap();

        let state_packed = Arc::new(RwLock::new(RuntimeView::default()));
        copy_preview_frame(
            &state_packed,
            Some(&surface),
            PreviewFrameDescriptor {
                slot: 0,
                generation: 100,
                sequence: 1,
                width,
                height,
                stride: packed_stride,
                nbytes: height * packed_stride,
                timestamp: 0.0,
                graph_revision: 11,
                resolution_generation: 2,
                source_timestamp: Some(1.25),
                submitted_at: Some(1_700_000_000.0),
                rendered_at: Some(1_700_000_000.012),
                render_ms: 12.0,
                ..PreviewFrameDescriptor::default()
            },
        )
        .unwrap();

        let state_padded = Arc::new(RwLock::new(RuntimeView::default()));
        copy_preview_frame(
            &state_padded,
            Some(&surface),
            PreviewFrameDescriptor {
                slot: 1,
                generation: 100,
                sequence: 1,
                width,
                height,
                stride: padded_stride,
                nbytes: (height - 1) * padded_stride + width * 3,
                timestamp: 0.0,
                ..PreviewFrameDescriptor::default()
            },
        )
        .unwrap();

        let packed_view = state_packed.read().unwrap();
        let padded_view = state_padded.read().unwrap();

        assert!(packed_view.preview.is_some());
        assert!(padded_view.preview.is_some());

        let packed_frame = packed_view.preview.as_ref().unwrap();
        let padded_frame = padded_view.preview.as_ref().unwrap();

        // Exact pixel equality between packed fast-path and padded fallback
        assert_eq!(packed_frame.rgb.as_ref(), padded_frame.rgb.as_ref());
        assert_eq!(packed_frame.rgb.len(), width * height * 3);
        assert_eq!(packed_frame.surface_generation, 100);
        assert_eq!(packed_frame.graph_revision, 11);
        assert_eq!(packed_frame.resolution_generation, 2);
        assert_eq!(packed_frame.source_timestamp, Some(1.25));
        assert_eq!(packed_frame.submitted_at, Some(1_700_000_000.0));
        assert_eq!(packed_frame.rendered_at, Some(1_700_000_000.012));
        assert_eq!(packed_frame.render_ms, 12.0);
        assert_eq!(packed_view.snapshot.preview_received_sequence, 1);

        // Bounds checks
        let state_err = Arc::new(RwLock::new(RuntimeView::default()));
        // 1. Invalid slot
        let err_slot = copy_preview_frame(
            &state_err,
            Some(&surface),
            PreviewFrameDescriptor {
                slot: 99,
                generation: 100,
                sequence: 1,
                width,
                height,
                stride: packed_stride,
                nbytes: height * packed_stride,
                timestamp: 0.0,
                ..PreviewFrameDescriptor::default()
            },
        );
        assert!(err_slot.is_err());

        // 2. Stale generation
        let err_gen = copy_preview_frame(
            &state_err,
            Some(&surface),
            PreviewFrameDescriptor {
                slot: 0,
                generation: 999,
                sequence: 1,
                width,
                height,
                stride: packed_stride,
                nbytes: height * packed_stride,
                timestamp: 0.0,
                ..PreviewFrameDescriptor::default()
            },
        );
        assert!(err_gen.is_err());

        // 3. Stride smaller than row
        let err_stride = copy_preview_frame(
            &state_err,
            Some(&surface),
            PreviewFrameDescriptor {
                slot: 0,
                generation: 100,
                sequence: 1,
                width,
                height,
                stride: 5,
                nbytes: height * packed_stride,
                timestamp: 0.0,
                ..PreviewFrameDescriptor::default()
            },
        );
        assert!(err_stride.is_err());

        drop(surface);
        let _ = std::fs::remove_file(file_path);
    }
}
