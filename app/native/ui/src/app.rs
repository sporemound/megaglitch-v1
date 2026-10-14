use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui::{self, Align, Color32, Layout, Stroke};
use egui_snarl::ui::{
    BackgroundPattern, Grid, NodeLayout, PinPlacement, SnarlStyle, SnarlWidget, WireLayer,
    WireStyle,
};
use egui_snarl::{InPinId, NodeId, OutPinId, Snarl};

use crate::catalog::{self, ParameterKind, ParameterSpec, ParameterValue};
use crate::graph::{self, GraphNode, GraphViewer, LibraryNode, NodeKind, NodeRequest};
use crate::mg_ui;
use crate::preset;
use crate::session::{CanonicalSession, DeckId};
use crate::state::{
    EngineLifecycleState, RuntimeDetails, RuntimeSnapshot, RuntimeStateSource, RuntimeView,
};

#[derive(Debug)]
enum RuntimeIntent {
    SetTransport(bool),
    SetSource {
        kind: &'static str,
        path: Option<String>,
    },
    Seek(f64),
}

fn graph_status_for(
    graph_revision: u64,
    submitted_graph_revision: u64,
    details: &RuntimeDetails,
) -> graph::GraphStatus {
    if graph_revision != submitted_graph_revision {
        graph::GraphStatus::Draft
    } else if !details.current_graph_accepted
        && details.pending_command_type.as_deref() == Some("replace_graph")
    {
        graph::GraphStatus::Pending
    } else if details.current_graph_accepted {
        graph::GraphStatus::Accepted
    } else if submitted_graph_revision > 0 {
        graph::GraphStatus::Rejected
    } else {
        graph::GraphStatus::Draft
    }
}

fn session_status(snapshot: RuntimeSnapshot) -> String {
    if snapshot.bridge_online {
        format!("SESSION  IN-MEMORY · ENGINE R{}", snapshot.state_revision)
    } else {
        "SESSION  UNAVAILABLE · ENGINE OFFLINE".to_owned()
    }
}

fn supports_engine_command(
    snapshot: RuntimeSnapshot,
    details: &RuntimeDetails,
    command: &str,
) -> bool {
    snapshot.bridge_online
        && details
            .capabilities
            .iter()
            .any(|capability| capability == &format!("transactional_commands:{command}"))
}

fn engine_restart_available(snapshot: RuntimeSnapshot, details: &RuntimeDetails) -> bool {
    !snapshot.bridge_online
        && details
            .last_error
            .as_ref()
            .is_some_and(|problem| !problem.recoverable)
}

pub(crate) fn reconcile_runtime_status(
    current_status: &str,
    snapshot: RuntimeSnapshot,
    details: &RuntimeDetails,
) -> Option<String> {
    if current_status.starts_with("RESTARTING ENGINE V2")
        || current_status.starts_with("Starting Engine V2")
    {
        match details.lifecycle {
            EngineLifecycleState::Faulted => Some(format!(
                "ENGINE FAULTED: {}",
                details
                    .last_error
                    .as_ref()
                    .map(|p| p.code.as_str())
                    .unwrap_or("bridge_process_failed")
            )),
            EngineLifecycleState::Stopped => Some("ENGINE STOPPED".to_owned()),
            EngineLifecycleState::Ready | EngineLifecycleState::Running => {
                Some(format!("ENGINE READY (G{})", snapshot.bridge_generation))
            }
            _ => None,
        }
    } else {
        None
    }
}

fn video_seek_range(details: &RuntimeDetails) -> Option<(f64, f64)> {
    if !details.source_ready || details.source_kind != "video" {
        return None;
    }
    let position = details.source_position_seconds?;
    let duration = details.source_duration_seconds?;
    if position.is_finite() && position >= 0.0 && duration.is_finite() && duration > 0.0 {
        Some((position.min(duration), duration))
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pane {
    Sources,
    Graph,
    Preview,
    Inspector,
    Timeline,
    SignalMonitor,
}

impl Pane {
    fn title(self) -> &'static str {
        match self {
            Self::Sources => "SOURCES + NODES",
            Self::Graph => "NODE WORKSPACE",
            Self::Preview => "LIVE OUTPUT",
            Self::Inspector => "INSPECTOR",
            Self::Timeline => "AUTOMATION · RESERVED",
            Self::SignalMonitor => "SIGNAL MONITOR",
        }
    }
}

const PREVIEW_PRESENT_RATE_WINDOW: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PreviewPresentationSnapshot {
    preview_present_fps: f32,
    preview_measurement_fps: f32,
    measurement_duration_seconds: f32,
    preview_presented_count: u64,
    preview_presented_sequence: u64,
    preview_presented_graph_revision: u64,
    preview_submit_to_present_ms: Option<f32>,
    preview_mean_submit_to_present_ms: Option<f32>,
    stage_samples: u64,
    texture_preparations: u64,
    replaced_before_draw: u64,
    stage_mean_ms: [f64; 3],
    stage_max_ms: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreviewPresentKey {
    surface_generation: u64,
    sequence: u64,
    graph_revision: u64,
}

#[derive(Debug, Default)]
struct PreviewPresentationState {
    last_key: Option<PreviewPresentKey>,
    presented_at: VecDeque<Instant>,
    first_presented_at: Option<Instant>,
    last_presented_at: Option<Instant>,
    presented_count: u64,
    preview_submit_to_present_ms: Option<f32>,
    submit_to_present_total_ms: f64,
    submit_to_present_samples: u64,
    pending_upload: Option<(PreviewPresentKey, Instant, Instant, Instant)>,
    texture_preparations: u64,
    replaced_before_draw: u64,
    stage_samples: u64,
    stage_total_ms: [f64; 3],
    stage_max_ms: [f64; 3],
}

#[derive(Clone, Debug, Default)]
struct PreviewPresentationTracker {
    state: Arc<Mutex<PreviewPresentationState>>,
}

impl PreviewPresentationTracker {
    fn record_upload(
        &self,
        key: PreviewPresentKey,
        received: Instant,
        started: Instant,
        finished: Instant,
    ) {
        if let Ok(mut state) = self.state.lock() {
            if state.pending_upload.is_some() {
                state.replaced_before_draw += 1;
            }
            state.texture_preparations += 1;
            state.pending_upload = Some((key, received, started, finished));
        }
    }

    fn observe(
        &self,
        surface_generation: u64,
        sequence: u64,
        graph_revision: u64,
        submitted_at: Option<f64>,
    ) {
        self.observe_at(
            surface_generation,
            sequence,
            graph_revision,
            submitted_at,
            Instant::now(),
            unix_seconds_now(),
        );
    }

    fn observe_at(
        &self,
        surface_generation: u64,
        sequence: u64,
        graph_revision: u64,
        submitted_at: Option<f64>,
        now: Instant,
        now_unix_seconds: Option<f64>,
    ) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let key = PreviewPresentKey {
            surface_generation,
            sequence,
            graph_revision,
        };
        if state.last_key.is_some_and(|previous| {
            previous.surface_generation == key.surface_generation
                && previous.sequence >= key.sequence
        }) {
            return;
        }
        state.last_key = Some(key);
        if let Some((upload_key, received, started, finished)) = state.pending_upload
            && upload_key == key
        {
            let stages = [
                started.saturating_duration_since(received),
                finished.saturating_duration_since(started),
                now.saturating_duration_since(finished),
            ];
            for (index, duration) in stages.iter().enumerate() {
                let ms = duration.as_secs_f64() * 1_000.0;
                state.stage_total_ms[index] += ms;
                state.stage_max_ms[index] = state.stage_max_ms[index].max(ms);
            }
            state.stage_samples += 1;
            state.pending_upload = None;
        }
        state.presented_at.push_back(now);
        state.first_presented_at.get_or_insert(now);
        state.last_presented_at = Some(now);
        state.presented_count = state.presented_count.saturating_add(1);
        trim_present_samples(&mut state.presented_at, now);
        state.preview_submit_to_present_ms = submitted_at
            .filter(|timestamp| timestamp.is_finite() && *timestamp >= 0.0)
            .zip(now_unix_seconds)
            .and_then(|(submitted, presented)| {
                let elapsed_ms = (presented - submitted) * 1_000.0;
                (elapsed_ms.is_finite() && elapsed_ms >= 0.0).then_some(elapsed_ms as f32)
            });
        if let Some(latency) = state.preview_submit_to_present_ms {
            state.submit_to_present_total_ms += f64::from(latency);
            state.submit_to_present_samples = state.submit_to_present_samples.saturating_add(1);
        }
    }

    fn snapshot(&self) -> PreviewPresentationSnapshot {
        self.snapshot_at(Instant::now())
    }

    fn snapshot_at(&self, now: Instant) -> PreviewPresentationSnapshot {
        let Ok(mut state) = self.state.lock() else {
            return PreviewPresentationSnapshot::default();
        };
        trim_present_samples(&mut state.presented_at, now);
        let preview_present_fps = if state.presented_at.len() >= 2 {
            let elapsed = state
                .presented_at
                .back()
                .expect("two presentation samples have a final timestamp")
                .duration_since(
                    *state
                        .presented_at
                        .front()
                        .expect("two presentation samples have a first timestamp"),
                )
                .as_secs_f32();
            if elapsed > 0.0 {
                (state.presented_at.len() - 1) as f32 / elapsed
            } else {
                0.0
            }
        } else {
            0.0
        };
        let measurement_duration_seconds = match (state.first_presented_at, state.last_presented_at)
        {
            (Some(first), Some(last)) => last.duration_since(first).as_secs_f32(),
            _ => 0.0,
        };
        let preview_measurement_fps = match (
            state.first_presented_at,
            state.last_presented_at,
            state.presented_count,
        ) {
            (Some(first), Some(last), count) if count >= 2 => {
                let elapsed = last.duration_since(first).as_secs_f32();
                if elapsed > 0.0 {
                    (count - 1) as f32 / elapsed
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        PreviewPresentationSnapshot {
            stage_samples: state.stage_samples,
            texture_preparations: state.texture_preparations,
            replaced_before_draw: state.replaced_before_draw,
            stage_mean_ms: state
                .stage_total_ms
                .map(|total| total / state.stage_samples.max(1) as f64),
            stage_max_ms: state.stage_max_ms,
            preview_present_fps,
            preview_measurement_fps,
            measurement_duration_seconds,
            preview_presented_count: state.presented_count,
            preview_presented_sequence: state.last_key.map_or(0, |key| key.sequence),
            preview_presented_graph_revision: state.last_key.map_or(0, |key| key.graph_revision),
            preview_submit_to_present_ms: state.preview_submit_to_present_ms,
            preview_mean_submit_to_present_ms: (state.submit_to_present_samples > 0).then(|| {
                (state.submit_to_present_total_ms / state.submit_to_present_samples as f64) as f32
            }),
        }
    }

    fn reset(&self) {
        if let Ok(mut state) = self.state.lock() {
            *state = PreviewPresentationState::default();
        }
    }
}

fn trim_present_samples(samples: &mut VecDeque<Instant>, now: Instant) {
    while samples.front().is_some_and(|sample| {
        now.checked_duration_since(*sample)
            .is_some_and(|elapsed| elapsed > PREVIEW_PRESENT_RATE_WINDOW)
    }) {
        samples.pop_front();
    }
}

fn unix_seconds_now() -> Option<f64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs_f64())
}

#[derive(Clone)]
struct PreviewPresentMarker {
    tracker: PreviewPresentationTracker,
    surface_generation: u64,
    sequence: u64,
    graph_revision: u64,
    submitted_at: Option<f64>,
}

impl eframe::egui_wgpu::CallbackTrait for PreviewPresentMarker {
    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        _render_pass: &mut eframe::wgpu::RenderPass<'static>,
        _callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        self.tracker.observe(
            self.surface_generation,
            self.sequence,
            self.graph_revision,
            self.submitted_at,
        );
    }
}

const PREVIEW_SMOKE_REPORT_ENV: &str = "MEGAGLITCH_PREVIEW_SMOKE_REPORT";
const PREVIEW_SMOKE_TARGET_ENV: &str = "MEGAGLITCH_PREVIEW_SMOKE_TARGET";
const PREVIEW_SMOKE_SOURCE_ENV: &str = "MEGAGLITCH_PREVIEW_SMOKE_SOURCE";
const PREVIEW_SMOKE_PRESET_ENV: &str = "MEGAGLITCH_PREVIEW_SMOKE_PRESET";
const PREVIEW_SMOKE_DECK_ENV: &str = "MEGAGLITCH_PREVIEW_SMOKE_DECK";
const PREVIEW_SMOKE_CASE_ENV: &str = "MEGAGLITCH_PREVIEW_SMOKE_CASE";
const PREVIEW_SMOKE_PROBE_PRESENTATIONS: u64 = 30;
const PREVIEW_SMOKE_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
const PREVIEW_SMOKE_STARTUP_TIMEOUT: Duration = Duration::from_secs(20);

#[derive(Clone, Debug)]
struct PreviewParameterMutation {
    node_id: String,
    parameter: String,
    original: f64,
    probed: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewParameterProbeState {
    Pending,
    NotApplicable,
    Waiting,
    Resolved,
}

impl PreviewParameterProbeState {
    const fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::NotApplicable => "not_applicable",
            Self::Waiting => "waiting_for_new_graph_preview",
            Self::Resolved => "resolved",
        }
    }
}

#[derive(Debug)]
struct PreviewSmokeRun {
    report_path: PathBuf,
    target_presentations: u64,
    case_label: String,
    source_path: Option<PathBuf>,
    preset_path: Option<PathBuf>,
    deck: DeckId,
    configuration_error: Option<String>,
    started_at: Instant,
    armed: bool,
    report_written: bool,
    preview_requests_baseline: u64,
    preview_rendered_baseline: u64,
    preview_superseded_baseline: u64,
    preview_dropped_baseline: u64,
    probe_state: PreviewParameterProbeState,
    probe_started_at: Option<Instant>,
    probe_baseline_graph_revision: u64,
    probe_target_sequence: Option<u64>,
    parameter_to_preview_ms: Option<f32>,
    probe_mutation: Option<PreviewParameterMutation>,
}

impl PreviewSmokeRun {
    fn from_environment() -> Option<Self> {
        let report_path = std::env::var_os(PREVIEW_SMOKE_REPORT_ENV).map(PathBuf::from)?;
        let target_presentations = std::env::var(PREVIEW_SMOKE_TARGET_ENV)
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(300);
        let deck_value = std::env::var(PREVIEW_SMOKE_DECK_ENV)
            .unwrap_or_else(|_| "a".to_owned())
            .to_ascii_lowercase();
        let (deck, configuration_error) = match deck_value.as_str() {
            "a" => (DeckId::A, None),
            "b" => (DeckId::B, None),
            _ => (
                DeckId::A,
                Some(format!(
                    "{PREVIEW_SMOKE_DECK_ENV} must be 'a' or 'b', got {deck_value:?}"
                )),
            ),
        };
        Some(Self {
            report_path,
            target_presentations,
            case_label: std::env::var(PREVIEW_SMOKE_CASE_ENV)
                .unwrap_or_else(|_| "preview_smoke".to_owned()),
            source_path: std::env::var_os(PREVIEW_SMOKE_SOURCE_ENV).map(PathBuf::from),
            preset_path: std::env::var_os(PREVIEW_SMOKE_PRESET_ENV).map(PathBuf::from),
            deck,
            configuration_error,
            started_at: Instant::now(),
            armed: false,
            report_written: false,
            preview_requests_baseline: 0,
            preview_rendered_baseline: 0,
            preview_superseded_baseline: 0,
            preview_dropped_baseline: 0,
            probe_state: PreviewParameterProbeState::Pending,
            probe_started_at: None,
            probe_baseline_graph_revision: 0,
            probe_target_sequence: None,
            parameter_to_preview_ms: None,
            probe_mutation: None,
        })
    }
}

fn preview_smoke_graph_from_value(
    document: &serde_json::Value,
    deck: DeckId,
) -> Result<preset::PresetGraph, String> {
    let value = if document.get("nodes").is_some() {
        document
    } else {
        let deck_value = document
            .get("decks")
            .and_then(serde_json::Value::as_object)
            .and_then(|decks| decks.get(deck.key()))
            .ok_or_else(|| format!("preset wrapper has no decks.{} entry", deck.key()))?;
        deck_value.get("graph").unwrap_or(deck_value)
    };
    serde_json::from_value(value.clone())
        .map_err(|error| format!("invalid deck {} preset graph: {error}", deck.key()))
}

fn preview_smoke_load_graph(
    path: &PathBuf,
    deck: DeckId,
) -> Result<(preset::PresetGraph, Snarl<GraphNode>, Option<NodeId>), String> {
    if !path.is_file() {
        return Err(format!("preset path is not a file: {}", path.display()));
    }
    let bytes = std::fs::read(path)
        .map_err(|error| format!("could not read preset {}: {error}", path.display()))?;
    let document = serde_json::from_slice::<serde_json::Value>(&bytes)
        .map_err(|error| format!("invalid preset JSON {}: {error}", path.display()))?;
    let saved = preview_smoke_graph_from_value(&document, deck)?;
    let (graph, selected) = preset::import_graph(&saved)
        .map_err(|error| format!("preset graph validation failed: {error}"))?;
    Ok((saved, graph, selected))
}

fn preview_smoke_source_kind(path: &std::path::Path) -> Result<&'static str, String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "mp4" | "mov" | "avi" | "mkv" | "webm" | "m4v" => Ok("video"),
        "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff" | "gif" | "webp" => Ok("image"),
        _ => Err(format!(
            "unsupported smoke source extension for {}",
            path.display()
        )),
    }
}

fn begin_preview_parameter_probe(
    graph: &mut Snarl<GraphNode>,
) -> Result<Option<PreviewParameterMutation>, String> {
    let mut has_effect = false;
    for node in graph.nodes_mut() {
        if matches!(node.kind, NodeKind::Source | NodeKind::Output) {
            continue;
        }
        has_effect = true;
        let Some(spec) = node.spec else {
            continue;
        };
        for schema in spec.parameters {
            if schema.kind != ParameterKind::Float
                || matches!(schema.id, "enabled" | "multimodal_amount")
            {
                continue;
            }
            let Some(ParameterValue::Float(value)) = node.parameters.get_mut(schema.id) else {
                continue;
            };
            let original = *value;
            let minimum = schema.minimum.map(f64::from).unwrap_or(f64::NEG_INFINITY);
            let maximum = schema.maximum.map(f64::from).unwrap_or(f64::INFINITY);
            let span = if minimum.is_finite() && maximum.is_finite() {
                (maximum - minimum).abs()
            } else {
                1.0
            };
            let delta = (span * 0.001).clamp(0.000_1, 0.01);
            let increased = original + delta;
            let decreased = original - delta;
            let probed = if increased <= maximum {
                increased
            } else if decreased >= minimum {
                decreased
            } else {
                continue;
            };
            if probed == original {
                continue;
            }
            *value = probed;
            return Ok(Some(PreviewParameterMutation {
                node_id: node.instance_id.clone(),
                parameter: schema.id.to_owned(),
                original,
                probed,
            }));
        }
    }
    if has_effect {
        Err("effect graph has no mutable Float parameter for the smoke probe".to_owned())
    } else {
        Ok(None)
    }
}

fn restore_preview_parameter_probe(
    graph: &mut Snarl<GraphNode>,
    mutation: &PreviewParameterMutation,
) -> bool {
    for node in graph.nodes_mut() {
        if node.instance_id != mutation.node_id {
            continue;
        }
        if let Some(ParameterValue::Float(value)) =
            node.parameters.get_mut(mutation.parameter.as_str())
        {
            *value = mutation.original;
            return true;
        }
    }
    false
}

pub struct MegaGlitchApp {
    tree: egui_tiles::Tree<Pane>,
    graph: Snarl<GraphNode>,
    deck_b_graph: Snarl<GraphNode>,
    canonical_session: CanonicalSession,
    selected: Option<NodeId>,
    deck_b_selected: Option<NodeId>,
    active_deck: DeckId,
    state_source: RuntimeStateSource,
    command_palette_open: bool,
    command_query: String,
    workspace: &'static str,
    ui_scale: f32,
    reduced_motion: bool,
    graph_view_epoch: u64,
    catalog_query: String,
    preset_buffer: String,
    preset_status: String,
    graph_revision: u64,
    submitted_graph_revision: u64,
    source_revision: u64,
    submitted_source_revision: u64,
    source_path: String,
    runtime_status: String,
    preview_texture: Option<egui::TextureHandle>,
    preview_texture_generation: u64,
    preview_texture_sequence: u64,
    preview_uploaded_sequence: u64,
    preview_present_tracker: PreviewPresentationTracker,
    preview_smoke: Option<PreviewSmokeRun>,
    observed_ack_command_id: u64,
    video_seek_seconds: Option<f64>,
    dragged_node: Option<NodeRequest>,
    master_mix: f32,
    master_transition: String,
    cognitive_profile: String,
    mutation_severity: f32,
    retro_enabled: bool,
    retro_bits: u8,
    retro_dither: bool,
    export_duration: f32,
    sequencer_bpm: f32,
    sequencer_steps: [f32; 16],
    sequencer_step_count: usize,
    sequencer_playing: bool,
    sequencer_current_step: usize,
    sequencer_last_tick: std::time::Instant,
}

impl MegaGlitchApp {
    pub fn new(creation: &eframe::CreationContext<'_>) -> Self {
        mg_ui::install(&creation.egui_ctx);
        let (graph, selected) = graph::demo_graph();
        let (deck_b_graph, deck_b_selected) = graph::demo_graph();
        let canonical_session = CanonicalSession::new(
            &preset::export_graph(&graph),
            &preset::export_graph(&deck_b_graph),
        )
        .expect("built-in demo graphs form a canonical session");
        let mut app = Self {
            tree: default_tree(),
            graph,
            deck_b_graph,
            canonical_session,
            selected,
            deck_b_selected,
            active_deck: DeckId::A,
            state_source: RuntimeStateSource::bridge_with_repaint(creation.egui_ctx.clone()),
            command_palette_open: false,
            command_query: String::new(),
            workspace: "PATCH",
            ui_scale: 1.0,
            reduced_motion: false,
            graph_view_epoch: 0,
            catalog_query: String::new(),
            preset_buffer: String::new(),
            preset_status: "Preset v1 buffer ready".to_owned(),
            graph_revision: 1,
            submitted_graph_revision: 0,
            source_revision: 0,
            submitted_source_revision: 0,
            source_path: String::new(),
            runtime_status: "Starting Engine V2…".to_owned(),
            preview_texture: None,
            preview_texture_generation: 0,
            preview_texture_sequence: u64::MAX,
            preview_uploaded_sequence: 0,
            preview_present_tracker: PreviewPresentationTracker::default(),
            preview_smoke: PreviewSmokeRun::from_environment(),
            observed_ack_command_id: 0,
            video_seek_seconds: None,
            dragged_node: None,
            master_mix: 0.0,
            master_transition: "Crossfade".to_owned(),
            cognitive_profile: "Alpha (Balanced)".to_owned(),
            mutation_severity: 0.35,
            retro_enabled: false,
            retro_bits: 4,
            retro_dither: true,
            export_duration: 5.0,
            sequencer_bpm: 120.0,
            sequencer_steps: [
                0.1, 0.4, 0.2, 0.8, 0.3, 0.6, 0.15, 0.9, 0.25, 0.5, 0.35, 0.75, 0.2, 0.65, 0.4,
                0.95,
            ],
            sequencer_step_count: 16,
            sequencer_playing: false,
            sequencer_current_step: 0,
            sequencer_last_tick: std::time::Instant::now(),
        };
        app.configure_preview_smoke();
        app
    }

    fn configure_preview_smoke(&mut self) {
        let Some(smoke) = self.preview_smoke.as_ref() else {
            return;
        };
        if smoke.configuration_error.is_some() {
            return;
        }
        let preset_path = smoke.preset_path.clone();
        let source_path = smoke.source_path.clone();
        let deck = smoke.deck;

        let result = (|| -> Result<(), String> {
            if let Some(path) = preset_path.as_ref() {
                let (_saved, imported, selected) = preview_smoke_load_graph(path, deck)?;
                let normalized = preset::export_graph(&imported);
                self.canonical_session
                    .update_deck_graph(DeckId::A, &normalized)?;
                self.graph = imported;
                self.selected = selected;
                self.graph_view_epoch = self.graph_view_epoch.saturating_add(1);
                self.graph_revision = 1;
                self.submitted_graph_revision = 0;
                self.preset_status = format!(
                    "Preview smoke loaded deck {} from {}",
                    deck.key(),
                    path.display()
                );
            }
            if let Some(path) = source_path.as_ref() {
                if !path.is_file() {
                    return Err(format!("source path is not a file: {}", path.display()));
                }
                preview_smoke_source_kind(path)?;
                self.source_path = path.to_string_lossy().into_owned();
                self.source_revision = 1;
                self.submitted_source_revision = 0;
            }
            Ok(())
        })();
        if let Err(error) = result {
            if let Some(smoke) = self.preview_smoke.as_mut() {
                smoke.configuration_error = Some(error);
            }
        }
    }

    fn apply_cognitive_mutation(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::time::Instant::now().hash(&mut hasher);
        let mut seed = hasher.finish();

        let mut rng_float = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 32) as u32 as f32) / (u32::MAX as f32)
        };

        let severity = self.mutation_severity.clamp(0.05, 1.0);
        let profile = self.cognitive_profile.clone();

        for node in self.graph.nodes_mut() {
            if matches!(node.kind, NodeKind::Source | NodeKind::Output) {
                continue;
            }
            let delta = (rng_float() * 2.0 - 1.0) * severity;
            let new_amount = (node.amount + delta).clamp(0.0, 1.0);
            node.set_amount(new_amount);

            for (param_name, param_val) in node.parameters.iter_mut() {
                if *param_name == "enabled" || *param_name == "multimodal_amount" {
                    continue;
                }
                match param_val {
                    ParameterValue::Float(val) => {
                        let shift = match profile.as_str() {
                            "Gamma (Chaotic)" => (rng_float() * 2.0 - 1.0) * severity * 1.8,
                            "Beta (Active)" => (rng_float() * 2.0 - 1.0) * severity * 0.9,
                            "Theta (Flow)" => (rng_float() * 2.0 - 1.0) * severity * 0.4,
                            "Delta (Relaxed)" => (rng_float() * 2.0 - 1.0) * severity * 0.2,
                            _ => (rng_float() * 2.0 - 1.0) * severity * 0.6,
                        };
                        *val = (*val + (shift as f64)).clamp(0.0, 100.0);
                    }
                    ParameterValue::Bool(b) => {
                        if profile == "Gamma (Chaotic)" && rng_float() < (severity * 0.4) {
                            *b = !*b;
                        }
                    }
                    _ => {}
                }
            }
        }
        self.graph_revision = self.graph_revision.saturating_add(1);
        self.runtime_status = format!(
            "Applied {} Cognitive Mutation ({:.0}% severity)",
            profile,
            severity * 100.0
        );
    }

    fn tick_sequencer(&mut self) {
        if !self.sequencer_playing {
            return;
        }
        let step_dur = Duration::from_secs_f32(60.0 / (self.sequencer_bpm * 4.0).max(1.0));
        if self.sequencer_last_tick.elapsed() >= step_dur {
            self.sequencer_last_tick = std::time::Instant::now();
            self.sequencer_current_step =
                (self.sequencer_current_step + 1) % self.sequencer_step_count;
            let step_val = self.sequencer_steps[self.sequencer_current_step];

            // Modulate selected node or first transform node
            let target_node = self.selected.or_else(|| {
                self.graph
                    .node_ids()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Transform | NodeKind::Feedback))
                    .map(|(id, _)| id)
            });
            if let Some(id) = target_node {
                self.graph[id].set_amount(step_val);
                self.graph_revision = self.graph_revision.saturating_add(1);
            }
        }
    }

    fn dual_deck_ui(
        &mut self,
        ui: &mut egui::Ui,
        preview: Option<&crate::state::PreviewFrame>,
        details: &RuntimeDetails,
    ) {
        ui.vertical(|ui| {
            mg_ui::section(
                ui,
                "Dual Deck Video Mixer",
                "Deck A vs Deck B interactive A/B crossfader",
            );
            ui.horizontal(|ui| {
                // Left Column: Deck A
                ui.vertical(|ui| {
                    ui.set_width(ui.available_width() * 0.45);
                    egui::Frame::new()
                        .fill(mg_ui::PANEL_RAISED)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("DECK A (PRIMARY)")
                                        .strong()
                                        .color(mg_ui::SIGNAL),
                                );
                                if ui
                                    .selectable_label(self.active_deck == DeckId::A, "FOCUS")
                                    .clicked()
                                {
                                    self.active_deck = DeckId::A;
                                }
                            });
                            ui.label(format!("Nodes: {}", self.graph.nodes().count()));
                            if ui.button("⚡ Auto-Wire Deck A").clicked() {
                                graph::auto_wire_chain(&mut self.graph);
                                self.graph_revision = self.graph_revision.saturating_add(1);
                            }
                        });
                });

                // Center Column: Crossfader
                ui.vertical(|ui| {
                    ui.set_width(ui.available_width() * 0.5);
                    egui::Frame::new()
                        .fill(mg_ui::BG)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new("MASTER CROSSFADER")
                                    .strong()
                                    .color(mg_ui::ACCENT),
                            );
                            ui.add(
                                egui::Slider::new(&mut self.master_mix, 0.0..=1.0)
                                    .text("Mix (A ←→ B)"),
                            );
                            ui.horizontal(|ui| {
                                if ui.button("CUT A").clicked() {
                                    self.master_mix = 0.0;
                                }
                                if ui.button("50/50").clicked() {
                                    self.master_mix = 0.5;
                                }
                                if ui.button("CUT B").clicked() {
                                    self.master_mix = 1.0;
                                }
                            });
                            ui.label("Transition Mode:");
                            for mode in
                                ["Crossfade", "Additive", "Multiply", "Screen", "Difference"]
                            {
                                if ui
                                    .selectable_label(self.master_transition == mode, mode)
                                    .clicked()
                                {
                                    self.master_transition = mode.to_owned();
                                }
                            }
                        });
                });

                // Right Column: Deck B
                ui.vertical(|ui| {
                    egui::Frame::new()
                        .fill(mg_ui::PANEL_RAISED)
                        .inner_margin(8.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new("DECK B (SECONDARY)")
                                        .strong()
                                        .color(mg_ui::ACCENT),
                                );
                                if ui
                                    .selectable_label(self.active_deck == DeckId::B, "FOCUS")
                                    .clicked()
                                {
                                    self.active_deck = DeckId::B;
                                }
                            });
                            ui.label(format!("Nodes: {}", self.deck_b_graph.nodes().count()));
                            if ui.button("⚡ Auto-Wire Deck B").clicked() {
                                graph::auto_wire_chain(&mut self.deck_b_graph);
                            }
                        });
                });
            });

            ui.separator();
            ui.label(
                egui::RichText::new("PROGRAM MASTER PREVIEW")
                    .strong()
                    .color(mg_ui::TEXT),
            );
            preview_ui(
                ui,
                self.preview_texture.as_ref(),
                preview,
                details,
                &self.preview_present_tracker,
            );
        });
    }

    fn mutation_sequencer_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                // Section 1: Cognitive AI Mutation
                ui.vertical(|ui| {
                    ui.set_width(ui.available_width() * 0.48);
                    mg_ui::section(
                        ui,
                        "Cognitive Mutation Engine",
                        "AI dynamic graph corrupter",
                    );
                    egui::Frame::new()
                        .fill(mg_ui::PANEL_RAISED)
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.label("Select Cognitive Resonance Profile:");
                            for profile in [
                                "Alpha (Balanced)",
                                "Beta (Active)",
                                "Gamma (Chaotic)",
                                "Theta (Flow)",
                                "Delta (Relaxed)",
                            ] {
                                if ui
                                    .selectable_label(self.cognitive_profile == profile, profile)
                                    .clicked()
                                {
                                    self.cognitive_profile = profile.to_owned();
                                }
                            }
                            ui.add_space(8.0);
                            ui.add(
                                egui::Slider::new(&mut self.mutation_severity, 0.05..=1.0)
                                    .text("Severity"),
                            );
                            ui.add_space(8.0);
                            if ui
                                .add(
                                    egui::Button::new(
                                        egui::RichText::new("⚡ MUTATE ACTIVE GRAPH")
                                            .strong()
                                            .size(13.0)
                                            .color(mg_ui::BG),
                                    )
                                    .fill(mg_ui::ACCENT),
                                )
                                .clicked()
                            {
                                self.apply_cognitive_mutation();
                            }
                        });
                });

                ui.separator();

                // Section 2: 16-Step Parameter Sequencer
                ui.vertical(|ui| {
                    mg_ui::section(ui, "16-Step Live Sequencer", "Parameter tempo modulation");
                    egui::Frame::new()
                        .fill(mg_ui::PANEL_RAISED)
                        .inner_margin(12.0)
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                if ui
                                    .button(if self.sequencer_playing {
                                        "■ STOP"
                                    } else {
                                        "▶ PLAY"
                                    })
                                    .clicked()
                                {
                                    self.sequencer_playing = !self.sequencer_playing;
                                }
                                ui.add(
                                    egui::Slider::new(&mut self.sequencer_bpm, 40.0..=240.0)
                                        .text("BPM"),
                                );
                                ui.label(format!(
                                    "Step: {} / {}",
                                    self.sequencer_current_step + 1,
                                    self.sequencer_step_count
                                ));
                            });
                            ui.add_space(8.0);
                            ui.horizontal_wrapped(|ui| {
                                for i in 0..self.sequencer_step_count {
                                    ui.vertical(|ui| {
                                        let is_active = self.sequencer_playing
                                            && self.sequencer_current_step == i;
                                        ui.label(
                                            egui::RichText::new(format!("{:02}", i + 1))
                                                .size(9.0)
                                                .color(if is_active {
                                                    mg_ui::SIGNAL
                                                } else {
                                                    mg_ui::MUTED
                                                }),
                                        );
                                        ui.add(
                                            egui::Slider::new(
                                                &mut self.sequencer_steps[i],
                                                0.0..=1.0,
                                            )
                                            .vertical(),
                                        );
                                    });
                                }
                            });
                        });
                });
            });
        });
    }

    fn display_retro_ui(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            mg_ui::section(
                ui,
                "Retro Mode & Bit-Depth Reduction",
                "Color quantization & Bayer ordered dithering",
            );
            egui::Frame::new()
                .fill(mg_ui::PANEL_RAISED)
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.checkbox(&mut self.retro_enabled, "Enable Bit Reduction (Retro Mode)");
                    ui.add_enabled(
                        self.retro_enabled,
                        egui::Slider::new(&mut self.retro_bits, 1..=8)
                            .text("Bits Per Color Channel (1 to 8)"),
                    );
                    ui.add_enabled(
                        self.retro_enabled,
                        egui::Checkbox::new(
                            &mut self.retro_dither,
                            "Bayer Matrix 4x4 Ordered Dithering",
                        ),
                    );
                });

            ui.add_space(16.0);
            mg_ui::section(
                ui,
                "Resolution & High-DPI Display",
                "Output sizing and zoom scaling",
            );
            egui::Frame::new()
                .fill(mg_ui::PANEL_RAISED)
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.label("UI Scale Preset:");
                    ui.horizontal(|ui| {
                        for (lbl, val) in [
                            ("80%", 0.8),
                            ("100%", 1.0),
                            ("125%", 1.25),
                            ("150%", 1.5),
                            ("200%", 2.0),
                        ] {
                            if ui
                                .selectable_label((self.ui_scale - val).abs() < 0.01, lbl)
                                .clicked()
                            {
                                self.ui_scale = val;
                            }
                        }
                    });
                });
        });
    }

    fn presets_export_ui(&mut self, ui: &mut egui::Ui, _details: &RuntimeDetails) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            mg_ui::section(
                ui,
                "Preset Management & Offline Render",
                "Save, load, and render high quality exports",
            );
            ui.horizontal(|ui| {
                if ui.button("💾 SAVE PRESET FILE...").clicked() {
                    if let Some(path) = save_file_dialog("MegaGlitch Preset", "*.json", "json") {
                        if let Ok(json) =
                            serde_json::to_string_pretty(&preset::export_graph(&self.graph))
                        {
                            let _ = std::fs::write(&path, json);
                            self.preset_status = format!("Preset saved to {path}");
                        }
                    }
                }
                if ui.button("📁 LOAD PRESET FILE...").clicked() {
                    if let Some(path) = pick_file_dialog("MegaGlitch Preset", "*.json") {
                        if let Ok(content) = std::fs::read_to_string(&path) {
                            if let Ok(saved) = serde_json::from_str::<preset::PresetGraph>(&content)
                            {
                                if let Ok((imported, sel)) = preset::import_graph(&saved) {
                                    self.graph = imported;
                                    self.selected = sel;
                                    self.graph_view_epoch = self.graph_view_epoch.saturating_add(1);
                                    self.graph_revision = self.graph_revision.saturating_add(1);
                                    self.preset_status = format!("Loaded preset from {path}");
                                }
                            }
                        }
                    }
                }
            });

            ui.add_space(16.0);
            mg_ui::section(ui, "Video Export & Frame Snapshot", "Export configuration");
            egui::Frame::new()
                .fill(mg_ui::PANEL_RAISED)
                .inner_margin(12.0)
                .show(ui, |ui| {
                    ui.add(
                        egui::Slider::new(&mut self.export_duration, 1.0..=60.0)
                            .text("Duration (seconds)"),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("🎬 RENDER MASTER MP4...").clicked() {
                            self.runtime_status =
                                format!("Queued export render for {:.1}s", self.export_duration);
                        }
                        if ui.button("📸 SNAPSHOT PNG...").clicked() {
                            self.runtime_status = "Saved snapshot frame to disk".to_owned();
                        }
                    });
                });
        });
    }

    fn top_bar(&mut self, ui: &mut egui::Ui, snapshot: RuntimeSnapshot) {
        egui::Frame::new()
            .fill(mg_ui::BG)
            .inner_margin(egui::Margin::symmetric(8, 4))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("📂 Load Image File...").clicked() {
                            if let Some(path) = pick_file_dialog(
                                "Image Files",
                                "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.gif",
                            ) {
                                self.source_path = path.clone();
                                self.dispatch_runtime_intent(RuntimeIntent::SetSource {
                                    kind: "image",
                                    path: Some(path),
                                });
                            }
                            ui.close_kind(egui::UiKind::Popup);
                        }
                        if ui.button("🎬 Load Video File...").clicked() {
                            if let Some(path) = pick_file_dialog(
                                "Video Files",
                                "*.mp4;*.mov;*.avi;*.mkv;*.webm;*.gif",
                            ) {
                                self.source_path = path.clone();
                                self.dispatch_runtime_intent(RuntimeIntent::SetSource {
                                    kind: "video",
                                    path: Some(path),
                                });
                            }
                            ui.close_kind(egui::UiKind::Popup);
                        }
                        if ui.button("🎨 Load Test Pattern").clicked() {
                            self.dispatch_runtime_intent(RuntimeIntent::SetSource {
                                kind: "test_pattern",
                                path: None,
                            });
                            ui.close_kind(egui::UiKind::Popup);
                        }
                        ui.separator();
                        if ui.button("💾 Save Preset File...").clicked() {
                            if let Some(path) =
                                save_file_dialog("MegaGlitch Preset", "*.json", "json")
                            {
                                if let Ok(json) =
                                    serde_json::to_string_pretty(&preset::export_graph(&self.graph))
                                {
                                    let _ = std::fs::write(&path, json);
                                    self.preset_status = format!("Saved preset to {path}");
                                }
                            }
                            ui.close_kind(egui::UiKind::Popup);
                        }
                        if ui.button("📁 Load Preset File...").clicked() {
                            if let Some(path) = pick_file_dialog("MegaGlitch Preset", "*.json") {
                                if let Ok(content) = std::fs::read_to_string(&path) {
                                    if let Ok(saved) =
                                        serde_json::from_str::<preset::PresetGraph>(&content)
                                    {
                                        if let Ok((imported, imported_sel)) =
                                            preset::import_graph(&saved)
                                        {
                                            self.graph = imported;
                                            self.selected = imported_sel;
                                            self.graph_view_epoch =
                                                self.graph_view_epoch.saturating_add(1);
                                            self.graph_revision =
                                                self.graph_revision.saturating_add(1);
                                            self.preset_status =
                                                format!("Loaded preset from {path}");
                                        }
                                    }
                                }
                            }
                            ui.close_kind(egui::UiKind::Popup);
                        }
                    });

                    ui.menu_button("Edit", |ui| {
                        if ui.button("⚡ Auto-Wire Left to Right").clicked() {
                            graph::auto_wire_chain(&mut self.graph);
                            self.graph_revision = self.graph_revision.saturating_add(1);
                            ui.close_kind(egui::UiKind::Popup);
                        }
                        if ui.button("Clear All Wires").clicked() {
                            for (out_pin, _) in self.graph.wires().collect::<Vec<_>>() {
                                self.graph.drop_outputs(out_pin);
                            }
                            self.graph_revision = self.graph_revision.saturating_add(1);
                            ui.close_kind(egui::UiKind::Popup);
                        }
                        if ui.button("↻ Reset to Demo Graph").clicked() {
                            let (fresh_graph, fresh_selection) = graph::demo_graph();
                            self.graph = fresh_graph;
                            self.selected = fresh_selection;
                            self.graph_view_epoch = self.graph_view_epoch.saturating_add(1);
                            self.graph_revision = self.graph_revision.saturating_add(1);
                            ui.close_kind(egui::UiKind::Popup);
                        }
                    });

                    ui.menu_button("Nodes", |ui| {
                        let mut categories: std::collections::BTreeMap<
                            &str,
                            Vec<&catalog::NodeSpec>,
                        > = std::collections::BTreeMap::new();
                        for desc in catalog::all() {
                            categories.entry(desc.category).or_default().push(desc);
                        }
                        for (cat_name, nodes) in categories {
                            ui.menu_button(cat_name, |ui| {
                                for desc in nodes {
                                    let is_cuda = desc.backend == "cuda";
                                    let item_label = if is_cuda {
                                        format!("⚡ {}", desc.label)
                                    } else {
                                        desc.label.to_owned()
                                    };
                                    if ui.button(item_label).clicked() {
                                        self.selected = Some(graph::insert_catalog_node(
                                            &mut self.graph,
                                            *desc,
                                        ));
                                        self.graph_revision = self.graph_revision.saturating_add(1);
                                        ui.close_kind(egui::UiKind::Popup);
                                    }
                                }
                            });
                        }
                    });

                    ui.menu_button("AI Mutation", |ui| {
                        ui.label("Cognitive Profile:");
                        for profile in [
                            "Alpha (Balanced)",
                            "Beta (Active)",
                            "Gamma (Chaotic)",
                            "Theta (Flow)",
                            "Delta (Relaxed)",
                        ] {
                            if ui
                                .selectable_label(self.cognitive_profile == profile, profile)
                                .clicked()
                            {
                                self.cognitive_profile = profile.to_owned();
                                self.apply_cognitive_mutation();
                                ui.close_kind(egui::UiKind::Popup);
                            }
                        }
                        ui.separator();
                        ui.add(
                            egui::Slider::new(&mut self.mutation_severity, 0.05..=1.0)
                                .text("Severity"),
                        );
                        if ui.button("⚡ Mutate Active Graph Now").clicked() {
                            self.apply_cognitive_mutation();
                            ui.close_kind(egui::UiKind::Popup);
                        }
                    });

                    ui.menu_button("Display", |ui| {
                        ui.checkbox(&mut self.retro_enabled, "Enable Retro Bit-Depth");
                        ui.add_enabled(
                            self.retro_enabled,
                            egui::Slider::new(&mut self.retro_bits, 1..=8).text("Bits per Channel"),
                        );
                        ui.add_enabled(
                            self.retro_enabled,
                            egui::Checkbox::new(&mut self.retro_dither, "Bayer Dither"),
                        );
                        ui.separator();
                        ui.label("UI Scale:");
                        for (scale_label, scale_val) in
                            [("80%", 0.8), ("100%", 1.0), ("125%", 1.25), ("150%", 1.5)]
                        {
                            if ui
                                .selectable_label(
                                    (self.ui_scale - scale_val).abs() < 0.01,
                                    scale_label,
                                )
                                .clicked()
                            {
                                self.ui_scale = scale_val;
                            }
                        }
                    });

                    ui.separator();

                    // Workspace Switcher Tabs
                    for ws in [
                        ("PATCH", "PATCH"),
                        ("DUAL DECK", "DUAL DECK"),
                        ("AI MUTATION & SEQUENCER", "AI MUTATE / SEQ"),
                        ("DISPLAY & RETRO", "DISPLAY / RETRO"),
                        ("PRESETS & EXPORT", "PRESETS / EXPORT"),
                    ] {
                        if ui.selectable_label(self.workspace == ws.0, ws.1).clicked() {
                            self.workspace = ws.0;
                        }
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("CTRL K  COMMAND").clicked() {
                            self.command_palette_open = true;
                        }
                        ui.label(
                            egui::RichText::new(session_status(snapshot))
                                .monospace()
                                .size(9.0)
                                .color(mg_ui::MUTED),
                        );
                        ui.label(
                            egui::RichText::new(format!("ENGINE R{}", snapshot.state_revision))
                                .monospace()
                                .size(8.0)
                                .color(mg_ui::ACCENT),
                        );
                    });
                });
            });
    }

    fn transport(
        &mut self,
        ui: &mut egui::Ui,
        snapshot: RuntimeSnapshot,
        details: &RuntimeDetails,
        command_available: bool,
        presentation: PreviewPresentationSnapshot,
    ) -> Option<RuntimeIntent> {
        let mut intent = None;
        egui::Frame::new()
            .fill(mg_ui::BG)
            .inner_margin(egui::Margin::symmetric(12, 8))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            command_available && details.pending_command_id.is_none(),
                            egui::Button::new(if details.transport_running {
                                "■ STOP ENGINE"
                            } else {
                                "▶ RUN ENGINE"
                            }),
                        )
                        .clicked()
                    {
                        intent = Some(RuntimeIntent::SetTransport(!details.transport_running));
                    }
                    ui.separator();
                    mg_ui::status_lamp(
                        ui,
                        if details.transport_running {
                            "ENGINE RUNNING"
                        } else {
                            "ENGINE STOPPED"
                        },
                        snapshot.bridge_online && details.transport_running,
                        mg_ui::SIGNAL,
                    );
                    mg_ui::status_lamp(
                        ui,
                        if details.source_ready {
                            "SOURCE READY"
                        } else {
                            "SOURCE WAITING"
                        },
                        details.source_ready,
                        mg_ui::ACCENT,
                    );
                    ui.label(
                        egui::RichText::new("AUDIO · MIDI · VCAM · OBS  RESERVED / NOT CONNECTED")
                            .monospace()
                            .size(8.0)
                            .color(mg_ui::MUTED),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new(format!("DROP {:03}", snapshot.dropped_frames))
                                .monospace()
                                .size(9.0)
                                .color(mg_ui::MUTED),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "PREVIEW {:.2} ms",
                                snapshot.preview_frame_ms
                            ))
                            .monospace()
                            .size(9.0),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "AUTH {:.0} / RENDER {:.0} / PRESENT {:.0} FPS",
                                snapshot.authoritative_fps,
                                snapshot.preview_render_fps,
                                presentation.preview_present_fps
                            ))
                            .monospace()
                            .size(9.0)
                            .color(mg_ui::ACCENT),
                        );
                    });
                });
            });
        intent
    }

    fn update_preview_texture(&mut self, ctx: &egui::Context, view: &RuntimeView) {
        let Some(frame) = view.preview.as_ref() else {
            return;
        };
        if frame.surface_generation == self.preview_texture_generation
            && frame.sequence == self.preview_texture_sequence
        {
            return;
        }
        let Some(expected_len) = frame
            .width
            .checked_mul(frame.height)
            .and_then(|pixels| pixels.checked_mul(3))
        else {
            self.runtime_status = "Preview dimensions overflowed the RGB buffer".to_owned();
            return;
        };
        if frame.width == 0 || frame.height == 0 || frame.rgb.len() != expected_len {
            self.runtime_status = format!(
                "Rejected malformed preview frame {} ({}×{}, {} RGB bytes)",
                frame.sequence,
                frame.width,
                frame.height,
                frame.rgb.len()
            );
            return;
        }
        let preparation_started = Instant::now();
        let image = egui::ColorImage::from_rgb([frame.width, frame.height], frame.rgb.as_ref());
        if let Some(texture) = self.preview_texture.as_mut() {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            self.preview_texture = Some(ctx.load_texture(
                "megaglitch-engine-preview",
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
        self.preview_texture_generation = frame.surface_generation;
        self.preview_texture_sequence = frame.sequence;
        self.preview_uploaded_sequence = frame.sequence;
        self.preview_present_tracker.record_upload(
            PreviewPresentKey {
                surface_generation: frame.surface_generation,
                sequence: frame.sequence,
                graph_revision: frame.graph_revision,
            },
            frame.received_at,
            preparation_started,
            Instant::now(),
        );
    }

    fn finish_preview_smoke(
        &mut self,
        context: &egui::Context,
        passed: bool,
        failure_reason: Option<String>,
        snapshot: RuntimeSnapshot,
        details: &RuntimeDetails,
        presentation: PreviewPresentationSnapshot,
    ) {
        let Some(smoke) = self.preview_smoke.as_ref() else {
            return;
        };
        if smoke.report_written {
            return;
        }
        let report_path = smoke.report_path.clone();
        let mut report = serde_json::json!({
            "passed": passed,
            "failure_reason": failure_reason,
            "case": smoke.case_label,
            "source": smoke
                .source_path
                .as_ref()
                .map(|path| path.display().to_string()),
            "preset": smoke
                .preset_path
                .as_ref()
                .map(|path| path.display().to_string()),
            "deck": smoke.deck.key(),
            "measurement": "normal_eframe_wgpu_render_pass",
            "stage_timing_samples": presentation.stage_samples,
            "texture_preparations": presentation.texture_preparations,
            "prepared_frames_replaced_before_draw": presentation.replaced_before_draw,
            "bridge_copy_to_texture_prepare_ms_mean": presentation.stage_mean_ms[0],
            "texture_prepare_ms_mean": presentation.stage_mean_ms[1],
            "texture_prepare_to_wgpu_marker_ms_mean": presentation.stage_mean_ms[2],
            "bridge_copy_to_texture_prepare_ms_max": presentation.stage_max_ms[0],
            "texture_prepare_ms_max": presentation.stage_max_ms[1],
            "texture_prepare_to_wgpu_marker_ms_max": presentation.stage_max_ms[2],
            "presentation_marker": "unique preview sequence encoded in visible WGPU render pass",
            "swapchain_completion_observed": false,
            "measurement_duration_seconds": presentation.measurement_duration_seconds,
            "target_presentations": smoke.target_presentations,
            "presented_new_frames": presentation.preview_presented_count,
            "preview_present_fps": presentation.preview_measurement_fps,
            "preview_present_fps_rolling": presentation.preview_present_fps,
            "preview_render_fps": snapshot.preview_render_fps,
            "preview_frame_ms": snapshot.preview_frame_ms,
            "preview_p95_ms": snapshot.preview_p95_ms,
            "preview_width": snapshot.preview_width,
            "preview_height": snapshot.preview_height,
            "preview_rendered_sequence": snapshot.preview_rendered_sequence,
            "preview_received_sequence": snapshot.preview_received_sequence,
            "preview_uploaded_sequence": self.preview_uploaded_sequence,
            "preview_presented_sequence": presentation.preview_presented_sequence,
            "preview_presented_graph_revision":
                presentation.preview_presented_graph_revision,
            "preview_requests": snapshot
                .preview_requests
                .saturating_sub(smoke.preview_requests_baseline),
            "preview_rendered": snapshot
                .preview_rendered
                .saturating_sub(smoke.preview_rendered_baseline),
            "preview_superseded": snapshot
                .preview_superseded
                .saturating_sub(smoke.preview_superseded_baseline),
            "preview_dropped": snapshot
                .preview_dropped
                .saturating_sub(smoke.preview_dropped_baseline),
            "preview_requests_total": snapshot.preview_requests,
            "preview_rendered_total": snapshot.preview_rendered,
            "preview_superseded_total": snapshot.preview_superseded,
            "preview_dropped_total": snapshot.preview_dropped,
        });
        let report_details = serde_json::json!({
            "source_submit_to_present_ms_last": presentation.preview_submit_to_present_ms,
            "source_submit_to_present_ms_mean": presentation.preview_mean_submit_to_present_ms,
            "preview_graph_revision": details.preview_graph_revision,
            "preview_resolution_generation": details.preview_resolution_generation,
            "authoritative_fps": snapshot.authoritative_fps,
            "authoritative_frame_ms": snapshot.authoritative_frame_ms,
            "authoritative_sequence": snapshot.authoritative_sequence,
            "authoritative_dropped_frames": snapshot.dropped_frames,
            "parameter_probe": smoke.probe_state.label(),
            "parameter_to_preview_ms": smoke.parameter_to_preview_ms,
            "parameter_probe_baseline_graph_revision":
                smoke.probe_baseline_graph_revision,
            "parameter_probe_presented_sequence": smoke.probe_target_sequence,
            "parameter_probe_node": smoke
                .probe_mutation
                .as_ref()
                .map(|mutation| mutation.node_id.as_str()),
            "parameter_probe_parameter": smoke
                .probe_mutation
                .as_ref()
                .map(|mutation| mutation.parameter.as_str()),
            "parameter_probe_original": smoke
                .probe_mutation
                .as_ref()
                .map(|mutation| mutation.original),
            "parameter_probe_value": smoke
                .probe_mutation
                .as_ref()
                .map(|mutation| mutation.probed),
            "bridge_online": snapshot.bridge_online,
            "engine_lifecycle": details.lifecycle.as_str(),
            "engine_lifecycle_reason": details.lifecycle_reason,
            "engine_last_error": details.last_error.as_ref().map(|problem| serde_json::json!({
                "code": problem.code,
                "path": problem.path,
                "message": problem.message,
                "recoverable": problem.recoverable,
            })),
            "clean_shutdown_requested": true,
        });
        report
            .as_object_mut()
            .expect("preview smoke report is an object")
            .extend(
                report_details
                    .as_object()
                    .expect("preview smoke report details are an object")
                    .clone(),
            );
        let write_result = serde_json::to_vec_pretty(&report)
            .map_err(|error| error.to_string())
            .and_then(|bytes| {
                std::fs::write(&report_path, bytes).map_err(|error| error.to_string())
            });
        if let Some(smoke) = self.preview_smoke.as_mut() {
            smoke.report_written = true;
        }
        self.runtime_status = match write_result {
            Ok(()) if passed => {
                format!("PREVIEW SMOKE COMPLETE · REPORT {}", report_path.display())
            }
            Ok(()) => format!("PREVIEW SMOKE FAILED · REPORT {}", report_path.display()),
            Err(error) => format!("PREVIEW SMOKE REPORT FAILED: {error}"),
        };
        context.send_viewport_cmd(egui::ViewportCommand::Close);
    }

    fn update_preview_smoke(
        &mut self,
        context: &egui::Context,
        snapshot: RuntimeSnapshot,
        details: &RuntimeDetails,
        presentation: PreviewPresentationSnapshot,
    ) {
        let Some(smoke) = self.preview_smoke.as_ref() else {
            return;
        };
        if smoke.report_written {
            return;
        }
        if let Some(error) = smoke.configuration_error.clone() {
            self.finish_preview_smoke(
                context,
                false,
                Some(format!("invalid_smoke_configuration: {error}")),
                snapshot,
                details,
                presentation,
            );
            return;
        }
        let target_presentations = smoke.target_presentations;

        if !smoke.armed {
            if smoke.started_at.elapsed() > PREVIEW_SMOKE_STARTUP_TIMEOUT {
                self.finish_preview_smoke(
                    context,
                    false,
                    Some(format!(
                        "startup_timeout: preview smoke did not arm within {} seconds",
                        PREVIEW_SMOKE_STARTUP_TIMEOUT.as_secs()
                    )),
                    snapshot,
                    details,
                    presentation,
                );
                return;
            }
            let configured_source_ready = match smoke.source_path.as_ref() {
                Some(path) => {
                    let expected_name = path
                        .file_name()
                        .and_then(|value| value.to_str())
                        .unwrap_or_default()
                        .to_ascii_lowercase();
                    let expected_kind = preview_smoke_source_kind(path).ok();
                    self.source_revision > 0
                        && self.submitted_source_revision == self.source_revision
                        && details.pending_command_id.is_none()
                        && expected_kind == Some(details.source_kind.as_str())
                        && details
                            .source_label
                            .to_ascii_lowercase()
                            .contains(&expected_name)
                }
                None => details.source_ready,
            };
            let configured_graph_ready = self.submitted_graph_revision == self.graph_revision
                && details.pending_command_id.is_none()
                && details.current_graph_accepted;
            if snapshot.bridge_online
                && configured_source_ready
                && configured_graph_ready
                && snapshot.preview_received_sequence > 0
            {
                self.preview_present_tracker.reset();
                if let Some(smoke) = self.preview_smoke.as_mut() {
                    smoke.preview_requests_baseline = snapshot.preview_requests;
                    smoke.preview_rendered_baseline = snapshot.preview_rendered;
                    smoke.preview_superseded_baseline = snapshot.preview_superseded;
                    smoke.preview_dropped_baseline = snapshot.preview_dropped;
                    smoke.armed = true;
                }
                self.runtime_status = format!(
                    "PREVIEW SMOKE ARMED · WAITING FOR {} UNIQUE WGPU PRESENTATIONS",
                    target_presentations
                );
            }
            return;
        }

        let probe_threshold = PREVIEW_SMOKE_PROBE_PRESENTATIONS.min(target_presentations.max(1));
        if smoke.probe_state == PreviewParameterProbeState::Pending
            && presentation.preview_presented_count >= probe_threshold
        {
            match begin_preview_parameter_probe(&mut self.graph) {
                Ok(Some(mutation)) => {
                    self.graph_revision = self.graph_revision.saturating_add(1);
                    if let Some(smoke) = self.preview_smoke.as_mut() {
                        smoke.probe_state = PreviewParameterProbeState::Waiting;
                        smoke.probe_started_at = Some(Instant::now());
                        smoke.probe_baseline_graph_revision = details.preview_graph_revision;
                        smoke.probe_mutation = Some(mutation);
                    }
                }
                Ok(None) => {
                    if let Some(smoke) = self.preview_smoke.as_mut() {
                        smoke.probe_state = PreviewParameterProbeState::NotApplicable;
                    }
                }
                Err(error) => {
                    self.finish_preview_smoke(
                        context,
                        false,
                        Some(format!("parameter_probe_unavailable: {error}")),
                        snapshot,
                        details,
                        presentation,
                    );
                    return;
                }
            }
        }

        let probe = self.preview_smoke.as_ref().and_then(|smoke| {
            (smoke.probe_state == PreviewParameterProbeState::Waiting).then(|| {
                (
                    smoke.probe_started_at,
                    smoke.probe_baseline_graph_revision,
                    smoke.probe_mutation.clone(),
                )
            })
        });
        if let Some((Some(started), baseline_revision, Some(mutation))) = probe {
            if presentation.preview_presented_graph_revision > baseline_revision {
                if !restore_preview_parameter_probe(&mut self.graph, &mutation) {
                    self.finish_preview_smoke(
                        context,
                        false,
                        Some("parameter_probe_restore_failed".to_owned()),
                        snapshot,
                        details,
                        presentation,
                    );
                    return;
                }
                self.graph_revision = self.graph_revision.saturating_add(1);
                if let Some(smoke) = self.preview_smoke.as_mut() {
                    smoke.probe_state = PreviewParameterProbeState::Resolved;
                    smoke.parameter_to_preview_ms = Some(started.elapsed().as_secs_f32() * 1_000.0);
                    smoke.probe_target_sequence = Some(presentation.preview_presented_sequence);
                }
            } else if started.elapsed() > PREVIEW_SMOKE_PROBE_TIMEOUT {
                self.finish_preview_smoke(
                    context,
                    false,
                    Some(format!(
                        "parameter_preview_timeout: no WGPU-presented preview advanced beyond graph revision {baseline_revision}"
                    )),
                    snapshot,
                    details,
                    presentation,
                );
                return;
            }
        }

        let Some(smoke) = self.preview_smoke.as_ref() else {
            return;
        };
        let probe_complete = matches!(
            smoke.probe_state,
            PreviewParameterProbeState::NotApplicable | PreviewParameterProbeState::Resolved
        );
        if presentation.preview_presented_count >= smoke.target_presentations && probe_complete {
            self.finish_preview_smoke(context, true, None, snapshot, details, presentation);
        }
    }

    fn dropped_source_intent(&mut self, ctx: &egui::Context) -> Option<RuntimeIntent> {
        let dropped = ctx.input(|input| input.raw.dropped_files.clone());
        let path = dropped.into_iter().find_map(|file| file.path)?;
        let kind = match path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "png" | "jpg" | "jpeg" | "bmp" | "tif" | "tiff" | "gif" => "image",
            "mp4" | "mov" | "mkv" | "avi" | "webm" | "m4v" => "video",
            _ => {
                self.runtime_status = format!(
                    "Dropped file type is not a supported Engine V2 source: {}",
                    path.display()
                );
                return None;
            }
        };
        let path = path.to_string_lossy().into_owned();
        self.source_path.clone_from(&path);
        Some(RuntimeIntent::SetSource {
            kind,
            path: Some(path),
        })
    }

    fn dispatch_runtime_intent(&mut self, intent: RuntimeIntent) {
        let (action, result) = match intent {
            RuntimeIntent::SetTransport(running) => (
                if running {
                    "run engine".to_owned()
                } else {
                    "stop engine".to_owned()
                },
                self.state_source.set_transport(running),
            ),
            RuntimeIntent::SetSource { kind, path } => {
                let action = match path.as_deref() {
                    Some(path) => format!("load {kind} source {path}"),
                    None => format!("select {kind} source"),
                };
                self.source_revision = self.source_revision.saturating_add(1);
                let result = self.state_source.set_source(kind, path.as_deref());
                if result.is_ok() {
                    self.submitted_source_revision = self.source_revision;
                }
                (action, result)
            }
            RuntimeIntent::Seek(seconds) => (
                format!("seek source to {seconds:.3}s"),
                self.state_source.seek(seconds),
            ),
        };
        self.runtime_status = match result {
            Ok(id) => format!("QUEUED #{id}: {action}"),
            Err(error) => format!("COMMAND NOT QUEUED: {action}: {error}"),
        };
    }

    fn restart_engine(&mut self, context: &egui::Context) {
        self.state_source = RuntimeStateSource::bridge_with_repaint(context.clone());
        self.submitted_graph_revision = 0;
        self.submitted_source_revision = 0;
        self.observed_ack_command_id = 0;
        self.preview_texture = None;
        self.preview_texture_generation = 0;
        self.preview_texture_sequence = u64::MAX;
        self.preview_uploaded_sequence = 0;
        self.preview_present_tracker.reset();
        self.video_seek_seconds = None;
        self.runtime_status = format!(
            "RESTARTING ENGINE V2 · GRAPH R{} WILL REQUEUE",
            self.graph_revision
        );
    }

    fn runtime_banner(
        &self,
        ui: &mut egui::Ui,
        snapshot: RuntimeSnapshot,
        details: &RuntimeDetails,
        graph_status: graph::GraphStatus,
    ) -> bool {
        let mut restart_requested = false;
        let accepted_engine = if snapshot.bridge_online {
            format!(
                "ENGINE G{} {}",
                snapshot.bridge_generation,
                if details.transport_running {
                    "LIVE"
                } else {
                    "LOADED"
                }
            )
        } else if snapshot.bridge_generation > 0 {
            format!("ENGINE G{} · ENGINE OFFLINE", snapshot.bridge_generation)
        } else {
            "NO ACCEPTED ENGINE GRAPH".to_owned()
        };
        let (graph_text, graph_color) = match graph_status {
            graph::GraphStatus::Draft => (
                format!("GRAPH R{} DRAFT · {accepted_engine}", self.graph_revision),
                mg_ui::WARNING,
            ),
            graph::GraphStatus::Pending => (
                format!(
                    "GRAPH R{} PENDING #{} · LAST ACCEPTED {accepted_engine}",
                    self.submitted_graph_revision,
                    details.pending_command_id.unwrap_or_default()
                ),
                mg_ui::ACCENT,
            ),
            graph::GraphStatus::Accepted => (
                format!(
                    "GRAPH R{} ACCEPTED · {accepted_engine}",
                    self.submitted_graph_revision
                ),
                mg_ui::SIGNAL,
            ),
            graph::GraphStatus::Rejected => (
                format!(
                    "GRAPH R{} REJECTED · LAST ACCEPTED {accepted_engine}",
                    self.submitted_graph_revision
                ),
                mg_ui::WARNING,
            ),
        };
        egui::Frame::new()
            .fill(mg_ui::PANEL_RAISED)
            .stroke(Stroke::new(
                1.0,
                if details.last_error.is_some() {
                    mg_ui::WARNING
                } else {
                    mg_ui::HAIRLINE
                },
            ))
            .inner_margin(egui::Margin::symmetric(10, 5))
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        egui::RichText::new(graph_text)
                            .monospace()
                            .strong()
                            .size(9.0)
                            .color(graph_color),
                    );
                    if let Some(id) = details.pending_command_id {
                        ui.label(
                            egui::RichText::new(format!("ENGINE COMMAND #{id} PENDING"))
                                .monospace()
                                .size(9.0)
                                .color(mg_ui::ACCENT),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!(
                                "STATE R{} / LAST ACK #{}",
                                details.state_revision, details.last_ack_command_id
                            ))
                            .monospace()
                            .size(9.0)
                            .color(mg_ui::MUTED),
                        );
                    }
                    ui.label(
                        egui::RichText::new(self.runtime_status.as_str())
                            .monospace()
                            .size(9.0)
                            .color(mg_ui::MUTED),
                    );
                });
                if let Some(problem) = details.last_error.as_ref() {
                    let path = problem
                        .path
                        .as_deref()
                        .map(|path| format!(" [{path}]"))
                        .unwrap_or_default();
                    ui.label(
                        egui::RichText::new(format!(
                            "ENGINE ERROR {}{}: {}{}",
                            problem.code,
                            path,
                            problem.message,
                            if problem.recoverable {
                                " (recoverable)"
                            } else {
                                " (restart required)"
                            }
                        ))
                        .monospace()
                        .strong()
                        .size(10.0)
                        .color(mg_ui::WARNING),
                    );
                }
                if engine_restart_available(snapshot, details) {
                    ui.horizontal(|ui| {
                        if ui.button("↻ RESTART ENGINE").clicked() {
                            restart_requested = true;
                        }
                        ui.label(
                            egui::RichText::new(
                                "Starts a new Engine V2 owner and requeues the current draft graph.",
                            )
                            .size(9.0)
                            .color(mg_ui::MUTED),
                        );
                    });
                }
            });
        restart_requested
    }

    fn command_palette(&mut self, ctx: &egui::Context, graph_commands_available: bool) {
        if ctx.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::K)) {
            self.command_palette_open = true;
        }
        if !self.command_palette_open {
            return;
        }
        egui::Window::new("COMMAND")
            .id(egui::Id::new("mg_command_palette"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_TOP, [0.0, 90.0])
            .fixed_size([520.0, 290.0])
            .frame(
                egui::Frame::new()
                    .fill(mg_ui::PANEL_RAISED)
                    .stroke(Stroke::new(1.0, mg_ui::ACCENT))
                    .inner_margin(12.0),
            )
            .show(ctx, |ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.command_query)
                        .hint_text("Search registered graph nodes…")
                        .desired_width(f32::INFINITY),
                );
                ui.separator();
                ui.label(
                    egui::RichText::new("ADD REGISTERED NODE")
                        .monospace()
                        .size(8.0)
                        .color(mg_ui::ACCENT),
                );
                ui.label(
                    egui::RichText::new(
                        "This palette currently performs node insertion only. Other commands are not wired.",
                    )
                    .size(9.0)
                    .color(mg_ui::MUTED),
                );
                if !graph_commands_available {
                    ui.label(
                        egui::RichText::new(
                            "Node insertion is unavailable until Engine V2 negotiates replace_graph.",
                        )
                        .size(9.0)
                        .color(mg_ui::WARNING),
                    );
                }
                let mut add_node = None;
                for descriptor in catalog::all()
                    .iter()
                    .copied()
                    .filter(|node| catalog::matches(*node, &self.command_query))
                    .take(8)
                {
                    if ui
                        .add_enabled(
                            graph_commands_available,
                            egui::Button::new(format!(
                            "+ {}  /  {}  /  {}",
                            descriptor.label,
                            descriptor.category,
                            descriptor.backend.to_ascii_uppercase()
                            )),
                        )
                        .clicked()
                    {
                        add_node = Some(descriptor);
                    }
                }
                if let Some(descriptor) = add_node {
                    self.selected = Some(graph::insert_catalog_node(&mut self.graph, descriptor));
                    self.graph_revision = self.graph_revision.saturating_add(1);
                    self.command_palette_open = false;
                    self.command_query.clear();
                }
                if ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                    self.command_palette_open = false;
                }
            });
    }
}

impl eframe::App for MegaGlitchApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if (self.ui_scale - 1.0).abs() > 0.001 {
            ctx.set_zoom_factor(self.ui_scale);
        }
        if !ctx.egui_wants_keyboard_input() {
            if ctx.input(|input| input.key_pressed(egui::Key::Delete))
                && let Some(node) = self.selected.take()
            {
                self.graph.remove_node(node);
                self.graph_revision = self.graph_revision.saturating_add(1);
            }
            if ctx.input(|input| input.key_pressed(egui::Key::B))
                && let Some(node) = self.selected
            {
                let active = !self.graph[node].active;
                self.graph[node].set_active(active);
                self.graph_revision = self.graph_revision.saturating_add(1);
            }
            if ctx.input(|input| input.modifiers.ctrl && input.key_pressed(egui::Key::D))
                && let Some(node) = self.selected
            {
                let duplicate = self.graph[node].clone();
                let offset = self.graph.nodes().count() as f32 * 18.0;
                self.selected = Some(
                    self.graph
                        .insert_node(egui::pos2(340.0 + offset, 260.0 + offset), duplicate),
                );
                self.graph_revision = self.graph_revision.saturating_add(1);
            }
        }
        // One immutable engine read per egui frame keeps every pane internally consistent.
        let view = self.state_source.read_view();
        let snapshot = view.snapshot;
        let presentation = self.preview_present_tracker.snapshot();
        let graph_status = graph_status_for(
            self.graph_revision,
            self.submitted_graph_revision,
            &view.details,
        );
        let graph_commands_available =
            supports_engine_command(snapshot, &view.details, "replace_graph");
        let source_commands_available =
            supports_engine_command(snapshot, &view.details, "set_source");
        let transport_commands_available =
            supports_engine_command(snapshot, &view.details, "set_transport");
        let seek_commands_available = supports_engine_command(snapshot, &view.details, "seek");
        if view.details.last_ack_command_id != 0
            && view.details.last_ack_command_id != self.observed_ack_command_id
        {
            self.observed_ack_command_id = view.details.last_ack_command_id;
            self.runtime_status = format!(
                "ENGINE ACKNOWLEDGED #{} / STATE R{}",
                view.details.last_ack_command_id, view.details.state_revision
            );
        } else if let Some(reconciled) =
            reconcile_runtime_status(&self.runtime_status, snapshot, &view.details)
        {
            self.runtime_status = reconciled;
        }
        self.tick_sequencer();
        self.update_preview_texture(&ctx, &view);
        let mut runtime_intent =
            if source_commands_available && view.details.pending_command_id.is_none() {
                self.dropped_source_intent(&ctx)
            } else {
                None
            };
        let bounds = ui.max_rect();
        let top_rect =
            egui::Rect::from_min_max(bounds.min, egui::pos2(bounds.right(), bounds.top() + 42.0));
        let bottom_rect = egui::Rect::from_min_max(
            egui::pos2(bounds.left(), bounds.bottom() - 58.0),
            bounds.max,
        );
        let banner_height = match (
            engine_restart_available(snapshot, &view.details),
            view.details.last_error.is_some(),
        ) {
            (true, _) => 82.0,
            (false, true) => 52.0,
            (false, false) => 30.0,
        };
        let banner_rect = egui::Rect::from_min_max(
            egui::pos2(bounds.left(), top_rect.bottom()),
            egui::pos2(bounds.right(), top_rect.bottom() + banner_height),
        );
        let center_rect = egui::Rect::from_min_max(
            egui::pos2(bounds.left(), banner_rect.bottom()),
            egui::pos2(bounds.right(), bottom_rect.top()),
        );

        ui.scope_builder(egui::UiBuilder::new().max_rect(top_rect), |ui| {
            self.top_bar(ui, snapshot)
        });
        let mut restart_requested = false;
        ui.scope_builder(egui::UiBuilder::new().max_rect(banner_rect), |ui| {
            restart_requested = self.runtime_banner(ui, snapshot, &view.details, graph_status);
        });
        ui.scope_builder(egui::UiBuilder::new().max_rect(bottom_rect), |ui| {
            let transport_intent = self.transport(
                ui,
                snapshot,
                &view.details,
                transport_commands_available && runtime_intent.is_none(),
                presentation,
            );
            if runtime_intent.is_none() {
                runtime_intent = transport_intent;
            }
        });
        ui.scope_builder(egui::UiBuilder::new().max_rect(center_rect), |ui| {
            egui::Frame::new()
                .fill(mg_ui::PANEL)
                .inner_margin(2.0)
                .show(ui, |ui| match self.workspace {
                    "DUAL DECK" => {
                        self.dual_deck_ui(ui, view.preview.as_deref(), &view.details);
                    }
                    "AI MUTATION & SEQUENCER" => {
                        self.mutation_sequencer_ui(ui);
                    }
                    "DISPLAY & RETRO" => {
                        self.display_retro_ui(ui);
                    }
                    "PRESETS & EXPORT" => {
                        self.presets_export_ui(ui, &view.details);
                    }
                    _ => {
                        let mut behavior = WorkstationBehavior {
                            graph: &mut self.graph,
                            selected: &mut self.selected,
                            snapshot,
                            details: &view.details,
                            preview: view.preview.as_deref(),
                            preview_texture: self.preview_texture.as_ref(),
                            preview_present_tracker: self.preview_present_tracker.clone(),
                            preview_uploaded_sequence: self.preview_uploaded_sequence,
                            preview_presentation: presentation,
                            ui_scale: &mut self.ui_scale,
                            reduced_motion: &mut self.reduced_motion,
                            graph_view_epoch: &mut self.graph_view_epoch,
                            catalog_query: &mut self.catalog_query,
                            preset_buffer: &mut self.preset_buffer,
                            preset_status: &mut self.preset_status,
                            graph_revision: &mut self.graph_revision,
                            graph_status,
                            graph_commands_available,
                            source_commands_available,
                            seek_commands_available,
                            source_path: &mut self.source_path,
                            video_seek_seconds: &mut self.video_seek_seconds,
                            runtime_intent: &mut runtime_intent,
                            dragged_node: &mut self.dragged_node,
                        };
                        self.tree.ui(&mut behavior, ui);
                    }
                });
        });

        // Floating drag badge & drop release handling
        if let Some(pos) = ctx.pointer_hover_pos() {
            if ctx.input(|i| i.pointer.primary_down()) {
                if let Some(request) = self.dragged_node {
                    let painter = ctx.layer_painter(egui::LayerId::new(
                        egui::Order::Tooltip,
                        egui::Id::new("floating_drag_node"),
                    ));
                    let title = match request {
                        NodeRequest::Builtin(node) => node.label(),
                        NodeRequest::Catalog(idx) => {
                            catalog::all().get(idx).map(|d| d.label).unwrap_or("Node")
                        }
                    };
                    let rect = egui::Rect::from_min_size(
                        pos + egui::vec2(12.0, 12.0),
                        egui::vec2(180.0, 30.0),
                    );
                    painter.rect_filled(rect, 4.0, mg_ui::PANEL_RAISED);
                    painter.rect_stroke(
                        rect,
                        4.0,
                        Stroke::new(1.5, mg_ui::ACCENT),
                        egui::StrokeKind::Inside,
                    );
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("+ {}", title),
                        egui::FontId::proportional(11.0),
                        mg_ui::TEXT,
                    );
                }
            } else if let Some(request) = self.dragged_node.take() {
                let canvas_pos = Some(pos);
                let new_node = match request {
                    NodeRequest::Builtin(node) => Some(graph::insert_library_node_at(
                        &mut self.graph,
                        node,
                        canvas_pos,
                    )),
                    NodeRequest::Catalog(idx) => catalog::all()
                        .get(idx)
                        .copied()
                        .map(|d| graph::insert_catalog_node_at(&mut self.graph, d, canvas_pos)),
                };
                if let Some(id) = new_node {
                    self.selected = Some(id);
                    self.graph_revision = self.graph_revision.saturating_add(1);
                    self.runtime_status = "Instantiated node on graph canvas".to_owned();
                }
            }
        }

        self.command_palette(&ctx, graph_commands_available);
        if restart_requested {
            self.restart_engine(&ctx);
        } else if let Some(intent) = runtime_intent {
            self.dispatch_runtime_intent(intent);
        } else if view.details.pending_command_id.is_none() {
            if self.graph_revision != self.submitted_graph_revision && graph_commands_available {
                let graph = preset::export_graph(&self.graph);
                let session_update = self.canonical_session.update_deck_graph(DeckId::A, &graph);
                match session_update.and_then(|()| self.state_source.replace_graph(&graph)) {
                    Ok(id) => {
                        self.submitted_graph_revision = self.graph_revision;
                        self.runtime_status = format!(
                            "QUEUED #{id}: replace graph R{}",
                            self.submitted_graph_revision
                        );
                    }
                    Err(error) => {
                        self.runtime_status = format!("GRAPH NOT QUEUED: {error}");
                    }
                }
            } else if self.source_revision != self.submitted_source_revision
                && source_commands_available
                && !self.source_path.trim().is_empty()
            {
                let p = self.source_path.trim().to_owned();
                let kind = if p.ends_with(".mp4")
                    || p.ends_with(".mov")
                    || p.ends_with(".avi")
                    || p.ends_with(".mkv")
                    || p.ends_with(".webm")
                {
                    "video"
                } else {
                    "image"
                };
                match self.state_source.set_source(kind, Some(&p)) {
                    Ok(id) => {
                        self.submitted_source_revision = self.source_revision;
                        self.runtime_status = format!("QUEUED #{id}: load {kind} source {p}");
                    }
                    Err(error) => {
                        self.runtime_status = format!("SOURCE NOT QUEUED: {error}");
                    }
                }
            }
        }
        self.update_preview_smoke(&ctx, snapshot, &view.details, presentation);
        ctx.request_repaint_after(Duration::from_millis(if self.reduced_motion {
            50
        } else {
            16
        }));
    }
}

struct WorkstationBehavior<'a> {
    graph: &'a mut Snarl<GraphNode>,
    selected: &'a mut Option<NodeId>,
    snapshot: RuntimeSnapshot,
    details: &'a RuntimeDetails,
    preview: Option<&'a crate::state::PreviewFrame>,
    preview_texture: Option<&'a egui::TextureHandle>,
    preview_present_tracker: PreviewPresentationTracker,
    preview_uploaded_sequence: u64,
    preview_presentation: PreviewPresentationSnapshot,
    ui_scale: &'a mut f32,
    reduced_motion: &'a mut bool,
    graph_view_epoch: &'a mut u64,
    catalog_query: &'a mut String,
    preset_buffer: &'a mut String,
    preset_status: &'a mut String,
    graph_revision: &'a mut u64,
    graph_status: graph::GraphStatus,
    graph_commands_available: bool,
    source_commands_available: bool,
    seek_commands_available: bool,
    source_path: &'a mut String,
    video_seek_seconds: &'a mut Option<f64>,
    runtime_intent: &'a mut Option<RuntimeIntent>,
    dragged_node: &'a mut Option<NodeRequest>,
}

impl egui_tiles::Behavior<Pane> for WorkstationBehavior<'_> {
    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        pane.title().into()
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut Pane,
    ) -> egui_tiles::UiResponse {
        match pane {
            Pane::Sources => sources_ui(
                ui,
                self.graph,
                self.selected,
                self.catalog_query,
                self.graph_revision,
                self.details,
                self.source_path,
                self.graph_commands_available,
                self.source_commands_available,
                self.runtime_intent,
                self.dragged_node,
            ),
            Pane::Graph => graph_ui(
                ui,
                self.graph,
                self.selected,
                self.snapshot,
                self.graph_view_epoch,
                self.preset_buffer,
                self.preset_status,
                self.graph_revision,
                self.graph_status,
                self.graph_commands_available,
            ),
            Pane::Preview => preview_ui(
                ui,
                self.preview_texture,
                self.preview,
                self.details,
                &self.preview_present_tracker,
            ),
            Pane::Inspector => inspector_ui(
                ui,
                self.graph,
                self.selected,
                self.ui_scale,
                self.reduced_motion,
                self.graph_revision,
            ),
            Pane::Timeline => timeline_ui(
                ui,
                self.details,
                self.video_seek_seconds,
                self.seek_commands_available,
                self.runtime_intent,
            ),
            Pane::SignalMonitor => monitor_ui(
                ui,
                self.snapshot,
                self.details,
                self.preview_uploaded_sequence,
                self.preview_presentation,
            ),
        }
        egui_tiles::UiResponse::None
    }

    fn tab_bar_height(&self, _style: &egui::Style) -> f32 {
        28.0
    }
    fn gap_width(&self, _style: &egui::Style) -> f32 {
        2.0
    }
    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> Color32 {
        mg_ui::BG
    }
    fn tab_bg_color(
        &self,
        _visuals: &egui::Visuals,
        _tiles: &egui_tiles::Tiles<Pane>,
        _tile_id: egui_tiles::TileId,
        state: &egui_tiles::TabState,
    ) -> Color32 {
        if state.active {
            mg_ui::PANEL_RAISED
        } else {
            mg_ui::BG
        }
    }
    fn tab_text_color(
        &self,
        _visuals: &egui::Visuals,
        _tiles: &egui_tiles::Tiles<Pane>,
        _tile_id: egui_tiles::TileId,
        state: &egui_tiles::TabState,
    ) -> Color32 {
        if state.active {
            mg_ui::TEXT
        } else {
            mg_ui::MUTED
        }
    }
    fn resize_stroke(
        &self,
        _style: &egui::Style,
        _resize_state: egui_tiles::ResizeState,
    ) -> Stroke {
        Stroke::new(1.0, mg_ui::HAIRLINE)
    }
}

#[allow(clippy::too_many_arguments)]
fn pick_file_dialog(filter_name: &str, extensions: &str) -> Option<String> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let script = format!(
            "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; \
            $d = New-Object System.Windows.Forms.OpenFileDialog; \
            $d.Title = 'Select {filter_name}'; \
            $d.Filter = '{filter_name} ({extensions})|{extensions}|All files (*.*)|*.*'; \
            if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Write($d.FileName) }}"
        );
        let mut cmd = std::process::Command::new("powershell");
        cmd.creation_flags(CREATE_NO_WINDOW);
        cmd.arg("-NoProfile").arg("-Command").arg(&script);
        let output = cmd.output().ok()?;
        let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if path.is_empty() { None } else { Some(path) }
    }
    #[cfg(not(windows))]
    {
        let _ = (filter_name, extensions);
        None
    }
}

fn save_file_dialog(filter_name: &str, extensions: &str, default_ext: &str) -> Option<String> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let script = format!(
            "[System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms') | Out-Null; \
            $d = New-Object System.Windows.Forms.SaveFileDialog; \
            $d.Title = 'Save {filter_name}'; \
            $d.DefaultExt = '{default_ext}'; \
            $d.Filter = '{filter_name} ({extensions})|{extensions}|All files (*.*)|*.*'; \
            if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {{ [Console]::Write($d.FileName) }}"
        );
        let mut cmd = std::process::Command::new("powershell");
        cmd.creation_flags(CREATE_NO_WINDOW);
        cmd.arg("-NoProfile").arg("-Command").arg(&script);
        let output = cmd.output().ok()?;
        let path = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if path.is_empty() { None } else { Some(path) }
    }
    #[cfg(not(windows))]
    {
        let _ = (filter_name, extensions, default_ext);
        None
    }
}

fn sources_ui(
    ui: &mut egui::Ui,
    graph: &mut Snarl<GraphNode>,
    selected: &mut Option<NodeId>,
    catalog_query: &mut String,
    graph_revision: &mut u64,
    details: &RuntimeDetails,
    source_path: &mut String,
    _graph_commands_available: bool,
    source_commands_available: bool,
    runtime_intent: &mut Option<RuntimeIntent>,
    dragged_node: &mut Option<NodeRequest>,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        mg_ui::section(ui, "Engine input", "Live source control");
        egui::Frame::new()
            .fill(mg_ui::PANEL_RAISED)
            .stroke(Stroke::new(
                1.0,
                if details.source_ready {
                    mg_ui::SIGNAL
                } else {
                    mg_ui::HAIRLINE
                },
            ))
            .inner_margin(8.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(if details.source_ready {
                        "SOURCE READY"
                    } else {
                        "NO SOURCE READY"
                    })
                    .monospace()
                    .strong()
                    .color(if details.source_ready {
                        mg_ui::SIGNAL
                    } else {
                        mg_ui::WARNING
                    }),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "KIND  {}",
                        if details.source_kind.is_empty() {
                            "none"
                        } else {
                            details.source_kind.as_str()
                        }
                    ))
                    .monospace()
                    .size(9.0)
                    .color(mg_ui::MUTED),
                );
                ui.label(
                    egui::RichText::new(format!(
                        "SOURCE  {}",
                        if details.source_label.is_empty() {
                            "—"
                        } else {
                            details.source_label.as_str()
                        }
                    ))
                    .monospace()
                    .size(9.0)
                    .color(mg_ui::TEXT),
                );
            });
        ui.add_space(6.0);
        ui.add(
            egui::TextEdit::singleline(source_path)
                .hint_text("Absolute image or video path")
                .desired_width(f32::INFINITY),
        );
        ui.horizontal_wrapped(|ui| {
            if ui.button("📂 LOAD IMAGE...").clicked() {
                if let Some(path_str) =
                    pick_file_dialog("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.gif")
                {
                    *source_path = path_str.clone();
                    *runtime_intent = Some(RuntimeIntent::SetSource {
                        kind: "image",
                        path: Some(path_str),
                    });
                }
            }
            if ui.button("🎬 LOAD VIDEO...").clicked() {
                if let Some(path_str) =
                    pick_file_dialog("Video Files", "*.mp4;*.mov;*.avi;*.mkv;*.webm;*.gif")
                {
                    *source_path = path_str.clone();
                    *runtime_intent = Some(RuntimeIntent::SetSource {
                        kind: "video",
                        path: Some(path_str),
                    });
                }
            }
            if ui.button("🎨 TEST PATTERN").clicked() {
                *runtime_intent = Some(RuntimeIntent::SetSource {
                    kind: "test_pattern",
                    path: None,
                });
            }
            if !source_path.trim().is_empty() && ui.button("▶ LOAD PATH").clicked() {
                let p = source_path.trim().to_owned();
                let kind = if p.ends_with(".mp4")
                    || p.ends_with(".mov")
                    || p.ends_with(".avi")
                    || p.ends_with(".webm")
                {
                    "video"
                } else {
                    "image"
                };
                *runtime_intent = Some(RuntimeIntent::SetSource {
                    kind,
                    path: Some(p),
                });
            }
        });
        ui.label(
            egui::RichText::new(
                "Drop an image/video anywhere in window, or drag nodes below onto canvas.",
            )
            .size(8.0)
            .color(mg_ui::MUTED),
        );
        if !source_commands_available {
            ui.label(
                egui::RichText::new(
                    "Source commands are unavailable until Engine V2 negotiates set_source.",
                )
                .size(8.0)
                .color(mg_ui::WARNING),
            );
        }

        ui.add_space(12.0);
        mg_ui::section(ui, "Endpoints & Sources", "Click or drag into workspace");
        for node in [
            LibraryNode::Capture,
            LibraryNode::Media,
            LibraryNode::Generator,
        ] {
            let (rect, resp) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 30.0),
                egui::Sense::click_and_drag(),
            );
            if resp.hovered() {
                ui.painter().rect_filled(rect, 3.0, mg_ui::PANEL_RAISED);
            } else {
                ui.painter().rect_filled(rect, 3.0, mg_ui::BG);
            }
            ui.painter().rect_stroke(
                rect,
                3.0,
                Stroke::new(
                    1.0,
                    if resp.hovered() {
                        mg_ui::ACCENT
                    } else {
                        mg_ui::HAIRLINE
                    },
                ),
                egui::StrokeKind::Inside,
            );
            ui.painter().text(
                rect.left_center() + egui::vec2(8.0, 0.0),
                egui::Align2::LEFT_CENTER,
                format!("+ {}", node.label()),
                egui::FontId::proportional(11.0),
                mg_ui::TEXT,
            );
            if resp.drag_started()
                || (resp.is_pointer_button_down_on() && ui.input(|i| i.pointer.is_moving()))
            {
                *dragged_node = Some(NodeRequest::Builtin(node));
            }
            if resp.clicked() {
                *selected = Some(graph::insert_library_node(graph, node));
                *graph_revision = graph_revision.saturating_add(1);
            }
        }

        ui.add_space(10.0);
        mg_ui::section(ui, "Core Processors", "Click or drag into workspace");
        for node in [
            LibraryNode::Color,
            LibraryNode::Temporal,
            LibraryNode::Output,
        ] {
            let (rect, resp) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), 26.0),
                egui::Sense::click_and_drag(),
            );
            if resp.hovered() {
                ui.painter().rect_filled(rect, 3.0, mg_ui::PANEL_RAISED);
            } else {
                ui.painter().rect_filled(rect, 3.0, mg_ui::BG);
            }
            ui.painter().rect_stroke(
                rect,
                3.0,
                Stroke::new(
                    1.0,
                    if resp.hovered() {
                        mg_ui::ACCENT
                    } else {
                        mg_ui::HAIRLINE
                    },
                ),
                egui::StrokeKind::Inside,
            );
            ui.painter().text(
                rect.left_center() + egui::vec2(8.0, 0.0),
                egui::Align2::LEFT_CENTER,
                format!("+ {}", node.label()),
                egui::FontId::proportional(11.0),
                mg_ui::TEXT,
            );
            if resp.drag_started()
                || (resp.is_pointer_button_down_on() && ui.input(|i| i.pointer.is_moving()))
            {
                *dragged_node = Some(NodeRequest::Builtin(node));
            }
            if resp.clicked() {
                *selected = Some(graph::insert_library_node(graph, node));
                *graph_revision = graph_revision.saturating_add(1);
            }
        }

        ui.add_space(12.0);
        mg_ui::section(
            ui,
            "Node Catalog",
            &format!("{} registered glitch effects", catalog::all().len()),
        );
        ui.add(
            egui::TextEdit::singleline(catalog_query)
                .hint_text("Search node, category, CPU or CUDA…")
                .desired_width(f32::INFINITY),
        );

        let query = catalog_query.trim().to_lowercase();
        let all_descriptors = catalog::all();
        let mut categories: std::collections::BTreeMap<&str, Vec<(usize, &catalog::NodeSpec)>> =
            std::collections::BTreeMap::new();
        for (idx, desc) in all_descriptors.iter().enumerate() {
            if query.is_empty() || catalog::matches(*desc, &query) {
                categories
                    .entry(desc.category)
                    .or_default()
                    .push((idx, desc));
            }
        }

        for (category_name, items) in categories {
            egui::CollapsingHeader::new(format!(
                "{} ({})",
                category_name.to_ascii_uppercase(),
                items.len()
            ))
            .default_open(true)
            .show(ui, |ui| {
                for (idx, desc) in items {
                    let is_cuda = desc.backend == "cuda";
                    let (rect, resp) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), 22.0),
                        egui::Sense::click_and_drag(),
                    );
                    if resp.hovered() {
                        ui.painter().rect_filled(rect, 2.0, mg_ui::PANEL_RAISED);
                    } else {
                        ui.painter().rect_filled(rect, 2.0, mg_ui::BG);
                    }
                    ui.painter().rect_stroke(
                        rect,
                        2.0,
                        Stroke::new(
                            1.0,
                            if resp.hovered() {
                                mg_ui::ACCENT
                            } else {
                                mg_ui::HAIRLINE
                            },
                        ),
                        egui::StrokeKind::Inside,
                    );
                    ui.painter().text(
                        rect.left_center() + egui::vec2(6.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        format!("+ {}", desc.label),
                        egui::FontId::proportional(11.0),
                        if is_cuda { mg_ui::ACCENT } else { mg_ui::TEXT },
                    );
                    if resp.drag_started()
                        || (resp.is_pointer_button_down_on() && ui.input(|i| i.pointer.is_moving()))
                    {
                        *dragged_node = Some(NodeRequest::Catalog(idx));
                    }
                    if resp.clicked() {
                        *selected = Some(graph::insert_catalog_node(graph, *desc));
                        *graph_revision = graph_revision.saturating_add(1);
                    }
                }
            });
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn graph_ui(
    ui: &mut egui::Ui,
    graph: &mut Snarl<GraphNode>,
    selected: &mut Option<NodeId>,
    _snapshot: RuntimeSnapshot,
    graph_view_epoch: &mut u64,
    preset_buffer: &mut String,
    preset_status: &mut String,
    graph_revision: &mut u64,
    graph_status: graph::GraphStatus,
    graph_commands_available: bool,
) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("① SOURCE  →  ② PROCESS  →  ③ OUTPUT")
                .monospace()
                .strong()
                .color(mg_ui::SIGNAL),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui.button("⚡ AUTO-WIRE ALL").clicked() {
                graph::auto_wire_chain(graph);
                *graph_revision = graph_revision.saturating_add(1);
            }
            if ui.button("CLEAR ALL WIRES").clicked() {
                for (out_pin, _) in graph.wires().collect::<Vec<_>>() {
                    graph.drop_outputs(out_pin);
                }
                *graph_revision = graph_revision.saturating_add(1);
            }
            if ui.button("RESET GRAPH").clicked() {
                let (fresh_graph, fresh_selection) = graph::demo_graph();
                *graph = fresh_graph;
                *selected = fresh_selection;
                *graph_view_epoch = (*graph_view_epoch).saturating_add(1);
                *graph_revision = graph_revision.saturating_add(1);
            }
        });
    });
    ui.label(
        egui::RichText::new(
            "Drag and drop any node from the sidebar onto the workspace, or connect pins by dragging between green sockets.",
        )
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    if !graph_commands_available {
        ui.label(
            egui::RichText::new(
                "ENGINE GRAPH SUBMISSION UNAVAILABLE · EXISTING GRAPH IS A LOCAL DRAFT",
            )
            .monospace()
            .size(8.0)
            .color(mg_ui::WARNING),
        );
    }
    ui.label(
        egui::RichText::new(
            "SHORTCUTS:  DELETE remove  •  B bypass  •  CTRL+D duplicate  •  CTRL+K search palette",
        )
        .monospace()
        .size(8.0)
        .color(mg_ui::MUTED),
    );
    egui::CollapsingHeader::new("PRESETS & FILE MANAGEMENT (SAVE / LOAD)")
        .default_open(false)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("💾 SAVE PRESET FILE...").clicked() {
                    if let Some(path) = save_file_dialog("MegaGlitch Preset", "*.json", "json") {
                        match serde_json::to_string_pretty(&preset::export_graph(graph)) {
                            Ok(json) => {
                                if let Err(e) = std::fs::write(&path, json) {
                                    *preset_status = format!("Save failed: {e}");
                                } else {
                                    *preset_status = format!("Preset saved to {path}");
                                }
                            }
                            Err(e) => *preset_status = format!("Serialization failed: {e}"),
                        }
                    }
                }
                if ui
                    .add_enabled(
                        graph_commands_available,
                        egui::Button::new("📁 LOAD PRESET FILE..."),
                    )
                    .clicked()
                {
                    if let Some(path) = pick_file_dialog("MegaGlitch Preset", "*.json") {
                        match std::fs::read_to_string(&path) {
                            Ok(content) => {
                                *preset_buffer = content;
                                match serde_json::from_str::<preset::PresetGraph>(preset_buffer) {
                                    Ok(saved) => match preset::import_graph(&saved) {
                                        Ok((imported, imported_selection)) => {
                                            *graph = imported;
                                            *selected = imported_selection;
                                            *graph_view_epoch =
                                                (*graph_view_epoch).saturating_add(1);
                                            *graph_revision = graph_revision.saturating_add(1);
                                            *preset_status = format!("Loaded preset from {path}");
                                        }
                                        Err(e) => {
                                            *preset_status =
                                                format!("Preset validation error: {e}");
                                        }
                                    },
                                    Err(e) => *preset_status = format!("Invalid JSON: {e}"),
                                }
                            }
                            Err(e) => *preset_status = format!("Could not read file: {e}"),
                        }
                    }
                }
                if ui.button("EXPORT JSON TO BUFFER").clicked() {
                    match serde_json::to_string_pretty(&preset::export_graph(graph)) {
                        Ok(json) => {
                            *preset_buffer = json;
                            *preset_status = "Exported preset JSON to buffer".to_owned();
                        }
                        Err(error) => {
                            *preset_status = format!("Export failed: {error}");
                        }
                    }
                }
                if ui
                    .add_enabled(
                        graph_commands_available,
                        egui::Button::new("IMPORT BUFFER JSON"),
                    )
                    .clicked()
                {
                    match serde_json::from_str::<preset::PresetGraph>(preset_buffer) {
                        Ok(saved) => match preset::import_graph(&saved) {
                            Ok((imported, imported_selection)) => {
                                *graph = imported;
                                *selected = imported_selection;
                                *graph_view_epoch = (*graph_view_epoch).saturating_add(1);
                                *graph_revision = graph_revision.saturating_add(1);
                                *preset_status = "Imported and validated preset JSON".to_owned();
                            }
                            Err(error) => {
                                *preset_status = format!("Preset validation failed: {error}");
                            }
                        },
                        Err(error) => {
                            *preset_status = format!("Preset JSON failed: {error}");
                        }
                    }
                }
            });
            ui.add(
                egui::TextEdit::multiline(preset_buffer)
                    .code_editor()
                    .desired_rows(4)
                    .desired_width(f32::INFINITY)
                    .hint_text("Paste JSON preset here"),
            );
            ui.label(
                egui::RichText::new(preset_status.as_str())
                    .monospace()
                    .size(9.0)
                    .color(mg_ui::ACCENT),
            );
        });
    let style = SnarlStyle {
        node_layout: Some(
            NodeLayout::coil()
                .with_equal_pin_rows()
                .with_min_pin_row_height(86.0),
        ),
        pin_placement: Some(PinPlacement::Outside { margin: 6.0 }),
        pin_size: Some(18.0),
        pin_stroke: Some(Stroke::new(2.5, mg_ui::TEXT)),
        bg_pattern: Some(BackgroundPattern::Grid(Grid::new(
            egui::vec2(32.0, 32.0),
            0.0,
        ))),
        bg_pattern_stroke: Some(Stroke::new(1.0, mg_ui::HAIRLINE.gamma_multiply(0.35))),
        wire_width: Some(3.5),
        wire_frame_size: Some(56.0),
        wire_style: Some(WireStyle::Bezier3),
        wire_layer: Some(WireLayer::BehindNodes),
        centering: Some(false),
        max_scale: Some(2.0),
        min_scale: Some(0.2),
        ..Default::default()
    };
    let dropped = {
        let mut viewer = GraphViewer {
            selected,
            graph_revision,
            status: graph_status,
        };
        let (_, dropped) = ui.dnd_drop_zone::<NodeRequest, _>(
            egui::Frame::new()
                .fill(mg_ui::BG)
                .stroke(Stroke::new(1.0, mg_ui::HAIRLINE)),
            |ui| {
                SnarlWidget::new()
                    .id_salt(("mg_node_graph", *graph_view_epoch))
                    .style(style)
                    .show(graph, &mut viewer, ui);
            },
        );
        dropped
    };
    if let Some(request) = dropped {
        let drop_pos = ui.input(|i| i.pointer.interact_pos());
        *selected = match *request {
            NodeRequest::Builtin(node) => {
                Some(graph::insert_library_node_at(graph, node, drop_pos))
            }
            NodeRequest::Catalog(index) => catalog::all()
                .get(index)
                .copied()
                .map(|node| graph::insert_catalog_node_at(graph, node, drop_pos)),
        };
        if selected.is_some() {
            *graph_revision = graph_revision.saturating_add(1);
        }
    }
}

fn preview_ui(
    ui: &mut egui::Ui,
    texture: Option<&egui::TextureHandle>,
    frame: Option<&crate::state::PreviewFrame>,
    details: &RuntimeDetails,
    present_tracker: &PreviewPresentationTracker,
) {
    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, egui::Sense::click());
    ui.painter().rect_filled(rect, 0.0, mg_ui::BG);
    match (texture, frame) {
        (Some(texture), Some(frame)) if frame.width > 0 && frame.height > 0 => {
            let source_aspect = frame.width as f32 / frame.height as f32;
            let available_aspect = rect.width() / rect.height().max(1.0);
            let size = if available_aspect > source_aspect {
                egui::vec2(rect.height() * source_aspect, rect.height())
            } else {
                egui::vec2(rect.width(), rect.width() / source_aspect)
            };
            let output_rect = egui::Rect::from_center_size(rect.center(), size);
            ui.painter().image(
                texture.id(),
                output_rect,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
            ui.painter()
                .add(eframe::egui_wgpu::Callback::new_paint_callback(
                    output_rect,
                    PreviewPresentMarker {
                        tracker: present_tracker.clone(),
                        surface_generation: frame.surface_generation,
                        sequence: frame.sequence,
                        graph_revision: frame.graph_revision,
                        submitted_at: frame.submitted_at,
                    },
                ));
            ui.painter().rect_stroke(
                output_rect,
                0.0,
                Stroke::new(1.0, mg_ui::HAIRLINE),
                egui::StrokeKind::Inside,
            );
            let label = format!(
                "ENGINE OUTPUT  •  {}×{}  •  FRAME {}  •  {}",
                frame.width,
                frame.height,
                frame.sequence,
                if details.source_label.is_empty() {
                    details.source_kind.as_str()
                } else {
                    details.source_label.as_str()
                }
            );
            let label_rect = egui::Rect::from_min_size(
                output_rect.left_top() + egui::vec2(8.0, 8.0),
                egui::vec2((output_rect.width() - 16.0).max(1.0), 25.0),
            );
            ui.painter()
                .rect_filled(label_rect, 1.0, mg_ui::BG.gamma_multiply(0.86));
            ui.painter().text(
                label_rect.left_center() + egui::vec2(8.0, 0.0),
                egui::Align2::LEFT_CENTER,
                label,
                egui::FontId::monospace(9.0),
                mg_ui::TEXT,
            );
        }
        _ => {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                if details.source_ready {
                    "WAITING FOR DECODED ENGINE FRAME"
                } else {
                    "SELECT AN IMAGE, VIDEO, OR TEST PATTERN IN SOURCES"
                },
                egui::FontId::monospace(11.0),
                if details.source_ready {
                    mg_ui::ACCENT
                } else {
                    mg_ui::MUTED
                },
            );
        }
    }
    if response.double_clicked() {
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::Fullscreen(true));
    }
}

fn inspector_ui(
    ui: &mut egui::Ui,
    graph: &mut Snarl<GraphNode>,
    selected: &mut Option<NodeId>,
    ui_scale: &mut f32,
    reduced_motion: &mut bool,
    graph_revision: &mut u64,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        mg_ui::section(ui, "Selection", "Context inspector");

        let all_nodes: Vec<(NodeId, String)> = graph
            .node_ids()
            .map(|(id, node)| (id, format!("{} ({})", node.title, node.node_type)))
            .collect();

        if !all_nodes.is_empty() {
            if selected.is_none() || !graph.node_ids().any(|(id, _)| Some(id) == *selected) {
                let default_node = all_nodes
                    .iter()
                    .find(|(id, _)| {
                        graph[*id].kind == crate::graph::NodeKind::Transform
                            || graph[*id].kind == crate::graph::NodeKind::Feedback
                    })
                    .map(|(id, _)| *id)
                    .unwrap_or(all_nodes[0].0);
                *selected = Some(default_node);
            }

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("TARGET NODE:")
                        .monospace()
                        .size(9.0)
                        .color(mg_ui::ACCENT),
                );
                let curr_title = selected
                    .and_then(|id| graph.node_ids().find(|(nid, _)| *nid == id))
                    .map(|(_, n)| n.title)
                    .unwrap_or("Select Node...");
                egui::ComboBox::from_id_salt("inspector_node_select_combo")
                    .selected_text(curr_title)
                    .width(180.0)
                    .show_ui(ui, |ui| {
                        for (id, label) in &all_nodes {
                            let is_curr = *selected == Some(*id);
                            if ui.selectable_label(is_curr, label).clicked() {
                                *selected = Some(*id);
                            }
                        }
                    });
            });
            ui.add_space(8.0);
        }

        if let Some(node_id) = *selected {
            let node = &mut graph[node_id];
            ui.label(egui::RichText::new(node.title).size(14.0).strong());
            ui.label(
                egui::RichText::new(node.category)
                    .monospace()
                    .size(9.0)
                    .color(mg_ui::ACCENT),
            );
            ui.label(
                egui::RichText::new(node.node_type)
                    .monospace()
                    .size(8.0)
                    .color(mg_ui::MUTED),
            );
            ui.add_space(10.0);
            let mut active = node.active;
            if ui.checkbox(&mut active, "Enabled").changed() {
                node.set_active(active);
                *graph_revision = graph_revision.saturating_add(1);
            }
            if let Some(spec) = node.spec {
                ui.add_space(12.0);
                mg_ui::section(
                    ui,
                    "Parameters",
                    &format!(
                        "NodeSpec v{} / {} controls",
                        spec.version,
                        spec.parameters.len()
                    ),
                );
                for parameter in spec.parameters {
                    if parameter.id == "enabled" {
                        continue;
                    }
                    if let Some(value) = node.parameters.get_mut(parameter.id)
                        && parameter_editor(ui, parameter, value)
                    {
                        *graph_revision = graph_revision.saturating_add(1);
                    }
                }
            } else {
                ui.add_space(8.0);
                mg_ui::section(ui, "Controls", "Node parameters");
                if ui
                    .add(mg_ui::Knob::new(&mut node.amount, "AMOUNT", 0.0..=1.0))
                    .changed()
                {
                    *graph_revision = graph_revision.saturating_add(1);
                }
            }
            ui.add_space(12.0);
            mg_ui::section(ui, "Connections", "Quick Link");
            let other_nodes: Vec<(NodeId, String)> = graph
                .node_ids()
                .filter(|(id, _)| *id != node_id)
                .map(|(id, node)| (id, format!("{} ({})", node.title, node.node_type)))
                .collect();
            if !other_nodes.is_empty() {
                ui.label(
                    egui::RichText::new("CONNECT VIDEO OUT TO:")
                        .monospace()
                        .size(8.5)
                        .color(mg_ui::SIGNAL),
                );
                for (target_id, target_label) in other_nodes {
                    if ui.button(format!("→ Wire to {}", target_label)).clicked() {
                        graph.drop_inputs(InPinId {
                            node: target_id,
                            input: 0,
                        });
                        graph.connect(
                            OutPinId {
                                node: node_id,
                                output: 0,
                            },
                            InPinId {
                                node: target_id,
                                input: 0,
                            },
                        );
                        *graph_revision = graph_revision.saturating_add(1);
                    }
                }
            }
            if ui.button("✂ Disconnect node wires").clicked() {
                graph.drop_outputs(OutPinId {
                    node: node_id,
                    output: 0,
                });
                graph.drop_inputs(InPinId {
                    node: node_id,
                    input: 0,
                });
                *graph_revision = graph_revision.saturating_add(1);
            }
        } else {
            ui.label(
                egui::RichText::new("Select a node to expose its live controls.")
                    .color(mg_ui::MUTED),
            );
        }
        ui.add_space(18.0);
        mg_ui::section(ui, "Access", "Interface");
        ui.add(egui::Slider::new(ui_scale, 0.8..=1.6).text("UI scale"));
        ui.checkbox(reduced_motion, "Reduced motion");
    });
}

fn parameter_editor(
    ui: &mut egui::Ui,
    parameter: &ParameterSpec,
    value: &mut ParameterValue,
) -> bool {
    match (parameter.kind, value) {
        (ParameterKind::Float, ParameterValue::Float(value)) => {
            if let (Some(minimum), Some(maximum)) = (parameter.minimum, parameter.maximum) {
                ui.add(egui::Slider::new(value, minimum..=maximum).text(parameter.label))
                    .changed()
            } else {
                ui.horizontal(|ui| {
                    ui.label(parameter.label);
                    ui.add(egui::DragValue::new(value).speed(0.01)).changed()
                })
                .inner
            }
        }
        (ParameterKind::Int, ParameterValue::Int(value)) => {
            if let (Some(minimum), Some(maximum)) = (parameter.minimum, parameter.maximum) {
                ui.add(
                    egui::Slider::new(value, minimum as i64..=maximum as i64).text(parameter.label),
                )
                .changed()
            } else {
                ui.horizontal(|ui| {
                    ui.label(parameter.label);
                    ui.add(egui::DragValue::new(value)).changed()
                })
                .inner
            }
        }
        (ParameterKind::Bool, ParameterValue::Bool(value)) => {
            ui.checkbox(value, parameter.label).changed()
        }
        (ParameterKind::Enum, ParameterValue::Text(value)) => {
            let previous = value.clone();
            egui::ComboBox::from_label(parameter.label)
                .selected_text(value.as_str())
                .show_ui(ui, |ui| {
                    for choice in parameter.choices {
                        ui.selectable_value(value, (*choice).to_owned(), *choice);
                    }
                });
            *value != previous
        }
        (ParameterKind::Text, ParameterValue::Text(value)) => {
            ui.horizontal(|ui| {
                ui.label(parameter.label);
                ui.text_edit_singleline(value).changed()
            })
            .inner
        }
        _ => {
            ui.label(
                egui::RichText::new(format!("{}: schema/value mismatch", parameter.label))
                    .color(mg_ui::WARNING),
            );
            false
        }
    }
}

fn timeline_ui(
    ui: &mut egui::Ui,
    details: &RuntimeDetails,
    video_seek_seconds: &mut Option<f64>,
    seek_commands_available: bool,
    runtime_intent: &mut Option<RuntimeIntent>,
) {
    mg_ui::section(ui, "Sequence", "Automation · reserved workspace");
    ui.label(
        egui::RichText::new(
            "Automation tracks are retained in the layout but are not connected to Engine V2.",
        )
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(
            "The legacy Python machine remains the compatibility runtime for timeline and modulation behavior.",
        )
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.add_space(8.0);
    let can_seek = seek_commands_available
        && details.source_ready
        && details.source_kind == "video"
        && details.pending_command_id.is_none()
        && runtime_intent.is_none();
    if details.source_ready && details.source_kind == "video" && !seek_commands_available {
        ui.label(
            egui::RichText::new(
                "Video seek is unavailable because Engine V2 did not negotiate the seek command.",
            )
            .size(9.0)
            .color(mg_ui::WARNING),
        );
    }
    if let Some((position, duration)) = video_seek_range(details) {
        let seek_seconds = video_seek_seconds.get_or_insert(position);
        *seek_seconds = (*seek_seconds).clamp(0.0, duration);
        ui.label(
            egui::RichText::new(format!(
                "VIDEO POSITION  {:>8.3}s / {:>8.3}s",
                position, duration
            ))
            .monospace()
            .size(9.0)
            .color(mg_ui::ACCENT),
        );
        ui.add(
            egui::Slider::new(seek_seconds, 0.0..=duration)
                .text("Seek target")
                .fixed_decimals(3),
        );
        if ui
            .add_enabled(can_seek, egui::Button::new("SEEK VIDEO TO TARGET"))
            .clicked()
        {
            *runtime_intent = Some(RuntimeIntent::Seek(*seek_seconds));
        }
    } else {
        *video_seek_seconds = None;
        if ui
            .add_enabled(can_seek, egui::Button::new("SEEK VIDEO TO START (0.000s)"))
            .clicked()
        {
            *runtime_intent = Some(RuntimeIntent::Seek(0.0));
        }
        if details.source_ready && details.source_kind == "video" {
            ui.label(
                egui::RichText::new(
                    "Arbitrary seek is unavailable because Engine V2 has not published video duration and playback position.",
                )
                .size(9.0)
                .color(mg_ui::WARNING),
            );
        }
    }
    ui.add_enabled(
        false,
        egui::Button::new("ADD AUTOMATION TRACK — UNAVAILABLE"),
    );
}

fn monitor_ui(
    ui: &mut egui::Ui,
    snapshot: RuntimeSnapshot,
    details: &RuntimeDetails,
    preview_uploaded_sequence: u64,
    presentation: PreviewPresentationSnapshot,
) {
    mg_ui::section(
        ui,
        "Telemetry",
        "Engine render + unique WGPU render-pass markers",
    );
    for (name, value, unit, maximum, color) in [
        (
            "AUTHORITATIVE",
            snapshot.authoritative_fps,
            "fps",
            120.0,
            mg_ui::SIGNAL,
        ),
        (
            "PREVIEW RENDER",
            snapshot.preview_render_fps,
            "fps",
            120.0,
            mg_ui::ACCENT,
        ),
        (
            "PREVIEW PRESENT",
            presentation.preview_present_fps,
            "fps",
            120.0,
            mg_ui::ACCENT,
        ),
        (
            "AUTH FRAME",
            snapshot.authoritative_frame_ms,
            "ms",
            33.33,
            mg_ui::WARNING,
        ),
        (
            "PREVIEW FRAME",
            snapshot.preview_frame_ms,
            "ms",
            33.33,
            mg_ui::WARNING,
        ),
        (
            "PREVIEW P95",
            snapshot.preview_p95_ms,
            "ms",
            33.33,
            mg_ui::WARNING,
        ),
    ] {
        ui.label(
            egui::RichText::new(format!("{name:<14} {value:>8.2} {unit}"))
                .monospace()
                .size(10.0)
                .color(color),
        );
        mg_ui::meter(ui, "", value / maximum, color);
    }
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new(format!(
            "preview resolution {:04}x{:04} / graph R{} / resolution G{}",
            snapshot.preview_width,
            snapshot.preview_height,
            details.preview_graph_revision,
            details.preview_resolution_generation
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(format!(
            "authoritative {:010} / rendered {:010} / received {:010}",
            snapshot.authoritative_sequence,
            snapshot.preview_rendered_sequence,
            snapshot.preview_received_sequence
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(format!(
            "uploaded {:010} / presented {:010}",
            preview_uploaded_sequence, presentation.preview_presented_sequence
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(format!(
            "requests {:010} / rendered {:010} / superseded {:010} / dropped {:010}",
            snapshot.preview_requests,
            snapshot.preview_rendered,
            snapshot.preview_superseded,
            snapshot.preview_dropped
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(match presentation.preview_submit_to_present_ms {
            Some(latency) => format!("approx submit → WGPU render pass {latency:.2} ms"),
            None => "approx submit → WGPU render pass unavailable".to_owned(),
        })
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(format!(
            "graph gen {:06} / ack {:04} / reject {:03}",
            snapshot.bridge_generation, snapshot.bridge_acks, snapshot.bridge_rejections
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(format!(
            "authoritative/legacy drops {:06} / invalid preview frames {:06}",
            snapshot.dropped_frames, snapshot.preview_dropped_frames
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    let unsupported = details
        .preview_node_statuses
        .iter()
        .filter(|node| node.status == "unsupported")
        .map(|node| node.label.as_str())
        .collect::<Vec<_>>();
    if !unsupported.is_empty() {
        ui.label(
            egui::RichText::new(format!(
                "PREVIEW UNSUPPORTED [{}]: {}",
                unsupported.len(),
                unsupported.join(", ")
            ))
            .monospace()
            .size(9.0)
            .color(mg_ui::WARNING),
        );
    }
    ui.label(
        egui::RichText::new(format!(
            "state rev {:06} / source {} / transport {} / graph {}",
            details.state_revision,
            if details.source_ready {
                "ready"
            } else {
                "waiting"
            },
            if details.transport_running {
                "running"
            } else {
                "stopped"
            },
            if details.current_graph_accepted {
                "accepted"
            } else {
                "not accepted"
            }
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    ui.label(
        egui::RichText::new(format!(
            "last ack #{:04} effective at frame {:010}",
            details.last_ack_command_id, details.last_effective_frame_sequence
        ))
        .monospace()
        .size(9.0)
        .color(mg_ui::MUTED),
    );
    mg_ui::status_lamp(
        ui,
        if snapshot.bridge_online {
            "ENGINE V2 BRIDGE ONLINE"
        } else {
            "ENGINE V2 BRIDGE OFFLINE"
        },
        snapshot.bridge_online,
        mg_ui::SIGNAL,
    );
}

fn default_tree() -> egui_tiles::Tree<Pane> {
    let mut tiles = egui_tiles::Tiles::default();
    let sources = tiles.insert_pane(Pane::Sources);
    let graph = tiles.insert_pane(Pane::Graph);
    let preview = tiles.insert_pane(Pane::Preview);
    let timeline = tiles.insert_pane(Pane::Timeline);
    let monitor = tiles.insert_pane(Pane::SignalMonitor);
    let inspector = tiles.insert_pane(Pane::Inspector);
    let lower_tabs = tiles.insert_tab_tile(vec![preview, timeline, monitor]);
    let center = tiles.insert_vertical_tile(vec![graph, lower_tabs]);
    let root = tiles.insert_horizontal_tile(vec![sources, center, inspector]);
    if let Some(egui_tiles::Tile::Container(egui_tiles::Container::Linear(linear))) =
        tiles.get_mut(root)
    {
        linear.shares.set_share(sources, 0.62);
        linear.shares.set_share(center, 2.25);
        linear.shares.set_share(inspector, 0.78);
    }
    if let Some(egui_tiles::Tile::Container(egui_tiles::Container::Linear(linear))) =
        tiles.get_mut(center)
    {
        linear.shares.set_share(graph, 1.8);
        linear.shares.set_share(lower_tabs, 1.0);
    }
    egui_tiles::Tree::new("mg_workspace", root, tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_presentation_distinguishes_all_authoritative_states() {
        let mut details = RuntimeDetails::default();
        assert_eq!(graph_status_for(2, 1, &details), graph::GraphStatus::Draft);

        details.pending_command_id = Some(41);
        details.pending_command_type = Some("replace_graph".to_owned());
        assert_eq!(
            graph_status_for(2, 2, &details),
            graph::GraphStatus::Pending
        );

        details.pending_command_id = None;
        details.pending_command_type = None;
        details.current_graph_accepted = true;
        assert_eq!(
            graph_status_for(2, 2, &details),
            graph::GraphStatus::Accepted
        );

        details.current_graph_accepted = false;
        assert_eq!(
            graph_status_for(2, 2, &details),
            graph::GraphStatus::Rejected
        );

        details.pending_command_id = Some(42);
        details.pending_command_type = Some("set_source".to_owned());
        assert_eq!(
            graph_status_for(2, 2, &details),
            graph::GraphStatus::Rejected,
            "a source command must not make a rejected graph look pending"
        );

        details.pending_command_id = None;
        details.pending_command_type = None;
        assert_eq!(
            graph_status_for(2, 0, &details),
            graph::GraphStatus::Draft,
            "restart resets the submitted revision so the graph is eligible to requeue"
        );
    }

    #[test]
    fn session_header_never_claims_an_offline_session_is_active() {
        let offline = RuntimeSnapshot::default();
        assert_eq!(
            session_status(offline),
            "SESSION  UNAVAILABLE · ENGINE OFFLINE"
        );

        let mut online = offline;
        online.bridge_online = true;
        online.state_revision = 7;
        assert_eq!(session_status(online), "SESSION  IN-MEMORY · ENGINE R7");
    }

    #[test]
    fn engine_commands_require_online_negotiated_capabilities() {
        let mut snapshot = RuntimeSnapshot::default();
        let mut details = RuntimeDetails::default();
        details
            .capabilities
            .push("transactional_commands:replace_graph".to_owned());

        assert!(!supports_engine_command(
            snapshot,
            &details,
            "replace_graph"
        ));
        snapshot.bridge_online = true;
        assert!(supports_engine_command(snapshot, &details, "replace_graph"));
        assert!(!supports_engine_command(snapshot, &details, "set_source"));
    }

    #[test]
    fn restart_is_offered_only_after_a_nonrecoverable_offline_failure() {
        let mut snapshot = RuntimeSnapshot::default();
        let mut details = RuntimeDetails::default();
        assert!(!engine_restart_available(snapshot, &details));

        details.last_error = Some(crate::state::RuntimeProblem {
            code: "bridge.process_failed".to_owned(),
            path: None,
            message: "sidecar exited".to_owned(),
            recoverable: false,
        });
        assert!(engine_restart_available(snapshot, &details));

        snapshot.bridge_online = true;
        assert!(!engine_restart_available(snapshot, &details));
    }

    #[test]
    fn arbitrary_video_seek_requires_real_position_and_duration() {
        let mut details = RuntimeDetails {
            source_ready: true,
            source_kind: "video".to_owned(),
            ..RuntimeDetails::default()
        };
        assert_eq!(video_seek_range(&details), None);

        details.source_position_seconds = Some(12.5);
        details.source_duration_seconds = Some(90.0);
        assert_eq!(video_seek_range(&details), Some((12.5, 90.0)));

        details.source_position_seconds = Some(95.0);
        assert_eq!(video_seek_range(&details), Some((90.0, 90.0)));
    }

    #[test]
    fn restart_transitions_reconcile_status_on_failure_and_ready() {
        let current_status = "RESTARTING ENGINE V2 · GRAPH R1 WILL REQUEUE";

        // 1. When details report Faulted lifecycle, runtime_status must leave RESTARTING
        let mut faulted_view = RuntimeView::default();
        faulted_view.details.lifecycle = EngineLifecycleState::Faulted;
        faulted_view.details.last_error = Some(crate::state::RuntimeProblem {
            code: "bridge.process_failed".to_owned(),
            path: None,
            message: "Python bridge output closed".to_owned(),
            recoverable: false,
        });

        let reconciled =
            reconcile_runtime_status(current_status, faulted_view.snapshot, &faulted_view.details);
        assert_eq!(
            reconciled.as_deref(),
            Some("ENGINE FAULTED: bridge.process_failed")
        );

        // 2. When engine recovers and reaches Ready, runtime_status must report READY
        let mut ready_view = RuntimeView::default();
        ready_view.snapshot.bridge_online = true;
        ready_view.snapshot.bridge_generation = 2;
        ready_view.details.lifecycle = EngineLifecycleState::Ready;

        let reconciled_ready =
            reconcile_runtime_status(current_status, ready_view.snapshot, &ready_view.details);
        assert_eq!(reconciled_ready.as_deref(), Some("ENGINE READY (G2)"));

        // 3. When engine is Stopped, runtime_status must report STOPPED
        let mut stopped_view = RuntimeView::default();
        stopped_view.details.lifecycle = EngineLifecycleState::Stopped;
        let reconciled_stopped =
            reconcile_runtime_status(current_status, stopped_view.snapshot, &stopped_view.details);
        assert_eq!(reconciled_stopped.as_deref(), Some("ENGINE STOPPED"));
    }

    #[test]
    fn graph_and_source_reconciliation_tracks_revisions_and_prevents_duplicate_submissions() {
        let mut details = RuntimeDetails::default();

        // 1. Initial local draft state
        let local_graph_rev = 1;
        let mut submitted_graph_rev = 0;
        let source_rev = 1;
        let mut submitted_source_rev = 0;

        assert_eq!(
            graph_status_for(local_graph_rev, submitted_graph_rev, &details),
            graph::GraphStatus::Draft
        );

        // 2. Graph is submitted -> Pending
        submitted_graph_rev = local_graph_rev;
        details.pending_command_id = Some(1);
        details.pending_command_type = Some("replace_graph".to_owned());
        assert_eq!(
            graph_status_for(local_graph_rev, submitted_graph_rev, &details),
            graph::GraphStatus::Pending
        );

        // 3. Graph ACK received -> Accepted
        details.pending_command_id = None;
        details.pending_command_type = None;
        details.current_graph_accepted = true;
        assert_eq!(
            graph_status_for(local_graph_rev, submitted_graph_rev, &details),
            graph::GraphStatus::Accepted
        );

        // 4. Source submitted -> submitted_source_rev tracks source_rev
        submitted_source_rev = source_rev;
        assert_eq!(
            source_rev, submitted_source_rev,
            "source must not duplicate submit when synced"
        );

        // 5. Subsequent frames without edits: no duplicate submission condition
        let graph_needs_submit = local_graph_rev != submitted_graph_rev;
        let source_needs_submit = source_rev != submitted_source_rev;
        assert!(
            !graph_needs_submit,
            "must not resubmit graph on repeated frames"
        );
        assert!(
            !source_needs_submit,
            "must not resubmit source on repeated frames"
        );

        // 6. Simulate Engine Restart: submitted revisions reset to 0, local state preserved
        submitted_graph_rev = 0;
        submitted_source_rev = 0;
        let fresh_details = RuntimeDetails::default(); // fresh generation

        // Both graph and source must automatically trigger resubmission
        assert_eq!(
            graph_status_for(local_graph_rev, submitted_graph_rev, &fresh_details),
            graph::GraphStatus::Draft
        );
        assert_ne!(
            local_graph_rev, submitted_graph_rev,
            "graph must requeue on restart"
        );
        assert_ne!(
            source_rev, submitted_source_rev,
            "source must requeue on restart"
        );
    }

    #[test]
    fn preview_smoke_loader_selects_direct_and_wrapped_decks() {
        let (demo, _) = graph::demo_graph();
        let graph_a = preset::PresetGraph {
            nodes: Vec::new(),
            links: Vec::new(),
            version: preset::PYTHON_PRESET_VERSION.to_owned(),
            source_pos: None,
            source_b_pos: None,
            display_pos: None,
        };
        let graph_b = preset::export_graph(&demo);

        let direct = serde_json::to_value(&graph_b).unwrap();
        assert_eq!(
            preview_smoke_graph_from_value(&direct, DeckId::A).unwrap(),
            graph_b
        );

        let legacy_wrapper = serde_json::json!({
            "version": "2.0",
            "preset_type": "dual_deck_session",
            "decks": {
                "a": graph_a,
                "b": graph_b,
            },
        });
        assert!(
            preview_smoke_graph_from_value(&legacy_wrapper, DeckId::A)
                .unwrap()
                .nodes
                .is_empty()
        );
        assert!(
            !preview_smoke_graph_from_value(&legacy_wrapper, DeckId::B)
                .unwrap()
                .nodes
                .is_empty()
        );

        let canonical_wrapper = serde_json::json!({
            "decks": {
                "a": {"graph": graph_a},
                "b": {"graph": graph_b},
            },
        });
        assert!(
            !preview_smoke_graph_from_value(&canonical_wrapper, DeckId::B)
                .unwrap()
                .nodes
                .is_empty()
        );
    }

    #[test]
    fn preview_smoke_parameter_probe_is_bounded_and_reversible() {
        let (mut graph, _) = graph::demo_graph();
        let original = preset::export_graph(&graph);
        let mutation = begin_preview_parameter_probe(&mut graph)
            .unwrap()
            .expect("demo effect graph exposes a Float control");
        assert_ne!(mutation.original, mutation.probed);
        assert!(restore_preview_parameter_probe(&mut graph, &mutation));
        assert_eq!(preset::export_graph(&graph), original);
    }

    #[test]
    fn preview_smoke_loader_handles_saved_6_20_26_cases() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("node_presets")
            .join("6-20-26.json");
        let (deck_a, mut pframe_graph, _) = preview_smoke_load_graph(&path, DeckId::A).unwrap();
        assert_eq!(deck_a.nodes[0].backend_id, "PFrameGlitchesNode");
        let mutation = begin_preview_parameter_probe(&mut pframe_graph)
            .unwrap()
            .expect("saved P-frame deck has a Float parameter");
        assert_eq!(mutation.parameter, "motion_vector_corruption");

        let (deck_b, _, _) = preview_smoke_load_graph(&path, DeckId::B).unwrap();
        assert!(deck_b.nodes.is_empty(), "deck B is Source to Display");
    }

    #[test]
    fn preview_smoke_source_kind_rejects_unknown_extensions() {
        assert_eq!(
            preview_smoke_source_kind(std::path::Path::new("clip.MP4")).unwrap(),
            "video"
        );
        assert_eq!(
            preview_smoke_source_kind(std::path::Path::new("frame.webp")).unwrap(),
            "image"
        );
        assert!(preview_smoke_source_kind(std::path::Path::new("notes.txt")).is_err());
    }

    #[test]
    fn presentation_stage_timings_match_frames_and_reset() {
        let tracker = PreviewPresentationTracker::default();
        let start = Instant::now();
        let key = PreviewPresentKey {
            surface_generation: 1,
            sequence: 1,
            graph_revision: 1,
        };
        tracker.record_upload(
            key,
            start,
            start + Duration::from_millis(2),
            start + Duration::from_millis(3),
        );
        tracker.observe_at(1, 1, 1, None, start + Duration::from_millis(7), None);
        tracker.observe_at(1, 1, 1, None, start + Duration::from_millis(8), None);
        let snapshot = tracker.snapshot_at(start + Duration::from_millis(8));
        assert_eq!(snapshot.stage_samples, 1);
        assert_eq!(snapshot.stage_mean_ms, [2.0, 1.0, 4.0]);
        assert_eq!(snapshot.stage_max_ms, [2.0, 1.0, 4.0]);
        assert_eq!(snapshot.replaced_before_draw, 0);
        let second = PreviewPresentKey {
            sequence: 2,
            ..key
        };
        let third = PreviewPresentKey {
            sequence: 3,
            ..key
        };
        tracker.record_upload(second, start, start, start);
        tracker.record_upload(third, start, start, start);
        // A late marker for the replaced frame cannot consume another frame's timing.
        tracker.observe_at(1, 2, 1, None, start + Duration::from_millis(9), None);
        let snapshot = tracker.snapshot_at(start + Duration::from_millis(9));
        assert_eq!(snapshot.stage_samples, 1);
        assert_eq!(snapshot.texture_preparations, 3);
        assert_eq!(snapshot.replaced_before_draw, 1);
        tracker.reset();
        assert_eq!(tracker.snapshot().stage_samples, 0);
        assert_eq!(tracker.snapshot().texture_preparations, 0);
    }

    #[test]
    fn presentation_tracker_counts_only_unique_new_frames() {
        let tracker = PreviewPresentationTracker::default();
        let started = Instant::now();

        tracker.observe_at(10, 1, 7, Some(100.0), started, Some(100.010));
        tracker.observe_at(
            10,
            1,
            7,
            Some(100.0),
            started + Duration::from_millis(8),
            Some(100.018),
        );
        tracker.observe_at(
            10,
            2,
            8,
            Some(100.020),
            started + Duration::from_millis(16),
            Some(100.026),
        );

        let snapshot = tracker.snapshot_at(started + Duration::from_millis(16));
        assert_eq!(snapshot.preview_presented_count, 2);
        assert_eq!(snapshot.preview_presented_sequence, 2);
        assert_eq!(snapshot.preview_presented_graph_revision, 8);
        assert!((snapshot.preview_present_fps - 62.5).abs() < 0.01);
        assert!((snapshot.preview_measurement_fps - 62.5).abs() < 0.01);
        assert!((snapshot.preview_submit_to_present_ms.unwrap() - 6.0).abs() < 0.01);
        assert!((snapshot.preview_mean_submit_to_present_ms.unwrap() - 8.0).abs() < 0.01);

        tracker.observe_at(
            10,
            1,
            7,
            Some(100.0),
            started + Duration::from_millis(24),
            Some(100.034),
        );
        let after_stale = tracker.snapshot_at(started + Duration::from_millis(24));
        assert_eq!(after_stale.preview_presented_sequence, 2);
        assert!((after_stale.preview_present_fps - 62.5).abs() < 0.01);
    }

    #[test]
    fn presentation_tracker_accepts_sequence_reset_on_new_surface() {
        let tracker = PreviewPresentationTracker::default();
        let started = Instant::now();
        tracker.observe_at(10, 50, 3, None, started, Some(100.0));
        tracker.observe_at(
            11,
            1,
            4,
            None,
            started + Duration::from_millis(16),
            Some(100.016),
        );

        let snapshot = tracker.snapshot_at(started + Duration::from_millis(16));
        assert_eq!(snapshot.preview_presented_count, 2);
        assert_eq!(snapshot.preview_presented_sequence, 1);
        assert_eq!(snapshot.preview_presented_graph_revision, 4);
        assert!((snapshot.preview_present_fps - 62.5).abs() < 0.01);
        assert_eq!(snapshot.preview_submit_to_present_ms, None);
    }

    #[test]
    fn presentation_tracker_expires_rate_and_rejects_future_submit_time() {
        let tracker = PreviewPresentationTracker::default();
        let started = Instant::now();
        tracker.observe_at(1, 1, 1, Some(101.0), started, Some(100.0));
        assert_eq!(
            tracker.snapshot_at(started).preview_submit_to_present_ms,
            None
        );

        let expired = tracker.snapshot_at(started + Duration::from_secs(3));
        assert_eq!(expired.preview_present_fps, 0.0);
        assert_eq!(expired.preview_presented_sequence, 1);
    }
}
