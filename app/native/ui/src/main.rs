#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod catalog;
mod graph;
mod mg_ui;
mod preset;
mod session;
mod state;

use serde_json::json;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

const HOST_ENGINE_SMOKE_TIMEOUT: Duration = Duration::from_secs(45);

fn engine_smoke_report_argument() -> Result<Option<PathBuf>, String> {
    let mut arguments = std::env::args_os().skip(1);
    let Some(command) = arguments.next() else {
        return Ok(None);
    };
    if command != "--verify-engine-v2" {
        return Ok(None);
    }
    let report = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "--verify-engine-v2 requires an absolute report path".to_owned())?;
    if arguments.next().is_some() {
        return Err("--verify-engine-v2 accepts exactly one report path".to_owned());
    }
    if !report.is_absolute() {
        return Err("--verify-engine-v2 report path must be absolute".to_owned());
    }
    Ok(Some(report))
}

fn run_engine_smoke() -> Result<serde_json::Value, String> {
    let started = Instant::now();
    let source = state::RuntimeStateSource::bridge_or_demo();
    let startup_deadline = Instant::now() + HOST_ENGINE_SMOKE_TIMEOUT;
    let initial_preview_sequence = loop {
        let view = source.read_view();
        if let Some(problem) = &view.details.last_error
            && !problem.recoverable
        {
            return Err(format!("{}: {}", problem.code, problem.message));
        }
        let has_pixels = view.preview.as_ref().is_some_and(|frame| {
            !frame.rgb.is_empty() && frame.rgb.iter().any(|value| *value != 0)
        });
        if view.snapshot.bridge_online
            && view.snapshot.source_ready
            && view.snapshot.preview_sequence > 0
            && has_pixels
        {
            break view.snapshot.preview_sequence;
        }
        if Instant::now() >= startup_deadline {
            return Err(
                "packaged Rust host did not receive a live RGB preview before timeout".to_owned(),
            );
        }
        thread::sleep(Duration::from_millis(10));
    };

    let (graph, _) = graph::demo_graph();
    let graph_command_id = source.replace_graph(&preset::export_graph(&graph))?;
    let graph_deadline = Instant::now() + HOST_ENGINE_SMOKE_TIMEOUT;
    let accepted_view = loop {
        let view = source.read_view();
        if let Some(problem) = &view.details.last_error
            && view.details.pending_command_id.is_none()
        {
            return Err(format!("{}: {}", problem.code, problem.message));
        }
        if view.details.last_ack_command_id == graph_command_id
            && view.snapshot.bridge_generation >= 1
            && view.details.current_graph_accepted
            && view.snapshot.graph_sequence > 0
            && view.snapshot.preview_sequence > initial_preview_sequence
            && view.snapshot.preview_sequence > view.details.last_effective_frame_sequence
        {
            break view;
        }
        if Instant::now() >= graph_deadline {
            return Err(
                "packaged Rust host did not receive an accepted processed graph before timeout"
                    .to_owned(),
            );
        }
        thread::sleep(Duration::from_millis(10));
    };

    let preview = accepted_view
        .preview
        .as_ref()
        .ok_or_else(|| "accepted graph had no Rust-owned preview frame".to_owned())?;
    let expected_preview_bytes = preview
        .width
        .checked_mul(preview.height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| "accepted RGB preview geometry overflowed usize".to_owned())?;
    if preview.width == 0 || preview.height == 0 || preview.rgb.len() != expected_preview_bytes {
        return Err(format!(
            "accepted RGB preview geometry was invalid: {}x{} produced {} bytes (expected {expected_preview_bytes})",
            preview.width,
            preview.height,
            preview.rgb.len()
        ));
    }
    if !preview.rgb.iter().any(|value| *value != 0) {
        return Err("accepted graph produced an all-zero RGB preview".to_owned());
    }
    if accepted_view.details.backend.trim().is_empty() || accepted_view.details.backend == "offline"
    {
        return Err("accepted graph had no live engine backend identity".to_owned());
    }
    let result = json!({
        "passed": true,
        "protocol_version": 2,
        "backend": accepted_view.details.backend,
        "state_revision": accepted_view.details.state_revision,
        "graph_accepted": accepted_view.details.current_graph_accepted,
        "graph_command_id": graph_command_id,
        "last_ack_command_id": accepted_view.details.last_ack_command_id,
        "last_effective_frame_sequence": accepted_view.details.last_effective_frame_sequence,
        "graph_generation": accepted_view.snapshot.bridge_generation,
        "graph_sequence": accepted_view.snapshot.graph_sequence,
        "initial_preview_sequence": initial_preview_sequence,
        "preview_sequence": preview.sequence,
        "preview_width": preview.width,
        "preview_height": preview.height,
        "preview_bytes": preview.rgb.len(),
        "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0,
    });
    drop(source);
    Ok(result)
}

fn main() -> eframe::Result {
    match engine_smoke_report_argument() {
        Ok(Some(report)) => {
            let outcome = run_engine_smoke();
            let document = match &outcome {
                Ok(value) => value.clone(),
                Err(message) => json!({"passed": false, "error": message}),
            };
            let write_result = std::fs::write(
                &report,
                serde_json::to_vec_pretty(&document).expect("smoke report is serializable"),
            );
            if outcome.is_err() || write_result.is_err() {
                std::process::exit(1);
            }
            std::process::exit(0);
        }
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
        Ok(None) => {}
    }

    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("MegaGlitchNode 00.9.4 — Functional Preview / Engine V2")
            .with_inner_size([1600.0, 960.0])
            .with_min_inner_size([960.0, 600.0])
            .with_maximized(true)
            .with_resizable(true),
        renderer: eframe::Renderer::Wgpu,
        centered: true,
        ..Default::default()
    };

    eframe::run_native(
        "MegaGlitchNode",
        options,
        Box::new(|creation| Ok(Box::new(app::MegaGlitchApp::new(creation)))),
    )
}
