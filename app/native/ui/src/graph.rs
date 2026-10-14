use eframe::egui::{self, Frame, Stroke};
use egui_snarl::ui::{PinInfo, SnarlViewer};
use egui_snarl::{InPin, InPinId, NodeId, OutPin, OutPinId, Snarl};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::catalog::{self, NodeSpec, ParameterValue, SocketDataType, SocketSpec};
use crate::mg_ui;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Source,
    Transform,
    Feedback,
    Output,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphStatus {
    Draft,
    Pending,
    Accepted,
    Rejected,
}

impl GraphStatus {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Draft => "DRAFT",
            Self::Pending => "PENDING",
            Self::Accepted => "ACCEPTED",
            Self::Rejected => "REJECTED",
        }
    }

    fn color(self) -> egui::Color32 {
        match self {
            Self::Draft => mg_ui::MUTED,
            Self::Pending => mg_ui::ACCENT,
            Self::Accepted => mg_ui::SIGNAL,
            Self::Rejected => mg_ui::WARNING,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LibraryNode {
    Capture,
    Media,
    Generator,
    Color,
    Temporal,
    Output,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeRequest {
    Builtin(LibraryNode),
    Catalog(usize),
}

impl LibraryNode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Capture => "PRIMARY SOURCE / SELECTED INPUT",
            Self::Media => "PRIMARY SOURCE / IMAGE OR VIDEO",
            Self::Generator => "GRAPH GENERATOR / LOCAL",
            Self::Color => "PROCESS / COLOR + LUT",
            Self::Temporal => "PROCESS / TEMPORAL",
            Self::Output => "VIDEO OUTPUT / PROGRAM",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphNode {
    /// Stable session identity. This must survive graph layout changes and
    /// export/import so timeline and modulation targets never follow a node's
    /// transient egui_snarl arena index.
    pub instance_id: String,
    pub node_type: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    pub kind: NodeKind,
    pub active: bool,
    pub cost_ms: f32,
    pub amount: f32,
    pub spec: Option<NodeSpec>,
    pub parameters: std::collections::BTreeMap<&'static str, ParameterValue>,
    fallback_inputs: &'static [SocketSpec],
    fallback_outputs: &'static [SocketSpec],
}

static NEXT_NODE_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

fn new_instance_id(node_type: &str) -> String {
    let ordinal = NEXT_NODE_INSTANCE_ID.fetch_add(1, Ordering::Relaxed);
    format!("{node_type}:rust-{ordinal:016x}")
}

impl GraphNode {
    pub fn from_spec(spec: NodeSpec) -> Self {
        let generator = spec.inputs.is_empty();
        let temporal = spec.category == "Temporal" || spec.label.contains("Feedback");
        let parameters = catalog::default_parameters(spec);
        let active = !matches!(parameters.get("enabled"), Some(ParameterValue::Bool(false)));
        Self {
            instance_id: new_instance_id(spec.backend_id),
            node_type: spec.backend_id,
            title: spec.label,
            category: spec.category,
            kind: if generator {
                NodeKind::Source
            } else if temporal {
                NodeKind::Feedback
            } else {
                NodeKind::Transform
            },
            active,
            cost_ms: 0.0,
            amount: 0.5,
            spec: Some(spec),
            parameters,
            fallback_inputs: &[],
            fallback_outputs: &[],
        }
    }

    fn builtin(
        node_type: &'static str,
        title: &'static str,
        category: &'static str,
        kind: NodeKind,
        cost_ms: f32,
        inputs: &'static [SocketSpec],
        outputs: &'static [SocketSpec],
    ) -> Self {
        Self {
            instance_id: new_instance_id(node_type),
            node_type,
            title,
            category,
            kind,
            active: true,
            cost_ms,
            amount: 1.0,
            spec: None,
            parameters: std::collections::BTreeMap::new(),
            fallback_inputs: inputs,
            fallback_outputs: outputs,
        }
    }

    pub fn input_specs(&self) -> &'static [SocketSpec] {
        self.spec
            .map(|spec| spec.inputs)
            .unwrap_or(self.fallback_inputs)
    }

    pub fn output_specs(&self) -> &'static [SocketSpec] {
        self.spec
            .map(|spec| spec.outputs)
            .unwrap_or(self.fallback_outputs)
    }

    fn inputs(&self) -> usize {
        self.input_specs().len()
    }

    fn outputs(&self) -> usize {
        self.output_specs().len()
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
        if let Some(ParameterValue::Bool(enabled)) = self.parameters.get_mut("enabled") {
            *enabled = active;
        }
    }

    pub fn set_amount(&mut self, amount: f32) {
        self.amount = amount.clamp(0.0, 1.0);
        if let Some(spec) = self.spec {
            for param in spec.parameters {
                if param.id == "enabled" || param.id == "multimodal_amount" {
                    continue;
                }
                if param.kind == catalog::ParameterKind::Float {
                    if let Some(ParameterValue::Float(val)) = self.parameters.get_mut(param.id) {
                        let min = param.minimum.unwrap_or(0.0);
                        let max = param.maximum.unwrap_or(1.0);
                        *val = min + (max - min) * (self.amount as f64);
                    }
                    break;
                }
            }
        }
    }
}

const VIDEO_IN: &[SocketSpec] = &[SocketSpec {
    id: "image_in",
    label: "Image In",
    data_type: SocketDataType::Video,
}];
const VIDEO_OUT: &[SocketSpec] = &[SocketSpec {
    id: "image_out",
    label: "Image Out",
    data_type: SocketDataType::Video,
}];

pub(crate) fn source_node(
    node_type: &'static str,
    title: &'static str,
    category: &'static str,
    cost_ms: f32,
) -> GraphNode {
    GraphNode::builtin(
        node_type,
        title,
        category,
        NodeKind::Source,
        cost_ms,
        &[],
        VIDEO_OUT,
    )
}

pub(crate) fn output_node() -> GraphNode {
    GraphNode::builtin(
        "DisplayNode",
        "OUTPUT",
        "PROGRAM / DISPLAY",
        NodeKind::Output,
        0.21,
        VIDEO_IN,
        &[],
    )
}

fn required_spec(backend_id: &str) -> NodeSpec {
    catalog::find(backend_id)
        .unwrap_or_else(|| panic!("generated NodeSpec missing required node {backend_id}"))
}

pub fn demo_graph() -> (Snarl<GraphNode>, Option<NodeId>) {
    let mut graph = Snarl::new();
    let source = graph.insert_node(
        egui::pos2(30.0, 100.0),
        source_node("SourceNode", "PRIMARY SOURCE", "TEST / IMAGE / VIDEO", 0.34),
    );
    let mut color_node = GraphNode::from_spec(required_spec("SpectralShiftNode"));
    color_node.cost_ms = 1.82;
    color_node.amount = 0.68;
    color_node.parameters.insert("frequency_shift", ParameterValue::Float(0.45));
    color_node.parameters.insert("magnitude_corrupt", ParameterValue::Float(0.25));
    let color = graph.insert_node(egui::pos2(360.0, 100.0), color_node);

    let mut feedback_node = GraphNode::from_spec(required_spec("FeedbackTrailNode"));
    feedback_node.cost_ms = 2.41;
    feedback_node.amount = 0.42;
    feedback_node.parameters.insert("feedback_intensity", ParameterValue::Float(0.65));
    feedback_node.parameters.insert("trailing_intensity", ParameterValue::Float(0.55));
    let feedback = graph.insert_node(egui::pos2(690.0, 100.0), feedback_node);

    let output = graph.insert_node(egui::pos2(1020.0, 100.0), output_node());
    graph.connect(
        OutPinId {
            node: source,
            output: 0,
        },
        InPinId {
            node: color,
            input: 0,
        },
    );
    graph.connect(
        OutPinId {
            node: color,
            output: 0,
        },
        InPinId {
            node: feedback,
            input: 0,
        },
    );
    graph.connect(
        OutPinId {
            node: feedback,
            output: 0,
        },
        InPinId {
            node: output,
            input: 0,
        },
    );
    (graph, Some(color))
}

pub fn auto_wire_chain(graph: &mut Snarl<GraphNode>) {
    let mut node_items: Vec<(NodeId, NodeKind, f32)> = graph
        .nodes_pos_ids()
        .map(|(id, pos, node)| (id, node.kind, pos.x))
        .collect();
    if node_items.len() < 2 {
        return;
    }
    node_items.sort_by(|a, b| {
        let kind_order = |k: NodeKind| match k {
            NodeKind::Source => 0,
            NodeKind::Transform | NodeKind::Feedback => 1,
            NodeKind::Output => 2,
        };
        let ka = kind_order(a.1);
        let kb = kind_order(b.1);
        if ka != kb {
            ka.cmp(&kb)
        } else {
            a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    for window in node_items.windows(2) {
        let from_node = window[0].0;
        let to_node = window[1].0;
        if graph[from_node].outputs() > 0 && graph[to_node].inputs() > 0 {
            graph.drop_inputs(InPinId {
                node: to_node,
                input: 0,
            });
            graph.connect(
                OutPinId {
                    node: from_node,
                    output: 0,
                },
                InPinId {
                    node: to_node,
                    input: 0,
                },
            );
        }
    }
}

pub fn insert_library_node(graph: &mut Snarl<GraphNode>, node: LibraryNode) -> NodeId {
    insert_library_node_at(graph, node, None)
}

pub fn insert_library_node_at(
    graph: &mut Snarl<GraphNode>,
    node: LibraryNode,
    pos: Option<egui::Pos2>,
) -> NodeId {
    let count = graph.nodes().count();
    let default_pos = match node {
        LibraryNode::Capture | LibraryNode::Media | LibraryNode::Generator => {
            egui::pos2(30.0, 100.0 + (count as f32 % 3.0) * 160.0)
        }
        LibraryNode::Output => egui::pos2(870.0, 100.0 + (count as f32 % 3.0) * 160.0),
        LibraryNode::Color | LibraryNode::Temporal => egui::pos2(
            300.0 + (count as f32 % 3.0) * 240.0,
            100.0 + ((count as f32 / 3.0).floor() % 3.0) * 160.0,
        ),
    };
    let position = pos.unwrap_or(default_pos);
    let value = match node {
        LibraryNode::Capture => source_node(
            "SourceNode",
            "PRIMARY SOURCE",
            "SELECTED ENGINE INPUT",
            0.34,
        ),
        LibraryNode::Media => source_node(
            "SourceNode",
            "PRIMARY MEDIA SOURCE",
            "IMAGE / VIDEO FILE",
            0.28,
        ),
        LibraryNode::Generator => {
            let mut node = GraphNode::from_spec(required_spec("GradientGeneratorNode"));
            node.cost_ms = 0.52;
            node
        }
        LibraryNode::Color => {
            let mut node = GraphNode::from_spec(required_spec("ColorAdjustNode"));
            node.cost_ms = 1.2;
            node.parameters.insert("brightness", ParameterValue::Float(0.35));
            node.parameters.insert("contrast", ParameterValue::Float(0.4));
            node.parameters.insert("saturation", ParameterValue::Float(0.6));
            node
        }
        LibraryNode::Temporal => {
            let mut node = GraphNode::from_spec(required_spec("FeedbackTrailNode"));
            node.cost_ms = 2.1;
            node.amount = 0.42;
            node.parameters.insert("feedback_intensity", ParameterValue::Float(0.65));
            node.parameters.insert("trailing_intensity", ParameterValue::Float(0.55));
            node
        }
        LibraryNode::Output => output_node(),
    };
    graph.insert_node(position, value)
}

pub fn insert_catalog_node(graph: &mut Snarl<GraphNode>, descriptor: NodeSpec) -> NodeId {
    insert_catalog_node_at(graph, descriptor, None)
}

pub fn insert_catalog_node_at(
    graph: &mut Snarl<GraphNode>,
    descriptor: NodeSpec,
    pos: Option<egui::Pos2>,
) -> NodeId {
    let count = graph.nodes().count();
    let default_pos = egui::pos2(
        300.0 + (count as f32 % 3.0) * 240.0,
        120.0 + ((count as f32 / 3.0).floor() % 3.0) * 160.0,
    );
    let position = pos.unwrap_or(default_pos);
    graph.insert_node(position, GraphNode::from_spec(descriptor))
}

pub struct GraphViewer<'a> {
    pub selected: &'a mut Option<NodeId>,
    pub graph_revision: &'a mut u64,
    pub status: GraphStatus,
}

impl SnarlViewer<GraphNode> for GraphViewer<'_> {
    fn title(&mut self, node: &GraphNode) -> String {
        node.title.into()
    }

    fn inputs(&mut self, node: &GraphNode) -> usize {
        node.inputs()
    }

    fn outputs(&mut self, node: &GraphNode) -> usize {
        node.outputs()
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<GraphNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let socket = snarl[pin.id.node].input_specs()[pin.id.input];
        ui.label(
            egui::RichText::new(format!("IN  ·  {}", socket.label.to_ascii_uppercase()))
                .monospace()
                .size(8.5)
                .strong()
                .color(mg_ui::SIGNAL),
        )
        .on_hover_text(format!("{} input", socket.label));
        PinInfo::circle()
            .with_fill(mg_ui::SIGNAL)
            .with_stroke(Stroke::new(2.0, mg_ui::TEXT))
            .with_wire_color(mg_ui::SIGNAL)
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        snarl: &mut Snarl<GraphNode>,
    ) -> impl egui_snarl::ui::SnarlPin + 'static {
        let socket = snarl[pin.id.node].output_specs()[pin.id.output];
        ui.label(
            egui::RichText::new(format!("{}  ·  OUT", socket.label.to_ascii_uppercase()))
                .monospace()
                .size(8.5)
                .strong()
                .color(mg_ui::SIGNAL),
        )
        .on_hover_text(format!("{} output", socket.label));
        PinInfo::circle()
            .with_fill(mg_ui::SIGNAL)
            .with_stroke(Stroke::new(2.0, mg_ui::TEXT))
            .with_wire_color(mg_ui::SIGNAL)
    }

    fn show_header(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<GraphNode>,
    ) {
        let graph_node = &snarl[node];
        let is_selected = *self.selected == Some(node);
        if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
            if (ui.input(|i| i.pointer.primary_down() || i.pointer.primary_clicked()))
                && ui.min_rect().expand(10.0).contains(pos)
            {
                *self.selected = Some(node);
            }
        }
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(graph_node.title)
                    .strong()
                    .size(10.0)
                    .color(if is_selected {
                        mg_ui::ACCENT
                    } else {
                        mg_ui::TEXT
                    }),
            );
        });
    }

    fn has_body(&mut self, _node: &GraphNode) -> bool {
        true
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        snarl: &mut Snarl<GraphNode>,
    ) {
        let graph_node = &mut snarl[node];
        const BODY_WIDTH: f32 = 175.0;
        let is_selected = *self.selected == Some(node);
        if let Some(pos) = ui.input(|i| i.pointer.interact_pos()) {
            if (ui.input(|i| i.pointer.primary_down() || i.pointer.primary_clicked()))
                && ui.min_rect().expand(16.0).contains(pos)
            {
                *self.selected = Some(node);
            }
        }
        let resp = ui.vertical(|ui| {
            ui.set_width(BODY_WIDTH);
            ui.horizontal(|ui| {
                if ui
                    .button(
                        egui::RichText::new(if is_selected { "★ ACTIVE" } else { "SELECT" })
                            .size(9.0)
                            .color(if is_selected { mg_ui::ACCENT } else { mg_ui::TEXT }),
                    )
                    .on_hover_text("Select this node to inspect and edit its parameters")
                    .clicked()
                {
                    *self.selected = Some(node);
                }
                ui.label(
                    egui::RichText::new(graph_node.category)
                        .size(8.5)
                        .color(if is_selected { mg_ui::ACCENT } else { mg_ui::MUTED }),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut active = graph_node.active;
                    if ui
                        .checkbox(&mut active, "")
                        .on_hover_text("Enable / Bypass node")
                        .changed()
                    {
                        graph_node.set_active(active);
                        *self.selected = Some(node);
                        *self.graph_revision = self.graph_revision.saturating_add(1);
                    }
                });
            });
            if matches!(graph_node.kind, NodeKind::Source | NodeKind::Output) {
                ui.label(
                    egui::RichText::new(if graph_node.kind == NodeKind::Source {
                        "START SIGNAL HERE"
                    } else {
                        "END SIGNAL HERE"
                    })
                    .monospace()
                    .size(9.0)
                    .strong()
                    .color(mg_ui::WARNING),
                );
            }
            let (rect, response) =
                ui.allocate_exact_size(egui::vec2(BODY_WIDTH, 26.0), egui::Sense::click());
            ui.painter().rect_filled(
                rect,
                2.0,
                if is_selected {
                    mg_ui::PANEL_RAISED
                } else {
                    mg_ui::BG
                },
            );
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                graph_node.node_type,
                egui::FontId::monospace(9.0),
                if is_selected {
                    mg_ui::ACCENT
                } else {
                    mg_ui::MUTED
                },
            );
            if response.clicked() {
                *self.selected = Some(node);
            }
            if graph_node.spec.is_none() {
                let mut amount = graph_node.amount;
                if ui
                    .add(egui::Slider::new(&mut amount, 0.0..=1.0).text("Amount"))
                    .changed()
                {
                    graph_node.set_amount(amount);
                    *self.selected = Some(node);
                    *self.graph_revision = self.graph_revision.saturating_add(1);
                }
            } else if let Some(spec) = graph_node.spec {
                for param in spec.parameters.iter().take(3) {
                    if param.id == "enabled" || param.id == "multimodal_amount" {
                        continue;
                    }
                    if let Some(crate::catalog::ParameterValue::Float(val)) =
                        graph_node.parameters.get_mut(param.id)
                    {
                        let min = param.minimum.unwrap_or(0.0);
                        let max = param.maximum.unwrap_or(1.0);
                        if ui
                            .add(egui::Slider::new(val, min..=max).text(param.label))
                            .changed()
                        {
                            *self.selected = Some(node);
                            *self.graph_revision = self.graph_revision.saturating_add(1);
                        }
                    }
                }
            }
            let (status_rect, _) =
                ui.allocate_exact_size(egui::vec2(BODY_WIDTH, 16.0), egui::Sense::hover());
            ui.painter().circle_filled(
                status_rect.left_center() + egui::vec2(4.0, 0.0),
                3.0,
                if graph_node.active {
                    mg_ui::SIGNAL
                } else {
                    mg_ui::MUTED
                },
            );
            ui.painter().text(
                status_rect.left_center() + egui::vec2(12.0, 0.0),
                egui::Align2::LEFT_CENTER,
                if graph_node.active {
                    "ENABLED"
                } else {
                    "BYPASSED"
                },
                egui::FontId::monospace(8.5),
                mg_ui::TEXT,
            );
            ui.painter().text(
                status_rect.right_center(),
                egui::Align2::RIGHT_CENTER,
                self.status.label(),
                egui::FontId::monospace(8.0),
                self.status.color(),
            );
        });
        if resp.response.clicked() {
            *self.selected = Some(node);
        }
    }

    fn node_frame(
        &mut self,
        default: Frame,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        _snarl: &Snarl<GraphNode>,
    ) -> Frame {
        default
            .fill(mg_ui::PANEL_RAISED)
            .stroke(Stroke::new(
                1.0,
                if *self.selected == Some(node) {
                    mg_ui::ACCENT
                } else {
                    mg_ui::HAIRLINE
                },
            ))
            .corner_radius(2.0)
    }

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<GraphNode>) {
        snarl.drop_inputs(to.id);
        snarl.connect(from.id, to.id);
        *self.graph_revision = self.graph_revision.saturating_add(1);
    }

    fn disconnect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<GraphNode>) {
        if snarl.disconnect(from.id, to.id) {
            *self.graph_revision = self.graph_revision.saturating_add(1);
        }
    }

    fn drop_outputs(&mut self, pin: &OutPin, snarl: &mut Snarl<GraphNode>) {
        if snarl.drop_outputs(pin.id) > 0 {
            *self.graph_revision = self.graph_revision.saturating_add(1);
        }
    }

    fn drop_inputs(&mut self, pin: &InPin, snarl: &mut Snarl<GraphNode>) {
        if snarl.drop_inputs(pin.id) > 0 {
            *self.graph_revision = self.graph_revision.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_run_graph_has_visible_signal_endpoints() {
        let (graph, _) = demo_graph();
        assert_eq!(
            graph
                .nodes()
                .filter(|node| node.kind == NodeKind::Source)
                .count(),
            1
        );
        assert_eq!(
            graph
                .nodes()
                .filter(|node| node.kind == NodeKind::Output)
                .count(),
            1
        );
        assert!(graph.nodes().any(|node| node.title == "PRIMARY SOURCE"));
        assert!(graph.nodes().any(|node| node.title == "OUTPUT"));
    }

    #[test]
    fn graph_status_labels_are_distinct_and_truthful() {
        assert_eq!(GraphStatus::Draft.label(), "DRAFT");
        assert_eq!(GraphStatus::Pending.label(), "PENDING");
        assert_eq!(GraphStatus::Accepted.label(), "ACCEPTED");
        assert_eq!(GraphStatus::Rejected.label(), "REJECTED");
    }

    #[test]
    fn source_process_and_output_expose_readable_socket_names() {
        let (graph, selected) = demo_graph();
        let source = graph
            .nodes()
            .find(|node| node.kind == NodeKind::Source)
            .expect("default graph source");
        let process = &graph[selected.expect("default selected process")];
        let output = graph
            .nodes()
            .find(|node| node.kind == NodeKind::Output)
            .expect("default graph output");

        assert_eq!(source.output_specs()[0].label, "Image Out");
        assert_eq!(process.input_specs()[0].label, "Image In");
        assert_eq!(process.output_specs()[0].label, "Image Out");
        assert_eq!(output.input_specs()[0].label, "Image In");
    }

    #[test]
    fn library_can_create_source_process_and_output_nodes() {
        let mut graph = Snarl::new();
        let source = insert_library_node(&mut graph, LibraryNode::Capture);
        let process = insert_library_node(&mut graph, LibraryNode::Color);
        let output = insert_library_node(&mut graph, LibraryNode::Output);
        assert_eq!(graph[source].kind, NodeKind::Source);
        assert_eq!(graph[process].kind, NodeKind::Transform);
        assert_eq!(graph[output].kind, NodeKind::Output);
    }
}
