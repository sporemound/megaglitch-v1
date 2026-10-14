use std::collections::{BTreeMap, HashMap};

use eframe::egui;
use egui_snarl::{InPinId, NodeId, OutPinId, Snarl};
use serde::{Deserialize, Serialize};

use crate::catalog::{self, ParameterKind, ParameterValue};
use crate::graph::{self, GraphNode};

pub const PYTHON_PRESET_VERSION: &str = "1.0";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PresetNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub backend_id: String,
    pub label: String,
    pub position: [f32; 2],
    #[serde(default)]
    pub params: BTreeMap<String, ParameterValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetLink {
    pub from_node: String,
    pub to_node: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from_node_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to_node_id: Option<String>,
    pub from_attr: String,
    pub to_attr: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PresetGraph {
    #[serde(default)]
    pub nodes: Vec<PresetNode>,
    #[serde(default)]
    pub links: Vec<PresetLink>,
    #[serde(default = "preset_version")]
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_pos: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_b_pos: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_pos: Option<[f32; 2]>,
}

fn preset_version() -> String {
    PYTHON_PRESET_VERSION.to_owned()
}

pub fn export_graph(graph: &Snarl<GraphNode>) -> PresetGraph {
    let mut preset = PresetGraph {
        nodes: Vec::new(),
        links: Vec::new(),
        version: preset_version(),
        source_pos: None,
        source_b_pos: None,
        display_pos: None,
    };
    let mut preset_ids: HashMap<NodeId, String> = HashMap::new();
    let mut identifiers: HashMap<NodeId, String> = HashMap::new();

    for (node_id, position, node) in graph.nodes_pos_ids() {
        if node.spec.is_none() {
            match node.node_type {
                "SourceNode" => {
                    preset.source_pos = Some([position.x, position.y]);
                    identifiers.insert(node_id, "SOURCE".to_owned());
                }
                "SecondarySourceNode" => {
                    preset.source_b_pos = Some([position.x, position.y]);
                    identifiers.insert(node_id, "SOURCE_B".to_owned());
                }
                "DisplayNode" => {
                    preset.display_pos = Some([position.x, position.y]);
                    identifiers.insert(node_id, "DISPLAY".to_owned());
                }
                _ => {}
            }
            continue;
        }

        let preset_id = node.instance_id.clone();
        let mut params: BTreeMap<String, ParameterValue> = node
            .parameters
            .iter()
            .map(|(name, value)| ((*name).to_owned(), value.clone()))
            .collect();
        if let Some(ParameterValue::Bool(enabled)) = params.get_mut("enabled") {
            *enabled = node.active;
        }
        preset.nodes.push(PresetNode {
            id: Some(preset_id.clone()),
            backend_id: node.node_type.to_owned(),
            label: node.title.to_owned(),
            position: [position.x, position.y],
            params,
        });
        preset_ids.insert(node_id, preset_id);
        identifiers.insert(node_id, node.title.to_owned());
    }

    for (out_pin, in_pin) in graph.wires() {
        let Some(from_node) = graph.get_node(out_pin.node) else {
            continue;
        };
        let Some(to_node) = graph.get_node(in_pin.node) else {
            continue;
        };
        let Some(from_socket) = from_node.output_specs().get(out_pin.output) else {
            continue;
        };
        let Some(to_socket) = to_node.input_specs().get(in_pin.input) else {
            continue;
        };
        preset.links.push(PresetLink {
            from_node: identifiers
                .get(&out_pin.node)
                .cloned()
                .unwrap_or_else(|| from_node.title.to_owned()),
            to_node: identifiers
                .get(&in_pin.node)
                .cloned()
                .unwrap_or_else(|| to_node.title.to_owned()),
            from_node_id: preset_ids.get(&out_pin.node).cloned(),
            to_node_id: preset_ids.get(&in_pin.node).cloned(),
            from_attr: from_socket.id.to_owned(),
            to_attr: to_socket.id.to_owned(),
        });
    }
    preset
}

pub fn import_graph(preset: &PresetGraph) -> Result<(Snarl<GraphNode>, Option<NodeId>), String> {
    if preset.version != PYTHON_PRESET_VERSION {
        return Err(format!(
            "unsupported Python node preset version {}",
            preset.version
        ));
    }
    let mut graph = Snarl::new();
    let source = graph.insert_node(
        position(preset.source_pos.unwrap_or([30.0, 100.0])),
        graph::source_node("SourceNode", "PRIMARY SOURCE", "TEST / IMAGE / VIDEO", 0.0),
    );
    let display = graph.insert_node(
        position(preset.display_pos.unwrap_or([790.0, 90.0])),
        graph::output_node(),
    );
    let secondary = preset.source_b_pos.map(|pos| {
        graph.insert_node(
            position(pos),
            graph::source_node(
                "SecondarySourceNode",
                "SOURCE B (UNAVAILABLE)",
                "GATE 2 / NOT OWNED",
                0.0,
            ),
        )
    });

    let mut explicit_ids: HashMap<String, NodeId> = HashMap::new();
    let mut legacy_labels: HashMap<String, NodeId> = HashMap::new();
    let mut selected = None;
    for saved in &preset.nodes {
        let spec = catalog::find(&saved.backend_id)
            .ok_or_else(|| format!("unknown backend node type {}", saved.backend_id))?;
        let mut node = GraphNode::from_spec(spec);
        if let Some(id) = &saved.id {
            if explicit_ids.contains_key(id) {
                return Err(format!("duplicate preset node id {id}"));
            }
            node.instance_id.clone_from(id);
        }
        for (name, value) in &saved.params {
            let schema = spec
                .parameters
                .iter()
                .find(|parameter| parameter.id == name)
                .ok_or_else(|| format!("unknown parameter {}.{}", saved.backend_id, name))?;
            let validated = validate_parameter(schema.kind, schema.choices, value)?;
            if name == "enabled"
                && let ParameterValue::Bool(enabled) = validated
            {
                node.set_active(enabled);
            } else {
                node.parameters.insert(schema.id, validated);
            }
        }
        let node_id = graph.insert_node(position(saved.position), node);
        selected.get_or_insert(node_id);
        if let Some(id) = &saved.id {
            explicit_ids.insert(id.clone(), node_id);
        }
        legacy_labels.entry(saved.label.clone()).or_insert(node_id);
    }

    for saved in &preset.links {
        let from_node = resolve_node(
            saved.from_node_id.as_deref().unwrap_or(&saved.from_node),
            source,
            secondary,
            display,
            &explicit_ids,
            &legacy_labels,
        )?;
        let to_node = resolve_node(
            saved.to_node_id.as_deref().unwrap_or(&saved.to_node),
            source,
            secondary,
            display,
            &explicit_ids,
            &legacy_labels,
        )?;
        let from_index = graph[from_node]
            .output_specs()
            .iter()
            .position(|socket| socket.id == saved.from_attr)
            .ok_or_else(|| {
                format!(
                    "unknown output socket {}.{}",
                    graph[from_node].node_type, saved.from_attr
                )
            })?;
        let to_index = graph[to_node]
            .input_specs()
            .iter()
            .position(|socket| socket.id == saved.to_attr)
            .ok_or_else(|| {
                format!(
                    "unknown input socket {}.{}",
                    graph[to_node].node_type, saved.to_attr
                )
            })?;
        graph.connect(
            OutPinId {
                node: from_node,
                output: from_index,
            },
            InPinId {
                node: to_node,
                input: to_index,
            },
        );
    }
    Ok((graph, selected))
}

fn position(value: [f32; 2]) -> egui::Pos2 {
    egui::pos2(value[0], value[1])
}

fn resolve_node(
    identifier: &str,
    source: NodeId,
    secondary: Option<NodeId>,
    display: NodeId,
    explicit_ids: &HashMap<String, NodeId>,
    legacy_labels: &HashMap<String, NodeId>,
) -> Result<NodeId, String> {
    match identifier {
        "SOURCE" => Ok(source),
        "SOURCE_B" => secondary.ok_or_else(|| "preset references missing SOURCE_B".to_owned()),
        "DISPLAY" => Ok(display),
        other => explicit_ids
            .get(other)
            .or_else(|| legacy_labels.get(other))
            .copied()
            .ok_or_else(|| format!("unknown preset node identifier {other}")),
    }
}

fn validate_parameter(
    kind: ParameterKind,
    choices: &[&str],
    value: &ParameterValue,
) -> Result<ParameterValue, String> {
    match (kind, value) {
        (ParameterKind::Float, ParameterValue::Float(value)) => Ok(ParameterValue::Float(*value)),
        (ParameterKind::Float, ParameterValue::Int(value)) => {
            Ok(ParameterValue::Float(*value as f64))
        }
        (ParameterKind::Int, ParameterValue::Int(value)) => Ok(ParameterValue::Int(*value)),
        (ParameterKind::Bool, ParameterValue::Bool(value)) => Ok(ParameterValue::Bool(*value)),
        (ParameterKind::Text, ParameterValue::Text(value)) => {
            Ok(ParameterValue::Text(value.clone()))
        }
        (ParameterKind::Enum, ParameterValue::Text(value)) if choices.contains(&value.as_str()) => {
            Ok(ParameterValue::Text(value.clone()))
        }
        _ => Err("parameter value does not match NodeSpec type or enum choices".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY_PRESET: &str = r#"{
      "nodes": [{
        "id": "PixelSortingNode:1",
        "type": "PixelSortingNode",
        "label": "Pixel Sorting",
        "position": [120, 80],
        "params": {
          "enabled": true,
          "pixel_sorting_intensity": 0.75,
          "pixel_sorting_direction": "Vertical",
          "sort_chance": 42,
          "pixel_sort_evolution": 0.2,
          "pixel_sort_evolution_speed": 1.4
        }
      }],
      "links": [
        {
          "from_node": "SOURCE",
          "to_node": "Pixel Sorting",
          "to_node_id": "PixelSortingNode:1",
          "from_attr": "image_out",
          "to_attr": "image_in"
        },
        {
          "from_node": "Pixel Sorting",
          "to_node": "DISPLAY",
          "from_node_id": "PixelSortingNode:1",
          "from_attr": "image_out",
          "to_attr": "image_in"
        }
      ],
      "version": "1.0",
      "source_pos": [5, -70],
      "display_pos": [685, -68]
    }"#;

    #[test]
    fn imports_python_preset_with_named_sockets_and_typed_values() {
        let preset: PresetGraph = serde_json::from_str(LEGACY_PRESET).unwrap();
        let (graph, selected) = import_graph(&preset).unwrap();
        let node = &graph[selected.unwrap()];
        assert_eq!(node.node_type, "PixelSortingNode");
        assert_eq!(
            node.parameters["pixel_sorting_direction"],
            ParameterValue::Text("Vertical".to_owned())
        );
        assert_eq!(graph.wires().count(), 2);
    }

    #[test]
    fn exports_python_compatible_ids_parameters_and_socket_names() {
        let preset: PresetGraph = serde_json::from_str(LEGACY_PRESET).unwrap();
        let (graph, _) = import_graph(&preset).unwrap();
        let exported = export_graph(&graph);
        assert_eq!(exported.version, "1.0");
        assert_eq!(exported.nodes[0].id.as_deref(), Some("PixelSortingNode:1"));
        assert_eq!(
            exported.nodes[0].params["sort_chance"],
            ParameterValue::Int(42)
        );
        assert!(exported.links.iter().any(|link| {
            link.from_node == "SOURCE"
                && link.from_attr == "image_out"
                && link.to_attr == "image_in"
        }));
        let json = serde_json::to_value(&exported).unwrap();
        assert_eq!(json["nodes"][0]["type"], "PixelSortingNode");
    }

    #[test]
    fn rejects_unknown_parameters_and_wrong_enum_values() {
        let mut preset: PresetGraph = serde_json::from_str(LEGACY_PRESET).unwrap();
        preset.nodes[0].params.insert(
            "pixel_sorting_direction".to_owned(),
            ParameterValue::Text("Diagonal".to_owned()),
        );
        assert!(import_graph(&preset).is_err());
    }

    #[test]
    fn stable_node_ids_survive_layout_and_repeated_export() {
        let preset: PresetGraph = serde_json::from_str(LEGACY_PRESET).unwrap();
        let (mut graph, selected) = import_graph(&preset).unwrap();
        let node_id = selected.unwrap();
        let first = export_graph(&graph);
        graph
            .nodes_ids_data_mut()
            .find(|(id, _)| *id == node_id)
            .expect("selected graph node")
            .1
            .pos = egui::pos2(999.0, -125.0);
        let second = export_graph(&graph);
        assert_eq!(first.nodes[0].id.as_deref(), Some("PixelSortingNode:1"));
        assert_eq!(second.nodes[0].id, first.nodes[0].id);
        assert_ne!(second.nodes[0].position, first.nodes[0].position);
    }

    #[test]
    fn duplicate_persistent_node_ids_are_rejected() {
        let mut preset: PresetGraph = serde_json::from_str(LEGACY_PRESET).unwrap();
        let mut duplicate = preset.nodes[0].clone();
        duplicate.position = [500.0, 200.0];
        preset.nodes.push(duplicate);
        assert!(
            import_graph(&preset)
                .unwrap_err()
                .contains("duplicate preset node id")
        );
    }
}
