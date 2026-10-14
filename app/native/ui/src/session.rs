use std::collections::HashMap;

use serde_json::{Map, Value, json};

use crate::preset::PresetGraph;

pub const SESSION_FORMAT: &str = "megaglitch.session";
pub const SESSION_FORMAT_VERSION: u64 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeckId {
    A,
    B,
}

impl DeckId {
    pub const fn key(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
        }
    }
}

/// Lossless Rust view of the Python-owned canonical session document.
///
/// The raw JSON remains authoritative so newer Python fields survive an older
/// Rust host. Typed graph views are projections used by the editor, never a
/// replacement project format.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalSession {
    document: Value,
}

impl CanonicalSession {
    pub fn new(deck_a: &PresetGraph, deck_b: &PresetGraph) -> Result<Self, String> {
        let document = json!({
            "format": SESSION_FORMAT,
            "format_version": SESSION_FORMAT_VERSION,
            "metadata": {},
            "decks": {
                "a": {
                    "source_bindings": {"SOURCE": "primary", "SOURCE_B": "secondary"},
                    "graph": serde_json::to_value(deck_a).map_err(|error| error.to_string())?,
                },
                "b": {
                    "source_bindings": {"SOURCE": "secondary", "SOURCE_B": "secondary"},
                    "graph": serde_json::to_value(deck_b).map_err(|error| error.to_string())?,
                },
            },
            "sources": {},
            "source_compositor": {},
            "master": {"mix": 0.5, "transition": "Normal Mix"},
            "macros": {},
            "modulation": {"enabled": false, "routes": [], "unresolved_legacy_connections": []},
            "timeline": {"enabled": false, "timebase": "seconds", "tracks": [], "unresolved_tracks": []},
            "audio": {},
            "transport": {},
            "render": {},
            "outputs": {},
            "ui": {},
            "extensions": {},
        });
        Self::from_value(document)
    }

    pub fn from_value(document: Value) -> Result<Self, String> {
        let root = document
            .as_object()
            .ok_or_else(|| "session document must be a JSON object".to_owned())?;
        if root.get("format").and_then(Value::as_str) != Some(SESSION_FORMAT) {
            return Err(format!("session format must be {SESSION_FORMAT}"));
        }
        if root.get("format_version").and_then(Value::as_u64) != Some(SESSION_FORMAT_VERSION) {
            return Err(format!(
                "session format_version must be {SESSION_FORMAT_VERSION}"
            ));
        }
        for deck in [DeckId::A, DeckId::B] {
            let graph = root
                .get("decks")
                .and_then(Value::as_object)
                .and_then(|decks| decks.get(deck.key()))
                .and_then(Value::as_object)
                .and_then(|state| state.get("graph"))
                .ok_or_else(|| format!("session deck {} has no graph", deck.key()))?;
            serde_json::from_value::<PresetGraph>(graph.clone())
                .map_err(|error| format!("invalid deck {} graph: {error}", deck.key()))?;
        }
        Ok(Self { document })
    }

    pub fn as_value(&self) -> &Value {
        &self.document
    }

    #[allow(dead_code)] // Used by the pending transactional replace_session command.
    pub fn into_value(self) -> Value {
        self.document
    }

    #[allow(dead_code)] // Deck B workspace import is wired in the next integration step.
    pub fn deck_graph(&self, deck: DeckId) -> Result<PresetGraph, String> {
        let graph = self
            .document
            .get("decks")
            .and_then(Value::as_object)
            .and_then(|decks| decks.get(deck.key()))
            .and_then(Value::as_object)
            .and_then(|state| state.get("graph"))
            .ok_or_else(|| format!("session deck {} has no graph", deck.key()))?;
        serde_json::from_value(graph.clone())
            .map_err(|error| format!("invalid deck {} graph: {error}", deck.key()))
    }

    /// Overlay an edited typed graph while retaining unmodelled graph, node,
    /// parameter and link fields from the authoritative Python document.
    pub fn update_deck_graph(&mut self, deck: DeckId, edited: &PresetGraph) -> Result<(), String> {
        let edited_value = serde_json::to_value(edited)
            .map_err(|error| format!("could not serialize deck graph: {error}"))?;
        let decks = self
            .document
            .get_mut("decks")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| "session has no decks object".to_owned())?;
        let state = decks
            .get_mut(deck.key())
            .and_then(Value::as_object_mut)
            .ok_or_else(|| format!("session has no deck {} object", deck.key()))?;
        let previous = state
            .get("graph")
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::new()));
        state.insert("graph".to_owned(), merge_graph(previous, edited_value)?);
        Ok(())
    }

    #[allow(dead_code)] // PERFORM owns this once replace_session is negotiated.
    pub fn set_master(&mut self, mix: f64, transition: &str) -> Result<(), String> {
        if !mix.is_finite() || !(0.0..=1.0).contains(&mix) {
            return Err("master mix must be finite and between 0 and 1".to_owned());
        }
        if transition.trim().is_empty() {
            return Err("master transition cannot be empty".to_owned());
        }
        let root = self
            .document
            .as_object_mut()
            .ok_or_else(|| "session document must remain an object".to_owned())?;
        let master = root
            .entry("master")
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .ok_or_else(|| "session master must be an object".to_owned())?;
        master.insert("mix".to_owned(), Value::from(mix));
        master.insert(
            "transition".to_owned(),
            Value::String(transition.to_owned()),
        );
        Ok(())
    }
}

fn merge_graph(previous: Value, edited: Value) -> Result<Value, String> {
    let mut result = previous.as_object().cloned().unwrap_or_default();
    let edited = edited
        .as_object()
        .ok_or_else(|| "edited graph must serialize as an object".to_owned())?;

    let prior_nodes = result
        .get("nodes")
        .and_then(Value::as_array)
        .map(|nodes| {
            nodes
                .iter()
                .filter_map(|node| {
                    let object = node.as_object()?;
                    let id = object.get("id")?.as_str()?;
                    Some((id.to_owned(), object.clone()))
                })
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let merged_nodes = edited
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "edited graph nodes must be an array".to_owned())?
        .iter()
        .map(|node| merge_node(node, &prior_nodes))
        .collect::<Result<Vec<_>, _>>()?;

    let prior_links = result
        .get("links")
        .and_then(Value::as_array)
        .map(|links| {
            links
                .iter()
                .filter_map(|link| Some((link_identity(link)?, link.as_object()?.clone())))
                .collect::<HashMap<_, _>>()
        })
        .unwrap_or_default();
    let merged_links = edited
        .get("links")
        .and_then(Value::as_array)
        .ok_or_else(|| "edited graph links must be an array".to_owned())?
        .iter()
        .map(|link| {
            let mut merged = link_identity(link)
                .and_then(|identity| prior_links.get(&identity).cloned())
                .unwrap_or_default();
            let object = link
                .as_object()
                .ok_or_else(|| "edited graph link must be an object".to_owned())?;
            merged.extend(object.clone());
            Ok(Value::Object(merged))
        })
        .collect::<Result<Vec<_>, String>>()?;

    for (key, value) in edited {
        match key.as_str() {
            "nodes" => {
                result.insert(key.clone(), Value::Array(merged_nodes.clone()));
            }
            "links" => {
                result.insert(key.clone(), Value::Array(merged_links.clone()));
            }
            _ => {
                result.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(Value::Object(result))
}

fn merge_node(
    edited: &Value,
    previous: &HashMap<String, Map<String, Value>>,
) -> Result<Value, String> {
    let edited = edited
        .as_object()
        .ok_or_else(|| "edited graph node must be an object".to_owned())?;
    let id = edited
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| "edited graph node requires a stable id".to_owned())?;
    let mut merged = previous.get(id).cloned().unwrap_or_default();
    let previous_params = merged
        .get("params")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    merged.extend(edited.clone());
    if let Some(edited_params) = edited.get("params").and_then(Value::as_object) {
        let mut params = previous_params;
        params.extend(edited_params.clone());
        merged.insert("params".to_owned(), Value::Object(params));
    }
    Ok(Value::Object(merged))
}

fn link_identity(link: &Value) -> Option<String> {
    let link = link.as_object()?;
    let from = link
        .get("from_node_id")
        .and_then(Value::as_str)
        .or_else(|| link.get("from_node").and_then(Value::as_str))?;
    let to = link
        .get("to_node_id")
        .and_then(Value::as_str)
        .or_else(|| link.get("to_node").and_then(Value::as_str))?;
    let from_attr = link.get("from_attr")?.as_str()?;
    let to_attr = link.get("to_attr")?.as_str()?;
    Some(format!("{from}\u{1f}{from_attr}\u{1f}{to}\u{1f}{to_attr}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{graph, preset};

    fn session() -> CanonicalSession {
        let (graph_a, _) = graph::demo_graph();
        let (graph_b, _) = graph::demo_graph();
        CanonicalSession::new(
            &preset::export_graph(&graph_a),
            &preset::export_graph(&graph_b),
        )
        .unwrap()
    }

    #[test]
    fn creates_both_decks_with_python_session_identity() {
        let session = session();
        assert_eq!(session.as_value()["format"], SESSION_FORMAT);
        assert_eq!(session.as_value()["format_version"], SESSION_FORMAT_VERSION);
        assert!(!session.deck_graph(DeckId::A).unwrap().nodes.is_empty());
        assert!(!session.deck_graph(DeckId::B).unwrap().nodes.is_empty());
    }

    #[test]
    fn graph_overlay_preserves_unknown_python_fields() {
        let mut value = session().into_value();
        value["future_top"] = json!({"kept": true});
        value["decks"]["a"]["graph"]["future_graph"] = json!(17);
        value["decks"]["a"]["graph"]["nodes"][0]["future_node"] = json!("retain");
        value["decks"]["a"]["graph"]["nodes"][0]["params"]["future_parameter"] = json!(0.75);
        value["decks"]["a"]["graph"]["links"][0]["future_link"] = json!("retain");
        let mut session = CanonicalSession::from_value(value).unwrap();
        let mut graph = session.deck_graph(DeckId::A).unwrap();
        graph.nodes[0].position = [777.0, -25.0];
        session.update_deck_graph(DeckId::A, &graph).unwrap();

        let value = session.as_value();
        assert_eq!(value["future_top"]["kept"], true);
        assert_eq!(value["decks"]["a"]["graph"]["future_graph"], 17);
        assert_eq!(
            value["decks"]["a"]["graph"]["nodes"][0]["future_node"],
            "retain"
        );
        assert_eq!(
            value["decks"]["a"]["graph"]["nodes"][0]["params"]["future_parameter"],
            0.75
        );
        assert_eq!(
            value["decks"]["a"]["graph"]["links"][0]["future_link"],
            "retain"
        );
    }

    #[test]
    fn master_patch_is_bounded_and_lossless() {
        let mut session = session();
        session.document["master"]["future_mode"] = json!("kept");
        session.set_master(0.25, "Difference").unwrap();
        assert_eq!(session.as_value()["master"]["mix"], 0.25);
        assert_eq!(session.as_value()["master"]["transition"], "Difference");
        assert_eq!(session.as_value()["master"]["future_mode"], "kept");
        assert!(session.set_master(f64::NAN, "Difference").is_err());
        assert!(session.set_master(0.5, "  ").is_err());
    }

    #[test]
    fn rejects_noncanonical_session_identity() {
        let mut value = session().into_value();
        value["format_version"] = json!(99);
        assert!(CanonicalSession::from_value(value).is_err());
    }
}
