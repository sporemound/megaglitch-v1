use std::collections::BTreeMap;

pub const NODE_SPEC_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SocketDataType {
    Video,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SocketSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub data_type: SocketDataType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterKind {
    Float,
    Int,
    Bool,
    Enum,
    Text,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParameterDefault {
    Float(f64),
    Int(i64),
    Bool(bool),
    Text(&'static str),
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum ParameterValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl ParameterValue {
    pub fn from_default(value: ParameterDefault) -> Self {
        match value {
            ParameterDefault::Float(value) => Self::Float(value),
            ParameterDefault::Int(value) => Self::Int(value),
            ParameterDefault::Bool(value) => Self::Bool(value),
            ParameterDefault::Text(value) => Self::Text(value.to_owned()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub kind: ParameterKind,
    pub default: ParameterDefault,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub choices: &'static [&'static str],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpec {
    pub version: u16,
    pub backend_id: &'static str,
    pub label: &'static str,
    pub category: &'static str,
    pub backend: &'static str,
    pub inputs: &'static [SocketSpec],
    pub outputs: &'static [SocketSpec],
    pub parameters: &'static [ParameterSpec],
}

#[allow(clippy::approx_constant)]
mod generated {
    include!("generated_node_catalog.rs");
}

pub fn all() -> &'static [NodeSpec] {
    generated::NODES
}

pub fn find(backend_id: &str) -> Option<NodeSpec> {
    all()
        .iter()
        .copied()
        .find(|spec| spec.backend_id == backend_id)
}

pub fn default_parameters(spec: NodeSpec) -> BTreeMap<&'static str, ParameterValue> {
    spec.parameters
        .iter()
        .map(|parameter| {
            (
                parameter.id,
                ParameterValue::from_default(parameter.default),
            )
        })
        .collect()
}

pub fn matches(spec: NodeSpec, query: &str) -> bool {
    let query = query.trim().to_ascii_lowercase();
    query.is_empty()
        || spec.label.to_ascii_lowercase().contains(&query)
        || spec.backend_id.to_ascii_lowercase().contains(&query)
        || spec.category.to_ascii_lowercase().contains(&query)
        || spec.backend.contains(&query)
        || spec.parameters.iter().any(|param| {
            param.id.to_ascii_lowercase().contains(&query)
                || param.label.to_ascii_lowercase().contains(&query)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(id: &str) -> &'static NodeSpec {
        all().iter().find(|node| node.backend_id == id).unwrap()
    }

    #[test]
    fn generated_catalog_has_unique_versioned_backend_ids() {
        assert!(all().iter().all(|node| node.version == NODE_SPEC_VERSION));
        let mut ids: Vec<_> = all().iter().map(|node| node.backend_id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), all().len());
    }

    #[test]
    fn representative_specs_have_named_sockets_and_typed_parameters() {
        let pixel_sort = spec("PixelSortingNode");
        assert_eq!(pixel_sort.inputs[0].id, "image_in");
        assert_eq!(pixel_sort.outputs[0].id, "image_out");
        assert!(pixel_sort.parameters.iter().any(|param| {
            param.id == "sort_chance"
                && param.kind == ParameterKind::Int
                && param.minimum == Some(0.0)
                && param.maximum == Some(100.0)
        }));
        assert!(pixel_sort.parameters.iter().any(|param| {
            param.id == "pixel_sorting_direction"
                && param.kind == ParameterKind::Enum
                && param.choices == ["Horizontal", "Vertical"]
        }));

        let composite = spec("GenesisPriorityCompositeNode");
        assert_eq!(composite.inputs[0].id, "base_image");
        assert_eq!(composite.inputs[4].id, "corruption_mask");
        assert_eq!(composite.outputs[0].id, "image_out");

        let cuda = spec("CUDAFlowWarpNode");
        assert_eq!(cuda.backend, "cuda");
        assert!(
            cuda.parameters
                .iter()
                .any(|param| param.kind == ParameterKind::Float)
        );

        let generator = spec("GradientGeneratorNode");
        assert!(generator.inputs.is_empty());
        assert_eq!(generator.outputs[0].id, "image_out");
        assert!(generator.parameters.iter().any(|param| param.id == "mode"));
    }

    #[test]
    fn every_spec_has_unique_socket_and_parameter_ids() {
        for node in all() {
            let mut inputs: Vec<_> = node.inputs.iter().map(|socket| socket.id).collect();
            let input_count = inputs.len();
            inputs.sort_unstable();
            inputs.dedup();
            assert_eq!(
                inputs.len(),
                input_count,
                "duplicate input on {}",
                node.backend_id
            );

            let mut outputs: Vec<_> = node.outputs.iter().map(|socket| socket.id).collect();
            let output_count = outputs.len();
            outputs.sort_unstable();
            outputs.dedup();
            assert_eq!(
                outputs.len(),
                output_count,
                "duplicate output on {}",
                node.backend_id
            );

            let mut params: Vec<_> = node.parameters.iter().map(|param| param.id).collect();
            let param_count = params.len();
            params.sort_unstable();
            params.dedup();
            assert_eq!(
                params.len(),
                param_count,
                "duplicate parameter on {}",
                node.backend_id
            );
        }
    }
}
