"""Generate the versioned Rust NodeSpec catalog from Python node definitions.

The exporter uses Python's AST rather than importing the application, so it cannot
initialize Dear PyGui, OpenCV/CUDA, audio, or MIDI during regeneration.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "native" / "ui" / "src" / "generated_node_catalog.rs"
BASE_CLASSES = {"Node", "GlitchEffectNode", "MultiInputNode", "GeneratorNode"}
NODE_SPEC_VERSION = 1


@dataclass(frozen=True)
class Socket:
    name: str
    label: str


@dataclass(frozen=True)
class Parameter:
    name: str
    label: str
    kind: str
    default: bool | int | float | str
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class NodeSpec:
    backend_id: str
    label: str
    category: str
    backend: str
    inputs: tuple[Socket, ...]
    outputs: tuple[Socket, ...]
    parameters: tuple[Parameter, ...]


def class_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def categories() -> tuple[dict[str, list[str]], dict[str, str]]:
    tree = ast.parse((ROOT / "megaglitchfm_app_new.py").read_text(encoding="utf-8"))
    category_map: dict[str, list[str]] | None = None
    labels: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "node_categories"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict) and "Core" in value:
                category_map = value
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "node_menu_labels"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                labels = value
    if category_map is None:
        raise RuntimeError("node_categories was not found in megaglitchfm_app_new.py")
    return category_map, labels


def _literal(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    corruption_modes = env.get("CorruptionMode")
    if (
        isinstance(corruption_modes, dict)
        and isinstance(node, ast.Attribute)
        and node.attr == "value"
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "CorruptionMode"
    ):
        return corruption_modes.get(node.value.attr)
    if (
        isinstance(corruption_modes, dict)
        and isinstance(node, ast.ListComp)
        and len(node.generators) == 1
        and isinstance(node.generators[0].iter, ast.Name)
        and node.generators[0].iter.id == "CorruptionMode"
        and isinstance(node.elt, ast.Attribute)
        and node.elt.attr == "value"
    ):
        return list(corruption_modes.values())
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _dict_value(node: ast.Dict, key_name: str, env: dict[str, Any]) -> Any:
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == key_name:
            return _literal(value, env)
    return None


def _parameter_from_dict(node: ast.Dict, env: dict[str, Any]) -> Parameter | None:
    name = _dict_value(node, "name", env)
    kind = _dict_value(node, "type", env)
    if not isinstance(name, str) or kind not in {
        "float", "int", "combo", "checkbox", "text"
    }:
        return None
    label = _dict_value(node, "label", env)
    default = _dict_value(node, "default_value", env)
    if default is None:
        default = _dict_value(node, "default", env)
    minimum = _dict_value(node, "min_value", env)
    if minimum is None:
        minimum = _dict_value(node, "min", env)
    maximum = _dict_value(node, "max_value", env)
    if maximum is None:
        maximum = _dict_value(node, "max", env)
    items = _dict_value(node, "items", env)
    contract_kind = {"combo": "enum", "checkbox": "bool"}.get(kind, kind)
    fallback: bool | int | float | str = {
        "float": 0.0,
        "int": 0,
        "enum": "",
        "bool": False,
        "text": "",
    }[contract_kind]
    if not isinstance(default, (bool, int, float, str)):
        default = fallback
    choices = tuple(str(item) for item in items) if isinstance(items, (list, tuple)) else ()
    return Parameter(
        name=name,
        label=str(label or name.replace("_", " ").title()),
        kind=contract_kind,
        default=default,
        minimum=float(minimum) if isinstance(minimum, (int, float)) else None,
        maximum=float(maximum) if isinstance(maximum, (int, float)) else None,
        choices=choices,
    )


def _parameters(node: ast.AST | None, env: dict[str, Any]) -> list[Parameter]:
    if node is None:
        return []
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        return list(value) if isinstance(value, list) else []
    if isinstance(node, ast.List):
        return [
            param
            for item in node.elts
            if isinstance(item, ast.Dict)
            and (param := _parameter_from_dict(item, env))
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _parameters(node.left, env) + _parameters(node.right, env)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "with_evolution_params"
    ):
        values = _parameters(node.args[0] if node.args else None, env)
        prefix = _literal(node.args[1], env) if len(node.args) > 1 else None
        if not isinstance(prefix, str):
            return values
        keywords = {
            item.arg: _literal(item.value, env) for item in node.keywords if item.arg
        }
        values.extend(
            [
                Parameter(
                    f"{prefix}_evolution",
                    str(keywords.get("amount_label") or "Evolve"),
                    "float",
                    float(keywords.get("amount_default") or 0.0),
                    0.0,
                    1.0,
                ),
                Parameter(
                    f"{prefix}_evolution_speed",
                    str(keywords.get("speed_label") or "Evolve Speed"),
                    "float",
                    float(keywords.get("speed_default") or 1.0),
                    0.0,
                    5.0,
                ),
            ]
        )
        return values
    return []


def _sockets(node: ast.AST | None, env: dict[str, Any]) -> list[Socket]:
    value = _literal(node, env) if node is not None else None
    if not isinstance(value, list):
        return []
    sockets: list[Socket] = []
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        label = (
            item.get("text")
            or item.get("label")
            or item.get("attr_label")
            or item["name"]
        )
        sockets.append(Socket(item["name"], str(label)))
    return sockets


def _target_name(target: ast.expr) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def python_classes(path: Path) -> dict[str, dict[str, object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, object]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = [name for base in node.bases if (name := class_name(base))]
        init = next(
            (
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            ),
            None,
        )
        env: dict[str, Any] = {"CorruptionMode": corruption_mode_values()}
        label: str | None = None
        params: list[Parameter] = []
        inputs: list[Socket] = []
        outputs: list[Socket] = []
        finalized_generator = False
        common_parameters: list[Parameter] = []
        common_method = next(
            (
                item
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                and item.name == "_common_generator_params"
            ),
            None,
        )
        if common_method is not None:
            returned = next(
                (
                    item
                    for item in ast.walk(common_method)
                    if isinstance(item, ast.Return)
                ),
                None,
            )
            if returned is not None:
                common_parameters = _parameters(returned.value, env)
        if init is not None:
            for child in ast.walk(init):
                if isinstance(child, ast.Assign):
                    value = _literal(child.value, env)
                    for target in child.targets:
                        name = _target_name(target)
                        if not name:
                            continue
                        if name in {"params", "params_def"}:
                            extracted = _parameters(child.value, env)
                            if extracted:
                                params = extracted
                            env[name] = extracted
                        elif name == "input_defs":
                            inputs = _sockets(child.value, env)
                            env[name] = value
                        elif name == "output_defs":
                            outputs = _sockets(child.value, env)
                            env[name] = value
                        else:
                            env[name] = value
                if not isinstance(child, ast.Call):
                    continue
                function = child.func
                is_super_init = (
                    isinstance(function, ast.Attribute)
                    and function.attr == "__init__"
                    and isinstance(function.value, ast.Call)
                    and isinstance(function.value.func, ast.Name)
                    and function.value.func.id == "super"
                )
                if is_super_init:
                    if (
                        child.args
                        and isinstance(child.args[0], ast.Constant)
                        and isinstance(child.args[0].value, str)
                    ):
                        label = child.args[0].value
                    keyword_map = {
                        item.arg: item.value for item in child.keywords if item.arg
                    }
                    param_node = keyword_map.get("params") or (
                        child.args[2] if len(child.args) > 2 else None
                    )
                    extracted = _parameters(param_node, env)
                    if extracted:
                        params = extracted
                    inputs = _sockets(keyword_map.get("input_defs"), env) or inputs
                    outputs = _sockets(keyword_map.get("output_defs"), env) or outputs
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr == "_finalize_generator_params"
                ):
                    finalized_generator = True
        result[node.name] = {
            "bases": bases,
            "label": label,
            "parameters": params,
            "inputs": inputs,
            "outputs": outputs,
            "finalized_generator": finalized_generator,
            "common_parameters": common_parameters,
            "module": path.stem,
        }
    return result


def corruption_mode_values() -> dict[str, str]:
    tree = ast.parse((ROOT / "glitch_effects.py").read_text(encoding="utf-8"))
    enum_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "CorruptionMode"
        ),
        None,
    )
    if enum_class is None:
        return {}
    values: dict[str, str] = {}
    for node in enum_class.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = _literal(node.value, {})
        if isinstance(target, ast.Name) and isinstance(value, str):
            values[target.id] = value
    return values


def derives_from(
    name: str, ancestor: str, classes: dict[str, dict[str, object]]
) -> bool:
    seen: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for base in classes.get(current, {}).get("bases", []):
            if base == ancestor:
                return True
            pending.append(base)
    return False


def _inherited(
    name: str, field: str, classes: dict[str, dict[str, object]]
) -> Any:
    pending = [name]
    seen: set[str] = set()
    while pending:
        current = pending.pop(0)
        if current in seen:
            continue
        seen.add(current)
        info = classes.get(current, {})
        value = info.get(field)
        if value:
            return value
        pending.extend(info.get("bases", []))
    return None


def build_specs() -> list[NodeSpec]:
    category_map, menu_labels = categories()
    classes: dict[str, dict[str, object]] = {}
    for filename in ("effect_nodes.py", "effect_nodes_cuda.py"):
        classes.update(python_classes(ROOT / filename))
    available = {
        name
        for name in classes
        if name not in BASE_CLASSES
        and (
            derives_from(name, "GlitchEffectNode", classes)
            or derives_from(name, "MultiInputNode", classes)
            or derives_from(name, "GeneratorNode", classes)
            or name == "VideoMixerNode"
        )
    }
    node_category = {
        name: category for category, names in category_map.items() for name in names
    }
    common_generator = list(
        classes.get("GeneratorNode", {}).get("common_parameters", [])
    )
    common_glitch_effect = list(
        classes.get("GlitchEffectNode", {}).get("parameters", [])
    )
    specs: list[NodeSpec] = []
    for name in sorted(
        available, key=lambda item: (node_category.get(item, "Uncategorized"), item)
    ):
        info = classes[name]
        category = node_category.get(name, "Uncategorized")
        label = menu_labels.get(name) or info.get("label") or name.removesuffix("Node")
        generator = derives_from(name, "GeneratorNode", classes)
        multi_input = derives_from(name, "MultiInputNode", classes)
        glitch_effect = derives_from(name, "GlitchEffectNode", classes)
        params = list(_inherited(name, "parameters", classes) or [])
        if glitch_effect:
            existing = {param.name for param in params}
            params = [
                param for param in common_glitch_effect if param.name not in existing
            ] + params
        elif multi_input and not any(param.name == "enabled" for param in params):
            params.insert(0, Parameter("enabled", "Enabled", "bool", True))
        if generator and info.get("finalized_generator"):
            existing = {param.name for param in params}
            params.extend(
                param for param in common_generator if param.name not in existing
            )
        inputs = list(_inherited(name, "inputs", classes) or [])
        outputs = list(_inherited(name, "outputs", classes) or [])
        if name == "VideoMixerNode":
            inputs = [Socket("image_a", "Channel A"), Socket("image_b", "Channel B")]
            outputs = [Socket("image_out", "Mixed Out")]
        elif generator:
            outputs = outputs or [Socket("image_out", "Generated Output")]
        else:
            inputs = inputs or [Socket("image_in", "Image In")]
            outputs = outputs or [Socket("image_out", "Image Out")]
        specs.append(
            NodeSpec(
                backend_id=name,
                label=str(label),
                category=category,
                backend=(
                    "cuda" if info["module"] == "effect_nodes_cuda" else "cpu"
                ),
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                parameters=tuple(params),
            )
        )
    return specs


def rust_string(value: str) -> str:
    return (
        '"'
        + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        + '"'
    )


def _option(value: float | None) -> str:
    return "None" if value is None else f"Some({value!r})"


def _default(value: bool | int | float | str, kind: str) -> str:
    if kind == "bool":
        return f"ParameterDefault::Bool({str(bool(value)).lower()})"
    if kind == "int":
        return f"ParameterDefault::Int({int(value)})"
    if kind == "float":
        return f"ParameterDefault::Float({float(value)!r})"
    return f"ParameterDefault::Text({rust_string(str(value))})"


def render(specs: list[NodeSpec]) -> str:
    rows: list[str] = []
    for spec in specs:
        inputs = ", ".join(
            "SocketSpec { "
            f"id: {rust_string(socket.name)}, label: {rust_string(socket.label)}, "
            "data_type: SocketDataType::Video }"
            for socket in spec.inputs
        )
        outputs = ", ".join(
            "SocketSpec { "
            f"id: {rust_string(socket.name)}, label: {rust_string(socket.label)}, "
            "data_type: SocketDataType::Video }"
            for socket in spec.outputs
        )
        params = []
        for param in spec.parameters:
            choices = ", ".join(rust_string(choice) for choice in param.choices)
            params.append(
                "ParameterSpec { "
                f"id: {rust_string(param.name)}, label: {rust_string(param.label)}, "
                f"kind: ParameterKind::{param.kind.title()}, "
                f"default: {_default(param.default, param.kind)}, "
                f"minimum: {_option(param.minimum)}, maximum: {_option(param.maximum)}, "
                f"choices: &[{choices}] "
                "}"
            )
        rows.append(
            "    NodeSpec { "
            f"version: NODE_SPEC_VERSION, backend_id: {rust_string(spec.backend_id)}, "
            f"label: {rust_string(spec.label)}, category: {rust_string(spec.category)}, "
            f"backend: {rust_string(spec.backend)}, inputs: &[{inputs}], "
            f"outputs: &[{outputs}], parameters: &[{', '.join(params)}] "
            "},"
        )
    return "\n".join(
        [
            "// @generated by tools/export_rust_node_catalog.py; do not hand-edit.",
            "use super::{NodeSpec, ParameterDefault, ParameterKind, ParameterSpec, "
            "SocketDataType, SocketSpec, NODE_SPEC_VERSION};",
            "",
            "pub const NODES: &[NodeSpec] = &[",
            *rows,
            "];",
            "",
        ]
    )


def main() -> None:
    specs = build_specs()
    OUTPUT.write_text(render(specs), encoding="utf-8", newline="\n")
    print(
        f"wrote {len(specs)} version-{NODE_SPEC_VERSION} node specs "
        f"to {OUTPUT.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
