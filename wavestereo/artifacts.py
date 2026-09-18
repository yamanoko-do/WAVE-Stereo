from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import yaml


BUNDLE_SCHEMA_VERSION = 2
METADATA_FILENAME = "metadata.yaml"
ONNX_FILENAME = "model.onnx"
TENSORRT_ENGINE_FILENAME = "model.engine"
EXPORT_PROFILES = frozenset({"standard", "intel_gpu", "intel_npu"})
CONTEXT_UPSAMPLE_MODES = frozenset({"standard", "consistency_guided"})
CONTEXT_UPSAMPLE_REFERENCE_MODES = frozenset({"argmax", "npu_reduce_max"})
PREPROCESSING_IMPLEMENTATIONS = frozenset({"reference", "lookup", "affine"})

_PROFILE_PREPROCESSING = {
    "standard": "reference",
    "intel_gpu": "lookup",
    "intel_npu": "affine",
}
_PROFILE_ALLOWED_DEVICES = {
    "standard": ["*"],
    "intel_gpu": ["GPU"],
    "intel_npu": ["NPU"],
}
_STANDARD_GRAPH_OPTIONS = {
    "correlation_chunk_size": 0,
    "correlation_builder": "padstack",
    "correlation_reducer": "mean",
    "geometry_sampler": "gather",
    "geometry_pyramid_builder": "pool",
    "context_splitter": "split",
    "dpt_conv_before_upsample": False,
    "batch_stereo_backbone": False,
    "hoist_warp_left": False,
    "late_image_stems": False,
    "polyphase_deconv": "none",
    "stem_groupnorm_scope": "none",
    "dpt_groupnorm": False,
}
_PROFILE_GRAPH_OPTIONS = {
    "standard": _STANDARD_GRAPH_OPTIONS,
    "intel_gpu": _STANDARD_GRAPH_OPTIONS
    | {
        "correlation_builder": "matmul",
        "geometry_sampler": "padded",
        "batch_stereo_backbone": True,
        "polyphase_deconv": "all",
        "stem_groupnorm_scope": "first",
        "dpt_groupnorm": True,
    },
    "intel_npu": _STANDARD_GRAPH_OPTIONS
    | {
        "correlation_reducer": "conv",
        "geometry_sampler": "grid",
        "geometry_pyramid_builder": "slice",
        "context_splitter": "slice",
    },
}


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Model bundle {field} must be a positive integer: {value!r}")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Model bundle {field} must be a non-negative integer: {value!r}"
        )
    return value


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Model bundle {field} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Model bundle {field} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"Model bundle {field} must be positive and finite")
    return result


def _validate_profile(profile: Any) -> str:
    if profile not in EXPORT_PROFILES:
        supported = ", ".join(sorted(EXPORT_PROFILES))
        raise ValueError(
            f"Unsupported export profile {profile!r}; expected one of: {supported}"
        )
    return str(profile)


def make_spec_name(height: int, width: int, valid_iters: int, max_disp: int) -> str:
    """Return the canonical directory name for one fixed model specification."""
    height = _positive_int(height, "height")
    width = _positive_int(width, "width")
    valid_iters = _positive_int(valid_iters, "valid_iters")
    max_disp = _positive_int(max_disp, "max_disp")
    return f"{height}x{width}_iter{valid_iters}_maxdisp{max_disp}"


def make_bundle_dir(
    save_path: str | Path,
    profile: str,
    height: int,
    width: int,
    valid_iters: int,
    max_disp: int,
) -> Path:
    """Resolve ``<save_path>/<profile>/<fixed-specification>``."""
    return Path(save_path) / _validate_profile(profile) / make_spec_name(
        height, width, valid_iters, max_disp
    )


def bundle_input_shape(metadata: dict[str, Any]) -> tuple[int, int, int, int]:
    input_metadata = metadata.get("input")
    if not isinstance(input_metadata, dict):
        raise ValueError("Model bundle is missing input metadata")
    shape = input_metadata.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        raise ValueError(f"Invalid model bundle input shape: {shape!r}")
    converted = tuple(_positive_int(value, "input.shape") for value in shape)
    if converted[0] != 1 or converted[3] != 3:
        raise ValueError(
            f"Model bundle input shape must be static [1, H, W, 3]: {shape!r}"
        )
    return converted


def bundle_input_hw(metadata: dict[str, Any]) -> tuple[int, int]:
    """Return the logical model input ``(height, width)``."""
    _, height, width, _ = bundle_input_shape(metadata)
    return height, width


def bundle_normalization(metadata: dict[str, Any]) -> tuple[list[float], list[float]]:
    input_metadata = metadata.get("input")
    if not isinstance(input_metadata, dict):
        raise ValueError("Model bundle is missing input metadata")
    preprocessing = input_metadata.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("Model bundle is missing graph preprocessing metadata")
    mean = preprocessing.get("mean")
    std = preprocessing.get("std")
    if (
        not isinstance(mean, list)
        or not isinstance(std, list)
        or len(mean) != 3
        or len(std) != 3
        or any(isinstance(value, bool) for value in mean + std)
    ):
        raise ValueError(
            "Model bundle preprocessing must contain three-channel mean/std"
        )
    try:
        mean_values = [float(value) for value in mean]
        std_values = [float(value) for value in std]
    except (TypeError, ValueError) as exc:
        raise ValueError("Model bundle preprocessing mean/std must be numeric") from exc
    if not all(math.isfinite(value) for value in mean_values + std_values):
        raise ValueError("Model bundle preprocessing mean/std must be finite")
    if any(value <= 0.0 for value in std_values):
        raise ValueError("Model bundle preprocessing std values must be positive")
    return mean_values, std_values


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(checkpoint: Any) -> None:
    if checkpoint is None:
        return
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"filename"}:
        raise ValueError(
            "Model bundle export.checkpoint must be null or contain only filename"
        )
    filename = checkpoint.get("filename")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError(
            "Model bundle checkpoint filename must not contain a directory"
        )


def _validate_onnx_contract(
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
    output_shape: tuple[int, int, int, int],
) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The onnx package is required to validate a model bundle"
        ) from exc

    try:
        graph = onnx.load(onnx_path, load_external_data=False)
        onnx.checker.check_model(graph)
    except Exception as exc:
        raise ValueError(f"Invalid ONNX artifact: {onnx_path}") from exc

    graph_inputs = list(graph.graph.input)
    if [value.name for value in graph_inputs] != ["left_image", "right_image"]:
        raise ValueError(
            "ONNX inputs must be exactly ['left_image', 'right_image']"
        )
    for value in graph_inputs:
        tensor = value.type.tensor_type
        dims = tuple(dim.dim_value for dim in tensor.shape.dim)
        if tensor.elem_type != onnx.TensorProto.UINT8:
            raise ValueError(f"ONNX input {value.name!r} must have dtype uint8")
        if dims != input_shape:
            raise ValueError(
                f"ONNX input {value.name!r} shape {dims!r} does not match "
                f"metadata {input_shape!r}"
            )

    graph_outputs = list(graph.graph.output)
    if [value.name for value in graph_outputs] != ["disparity"]:
        raise ValueError("ONNX output must be exactly ['disparity']")
    tensor = graph_outputs[0].type.tensor_type
    dims = tuple(dim.dim_value for dim in tensor.shape.dim)
    if tensor.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("ONNX output 'disparity' must have dtype float32")
    if dims != output_shape:
        raise ValueError(
            f"ONNX output shape {dims!r} does not match metadata {output_shape!r}"
        )


def validate_profile_device(metadata: dict[str, Any], device: str) -> None:
    """Reject an OpenVINO device that is outside the exported profile contract."""
    allowed = metadata.get("allowed_devices")
    if allowed == ["*"]:
        return
    normalized = str(device).strip().upper()
    if not isinstance(allowed, list) or not any(
        normalized == item or normalized.startswith(f"{item}.") for item in allowed
    ):
        raise ValueError(
            f"Export profile {metadata.get('profile')!r} does not allow OpenVINO "
            f"device {device!r}; allowed devices: {allowed!r}"
        )


def load_bundle_metadata(model_dir: str | Path) -> dict[str, Any]:
    """Load schema-v2 metadata and verify it against the exact ONNX bytes."""
    model_dir = Path(model_dir)
    metadata_path = model_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model bundle metadata not found: {metadata_path}")

    try:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid model bundle metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid model bundle metadata: {metadata_path}")

    schema_version = metadata.get("schema_version")
    if schema_version == 1 and not isinstance(schema_version, bool):
        raise ValueError(
            "Model bundle schema v1 is no longer supported; re-export the bundle "
            "with the current tools/export_onnx.py"
        )
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported model bundle schema in {metadata_path}: {schema_version!r}; "
            "re-export the bundle"
        )

    required_sections = {
        "schema_version",
        "profile",
        "allowed_devices",
        "model",
        "input",
        "output",
        "artifacts",
        "export",
    }
    if set(metadata) != required_sections:
        missing = sorted(required_sections - set(metadata))
        extra = sorted(set(metadata) - required_sections)
        raise ValueError(
            f"Invalid model bundle top-level sections: missing={missing}, extra={extra}"
        )

    profile = _validate_profile(metadata["profile"])
    if model_dir.parent.name != profile:
        raise ValueError(
            f"Model bundle profile directory must be {profile!r}: {model_dir.parent}"
        )
    expected_devices = _PROFILE_ALLOWED_DEVICES[profile]
    if metadata["allowed_devices"] != expected_devices:
        raise ValueError(
            f"Model bundle allowed_devices must be {expected_devices!r} for {profile}"
        )

    model_metadata = metadata["model"]
    model_fields = {
        "name",
        "valid_iters",
        "max_disp",
        "corr_radius",
        "corr_levels",
        "hidden_dim",
        "context_upsample_mode",
        "context_upsample_sigma",
        "context_upsample_reference",
    }
    if not isinstance(model_metadata, dict) or set(model_metadata) != model_fields:
        raise ValueError("Missing or invalid 'model' section in model bundle")
    if model_metadata.get("name") != "WAVEStereo":
        raise ValueError("Model bundle model.name must be 'WAVEStereo'")
    valid_iters = _positive_int(model_metadata.get("valid_iters"), "valid_iters")
    max_disp = _positive_int(model_metadata.get("max_disp"), "max_disp")
    _positive_int(model_metadata.get("corr_radius"), "corr_radius")
    _positive_int(model_metadata.get("corr_levels"), "corr_levels")
    _positive_int(model_metadata.get("hidden_dim"), "hidden_dim")
    mode = model_metadata.get("context_upsample_mode")
    if mode not in CONTEXT_UPSAMPLE_MODES:
        raise ValueError(f"Unsupported model bundle context_upsample_mode: {mode!r}")
    _positive_float(
        model_metadata.get("context_upsample_sigma"), "context_upsample_sigma"
    )
    reference = model_metadata.get("context_upsample_reference")
    if reference not in CONTEXT_UPSAMPLE_REFERENCE_MODES:
        raise ValueError(
            f"Unsupported model bundle context_upsample_reference: {reference!r}"
        )
    expected_reference = "npu_reduce_max" if profile == "intel_npu" else "argmax"
    if reference != expected_reference:
        raise ValueError(
            f"Export profile {profile!r} requires context reference "
            f"{expected_reference!r}"
        )

    input_metadata = metadata["input"]
    input_fields = {
        "names",
        "shape",
        "dtype",
        "layout",
        "color_order",
        "preprocessing",
        "internal_padding",
    }
    if not isinstance(input_metadata, dict) or set(input_metadata) != input_fields:
        raise ValueError("Missing or invalid 'input' section in model bundle")
    input_shape = bundle_input_shape(metadata)
    if input_metadata.get("names") != ["left_image", "right_image"]:
        raise ValueError(
            "Model bundle input.names must be ['left_image', 'right_image']"
        )
    if input_metadata.get("dtype") != "uint8":
        raise ValueError("Model bundle input.dtype must be 'uint8'")
    if input_metadata.get("layout") != "NHWC":
        raise ValueError("Model bundle input.layout must be 'NHWC'")
    if input_metadata.get("color_order") != "BGR":
        raise ValueError("Model bundle input.color_order must be 'BGR'")
    preprocessing = input_metadata.get("preprocessing")
    if not isinstance(preprocessing, dict) or set(preprocessing) != {
        "location",
        "implementation",
        "mean",
        "std",
    }:
        raise ValueError("Model bundle input.preprocessing must be a mapping")
    if preprocessing.get("location") != "graph":
        raise ValueError("Model bundle preprocessing.location must be 'graph'")
    implementation = preprocessing.get("implementation")
    if implementation not in PREPROCESSING_IMPLEMENTATIONS:
        raise ValueError(
            f"Unsupported graph preprocessing implementation: {implementation!r}"
        )
    if implementation != _PROFILE_PREPROCESSING[profile]:
        raise ValueError(
            f"Export profile {profile!r} requires preprocessing "
            f"{_PROFILE_PREPROCESSING[profile]!r}"
        )
    bundle_normalization(metadata)

    padding = input_metadata.get("internal_padding")
    if not isinstance(padding, dict) or set(padding) != {
        "mode",
        "divisible_by",
        "top",
        "right",
    }:
        raise ValueError("Model bundle input.internal_padding must be a mapping")
    if padding.get("mode") != "top_right_replicate":
        raise ValueError(
            "Model bundle internal_padding.mode must be 'top_right_replicate'"
        )
    if padding.get("divisible_by") != 32:
        raise ValueError("Model bundle internal_padding.divisible_by must be 32")
    pad_top = _nonnegative_int(padding.get("top"), "internal_padding.top")
    pad_right = _nonnegative_int(padding.get("right"), "internal_padding.right")
    _, height, width, _ = input_shape
    if pad_top != (-height) % 32 or pad_right != (-width) % 32:
        raise ValueError("Model bundle internal /32 padding does not match input shape")

    output_metadata = metadata["output"]
    expected_output = {
        "name": "disparity",
        "shape": [1, 1, height, width],
        "dtype": "float32",
        "units": "model_input_pixels_x",
    }
    if output_metadata != expected_output:
        raise ValueError(f"Model bundle output metadata must be {expected_output!r}")

    artifacts = metadata["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"onnx"}:
        raise ValueError("Model bundle artifacts must contain only 'onnx'")
    onnx_metadata = artifacts["onnx"]
    if (
        not isinstance(onnx_metadata, dict)
        or set(onnx_metadata) != {"filename", "sha256"}
        or onnx_metadata.get("filename") != ONNX_FILENAME
    ):
        raise ValueError(
            "Model bundle artifacts.onnx must contain model.onnx and its SHA-256"
        )
    expected_sha256 = onnx_metadata.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise ValueError("Model bundle ONNX SHA-256 must be lowercase hexadecimal")

    export_metadata = metadata["export"]
    export_fields = {
        "config",
        "model_config",
        "checkpoint",
        "requested_opset",
        "effective_opset",
        "graph",
    }
    if not isinstance(export_metadata, dict) or set(export_metadata) != export_fields:
        raise ValueError("Missing or invalid 'export' section in model bundle")
    _positive_int(export_metadata.get("requested_opset"), "export.requested_opset")
    _positive_int(export_metadata.get("effective_opset"), "export.effective_opset")
    if export_metadata.get("graph") != _PROFILE_GRAPH_OPTIONS[profile]:
        raise ValueError(
            f"Model bundle export.graph does not match profile {profile!r}"
        )
    for field in ("config", "model_config"):
        value = export_metadata.get(field)
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError(f"Model bundle export.{field} must be repo-relative")
    _validate_checkpoint(export_metadata.get("checkpoint"))

    expected_name = make_spec_name(height, width, valid_iters, max_disp)
    if model_dir.name != expected_name:
        raise ValueError(
            f"Model bundle directory must match its metadata: expected "
            f"{expected_name!r}, got {model_dir.name!r}"
        )

    onnx_path = model_dir / ONNX_FILENAME
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Model bundle artifact not found: {onnx_path}")
    actual_sha256 = sha256_file(onnx_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"ONNX SHA-256 mismatch for {onnx_path}; metadata and model do not match"
        )
    _validate_onnx_contract(
        onnx_path,
        input_shape=input_shape,
        output_shape=(1, 1, height, width),
    )
    return metadata


def resolve_bundle_artifact(
    model_dir: str | Path,
    metadata: dict[str, Any],
    artifact: str,
    *,
    must_exist: bool = True,
) -> Path:
    if artifact != "onnx":
        raise ValueError(
            f"Unknown schema-v2 model bundle artifact {artifact!r}; only 'onnx' is supported"
        )
    artifact_metadata = metadata.get("artifacts", {}).get("onnx")
    if not isinstance(artifact_metadata, dict):
        raise ValueError("Model bundle is missing ONNX artifact metadata")
    path = Path(model_dir) / ONNX_FILENAME
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Model bundle artifact not found: {path}")
    return path


def resolve_openvino_model(
    model_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(
            f"OpenVINO model path must be a model bundle directory: {model_dir}"
        )
    metadata = load_bundle_metadata(model_dir)
    return resolve_bundle_artifact(model_dir, metadata, "onnx"), metadata


def resolve_tensorrt_engine(
    model_dir: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Keep the dormant API importable while schema v2 has no TRT artifact."""
    raise RuntimeError(
        "TensorRT is not part of the schema-v2 public deployment contract; "
        "use a future tensorrt_cuda export profile"
    )
