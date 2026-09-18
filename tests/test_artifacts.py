from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from wavestereo.artifacts import (
    BUNDLE_SCHEMA_VERSION,
    METADATA_FILENAME,
    ONNX_FILENAME,
    _PROFILE_GRAPH_OPTIONS,
    bundle_input_hw,
    bundle_input_shape,
    bundle_normalization,
    load_bundle_metadata,
    make_bundle_dir,
    make_spec_name,
    resolve_openvino_model,
    sha256_file,
    validate_profile_device,
)


HEIGHT = 180
WIDTH = 360
VALID_ITERS = 4
MAX_DISP = 192
SPEC_NAME = "180x360_iter4_maxdisp192"


def _write_tiny_onnx(path: Path, *, dtype=None, width=WIDTH):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    dtype = TensorProto.UINT8 if dtype is None else dtype
    shape = [1, HEIGHT, width, 3]
    left = helper.make_tensor_value_info("left_image", dtype, shape)
    right = helper.make_tensor_value_info("right_image", dtype, shape)
    output = helper.make_tensor_value_info(
        "disparity", TensorProto.FLOAT, [1, 1, HEIGHT, width]
    )
    nodes = [
        helper.make_node("Cast", ["left_image"], ["left_float"], to=TensorProto.FLOAT),
        helper.make_node(
            "Transpose", ["left_float"], ["left_nchw"], perm=[0, 3, 1, 2]
        ),
        helper.make_node(
            "Slice",
            ["left_nchw", "starts", "ends", "axes", "steps"],
            ["disparity"],
        ),
    ]
    initializers = [
        helper.make_tensor("starts", TensorProto.INT64, [1], [0]),
        helper.make_tensor("ends", TensorProto.INT64, [1], [1]),
        helper.make_tensor("axes", TensorProto.INT64, [1], [1]),
        helper.make_tensor("steps", TensorProto.INT64, [1], [1]),
    ]
    graph = helper.make_graph(
        nodes, "bundle-contract", [left, right], [output], initializers
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _metadata(profile="standard", *, onnx_sha256="0" * 64):
    preprocessing = {
        "standard": "reference",
        "intel_gpu": "lookup",
        "intel_npu": "affine",
    }[profile]
    allowed = {
        "standard": ["*"],
        "intel_gpu": ["GPU"],
        "intel_npu": ["NPU"],
    }[profile]
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "profile": profile,
        "allowed_devices": allowed,
        "model": {
            "name": "WAVEStereo",
            "valid_iters": VALID_ITERS,
            "max_disp": MAX_DISP,
            "corr_radius": 4,
            "corr_levels": 2,
            "hidden_dim": 64,
            "context_upsample_mode": "consistency_guided",
            "context_upsample_sigma": 8.0,
            "context_upsample_reference": (
                "npu_reduce_max" if profile == "intel_npu" else "argmax"
            ),
        },
        "input": {
            "names": ["left_image", "right_image"],
            "shape": [1, HEIGHT, WIDTH, 3],
            "dtype": "uint8",
            "layout": "NHWC",
            "color_order": "BGR",
            "preprocessing": {
                "location": "graph",
                "implementation": preprocessing,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "internal_padding": {
                "mode": "top_right_replicate",
                "divisible_by": 32,
                "top": 12,
                "right": 24,
            },
        },
        "output": {
            "name": "disparity",
            "shape": [1, 1, HEIGHT, WIDTH],
            "dtype": "float32",
            "units": "model_input_pixels_x",
        },
        "artifacts": {
            "onnx": {"filename": ONNX_FILENAME, "sha256": onnx_sha256}
        },
        "export": {
            "config": f"cfgs/export/{profile}.yaml",
            "model_config": "cfgs/wavestereo.yaml",
            "checkpoint": {"filename": "checkpoint.pth"},
            "requested_opset": 17,
            "effective_opset": 17,
            "graph": deepcopy(_PROFILE_GRAPH_OPTIONS[profile]),
        },
    }


def _write_bundle(root: Path, profile="standard", metadata=None):
    bundle = root / profile / SPEC_NAME
    bundle.mkdir(parents=True)
    _write_tiny_onnx(bundle / ONNX_FILENAME)
    if metadata is None:
        metadata = _metadata(
            profile, onnx_sha256=sha256_file(bundle / ONNX_FILENAME)
        )
    with (bundle / METADATA_FILENAME).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)
    return bundle


def _rewrite_metadata(bundle: Path, metadata):
    with (bundle / METADATA_FILENAME).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)


def test_canonical_bundle_path_is_profile_and_fixed_specification(tmp_path):
    assert make_spec_name(HEIGHT, WIDTH, VALID_ITERS, MAX_DISP) == SPEC_NAME
    assert make_bundle_dir(
        tmp_path, "intel_npu", HEIGHT, WIDTH, VALID_ITERS, MAX_DISP
    ) == tmp_path / "intel_npu" / SPEC_NAME


@pytest.mark.parametrize("profile", ["standard", "intel_gpu", "intel_npu"])
def test_schema_v2_contract_and_actual_onnx_are_strictly_validated(tmp_path, profile):
    bundle = _write_bundle(tmp_path, profile)
    metadata = load_bundle_metadata(bundle)

    assert bundle_input_shape(metadata) == (1, HEIGHT, WIDTH, 3)
    assert bundle_input_hw(metadata) == (HEIGHT, WIDTH)
    assert bundle_normalization(metadata) == (
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    model_path, loaded = resolve_openvino_model(bundle)
    assert model_path == bundle / ONNX_FILENAME
    assert loaded == metadata


def test_schema_v1_has_explicit_reexport_error(tmp_path):
    metadata = _metadata()
    metadata["schema_version"] = 1
    bundle = _write_bundle(tmp_path, metadata=metadata)
    with pytest.raises(ValueError, match="schema v1.*re-export"):
        load_bundle_metadata(bundle)


@pytest.mark.parametrize(
    ("profile", "accepted", "rejected"),
    [
        ("standard", "AUTO", None),
        ("intel_gpu", "GPU.0", "NPU"),
        ("intel_npu", "NPU", "GPU.0"),
    ],
)
def test_profile_device_constraints(tmp_path, profile, accepted, rejected):
    metadata = load_bundle_metadata(_write_bundle(tmp_path, profile))
    validate_profile_device(metadata, accepted)
    if rejected is not None:
        with pytest.raises(ValueError, match="does not allow"):
            validate_profile_device(metadata, rejected)


def test_checksum_mismatch_is_rejected(tmp_path):
    bundle = _write_bundle(tmp_path)
    with (bundle / ONNX_FILENAME).open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_bundle_metadata(bundle)


def test_onnx_dtype_is_checked_against_metadata(tmp_path):
    onnx = pytest.importorskip("onnx")
    bundle = _write_bundle(tmp_path)
    _write_tiny_onnx(bundle / ONNX_FILENAME, dtype=onnx.TensorProto.FLOAT)
    metadata = _metadata(onnx_sha256=sha256_file(bundle / ONNX_FILENAME))
    _rewrite_metadata(bundle, metadata)
    with pytest.raises(ValueError, match="dtype uint8"):
        load_bundle_metadata(bundle)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda m: m["input"].update(dtype="float32"), "dtype"),
        (lambda m: m["input"].update(layout="NCHW"), "layout"),
        (lambda m: m["input"].update(color_order="RGB"), "color_order"),
        (
            lambda m: m["input"]["internal_padding"].update(top=0),
            "padding",
        ),
        (lambda m: m["output"].update(units="pixels"), "output metadata"),
        (lambda m: m["artifacts"].update(extra={}), "only 'onnx'"),
        (lambda m: m["export"].update(graph={}), "export.graph"),
    ],
)
def test_invalid_contract_fields_are_rejected(tmp_path, mutator, message):
    bundle = _write_bundle(tmp_path)
    metadata = deepcopy(load_bundle_metadata(bundle))
    mutator(metadata)
    _rewrite_metadata(bundle, metadata)
    with pytest.raises(ValueError, match=message):
        load_bundle_metadata(bundle)


def test_metadata_and_directory_spec_must_match(tmp_path):
    bundle = _write_bundle(tmp_path)
    metadata = deepcopy(load_bundle_metadata(bundle))
    metadata["model"]["valid_iters"] = 5
    _rewrite_metadata(bundle, metadata)
    with pytest.raises(ValueError, match="directory"):
        load_bundle_metadata(bundle)


def test_resolver_requires_bundle_directory(tmp_path):
    standalone = tmp_path / ONNX_FILENAME
    standalone.touch()
    with pytest.raises(NotADirectoryError, match="bundle directory"):
        resolve_openvino_model(standalone)
