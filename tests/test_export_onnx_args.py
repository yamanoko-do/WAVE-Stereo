from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import yaml

from wavestereo.deployment import export_onnx
from wavestereo.deployment.export_onnx import (
    BgrUint8OnnxInput,
    load_export_config,
    parse_args,
)


def test_export_cli_contains_only_config_and_optional_weights():
    args = parse_args(
        [
            "--export-config",
            "cfgs/export/intel_npu.yaml",
            "--weights",
            "/models/checkpoint.pth",
        ]
    )
    assert vars(args) == {
        "export_config": "cfgs/export/intel_npu.yaml",
        "weights": "/models/checkpoint.pth",
    }

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--export-config",
                "cfgs/export/intel_npu.yaml",
                "--height",
                "720",
            ]
        )


def test_export_config_is_required():
    with pytest.raises(SystemExit):
        parse_args([])


def test_three_controlled_profiles_share_model_semantics():
    configs = {
        name: load_export_config(f"cfgs/export/{name}.yaml")
        for name in ("standard", "intel_gpu", "intel_npu")
    }
    semantics = {
        (
            config.height,
            config.width,
            config.valid_iters,
            config.max_disp,
            config.context_upsample_mode,
            config.context_upsample_sigma,
        )
        for config in configs.values()
    }
    assert semantics == {(180, 360, 4, 192, "consistency_guided", 8.0)}
    assert configs["standard"].preprocessing == "reference"
    assert configs["intel_gpu"].preprocessing == "lookup"
    assert configs["intel_npu"].preprocessing == "affine"
    assert configs["standard"].graph == export_onnx._STANDARD_GRAPH_OPTIONS
    assert configs["intel_gpu"].graph == export_onnx._INTEL_GPU_GRAPH_OPTIONS
    assert configs["intel_npu"].graph == export_onnx._INTEL_NPU_GRAPH_OPTIONS


def test_export_config_paths_are_resolved_from_repo_root(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    (root / "cfgs/export").mkdir(parents=True)
    (root / "cfgs/model.yaml").write_text("MODEL: {}\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "profile": "standard",
        "model_config": "cfgs/model.yaml",
        "output_root": "outputs/onnx",
        "input": {"height": 180, "width": 360},
        "model": {
            "valid_iters": 4,
            "max_disp": 192,
            "context_upsample_mode": "consistency_guided",
            "context_upsample_sigma": 8.0,
        },
        "onnx": {"opset": 17, "preprocessing": "reference"},
        "graph": deepcopy(export_onnx._STANDARD_GRAPH_OPTIONS),
    }
    path = root / "cfgs/export/standard.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(export_onnx, "REPO_ROOT", root)
    monkeypatch.chdir(tmp_path)

    loaded = load_export_config("cfgs/export/standard.yaml")
    assert loaded.model_config == root / "cfgs/model.yaml"
    assert loaded.output_root == root / "outputs/onnx"
    assert loaded.path_relative == "cfgs/export/standard.yaml"


def test_profile_cannot_silently_drift_from_graph_or_preprocessing(
    monkeypatch, tmp_path
):
    root = tmp_path / "repo"
    (root / "cfgs/export").mkdir(parents=True)
    (root / "cfgs/wavestereo.yaml").write_text("MODEL: {}\n", encoding="utf-8")
    source = yaml.safe_load(
        (export_onnx.REPO_ROOT / "cfgs/export/intel_npu.yaml").read_text(
            encoding="utf-8"
        )
    )
    monkeypatch.setattr(export_onnx, "REPO_ROOT", root)

    source["graph"]["geometry_sampler"] = "gather"
    path = root / "cfgs/export/intel_npu.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="controlled preset"):
        load_export_config("cfgs/export/intel_npu.yaml")

    source["graph"] = deepcopy(export_onnx._INTEL_NPU_GRAPH_OPTIONS)
    source["onnx"]["preprocessing"] = "lookup"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="requires ONNX preprocessing 'affine'"):
        load_export_config("cfgs/export/intel_npu.yaml")


class _Capture(torch.nn.Module):
    def forward(self, left, right):
        return left + right * 0.0


class _StereoOutput(torch.nn.Module):
    def forward(self, left, right):
        return left[:, :1] + right[:, :1]


@pytest.mark.parametrize(
    ("implementation", "max_error"),
    [("reference", 1e-6), ("lookup", 1e-6), ("affine", 1e-5)],
)
def test_baked_bgr_uint8_preprocessing_matches_host_normalization(
    implementation, max_error
):
    torch.manual_seed(9)
    image = torch.randint(0, 256, (1, 11, 17, 3), dtype=torch.uint8)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    wrapper = BgrUint8OnnxInput(_Capture(), implementation, mean, std)
    actual = wrapper(image, image)

    expected = torch.stack(
        (image[..., 2], image[..., 1], image[..., 0]), dim=1
    ).float()
    expected = expected / 255.0
    expected = (expected - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(
        std
    ).view(1, 3, 1, 1)
    torch.testing.assert_close(actual, expected, atol=max_error, rtol=0.0)


@pytest.mark.parametrize(
    ("implementation", "required_ops", "forbidden_ops"),
    [
        ("reference", {"Div", "Sub"}, set()),
        ("lookup", {"Gather"}, {"Div", "Sub"}),
        ("affine", {"Mul", "Add"}, {"Div", "Sub"}),
    ],
)
def test_exported_preprocessing_has_static_uint8_nhwc_contract_without_transpose(
    tmp_path, implementation, required_ops, forbidden_ops
):
    onnx = pytest.importorskip("onnx")
    image = torch.randint(0, 256, (1, 5, 7, 3), dtype=torch.uint8)
    path = tmp_path / f"{implementation}.onnx"
    torch.onnx.export(
        BgrUint8OnnxInput(
            _StereoOutput(),
            implementation,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
        (image, image),
        path,
        input_names=["left_image", "right_image"],
        output_names=["disparity"],
        opset_version=17,
    )
    graph = onnx.load(path)
    operators = {node.op_type for node in graph.graph.node}

    assert [value.name for value in graph.graph.input] == [
        "left_image",
        "right_image",
    ]
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.UINT8
        for value in graph.graph.input
    )
    assert all(
        [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        == [1, 5, 7, 3]
        for value in graph.graph.input
    )
    assert "Transpose" not in operators
    assert required_ops <= operators
    assert not (forbidden_ops & operators)
