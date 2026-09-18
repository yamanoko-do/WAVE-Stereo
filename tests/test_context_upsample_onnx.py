from collections import Counter

import pytest
import torch

from wavestereo.layers.disp_refinement import context_upsample


class _ContextUpsampleWrapper(torch.nn.Module):
    def __init__(self, mode, reference_mode="argmax"):
        super().__init__()
        self.mode = mode
        self.reference_mode = reference_mode

    def forward(self, disparity, weights):
        return context_upsample(
            disparity,
            weights,
            mode=self.mode,
            consistency_sigma=2.0,
            reference_mode=self.reference_mode,
        )


def test_onnx_graph_uses_selected_context_upsample_implementation(tmp_path):
    onnx = pytest.importorskip("onnx")
    disparity = torch.rand(1, 1, 3, 5)
    weights = torch.softmax(torch.randn(1, 9, 12, 20), dim=1)
    operator_counts = {}

    cases = {
        "standard": ("standard", "argmax"),
        "consistency_guided": ("consistency_guided", "argmax"),
        "consistency_guided_npu": ("consistency_guided", "npu_reduce_max"),
    }
    for name, (mode, reference_mode) in cases.items():
        path = tmp_path / f"{name}.onnx"
        torch.onnx.export(
            _ContextUpsampleWrapper(mode, reference_mode),
            (disparity, weights),
            path,
            input_names=["disparity", "weights"],
            output_names=["upsampled_disparity"],
            opset_version=17,
            dynamo=False,
        )
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        operator_counts[name] = Counter(
            node.op_type for node in graph.graph.node
        )

    assert operator_counts["standard"]["Exp"] == 0
    assert operator_counts["standard"]["ArgMax"] == 0
    assert operator_counts["standard"]["GatherElements"] == 0
    assert operator_counts["consistency_guided"]["Exp"] == 1
    assert operator_counts["consistency_guided"]["ArgMax"] == 1
    assert operator_counts["consistency_guided"]["GatherElements"] == 1
    assert operator_counts["consistency_guided_npu"]["Exp"] == 1
    assert operator_counts["consistency_guided_npu"]["ArgMax"] == 0
    assert operator_counts["consistency_guided_npu"]["GatherElements"] == 0
    assert operator_counts["consistency_guided_npu"]["ReduceMax"] == 2
