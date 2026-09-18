import torch

from wavestereo.deployment import export_onnx


def test_correlation_conv_reducer_matches_mean(monkeypatch):
    torch.manual_seed(7)
    left = torch.randn(1, 24, 5, 9)
    right = torch.randn(1, 24, 5, 9)
    monkeypatch.setattr(export_onnx, "_correlation_builder", "padstack")
    monkeypatch.setattr(export_onnx, "_correlation_chunk_size", 0)

    monkeypatch.setattr(export_onnx, "_correlation_reducer", "mean")
    expected = export_onnx.correlation_volume_onnx(left, right, 8)
    monkeypatch.setattr(export_onnx, "_correlation_reducer", "conv")
    actual = export_onnx.correlation_volume_onnx(left, right, 8)

    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)


def test_slice_geometry_pyramid_matches_average_pool(monkeypatch):
    torch.manual_seed(11)
    volume = torch.randn(1, 1, 8, 3, 5)

    monkeypatch.setattr(export_onnx, "_geometry_pyramid_builder", "pool")
    pooled = export_onnx.Geo_Encoding_Volume_onnx(volume, num_levels=2).geo_volume_pyramid
    monkeypatch.setattr(export_onnx, "_geometry_pyramid_builder", "slice")
    sliced = export_onnx.Geo_Encoding_Volume_onnx(volume, num_levels=2).geo_volume_pyramid

    assert torch.equal(sliced[0], pooled[0])
    assert torch.equal(sliced[1], pooled[1])
