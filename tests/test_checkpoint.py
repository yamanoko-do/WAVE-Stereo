import torch

from wavestereo.inference.checkpoint import extract_state_dict, strip_prefix_if_present


def test_extract_state_dict_from_model_key():
    state = {"layer.weight": torch.zeros(1)}
    assert extract_state_dict({"model": state}) is state


def test_strip_module_prefix():
    state = {"module.layer.weight": torch.zeros(1)}
    stripped = strip_prefix_if_present(state, "module.")
    assert "layer.weight" in stripped
