import pytest
import torch

from wavestereo.config import load_config
from wavestereo.model import WAVEStereo


def test_model_constructs_from_public_config():
    pytest.importorskip("timm")
    cfg = load_config("cfgs/wavestereo.yaml")
    model = WAVEStereo(cfg.MODEL)
    assert model.max_disp == cfg.MODEL.MAX_DISP
