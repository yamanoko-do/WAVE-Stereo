from pathlib import Path

import numpy as np

from wavestereo.inference.visualization import save_outputs


def test_save_outputs_appends_extensions_to_dotted_prefix(tmp_path):
    prefix = tmp_path / "GPU.0_result"
    disparity = np.zeros((2, 3), dtype=np.float32)

    save_outputs(prefix, disparity, max_disp=192, save_npy=True)

    assert Path(f"{prefix}.npy").is_file()
    assert Path(f"{prefix}.png").is_file()
