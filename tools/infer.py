#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wavestereo.inference.predictor import WAVEStereoPredictor
from wavestereo.inference.visualization import save_outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Run WAVEStereo PyTorch inference on one stereo pair")
    parser.add_argument("--config", default="cfgs/wavestereo.yaml", help="Path to config YAML")
    parser.add_argument("--weights", default=None, help="Path to model checkpoint")
    parser.add_argument("--left", required=True, help="Path to left image")
    parser.add_argument("--right", required=True, help="Path to right image")
    parser.add_argument("--output", default="outputs/disparity", help="Output prefix or directory")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = WAVEStereoPredictor(
        args.config,
        weights=args.weights,
        device=args.device,
    )
    disparity = predictor.predict(args.left, args.right)
    output = Path(args.output)
    if output.suffix:
        output = output.with_suffix("")
    elif output.is_dir() or str(args.output).endswith("/"):
        output = output / "disparity"
    save_outputs(output, disparity, max_disp=None, save_npy=True)
    print(f"[INFO] Saved outputs with prefix: {output}")


if __name__ == "__main__":
    main()
