<div align="center">

<h1>WAVE-Stereo: Warp-Aligned Volume Encoding for Stereo Matching</h1>

</div>

<div align="center">

[![arXiv](https://img.shields.io/static/v1?label=Paper&message=arXiv&color=lightgrey&logo=arxiv)](https://arxiv.org/abs/2607.13674)
[![HuggingFace](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange)](https://huggingface.co/Yama222/WAVE-Stereo/tree/main)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

</div>

## Install

- Clone this repository:

```bash
git clone https://github.com/yamanoko-do/WAVE-Stereo.git
cd WAVE-Stereo
```

- Create the environment:

```bash
conda create -n wavestereo python=3.12 -y
conda activate wavestereo
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## TensorRT Environment Setup (Optional)

- Install the local TensorRT repository package:

```bash
wget https://developer.download.nvidia.com/compute/tensorrt/10.11.0/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb

apt install -y ./nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb
```

- Add the GPG key and install the TensorRT toolkit:

```bash
cp /var/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9/nv-tensorrt-local-5BF87A98-keyring.gpg /usr/share/keyrings/

apt update

apt install -y libnvinfer-bin
echo 'alias trtexec="/usr/src/tensorrt/bin/trtexec"' >> ~/.bashrc
source ~/.bashrc
trtexec --version
```

- Install the Python bindings:

```bash
pip install -v --no-deps tensorrt-cu12==10.11.0.33 tensorrt-cu12-bindings==10.11.0.33 tensorrt-cu12-libs==10.11.0.33
```

## Quick Start

### Inference

- Run inference on a stereo image pair:

```bash
python tools/infer.py \
  --config ./cfgs/wavestereo.yaml \
  --weights /path2ckpt/zeroshot.pth \
  --left ./assets/explorer_20-41-07_left.png \
  --right ./assets/explorer_20-41-07_right.png \
  --output ./outputs/demo
```

- Run realtime inference on a camera:

```bash
python tools/infercam.py \
  --config ./cfgs/wavestereo.yaml \
  --weights /path2ckpt/zeroshot.pth \
  --cam_file ./cfgs/camera/zed_calib/hd720 \
  --zfar 9
```

### TensorRT

- Export ONNX. For a 720p camera stream, use an input resolution divisible by 32. The following example exports a model with 4 refinement iterations:

```bash
python tools/export_onnx.py \
  --config cfgs/wavestereo.yaml \
  --weights /path2ckpt/zeroshot.pth \
  --save_path outputs/onnx \
  --height 736 \
  --width 1280 \
  --valid_iters 4
```

- Build the TensorRT inference engine:

```bash
trtexec --onnx=outputs/onnx/wavestereo_736x1280_iter4.onnx --saveEngine=outputs/onnx/wavestereo_736x1280_iter4.engine --fp16
```

- Run realtime camera inference with the TensorRT engine:

```bash
python tools/infercam.py \
  --trt \
  --onnx-dir outputs/onnx \
  --cam_file ./cfgs/camera/zed_calib/hd720
```

## Citation

If you use WAVE-Stereo, please cite the paper:

```bibtex
@article{wavestereo2026,
  title={WAVE-Stereo: Warp-Aligned Volume Encoding for Stereo Matching},
  author={Zehan Liu and Yage He},
  journal={arXiv preprint arXiv:2607.13674},
  year={2026}
}
```

## License

This project is released under the Apache License 2.0.

## Acknowledgement

This project is based on [IGEV++](https://github.com/gangweiX/IGEV-plusplus), [LightStereo](https://github.com/XiandaGuo/OpenStereo), [WAFT-Stereo](https://github.com/princeton-vl/WAFT-Stereo), and [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo). We thank the original authors for their excellent work.
