# v4l-background-buster

Real-time AI background removal for Linux virtual cameras.

A standalone C++ application that captures video from a USB camera via V4L2, runs [Robust Video Matting (RVM)](https://github.com/PeterL1n/RobustVideoMatting) inference with TensorRT FP16, and outputs to a v4l2loopback virtual camera. Any application — OBS, Discord, Zoom, Teams — can use the output as a normal camera device. No plugins required.

**Target**: ~5-7ms end-to-end latency at 1080p60 on NVIDIA RTX GPUs.

## How It Works

```
USB Camera (/dev/video0)
    → V4L2 MMAP capture
    → CUDA upload (async)
    → Preprocess kernel (BGRA → RGB FP16)
    → TensorRT RVM inference (FP16, recurrent)
    → Composite kernel (alpha blend)
    → Color convert kernel (BGRA → YUYV)
    → CUDA download (async)
    → v4l2loopback write (/dev/video10)
    → Any app sees a normal camera
```

## Requirements

- **GPU**: NVIDIA RTX (tested on RTX 2060 Super)
- **OS**: Linux (Ubuntu 24.04)
- **NVIDIA stack**: Driver 590+, CUDA 12.x, TensorRT 10.x
- **V4L2**: libv4l-dev, v4l-utils, v4l2loopback-dkms

### Install Dependencies

```bash
# Build tools
sudo apt install build-essential cmake

# TensorRT (requires NVIDIA apt repo)
sudo apt install tensorrt libnvinfer-dev libnvonnxparsers-dev

# V4L2
sudo apt install libv4l-dev v4l-utils v4l2loopback-dkms v4l2loopback-utils
```

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Load v4l2loopback
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI-Camera" exclusive_caps=1

# Run
./rvm-vcam -i /dev/video0 -o /dev/video10
```

### Options

```
  -i, --input DEVICE        Camera device (default: /dev/video0)
  -o, --output DEVICE       v4l2loopback device (default: /dev/video10)
  -W, --width N             Frame width (default: 1920)
  -H, --height N            Frame height (default: 1080)
  -f, --fps N               Target framerate (default: 30)
  -m, --model PATH          ONNX model path
  -e, --engine PATH         TensorRT plan cache path
  -d, --downsample RATIO    RVM downsample ratio (default: 0.25)
  -b, --background MODE     transparent|green|blur|color
  -c, --color R,G,B         Background color
      --no-fp16             Disable FP16 (use FP32)
      --benchmark           Print per-frame timing stats
  -v, --verbose             Verbose logging
```

## License

TBD
