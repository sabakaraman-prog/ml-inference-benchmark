# ML Inference Benchmark

A PyTorch benchmarking tool for comparing CPU and GPU inference performance across CNN models.

## Features
- CPU vs GPU inference comparison
- Batch size scaling
- Latency measurement
- Throughput (images/sec)
- Optional CSV output
- Visualization graphs

## Supported Models
- ResNet18
- ResNet50
- MobileNetV2
- EfficientNetB0

## Example

python benchmark.py --model resnet50 --batch-sizes 1,8,32 --plot

## Requirements

pip install torch torchvision matplotlib
