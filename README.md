#ML Inference Benchmark

A PyTorch-based benchmarking tool for evaluating deep learning inference performance across CPU and GPU environments.

##Overview

This project measures latency, throughput, and memory usage for CNN models under different batch sizes, helping analyze how hardware impacts inference performance.

The tool demonstrates significant performance gains with GPU acceleration, achieving up to 29.97x higher throughput compared to CPU in controlled experiments.

##Features
CPU vs GPU inference benchmarking
Batch size scaling analysis
Latency and throughput measurement
GPU memory usage tracking
Optional CSV export for results
Built-in visualization (matplotlib)

##Supported Models
ResNet18
ResNet50
MobileNetV2
EfficientNetB0

##Example Usage
python benchmark.py --model resnet50 --batch-sizes 1,8,32 --plot

##Sample Results:

Up to 29.97x throughput improvement with GPU acceleration

Reduced latency from ~13.4 ms to ~1.6 ms per inference

Peak GPU memory usage up to ~268 MB

<img width="470" height="353" alt="ML_graph" src="https://github.com/user-attachments/assets/54ce6c63-efda-4bd8-885d-144505762a59" />

##Requirements:
pip install torch torchvision matplotlib

##Key Concepts:

Performance benchmarking

Parallel computing (GPU acceleration)

Inference optimization

Experimental evaluation of ML systems
