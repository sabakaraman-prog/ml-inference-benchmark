import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import models


MODEL_FACTORIES = {
    "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "mobilenet_v2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT),
    "efficientnet_b0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
}


def parse_batch_sizes(batch_sizes_str):
    parts = batch_sizes_str.split(",")
    batch_sizes = []

    for part in parts:
        part = part.strip()
        if part:
            value = int(part)
            if value <= 0:
                raise ValueError("Batch sizes must be positive integers.")
            batch_sizes.append(value)

    if not batch_sizes:
        raise ValueError("You must provide at least one batch size.")

    return batch_sizes


def get_model(model_name, device):
    model = MODEL_FACTORIES[model_name]()
    model.eval()
    model.to(device)
    return model


def time_inference(model, x, iters=100, warmup=20, device_name="cpu", use_amp=False):
    if device_name == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for _ in range(warmup):
            if device_name == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)

        if device_name == "cuda":
            torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(iters):
            if device_name == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = model(x)
            else:
                _ = model(x)

        if device_name == "cuda":
            torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    avg_ms = (total_time / iters) * 1000
    throughput = (x.shape[0] * iters) / total_time if total_time > 0 else float("inf")

    peak_memory_mb = None
    if device_name == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "total_time_s": total_time,
        "avg_ms": avg_ms,
        "throughput_img_s": throughput,
        "peak_memory_mb": peak_memory_mb,
    }


def save_csv(results, output_file):
    fieldnames = [
        "model",
        "device",
        "precision",
        "batch_size",
        "iters",
        "warmup",
        "total_time_s",
        "avg_ms",
        "throughput_img_s",
        "peak_memory_mb",
        "speedup_vs_cpu",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def plot_results(results, output_dir, model_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_rows = [row for row in results if row["device"] == "cpu"]
    gpu_rows = [row for row in results if row["device"] == "cuda"]

    if not cpu_rows:
        return

    cpu_rows = sorted(cpu_rows, key=lambda row: row["batch_size"])
    batch_sizes = [row["batch_size"] for row in cpu_rows]
    cpu_avg = [row["avg_ms"] for row in cpu_rows]
    cpu_throughput = [row["throughput_img_s"] for row in cpu_rows]

    if gpu_rows:
        gpu_rows = sorted(gpu_rows, key=lambda row: row["batch_size"])
        gpu_avg = [row["avg_ms"] for row in gpu_rows]
        gpu_throughput = [row["throughput_img_s"] for row in gpu_rows]
    else:
        gpu_avg = None
        gpu_throughput = None

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, cpu_avg, marker="o", label="CPU")
    if gpu_avg is not None:
        plt.plot(batch_sizes, gpu_avg, marker="o", label="GPU")
    plt.xlabel("Batch Size")
    plt.ylabel("Average Latency (ms)")
    plt.title(f"{model_name} Latency vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_latency.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(batch_sizes, cpu_throughput, marker="o", label="CPU")
    if gpu_throughput is not None:
        plt.plot(batch_sizes, gpu_throughput, marker="o", label="GPU")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (images/sec)")
    plt.title(f"{model_name} Throughput vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_throughput.png")
    plt.close()


def benchmark(args):
    cpu_device = torch.device("cpu")
    cuda_available = torch.cuda.is_available()
    gpu_device = torch.device("cuda") if cuda_available else None

    if cuda_available:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version (PyTorch build): {torch.version.cuda}")
    else:
        print("No CUDA GPU detected.")
        print("Running CPU benchmark only.")

    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Iters per test: {args.iters}")
    print(f"Warmup iters: {args.warmup}")

    use_amp = args.precision == "fp16"

    if use_amp and not cuda_available:
        print("\nWarning: fp16 requested but CUDA is not available. Falling back to CPU fp32.\n")
        use_amp = False

    model_cpu = get_model(args.model, cpu_device)
    model_gpu = None
    if cuda_available:
        model_gpu = get_model(args.model, gpu_device)

    results = []

    print("\n--- Inference Benchmark ---")

    for bs in args.batch_sizes:
        print(f"\nBatch size: {bs}")

        x_cpu = torch.randn(bs, 3, args.image_size, args.image_size, device=cpu_device)
        cpu_metrics = time_inference(
            model=model_cpu,
            x=x_cpu,
            iters=args.iters,
            warmup=args.warmup,
            device_name="cpu",
            use_amp=False,
        )

        print(
            f"CPU total: {cpu_metrics['total_time_s']:.4f}s | "
            f"avg: {cpu_metrics['avg_ms']:.3f} ms/iter | "
            f"throughput: {cpu_metrics['throughput_img_s']:.2f} img/s"
        )

        cpu_row = {
            "model": args.model,
            "device": "cpu",
            "precision": "fp32",
            "batch_size": bs,
            "iters": args.iters,
            "warmup": args.warmup,
            "total_time_s": round(cpu_metrics["total_time_s"], 6),
            "avg_ms": round(cpu_metrics["avg_ms"], 6),
            "throughput_img_s": round(cpu_metrics["throughput_img_s"], 6),
            "peak_memory_mb": "",
            "speedup_vs_cpu": 1.0,
        }
        results.append(cpu_row)

        if cuda_available and model_gpu is not None:
            x_gpu = torch.randn(bs, 3, args.image_size, args.image_size, device=gpu_device)
            gpu_metrics = time_inference(
                model=model_gpu,
                x=x_gpu,
                iters=args.iters,
                warmup=args.warmup,
                device_name="cuda",
                use_amp=use_amp,
            )

            speedup = (
                cpu_metrics["total_time_s"] / gpu_metrics["total_time_s"]
                if gpu_metrics["total_time_s"] > 0
                else float("inf")
            )

            print(
                f"GPU total: {gpu_metrics['total_time_s']:.4f}s | "
                f"avg: {gpu_metrics['avg_ms']:.3f} ms/iter | "
                f"throughput: {gpu_metrics['throughput_img_s']:.2f} img/s | "
                f"peak mem: {gpu_metrics['peak_memory_mb']:.2f} MB"
            )
            print(f"Speedup: {speedup:.2f}x")

            gpu_row = {
                "model": args.model,
                "device": "cuda",
                "precision": args.precision if use_amp else "fp32",
                "batch_size": bs,
                "iters": args.iters,
                "warmup": args.warmup,
                "total_time_s": round(gpu_metrics["total_time_s"], 6),
                "avg_ms": round(gpu_metrics["avg_ms"], 6),
                "throughput_img_s": round(gpu_metrics["throughput_img_s"], 6),
                "peak_memory_mb": round(gpu_metrics["peak_memory_mb"], 6),
                "speedup_vs_cpu": round(speedup, 6),
            }
            results.append(gpu_row)
        else:
            print("GPU: N/A")

    if args.csv:
        save_csv(results, args.csv)
        print(f"\nSaved CSV results to: {args.csv}")

    if args.plot:
        plot_results(results, args.plot_dir, args.model)
        print(f"Saved plots to: {args.plot_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="PyTorch inference benchmark tool")

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=MODEL_FACTORIES.keys(),
        help="Model to benchmark",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=[1, 8, 32],
        help="Comma-separated batch sizes, e.g. 1,8,32",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of timed iterations",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="Precision mode for CUDA benchmarking",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV output file",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate latency and throughput plots",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    benchmark(args)

if __name__ == "__main__":
    main()