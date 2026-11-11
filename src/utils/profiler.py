"""
Hardware & Computational Efficiency Profiler for PyTorch Models.
Measures:
1. FLOPs / MACs via PyTorch native FlopCounterMode (zero external dependencies).
2. Peak GPU Memory (VRAM) consumption in MB.
3. Inference Latency (ms/sample) and Throughput (samples/sec) with warm-up.
4. Parameter footprint (trainable vs total).
"""
import time
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def format_flops(flops: int) -> str:
    """Formats raw FLOPs integer into human-readable string."""
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    return f"{flops} FLOPs"


def profile_model_efficiency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup_runs: int = 20,
    timed_runs: int = 100
) -> Dict[str, Any]:
    """
    Conducts empirical hardware efficiency profiling.
    Args:
        model: PyTorch model (nn.Module)
        sample_input: Input tensor matching inference batch shape (e.g. (1, 7, 24) or (1, 3, 224, 224))
        device: torch.device (CUDA or CPU)
    Returns:
        Dict containing parameter counts, FLOPs, latency, throughput, and peak VRAM.
    """
    model.eval()
    model = model.to(device)
    sample_input = sample_input.to(device)

    # 1. Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 2. FLOPs counting via PyTorch native FlopCounterMode
    try:
        # Clone sample on CPU for reliable flop tracing
        cpu_model = model.to("cpu").eval()
        cpu_input = sample_input.to("cpu")
        with torch.no_grad():
            with FlopCounterMode(display=False) as flop_counter:
                cpu_model(cpu_input)
            total_flops = int(flop_counter.get_total_flops())
    except Exception as e:
        total_flops = 0
    finally:
        model.to(device)
        sample_input = sample_input.to(device)

    # 3. Peak VRAM & Inference Latency Measurement
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    # Timed inference phase
    latencies = []
    with torch.no_grad():
        for _ in range(timed_runs):
            t_start = time.perf_counter()
            _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)  # Convert to milliseconds

    avg_latency_ms = float(sum(latencies) / len(latencies))
    batch_size = sample_input.size(0)
    latency_per_sample_ms = avg_latency_ms / max(batch_size, 1)
    throughput_fps = (1000.0 / latency_per_sample_ms) if latency_per_sample_ms > 0 else 0.0

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "total_flops": total_flops,
        "flops_formatted": format_flops(total_flops),
        "inference_latency_ms": round(latency_per_sample_ms, 3),
        "throughput_fps": round(throughput_fps, 1),
        "peak_vram_mb": peak_vram_mb,
        "device_used": str(device),
        "input_tensor_shape": list(sample_input.shape),
    }
