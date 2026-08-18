"""
Model-size / computational-cost profiling utilities (Sec. 7.5 / Table 1).
"""

import time

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    """Approximate on-disk size assuming float32 parameters."""
    n_params = count_parameters(model)
    return n_params * 4 / (1024 ** 2)


def estimate_flops(model: torch.nn.Module, input_shape, device="cpu") -> float:
    """FLOPs for one forward pass, using `thop` if available.

    Falls back to a parameter-based rough estimate (2x param count) with a
    printed warning if `thop` is not installed -- install it
    (`pip install thop`) for an accurate number to report in the paper.
    """
    try:
        from thop import profile
        dummy = torch.randn(1, *input_shape).to(device)
        model_device = model.to(device)
        flops, _ = profile(model_device, inputs=(dummy,), verbose=False)
        return float(flops)
    except ImportError:
        print("[complexity] `thop` not installed -- returning a rough "
              "parameter-based FLOPs estimate. Run `pip install thop` for "
              "an accurate figure.")
        return 2.0 * count_parameters(model)


def measure_inference_time(model: torch.nn.Module, input_shape,
                            device="cpu", n_warmup=10, n_runs=100) -> float:
    """Average single-sample inference latency in milliseconds."""
    model = model.to(device).eval()
    dummy = torch.randn(1, *input_shape).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    return (end - start) / n_runs * 1000.0  # ms per sample


def profile_model(model: torch.nn.Module, window_size: int, input_dim: int,
                   device="cpu") -> dict:
    input_shape = (window_size, input_dim)
    return {
        "params": count_parameters(model),
        "size_mb": round(model_size_mb(model), 4),
        "flops": estimate_flops(model, input_shape, device),
        "inference_ms": round(measure_inference_time(model, input_shape, device), 4),
    }
