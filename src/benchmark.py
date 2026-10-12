"""Training-throughput benchmark: CPU vs GPU, fp32 vs mixed precision.

Measures samples/second for a full training step (forward + backward + AdamW)
on random data, so the numbers are about the hardware and not the dataset.
Models are larger than the ones used in train.py so the GPU has real work to do.

    python -m src.benchmark            # full benchmark (use a GPU runtime, e.g. Colab)
    python -m src.benchmark --quick    # tiny sizes, just to check the script works
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from .models import LSTMForecaster, TransformerForecaster


def make(name, size, seq_len, n_feat=3):
    if name == "lstm":
        return LSTMForecaster(n_feat, hidden=size, layers=2)
    return TransformerForecaster(n_feat, seq_len, d_model=size, nhead=8, layers=4, dim_ff=size * 4)


def bench_one(name, size, batch, seq_len, device, amp, steps, warmup):
    torch.manual_seed(0)
    model = make(name, size, seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    x = torch.randn(batch, seq_len, 3, device=device)
    y = torch.randn(batch, device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            loss = nn.functional.mse_loss(model(x).float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return batch * steps / dt, dt / steps * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if args.quick:
        sizes, batch, seq_len, steps, warmup = {"lstm": 32, "transformer": 32}, 128, 22, 5, 1
    else:
        sizes, batch, seq_len, steps, warmup = {"lstm": 256, "transformer": 256}, 1024, 64, 30, 5

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("No CUDA device found: benchmarking CPU only. Use a GPU runtime for the GPU rows.")

    rows = []
    for name in ["lstm", "transformer"]:
        for dev in devices:
            for amp in ([False, True] if dev.type == "cuda" else [False]):
                sps, ms = bench_one(name, sizes[name], batch, seq_len, dev, amp, steps, warmup)
                label = (torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU")
                rows.append({"model": name, "device": label,
                             "precision": "fp16 AMP" if amp else "fp32",
                             "batch": batch, "seq_len": seq_len, "width": sizes[name],
                             "samples_per_sec": round(sps), "ms_per_step": round(ms, 1)})
                print(rows[-1])

    df = pd.DataFrame(rows)
    base = df[df.device == "CPU"].set_index("model").samples_per_sec
    df["speedup_vs_cpu"] = (df.samples_per_sec / df.model.map(base)).round(1)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / ("benchmark_quick.csv" if args.quick else "benchmark.csv"), index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
