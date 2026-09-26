#!/usr/bin/env python3
"""
Experiment Runner for Lightweight Adaptive Multi-Scale Industrial Anomaly Detection
Supports:
  1. Full training and multi-domain evaluation on AeBAD-S (same, view, illumination, background)
  2. Computational hardware efficiency benchmarking (FLOPs, latency, active params, RAM, size)
"""

import os
import sys
import argparse
import subprocess
import torch

from utils import parse_args, load_config, setup_logging
from tools import train


def run_benchmark():
    """Run standalone computational efficiency benchmark."""
    benchmark_script = os.path.join(os.path.dirname(__file__), "tools", "benchmark_efficiency.py")
    print(f"\n>>> Running Computational Efficiency Benchmark: {benchmark_script}")
    subprocess.run([sys.executable, benchmark_script], check=True)


def run_pipeline(cfg_path, epochs=None, batch_size=None, test_only=False):
    """Run training and evaluation pipeline on AeBAD-S."""
    class CustomArgs:
        device = "0" if torch.cuda.is_available() else "cpu"
        cfg_files = [cfg_path]
        opts = []

    args = CustomArgs()
    if epochs is not None:
        args.opts.extend(["TRAIN_SETUPS.epochs", int(epochs)])
    if batch_size is not None:
        args.opts.extend(["TRAIN_SETUPS.batch_size", int(batch_size), "TEST_SETUPS.batch_size", int(batch_size)])
    if test_only:
        args.opts.extend(["TRAIN.enable", False, "TEST.enable", True])

    cfg = load_config(args, path_to_config=cfg_path)
    setup_logging(cfg)
    train(cfg=cfg)


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight Adaptive Multi-Scale Anomaly Detection on AeBAD-S"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="benchmark",
        choices=["benchmark", "train", "eval"],
        help="Execution mode: benchmark (hardware profile), train (fit+eval), or eval (evaluation only)"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="method_config/AeBAD_S/LightweightAdaptive.yaml",
        help="Path to configuration YAML"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")

    args = parser.parse_args()

    if args.mode == "benchmark":
        run_benchmark()
    elif args.mode == "train":
        print(f"\n>>> Launching Lightweight Adaptive Model Training on AeBAD-S with config: {args.cfg}")
        run_pipeline(args.cfg, epochs=args.epochs, batch_size=args.batch_size, test_only=False)
    elif args.mode == "eval":
        print(f"\n>>> Launching Lightweight Adaptive Model Multi-Domain Evaluation on AeBAD-S with config: {args.cfg}")
        run_pipeline(args.cfg, batch_size=args.batch_size, test_only=True)


if __name__ == "__main__":
    main()
