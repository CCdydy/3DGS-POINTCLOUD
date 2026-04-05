"""
绘制 Sensitivity Curve：D1 PSNR vs. Rendering PSNR
核心实验结果可视化 - 证明传统几何指标不能预测渲染质量

RENO 使用 posQ 参数控制量化粒度:
  posQ=8  -> 最精细 (最多 bit, 最高几何质量)
  posQ=16 -> 默认 (14-bit 等效)
  posQ=32 -> 中等
  posQ=64 -> 粗 (12-bit 等效)
  posQ=128 -> 最粗 (最少 bit)
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str, posq_values: list):
    """加载各 posQ 配置的评估结果"""
    results_dir = Path(results_dir)
    rendering = {}

    for posq in posq_values + ["raw"]:
        result_file = results_dir / f"sensitivity_{posq}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            rendering[posq] = {
                "psnr": data.get("camera_psnr", data.get("psnr", data.get("results", {}).get("psnr", 0))),
                "ssim": data.get("camera_ssim", data.get("ssim", data.get("results", {}).get("ssim", 0))),
                "lpips": data.get("camera_lpips", data.get("lpips", data.get("results", {}).get("lpips", 0))),
            }
        else:
            print(f"[WARN] Missing result: {result_file}")

    return rendering


def plot_sensitivity_curve(geometry_results: dict, rendering_results: dict, output_path: str):
    posq_values = sorted([k for k in geometry_results.keys() if isinstance(k, int)])

    bpps = [geometry_results[q]["bpp"] for q in posq_values]
    d1_psnrs = [geometry_results[q]["d1_psnr"] for q in posq_values]
    render_psnrs = [rendering_results[q]["psnr"] for q in posq_values if q in rendering_results]

    raw_psnr = rendering_results.get("raw", {}).get("psnr", None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Plot 1: Geometry D1 PSNR vs Bitrate
    ax1 = axes[0]
    ax1.plot(bpps, d1_psnrs, "b-o", linewidth=2, markersize=8, label="D1 PSNR (geometry)")
    ax1.set_xlabel("Bits per point (BPP)", fontsize=12)
    ax1.set_ylabel("D1 PSNR (dB)", fontsize=12)
    ax1.set_title("Geometry Quality vs. Bitrate", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rendering PSNR vs Bitrate
    ax2 = axes[1]
    valid_bpps = [bpps[i] for i, q in enumerate(posq_values) if q in rendering_results]
    ax2.plot(valid_bpps, render_psnrs, "g-s", linewidth=2, markersize=8, label="Camera PSNR (rendering)")
    if raw_psnr is not None:
        ax2.axhline(y=raw_psnr, color="r", linestyle="--", linewidth=1.5,
                     label=f"Raw LiDAR ceiling ({raw_psnr:.1f} dB)")
    ax2.set_xlabel("Bits per point (BPP)", fontsize=12)
    ax2.set_ylabel("Camera PSNR (dB)", fontsize=12)
    ax2.set_title("Rendering Quality vs. Bitrate", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: D1 PSNR vs Rendering PSNR (key insight)
    ax3 = axes[2]
    valid_d1 = [d1_psnrs[i] for i, q in enumerate(posq_values) if q in rendering_results]
    ax3.plot(valid_d1, render_psnrs, "r-D", linewidth=2, markersize=8)
    for i, q in enumerate([q for q in posq_values if q in rendering_results]):
        ax3.annotate(f"posQ={q}", (valid_d1[i], render_psnrs[i]),
                     textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax3.set_xlabel("D1 PSNR (dB)", fontsize=12)
    ax3.set_ylabel("Camera PSNR (dB)", fontsize=12)
    ax3.set_title("Geometry vs. Rendering Quality", fontsize=13)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sensitivity curve")
    parser.add_argument("--results_dir", default="results", help="Directory with sensitivity_*.json files")
    parser.add_argument("--output", default="results/sensitivity_curve.pdf", help="Output figure path")
    args = parser.parse_args()

    # Geometry results from RENO KITTI smoke test (2026-04-05)
    # Replace with PandaSet results when available
    geometry_results = {
        8:   {"bpp": 9.644, "d1_psnr": 88.12},   # finest
        16:  {"bpp": 7.049, "d1_psnr": 82.20},   # default (14-bit)
        32:  {"bpp": 4.555, "d1_psnr": 76.22},
        64:  {"bpp": 2.552, "d1_psnr": 70.18},   # ~12-bit
        128: {"bpp": 1.293, "d1_psnr": 64.18},   # coarsest
    }

    posq_values = [8, 16, 32, 64, 128]
    rendering_results = load_results(args.results_dir, posq_values)

    if rendering_results:
        plot_sensitivity_curve(geometry_results, rendering_results, args.output)
    else:
        print("[ERROR] No rendering results found. Run SplatAD evaluation first.")
