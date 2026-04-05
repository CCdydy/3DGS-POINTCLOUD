"""
可视化 RENO 压缩失真：原始 vs 各 posQ 解码点云
生成:
  1. 鸟瞰图 (BEV) 对比
  2. 近距离区域放大
  3. 逐点误差热力图
  4. 点数统计
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def read_ply_ascii(filepath: str) -> np.ndarray:
    """读取 ASCII PLY"""
    coords = []
    header_ended = False
    with open(filepath, "r") as f:
        for line in f:
            if header_ended:
                parts = line.strip().split()
                if len(parts) >= 3:
                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            elif line.strip() == "end_header":
                header_ended = True
    return np.array(coords, dtype=np.float32)


def compute_per_point_error(original: np.ndarray, decoded: np.ndarray) -> tuple:
    """计算每个解码点到最近原始点的距离"""
    from scipy.spatial import cKDTree
    tree = cKDTree(original)
    distances, _ = tree.query(decoded, k=1)
    return distances


def plot_bev_comparison(original: np.ndarray, decoded_dict: dict, output_path: str,
                        xlim=None, ylim=None, title_prefix=""):
    """鸟瞰图对比: 原始 + 各 posQ 解码"""
    n_cols = 1 + len(decoded_dict)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    point_size = 0.05
    alpha = 0.3

    # 原始
    ax = axes[0]
    ax.scatter(original[:, 0], original[:, 1], s=point_size, c='black', alpha=alpha)
    ax.set_title(f"{title_prefix}Original\n({len(original):,} pts)", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # 各 posQ
    for i, (posq, decoded) in enumerate(sorted(decoded_dict.items())):
        ax = axes[i + 1]
        ax.scatter(decoded[:, 0], decoded[:, 1], s=point_size, c='blue', alpha=alpha)
        ax.set_title(f"posQ={posq}\n({len(decoded):,} pts, {len(decoded)/len(original)*100:.1f}%)",
                     fontsize=11)
        ax.set_xlabel("X (m)")
        ax.set_aspect("equal")
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_error_heatmap(original: np.ndarray, decoded_dict: dict, output_path: str):
    """逐点误差热力图 (BEV)"""
    n_cols = len(decoded_dict)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    vmax = 0.5  # 0.5m 作为最大显示误差

    for i, (posq, decoded) in enumerate(sorted(decoded_dict.items())):
        errors = compute_per_point_error(original, decoded)
        ax = axes[i]
        sc = ax.scatter(decoded[:, 0], decoded[:, 1], s=0.1,
                        c=errors, cmap='hot_r', vmin=0, vmax=vmax, alpha=0.5)
        ax.set_title(f"posQ={posq}\nmean err={errors.mean():.4f}m, max={errors.max():.4f}m",
                     fontsize=10)
        ax.set_xlabel("X (m)")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Error (m)", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_side_view(original: np.ndarray, decoded_dict: dict, output_path: str):
    """侧视图 (X-Z) 对比，看高度方向的失真"""
    n_cols = 1 + len(decoded_dict)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 3))

    point_size = 0.05
    alpha = 0.3

    ax = axes[0]
    ax.scatter(original[:, 0], original[:, 2], s=point_size, c='black', alpha=alpha)
    ax.set_title(f"Original", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")

    for i, (posq, decoded) in enumerate(sorted(decoded_dict.items())):
        ax = axes[i + 1]
        ax.scatter(decoded[:, 0], decoded[:, 2], s=point_size, c='blue', alpha=alpha)
        ax.set_title(f"posQ={posq}", fontsize=11)
        ax.set_xlabel("X (m)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_stats_summary(original: np.ndarray, decoded_dict: dict, bpp_dict: dict, output_path: str):
    """统计图: 点数变化 + BPP + 误差"""
    posq_values = sorted(decoded_dict.keys())
    n_points = [len(decoded_dict[q]) for q in posq_values]
    bpps = [bpp_dict.get(q, 0) for q in posq_values]
    mean_errors = []
    for q in posq_values:
        err = compute_per_point_error(original, decoded_dict[q])
        mean_errors.append(err.mean())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 点数
    ax = axes[0]
    ax.bar(range(len(posq_values)), n_points, color='steelblue')
    ax.axhline(y=len(original), color='r', linestyle='--', label=f'Original ({len(original):,})')
    ax.set_xticks(range(len(posq_values)))
    ax.set_xticklabels([f'posQ={q}' for q in posq_values], rotation=30)
    ax.set_ylabel("Point count")
    ax.set_title("Decoded Point Count")
    ax.legend()

    # BPP
    ax = axes[1]
    ax.bar(range(len(posq_values)), bpps, color='coral')
    ax.set_xticks(range(len(posq_values)))
    ax.set_xticklabels([f'posQ={q}' for q in posq_values], rotation=30)
    ax.set_ylabel("Bits per point")
    ax.set_title("Compression Rate")

    # 误差
    ax = axes[2]
    ax.bar(range(len(posq_values)), [e * 100 for e in mean_errors], color='seagreen')
    ax.set_xticks(range(len(posq_values)))
    ax.set_xticklabels([f'posQ={q}' for q in posq_values], rotation=30)
    ax.set_ylabel("Mean error (cm)")
    ax.set_title("Mean Point-to-Point Error")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RENO compression artifacts")
    parser.add_argument("--original", required=True, help="Original PLY file")
    parser.add_argument("--decoded_dir", required=True, help="Base dir containing decoded_posQ*/ subdirs")
    parser.add_argument("--output_dir", default="results/vis_compression", help="Output directory")
    parser.add_argument("--posq_values", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始点云
    print("Loading original...")
    original = read_ply_ascii(args.original)
    print(f"  Original: {len(original):,} points")

    # 加载各 posQ 解码点云
    decoded_dict = {}
    decoded_base = Path(args.decoded_dir)
    original_name = Path(args.original).name

    for posq in args.posq_values:
        # RENO 解码文件名: <original>.bin.ply
        decoded_path = decoded_base / f"decoded_posQ{posq}" / f"{original_name}.bin.ply"
        if decoded_path.exists():
            decoded_dict[posq] = read_ply_ascii(str(decoded_path))
            print(f"  posQ={posq}: {len(decoded_dict[posq]):,} points")
        else:
            print(f"  posQ={posq}: NOT FOUND at {decoded_path}")

    if not decoded_dict:
        print("[ERROR] No decoded files found!")
        exit(1)

    # BPP 数据 (PandaSet frame 00)
    bpp_dict = {8: 13.089, 16: 10.205, 32: 7.410, 64: 4.761, 128: 2.549}

    # 1. 全局鸟瞰图
    print("\nPlotting BEV overview...")
    plot_bev_comparison(original, decoded_dict,
                        str(output_dir / "bev_overview.png"))

    # 2. 近距离放大 (ego 周围 30m)
    print("Plotting BEV close-up...")
    plot_bev_comparison(original, decoded_dict,
                        str(output_dir / "bev_closeup.png"),
                        xlim=(-30, 30), ylim=(-30, 30),
                        title_prefix="Close-up: ")

    # 3. 侧视图
    print("Plotting side view...")
    plot_side_view(original, decoded_dict,
                   str(output_dir / "side_view.png"))

    # 4. 误差热力图
    print("Computing error heatmaps...")
    plot_error_heatmap(original, decoded_dict,
                       str(output_dir / "error_heatmap.png"))

    # 5. 统计汇总
    print("Plotting statistics...")
    plot_stats_summary(original, decoded_dict, bpp_dict,
                       str(output_dir / "stats_summary.png"))

    print(f"\nAll visualizations saved to {output_dir}/")
