"""
将 RENO 解码的点云注入 PandaSet scene，替换 lidar/ 目录。
保留相机、标注、ego-pose 等全部原始数据，只替换 LiDAR 几何。
RENO 只压缩几何 (xyz)，属性 (intensity, timestamp, d) 通过最近邻从原始点云插值。
"""
import argparse
import gzip
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def read_ply_ascii(filepath: str) -> np.ndarray:
    """读取 RENO 输出的 ASCII PLY"""
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


def inject_lidar(original_scene: str, decoded_ply_dir: str, output_scene: str):
    """
    复制 scene 目录，只替换 lidar/ 子目录为 RENO 解码版本。
    保留所有相机、标注、ego-pose 数据不变。
    """
    src = Path(original_scene)
    dst = Path(output_scene)
    decoded_dir = Path(decoded_ply_dir)

    if dst.exists():
        shutil.rmtree(dst)

    # 复制 scene（跳过 lidar 目录，避免浪费时间复制再删除）
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("lidar"))

    # 重建 lidar 目录
    lidar_dst = dst / "lidar"
    lidar_dst.mkdir()
    lidar_src = src / "lidar"

    for orig_pkl in sorted(lidar_src.glob("*.pkl.gz")):
        frame_id = orig_pkl.stem.split(".")[0]  # "00.pkl" -> "00"

        # RENO 解码文件名: <frame_id>.pkl.ply.bin.ply
        decoded_path = decoded_dir / f"{frame_id}.pkl.ply.bin.ply"

        if not decoded_path.exists():
            # fallback: 直接复制原始
            shutil.copy2(orig_pkl, lidar_dst / orig_pkl.name)
            print(f"  [WARN] frame {frame_id}: no decoded file, using original")
            continue

        # 读取 RENO 解码几何
        decoded_xyz = read_ply_ascii(str(decoded_path))

        # 读取原始点云（获取属性列）
        with gzip.open(orig_pkl, "rb") as f:
            orig_df = pickle.load(f)

        orig_xyz = orig_df[["x", "y", "z"]].values.astype(np.float32)

        # 最近邻插值: 为每个解码点找到原始点云中最近的点, 继承其属性
        tree = cKDTree(orig_xyz)
        _, idx = tree.query(decoded_xyz, k=1)

        # 构建新 DataFrame: 几何用解码版, 属性用原始版(最近邻)
        new_data = {"x": decoded_xyz[:, 0], "y": decoded_xyz[:, 1], "z": decoded_xyz[:, 2]}
        for col in orig_df.columns:
            if col not in ("x", "y", "z"):
                new_data[col] = orig_df[col].values[idx]

        new_df = pd.DataFrame(new_data)

        with gzip.open(lidar_dst / orig_pkl.name, "wb") as f:
            pickle.dump(new_df, f)

    print(f"  Done: {output_scene} ({len(list(lidar_dst.glob('*.pkl.gz')))} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject RENO decoded LiDAR into PandaSet scene")
    parser.add_argument("--original_scene", required=True, help="Original PandaSet scene path")
    parser.add_argument("--decoded_base", default="/media/zzy/SN5601/radar/data/reno_decoded",
                        help="Base directory containing 001_posQ*/ subdirs")
    parser.add_argument("--output_base", default="/media/zzy/SN5601/radar/data/pandaset_compressed",
                        help="Output base directory")
    parser.add_argument("--posq_values", nargs="+", type=int, default=[8, 32, 128],
                        help="posQ values to inject")
    args = parser.parse_args()

    scene_name = Path(args.original_scene).name  # e.g. "001"

    for posq in args.posq_values:
        decoded_dir = Path(args.decoded_base) / f"{scene_name}_posQ{posq}"
        output_scene = Path(args.output_base) / f"{scene_name}_posQ{posq}"
        print(f"\n=== posQ={posq} ===")

        if not decoded_dir.exists():
            print(f"  [ERROR] Decoded dir not found: {decoded_dir}")
            continue

        inject_lidar(args.original_scene, str(decoded_dir), str(output_scene))
