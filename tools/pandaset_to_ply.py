"""
PandaSet LiDAR (.pkl.gz) -> PLY (ASCII) 转换工具
用于将 PandaSet 格式的 LiDAR 数据转为 RENO 可读的 PLY 格式
"""
import argparse
import gzip
import pickle
from pathlib import Path

import numpy as np


def save_ply_ascii_geo(coords: np.ndarray, filepath: str):
    """保存为 ASCII PLY（与 RENO 的 kit.io.save_ply_ascii_geo 兼容）"""
    coords = coords.astype("float32")
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {coords.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for xyz in coords:
            f.write(f"{xyz[0]} {xyz[1]} {xyz[2]}\n")


def extract_pandaset_lidar(scene_dir: str, output_dir: str):
    lidar_dir = Path(scene_dir) / "lidar"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(lidar_dir.glob("*.pkl.gz"))
    if not pkl_files:
        print(f"[ERROR] No .pkl.gz files found in {lidar_dir}")
        return

    print(f"Found {len(pkl_files)} frames in {lidar_dir}")
    for pkl_file in pkl_files:
        with gzip.open(pkl_file, "rb") as f:
            pc = pickle.load(f)  # pandas DataFrame: x, y, z, intensity, ...

        xyz = pc[["x", "y", "z"]].values.astype(np.float32)
        frame_id = pkl_file.stem  # e.g., "00", "01", ...
        ply_path = output_dir / f"{frame_id}.ply"
        save_ply_ascii_geo(xyz, str(ply_path))
        print(f"  {frame_id}: {len(xyz)} points -> {ply_path}")

    print(f"Done. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PandaSet LiDAR to PLY for RENO")
    parser.add_argument("--scene_dir", required=True, help="PandaSet scene dir (e.g., pandaset/001)")
    parser.add_argument("--output_dir", required=True, help="Output dir for .ply files")
    args = parser.parse_args()

    extract_pandaset_lidar(args.scene_dir, args.output_dir)
