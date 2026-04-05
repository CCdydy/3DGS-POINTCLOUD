"""
用 RENO 解码的点云替换 PandaSet 原始 LiDAR
RENO 输出为 .ply.bin.ply 格式，需转回 PandaSet 的 .pkl.gz 格式
"""
import argparse
import gzip
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_ply_ascii(filepath: str) -> np.ndarray:
    """读取 RENO 输出的 ASCII PLY 文件"""
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


def swap_lidar_for_scene(pandaset_scene: str, reno_decoded_dir: str, output_scene: str):
    """
    复制一个 PandaSet scene，只把 lidar/ 子目录替换成 RENO 解码版本
    RENO decompress 输出文件名格式: <original_name>.ply.bin.ply
    """
    pandaset_scene = Path(pandaset_scene)
    reno_decoded_dir = Path(reno_decoded_dir)
    output_scene = Path(output_scene)

    if output_scene.exists():
        print(f"[WARN] Output scene already exists, removing: {output_scene}")
        shutil.rmtree(output_scene)

    # 复制整个 scene
    shutil.copytree(pandaset_scene, output_scene)

    # 清空原始 lidar
    lidar_out = output_scene / "lidar"
    shutil.rmtree(lidar_out)
    lidar_out.mkdir()

    # RENO 解码输出的文件名格式: XX.ply.bin.ply
    ply_files = sorted(reno_decoded_dir.glob("*.ply.bin.ply"))
    if not ply_files:
        # 也试试直接 .ply 文件
        ply_files = sorted(reno_decoded_dir.glob("*.ply"))
    if not ply_files:
        print(f"[ERROR] No PLY files found in {reno_decoded_dir}")
        return

    for ply_file in ply_files:
        xyz = read_ply_ascii(str(ply_file))
        # 从文件名提取 frame_id: "00.ply.bin.ply" -> "00"
        frame_id = ply_file.name.split(".ply")[0]
        df = pd.DataFrame(xyz, columns=["x", "y", "z"])
        with gzip.open(lidar_out / f"{frame_id}.pkl.gz", "wb") as f:
            pickle.dump(df, f)
        print(f"  Swapped frame {frame_id}: {len(xyz)} points")

    print(f"Done. Output scene: {output_scene}")


def batch_swap(pandaset_scene: str, reno_base_dir: str, output_base_dir: str, posq_values: list[int]):
    """对多个 posQ 档位批量生成替换后的 scene"""
    scene_name = Path(pandaset_scene).name
    for posq in posq_values:
        reno_decoded = Path(reno_base_dir) / f"{scene_name}_posQ{posq}"
        output_scene = Path(output_base_dir) / f"{scene_name}_posQ{posq}"
        print(f"\n=== posQ={posq} ===")
        swap_lidar_for_scene(pandaset_scene, str(reno_decoded), str(output_scene))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swap PandaSet LiDAR with RENO decoded")
    parser.add_argument("--pandaset_scene", required=True, help="Original PandaSet scene dir")
    parser.add_argument("--reno_decoded_dir", help="RENO decoded dir (single mode)")
    parser.add_argument("--output_scene", help="Output scene dir (single mode)")
    parser.add_argument("--batch", action="store_true", help="Batch mode: sweep posQ values")
    parser.add_argument("--reno_base_dir", default="data/reno_decoded", help="Base dir for RENO decoded outputs")
    parser.add_argument("--output_base_dir", default="data/pandaset_compressed", help="Base dir for output scenes")
    parser.add_argument("--posq_values", nargs="+", type=int, default=[8, 16, 32, 64, 128],
                        help="posQ values for batch mode (lower=finer, higher=coarser)")
    args = parser.parse_args()

    if args.batch:
        batch_swap(args.pandaset_scene, args.reno_base_dir, args.output_base_dir, args.posq_values)
    else:
        if not args.reno_decoded_dir or not args.output_scene:
            parser.error("Single mode requires --reno_decoded_dir and --output_scene")
        swap_lidar_for_scene(args.pandaset_scene, args.reno_decoded_dir, args.output_scene)
