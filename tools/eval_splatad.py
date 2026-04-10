"""
轻量 SplatAD 评估脚本
直接加载 checkpoint，渲染测试帧，计算 PSNR/SSIM/LPIPS
避免 ns-eval 的 chamfer distance OOM 问题
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yml")
    parser.add_argument("--data-root", required=True, help="PandaSet root path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    # 禁用 torch.compile
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

    from nerfstudio.utils.eval_utils import eval_setup

    # 修改 config 中的 data 路径
    def update_config(config):
        config.pipeline.datamanager.dataparser.data = Path(args.data_root)
        return config

    config, pipeline, checkpoint_path, _ = eval_setup(
        Path(args.config),
        update_config_callback=update_config,
    )

    pipeline.eval()

    # 手动计算指标，跳过 chamfer distance
    metrics_list = []
    num_eval = len(pipeline.datamanager.eval_dataset)
    print(f"Evaluating {num_eval} test frames...")

    with torch.no_grad():
        for i in range(num_eval):
            try:
                ray_bundle, batch = pipeline.datamanager.next_eval(i)
                outputs = pipeline.model(ray_bundle)

                # 只计算图像指标
                from torchmetrics.image import PeakSignalNoiseRatio
                from torchmetrics.image import StructuralSimilarityIndexMeasure
                from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

                psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(pipeline.device)
                ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(pipeline.device)
                lpips_fn = LearnedPerceptualImagePatchSimilarity(normalize=True).to(pipeline.device)

                gt_img = batch["image"].to(pipeline.device)  # [H, W, 3]
                pred_img = outputs["rgb"]  # [H, W, 3]

                # 转为 [1, 3, H, W] 格式
                gt = gt_img.permute(2, 0, 1).unsqueeze(0)
                pred = pred_img.permute(2, 0, 1).unsqueeze(0).clamp(0, 1)

                psnr = float(psnr_fn(pred, gt))
                ssim = float(ssim_fn(pred, gt))
                lpips = float(lpips_fn(pred, gt))

                metrics_list.append({"psnr": psnr, "ssim": ssim, "lpips": lpips})
                print(f"  Frame {i}/{num_eval}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}")

                # 释放显存
                del outputs, gt_img, pred_img, gt, pred
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Frame {i}: ERROR - {e}")
                continue

    if metrics_list:
        avg = {
            "psnr": np.mean([m["psnr"] for m in metrics_list]),
            "ssim": np.mean([m["ssim"] for m in metrics_list]),
            "lpips": np.mean([m["lpips"] for m in metrics_list]),
            "num_frames": len(metrics_list),
        }
        print(f"\n=== Average over {len(metrics_list)} frames ===")
        print(f"  PSNR:  {avg['psnr']:.2f} dB")
        print(f"  SSIM:  {avg['ssim']:.4f}")
        print(f"  LPIPS: {avg['lpips']:.4f}")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"average": avg, "per_frame": metrics_list}, f, indent=2)
        print(f"Saved: {output_path}")
    else:
        print("[ERROR] No frames evaluated successfully")


if __name__ == "__main__":
    main()
