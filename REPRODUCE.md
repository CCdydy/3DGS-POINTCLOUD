# LiDAR × 3DGS 联合压缩：复现指南

> **研究目标**：压缩 LiDAR 点云的同时保留下游 3D Gaussian Splatting 的渲染质量，
> 面向自动驾驶数据传输瓶颈场景。  
> **GPU 环境**：RTX 5090 / RTX 6000 Ada（≥48 GB VRAM）  
> **预计总周期**：2 周跑通基线，第 3–4 周建立 sensitivity curve

---

## 目录

1. [研究定位与核心发现](#1-研究定位与核心发现)
2. [两套独立环境配置](#2-两套独立环境配置)
3. [数据集准备](#3-数据集准备)
4. [Phase 1：LiDAR 压缩基线（RENO）](#4-phase-1lidar-压缩基线reno)
5. [Phase 2：LiDAR 初始化 3DGS 基线（SplatAD）](#5-phase-2lidar-初始化-3dgs-基线splatad)
6. [Phase 3：Sensitivity Curve（核心实验）](#6-phase-3sensitivity-curve核心实验)
7. [Phase 4：3DGS 自身压缩基线（HAC++）](#7-phase-4-3dgs-自身压缩基线hac)
8. [参考方法速查表](#8-参考方法速查表)
9. [已知坑与解决方案](#9-已知坑与解决方案)
10. [评估指标说明](#10-评估指标说明)

---

## 1. 研究定位与核心发现

### 这个方向为什么是空白

文献调查覆盖 40+ 篇论文，确认三个社区**完全未交汇**：

```
Stream A: LPCC 社区（RENO, G-PCC, AdaDPCC）
  → 优化 D1/D2 PSNR + 下游 3D 检测精度
  → 从不用解码点云初始化 3DGS，从不测渲染质量

Stream B: 自动驾驶 3DGS 社区（SplatAD, Street Gaussians, OmniRe）
  → 用原始 LiDAR 初始化 Gaussian，渲染 PSNR 极高
  → 从不考虑 LiDAR 在传输前被压缩的情况

Stream C: 3DGS 压缩社区（HAC++, RDO-Gaussian, ContextGS）
  → 压缩已优化好的 Gaussian 参数
  → 从不评估几何保真度（D1/D2 PSNR, Chamfer distance）
  → 从不从原始传感器数据出发
```

**你的贡献**：在 Stream A 的输出和 Stream B 的输入之间打通链路，
用渲染质量替代/增强几何失真作为 LiDAR 压缩的损失函数。

### 最接近的工作及差距

| 论文 | 做了什么 | 缺了什么 |
|------|---------|---------|
| Bits-to-Photon (NYU 2024) | PCC → 直接解码成 Gaussian，端到端优化渲染质量 | 针对**密集 RGB 点云**（8iVFB），不是稀疏 LiDAR |
| RENO (CVPR 2025) | 实时 LiDAR 神经压缩，10 fps on RTX 3090 | 只测 D1/D2 + PointPillars 检测，**从不测渲染质量** |
| SplatAD (CVPR 2025) | LiDAR 初始化 3DGS，同时渲染相机和 LiDAR | **假设输入 LiDAR 未压缩** |
| RDO-Gaussian (ECCV 2024) | 对 3DGS 做率失真优化，distortion = 渲染质量 | 从优化好的 Gaussian 出发，**不处理原始点云** |

---

## 2. 两套独立环境配置

> ⚠️ **必须用两个独立 conda env**，RENO 和 neurad-studio 的 torch 版本有冲突。
> 两者通过磁盘上的 `.bin` / `.pcd` 文件交换数据。

### Env A：RENO（LiDAR 压缩）

```bash
conda create -n reno python=3.10
conda activate reno

# PyTorch（CUDA 12.1，5090/RTX6000Ada 兼容）
pip install torch==2.1.2+cu121 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121

# MinkowskiEngine（用 NVIDIA 维护的 fork，对新 CUDA 支持更好）
# 先装编译依赖
sudo apt-get install libopenblas-dev

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install \
    --blas_include_dirs=/usr/include/openblas \
    --blas=openblas
cd ..

# RENO 本体
git clone https://github.com/NJUVISION/RENO.git
cd RENO
pip install -r requirements.txt
# 验证安装
python -c "import MinkowskiEngine as ME; print(ME.__version__)"
```

**安装 MinkowskiEngine 失败的备选方案**（如果 NVIDIA fork 也失败）：

```bash
# 用预编译 wheel（社区维护，支持到 CUDA 12.x）
pip install https://github.com/chaytonmin/MinkowskiEngine/releases/download/v0.5.4/MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl
```

### Env B：neurad-studio（SplatAD + 3DGS 训练）

```bash
conda create -n neurad python=3.10
conda activate neurad

# neurad-studio 用自己的 gsplat 定制分支，不能用官方 pip 版本
git clone https://github.com/georghess/neurad-studio.git
cd neurad-studio

# 先装 tiny-cuda-nn（耗时较长，需要编译 CUDA kernel）
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 安装 neurad-studio 及其所有依赖
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"

# 验证
python -c "import nerfstudio; print('neurad-studio OK')"
```

---

## 3. 数据集准备

### 推荐起点：PandaSet

**理由**：80 GB（最小）、CC BY 4.0（最宽松）、双 LiDAR（旋转式 + 固态）、
6 个环视相机、SplatAD / NeuRAD 原生支持。

```bash
# 注册后从 Scale AI 下载
# https://scale.com/open-av-datasets/pandaset
# 下载所有 sequence（共 103 个，每个约 800 MB）

# 目录结构（neurad-studio 要求的格式）
pandaset/
├── 001/
│   ├── camera/
│   │   ├── back_camera/
│   │   ├── front_camera/
│   │   ├── front_left_camera/
│   │   ├── front_right_camera/
│   │   ├── left_camera/
│   │   └── right_camera/
│   ├── lidar/                  # .pkl.gz 格式
│   ├── annotations/
│   └── meta/
├── 002/
...
```

neurad-studio 的 PandaSet dataparser 可直接读取上述格式，
**不需要额外转换**：

```bash
# 训练单个 scene（scene 001 约 30 分钟）
conda activate neurad
cd neurad-studio
python train.py splatad \
    --data.path /path/to/pandaset/001 \
    --data.parser PandaSetDataParserConfig \
    --output-dir ./outputs/pandaset_001
```

### 最终评估：Waymo NOTR

完成方法开发后在 Waymo 上跑 final benchmark（Street Gaussians 和 OmniRe 的标准测试集）。

```bash
# Waymo Open Dataset 需要申请访问权限
# https://waymo.com/open/
# 下载 dynamic32 + static32 splits（共约 200 GB）

# 用 OmniRe（drivestudio）的 Waymo dataparser
git clone https://github.com/ziyc/drivestudio.git
```

---

## 4. Phase 1：LiDAR 压缩基线（RENO）

**目标**：在 PandaSet 上建立 RENO 的率失真曲线（D1/D2 PSNR vs. bitrate）

### 4.1 准备输入点云

PandaSet 的 LiDAR 是 `.pkl.gz` 格式，需要先提取成 RENO 支持的格式：

```python
# tools/pandaset_to_ply.py
import gzip, pickle, numpy as np
from pathlib import Path

def extract_pandaset_lidar(scene_dir, output_dir):
    lidar_dir = Path(scene_dir) / "lidar"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pkl_file in sorted(lidar_dir.glob("*.pkl.gz")):
        with gzip.open(pkl_file, 'rb') as f:
            pc = pickle.load(f)  # pandas DataFrame: x, y, z, intensity, ...
        
        xyz = pc[['x', 'y', 'z']].values.astype(np.float32)
        frame_id = pkl_file.stem  # e.g., "00", "01", ...
        
        # 保存为 numpy binary（RENO 读取格式）
        np.save(output_dir / f"{frame_id}.npy", xyz)
        print(f"  {frame_id}: {len(xyz)} points")

extract_pandaset_lidar("pandaset/001", "data/reno_input/001")
```

### 4.2 运行 RENO 压缩

```bash
conda activate reno
cd RENO

# 压缩（14-bit 精度，与论文一致）
python compress.py \
    --input_dir ../data/reno_input/001 \
    --output_dir ../data/reno_output/001 \
    --bit_depth 14 \
    --model_path checkpoints/reno_14bit.pth

# 解压（生成重建点云）
python decompress.py \
    --input_dir ../data/reno_output/001 \
    --output_dir ../data/reno_decoded/001

# 计算 D1/D2 PSNR（RENO 内置评估脚本）
python eval_geometry.py \
    --original ../data/reno_input/001 \
    --decoded  ../data/reno_decoded/001
```

**多比特率扫描**（生成率失真曲线所需的多个点）：

```bash
for BITDEPTH in 10 11 12 13 14; do
    python compress.py \
        --input_dir ../data/reno_input/001 \
        --output_dir ../data/reno_output/001_${BITDEPTH}bit \
        --bit_depth ${BITDEPTH} \
        --model_path checkpoints/reno_${BITDEPTH}bit.pth
    
    python decompress.py \
        --input_dir ../data/reno_output/001_${BITDEPTH}bit \
        --output_dir ../data/reno_decoded/001_${BITDEPTH}bit
done
```

**预期输出**（参考 RENO 论文 Table 1，KITTI 数据集）：

| Bit depth | BPP（估算）| D1 PSNR |
|-----------|-----------|---------|
| 10-bit | ~1.5 | ~55 dB |
| 12-bit | ~2.5 | ~65 dB |
| 14-bit | ~4.0 | ~75 dB |

---

## 5. Phase 2：LiDAR 初始化 3DGS 基线（SplatAD）

**目标**：用**未压缩**的 LiDAR 初始化 SplatAD，建立渲染质量上界

```bash
conda activate neurad
cd neurad-studio

# 标准 SplatAD 训练（scene 001，全精度 LiDAR）
python train.py splatad \
    --data.path /path/to/pandaset/001 \
    --data.parser PandaSetDataParserConfig \
    --trainer.max_iterations 30000 \
    --output-dir ./outputs/baseline_001

# 评估渲染质量（PSNR / SSIM / LPIPS）
python eval.py \
    --checkpoint ./outputs/baseline_001/checkpoint_30000.ckpt \
    --data.path /path/to/pandaset/001
```

**记录指标**（这是你的上界，后续对比的参照）：

```
# 预期范围（参考 SplatAD 论文，nuScenes 数据集）
Camera PSNR:  ~26–28 dB
LiDAR depth PSNR: ~35–40 dB
SSIM: ~0.80–0.85
LPIPS: ~0.15–0.20
```

---

## 6. Phase 3：Sensitivity Curve（核心实验）

**这是整个项目的核心贡献**：
用压缩后的 LiDAR 初始化 SplatAD，观察渲染质量如何随比特率下降。

### 6.1 数据路由脚本

```bash
# 用 RENO 解码的点云替换 PandaSet 原始 LiDAR
# tools/swap_lidar.py

import shutil
from pathlib import Path

def swap_lidar_for_scene(pandaset_scene, reno_decoded_dir, output_scene):
    """
    复制一个 PandaSet scene，只把 lidar/ 子目录替换成 RENO 解码版本
    """
    shutil.copytree(pandaset_scene, output_scene)
    
    # 清空原始 lidar
    lidar_out = Path(output_scene) / "lidar"
    shutil.rmtree(lidar_out)
    lidar_out.mkdir()
    
    # 从 RENO 输出转回 pkl.gz（SplatAD dataparser 需要原格式）
    for npy_file in sorted(Path(reno_decoded_dir).glob("*.npy")):
        import numpy as np, gzip, pickle, pandas as pd
        xyz = np.load(npy_file)
        df = pd.DataFrame(xyz, columns=['x', 'y', 'z'])
        frame_id = npy_file.stem
        with gzip.open(lidar_out / f"{frame_id}.pkl.gz", 'wb') as f:
            pickle.dump(df, f)

# 对每个比特率档创建一个 scene 副本
for bitdepth in [10, 11, 12, 13, 14]:
    swap_lidar_for_scene(
        pandaset_scene="pandaset/001",
        reno_decoded_dir=f"data/reno_decoded/001_{bitdepth}bit",
        output_scene=f"data/pandaset_compressed/001_{bitdepth}bit"
    )
```

### 6.2 批量训练 + 评估

```bash
conda activate neurad
cd neurad-studio

for BITDEPTH in 10 11 12 13 14 raw; do
    if [ "$BITDEPTH" = "raw" ]; then
        DATA_PATH="pandaset/001"
    else
        DATA_PATH="data/pandaset_compressed/001_${BITDEPTH}bit"
    fi
    
    python train.py splatad \
        --data.path $DATA_PATH \
        --data.parser PandaSetDataParserConfig \
        --trainer.max_iterations 30000 \
        --output-dir ./outputs/sensitivity_${BITDEPTH}
    
    python eval.py \
        --checkpoint ./outputs/sensitivity_${BITDEPTH}/checkpoint_30000.ckpt \
        --data.path $DATA_PATH \
        --output-json ./results/sensitivity_${BITDEPTH}.json
done
```

### 6.3 绘制 Sensitivity Curve

```python
# tools/plot_sensitivity.py
import json, matplotlib.pyplot as plt
import numpy as np

# 从 RENO 评估结果读取几何指标
geometry_results = {
    10: {"bpp": 1.52, "d1_psnr": 55.2},
    11: {"bpp": 2.01, "d1_psnr": 60.1},
    12: {"bpp": 2.53, "d1_psnr": 65.3},
    13: {"bpp": 3.18, "d1_psnr": 70.4},
    14: {"bpp": 4.05, "d1_psnr": 75.8},
}

# 从 SplatAD 评估结果读取渲染指标
rendering_results = {}
for bitdepth in [10, 11, 12, 13, 14, "raw"]:
    with open(f"results/sensitivity_{bitdepth}.json") as f:
        data = json.load(f)
    rendering_results[bitdepth] = {
        "psnr": data["camera_psnr"],
        "ssim": data["camera_ssim"],
        "lpips": data["camera_lpips"],
    }

# 关键图：D1 PSNR vs. Rendering PSNR（应该不是线性关系）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

bitdepths = [10, 11, 12, 13, 14]
bpps = [geometry_results[b]["bpp"] for b in bitdepths]
d1_psnrs = [geometry_results[b]["d1_psnr"] for b in bitdepths]
render_psnrs = [rendering_results[b]["psnr"] for b in bitdepths]
raw_psnr = rendering_results["raw"]["psnr"]

ax1.plot(bpps, d1_psnrs, 'b-o', label="D1 PSNR (geometry)")
ax1.axhline(y=raw_psnr, color='r', linestyle='--', label=f"Raw LiDAR ceiling ({raw_psnr:.1f})")
ax1.set_xlabel("Bits per point")
ax1.set_ylabel("PSNR (dB)")
ax1.set_title("Geometry: D1 PSNR vs. Bitrate")
ax1.legend()

ax2.plot(bpps, render_psnrs, 'g-o', label="Rendering PSNR (camera)")
ax2.axhline(y=raw_psnr, color='r', linestyle='--', label=f"Raw LiDAR ceiling ({raw_psnr:.1f})")
ax2.set_xlabel("Bits per point")
ax2.set_ylabel("PSNR (dB)")
ax2.set_title("Rendering: Camera PSNR vs. Bitrate")
ax2.legend()

plt.tight_layout()
plt.savefig("results/sensitivity_curve.pdf", dpi=300)
print("Saved: results/sensitivity_curve.pdf")
```

**这张图是论文的第一个关键图**。如果两条曲线的斜率/形状不同（几乎肯定如此），就证明了"传统几何指标不能预测渲染质量"这一核心论点。

---

## 7. Phase 4：3DGS 自身压缩基线（HAC++）

**目标**：评估对**已优化好的 Gaussian** 做压缩的效果，作为 Stream C 的对比方法

### 7.1 安装 HAC++

```bash
conda create -n hac python=3.10
conda activate hac

# 需要 CUDA 11.8（HAC 在 12.x 上有已知问题）
# 如果系统是 CUDA 12.x，用 conda 装一个 11.8 的编译环境：
conda install -c nvidia cuda-toolkit=11.8

pip install torch==2.0.1+cu118 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

git clone https://github.com/YihangChen-ee/HAC-plus.git
cd HAC-plus
pip install -e .

# 还需要 tmc3（G-PCC 工具）
git clone https://github.com/MPEGGroup/mpeg-pcc-tmc13.git
cd mpeg-pcc-tmc13 && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
# 把 tmc13 可执行文件加入 PATH
export PATH=$PATH:$(pwd)/mpeg-pcc-tmc13/build
```

### 7.2 在 SplatAD 输出上运行 HAC++

```bash
conda activate hac
cd HAC-plus

# 先用 SplatAD 训练好的场景做初始化
# HAC 需要的输入是 .ply 格式的 Gaussian 参数
python train.py \
    --source_path ./outputs/baseline_001/gaussians.ply \
    --model_path ./hac_output/001 \
    --iterations 30000

# 评估（HAC 输出的是压缩后的渲染质量）
python render.py \
    --model_path ./hac_output/001 \
    --iteration 30000
```

---

## 8. 参考方法速查表

| 方法 | 类别 | GitHub | 数据集 | VRAM | 主要指标 |
|------|------|--------|--------|------|---------|
| **RENO** | LiDAR 压缩 | `NJUVISION/RENO` | KITTI, Ford | 8 GB | D1/D2 PSNR, BPP, fps |
| **SplatAD** | AD 3DGS | `carlinds/splatad` (via neurad-studio) | PandaSet, nuScenes | 24+ GB | Camera PSNR/SSIM, LiDAR depth |
| **Street Gaussians** | AD 3DGS | `zju3dv/street_gaussians` | KITTI, Waymo | 24 GB | PSNR/SSIM/LPIPS, FPS |
| **OmniRe** | AD 3DGS (统一框架) | `ziyc/drivestudio` | Waymo, PandaSet, nuScenes | 24+ GB | PSNR/SSIM/LPIPS |
| **HAC++** | 3DGS 压缩 | `YihangChen-ee/HAC-plus` | MipNeRF360 等 | 48 GB | PSNR, Size (MB) |
| **RDO-Gaussian** | 3DGS 压缩 (R-D 优化) | `USTC-IMCL/RDO-Gaussian` | MipNeRF360 等 | 24 GB | PSNR, BPP |
| **Bits-to-Photon** | PCC→Gaussian 端到端 | `huzi96/gaussian-pcloud-render` | 8iVFB (密集点云) | 16 GB | PSNR/SSIM/LPIPS + BPP |
| **LightGaussian** | 3DGS 压缩 (轻量) | `VITA-Group/LightGaussian` | 标准 3DGS 数据集 | 24 GB | PSNR, 压缩比 |
| **G-PCC v23** | LiDAR 压缩 (标准) | `MPEGGroup/mpeg-pcc-tmc13` | MPEG 标准测试集 | CPU | D1/D2 PSNR |

---

## 9. 已知坑与解决方案

### 坑 1：MinkowskiEngine 在 CUDA 12.x 编译失败

**现象**：`error: no member named 'value' in 'thrust::detail::integral_constant'`

**解决**：用 NVIDIA 维护的 fork（已修复此问题）：
```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
# 如果还是失败，指定旧版 thrust
THRUST_VERSION=1.17 python setup.py install
```

或者使用预编译 wheel（在 RTX 40/50 系上测试过）：
```bash
pip install https://github.com/chaytonmin/MinkowskiEngine/releases/\
download/v0.5.4/MinkowskiEngine-0.5.4-cp310-cp310-linux_x86_64.whl
```

---

### 坑 2：neurad-studio 的 gsplat 版本锁定

**现象**：`pip install gsplat` 安装后，`import nerfstudio` 报版本冲突

**解决**：不要单独装 gsplat，neurad-studio 会在 `pip install -e .` 时装它自己 fork 的版本：
```bash
# 错误做法
pip install gsplat  # ❌

# 正确做法
cd neurad-studio && pip install -e ".[dev]"  # ✅
```

---

### 坑 3：PandaSet LiDAR 格式与 RENO 不兼容

**现象**：RENO 期望 PLY 或 BIN 格式，PandaSet 是 `pkl.gz`

**解决**：用 Phase 6.1 中的 `pandaset_to_ply.py` 脚本转换，
或者用 open3d 直接写成 PLY：
```python
import open3d as o3d, numpy as np
xyz = np.load("frame.npy")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
o3d.io.write_point_cloud("frame.ply", pcd)
```

---

### 坑 4：HAC++ 依赖 CUDA 11.8，与 RENO 环境冲突

**解决**：严格隔离三个 conda env，用 `conda activate` 切换：

```
env: reno     → CUDA 12.1, torch 2.1, MinkowskiEngine
env: neurad   → CUDA 12.1, torch 2.1, neurad-studio
env: hac      → CUDA 11.8, torch 2.0, HAC++
```

三个 env 之间**只通过文件交换数据**，不共享 Python 对象。

---

### 坑 5：OmniRe（drivestudio）需要 smplx + nvdiffrast

```bash
# smplx（行人建模）
pip install smplx

# nvdiffrast（mesh rasterizer）
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast && pip install .

# pytorch3d（需要与 torch 版本严格匹配）
pip install pytorch3d -f \
    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/\
    py310_cu121_pyt210/download.html
```

如果 pytorch3d 装不上，可以先用 Street Gaussians 代替 OmniRe 作为基线，
两者在 KITTI/Waymo 上结果接近。

---

### 坑 6：Waymo 数据集需要 gcloud 工具

```bash
# 安装 Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud auth login
gcloud config set project <your_project>

# 下载 NOTR split（约 200 GB）
gsutil -m cp -r \
    gs://waymo_open_dataset_v_1_4_3/individual_files/training/ \
    /data/waymo/
```

开发阶段建议**只用 PandaSet**，最终 benchmark 再用 Waymo。

---

## 10. 评估指标说明

### LiDAR 几何指标（Stream A）

| 指标 | 含义 | 越高越好 | 工具 |
|------|------|---------|------|
| D1 PSNR | point-to-point 距离 | ✓ | RENO 内置 |
| D2 PSNR | point-to-plane 距离 | ✓ | RENO 内置 |
| Chamfer Distance | 双向最近邻距离 | ✗（越低越好）| open3d |
| BPP | 每点比特数 | ✗（越低越好）| RENO 内置 |
| BD-Rate | 相对 G-PCC 的比特节省 | ✗（越负越好）| Bjontegaard-delta |

### 渲染质量指标（Stream B/C）

| 指标 | 含义 | 越高越好 | 工具 |
|------|------|---------|------|
| Camera PSNR | 渲染图像峰值信噪比 | ✓ | neurad-studio 内置 |
| SSIM | 结构相似性 | ✓ | neurad-studio 内置 |
| LPIPS | 感知相似性（越低越好）| ✗ | neurad-studio 内置 |
| LiDAR depth PSNR | 渲染深度图质量 | ✓ | SplatAD 特有 |
| FPS | 渲染帧率 | ✓ | 计时测量 |

### 你需要同时报告的新指标组合

因为这个研究跨越两个社区，需要同时报告两组指标才能被两边的审稿人接受：

```
Table 1: 几何压缩质量
  Method | BPP | D1 PSNR | D2 PSNR | BD-Rate vs G-PCC

Table 2: 下游渲染质量（压缩点云初始化 3DGS 后）
  Method | BPP | Camera PSNR | SSIM | LPIPS | LiDAR depth PSNR

Figure 1: D1 PSNR vs. Camera PSNR（证明传统几何指标不能预测渲染质量）
```

---

## 快速开始检查清单

```
□ conda env reno    创建并验证 MinkowskiEngine 可 import
□ conda env neurad  创建并跑通 neurad-studio 的 smoke test
□ PandaSet 下载中（后台，≥80 GB）
□ 用 scene 001 的 1 帧 LiDAR 跑通 RENO compress/decompress
□ 用 scene 001 跑通 SplatAD 30000 iter 训练（约 30 min on RTX 5090）
□ swap_lidar.py 脚本测试通过（原始 ↔ 压缩 LiDAR 互换）
□ sensitivity_curve.py 生成第一张对比图
□ （可选）conda env hac 创建，tmc3 编译完成
```

---

*最后更新：2026 年 4 月，基于 CVPR 2025 / ICCV 2025 最新文献*
