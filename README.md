# 3DGS-POINTCLOUD

这是一个面向自动驾驶场景的研究工作区，用于复现并扩展
“LiDAR 点云压缩 × 3D Gaussian Splatting 渲染质量”方向的实验。

当前仓库采用“顶层工作区 + 上游子模块”的组织方式：

- `RENO/`: LiDAR 点云压缩基线
- `neurad-studio/`: NeuRAD / SplatAD 训练与评估代码
- `torchsparse/`: 稀疏卷积依赖
- `mind_map/`: 调研文档与阶段性工作总结
- `tools/`: 数据转换、批量实验、结果绘图脚本
- `REPRODUCE.md`: 端到端复现流程
- `mind_map/compass_artifact_学术调查.md`: 方向调研与论文对比
- `mind_map/复现后工作.md`: 当前阶段复现工作总结

## 克隆

首次克隆请连同子模块一起拉取：

```bash
git clone --recurse-submodules https://github.com/CCdydy/3DGS-POINTCLOUD.git
cd 3DGS-POINTCLOUD
```

如果已经克隆过主仓库，再执行：

```bash
git submodule update --init --recursive
```

## 工作区说明

这个仓库本身主要维护：

- 研究路线和复现文档
- PandaSet 与 RENO / SplatAD 之间的数据转换脚本
- 批量 sweep 与 sensitivity 分析脚本
- 第三方依赖的版本固定点

大体实验流程见 [REPRODUCE.md](REPRODUCE.md)。

## 目录概览

```text
3DGS-POINTCLOUD/
├── README.md
├── REPRODUCE.md
├── mind_map/
├── tools/
├── data/             # 本地数据
├── results/          # 本地评估与可视化结果
├── outputs/          # 本地训练输出
├── RENO/             # 子模块
├── neurad-studio/    # 子模块
└── torchsparse/      # 子模块
```

## 工具脚本

常用脚本汇总见 [tools/README.md](tools/README.md)。

## 数据与结果

以下目录默认不纳入版本控制：

- `data/`
- `results/`
- `outputs/`

请将原始数据集、解码结果、训练输出和评估结果放在这些目录下。
