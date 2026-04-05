# Tools

这个目录集中维护 PandaSet、RENO、SplatAD 之间的数据转换和批处理脚本。

## 脚本索引

### 数据转换

- `pandaset_to_ply.py`
  - 将 PandaSet 的 `lidar/*.pkl.gz` 转为 RENO 可读的 ASCII PLY。
- `swap_lidar.py`
  - 简单替换 PandaSet scene 下的 `lidar/` 目录，只保留解码后的 xyz。
  - 适合快速验证目录流转。
- `inject_compressed_lidar.py`
  - 推荐用于 PandaSet 压缩注入。
  - 用 RENO 解码几何替换原始 xyz，并通过最近邻继承原始点属性，保留 `intensity`、时间戳等字段。

### 批量实验

- `run_reno_sweep.sh`
  - 在多个 `posQ` 档位下批量运行 RENO 压缩和解压。
- `run_splatad_sensitivity.sh`
  - 在原始 LiDAR 和多个压缩 LiDAR scene 上批量训练 / 评估 SplatAD。

### 结果分析

- `plot_sensitivity.py`
  - 读取几何指标和渲染指标，绘制 sensitivity curve。
- `visualize_compression.py`
  - 生成原始点云与多档位压缩点云的 BEV、侧视图、误差热力图和统计图。

## 建议用法

建议按下面顺序使用：

1. `pandaset_to_ply.py`
2. `run_reno_sweep.sh`
3. `inject_compressed_lidar.py`
4. `run_splatad_sensitivity.sh`
5. `plot_sensitivity.py`
6. `visualize_compression.py`

如果只是验证目录和文件名是否通顺，可以先用 `swap_lidar.py` 做最小闭环。
