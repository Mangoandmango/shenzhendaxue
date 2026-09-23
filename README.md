# 华为杯无人机运输与通信协同优化

本项目保存 2026 年中国研究生数学建模竞赛 D 题的公共物理计算层、四问模型、计算结果和论文材料。当前问题一已经实现并验证；问题二至四保留配置与模块目录，待模型确定后继续实现。

## 环境

- Python 3.11 或更高版本
- `openpyxl`：读取赛题 Excel 文件
- `Pillow`：读取 GeoTIFF DEM
- `matplotlib`：生成结果图
- `pytest`：运行公共公式测试

在 PyCharm 中打开本目录，创建虚拟环境后执行：

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/run_q1.py
python scripts/run_q1_sensitivity.py
python -m unittest discover -s tests -v
```

也可以将项目安装为可编辑包：

```bash
python -m pip install -e .
huawei-q1
```

## 数据纪律

- `data/raw/` 保存原始赛题和附件，程序不向该目录写入文件。
- `data/processed/` 保存清洗后数据。
- `data/cache/` 保存 DEM 航段、可见性和候选路线等可重建缓存。
- `outputs/q*/tables/` 与 `outputs/q*/figures/` 保存论文可追溯结果。

运行 `python scripts/prepare_data.py` 后，`data/processed/` 会生成节点、逐箱货箱、运输与中继资源、通信参数、需求汇总等标准 CSV，以及 `data_audit.json`。节点表同时保留原始 WGS84 经纬度和以 O01 为原点的 ENU（东、北、天）米制坐标；其坐标口径记录在 `coordinate_reference.json`。`data/cache/leg_geometry.csv` 预先保存 O01 和 15 个服务区的 240 条有向航段距离、沿线最高高程、巡航海拔和 DEM 像元数；`leg_terrain_profile.csv` 按航段穿越顺序保存完整 DEM 剖面，供问题三的通信视线判定复用。

## 当前问题一口径

载荷—航程采用题面给出的 3/2 次关系。单点往返任务去程载荷为该架次全部货箱质量，返程载荷为 0。水平能耗按可用电量与等效航程折算，爬升能耗按重力势能除以爬升效率计算。组批先精确最小化往返架次数，再在该固定最少架次数下计算“总能耗—累计作业时间”的 Pareto 前沿；不人为指定能耗与时间的加权系数。程序同时导出 ε-约束口径的代表解。

运行后还会生成架次数下界验证、候选支配筛选统计、逐架次独立复核、18—20 架次权衡和返航余量精确临界点等可审计表。最大安全载荷表同时标记额定载重或能源安全余量这一受限原因。任何正式提交前都应核对赛题是否发布了更具体的水平能耗或爬升能耗补充公式。

`python scripts/run_q1_sensitivity.py` 默认对 10% 至 40% 的七个返航安全余量情景独立求解，并在 `outputs/q1/sensitivity/` 输出各机型安全载荷变化、服务区组批变化、总体指标汇总及两张敏感性图。
