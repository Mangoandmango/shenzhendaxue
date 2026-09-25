# 华为杯无人机运输与通信协同优化

本项目保存 2026 年中国研究生数学建模竞赛 D 题的公共物理计算层、四问模型、计算结果和论文材料。当前问题一和问题二已经实现并配有独立复核；问题三、四保留配置与模块目录。

## 环境

- Python 3.11 或更高版本
- `openpyxl`：读取赛题 Excel 文件
- `Pillow`：读取 GeoTIFF DEM
- `matplotlib`：生成结果图
- `pytest`：运行公共公式测试

在 PyCharm 中打开本目录，创建虚拟环境后安装依赖：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

也可以将项目安装为可编辑包：

```bash
python -m pip install -e .
huawei-q1
huawei-q2
huawei-q2-soft
huawei-q2b
huawei-q2b-time
huawei-q2b-soft
huawei-q2b-joint --policy ordinary_soft
huawei-q2b-joint --policy all_hard
```

## 运行入口：正式、对比与历史脚本

不要按脚本文件名猜测其结果口径。下表是当前唯一有效的分类；所有正式运行均使用
`O01_WGS84_ENU_SUPERCOVER_V2` 的 ENU—supercover 几何缓存，**不使用 Haversine 距离**。

| 类别 | 脚本/命令 | 用途、输出与注意事项 |
|---|---|---|
| 公共预处理（正式） | `python scripts/prepare_data.py` | 从原始附件重建清洗数据与 ENU 航段缓存。仅在原始节点、DEM 或公共预处理代码改变后运行；随后应重新运行受影响的问题。 |
| 问题一基准（正式） | `python scripts/run_q1.py` | 当前问题一的正式复现入口，输出至 `outputs/q1/`。已调用 ENU 航段、RasterPixelIsPoint 半像元和 supercover DEM 穿越。 |
| 问题一敏感性（正式补充） | `python scripts/run_q1_sensitivity.py` | 在不同返航安全余量下独立重算，输出至 `outputs/q1/sensitivity/`；不替代问题一基准结果。 |
| 问题二 A-S（正式基线） | `python scripts/run_q2_soft.py` | 方案 A，普通物资期望送达时间为软窗；输出至 `outputs/q2_soft/`。它既是正式基线，也是 B+ 的热启动来源。 |
| 问题二 B+-S（正式强化） | `python scripts/run_q2b_joint.py --policy ordinary_soft` | 强化方案 B+，普通物资期望时间为软窗；输出至 `outputs/q2b_joint_soft/`。 |
| 问题二 B+-H（正式强化） | `python scripts/run_q2b_joint.py --policy all_hard` | 强化方案 B+，普通物资期望时间也为硬窗；输出至 `outputs/q2b_joint_hard/`。 |
| 问题二软/硬对比（诊断） | `python scripts/run_q2b_joint_comparison.py` | 把两次 B+ 的候选运输结构置于同一候选池，以两种时间窗分别 MILP 复评；输出至 `outputs/q2b_joint_comparison/`。用于控制候选覆盖差异，不取代两次正式独立运行。 |
| 问题三入口 | `python scripts/run_q3.py` | 问题三通信视线与中继基线；与问题一、二的正式结果无关，当前不应因运行问题一、二而自动重跑。 |
| 历史基线/消融 | `run_q2.py`、`run_q2b.py`、`run_q2b_time.py`、`run_q2b_soft.py` | 保留以复现早期方案、B0--B3 消融和迁移核验；**不得**作为 A-S、B+-S、B+-H 的正式最终结果入口。 |

问题二的正式复现顺序为：先确认 ENU 缓存已经由 `prepare_data.py` 构建，再运行 A-S、B+-S、B+-H，最后按需要运行共享候选池比较。B+-S 与 B+-H 的正式结果必须分别读取各自目录的 `run_metadata.json`、`tables/MILP逐层求解记录.csv` 与最终方案表；限时 MILP 的 `MIP gap` 仅反映固定候选运输结构下的排程界差，不能被写成整体全局最优差距。

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

## 当前问题二口径

正式比较只保留三种情况：

1. **A-S**：方案 A；医疗物资期望送达时间与首批保障截止时间为硬约束，普通物资期望送达时间为软时间窗。
2. **B+-S**：强化版方案 B；时间窗口径与 A-S 相同。
3. **B+-H**：强化版方案 B；所有物资期望送达时间均为硬时间窗，首批保障货箱还同时受首批截止时间限制，取较早者。

A-S 由 `python scripts/run_q2_soft.py` 运行。B+-S 与 B+-H 统一由 `python scripts/run_q2b_joint.py` 运行，仅通过 `--policy ordinary_soft` 或 `--policy all_hard` 切换 MILP 时间窗口径。两版 B+ 共用旧 B3 的强 ALNS 外层：方案 A 热启动、7 类基础破坏算子、4 类关键资源链破坏算子、6 类修复算子、关键链邻域、Top-K 事件驱动筛选、随机种子和计算预算。旧 B3 的 23 架次路线也作为两版不可缺少的候选结构。

B+ 采用“两层协调”求解：外层 B3-ALNS 搜索货箱组批、服务区访问次序和架次边界，并以事件驱动排程作 Top-K 快速筛选；内层连续时间 MILP 对 Top-K、方案 A 热启动和旧 B3 路线同时决定可行机型、实体无人机、共享电池、架次先后关系与开始时刻。MILP 对已形成的运输结构不再使用贪心资源解码。MILP 排程形成的完工关键链与电池关键链会反馈为架次切分和跨架次服务区移动邻域，再由 MILP 复评。

目标按字典序逐层求解，不使用任意大权重：硬违约数、硬迟到、普通物资加权软迟到（仅软口径）、完工时间、总能耗。电池从架次开始占用到返航后充至 100%，无人机返航后即可换用同型号满电电池继续执行任务。默认优先使用 Gurobi；不可用时回退到 SciPy/HiGHS。限时求解时必须连同每层状态、对偶界和 MIP gap 报告，不能把限时可行解表述为全局最优解。

正式配置位于 `configs/q2b_joint_soft.toml` 与 `configs/q2b_joint_hard.toml`，两者搜索预算保持一致；独立搜索结果分别写入 `outputs/q2b_joint_soft/` 与 `outputs/q2b_joint_hard/`。为避免启发式候选覆盖差异干扰软/硬口径比较，还要把两次独立运行的运输结构合并为共享候选池，在两种口径下分别用 MILP 复评；公平比较结果写入 `outputs/q2b_joint_comparison/`。每次运行输出最终运输、无人机、电池和逐箱时序，MILP 逐层求解记录、反馈候选记录、独立复核与 `run_metadata.json`。

`run_q2.py`、`run_q2b.py`、`run_q2b_time.py` 和 `run_q2b_soft.py` 保留为历史基线、消融与迁移核验入口，不再作为上述三种正式方案的结果入口。
