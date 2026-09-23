"""Matplotlib 中文显示和统一样式。"""

import matplotlib.pyplot as plt


def configure_matplotlib() -> None:
    """配置常见中文字体回退和负号显示。"""

    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

