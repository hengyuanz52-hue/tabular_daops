"""
可视化 train_vortex（或其它使用 SupervisedTrainer / ReptileMetaTrainer 等）的训练结果。

用法:
  # 从 checkpoint 目录下的 history_epoch.csv 绘制
  python visualize_training.py experiments/GNOT-F_VELOCITY-denormal_masking/vortex/checkpoints

  # 或直接指定 CSV 文件
  python visualize_training.py experiments/GNOT-F_VELOCITY-denormal_masking/vortex/checkpoints/history_epoch.csv

  # 保存图片到指定路径
  python visualize_training.py experiments/.../checkpoints -o training_curve.png
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# 从项目内加载 history 读写
from daops.utils.serial import load_history_csv


def to_float_list(values):
    """将 history 中的值转为 Python float 列表，支持 nan。"""
    out = []
    for x in values:
        try:
            v = float(np.asarray(x))
        except (ValueError, TypeError):
            v = np.nan
        out.append(v)
    return out


def plot_history(history, save_path=None, title="Training Loss"):
    """根据 history 字典绘制曲线。支持 'loss' 或 'loss_sim'/'loss_geo'。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    if "loss" in history and history["loss"]:
        losses = to_float_list(history["loss"])
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, "b-", linewidth=1, label="Loss")
    if "loss_sim" in history and history["loss_sim"]:
        loss_sim = to_float_list(history["loss_sim"])
        epochs = range(1, len(loss_sim) + 1)
        ax.plot(epochs, loss_sim, "g-", linewidth=1, label="Loss Sim")
    if "loss_geo" in history and history["loss_geo"]:
        loss_geo = to_float_list(history["loss_geo"])
        epochs = range(1, len(loss_geo) + 1)
        ax.plot(epochs, loss_geo, "orange", linewidth=1, label="Loss Geo")
    if "val_loss" in history and history["val_loss"]:
        val = to_float_list(history["val_loss"])
        epochs = range(1, len(val) + 1)
        ax.plot(epochs, val, "r--", linewidth=1, label="Val Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化训练 history（CSV）")
    parser.add_argument(
        "path",
        type=str,
        help="checkpoint 目录（会查找 history_epoch.csv）或 CSV 文件路径",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="保存图片的路径，不指定则只弹窗显示",
    )
    parser.add_argument(
        "-t", "--title",
        type=str,
        default="Training Loss",
        help="图表标题",
    )
    args = parser.parse_args()

    path = args.path.strip()
    if path.lower().endswith(".csv"):
        csv_path = path
    else:
        csv_path = os.path.join(path, "history_epoch.csv")

    if not os.path.isfile(csv_path):
        print(f"未找到文件: {csv_path}")
        return

    history = load_history_csv(csv_path)
    plot_history(history, save_path=args.output, title=args.title)


if __name__ == "__main__":
    main()
