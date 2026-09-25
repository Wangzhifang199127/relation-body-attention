# crop_fig01_fig02.py
"""
从四合一图中裁剪出 fig01（关系体）与 fig02（Scheme A/B/C）。
输入：原始四合一图（需先保存为 PNG）
输出：figures/fig01_relation_body.png, figures/fig02_scheme_abc.png
"""

import os
from PIL import Image
import matplotlib.pyplot as plt

# ============ 配置 ============
SOURCE_IMAGE = "figures/raw/four_panel.png"  # 原始四合一图
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============ 加载原始图 ============
img = Image.open(SOURCE_IMAGE)
w, h = img.size
print(f"Source image size: {w} x {h}")

# 四宫格：左上、右上、左下、右下
half_w = w // 2
half_h = h // 2

# ============ 裁剪 ============
# 右下角：关系体图
fig01 = img.crop((half_w, half_h, w, h))
fig01_path = os.path.join(OUTPUT_DIR, "fig01_relation_body.png")
fig01.save(fig01_path, dpi=(300, 300))
print(f"Saved: {fig01_path}")

# 左上 + 右上 + 左下：Scheme A/B/C
# 组合成一张图
fig02 = img.crop((0, 0, w, half_h))  # 上半部分（A + B）
fig02_lower = img.crop((0, half_h, half_w, h))  # 左下角（C）

# 用 matplotlib 组合
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(fig02.crop((0, 0, half_w, half_h)))
axes[0].axis('off')
axes[0].set_title("(a) Scheme A: Local redundancy", fontsize=12)

axes[1].imshow(fig02.crop((half_w, 0, w, half_h)))
axes[1].axis('off')
axes[1].set_title("(b) Scheme B: Global PC energy", fontsize=12)

axes[2].imshow(fig02_lower)
axes[2].axis('off')
axes[2].set_title("(c) Scheme C: Global residual", fontsize=12)

plt.tight_layout()
fig02_path = os.path.join(OUTPUT_DIR, "fig02_scheme_abc.png")
plt.savefig(fig02_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {fig02_path}")