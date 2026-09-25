# plot_figA2.py
"""
生成 figA2：所有层/头的 rA 分布直方图。
数据来源：data/raw/cross_model_results.csv
输出：figures/appendix/figA2_ra_distribution.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ 配置 ============
DATA_PATH = "data/raw/cross_model_results.csv"
OUTPUT_DIR = "figures/appendix"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "figA2_ra_distribution.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

# ============ 加载数据 ============
df = pd.read_csv(DATA_PATH)

# 过滤有效 rA
df_small = df[(df["model"] == "gpt2") & df["rA"].notna()]
df_medium = df[(df["model"] == "gpt2-medium") & df["rA"].notna()]

print(f"GPT-2 small: {len(df_small)} heads, "
      f"mean rA = {df_small['rA'].mean():.4f}, "
      f"negative fraction = {(df_small['rA'] < 0).mean():.2%}")
print(f"GPT-2 medium: {len(df_medium)} heads, "
      f"mean rA = {df_medium['rA'].mean():.4f}, "
      f"negative fraction = {(df_medium['rA'] < 0).mean():.2%}")

# ============ 绘图 ============
fig, ax = plt.subplots(figsize=(8, 5))

bins = np.linspace(-1.0, 0.2, 30)

ax.hist(df_small["rA"], bins=bins, alpha=0.6,
        color="steelblue", label=f"GPT-2 small (n={len(df_small)})",
        edgecolor="white", linewidth=0.5)
ax.hist(df_medium["rA"], bins=bins, alpha=0.5,
        color="coral", label=f"GPT-2 medium (n={len(df_medium)})",
        edgecolor="white", linewidth=0.5)

# 均值线
ax.axvline(df_small["rA"].mean(), color="steelblue",
           linestyle="--", linewidth=1.5,
           label=f"mean (small) = {df_small['rA'].mean():.3f}")
ax.axvline(df_medium["rA"].mean(), color="coral",
           linestyle="--", linewidth=1.5,
           label=f"mean (medium) = {df_medium['rA'].mean():.3f}")

# 零线
ax.axvline(0, color="black", linestyle=":", linewidth=1.0,
           label="rA = 0")

ax.set_xlabel(r"$r_A = \mathrm{corr}(p_{ij}, \Delta_{ij})$")
ax.set_ylabel("Count (layer, head) pairs")
ax.set_title("Distribution of local redundancy correlation across all heads")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_PATH}")