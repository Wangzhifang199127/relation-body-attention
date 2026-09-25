# code/utils/visualization.py
"""
统一绘图风格与工具函数。
所有论文图使用相同字体、颜色、尺寸规范。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============ 全局风格 ============

def setup_style():
    """设置论文级绘图风格。"""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============ 常用色板 ============

COLORS = {
    "small": "steelblue",
    "medium": "coral",
    "high_pc1": "crimson",
    "low_pc1": "steelblue",
    "fit": "red",
    "baseline": "black",
    "reference": "orange",
}

DOMAIN_COLORS = {
    "repeat": "#1f77b4",
    "single": "#2ca02c",
    "double": "#ff7f0e",
    "multi": "#d62728",
}


# ============ 通用绘图函数 ============

def save_figure(fig, path, dpi=300):
    """保存图片，自动创建目录。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")


def plot_pc1_heatmap(pc1_matrix, title="PC1 concentration",
                     output_path=None, vmin=0.2, vmax=0.8):
    """
    绘制 PC1 热力图（层 × 头）。
    pc1_matrix: [n_layer, n_head]
    """
    setup_style()
    n_layer, n_head = pc1_matrix.shape

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(pc1_matrix, aspect="auto", cmap="viridis",
                   vmin=vmin, vmax=vmax, origin="upper")

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(n_head))
    ax.set_yticks(range(0, n_layer, max(1, n_layer // 12)))
    ax.set_yticklabels(range(0, n_layer, max(1, n_layer // 12)))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean PC1 concentration")

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_scheme_scatter(p_flat, delta_flat, r_value, title,
                        xlabel=r"Attention probability $p_{ij}$",
                        ylabel=r"$\Delta_{ij} = \|V_j - \bar{V}_i\|$",
                        output_path=None):
    """方案 A/B/C 的散点图。"""
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(p_flat, delta_flat, alpha=0.35, s=6, color="steelblue")

    z = np.polyfit(p_flat, delta_flat, 1)
    xs = np.linspace(p_flat.min(), p_flat.max(), 100)
    ax.plot(xs, np.polyval(z, xs), "r--", lw=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: r = {r_value:+.3f}")
    ax.grid(alpha=0.3)

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_relation_body(cos_vals, sin_vals, p_vals, output_path=None):
    """
    绘制关系体单位圆（上半圆）。
    cos_vals, sin_vals, p_vals: 长度相同的数组
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(cos_vals, sin_vals, c=p_vals,
                    cmap="viridis", s=12, alpha=0.7)

    theta = np.linspace(0, np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=1)

    ax.set_xlabel(r"$\cos\theta$ (same / dot)")
    ax.set_ylabel(r"$\sin\theta$ (different / wedge)")
    ax.set_title("Relation body (upper unit circle)")
    ax.axis("equal")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("attention p")

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_layer_trend(layer_means_dict, output_path=None,
                     baseline=None, title="Layer trend",
                     ylabel="Mean PC1"):
    """
    绘制层趋势曲线。
    layer_means_dict: {label: array_of_means}
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, values in layer_means_dict.items():
        ax.plot(range(len(values)), values, "o-", label=label, markersize=4)

    if baseline is not None:
        ax.axhline(baseline, color="red", linestyle="--", lw=1,
                   label=f"random baseline = {baseline:.4f}")

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_ra_histogram(ra_dict, output_path=None, bins=30):
    """
    绘制 rA 分布直方图。
    ra_dict: {label: array_of_rA}
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["steelblue", "coral", "green", "orange"]
    for (label, values), color in zip(ra_dict.items(), colors):
        values = np.array(values)
        values = values[~np.isnan(values)]
        ax.hist(values, bins=bins, alpha=0.6, color=color,
                label=f"{label} (n={len(values)})",
                edgecolor="white", linewidth=0.5)
        ax.axvline(values.mean(), color=color, linestyle="--",
                   lw=1.5, label=f"mean = {values.mean():.3f}")

    ax.axvline(0, color="black", linestyle=":", lw=1, label="rA = 0")

    ax.set_xlabel(r"$r_A = \mathrm{corr}(p_{ij}, \Delta_{ij})$")
    ax.set_ylabel("Count (layer, head) pairs")
    ax.set_title("Distribution of local redundancy correlation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_cumulative_ablation(df, output_path=None):
    """
    绘制累积消融曲线。
    df: pandas DataFrame，含 k, syntax_delta, semantic_delta
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["k"], df["syntax_delta"], "o-",
            label="Syntax Δ", color="steelblue")
    ax.plot(df["k"], df["semantic_delta"], "s-",
            label="Semantic Δ", color="coral")
    ax.axhline(0, color="k", linestyle="--", lw=0.8)

    ax.set_xlabel("Number of ablated middle layers (k)")
    ax.set_ylabel("Log-prob delta")
    ax.set_title("Cumulative ablation: compensation collapse")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_domain_width(df, output_path=None):
    """
    绘制受控领域 PC1 柱状图。
    df: pandas DataFrame，含 domain, mean_conc
    """
    setup_style()
    domain_order = ["repeat", "single", "double", "multi"]
    means = [df[df["domain"] == d]["mean_conc"].mean() for d in domain_order]
    stds = [df[df["domain"] == d]["mean_conc"].std() for d in domain_order]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [DOMAIN_COLORS[d] for d in domain_order]
    ax.bar(domain_order, means, yerr=stds, capsize=5, color=colors)

    random_baseline = 1.0 / 64
    ax.axhline(random_baseline, color="red", linestyle="--",
               label=f"random = 1/64 = {random_baseline:.4f}")

    ax.set_ylabel("Mean PC1 concentration")
    ax.set_title("PC1 vs semantic width")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    if output_path:
        save_figure(fig, output_path)
    return fig


# ============ 初始化 ============

setup_style()