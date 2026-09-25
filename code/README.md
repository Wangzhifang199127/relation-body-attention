# 代码说明（code/README.md）

本目录包含论文《From Pythagoras to Transformers: A Geometric Theory of Attention via the Relation Body》的全部实验代码。所有实验基于 HuggingFace Transformers 实现，可在单张 GPU 上完整复现。

---

## 目录结构

```
code/
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖
├── analysis/                     # 核心分析脚本
│   ├── batch_scan.py
│   ├── local_pca_analysis.py
│   ├── cross_model_analysis.py
│   ├── head11_deep_analysis.py
│   ├── layer11_analysis.py
│   ├── ablation_v2.py
│   ├── cumulative_ablation.py
│   └── domain_width_experiment.py
├── utils/                        # 工具模块与辅助脚本
│   ├── hooks.py                  # QKV 提取、消融、修补
│   ├── metrics.py                # 方案 A/B/C、局部 PCA、熵
│   ├── visualization.py          # 统一绘图风格与函数
│   ├── plot_figA1.py             # 附录图 A1：GPT-2 medium PC1 热力图
│   ├── plot_figA2.py             # 附录图 A2：rA 分布直方图
│   ├── plot_figA3.py             # 附录图 A3：SVD 奇异值谱 vs MP
│   ├── plot_figA4.py             # 附录图 A4：多种子 PC1 稳健性
│   └── crop_fig01_fig02.py       # 裁剪正文图 1 与图 2
└── reproduce/                    # 一键复现脚本
    └── run_all.sh
```

---

## 环境要求

| 依赖 | 版本 |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 |
| transformers | ≥ 4.40 |
| numpy | ≥ 1.26 |
| pandas | ≥ 2.2 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |
| scipy | ≥ 1.12 |

**硬件**：单张 NVIDIA T4（16 GB）或更高。所有实验可在 1 小时内完成。

---

## 安装

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 一键复现所有实验

```bash
bash reproduce/run_all.sh
```

### 单独运行某个脚本

```bash
python analysis/batch_scan.py
```

### 使用工具模块

```python
from utils.hooks import register_qkv_hooks, remove_hooks
from utils.metrics import compute_all_metrics
from utils.visualization import plot_pc1_heatmap, save_figure
```

---

## 脚本详细说明

### analysis/ 核心分析脚本

#### batch_scan.py

- **目的**：全层全头扫描，计算方案 A/B/C 相关系数。
- **输入**：GPT-2 small + 一段英文文本。
- **输出**：`data/raw/batch_scan_results.csv`
- **运行时间**：约 30 秒。

#### local_pca_analysis.py

- **目的**：计算每个头的局部 PCA 集中度与熵的关系。
- **输入**：GPT-2 small + 英文文本。
- **输出**：`data/raw/local_pca_results.csv`
- **运行时间**：约 1 分钟。

#### cross_model_analysis.py

- **目的**：在 GPT-2 small 与 medium 上重复分析。
- **输入**：两个模型 + 英文文本。
- **输出**：`data/raw/cross_model_results.csv`
- **运行时间**：约 3 分钟。

#### head11_deep_analysis.py

- **目的**：高 PC1 头（L1 H11）专项分析（注意力矩阵、Top-1 分布、熵曲线）。
- **输入**：GPT-2 small + 英文文本。
- **输出**：`figures/fig03_high_low_pc1.png`
- **运行时间**：约 30 秒。

#### layer11_analysis.py

- **目的**：Layer 11 逐头消融与几何特征分析。
- **输入**：GPT-2 small + 语义测试集。
- **输出**：
  - `data/raw/layer11_head_ablation.csv`
  - `data/raw/layer11_head_features.csv`
  - `data/raw/layer_compare_features.csv`
  - `figures/fig04_layer11_head7.png`
- **运行时间**：约 2 分钟。

#### ablation_v2.py

- **目的**：单头消融、整层消融、激活修补。
- **输入**：GPT-2 small + 60 句法 + 60 语义测试集。
- **输出**：
  - `data/raw/ablation_layer.csv`
  - `data/raw/ablation_single.csv`
  - `data/raw/ablation_patch.csv`
- **运行时间**：约 5 分钟。

#### cumulative_ablation.py

- **目的**：累积消融，寻找相变点。
- **输入**：GPT-2 small + 测试集。
- **输出**：
  - `data/raw/cumulative_ablation.csv`
  - `figures/fig05_cumulative_ablation.png`
- **运行时间**：约 3 分钟。

#### domain_width_experiment.py

- **目的**：受控领域实验（repeat/single/double/multi）。
- **输入**：GPT-2 small + 四种语义宽度文本。
- **输出**：
  - `data/raw/domain_width_results.csv`
  - `figures/fig06_domain_width.png`
- **运行时间**：约 2 分钟。

---

### utils/ 工具模块与辅助脚本

#### hooks.py

- **功能**：封装 QKV 提取、消融、激活修补的 Hook。
- **主要类**：
  - `QKVHook`：提取 `c_attn` 输出的 Q/K/V。
  - `AttentionOutputHook`：消融指定头或整层。
  - `PatchHook`：将指定头输出替换为均值。
- **主要函数**：
  - `register_qkv_hooks(model, n_layer)`：注册所有层的 QKV Hook。
  - `remove_hooks(handles)`：移除所有 Hook。

#### metrics.py

- **功能**：封装关系体坐标、方案 A/B/C、局部 PCA、熵等度量。
- **主要函数**：
  - `relation_coords(a, b)`：计算关系体坐标 \((c, s)\)。
  - `wedge_norm_squared(a, b)`：高维外积范数平方。
  - `compute_attention(q_h, k_h, head_dim, causal=True)`：注意力概率。
  - `attention_entropy(p)`：注意力熵。
  - `compute_scheme_A(p, v_h)`：方案 A 相关系数。
  - `compute_scheme_BC(p, v_h, top_k=5)`：方案 B/C 相关系数。
  - `compute_local_pca(p, v_h, top_k=10)`：局部 PC1 集中度。
  - `compute_all_metrics(...)`：一次性计算所有度量。

#### visualization.py

- **功能**：统一论文绘图风格，提供常用绘图函数。
- **主要函数**：
  - `setup_style()`：设置全局字体、尺寸、色板。
  - `save_figure(fig, path)`：保存图片。
  - `plot_pc1_heatmap(...)`：PC1 热力图。
  - `plot_scheme_scatter(...)`：方案 A/B/C 散点图。
  - `plot_relation_body(...)`：关系体单位圆。
  - `plot_layer_trend(...)`：层趋势曲线。
  - `plot_ra_histogram(...)`：rA 分布直方图。
  - `plot_cumulative_ablation(...)`：累积消融曲线。
  - `plot_domain_width(...)`：受控领域柱状图。

#### plot_figA1.py

- **目的**：生成 GPT-2 medium 的 PC1 热力图。
- **输入**：`data/raw/cross_model_results.csv`
- **输出**：`figures/appendix/figA1_gpt2_medium_pc1.png`

#### plot_figA2.py

- **目的**：生成 rA 分布直方图。
- **输入**：`data/raw/cross_model_results.csv`
- **输出**：`figures/appendix/figA2_ra_distribution.png`

#### plot_figA3.py

- **目的**：生成 SVD 奇异值谱 vs Marchenko-Pastur 曲线。
- **输入**：实时从 GPT-2 提取 QKV。
- **输出**：`figures/appendix/figA3_svd_spectrum.png`

#### plot_figA4.py

- **目的**：生成多种子 PC1 稳健性检验图。
- **输入**：5 种不同文本。
- **输出**：
  - `data/raw/multi_seed_results.csv`
  - `figures/appendix/figA4_multi_seed.png`

#### crop_fig01_fig02.py

- **目的**：从四合一图中裁剪出 fig01 与 fig02。
- **输入**：`figures/raw/four_panel.png`
- **输出**：
  - `figures/fig01_relation_body.png`
  - `figures/fig02_scheme_abc.png`

---

### reproduce/ 一键复现脚本

#### run_all.sh

```bash
#!/bin/bash
set -e

echo "=== Running all experiments ==="

python analysis/batch_scan.py
python analysis/local_pca_analysis.py
python analysis/cross_model_analysis.py
python analysis/head11_deep_analysis.py
python analysis/layer11_analysis.py
python analysis/ablation_v2.py
python analysis/cumulative_ablation.py
python analysis/domain_width_experiment.py

echo "=== Generating appendix figures ==="

python utils/plot_figA1.py
python utils/plot_figA2.py
python utils/plot_figA3.py
python utils/plot_figA4.py
python utils/crop_fig01_fig02.py

echo "=== All done ==="
```

---

## 数据依赖

所有脚本依赖 `data/raw/` 下的 CSV 文件，具体字段与来源请参阅 `data/README.md`。

---

## 注意事项

1. **模型下载**：首次运行会自动下载 GPT-2，需网络连接。模型缓存于 `~/.cache/huggingface/`。
2. **GPU**：建议使用 GPU，CPU 也可运行但较慢。
3. **随机性**：所有实验使用 `model.eval()`，不涉及训练，无随机种子问题。
4. **路径**：脚本使用相对路径，请在项目根目录下运行。
5. **数据覆盖**：重新运行会覆盖 `data/raw/` 下的 CSV 文件。

---

## 引用

如使用本代码，请引用：

```bibtex
@article{王志方2026relation,
  title={From Pythagoras to Transformers: A Geometric Theory of Attention via the Relation Body},
  author={王志方},
  year={2026}
}
```

---

## 联系
**王志方**  
独立研究者  
ORCID: [0009-0001-0374-3825](https://orcid.org/0009-0001-0374-3825)  
邮箱：1250191535@qq.com

---

*最后更新：2026-09-24*
独立研究者，联系方式待补充。