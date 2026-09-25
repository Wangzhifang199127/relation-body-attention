# 从勾股定理到 Transformer：基于关系体的注意力几何理论
# From Pythagoras to Transformers: A Geometric Theory of Attention via the Relation Body

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**作者**：王志方（独立研究者）  
**ORCID**：0009-0001-0374-3825  
**日期**：2026-09-24

---

## 📖 简介

本仓库包含论文《从勾股定理到 Transformer：基于关系体的注意力几何理论》的全部实验代码、数据与图表生成脚本。论文提出一个统一的几何框架——“关系体”（Relation Body），将任意两个向量之间的关系分解为内积（相同/冗余）与外积（不同/独立）两个正交状态，并应用于 Transformer 注意力机制的可解释性研究。

---

## 📄 摘要

**中文摘要**  
理解大型语言模型（LLMs）内部机制是当前人工智能研究的核心挑战之一。本文提出一个统一的几何框架——“关系体”（Relation Body），将任意两个向量之间的关系分解为两个正交状态：内积所度量的“相同/冗余”分量与外积所度量的“不同/独立”分量。二者的归一化满足 \(c^2 + s^2 = 1\)，即单位圆上的勾股定理。我们将这一框架应用于 Transformer 的注意力机制，推导出四个定理：关系体定理（T1）、冗余-概率定理（T2）、无主干定理（T3）与冗余补偿定理（T4）。在 GPT-2 small 与 GPT-2 medium 上的实验表明：注意力机制呈现“全局无主干、局部有叶柄”的灌木丛结构；高 PC1 的注意力头对应句法/位置功能，低 PC1 的头对应语义功能；Layer 0 与最后一层分别是信息入口与语义出口瓶颈，而中间层具有高度冗余补偿能力。进一步的受控输入领域实验直接验证了核心因果机制：输入语义分布越窄，PC1 越高，结构越接近具有主干的“传统树”。基于此，我们预测混合专家（MoE）模型中的单个专家应呈现比稠密模型更强的局部主干结构。本文为理解注意力机制的信息组织方式提供了可计算、可证伪的几何理论，并为大模型的可解释性研究开辟了“几何可解释性”新路径。

**English Abstract**  
Understanding the internal mechanisms of large language models (LLMs) remains a central challenge in artificial intelligence. This paper proposes a unified geometric framework—the **Relation Body**—which decomposes the relationship between any two vectors into two orthogonal states: the “same/redundant” component measured by the inner product, and the “different/independent” component measured by the outer product. Their normalized forms satisfy \(c^2 + s^2 = 1\), i.e., the Pythagorean theorem on the unit circle. We apply this framework to the attention mechanism of Transformers and derive four theorems: the Relation Body Theorem (T1), the Redundancy-Probability Theorem (T2), the No-Trunk Theorem (T3), and the Redundancy Compensation Theorem (T4). Experiments on GPT-2 small and GPT-2 medium reveal that attention exhibits a “shrub” structure: no global trunk, but local stems. High-PC1 heads correspond to syntactic/positional functions, while low-PC1 heads correspond to semantic functions. Layer 0 and the final layer serve as information entry and semantic exit bottlenecks, respectively, while middle layers possess strong redundancy compensation. Controlled-domain experiments directly verify the core causal mechanism: the narrower the input semantic distribution, the higher the PC1, and the closer the structure is to a traditional tree with a trunk. Based on this, we predict that individual experts in Mixture-of-Experts (MoE) models should exhibit stronger local trunk structures than dense models. This work provides a computable, falsifiable geometric theory for understanding how attention organizes information, and opens a new path of “geometric interpretability” for large-model research.

**关键词**：注意力机制，可解释性，关系体，内积与外积，主成分分析，混合专家  
**Keywords**: attention mechanism, interpretability, relation body, inner and outer product, principal component analysis, mixture of experts

---

## 📂 目录结构

```
relation-body-attention/
├── README.md                     # 本文件
├── LICENSE                       # MIT 许可证
├── CITATION.cff                  # 引用信息
├── paper/
│   └── paper.pdf                 # 论文 PDF
├── code/
│   ├── analysis/                 # 核心分析脚本
│   │   ├── batch_scan.py
│   │   ├── local_pca_analysis.py
│   │   ├── cross_model_analysis.py
│   │   ├── head11_deep_analysis.py
│   │   ├── layer11_analysis.py
│   │   ├── ablation_v2.py
│   │   ├── cumulative_ablation.py
│   │   └── domain_width_experiment.py
│   ├── utils/                    # 工具模块
│   │   ├── hooks.py
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   ├── plot_figA1.py
│   │   ├── plot_figA2.py
│   │   ├── plot_figA3.py
│   │   ├── plot_figA4.py
│   │   └── crop_fig01_fig02.py
│   ├── requirements.txt
│   └── reproduce/
│       └── run_all.sh            # 一键复现脚本
├── data/
│   ├── raw/                      # 原始实验数据（CSV）
│   │   ├── batch_scan_results.csv
│   │   ├── local_pca_results.csv
│   │   ├── cross_model_results.csv
│   │   ├── layer11_head_ablation.csv
│   │   ├── layer11_head_features.csv
│   │   ├── layer_compare_features.csv
│   │   ├── ablation_layer.csv
│   │   ├── ablation_single.csv
│   │   ├── ablation_patch.csv
│   │   ├── cumulative_ablation.csv
│   │   ├── domain_width_results.csv
│   │   └── multi_seed_results.csv
│   └── README.md                 # 数据说明
├── figures/
│   ├── fig01_relation_body.png
│   ├── fig02_scheme_abc.png
│   ├── fig03_high_low_pc1.png
│   ├── fig04_layer11_head7.png
│   ├── fig05_cumulative_ablation.png
│   ├── fig06_domain_width.png
│   └── appendix/
│       ├── figA1_gpt2_medium_pc1.png
│       ├── figA2_ra_distribution.png
│       ├── figA3_svd_spectrum.png
│       ├── figA4_multi_seed.png
│       └── figA5_cross_model_trend.png
└── refs.bib                      # 参考文献
```

---

## ⚙️ 环境依赖

- Python ≥ 3.10
- PyTorch ≥ 2.1
- transformers ≥ 4.40
- numpy ≥ 1.26
- pandas ≥ 2.2
- matplotlib ≥ 3.8
- seaborn ≥ 0.13
- scipy ≥ 1.12

安装依赖：

```bash
pip install -r code/requirements.txt
```

**硬件**：单张 NVIDIA T4（16 GB）或更高。所有实验可在 1 小时内完成。

---

## 🚀 快速开始

### 一键复现所有实验

```bash
bash code/reproduce/run_all.sh
```

### 单独运行某个脚本

```bash
python code/analysis/batch_scan.py
```

### 使用工具模块

```python
from utils.hooks import register_qkv_hooks, remove_hooks
from utils.metrics import compute_all_metrics
from utils.visualization import plot_pc1_heatmap, save_figure
```

---

## 📊 数据说明

所有原始实验数据位于 `data/raw/`，均为 CSV 格式，由 `code/` 下的脚本生成。详细字段说明请参阅 `data/README.md`。

---

## 📝 引用

如果本工作对您有帮助，请引用：

```bibtex
@article{wang2026relation,
  title={从勾股定理到 Transformer：基于关系体的注意力几何理论},
  author={王志方},
  year={2026},
  doi={10.5281/zenodo.22952945},
  note={Preprint}
}
```

或英文：

```bibtex
@article{wang2026relation,
  title={From Pythagoras to Transformers: A Geometric Theory of Attention via the Relation Body},
  author={Wang, Zhifang},
  year={2026},
  doi={10.5281/zenodo.22952945},
  note={Preprint}
}
```

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源。

---

## 📧 联系

**王志方**  
独立研究者  
ORCID: [0009-0001-0374-3825](https://orcid.org/0009-0001-0374-3825)  
邮箱：1250191535@qq.com

---

*最后更新：2026-09-24*
