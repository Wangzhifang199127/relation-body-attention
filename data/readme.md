# 数据说明

本目录包含论文所有实验的原始数据与处理数据。所有数据由 `code/` 目录下的脚本生成，未做任何手工修改。

---

## 目录结构

data/
├── raw/ # 原始实验输出
│ ├── batch_scan_results.csv
│ ├── local_pca_results.csv
│ ├── cross_model_results.csv
│ ├── layer11_head_ablation.csv
│ ├── layer11_head_features.csv
│ ├── layer_compare_features.csv
│ ├── ablation_layer.csv
│ ├── ablation_single.csv
│ ├── ablation_patch.csv
│ ├── cumulative_ablation.csv
│ ├── domain_width_results.csv
│ └── multi_seed_results.csv
├── processed/ # 汇总数据（可选）
│ └── summary_tables.csv
└── README.md

---

## 一、原始数据（raw/）

### 1. batch_scan_results.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| layer | int | 层索引（0–11） |
| head | int | 头索引（0–11） |
| rA | float | 方案 A 相关系数 |
| rB | float | 方案 B 相关系数 |
| rC | float | 方案 C 相关系数 |

- **来源**：GPT-2 small 全层全头扫描
- **生成脚本**：`code/analysis/batch_scan.py`
- **规模**：144 行（12 层 × 12 头）

---

### 2. local_pca_results.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| layer | int | 层索引 |
| head | int | 头索引 |
| mean_conc | float | 局部 PC1 集中度均值 |
| r_conc_entropy | float | 集中度与熵的相关系数 |

- **来源**：GPT-2 small 局部 PCA
- **生成脚本**：`code/analysis/local_pca_analysis.py`
- **规模**：144 行

---

### 3. cross_model_results.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| model | str | 模型名称（"gpt2" / "gpt2-medium"） |
| layer | int | 层索引 |
| head | int | 头索引 |
| rA | float | 方案 A 相关系数 |
| mean_conc | float | 局部 PC1 集中度 |
| r_conc_entropy | float | 集中度与熵的相关系数 |
| head_dim | int | 头维度（64） |

- **来源**：GPT-2 small + medium 跨模型实验
- **生成脚本**：`code/analysis/cross_model_analysis.py`
- **规模**：528 行（144 + 384）

---

### 4. layer11_head_ablation.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| head | int | 头索引（0–11） |
| semantic | float | 消融后的语义得分 |
| semantic_delta | float | 语义得分变化量 |

- **来源**：Layer 11 逐头消融
- **生成脚本**：`code/analysis/layer11_analysis.py`
- **规模**：12 行

---

### 5. layer11_head_features.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| head | int | 头索引 |
| mean_entropy | float | 平均注意力熵 |
| mean_conc | float | 平均 PC1 集中度 |

- **来源**：Layer 11 几何特征
- **生成脚本**：`code/analysis/layer11_analysis.py`
- **规模**：12 行

---

### 6. layer_compare_features.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| layer | int | 层索引（0 或 11） |
| head | int | 头索引 |
| mean_entropy | float | 平均熵 |
| mean_conc | float | 平均 PC1 |

- **来源**：Layer 0 vs Layer 11 对比
- **生成脚本**：`code/analysis/layer11_analysis.py`
- **规模**：24 行

---

### 7. ablation_layer.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| layer | int | 消融的层索引 |
| syntax | float | 消融后句法得分 |
| semantic | float | 消融后语义得分 |
| syntax_delta | float | 句法得分变化 |
| semantic_delta | float | 语义得分变化 |

- **来源**：整层消融实验
- **生成脚本**：`code/analysis/ablation_v2.py`
- **规模**：12 行

---

### 8. ablation_single.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| layer | int | 层索引 |
| head | int | 头索引 |
| syntax | float | 消融后句法得分 |
| semantic | float | 消融后语义得分 |
| syntax_delta | float | 句法变化 |
| semantic_delta | float | 语义变化 |

- **来源**：单头消融实验（4 个代表头）
- **生成脚本**：`code/analysis/ablation_v2.py`
- **规模**：4 行

---

### 9. ablation_patch.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| layer | int | 层索引 |
| head | int | 头索引 |
| syntax | float | 修补后句法得分 |
| semantic | float | 修补后语义得分 |
| syntax_delta | float | 句法变化 |
| semantic_delta | float | 语义变化 |

- **来源**：激活修补实验（4 个代表头）
- **生成脚本**：`code/analysis/ablation_v2.py`
- **规模**：4 行

---

### 10. cumulative_ablation.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| k | int | 消融层数 |
| ablated_layers | str | 消融的层列表 |
| syntax | float | 句法得分 |
| semantic | float | 语义得分 |
| syntax_delta | float | 句法变化 |
| semantic_delta | float | 语义变化 |

- **来源**：累积消融实验
- **生成脚本**：`code/analysis/cumulative_ablation.py`
- **规模**：11 行（k = 0–10）

---

### 11. domain_width_results.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| domain | str | 领域（repeat/single/double/multi） |
| layer | int | 层索引 |
| head | int | 头索引 |
| rA | float | 方案 A 相关系数 |
| mean_conc | float | 局部 PC1 集中度 |

- **来源**：受控领域实验
- **生成脚本**：`code/analysis/domain_width_experiment.py`
- **规模**：576 行（4 领域 × 12 层 × 12 头）

---

### 12. multi_seed_results.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| text | str | 输入文本类型（news/code/poetry/dialogue/math） |
| layer | int | 层索引 |
| head | int | 头索引 |
| mean_conc | float | 局部 PC1 集中度 |

- **来源**：多种子/多输入重复实验
- **生成脚本**：`code/utils/plot_figA4.py`
- **规模**：720 行（5 文本 × 12 层 × 12 头）

---

## 二、处理数据（processed/）

### summary_tables.csv

| 字段 | 类型 | 说明 |
|---|---|---|
| table_id | str | 表格编号（如 table01） |
| description | str | 表格描述 |
| value | float | 汇总值 |

- **来源**：从原始数据汇总
- **生成脚本**：`code/utils/summarize.py`
- **用途**：生成正文表格

---

## 三、数据可用性声明

所有数据已公开：

- **GitHub**：`https://github.com/yourname/relation-body`
- **Zenodo DOI**：`10.5281/zenodo.xxxxxxx`（待补充）

数据格式为 CSV，可用 pandas 直接读取。所有数据由 `code/` 目录下脚本生成，可完整复现。

---

## 四、引用

如使用本数据，请引用：

```bibtex
@article{王志方2026relation,
  title={From Pythagoras to Transformers: A Geometric Theory of Attention via the Relation Body},
  author={王志方},
  year={2026}
}
