# plot_figA4.py
"""
生成 figA4：多种子/多输入重复实验的 PC1 对比。
数据来源：实时跑 5 种输入文本。
输出：figures/appendix/figA4_multi_seed.png
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ 配置 ============
MODEL_NAME = "gpt2"
TOP_K_LOCAL = 10
OUTPUT_DIR = "figures/appendix"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "figA4_multi_seed.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 5 种不同输入文本（不同主题、长度、结构）
TEXTS = {
    "news": (
        "The stock market rose today as investors reacted to positive "
        "earnings reports from major technology companies. The Dow Jones "
        "Industrial Average gained three hundred points, while the Nasdaq "
        "Composite rose by two percent. Analysts expect continued growth."
    ),
    "code": (
        "def factorial(n):\n"
        "    if n == 0:\n"
        "        return 1\n"
        "    return n * factorial(n - 1)\n\n"
        "def fibonacci(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)"
    ),
    "poetry": (
        "The rose blooms in the garden fair, "
        "Its petals soft with morning dew. "
        "The poet writes with gentle care, "
        "Of love and loss and skies so blue."
    ),
    "dialogue": (
        "She said, 'I love you.' "
        "He replied, 'I love you too.' "
        "'Then why did you leave?' she asked. "
        "'I had to,' he answered softly, 'I had no choice.' "
        "They stood in silence for a long moment."
    ),
    "math": (
        "The integral of x squared is x cubed over three. "
        "The derivative of sine is cosine. "
        "The matrix has eigenvalues that satisfy the characteristic "
        "polynomial. The Fourier transform decomposes a signal into "
        "its frequency components."
    ),
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})


# ============ 加载模型 ============
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

n_layer = model.config.n_layer
n_head = model.config.n_head
n_embd = model.config.n_embd
head_dim = n_embd // n_head


# ============ 单次分析函数 ============
def analyze_text(text, text_name):
    """对一个输入文本，计算所有层/头的 PC1"""
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    qkv_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            qkv_store[layer_idx] = out.detach()
        return hook_fn

    hooks = []
    for l in range(n_layer):
        h = model.transformer.h[l].attn.c_attn.register_forward_hook(
            make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    records = []
    for layer in range(n_layer):
        qkv = qkv_store[layer][0]
        q, k, v = qkv.split(n_embd, dim=-1)
        q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
        k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
        v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

        for head in range(n_head):
            q_h, k_h, v_h = q[head], k[head], v[head]

            scale = 1.0 / np.sqrt(head_dim)
            scores = (q_h @ k_h.T) * scale
            mask = torch.tril(torch.ones(seq_len, seq_len))
            scores = scores.masked_fill(mask == 0, float('-inf'))
            p = torch.softmax(scores, dim=-1)

            conc = np.full(seq_len, np.nan)
            for i in range(seq_len):
                k_actual = min(TOP_K_LOCAL, i + 1)
                if k_actual < 3:
                    continue
                p_i = p[i, :i + 1]
                top_idx = torch.topk(p_i, k_actual).indices
                V_top = v_h[top_idx].detach().cpu().numpy()
                V_c = V_top - V_top.mean(axis=0, keepdims=True)
                try:
                    _, S, _ = np.linalg.svd(V_c, full_matrices=False)
                    e = S ** 2
                    if e.sum() > 1e-12:
                        conc[i] = e[0] / e.sum()
                except Exception:
                    pass

            mean_conc = float(np.nanmean(conc))
            records.append({
                "text": text_name,
                "layer": layer,
                "head": head,
                "mean_conc": mean_conc,
            })

    return pd.DataFrame(records)


# ============ 运行所有文本 ============
print("\n" + "=" * 60)
all_results = []
for text_name, text in TEXTS.items():
    print(f"\nAnalyzing text: {text_name}")
    inputs = tokenizer(text, return_tensors="pt")
    print(f"  Seq length: {inputs['input_ids'].shape[1]}")
    df = analyze_text(text, text_name)
    all_results.append(df)

df_all = pd.concat(all_results, ignore_index=True)
csv_path = os.path.join("data/raw", "multi_seed_results.csv")
os.makedirs("data/raw", exist_ok=True)
df_all.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ============ 汇总 ============
print("\n===== Summary =====")
summary = df_all.groupby('text')['mean_conc'].agg(['mean', 'std', 'count'])
print(summary)


# ============ 可视化 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：按文本分组的箱线图
ax = axes[0]
text_order = list(TEXTS.keys())
data_by_text = [df_all[df_all['text'] == t]['mean_conc'].values
                for t in text_order]
bp = ax.boxplot(data_by_text, labels=text_order, patch_artist=True)

colors = ['steelblue', 'coral', 'green', 'orange', 'purple']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.axhline(1/head_dim, color='red', linestyle='--', linewidth=1,
           label=f'random baseline = 1/{head_dim}')
ax.axhline(df_all['mean_conc'].mean(), color='black', linestyle=':',
           linewidth=1, label=f'overall mean = {df_all["mean_conc"].mean():.3f}')

ax.set_xlabel("Input text type")
ax.set_ylabel("Mean PC1 concentration")
ax.set_title("PC1 distribution across different input texts")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')

# 右图：按文本的层趋势
ax = axes[1]
for t, color in zip(text_order, colors):
    sub = df_all[df_all['text'] == t]
    layer_mean = sub.groupby('layer')['mean_conc'].mean()
    ax.plot(layer_mean.index, layer_mean.values, 'o-',
            label=t, color=color, markersize=4)

ax.axhline(1/head_dim, color='red', linestyle='--', linewidth=1,
           label=f'random = 1/{head_dim}')
ax.set_xlabel("Layer")
ax.set_ylabel("Mean PC1")
ax.set_title("Layer trend across different input texts")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_PATH}")


# ============ 稳健性检验 ============
print("\n===== Robustness Check =====")
overall_mean = df_all['mean_conc'].mean()
overall_std = df_all['mean_conc'].std()
text_means = df_all.groupby('text')['mean_conc'].mean()
text_stds = df_all.groupby('text')['mean_conc'].std()

print(f"Overall mean PC1: {overall_mean:.4f} ± {overall_std:.4f}")
print(f"Text-level means:")
for t in text_order:
    print(f"  {t:<10}: {text_means[t]:.4f} ± {text_stds[t]:.4f}")
print(f"Text-level range: {text_means.min():.4f} – {text_means.max():.4f}")
print(f"All above random baseline: {(text_means > 1/head_dim).all()}")