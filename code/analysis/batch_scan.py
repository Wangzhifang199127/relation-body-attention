# batch_scan.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.stats import pearsonr

# ============ 配置 ============
MODEL_NAME = "gpt2"
TOP_K = 5
TEXT = ("The cat sat on the mat. It was a sunny day and the cat wanted "
        "to play outside. The dog ran to the park and saw a bird.")
OUTPUT_DIR = "attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============ 加载模型 ============
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

inputs = tokenizer(TEXT, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]
n_layer = model.config.n_layer
n_head = model.config.n_head
n_embd = model.config.n_embd
head_dim = n_embd // n_head

print(f"Layers: {n_layer}, Heads: {n_head}, Seq: {seq_len}")


# ============ 一次前向，收集所有层的 QKV ============
qkv_store = {}

def make_hook(layer_idx):
    def hook_fn(module, inp, out):
        qkv_store[layer_idx] = out.detach()
    return hook_fn

hooks = []
for l in range(n_layer):
    h = model.transformer.h[l].attn.c_attn.register_forward_hook(make_hook(l))
    hooks.append(h)

with torch.no_grad():
    model(**inputs)

for h in hooks:
    h.remove()

print(f"Collected QKV for {len(qkv_store)} layers")


# ============ 核心分析函数 ============
def analyze(layer_idx, head_idx):
    """
    返回 (rA, rB, rC) 三个相关系数。
    若某层/头数据异常，返回 (nan, nan, nan)。
    """
    qkv = qkv_store[layer_idx][0]
    q, k, v = qkv.split(n_embd, dim=-1)

    q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
    k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
    v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

    q_h = q[head_idx]
    k_h = k[head_idx]
    v_h = v[head_idx]

    # 注意力概率
    scale = 1.0 / np.sqrt(head_dim)
    scores = (q_h @ k_h.T) * scale
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    p = torch.softmax(scores, dim=-1)
    mask_bool = mask.bool()

    # 方案 A
    V_bar = p @ v_h
    delta = torch.norm(v_h.unsqueeze(0) - V_bar.unsqueeze(1), dim=-1)
    p_flat = p[mask_bool].numpy()
    delta_flat = delta[mask_bool].numpy()

    if np.std(p_flat) < 1e-8 or np.std(delta_flat) < 1e-8:
        rA = np.nan
    else:
        rA, _ = pearsonr(p_flat, delta_flat)

    # 方案 B / C
    V_np = v_h.numpy()
    U, S, Vt = np.linalg.svd(V_np, full_matrices=False)
    k_use = min(TOP_K, Vt.shape[0])
    top_components = Vt[:k_use]

    energy_ratio = np.zeros(seq_len)
    residual_norm = np.zeros(seq_len)
    for j in range(seq_len):
        vj = V_np[j]
        coords = vj @ top_components.T
        proj = coords @ top_components
        energy_ratio[j] = np.sum(proj ** 2) / (np.sum(vj ** 2) + 1e-12)
        residual_norm[j] = np.linalg.norm(vj - proj)

    # 每个 token 的因果平均注意力
    avg_p = np.array([p[j:, j].mean().item() for j in range(seq_len)])

    if np.std(avg_p) < 1e-8:
        rB = rC = np.nan
    else:
        rB, _ = pearsonr(avg_p, energy_ratio)
        rC, _ = pearsonr(avg_p, residual_norm)

    return rA, rB, rC


# ============ 批量扫描 ============
print("\nScanning all layers and heads...")
records = []
for layer in range(n_layer):
    for head in range(n_head):
        rA, rB, rC = analyze(layer, head)
        records.append({
            "layer": layer,
            "head": head,
            "rA": rA,
            "rB": rB,
            "rC": rC,
        })
    print(f"  Layer {layer} done")

df = pd.DataFrame(records)
csv_path = os.path.join(OUTPUT_DIR, "batch_scan_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ============ 汇总统计 ============
print("\n===== Summary =====")
print(f"Total (layer, head) pairs: {len(df)}")
print(f"\nScheme A (expect negative):")
print(f"  mean rA = {df['rA'].mean():+.4f}")
print(f"  negative fraction = {(df['rA'] < 0).mean():.2%}")
print(f"\nScheme B (expect positive):")
print(f"  mean rB = {df['rB'].mean():+.4f}")
print(f"  positive fraction = {(df['rB'] > 0).mean():.2%}")
print(f"\nScheme C (expect negative):")
print(f"  mean rC = {df['rC'].mean():+.4f}")
print(f"  negative fraction = {(df['rC'] < 0).mean():.2%}")


# ============ 可视化 ============
# 图1：热力图 rA / rB / rC
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

def heatmap(ax, values, title, cmap, vmin, vmax):
    mat = values.reshape(n_layer, n_head)
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

heatmap(axes[0], df['rA'].values, "Scheme A: r(p, Δ)", 'RdBu_r', -1, 1)
heatmap(axes[1], df['rB'].values, "Scheme B: r(p, energy)", 'RdBu_r', -1, 1)
heatmap(axes[2], df['rC'].values, "Scheme C: r(p, residual)", 'RdBu_r', -1, 1)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "batch_scan_heatmaps.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved: {fig_path}")


# 图2：层趋势
fig, ax = plt.subplots(figsize=(10, 6))
layer_mean_A = df.groupby('layer')['rA'].mean()
layer_mean_B = df.groupby('layer')['rB'].mean()
layer_mean_C = df.groupby('layer')['rC'].mean()

ax.plot(layer_mean_A.index, layer_mean_A.values, 'o-', label='A: r(p, Δ)')
ax.plot(layer_mean_B.index, layer_mean_B.values, 's-', label='B: r(p, energy)')
ax.plot(layer_mean_C.index, layer_mean_C.values, '^-', label='C: r(p, residual)')
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel("Layer")
ax.set_ylabel("Mean Pearson r across heads")
ax.set_title("Layer-wise trend")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
trend_path = os.path.join(OUTPUT_DIR, "batch_scan_trend.png")
plt.savefig(trend_path, dpi=150)
plt.show()
print(f"Saved: {trend_path}")

print("\nDone.")