# plot_figA3.py
"""
生成 figA3：SVD 奇异值谱 vs Marchenko-Pastur 理论曲线。
数据来源：实时从 GPT-2 提取 QKV。
输出：figures/appendix/figA3_svd_spectrum.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ 配置 ============
MODEL_NAME = "gpt2"
TARGET_LAYER = 1
TARGET_HEAD = 11
OUTPUT_DIR = "figures/appendix"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "figA3_svd_spectrum.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT = ("The cat sat on the mat. It was a sunny day and the cat wanted "
        "to play outside. The dog ran to the park and saw a bird. "
        "The bird flew over the house and landed on the roof. "
        "She watched the bird from the window and smiled.")

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

inputs = tokenizer(TEXT, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]

n_embd = model.config.n_embd
n_head = model.config.n_head
head_dim = n_embd // n_head

print(f"Sequence length: {seq_len}, head_dim: {head_dim}")


# ============ 提取 QKV ============
qkv_store = {}

def hook_fn(module, inp, out):
    qkv_store['out'] = out.detach()

hook = model.transformer.h[TARGET_LAYER].attn.c_attn.register_forward_hook(hook_fn)
with torch.no_grad():
    model(**inputs)
hook.remove()

qkv = qkv_store['out'][0]
q, k, v = qkv.split(n_embd, dim=-1)
q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

v_h = v[TARGET_HEAD].cpu().numpy()  # [seq_len, head_dim]

print(f"V matrix shape: {v_h.shape}")


# ============ SVD ============
V_centered = v_h - v_h.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(V_centered, full_matrices=False)

# 归一化奇异值
S_normalized = S ** 2 / (S ** 2).sum()
print(f"Top-5 normalized singular values: {S_normalized[:5]}")
print(f"PC1 concentration: {S_normalized[0]:.4f}")


# ============ Marchenko-Pastur 理论曲线 ============
def marchenko_pastur(x, gamma, sigma2=1.0):
    """MP 分布密度"""
    lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    mask = (x >= lambda_minus) & (x <= lambda_plus)
    result = np.zeros_like(x)
    result[mask] = np.sqrt((lambda_plus - x[mask]) * (x[mask] - lambda_minus)) \
                   / (2 * np.pi * sigma2 * gamma * x[mask])
    return result


# 估计 gamma = d_h / n
gamma = head_dim / seq_len
print(f"gamma = d_h / n = {gamma:.4f}")

# 估计 sigma^2（从经验方差）
sigma2 = V_centered.var()
print(f"sigma^2 = {sigma2:.4f}")

# 理论谱
lambda_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
lambda_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2

x_theory = np.linspace(lambda_minus, lambda_plus, 500)
mp_theory = marchenko_pastur(x_theory, gamma, sigma2)


# ============ 绘图 ============
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：经验奇异值 vs MP 理论
ax = axes[0]
ax.bar(range(len(S)), S ** 2, color='steelblue', alpha=0.7,
       label='Empirical singular values$^2$')
ax.axhline(lambda_plus, color='red', linestyle='--', linewidth=1.5,
           label=f'MP upper bound $\\lambda_+ = {lambda_plus:.2f}$')
ax.axhline(lambda_minus, color='orange', linestyle=':', linewidth=1.5,
           label=f'MP lower bound $\\lambda_- = {lambda_minus:.2f}$')

# 标记 PC1
ax.bar(0, S[0]**2, color='crimson', alpha=0.9, label=f'PC1 = {S_normalized[0]:.3f}')

ax.set_xlabel("Singular value index")
ax.set_ylabel("Squared singular value")
ax.set_title(f"SVD spectrum: Layer {TARGET_LAYER}, Head {TARGET_HEAD}")
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# 右图：归一化奇异值累积分布
ax = axes[1]
cumulative = np.cumsum(S_normalized)
ax.plot(range(1, len(cumulative)+1), cumulative, 'o-',
        color='steelblue', markersize=4)
ax.axhline(1.0, color='k', linestyle=':', linewidth=1)
ax.axvline(5, color='red', linestyle='--', linewidth=1,
           label='Top-5 components')
ax.set_xlabel("Number of components")
ax.set_ylabel("Cumulative energy ratio")
ax.set_title("Cumulative singular value energy")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved: {OUTPUT_PATH}")


# ============ 汇总 ============
print("\n===== Summary =====")
print(f"Sequence length n = {seq_len}")
print(f"Head dimension d_h = {head_dim}")
print(f"gamma = {gamma:.4f}")
print(f"MP range: [{lambda_minus:.4f}, {lambda_plus:.4f}]")
print(f"PC1 concentration: {S_normalized[0]:.4f}")
print(f"Top-5 energy: {cumulative[4]:.4f}")
print(f"Random baseline 1/d_h: {1/head_dim:.4f}")