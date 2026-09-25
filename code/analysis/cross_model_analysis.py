# cross_model_analysis.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import pearsonr

# ============ 配置 ============
MODELS = {
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    # 可选：需要 HuggingFace 登录
    # "meta-llama/Llama-3.2-1B": {"n_layer": 16, "n_head": 32, "n_embd": 2048},
    # "Qwen/Qwen2.5-1.5B": {"n_layer": 28, "n_head": 12, "n_embd": 1536},
}

TEXT = ("The cat sat on the mat. It was a sunny day and the cat wanted "
        "to play outside. The dog ran to the park and saw a bird.")
TOP_K_LOCAL = 10
OUTPUT_DIR = "attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_model(model_name, config):
    """对一个模型做完整的局部PCA + 全局PCA分析"""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(TEXT, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    n_layer = config["n_layer"]
    n_head = config["n_head"]
    n_embd = config["n_embd"]
    head_dim = n_embd // n_head

    print(f"Layers: {n_layer}, Heads: {n_head}, Seq: {seq_len}, head_dim: {head_dim}")

    # 收集所有层的 QKV
    qkv_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            qkv_store[layer_idx] = out.detach()
        return hook_fn

    hooks = []
    for l in range(n_layer):
        # 不同的模型有不同的 attn 路径
        if "gpt2" in model_name:
            module = model.transformer.h[l].attn.c_attn
        else:
            module = model.model.layers[l].self_attn.q_proj
        h = module.register_forward_hook(make_hook(l))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # 分析每个头
    records = []
    layer_pc1 = {l: [] for l in range(n_layer)}

    for layer in range(n_layer):
        qkv = qkv_store[layer][0]
        if "gpt2" in model_name:
            q, k, v = qkv.split(n_embd, dim=-1)
            q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
            k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
            v = v.view(seq_len, n_head, head_dim).transpose(0, 1)
        else:
            # 其他模型结构不同，这里简化处理
            q = qkv.view(seq_len, n_head, head_dim).transpose(0, 1)
            k = q
            v = q

        for head in range(n_head):
            q_h, k_h, v_h = q[head], k[head], v[head]

            # 注意力概率
            scale = 1.0 / np.sqrt(head_dim)
            scores = (q_h @ k_h.T) * scale
            mask = torch.tril(torch.ones(seq_len, seq_len))
            scores = scores.masked_fill(mask == 0, float('-inf'))
            p = torch.softmax(scores, dim=-1)

            # 方案A：局部冗余
            V_bar = p @ v_h
            delta = torch.norm(v_h.unsqueeze(0) - V_bar.unsqueeze(1), dim=-1)
            mask_bool = mask.bool()
            p_flat = p[mask_bool].numpy()
            delta_flat = delta[mask_bool].numpy()

            if np.std(p_flat) > 1e-8 and np.std(delta_flat) > 1e-8:
                rA, _ = pearsonr(p_flat, delta_flat)
            else:
                rA = np.nan

            # 局部PCA
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
                except:
                    pass

            mean_conc = np.nanmean(conc)
            layer_pc1[layer].append(mean_conc)

            # 注意力熵
            p_np = p.cpu().numpy()
            entropy = -np.nansum(
                np.where(p_np > 0, p_np * np.log(p_np + 1e-12), 0), axis=-1
            )
            valid = ~np.isnan(conc)
            if valid.sum() > 3 and np.std(conc[valid]) > 1e-8:
                r_ce, _ = pearsonr(conc[valid], entropy[valid])
            else:
                r_ce = np.nan

            records.append({
                "model": model_name,
                "layer": layer,
                "head": head,
                "rA": rA,
                "mean_conc": mean_conc,
                "r_conc_entropy": r_ce,
                "head_dim": head_dim,
            })

    return pd.DataFrame(records)


# ============ 批量分析 ============
all_dfs = []
for model_name, config in MODELS.items():
    try:
        df = analyze_model(model_name, config)
        all_dfs.append(df)
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        continue

df_all = pd.concat(all_dfs, ignore_index=True)
csv_path = os.path.join(OUTPUT_DIR, "cross_model_results.csv")
df_all.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ============ 汇总统计 ============
print("\n===== Cross-Model Summary =====")
for model_name in df_all['model'].unique():
    sub = df_all[df_all['model'] == model_name]
    print(f"\n{model_name}:")
    print(f"  Total heads: {len(sub)}")
    print(f"  Mean rA: {sub['rA'].mean():+.4f}")
    print(f"  rA < 0 fraction: {(sub['rA'] < 0).mean():.2%}")
    print(f"  Mean PC1: {sub['mean_conc'].mean():.4f}")
    print(f"  PC1 > 0.3 fraction: {(sub['mean_conc'] > 0.3).mean():.2%}")
    print(f"  Random baseline (1/d): {1/sub['head_dim'].iloc[0]:.4f}")


# ============ 可视化 ============
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, model_name in enumerate(df_all['model'].unique()):
    sub = df_all[df_all['model'] == model_name]
    n_layer = sub['layer'].max() + 1
    n_head = sub['head'].max() + 1
    head_dim = sub['head_dim'].iloc[0]

    # 图1：PC1 热力图
    ax = axes[0, idx]
    mat = sub['mean_conc'].values.reshape(n_layer, n_head)
    im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"{model_name}: PC1 concentration")
    plt.colorbar(im, ax=ax)

    # 图2：层趋势
    ax = axes[1, idx]
    layer_mean = sub.groupby('layer')['mean_conc'].mean()
    ax.plot(layer_mean.index, layer_mean.values, 'o-', label='mean PC1')
    ax.axhline(1/head_dim, color='r', ls='--',
               label=f'random baseline = 1/{head_dim}')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean PC1")
    ax.set_title(f"{model_name}: Layer trend")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "cross_model_heatmaps.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved: {fig_path}")


# ============ 跨模型对比 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左：层趋势对比
ax = axes[0]
for model_name in df_all['model'].unique():
    sub = df_all[df_all['model'] == model_name]
    layer_mean = sub.groupby('layer')['mean_conc'].mean()
    ax.plot(layer_mean.index, layer_mean.values, 'o-', label=model_name)
ax.set_xlabel("Layer")
ax.set_ylabel("Mean PC1 concentration")
ax.set_title("Cross-model layer trend")
ax.legend()
ax.grid(alpha=0.3)

# 右：rA 分布对比
ax = axes[1]
for model_name in df_all['model'].unique():
    sub = df_all[df_all['model'] == model_name]
    ax.hist(sub['rA'].dropna(), bins=20, alpha=0.5, label=model_name)
ax.axvline(0, color='k', ls='--')
ax.set_xlabel("rA (correlation p vs Delta)")
ax.set_ylabel("Count")
ax.set_title("Cross-model rA distribution")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "cross_model_comparison.png")
plt.savefig(fig_path, dpi=150)
plt.show()
print(f"Saved: {fig_path}")

print("\nDone.")