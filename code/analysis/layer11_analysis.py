# layer11_analysis.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ 配置 ============
MODEL_NAME = "gpt2"
TARGET_LAYER = 11          # 语义出口
COMPARE_LAYER = 0          # 入口
OUTPUT_DIR = "attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT = ("The cat sat on the mat. It was a sunny day and the cat wanted "
        "to play outside. The dog ran to the park and saw a bird.")

# 语义测试集（复用）
SEMANTIC_TESTS = [
    ("The cat sat on the ___.", "mat", "sky"),
    ("The fish swims in the ___.", "water", "tree"),
    ("The sun rises in the ___.", "east", "west"),
    ("She drank a cup of ___.", "coffee", "bread"),
    ("The bird flew over the ___.", "house", "water"),
    ("He wrote a letter with a ___.", "pen", "shoe"),
    ("The baby cried for ___.", "milk", "stone"),
    ("They planted flowers in the ___.", "garden", "ocean"),
    ("The chef cooked a delicious ___.", "meal", "book"),
    ("She wore a beautiful ___.", "dress", "car"),
    ("The dog chased the ___.", "cat", "cloud"),
    ("He drove his ___ to work.", "car", "tree"),
    ("The student studied for the ___.", "exam", "beach"),
    ("She painted a picture of a ___.", "sunset", "stone"),
    ("The farmer planted ___ in the field.", "corn", "metal"),
    ("He read a ___ before bed.", "book", "stone"),
    ("The doctor examined the ___.", "patient", "planet"),
    ("She baked a ___ for the party.", "cake", "table"),
    ("The athlete ran around the ___.", "track", "soup"),
    ("He bought a new ___ for his computer.", "keyboard", "cloud"),
    ("The teacher wrote on the ___.", "blackboard", "ocean"),
    ("She drank a glass of ___.", "water", "sand"),
    ("The musician played the ___.", "piano", "cake"),
    ("He climbed the tall ___.", "mountain", "soup"),
    ("The bird built a ___ in the tree.", "nest", "house"),
    ("She wore a warm ___ in winter.", "coat", "fan"),
    ("The ship sailed across the ___.", "ocean", "desert"),
    ("He ate a slice of ___.", "pizza", "paper"),
    ("The photographer took a ___.", "photo", "sandwich"),
    ("She solved the math ___.", "problem", "soup"),
]


# ============ 评估函数 ============
def get_choice_logprob(context, choice, model, tokenizer):
    full_text = context.replace("___", choice)
    if "___" in context:
        prefix = context.split("___")[0]
        if prefix.endswith(" "):
            prefix = prefix[:-1]
    else:
        prefix = context

    prefix_ids = tokenizer(prefix, return_tensors="pt",
                           add_special_tokens=False)["input_ids"]
    prefix_len = prefix_ids.shape[1]

    full_ids = tokenizer(full_text, return_tensors="pt",
                         add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    log_prob = 0.0
    for pos in range(prefix_len, full_ids.shape[1]):
        if pos == 0:
            continue
        log_probs = torch.log_softmax(logits[0, pos - 1], dim=-1)
        token_id = full_ids[0, pos].item()
        log_prob += log_probs[token_id].item()

    return log_prob


def evaluate(tests, model, tokenizer):
    scores = []
    for context, correct, wrong in tests:
        sc = get_choice_logprob(context, correct, model, tokenizer)
        sw = get_choice_logprob(context, wrong, model, tokenizer)
        scores.append(sc - sw)
    return float(np.mean(scores))


# ============ 消融 Hook ============
class HeadAblationHook:
    def __init__(self, layer, head, head_dim):
        self.layer = layer
        self.head = head
        self.head_dim = head_dim
        self.enabled = False

    def __call__(self, module, inp, out):
        if not self.enabled:
            return out
        if isinstance(out, tuple):
            attn_output = out[0].clone()
            rest = out[1:]
        else:
            attn_output = out.clone()
            rest = None

        start = self.head * self.head_dim
        end = start + self.head_dim
        attn_output[:, :, start:end] = 0.0

        if rest is not None:
            return (attn_output,) + rest
        return attn_output


# ============ 加载模型 ============
print("Loading GPT-2...")
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model.eval()

n_layer = model.config.n_layer
n_head = model.config.n_head
n_embd = model.config.n_embd
head_dim = n_embd // n_head

print(f"Layers: {n_layer}, Heads: {n_head}, head_dim: {head_dim}")


# ============ 提取 Q/K/V ============
def extract_qkv(layer):
    qkv_store = {}

    def hook_fn(module, inp, out):
        qkv_store['out'] = out.detach()

    hook = model.transformer.h[layer].attn.c_attn.register_forward_hook(hook_fn)
    inputs = tokenizer(TEXT, return_tensors="pt")
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    qkv = qkv_store['out'][0]
    seq_len = qkv.shape[0]

    q, k, v = qkv.split(n_embd, dim=-1)
    q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
    k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
    v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

    return q, k, v, seq_len


# ============ 逐个头消融 ============
print(f"\n===== Single head ablation: Layer {TARGET_LAYER} =====")
base_semantic = evaluate(SEMANTIC_TESTS, model, tokenizer)
print(f"Baseline semantic: {base_semantic:+.4f}")

hooks = {}
for head in range(n_head):
    hook = HeadAblationHook(TARGET_LAYER, head, head_dim)
    handle = model.transformer.h[TARGET_LAYER].attn.register_forward_hook(hook)
    hooks[head] = (hook, handle)

head_results = []
for head in range(n_head):
    hook, _ = hooks[head]
    hook.enabled = True
    sem = evaluate(SEMANTIC_TESTS, model, tokenizer)
    hook.enabled = False

    head_results.append({
        "head": head,
        "semantic": sem,
        "semantic_delta": sem - base_semantic,
    })
    print(f"  Head {head:2d}: semantic Δ={sem - base_semantic:+.4f}")

for hook, handle in hooks.values():
    handle.remove()

df_head = pd.DataFrame(head_results)
df_head.to_csv(os.path.join(OUTPUT_DIR, "layer11_head_ablation.csv"),
               index=False)


# ============ 计算每个头的几何特征 ============
print(f"\n===== Geometric features: Layer {TARGET_LAYER} =====")
q, k, v, seq_len = extract_qkv(TARGET_LAYER)

features = []
for head in range(n_head):
    q_h, k_h, v_h = q[head], k[head], v[head]

    scale = 1.0 / np.sqrt(head_dim)
    scores = (q_h @ k_h.T) * scale
    mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    p = torch.softmax(scores, dim=-1)

    # 注意力熵
    p_np = p.cpu().numpy()
    entropy = -np.nansum(np.where(p_np > 0, p_np * np.log(p_np + 1e-12), 0),
                         axis=-1)

    # 局部 PC1 集中度
    conc = np.full(seq_len, np.nan)
    for i in range(seq_len):
        k_actual = min(10, i + 1)
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

    features.append({
        "head": head,
        "mean_entropy": float(entropy.mean()),
        "mean_conc": mean_conc,
    })

df_feat = pd.DataFrame(features)
df_feat.to_csv(os.path.join(OUTPUT_DIR, "layer11_head_features.csv"),
               index=False)

print(df_feat.to_string(index=False))


# ============ 合并结果 ============
df_merged = df_head.merge(df_feat, on="head")
print("\n===== Combined =====")
print(df_merged.to_string(index=False))


# ============ 注意力目标分析 ============
print(f"\n===== Attention target analysis: Layer {TARGET_LAYER} =====")
tokens = tokenizer.convert_ids_to_tokens(
    tokenizer(TEXT, return_tensors="pt")["input_ids"][0])

# 选语义 Δ 最大的头
top_semantic_head = int(df_head.loc[df_head['semantic_delta'].idxmin(), 'head'])
print(f"Top semantic head: {top_semantic_head}")

# 该头的注意力目标
q_h, k_h, v_h = q[top_semantic_head], k[top_semantic_head], v[top_semantic_head]
scale = 1.0 / np.sqrt(head_dim)
scores = (q_h @ k_h.T) * scale
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))
p_top = torch.softmax(scores, dim=-1).detach().cpu().numpy()

top1 = p_top.argmax(axis=1)
top1_names = [tokens[i] for i in top1]

print(f"\nTop-1 target distribution (Head {top_semantic_head}):")
cnt = Counter(top1_names)
for tok, c in cnt.most_common(15):
    print(f"  '{tok}': {c} times ({c/seq_len*100:.1f}%)")


# ============ 对比 Layer 0 与 Layer 11 ============
print(f"\n===== Compare Layer {COMPARE_LAYER} vs Layer {TARGET_LAYER} =====")

q0, k0, v0, _ = extract_qkv(COMPARE_LAYER)

compare_features = []
for layer_idx, (ql, kl, vl) in [(COMPARE_LAYER, (q0, k0, v0)),
                                 (TARGET_LAYER, (q, k, v))]:
    for head in range(n_head):
        q_h, k_h, v_h = ql[head], kl[head], vl[head]
        scores = (q_h @ k_h.T) * scale
        scores = scores.masked_fill(mask == 0, float('-inf'))
        p = torch.softmax(scores, dim=-1)
        p_np = p.cpu().numpy()
        entropy = -np.nansum(
            np.where(p_np > 0, p_np * np.log(p_np + 1e-12), 0), axis=-1)

        conc = np.full(seq_len, np.nan)
        for i in range(seq_len):
            k_actual = min(10, i + 1)
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

        compare_features.append({
            "layer": layer_idx,
            "head": head,
            "mean_entropy": float(entropy.mean()),
            "mean_conc": float(np.nanmean(conc)),
        })

df_compare = pd.DataFrame(compare_features)
df_compare.to_csv(os.path.join(OUTPUT_DIR, "layer_compare_features.csv"),
                  index=False)


# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1：Layer 11 每个头的语义 Δ
ax = axes[0, 0]
colors = ['coral' if d < -0.05 else 'steelblue'
          for d in df_head['semantic_delta']]
ax.bar(df_head['head'], df_head['semantic_delta'], color=colors)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel("Head")
ax.set_ylabel("Semantic Δ")
ax.set_title(f"Layer {TARGET_LAYER}: semantic contribution per head")
ax.grid(alpha=0.3, axis='y')

# 图2：PC1 vs 熵
ax = axes[0, 1]
sc = ax.scatter(df_merged['mean_conc'], df_merged['mean_entropy'],
                c=df_merged['semantic_delta'], cmap='RdBu_r',
                s=120, edgecolors='k')
for _, row in df_merged.iterrows():
    ax.annotate(f"H{int(row['head'])}",
                (row['mean_conc'], row['mean_entropy']),
                fontsize=8, ha='center', va='center')
ax.set_xlabel("Mean PC1 concentration")
ax.set_ylabel("Mean attention entropy")
ax.set_title(f"Layer {TARGET_LAYER}: PC1 vs entropy (color = semantic Δ)")
plt.colorbar(sc, ax=ax, label="Semantic Δ")
ax.grid(alpha=0.3)

# 图3：Layer 0 vs Layer 11 对比
ax = axes[1, 0]
for layer_idx, color, label in [(COMPARE_LAYER, 'steelblue', 'Layer 0'),
                                 (TARGET_LAYER, 'coral', 'Layer 11')]:
    sub = df_compare[df_compare['layer'] == layer_idx]
    ax.scatter(sub['mean_conc'], sub['mean_entropy'],
               c=color, s=100, alpha=0.7, label=label)
    for _, row in sub.iterrows():
        ax.annotate(f"H{int(row['head'])}",
                    (row['mean_conc'], row['mean_entropy']),
                    fontsize=7)
ax.set_xlabel("Mean PC1 concentration")
ax.set_ylabel("Mean attention entropy")
ax.set_title("Layer 0 (entry) vs Layer 11 (exit)")
ax.legend()
ax.grid(alpha=0.3)

# 图4：Top semantic head 的注意力矩阵
ax = axes[1, 1]
sns.heatmap(p_top, xticklabels=tokens, yticklabels=tokens,
            cmap='viridis', ax=ax, cbar_kws={'label': 'attention'})
ax.set_xlabel("Key")
ax.set_ylabel("Query")
ax.set_title(f"Attention matrix: Layer {TARGET_LAYER}, Head {top_semantic_head}")
plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "layer11_analysis.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\nSaved: {save_path}")


# ============ 汇总 ============
print("\n===== Summary =====")
print(f"Baseline semantic: {base_semantic:+.4f}")
print(f"\nTop 3 most important heads for semantics:")
df_sorted = df_head.sort_values('semantic_delta')
for _, row in df_sorted.head(3).iterrows():
    h = int(row['head'])
    feat = df_feat[df_feat['head'] == h].iloc[0]
    print(f"  Head {h}: semantic Δ={row['semantic_delta']:+.4f}, "
          f"PC1={feat['mean_conc']:.3f}, entropy={feat['mean_entropy']:.3f}")

print(f"\nLayer {COMPARE_LAYER} vs Layer {TARGET_LAYER}:")
for layer_idx in [COMPARE_LAYER, TARGET_LAYER]:
    sub = df_compare[df_compare['layer'] == layer_idx]
    print(f"  Layer {layer_idx}: mean PC1={sub['mean_conc'].mean():.3f}, "
          f"mean entropy={sub['mean_entropy'].mean():.3f}")

print("\nDone.")