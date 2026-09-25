# cumulative_ablation.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ============ 配置 ============
MODEL_NAME = "gpt2"
OUTPUT_DIR = "attention_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中间层（排除 L0 入口和 L11 出口）
MIDDLE_LAYERS = list(range(1, 11))   # [1, 2, ..., 10]

# ============ 测试集（复用实验 1 的 60+60） ============
SYNTAX_TESTS = [
    ("The cat ___ on the mat.", "is", "are"),
    ("The dog ___ in the garden.", "is", "are"),
    ("The book ___ on the table.", "is", "are"),
    ("The boy ___ to school.", "goes", "go"),
    ("The girl ___ a song.", "sings", "sing"),
    ("The bird ___ in the tree.", "sits", "sit"),
    ("The man ___ a doctor.", "is", "are"),
    ("The woman ___ a teacher.", "is", "are"),
    ("The child ___ happy.", "is", "are"),
    ("The flower ___ beautiful.", "is", "are"),
    ("The cats ___ on the mat.", "are", "is"),
    ("The dogs ___ in the garden.", "are", "is"),
    ("The books ___ on the table.", "are", "is"),
    ("The boys ___ to school.", "go", "goes"),
    ("The girls ___ a song.", "sing", "sings"),
    ("The birds ___ in the tree.", "sit", "sits"),
    ("The men ___ doctors.", "are", "is"),
    ("The women ___ teachers.", "are", "is"),
    ("The children ___ happy.", "are", "is"),
    ("The flowers ___ beautiful.", "are", "is"),
    ("Yesterday, she ___ to the store.", "went", "goes"),
    ("Tomorrow, he ___ to school.", "will go", "went"),
    ("Last week, they ___ the movie.", "watched", "watch"),
    ("Now, I ___ a book.", "am reading", "read"),
    ("Last night, it ___ heavily.", "rained", "rains"),
    ("Next year, we ___ to Japan.", "will go", "went"),
    ("Yesterday, the cat ___ the fish.", "ate", "eats"),
    ("Tomorrow, she ___ the cake.", "will bake", "baked"),
    ("Last month, he ___ a new car.", "bought", "buys"),
    ("Now, they ___ dinner.", "are eating", "ate"),
    ("John said ___ was tired.", "he", "they"),
    ("Mary said ___ was happy.", "she", "he"),
    ("The dog wagged ___ tail.", "its", "their"),
    ("The children lost ___ toys.", "their", "its"),
    ("I saw the man ___ was tall.", "who", "which"),
    ("I saw the book ___ was red.", "which", "who"),
    ("She gave ___ a gift.", "him", "he"),
    ("They invited ___ to the party.", "us", "we"),
    ("The cat licked ___ paws.", "its", "their"),
    ("The students raised ___ hands.", "their", "its"),
    ("___ apple a day keeps the doctor away.", "An", "A"),
    ("I saw ___ elephant at the zoo.", "an", "a"),
    ("She is ___ honest person.", "an", "a"),
    ("He is ___ university student.", "a", "an"),
    ("___ sun rises in the east.", "The", "A"),
    ("___ moon is bright tonight.", "The", "A"),
    ("I need ___ umbrella.", "an", "a"),
    ("She bought ___ new car.", "a", "an"),
    ("___ earth is round.", "The", "A"),
    ("He ate ___ orange.", "an", "a"),
    ("The book is ___ the table.", "on", "in"),
    ("The cat is ___ the box.", "in", "on"),
    ("She walked ___ the park.", "through", "on"),
    ("He arrived ___ 5 o'clock.", "at", "in"),
    ("They live ___ London.", "in", "at"),
    ("The picture is ___ the wall.", "on", "in"),
    ("She is good ___ math.", "at", "in"),
    ("He is interested ___ science.", "in", "at"),
    ("The plane flew ___ the clouds.", "above", "on"),
    ("She sat ___ the chair.", "on", "in"),
]

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
    ("The gardener watered the ___.", "flowers", "stones"),
    ("He fixed the broken ___.", "car", "cloud"),
    ("The librarian organized the ___.", "books", "snacks"),
    ("She caught the ___ in the net.", "fish", "cloud"),
    ("The painter used a ___.", "brush", "spoon"),
    ("He drove through the ___.", "tunnel", "sky"),
    ("The bee collected ___ from the flower.", "nectar", "sand"),
    ("She wrote a ___ to her friend.", "letter", "stone"),
    ("The firefighter climbed the ___.", "ladder", "pillow"),
    ("He played ___ with his friends.", "soccer", "soup"),
    ("The scientist conducted an ___.", "experiment", "umbrella"),
    ("She cut the ___ with a knife.", "bread", "cloud"),
    ("The pilot flew the ___.", "airplane", "sofa"),
    ("He washed his ___ in the sink.", "hands", "hats"),
    ("The elephant has a long ___.", "trunk", "tail"),
    ("She wore ___ on her feet.", "shoes", "gloves"),
    ("The moon shines at ___.", "night", "noon"),
    ("He ate breakfast in the ___.", "morning", "evening"),
    ("The snow falls in ___.", "winter", "summer"),
    ("She put the keys in her ___.", "pocket", "hair"),
    ("The train arrived at the ___.", "station", "kitchen"),
    ("He swam in the ___.", "pool", "tree"),
    ("The clock on the wall shows the ___.", "time", "food"),
    ("She opened the ___ to enter the room.", "door", "book"),
    ("The cat licked its ___.", "paws", "wings"),
    ("He drank soup with a ___.", "spoon", "fork"),
    ("The tree has many ___.", "leaves", "wheels"),
    ("She lit a ___ in the dark.", "candle", "stone"),
    ("The phone rang, so he ___ it.", "answered", "cooked"),
    ("The teacher gave ___ to the students.", "homework", "clouds"),
    ("She cooked rice in a ___.", "pot", "cup"),
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
print(f"Middle layers to ablate: {MIDDLE_LAYERS}")


# ============ 消融 Hook ============
class LayerAblationHook:
    def __init__(self, layer):
        self.layer = layer
        self.enabled = False

    def __call__(self, module, inp, out):
        if not self.enabled:
            return out
        if isinstance(out, tuple):
            attn_output = out[0].clone()
            rest = out[1:]
            attn_output[:, :, :] = 0.0
            return (attn_output,) + rest
        else:
            out = out.clone()
            out[:, :, :] = 0.0
            return out


# 注册所有中间层的 hook
hooks = {}
for layer in MIDDLE_LAYERS:
    hook = LayerAblationHook(layer)
    handle = model.transformer.h[layer].attn.register_forward_hook(hook)
    hooks[layer] = (hook, handle)

print(f"Registered {len(hooks)} layer hooks")


# ============ 基线 ============
print("\n===== Baseline =====")
base_syntax = evaluate(SYNTAX_TESTS, model, tokenizer)
base_semantic = evaluate(SEMANTIC_TESTS, model, tokenizer)
print(f"Syntax: {base_syntax:+.4f}")
print(f"Semantic: {base_semantic:+.4f}")


# ============ 累积消融 ============
print("\n===== Cumulative Ablation =====")

# 消融顺序：从中间向外，或按 PC1 排序
# 这里按层号从小到大
ablation_order = MIDDLE_LAYERS

results = []
for k in range(0, len(ablation_order) + 1):
    # 启用前 k 个 hook
    for i, layer in enumerate(ablation_order):
        hook, _ = hooks[layer]
        hook.enabled = (i < k)

    syn = evaluate(SYNTAX_TESTS, model, tokenizer)
    sem = evaluate(SEMANTIC_TESTS, model, tokenizer)

    results.append({
        "k": k,
        "ablated_layers": ablation_order[:k],
        "syntax": syn,
        "semantic": sem,
        "syntax_delta": syn - base_syntax,
        "semantic_delta": sem - base_semantic,
    })

    print(f"  k={k:2d}: syntax Δ={syn - base_syntax:+.4f}, "
          f"semantic Δ={sem - base_semantic:+.4f}")

# 关闭所有 hook
for hook, _ in hooks.values():
    hook.enabled = False


df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "cumulative_ablation.csv")
df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")


# ============ 计算崩溃点 ============
# 定义崩溃：|Δ| 超过基线的 50%
syntax_threshold = 0.5 * abs(base_syntax)
semantic_threshold = 0.5 * abs(base_semantic)

syntax_collapse_k = None
semantic_collapse_k = None

for _, row in df.iterrows():
    if syntax_collapse_k is None and abs(row['syntax_delta']) > syntax_threshold:
        syntax_collapse_k = int(row['k'])
    if semantic_collapse_k is None and abs(row['semantic_delta']) > semantic_threshold:
        semantic_collapse_k = int(row['k'])

print(f"\n===== Collapse Points =====")
print(f"Syntax collapse at k = {syntax_collapse_k} "
      f"(ablating {syntax_collapse_k} layers)")
print(f"Semantic collapse at k = {semantic_collapse_k} "
      f"(ablating {semantic_collapse_k} layers)")


# ============ 可视化 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 图1：累积消融曲线
ax = axes[0]
ax.plot(df['k'], df['syntax_delta'], 'o-', label='Syntax Δ', color='steelblue')
ax.plot(df['k'], df['semantic_delta'], 's-', label='Semantic Δ', color='coral')
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.axhline(-syntax_threshold, color='steelblue', ls=':', lw=1,
           alpha=0.5, label='Syntax collapse threshold')
ax.axhline(-semantic_threshold, color='coral', ls=':', lw=1,
           alpha=0.5, label='Semantic collapse threshold')

if syntax_collapse_k is not None:
    ax.axvline(syntax_collapse_k, color='steelblue', ls='--', alpha=0.3)
if semantic_collapse_k is not None:
    ax.axvline(semantic_collapse_k, color='coral', ls='--', alpha=0.3)

ax.set_xlabel("Number of ablated middle layers (k)")
ax.set_ylabel("Log-prob delta")
ax.set_title("Cumulative ablation: compensation collapse")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# 图2：相对保留率
ax = axes[1]
syntax_retention = df['syntax'] / base_syntax
semantic_retention = df['semantic'] / base_semantic

ax.plot(df['k'], syntax_retention, 'o-', label='Syntax retention',
        color='steelblue')
ax.plot(df['k'], semantic_retention, 's-', label='Semantic retention',
        color='coral')
ax.axhline(1.0, color='k', ls='--', lw=0.8, label='Baseline')
ax.axhline(0.5, color='gray', ls=':', lw=1, label='50% threshold')
ax.set_xlabel("Number of ablated middle layers (k)")
ax.set_ylabel("Retention ratio (score / baseline)")
ax.set_title("Information retention under cumulative ablation")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
save_path = os.path.join(OUTPUT_DIR, "cumulative_ablation_results.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\nSaved: {save_path}")


# ============ 移除 Hook ============
for hook, handle in hooks.values():
    handle.remove()


# ============ 汇总 ============
print("\n===== Summary =====")
print(f"Baseline: syntax={base_syntax:+.4f}, semantic={base_semantic:+.4f}")
print(f"\nAblation order: {ablation_order}")
print(f"\n{'k':>3} {'Layers ablated':<30} {'Syntax Δ':>12} {'Semantic Δ':>12}")
print("-" * 62)
for _, row in df.iterrows():
    layers_str = str(row['ablated_layers'])
    if len(layers_str) > 28:
        layers_str = layers_str[:25] + "..."
    print(f"{int(row['k']):>3} {layers_str:<30} "
          f"{row['syntax_delta']:>+12.4f} {row['semantic_delta']:>+12.4f}")

print(f"\nSyntax collapse at k = {syntax_collapse_k}")
print(f"Semantic collapse at k = {semantic_collapse_k}")

print("\nDone.")