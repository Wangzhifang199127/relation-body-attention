# code/utils/metrics.py
"""
度量指标工具。
封装方案 A/B/C、局部 PCA、熵、外积范数等计算。
"""

import numpy as np
import torch
from scipy.stats import pearsonr


# ============ 关系体基础 ============

def relation_coords(a, b):
    """
    计算两个向量的关系体坐标 (c, s)。
    c = cos(theta), s = |sin(theta)|。
    支持 numpy 数组或 torch 张量。
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0, 0.0

    c = np.dot(a, b) / (norm_a * norm_b)
    c = np.clip(c, -1.0, 1.0)
    s = np.sqrt(1.0 - c ** 2)
    return c, s


def wedge_norm_squared(a, b):
    """
    高维外积范数平方：
    ||a ^ b||^2 = ||a||^2 ||b||^2 - <a, b>^2
    """
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu().numpy()

    norm_a2 = np.dot(a, a)
    norm_b2 = np.dot(b, b)
    inner2 = np.dot(a, b) ** 2
    return max(0.0, norm_a2 * norm_b2 - inner2)


# ============ 注意力概率与熵 ============

def compute_attention(q_h, k_h, head_dim, causal=True):
    """
    计算注意力概率矩阵。
    q_h, k_h: torch.Tensor [seq, head_dim]
    返回: p [seq, seq] (torch.Tensor)
    """
    seq_len = q_h.shape[0]
    scale = 1.0 / np.sqrt(head_dim)
    scores = (q_h @ k_h.T) * scale

    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

    p = torch.softmax(scores, dim=-1)
    return p


def attention_entropy(p):
    """
    计算每行的注意力熵。
    p: torch.Tensor [seq, seq]
    返回: entropy [seq] (numpy)
    """
    p_np = p.detach().cpu().numpy()
    entropy = -np.nansum(
        np.where(p_np > 0, p_np * np.log(p_np + 1e-12), 0), axis=-1
    )
    return entropy


# ============ 方案 A：局部冗余度 ============

def compute_scheme_A(p, v_h):
    """
    计算方案 A 相关系数 rA = corr(p_ij, Delta_ij)。
    p: [seq, seq] 注意力概率
    v_h: [seq, head_dim] Value
    返回: rA (float)
    """
    V_bar = p @ v_h  # [seq, head_dim]
    delta = torch.norm(v_h.unsqueeze(0) - V_bar.unsqueeze(1), dim=-1)  # [seq, seq]

    mask = torch.tril(torch.ones_like(p, dtype=torch.bool))
    p_flat = p[mask].detach().cpu().numpy()
    delta_flat = delta[mask].detach().cpu().numpy()

    if np.std(p_flat) < 1e-8 or np.std(delta_flat) < 1e-8:
        return np.nan

    rA, _ = pearsonr(p_flat, delta_flat)
    return float(rA)


# ============ 方案 B/C：全局主成分 ============

def compute_scheme_BC(p, v_h, top_k=5):
    """
    计算方案 B（能量占比）与方案 C（残差范数）。
    返回: (rB, rC)
    """
    V_np = v_h.detach().cpu().numpy()
    seq_len = V_np.shape[0]

    U, S, Vt = np.linalg.svd(V_np, full_matrices=False)
    k_use = min(top_k, Vt.shape[0])
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
    p_np = p.detach().cpu().numpy()
    avg_p = np.array([p_np[j:, j].mean() for j in range(seq_len)])

    if np.std(avg_p) < 1e-8:
        return np.nan, np.nan

    rB, _ = pearsonr(avg_p, energy_ratio)
    rC, _ = pearsonr(avg_p, residual_norm)
    return float(rB), float(rC)


# ============ 局部 PCA ============

def compute_local_pca(p, v_h, top_k=10):
    """
    计算每个 query 的局部 PC1 集中度。
    返回: concentration [seq] (numpy)
    """
    seq_len = p.shape[0]
    v_np = v_h.detach().cpu().numpy()
    conc = np.full(seq_len, np.nan)

    for i in range(seq_len):
        k_actual = min(top_k, i + 1)
        if k_actual < 3:
            continue

        p_i = p[i, :i + 1]
        top_idx = torch.topk(p_i, k_actual).indices
        V_top = v_np[top_idx.cpu().numpy()]
        V_c = V_top - V_top.mean(axis=0, keepdims=True)

        try:
            _, S, _ = np.linalg.svd(V_c, full_matrices=False)
            e = S ** 2
            if e.sum() > 1e-12:
                conc[i] = e[0] / e.sum()
        except Exception:
            pass

    return conc


# ============ 综合度量 ============

def compute_all_metrics(q_h, k_h, v_h, head_dim, top_k_local=10, top_k_global=5):
    """
    对单个头计算所有度量。
    返回 dict: {rA, rB, rC, mean_conc, entropy_mean, r_conc_entropy}
    """
    p = compute_attention(q_h, k_h, head_dim)

    rA = compute_scheme_A(p, v_h)
    rB, rC = compute_scheme_BC(p, v_h, top_k=top_k_global)
    conc = compute_local_pca(p, v_h, top_k=top_k_local)

    entropy = attention_entropy(p)
    mean_conc = float(np.nanmean(conc))
    entropy_mean = float(np.nanmean(entropy))

    valid = ~np.isnan(conc)
    if valid.sum() > 3 and np.std(conc[valid]) > 1e-8:
        r_ce, _ = pearsonr(conc[valid], entropy[valid])
    else:
        r_ce = np.nan

    return {
        "rA": rA,
        "rB": rB,
        "rC": rC,
        "mean_conc": mean_conc,
        "entropy_mean": entropy_mean,
        "r_conc_entropy": float(r_ce) if not np.isnan(r_ce) else np.nan,
    }