# code/utils/hooks.py
"""
QKV 提取 Hook 工具。
提供统一接口，用于从 GPT-2 (及兼容模型) 的注意力层提取 Q/K/V。
"""

import torch
import numpy as np


class QKVHook:
    """
    提取 c_attn 输出的 Hook。
    GPT-2 的 attn.c_attn 输出形状为 [batch, seq, 3*n_embd]，
    前 n_embd 是 Q，中间是 K，后是 V。
    """

    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
        self.output = None

    def __call__(self, module, inp, out):
        self.output = out.detach()

    def get_qkv(self, n_embd, n_head, head_dim, batch_idx=0):
        """
        返回 Q, K, V，形状 [n_head, seq, head_dim]。
        """
        if self.output is None:
            raise RuntimeError("Hook 未捕获到输出，请先运行前向传播。")

        qkv = self.output[batch_idx]  # [seq, 3*n_embd]
        seq_len = qkv.shape[0]

        q, k, v = qkv.split(n_embd, dim=-1)
        q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
        k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
        v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

        return q, k, v


class AttentionOutputHook:
    """
    提取 attn 模块输出（attention 加权后的结果）。
    用于消融实验。
    """

    def __init__(self, layer_idx, head_indices=None):
        self.layer_idx = layer_idx
        self.head_indices = head_indices  # None 表示整层
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

        # 处理消融逻辑
        if self.head_indices is None:
            attn_output[:, :, :] = 0.0
        else:
            for h in self.head_indices:
                start = h * (attn_output.shape[-1] // max(1, self._n_head))
                end = start + (attn_output.shape[-1] // max(1, self._n_head))
                attn_output[:, :, start:end] = 0.0

        if rest is not None:
            return (attn_output,) + rest
        return attn_output


class PatchHook:
    """
    激活修补 Hook：把指定头的输出替换为该头在所有 token 上的均值。
    """

    def __init__(self, layer_idx, head_idx, head_dim):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
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

        start = self.head_idx * self.head_dim
        end = start + self.head_dim
        mean_val = attn_output[:, :, start:end].mean(dim=1, keepdim=True)
        attn_output[:, :, start:end] = mean_val

        if rest is not None:
            return (attn_output,) + rest
        return attn_output


def register_qkv_hooks(model, n_layer):
    """
    给所有层注册 QKV Hook，返回 hooks 列表与句柄列表。
    """
    hooks = []
    handles = []
    for l in range(n_layer):
        h = QKVHook(l)
        handle = model.transformer.h[l].attn.c_attn.register_forward_hook(h)
        hooks.append(h)
        handles.append(handle)
    return hooks, handles


def remove_hooks(handles):
    """移除所有 hook 句柄。"""
    for handle in handles:
        handle.remove()