"""Multi-head attention with an additive bias built from sparse typed relations."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def relation_bias(table, edges, valid):
    """Scatter relation embeddings into a dense attention bias.

    ``edges`` holds rows of (sequence, query, key, relation bucket); ``table``
    maps a bucket to one scalar per head. Several relations between the same
    pair add up, unrelated pairs receive no bias, and padded keys are excluded.
    The result is [sequences, heads, tokens, tokens].
    """
    sequences, tokens = valid.shape
    flat = table.weight.new_zeros((sequences * tokens * tokens, table.embedding_dim))
    if edges.numel():
        sequence, query, key, role = edges.unbind(-1)
        flat = flat.index_add(
            0, (sequence * tokens + query) * tokens + key, table(role)
        )
    bias = flat.view(sequences, tokens, tokens, -1).permute(0, 3, 1, 2)
    return bias.masked_fill(~valid[:, None, None, :], -float("inf"))


class Attention(nn.Module):
    def __init__(self, width, heads, *, backend="sdpa"):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must divide evenly across heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.backend = backend

    def forward(self, x, valid, bias):
        batch_size, tokens, width = x.shape
        q, k, v = (
            self.qkv(x)
            .view(batch_size, tokens, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        if self.backend == "reference":
            exact = torch.promote_types(q.dtype, torch.float32)
            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = (
                    q.to(exact) @ k.to(exact).transpose(-1, -2) / math.sqrt(self.head_dim)
                )
                out = ((scores + bias.to(exact)).softmax(-1) @ v.to(exact)).to(v.dtype)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias.to(q.dtype), dropout_p=0.0
            )
        out = out.transpose(1, 2).reshape(batch_size, tokens, width)
        return self.output(out) * valid.unsqueeze(-1)
