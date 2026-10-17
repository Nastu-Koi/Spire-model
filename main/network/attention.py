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


# Elements of bias (pairs x heads x layers) one scatter of `relation_biases` builds at
# most: a quarter of a gigabyte in FP32. A sampling pass (a few thousand pairs) gets
# every layer's bias from one scatter; a training microbatch at its pair budget
# (tens of millions of pairs) goes layer by layer, as `relation_bias` would.
BIAS_ELEMENTS = 1 << 26


def relation_biases(tables, edges, valid, *, dtype=None, elements=BIAS_ELEMENTS):
    """The biases of several layers' tables at once, equal in value to `relation_bias`
    of each: the tables are looked up side by side, so that a pass makes a few
    scatters and fills instead of one of each per layer. Returns a list, one
    [sequences, heads, tokens, tokens] bias per table.

    With `dtype` (the attention's, under autocast) the dense bias is built in that
    dtype from the start, sparing a pass that converts it: a pair with one relation
    gets the same rounded value, one with several has their sum rounded in that dtype
    rather than in FP32, within one unit of its last place."""
    sequences, tokens = valid.shape
    pairs = sequences * tokens * tokens
    heads = tables[0].embedding_dim
    chunk = max(1, min(len(tables), elements // max(pairs * heads, 1)))
    if edges.numel():
        sequence, query, key, role = edges.unbind(-1)
        # Scattered straight into the [layer, sequence, head, query, key] layout the
        # layers read, so that no pass over the dense bias is spent rearranging it:
        # an edge's cell in every (layer, head) plane.
        cells = (sequence * heads * tokens + query) * tokens + key
        planes = torch.arange(heads, device=edges.device) * tokens * tokens
    biases = []
    for start in range(0, len(tables), chunk):
        part = tables[start:start + chunk]
        weight = torch.cat([table.weight for table in part], dim=1) if len(part) > 1 else part[0].weight
        bias = weight.new_zeros((len(part), sequences, heads, tokens, tokens), dtype=dtype or weight.dtype)
        if edges.numel():
            layers = torch.arange(len(part), device=edges.device) * sequences * heads * tokens * tokens
            targets = cells[:, None, None] + layers[None, :, None] + planes[None, None, :]
            values = F.embedding(role, weight, padding_idx=0).to(bias.dtype)
            # In place into the fresh buffer: one pass over the dense bias fewer.
            bias.view(-1).index_add_(0, targets.reshape(-1), values.reshape(-1))
        bias = bias.masked_fill_(~valid[None, :, None, None, :], -float("inf"))
        biases.extend(bias.unbind(0))
    return biases


class Attention(nn.Module):
    def __init__(self, width, heads, *, backend="sdpa"):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must divide evenly across heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        # Normalizing queries and keys per head bounds the attention logits, so a
        # high learning rate cannot drive them to saturation as the model widens.
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.output = nn.Linear(width, width, bias=False)
        self.backend = backend

    def forward(self, x, valid, bias):
        batch_size, tokens, width = x.shape
        q, k, v = (
            self.qkv(x)
            .view(batch_size, tokens, 3, self.heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        # Under autocast the norm may return float32; SDPA wants q, k and v alike.
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
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
