"""Attention implementations and compact map-only global biases."""
from dataclasses import dataclass
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class MapAttentionBias:
    """Compact map metadata used directly by FlexAttention's score modifier.

    ``node_slots`` maps a packed token to its map-table row, or -1 for a
    non-map token.  The quadratic table is consequently MxM rather than NxN,
    where M is the number of map nodes and N is the full observation length.
    """

    node_slots: torch.Tensor  # [batch, tokens], -1 outside the map
    categories: torch.Tensor  # [batch, map_nodes, map_nodes], 0 for padding
    floors: torch.Tensor  # [batch, map_nodes]


class MapAttention(nn.Module):
    def __init__(self, width, heads, *, backend="reference", relations=128):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must divide evenly across heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.map_relation = nn.Embedding(relations, heads, padding_idx=0)
        self.floor_delta = nn.Embedding(33, heads)
        self.backend = backend
        self.actual_backend = backend
        self._flex = None

    def _score_bias(self, metadata, batch, head, query, key, map_weight=None, floor_weight=None):
        map_weight = self.map_relation.weight if map_weight is None else map_weight
        floor_weight = self.floor_delta.weight if floor_weight is None else floor_weight
        last_token = metadata.node_slots.shape[1] - 1
        query_in_bounds = query < metadata.node_slots.shape[1]
        key_in_bounds = key < metadata.node_slots.shape[1]
        query_token = query.clamp(max=last_token)
        key_token = key.clamp(max=last_token)
        query_slot = metadata.node_slots[batch, query_token]
        key_slot = metadata.node_slots[batch, key_token]
        is_map_pair = query_in_bounds & key_in_bounds & (query_slot >= 0) & (key_slot >= 0)
        query_slot = query_slot.clamp_min(0)
        key_slot = key_slot.clamp_min(0)
        category = metadata.categories[batch, query_slot, key_slot]
        delta = (metadata.floors[batch, key_slot] - metadata.floors[batch, query_slot]).clamp(-16, 16) + 16
        # Direct two-dimensional indexing is supported by FlexAttention's
        # score_mod backward; embedding_dense_backward is not.
        learned = map_weight[category, head] + floor_weight[delta, head]
        return torch.where(is_map_pair, learned, learned.new_zeros(()))

    def dense_bias(self, metadata):
        """Materialize a reference bias for CPU and SDPA correctness checks."""
        slots = metadata.node_slots
        safe = slots.clamp_min(0)
        batch = torch.arange(slots.shape[0], device=slots.device)[:, None, None]
        query = safe[:, :, None]
        key = safe[:, None, :]
        category = metadata.categories[batch, query, key]
        query_floor = metadata.floors.gather(1, safe)
        delta = (query_floor[:, None, :] - query_floor[:, :, None]).clamp(-16, 16) + 16
        bias = self.map_relation(category) + self.floor_delta(delta)
        is_map_pair = (slots[:, :, None] >= 0) & (slots[:, None, :] >= 0)
        return (bias * is_map_pair.unsqueeze(-1)).permute(0, 3, 1, 2)

    def forward(self, x, valid, metadata):
        batch_size, tokens, width = x.shape
        q, k, v = (self.qkv(x).view(batch_size, tokens, 3, self.heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        if self.backend in {"flex", "auto"} and x.is_cuda:
            from torch.nn.attention.flex_attention import flex_attention
            if self._flex is None:
                self._flex = torch.compile(flex_attention, dynamic=True)

            def score_mod(score, batch, head, query, key):
                score = score + self._score_bias(metadata, batch, head, query, key)
                key_in_bounds = key < valid.shape[1]
                key_token = key.clamp(max=valid.shape[1] - 1)
                return torch.where(key_in_bounds & valid[batch, key_token], score, -float("inf"))

            out = self._flex(q, k, v, score_mod=score_mod)
            self.actual_backend = "flex"
        else:
            bias = self.dense_bias(metadata)
            mask = bias.masked_fill(~valid[:, None, None, :], -float("inf"))
            if self.backend == "sdpa":
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype), dropout_p=0.0)
                self.actual_backend = "sdpa"
            else:
                if self.backend == "flex" and self.actual_backend != "reference":
                    warnings.warn("FlexAttention requires CUDA; using reference attention")
                self.actual_backend = "reference"
                with torch.autocast(device_type=x.device.type, enabled=False):
                    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
                    scores = scores + mask.float()
                    out = (scores.softmax(-1) @ v.float()).to(v.dtype)
        out = out.transpose(1, 2).reshape(batch_size, tokens, width)
        return self.output(out) * valid.unsqueeze(-1)


class RelationAttention(nn.Module):
    """Small dense reference attention for ordered local effect programs."""

    def __init__(self, width, heads, *, relations=128, backend="reference"):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must divide evenly across heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.relation = nn.Embedding(relations, heads, padding_idx=0)
        self.backend = backend
        self.actual_backend = "reference" if backend == "reference" else "sdpa"

    def forward(self, x, valid, edges):
        batch_size, tokens, width = x.shape
        q, k, v = (self.qkv(x).view(batch_size, tokens, 3, self.heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4).unbind(0))
        flat = x.new_zeros((batch_size * tokens * tokens, self.heads), dtype=self.relation.weight.dtype)
        if edges.numel():
            batch, source, target, role = edges.unbind(-1)
            flat = flat.index_add(0, batch * tokens * tokens + source * tokens + target,
                                  self.relation(role))
        bias = flat.view(batch_size, tokens, tokens, self.heads).permute(0, 3, 1, 2)
        mask = bias.masked_fill(~valid[:, None, None, :], -float("inf"))
        if self.backend == "reference":
            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
                scores = scores + mask.float()
                out = (scores.softmax(-1) @ v.float()).to(v.dtype)
            self.actual_backend = "reference"
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype), dropout_p=0.0)
            self.actual_backend = "sdpa"
        out = out.transpose(1, 2).reshape(batch_size, tokens, width)
        return self.output(out) * valid.unsqueeze(-1)
