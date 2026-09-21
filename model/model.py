"""Shared typed encoder → [Full, Linear]×n+Full → session GRU pointer."""
from dataclasses import dataclass
import math
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .representation import Observation, Vocabulary, bucket, field_tensors, fields_of


def linear_attention(q, k, v, valid, epsilon=1e-6):
    """Non-causal ELU+1 attention. FP32 aggregation, independently per batch."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        query = F.elu(q.float()) + 1
        key = (F.elu(k.float()) + 1) * valid[:, None, :, None]
        values = v.float() * valid[:, None, :, None]
        summary = key.transpose(-1, -2) @ values
        normalizer = key.sum(dim=-2)
        denominator = (query * normalizer.unsqueeze(-2)).sum(-1, keepdim=True).clamp_min(epsilon)
        out = (query @ summary) / denominator
        out = out * valid[:, None, :, None]
    return out.to(v.dtype)


class Attention(nn.Module):
    def __init__(self, width, heads, *, linear=False, backend="reference"):
        super().__init__()
        self.heads, self.head_dim, self.linear = heads, width // heads, linear
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.backend = backend
        self.actual_backend = "linear" if linear else backend
        self._flex = None

    def forward(self, x, valid, bias=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        if self.linear:
            out = linear_attention(q, k, v, valid)
        elif self.backend in {"flex", "auto"} and x.is_cuda:
            from torch.nn.attention.flex_attention import flex_attention
            if self._flex is None:
                self._flex = torch.compile(flex_attention)

            def score_mod(score, batch, head, query, key):
                return torch.where(valid[batch, key], score + bias[batch, head, query, key], -float("inf"))

            out = self._flex(q, k, v, score_mod=score_mod)
            self.actual_backend = "flex"
        elif self.backend == "sdpa":
            mask = bias.masked_fill(~valid[:, None, None, :], -float("inf"))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.to(q.dtype), dropout_p=0.0)
            self.actual_backend = "sdpa"
        else:
            if self.backend == "flex" and self.actual_backend != "reference":
                warnings.warn("FlexAttention requires CUDA; retaining relation bias with reference attention")
            self.actual_backend = "reference"
            with torch.autocast(device_type=x.device.type, enabled=False):
                scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
                scores = scores + bias.float()
                scores = scores.masked_fill(~valid[:, None, None, :], -float("inf"))
                out = (scores.softmax(-1) @ v.float()).to(v.dtype)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.output(out) * valid.unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, width, heads, ffn_size, *, linear=False, backend="reference", relations=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = Attention(width, heads, linear=linear, backend=backend)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, ffn_size), nn.GELU(), nn.Linear(ffn_size, width))
        if not linear:
            self.relation = nn.Embedding(relations, heads)
            self.floor = nn.Embedding(33, heads)

    def forward(self, x, valid, edges, floors):
        bias = None
        if not self.attention.linear:
            b, n, _ = x.shape
            # Sparse additive relationships retain overlapping roles.
            flat = x.new_zeros((b * n * n, self.attention.heads), dtype=self.relation.weight.dtype)
            if edges.numel():
                batch, source, target, role = edges.unbind(-1)
                flat = flat.index_add(0, batch * n * n + source * n + target, self.relation(role))
            if floors.numel():
                batch, source, target, delta = floors.unbind(-1)
                flat = flat.index_add(0, batch * n * n + source * n + target, self.floor(delta))
            bias = flat.view(b, n, n, self.attention.heads).permute(0, 3, 1, 2)
        x = x + self.attention(self.norm1(x), valid, bias)
        return (x + self.ffn(self.norm2(x))) * valid.unsqueeze(-1)


class SharedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        local = config.local_size
        self.symbol = nn.Embedding(config.vocabulary_size, local, padding_idx=0)
        self.field = nn.Embedding(config.field_buckets, local, padding_idx=0)
        self.numeric = nn.Sequential(nn.Linear(5, local), nn.GELU(), nn.Linear(local, local))
        self.field_norm = nn.LayerNorm(local)
        self.local_blocks = nn.ModuleList([Block(local, config.local_heads, local * 4,
                                                relations=config.relation_buckets)
                                           for _ in range(config.local_layers)])
        self.reference = nn.Linear(local, local)
        self.program_binding = nn.Linear(local, local)
        self.local_norm = nn.LayerNorm(local)
        self.projection = nn.Linear(local, config.hidden_size)

    def fields(self, rows, vocabulary, length=None):
        ids, kinds, nums, mask = field_tensors(rows, vocabulary, self.config.field_buckets,
                                               self.symbol.weight.device, length)
        x = self.symbol(ids) + self.field(kinds) + self.numeric(nums)
        # Sum retains multiplicity; sqrt normalization controls magnitude only.
        return self.field_norm((x * mask.unsqueeze(-1)).sum(1) / mask.sum(1).clamp_min(1).sqrt().unsqueeze(-1))

    def forward(self, obs, vocabulary):
        base = self.fields(obs.tokens, vocabulary)
        fused = base.clone()
        for i, j, role in obs.edges:
            if role in {"source", "target", "option", "owner"}:
                role_vector = self.field.weight[bucket("reference." + role, self.config.field_buckets)]
                fused[i] = fused[i] + self.reference(base[j] + role_vector)
        n = len(obs.effects)
        program_length = max(len(e.nodes) for e in obs.effects)
        rows = [row for effect in obs.effects for row in effect.nodes + [[]] * (program_length - len(effect.nodes))]
        programs = self.fields(rows, vocabulary).view(n, program_length, -1)
        valid = torch.tensor([[j < len(e.nodes) for j in range(program_length)] for e in obs.effects],
                             device=base.device, dtype=torch.bool)
        program_edges = []
        for token, effect in enumerate(obs.effects):
            for node, ref, role in effect.bindings:
                if ref not in obs.refs:
                    raise ValueError("Unresolved effect-program entity reference")
                programs[token, node] = programs[token, node] + self.program_binding(
                    base[obs.refs[ref]] + self.field.weight[bucket("binding." + role, self.config.field_buckets)])
            program_edges.extend((token, i, j, bucket(role, self.config.relation_buckets)) for i, j, role in effect.edges)
        edges = torch.tensor(program_edges, dtype=torch.long, device=base.device).reshape(-1, 4)
        floors = edges.new_empty((0, 4))
        for block in self.local_blocks:
            programs = block(programs, valid, edges, floors)
        pooled = (programs * valid.unsqueeze(-1)).sum(1) / valid.sum(1).sqrt().unsqueeze(-1)
        local = self.local_norm(fused + programs[:, 0] + pooled)
        return self.projection(local), local


@dataclass
class Encoded:
    hidden: torch.Tensor
    local: torch.Tensor
    actions: torch.Tensor
    keys: torch.Tensor
    observation: Observation


@dataclass
class DecisionOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    value: torch.Tensor | None

    def distribution(self):
        return torch.distributions.Categorical(logits=self.logits.float())

    def topk(self, k):
        if k < 1:
            raise ValueError("k must be positive")
        count = int(torch.isfinite(self.logits).sum())
        values, indices = torch.topk(self.logits, min(k, count))
        return indices, values


class PolicyValue(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d, local = config.hidden_size, config.local_size
        self.encoder = SharedEncoder(config)
        self.blocks = nn.ModuleList([Block(d, config.num_heads, config.ffn_size, linear=bool(i % 2),
                                          backend=config.backend, relations=config.relation_buckets)
                                     for i in range(config.layers)])
        self.final_norm = nn.LayerNorm(d)
        self.prefix_order = nn.GRUCell(local, local)
        self.prefix_projection = nn.Linear(local, d)
        self.previous_projection = nn.Linear(d, d)
        self.input_norm = nn.LayerNorm(d)
        self.bos = nn.Parameter(torch.zeros(d))
        self.gru = nn.GRUCell(d, d)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, 1)
        self.encoder_calls = 0
        self.decoder_calls = 0
        if config.compile_blocks:
            # Module.compile keeps checkpoint state_dict names unchanged.
            for block in self.blocks:
                block.compile(dynamic=False)

    @property
    def device(self):
        return self.bos.device

    def encode(self, observations, vocabulary, *, pad_to=None):
        self.encoder_calls += 1
        encoded = [self.encoder(o, vocabulary) for o in observations]
        n = max(len(o.tokens) for o in observations)
        n = max(n, pad_to or 0)
        hidden = torch.stack([F.pad(x, (0, 0, 0, n - x.shape[0])) for x, _ in encoded])
        valid = torch.tensor([[i < len(o.tokens) for i in range(n)] for o in observations], device=hidden.device)
        edge_rows, floor_rows = [], []
        for batch, obs in enumerate(observations):
            edge_rows.extend((batch, i, j, bucket(role, self.config.relation_buckets)) for i, j, role in obs.edges)
            floor_rows.extend((batch, i, j, max(-16, min(16, fj - fi)) + 16)
                              for i, fi in obs.map_floors.items() for j, fj in obs.map_floors.items())
        edges = torch.tensor(edge_rows, dtype=torch.long, device=hidden.device).reshape(-1, 4)
        floors = torch.tensor(floor_rows, dtype=torch.long, device=hidden.device).reshape(-1, 4)
        for block in self.blocks:
            if self.config.checkpoint_layers and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, valid, edges, floors, use_reentrant=False)
            else:
                hidden = block(hidden, valid, edges, floors)
        hidden = self.final_norm(hidden)
        result = []
        for b, obs in enumerate(observations):
            h = hidden[b, :len(obs.tokens)]
            actions = h[obs.action_indices]
            result.append(Encoded(h, encoded[b][1], actions, self.key(actions), obs))
        return result

    def prefix(self, encoded, context, vocabulary):
        local = self.encoder.fields([fields_of(context)], vocabulary)[0]
        selected = context.get("selected_refs", [])
        order_known = (context.get("known_masks") or {}).get("order_matters", True)
        if context.get("order_matters") and order_known:
            state = torch.zeros_like(local)
            for ref in selected:
                state = self.prefix_order(encoded.local[encoded.observation.refs[ref]], state)
            local = local + state
        else:
            for ref in selected:
                local = local + encoded.local[encoded.observation.refs[ref]]
        return self.prefix_projection(local)

    def decode(self, encoded, legal_mask, context, vocabulary, *, hidden=None, previous=None, value=True):
        if legal_mask.dtype != torch.bool or legal_mask.shape != (encoded.actions.shape[0],):
            raise ValueError("Invalid engine mask")
        if int(legal_mask.sum()) < 2:
            raise ValueError("Forced actions must bypass every model head")
        self.decoder_calls += 1
        prefix = self.prefix(encoded, context, vocabulary)
        start = encoded.hidden[0] + encoded.actions[legal_mask].mean(0) + prefix
        previous_embedding = self.bos if previous is None else previous if isinstance(previous, torch.Tensor) else encoded.actions[previous]
        current = self.input_norm(start + self.previous_projection(previous_embedding))
        if hidden is None:
            hidden = torch.zeros_like(current)
        hidden = self.gru(current, hidden)
        query = self.query(hidden)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = query.float() @ encoded.keys.float().T / math.sqrt(self.config.hidden_size)
            logits = logits.masked_fill(~legal_mask, -float("inf"))
        return DecisionOutput(logits, hidden, self.value(start).float().squeeze(-1) if value else None)

    def parameter_report(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "backbone": sum(p.numel() for b in self.blocks for p in b.parameters()) + sum(p.numel() for p in self.final_norm.parameters()),
                "gru": sum(p.numel() for p in self.gru.parameters()),
                "full_layers": sum(not b.attention.linear for b in self.blocks),
                "linear_layers": sum(b.attention.linear for b in self.blocks)}
