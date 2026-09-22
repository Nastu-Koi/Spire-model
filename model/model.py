"""Batched typed encoder, full map-aware Transformer, and multi-select GRU."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .attention import MapAttention, MapAttentionBias, RelationAttention
from .config import ModelConfig
from .representation import Observation, Vocabulary, bucket, field_tensors, fields_of


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer with an explicit intermediate width."""

    def __init__(self, width, intermediate):
        super().__init__()
        self.input = nn.Linear(width, 2 * intermediate)
        self.output = nn.Linear(intermediate, width)

    def forward(self, x):
        gate, value = self.input(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


class GlobalBlock(nn.Module):
    def __init__(self, width, heads, ffn_size, *, backend="reference", relations=128):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = MapAttention(width, heads, backend=backend, relations=relations)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = SwiGLU(width, ffn_size)

    def forward(self, x, valid, map_bias):
        x = x + self.attention(self.norm1(x), valid, map_bias)
        return (x + self.ffn(self.norm2(x))) * valid.unsqueeze(-1)


class LocalBlock(nn.Module):
    def __init__(self, width, heads, ffn_size, *, relations=128, backend="reference"):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = RelationAttention(width, heads, relations=relations, backend=backend)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = SwiGLU(width, ffn_size)

    def forward(self, x, valid, edges):
        x = x + self.attention(self.norm1(x), valid, edges)
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
        # 8*ceil(d/3) keeps SwiGLU near the parameter count of a 4d GELU FFN.
        local_ffn = 8 * math.ceil(local / 3)
        self.local_blocks = nn.ModuleList([LocalBlock(local, config.local_heads, local_ffn,
                                                     relations=config.relation_buckets, backend=config.backend)
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

    def forward(self, observations, vocabulary):
        """Encode every token and effect program in one cross-session batch."""
        if not observations:
            return []
        token_offsets, token_rows, effects = [], [], []
        offset = 0
        for obs in observations:
            token_offsets.append(offset)
            token_rows.extend(obs.tokens)
            effects.extend(obs.effects)
            offset += len(obs.tokens)

        base = self.fields(token_rows, vocabulary)
        reference_targets, reference_sources, reference_roles = [], [], []
        accepted = {"source", "target", "option", "owner"}
        for obs, token_offset in zip(observations, token_offsets):
            for target, source, role in obs.edges:
                if role in accepted:
                    reference_targets.append(token_offset + target)
                    reference_sources.append(token_offset + source)
                    reference_roles.append(bucket("reference." + role, self.config.field_buckets))
        fused = base
        if reference_targets:
            targets = torch.tensor(reference_targets, dtype=torch.long, device=base.device)
            sources = torch.tensor(reference_sources, dtype=torch.long, device=base.device)
            roles = torch.tensor(reference_roles, dtype=torch.long, device=base.device)
            contributions = self.reference(base.index_select(0, sources) + self.field(roles))
            fused = fused.index_add(0, targets, contributions.to(fused.dtype))

        program_length = max(max(len(effect.nodes), 1) for effect in effects)
        rows = []
        for effect in effects:
            rows.extend(list(effect.nodes) + [[]] * (program_length - len(effect.nodes)))
        programs = self.fields(rows, vocabulary).view(len(effects), program_length, -1)
        lengths = torch.tensor([len(effect.nodes) for effect in effects], device=base.device)
        valid = torch.arange(program_length, device=base.device).unsqueeze(0) < lengths.unsqueeze(1)

        binding_targets, binding_sources, binding_roles = [], [], []
        program_edges = []
        effect_offset = 0
        for obs, token_offset in zip(observations, token_offsets):
            for token, effect in enumerate(obs.effects):
                for node, ref, role in effect.bindings:
                    if ref not in obs.refs:
                        raise ValueError("Unresolved effect-program entity reference")
                    binding_targets.append((effect_offset + token) * program_length + node)
                    binding_sources.append(token_offset + obs.refs[ref])
                    binding_roles.append(bucket("binding." + role, self.config.field_buckets))
                program_edges.extend((effect_offset + token, source, target,
                                      bucket(role, self.config.relation_buckets))
                                     for source, target, role in effect.edges)
            effect_offset += len(obs.effects)

        if binding_targets:
            targets = torch.tensor(binding_targets, dtype=torch.long, device=base.device)
            sources = torch.tensor(binding_sources, dtype=torch.long, device=base.device)
            roles = torch.tensor(binding_roles, dtype=torch.long, device=base.device)
            additions = self.program_binding(base.index_select(0, sources) + self.field(roles))
            flat_programs = programs.flatten(0, 1)
            programs = flat_programs.index_add(0, targets, additions.to(flat_programs.dtype)).view_as(programs)
        edges = torch.tensor(program_edges, dtype=torch.long, device=base.device).reshape(-1, 4)
        for block in self.local_blocks:
            programs = block(programs, valid, edges)
        pooled = (programs * valid.unsqueeze(-1)).sum(1) / valid.sum(1).clamp_min(1).sqrt().unsqueeze(-1)
        local = self.local_norm(fused + programs[:, 0] + pooled)
        projected = self.projection(local)

        result = []
        for obs, token_offset in zip(observations, token_offsets):
            end = token_offset + len(obs.tokens)
            result.append((projected[token_offset:end], local[token_offset:end]))
        return result


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
        self.blocks = nn.ModuleList([GlobalBlock(d, config.num_heads, config.ffn_size,
                                                backend=config.backend, relations=config.relation_buckets)
                                     for _ in range(config.layers)])
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
                block.compile(dynamic=True)

    @property
    def device(self):
        return self.bos.device

    def _map_bias(self, observations, token_count, device):
        map_count = max(max(len(obs.map_floors) for obs in observations), 1)
        node_rows = [[-1] * token_count for _ in observations]
        category_rows = [[[0] * map_count for _ in range(map_count)] for _ in observations]
        floor_rows = [[0] * map_count for _ in observations]
        for batch, obs in enumerate(observations):
            nodes = list(obs.map_floors)
            slots = {node: slot for slot, node in enumerate(nodes)}
            for slot, node in enumerate(nodes):
                node_rows[batch][node] = slot
                floor_rows[batch][slot] = obs.map_floors[node]
            for source, target, role in obs.edges:
                if role.startswith("map_") and source in slots and target in slots:
                    category_rows[batch][slots[source]][slots[target]] = bucket(role, self.config.relation_buckets)
        # One transfer per compact table avoids scalar CUDA assignments.
        return MapAttentionBias(torch.tensor(node_rows, dtype=torch.long, device=device),
                                torch.tensor(category_rows, dtype=torch.long, device=device),
                                torch.tensor(floor_rows, dtype=torch.long, device=device))

    def encode(self, observations, vocabulary, *, pad_to=None):
        if not observations:
            return []
        self.encoder_calls += 1
        encoded = self.encoder(observations, vocabulary)
        n = max(max(len(o.tokens) for o in observations), pad_to or 0)
        hidden = torch.stack([F.pad(x, (0, 0, 0, n - x.shape[0])) for x, _ in encoded])
        lengths = torch.tensor([len(obs.tokens) for obs in observations], device=hidden.device)
        valid = torch.arange(n, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
        map_bias = self._map_bias(observations, n, hidden.device)
        for block in self.blocks:
            if self.config.checkpoint_layers and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, valid, map_bias, use_reentrant=False)
            else:
                hidden = block(hidden, valid, map_bias)
        hidden = self.final_norm(hidden)
        result = []
        for b, obs in enumerate(observations):
            h = hidden[b, :len(obs.tokens)]
            actions = h[obs.action_indices]
            result.append(Encoded(h, encoded[b][1], actions, self.key(actions), obs))
        return result

    def _prefix_batch(self, encoded_list, contexts, vocabulary):
        local = self.encoder.fields([fields_of(context) for context in contexts], vocabulary)
        prefixes = []
        for encoded, context, context_local in zip(encoded_list, contexts, local):
            selected = context.get("selected_refs", [])
            order_known = (context.get("known_masks") or {}).get("order_matters", True)
            if context.get("order_matters") and order_known:
                state = torch.zeros_like(context_local)
                for ref in selected:
                    state = self.prefix_order(encoded.local[encoded.observation.refs[ref]], state)
                context_local = context_local + state
            elif selected:
                indices = torch.tensor([encoded.observation.refs[ref] for ref in selected],
                                       dtype=torch.long, device=context_local.device)
                context_local = context_local + encoded.local.index_select(0, indices).sum(0)
            prefixes.append(context_local)
        return self.prefix_projection(torch.stack(prefixes))

    def prefix(self, encoded, context, vocabulary):
        return self._prefix_batch([encoded], [context], vocabulary)[0]

    def decode_batch(self, encoded_list, legal_masks, contexts, vocabulary, *,
                     hidden=None, previous=None, value=None):
        """Decode independent multi-select sessions with one batched head pass."""
        count = len(encoded_list)
        if not count or len(legal_masks) != count or len(contexts) != count:
            raise ValueError("Decoder batch inputs must have the same nonzero length")
        hidden = [None] * count if hidden is None else hidden
        previous = [None] * count if previous is None else previous
        value = [True] * count if value is None else value
        if len(hidden) != count or len(previous) != count or len(value) != count:
            raise ValueError("Decoder state lists must match the encoded batch")
        for encoded, mask in zip(encoded_list, legal_masks):
            if mask.dtype != torch.bool or mask.shape != (encoded.actions.shape[0],):
                raise ValueError("Invalid engine mask")
        legal_counts = torch.stack([mask.sum() for mask in legal_masks])
        if bool((legal_counts < 2).any()):
            raise ValueError("Forced actions must bypass every model head")

        self.decoder_calls += count
        prefix = self._prefix_batch(encoded_list, contexts, vocabulary)
        starts = torch.stack([encoded.hidden[0] + encoded.actions[mask].mean(0) + item_prefix
                              for encoded, mask, item_prefix in zip(encoded_list, legal_masks, prefix)])
        previous_embeddings = []
        for encoded, item in zip(encoded_list, previous):
            if item is None:
                previous_embeddings.append(self.bos)
            elif isinstance(item, torch.Tensor):
                previous_embeddings.append(item)
            else:
                previous_embeddings.append(encoded.actions[item])
        current = self.input_norm(starts + self.previous_projection(torch.stack(previous_embeddings)))
        hidden_batch = torch.stack([torch.zeros_like(current_item) if state is None else state
                                    for current_item, state in zip(current, hidden)])
        next_hidden = self.gru(current, hidden_batch)
        queries = self.query(next_hidden)

        max_actions = max(encoded.actions.shape[0] for encoded in encoded_list)
        keys = torch.stack([F.pad(encoded.keys, (0, 0, 0, max_actions - encoded.keys.shape[0]))
                            for encoded in encoded_list])
        masks = torch.stack([F.pad(mask, (0, max_actions - mask.shape[0]), value=False)
                             for mask in legal_masks])
        with torch.autocast(device_type=queries.device.type, enabled=False):
            logits = torch.bmm(queries.float().unsqueeze(1), keys.float().transpose(1, 2)).squeeze(1)
            logits = logits / math.sqrt(self.config.hidden_size)
            logits = logits.masked_fill(~masks, -float("inf"))
        values = self.value(starts).float().squeeze(-1) if any(value) else None
        return [DecisionOutput(logits[index, :encoded.actions.shape[0]], next_hidden[index],
                               values[index] if values is not None and value[index] else None)
                for index, encoded in enumerate(encoded_list)]

    def decode(self, encoded, legal_mask, context, vocabulary, *, hidden=None, previous=None, value=True):
        return self.decode_batch([encoded], [legal_mask], [context], vocabulary,
                                 hidden=[hidden], previous=[previous], value=[value])[0]

    def parameter_report(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "backbone": sum(p.numel() for b in self.blocks for p in b.parameters()) + sum(p.numel() for p in self.final_norm.parameters()),
                "gru": sum(p.numel() for p in self.gru.parameters()),
                "full_layers": len(self.blocks),
                "linear_layers": 0}
