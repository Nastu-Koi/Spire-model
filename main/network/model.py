"""Batched typed encoder, relation-aware Transformer, and a stateless decision head."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from main.network.attention import Attention, relation_bias
from main.encoding.batch import collate
from main.network.config import ModelConfig
from main.encoding.representation import Observation

# Periods of the sinusoidal number features, from 4 up to 2048.
NUMERIC_BANDS = 8
NUMERIC_FEATURES = 5 + 2 * NUMERIC_BANDS


def numeric_features(numbers, frequencies):
    """Expand (value, known, applicable, is_numeric) into magnitude and periodic features.

    The linear and logarithmic terms order values; the periodic terms keep
    nearby integers apart at every magnitude.
    """
    value, flags, numeric = numbers[..., :1], numbers[..., 1:], numbers[..., 3:]
    phase = value * frequencies
    return torch.cat(
        [
            value / 100,
            value.sign() * value.abs().log1p(),
            flags,
            phase.sin() * numeric,
            phase.cos() * numeric,
        ],
        dim=-1,
    )


class SwiGLU(nn.Module):
    """SwiGLU feed-forward layer with an explicit intermediate width."""

    def __init__(self, width, intermediate):
        super().__init__()
        self.input = nn.Linear(width, 2 * intermediate, bias=False)
        self.output = nn.Linear(intermediate, width, bias=False)

    def forward(self, x):
        gate, value = self.input(x).chunk(2, dim=-1)
        return self.output(F.silu(gate) * value)


def scale_residual_outputs(blocks):
    """Shrink each block's residual-branch output projections by 1/sqrt(2 * blocks).

    A residual stream summing 2L branches of unit-scale outputs grows like
    sqrt(2L); scaling the branches down at initialization keeps the stream's
    scale, and hence the usable learning rate, independent of depth.
    """
    factor = 1 / math.sqrt(2 * len(blocks))
    with torch.no_grad():
        for block in blocks:
            block.attention.output.weight.mul_(factor)
            block.ffn.output.weight.mul_(factor)


class Block(nn.Module):
    def __init__(self, width, heads, ffn_size, *, backend="sdpa"):
        super().__init__()
        self.norm1 = nn.RMSNorm(width)
        self.attention = Attention(width, heads, backend=backend)
        self.norm2 = nn.RMSNorm(width)
        self.ffn = SwiGLU(width, ffn_size)

    def forward(self, x, valid, bias):
        x = x + self.attention(self.norm1(x), valid, bias)
        return (x + self.ffn(self.norm2(x))) * valid.unsqueeze(-1)


class SharedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        local = config.local_size
        self.symbol = nn.Embedding(config.vocabulary_size, local, padding_idx=0)
        self.field = nn.Embedding(config.field_size, local, padding_idx=0)
        # The number encoder sees which field it encodes: a field-blind encoder
        # followed by a sum over fields cannot tell hp=12, block=40 from
        # hp=40, block=12.
        self.numeric = nn.Sequential(
            nn.Linear(local + NUMERIC_FEATURES, local, bias=False),
            nn.GELU(),
            nn.Linear(local, local, bias=False),
        )
        self.register_buffer(
            "frequencies",
            2 * math.pi / (4 * 512 ** (torch.arange(NUMERIC_BANDS) / (NUMERIC_BANDS - 1))),
        )
        self.field_norm = nn.RMSNorm(local)
        # 8*ceil(d/3) keeps SwiGLU near the parameter count of a 4d GELU FFN.
        local_ffn = 8 * math.ceil(local / 3)
        # The fields of one entity attend to each other before they are summed, so
        # a combination (a cost with a keyword) is seen at the entity, not left for
        # the global layers to rediscover.
        self.field_block = Block(local, config.local_heads, local_ffn, backend=config.backend)
        self.local_blocks = nn.ModuleList(
            [
                Block(local, config.local_heads, local_ffn, backend=config.backend)
                for _ in range(config.local_layers)
            ]
        )
        scale_residual_outputs(self.local_blocks)
        scale_residual_outputs([self.field_block])
        self.program_relation = nn.Embedding(
            config.relation_size, config.local_heads, padding_idx=0
        )
        self.reference = nn.Linear(local, local, bias=False)
        self.program_binding = nn.Linear(local, local, bias=False)
        self.local_norm = nn.RMSNorm(local)
        self.projection = nn.Linear(local, config.hidden_size, bias=False)

    def fields(self, packed):
        identity = self.symbol(packed.ids) + self.field(packed.kinds)
        numbers = numeric_features(packed.numbers.to(identity.dtype), self.frequencies)
        x = identity + self.numeric(torch.cat([identity, numbers], dim=-1))
        mask = packed.mask
        valid = mask.bool()
        # Padded fields are excluded as keys; a row without fields (padding of a
        # program group) attends uniformly to stay finite and is zeroed by the block.
        excluded = ~valid & valid.any(1, keepdim=True)
        bias = x.new_zeros((len(x), 1, 1, valid.shape[1])).masked_fill(
            excluded[:, None, None, :], -float("inf")
        )
        x = self.field_block(x, valid, bias)
        # Sum retains multiplicity; sqrt normalization controls magnitude only.
        return self.field_norm(
            (x * mask.unsqueeze(-1)).sum(1)
            / mask.sum(1).clamp_min(1).sqrt().unsqueeze(-1)
        )

    def programs(self, batch, base):
        """Encode each distinct local program once; one row per program."""
        result = base.new_zeros((batch.program_count, base.shape[-1]))
        for group in batch.programs:
            count, length = len(group.lengths), group.length
            x = self.fields(group.fields).view(count, length, -1)
            if group.binding_targets.numel():
                additions = self.program_binding(
                    base.index_select(0, group.binding_sources)
                    + self.field(group.binding_roles)
                )
                x = x.flatten(0, 1).index_add(0, group.binding_targets, additions).view_as(x)
            valid = torch.arange(length, device=base.device).unsqueeze(
                0
            ) < group.lengths.unsqueeze(1)
            bias = relation_bias(self.program_relation, group.edges, valid)
            for block in self.local_blocks:
                x = block(x, valid, bias)
            pooled = (x * valid.unsqueeze(-1)).sum(1) / group.lengths.sqrt().unsqueeze(-1)
            result = result.index_copy(0, group.slots, x[:, 0] + pooled)
        return result

    def forward(self, batch):
        """Encode every token of a packed batch; returns [tokens, hidden]."""
        base = self.fields(batch.tokens)
        fused = base
        if batch.reference_targets.numel():
            contributions = self.reference(
                base.index_select(0, batch.reference_sources)
                + self.field(batch.reference_roles)
            )
            fused = fused.index_add(0, batch.reference_targets, contributions)
        programs = self.programs(batch, base).index_select(0, batch.token_programs)
        return self.projection(self.local_norm(fused + programs))


@dataclass
class Encoded:
    hidden: torch.Tensor
    actions: torch.Tensor
    keys: torch.Tensor
    observation: Observation


@dataclass
class DecisionOutput:
    logits: torch.Tensor

    def distribution(self, temperature=1.0):
        return torch.distributions.Categorical(logits=self.logits.float() / temperature)

    def topk(self, k):
        if k < 1:
            raise ValueError("k must be positive")
        count = int(torch.isfinite(self.logits).sum())
        values, indices = torch.topk(self.logits, min(k, count))
        return indices, values


class PolicyNetwork(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.encoder = SharedEncoder(config)
        self.blocks = nn.ModuleList(
            [
                Block(d, config.num_heads, config.ffn_size, backend=config.backend)
                for _ in range(config.layers)
            ]
        )
        scale_residual_outputs(self.blocks)
        # One bias per relation type and head: entity references, their reverses
        # and engine relations. A table per global layer, so that each layer
        # follows relations in its own way.
        self.relation = nn.ModuleList(
            [
                nn.Embedding(config.relation_size, config.num_heads, padding_idx=0)
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nn.RMSNorm(d)
        # The global token (index 0) carries phase and selection progress; its
        # final state is the decision query. Selection progress is written into
        # the observation, so no state is carried between decisions.
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        # Normalized query and keys bound the decision logits as the width grows.
        self.query_norm = nn.RMSNorm(d)
        self.key_norm = nn.RMSNorm(d)
        self.encoder_calls = 0
        self.decoder_calls = 0
        if config.compile_blocks:
            # Module.compile keeps checkpoint state_dict names unchanged.
            for block in self.blocks:
                block.compile(dynamic=True)

    @property
    def device(self):
        return self.query.weight.device

    def _relation_bias(self, table, edges, valid):
        bias = relation_bias(table, edges, valid)
        # Every head of the layer reads this bias: convert it to the attention dtype once.
        device = valid.device.type
        if torch.is_autocast_enabled(device):
            bias = bias.to(torch.get_autocast_dtype(device))
        return bias.contiguous()

    def _entities(self, batch):
        """Run the entity encoder in FP32 and return its tokens.

        Its batches are small and vary in size with what a decision is batched
        with, so FP32 kernels differ in the last bits between batches, and under
        BF16 the backbone turns that into log-probability differences of about
        1e-3 per decision. The log-probabilities a sampled run records carry
        this noise; nothing is learned from them as a ratio. An FP64 encoder
        would remove it at several times the backbone's cost.
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            return self.encoder(batch)

    def hidden(self, batch):
        """Final token states of a packed batch: [observations, pad, hidden]."""
        # Sampling and differentiable replay share this boundary.
        if self.config.encoder_precision == "fp32":
            with torch.autocast(device_type=self.device.type, enabled=False):
                return self._hidden(batch)
        return self._hidden(batch)

    def _hidden(self, batch):
        self.encoder_calls += 1
        tokens = self._entities(batch)
        rows, pad, total = len(batch.lengths), batch.pad, tokens.shape[0]
        valid = torch.arange(pad, device=tokens.device).unsqueeze(
            0
        ) < batch.lengths.unsqueeze(1)
        # Place every token by a computed slot: a boolean index would make the
        # host wait for the token count before the backbone can be queued.
        row = torch.repeat_interleave(
            torch.arange(rows, device=tokens.device), batch.lengths, output_size=total
        )
        starts = batch.lengths.cumsum(0) - batch.lengths
        slots = row * pad + torch.arange(total, device=tokens.device) - starts[row]
        hidden = (
            tokens.new_zeros((rows * pad, tokens.shape[-1]))
            .index_copy(0, slots, tokens)
            .view(rows, pad, -1)
        )
        for block, table in zip(self.blocks, self.relation):
            bias = self._relation_bias(table, batch.relations, valid)
            if (
                self.config.checkpoint_layers
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(block, hidden, valid, bias, use_reentrant=False)
            else:
                hidden = block(hidden, valid, bias)
        return self.final_norm(hidden)

    def encode(self, observations, vocabulary, *, pad_to=None):
        if not observations:
            return []
        batch = collate(observations, vocabulary, pad_to).to(self.device)
        hidden = self.hidden(batch)
        actions = self._actions(hidden, batch)
        keys = self._keys(actions)
        result = []
        for b, obs in enumerate(observations):
            count = len(obs.action_indices)
            result.append(
                Encoded(hidden[b, : len(obs.tokens)], actions[b, :count], keys[b, :count], obs)
            )
        return result

    @staticmethod
    def _actions(hidden, batch):
        return hidden.gather(
            1, batch.actions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )

    def _heads(self, head, x):
        # The decision heads see a handful of rows whose count varies with the
        # batch; FP32 keeps their output independent of that count.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return head(x.float())

    def _query(self, state):
        return self.query_norm(self._heads(self.query, state))

    def _keys(self, actions):
        return self.key_norm(self._heads(self.key, actions))

    def _score(self, queries, keys, masks):
        with torch.autocast(device_type=queries.device.type, enabled=False):
            logits = torch.bmm(queries.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
            logits = logits / math.sqrt(self.config.hidden_size)
            return logits.masked_fill(~masks, -float("inf"))

    def decode_batch(self, encoded_list, legal_masks):
        """Score legal actions of independent decisions in one batched head pass."""
        count = len(encoded_list)
        if not count or len(legal_masks) != count:
            raise ValueError("Decoder batch inputs must have the same nonzero length")
        for encoded, mask in zip(encoded_list, legal_masks):
            if mask.dtype != torch.bool or mask.shape != (encoded.actions.shape[0],):
                raise ValueError("Invalid engine mask")
        legal_counts = torch.stack([mask.sum() for mask in legal_masks])
        if bool((legal_counts < 2).any()):
            raise ValueError("Forced actions must bypass every model head")
        self.decoder_calls += count
        state = torch.stack([encoded.hidden[0] for encoded in encoded_list])
        max_actions = max(encoded.actions.shape[0] for encoded in encoded_list)
        keys = torch.stack(
            [
                F.pad(encoded.keys, (0, 0, 0, max_actions - encoded.keys.shape[0]))
                for encoded in encoded_list
            ]
        )
        masks = torch.stack(
            [
                F.pad(mask, (0, max_actions - mask.shape[0]), value=False)
                for mask in legal_masks
            ]
        )
        logits = self._score(self._query(state), keys, masks)
        return [
            DecisionOutput(logits[index, : encoded.actions.shape[0]])
            for index, encoded in enumerate(encoded_list)
        ]

    def _replay_scores(self, prepared):
        batch = prepared.batch
        hidden = self.hidden(batch)
        self.decoder_calls += len(batch.lengths)
        state = hidden[:, 0]
        keys = self._keys(self._actions(hidden, batch))
        return self._score(self._query(state), keys, prepared.masks)

    @staticmethod
    def _tempered(logits, prepared, temperatures):
        if temperatures is None:
            return logits.log_softmax(-1)
        temperatures = torch.as_tensor(temperatures, dtype=logits.dtype, device=logits.device)
        return (logits / temperatures[prepared.owners].unsqueeze(1)).log_softmax(-1)

    def decision_log_probs(self, prepared, temperatures=None):
        """[decisions, slots] log-probabilities of every decision of a prepared replay
        batch, each at its macro-action's temperature (`temperatures`, one per
        macro-action; 1 without): what another policy is compared with (`replay`)."""
        return self._tempered(self._replay_scores(prepared), prepared, temperatures)

    def replay(self, prepared, *, with_accuracy=False, temperatures=None, reference=None):
        """Differentiable macro-action statistics of a prepared replay batch.

        Returns the summed log-probability and the summed entropy of every
        macro-action, as two [macros] tensors. With accuracy enabled, also return
        [macros, 2] integer counts of correct Top-1 decisions and all decisions.
        Forced choices do not count.

        With `temperatures` (one per macro-action) the log-probability is the one at
        that temperature, the distribution the macro-action was drawn from; the
        entropy stays the one at temperature 1. With `reference` (another policy's
        `decision_log_probs` of the same batch, at the same temperatures) the summed
        KL(this ‖ reference) of every macro-action's decisions comes last.
        """
        logits = self._replay_scores(prepared)
        log_probs = logits.log_softmax(-1)
        drawn = log_probs if temperatures is None else self._tempered(logits, prepared, temperatures)
        chosen = drawn.gather(1, prepared.labels.unsqueeze(1)).squeeze(1)
        entropy = -(log_probs.exp() * log_probs.masked_fill(~prepared.masks, 0)).sum(-1)
        macros = len(prepared.first)
        result = [
            chosen.new_zeros(macros).index_add(0, prepared.owners, chosen),
            entropy.new_zeros(macros).index_add(0, prepared.owners, entropy),
        ]
        if with_accuracy:
            # Observe the same pre-update forward pass, without adding gradients
            # or weighting short and long macros equally in the decision metric.
            decisions = prepared.masks.sum(-1) > 1
            correct = (logits.detach().argmax(-1) == prepared.labels) & decisions
            counts = torch.stack((correct.long(), decisions.long()), dim=1)
            result.append(counts.new_zeros((macros, 2)).index_add(0, prepared.owners, counts))
        if reference is not None:
            # Illegal slots are -inf in both: their difference is set aside before it is weighed.
            kl = (drawn.exp() * (drawn - reference).masked_fill(~prepared.masks, 0)).sum(-1)
            result.append(kl.new_zeros(macros).index_add(0, prepared.owners, kl))
        return tuple(result)

    def choose(self, decisions, *, sample=True, temperature=1.0):
        """One action for every decision of a packed batch, in a single pass.

        Returns the chosen slot, its log-probability and its log-probability at
        temperature 1, each [decisions]: drawn from the scores divided by
        `temperature` (one number, or one per decision), or the most likely slot
        when not sampling. The last is what a sample taken at another temperature
        is corrected by (`main/learn/awr.py`).
        """
        batch = decisions.batch
        hidden = self.hidden(batch)
        self.decoder_calls += len(batch.lengths)
        state = hidden[:, 0]
        keys = self._keys(self._actions(hidden, batch))
        scores = self._score(self._query(state), keys, decisions.masks)
        temperature = torch.as_tensor(temperature, dtype=scores.dtype, device=scores.device)
        log_probs = (scores / temperature.view(-1, 1)).log_softmax(-1)
        slots = (
            torch.multinomial(log_probs.exp(), 1).squeeze(1) if sample else log_probs.argmax(-1)
        )
        at = slots.unsqueeze(1)
        return slots, log_probs.gather(1, at).squeeze(1), scores.log_softmax(-1).gather(1, at).squeeze(1)

    def decode(self, encoded, legal_mask):
        return self.decode_batch([encoded], [legal_mask])[0]

    def parameter_report(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "backbone": sum(p.numel() for b in self.blocks for p in b.parameters())
            + sum(p.numel() for p in self.final_norm.parameters()),
            "full_layers": len(self.blocks),
        }
