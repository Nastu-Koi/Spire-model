"""Batched typed encoder, relation-aware Transformer, and a stateless decision head."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from main.env.clock import Laps
from main.network.attention import Attention, relation_bias, relation_biases
from main.encoding.batch import Fields, TableBatch, collate
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

    def _program(self, group, nodes, entities):
        """The vectors of one group's programs, from the field encodings of their real
        nodes; a bound node adds the encoding of its entity, a row of `entities`."""
        count, length = len(group.lengths), group.length
        x = nodes.new_zeros((count * length, nodes.shape[-1])).index_copy_(0, group.nodes, nodes)
        x = x.view(count, length, -1)
        if group.binding_targets.numel():
            additions = self.program_binding(
                entities.index_select(0, group.binding_sources)
                + self.field(group.binding_roles)
            )
            x = x.flatten(0, 1).index_add(0, group.binding_targets, additions).view_as(x)
        valid = torch.arange(length, device=nodes.device).unsqueeze(
            0
        ) < group.lengths.unsqueeze(1)
        bias = relation_bias(self.program_relation, group.edges, valid)
        for block in self.local_blocks:
            x = block(x, valid, bias)
        pooled = (x * valid.unsqueeze(-1)).sum(1) / group.lengths.sqrt().unsqueeze(-1)
        return x[:, 0] + pooled

    def programs(self, batch, base):
        """Encode each distinct local program once; one row per program. The fields
        of a group are those of its real nodes, a small part of the padded rows in a
        late act's pass, whose padding would otherwise bound the pass on the GPU. They are
        encoded compact and scattered into the padded layout, where the padding rows
        stay the zeros the field block would have left."""
        result = base.new_zeros((batch.program_count, base.shape[-1]))
        for group in batch.programs:
            result = result.index_copy(0, group.slots, self._program(group, self.fields(group.fields), base))
        return result

    def _referenced(self, base, batch):
        """Every token's encoding with what the entities it references add to it."""
        if not batch.reference_targets.numel():
            return base
        contributions = self.reference(
            base.index_select(0, batch.reference_sources)
            + self.field(batch.reference_roles)
        )
        return base.index_add(0, batch.reference_targets, contributions)

    def forward(self, batch):
        """Encode every token of a packed batch; returns [tokens, hidden]."""
        base = self.fields(batch.tokens)
        programs = self.programs(batch, base).index_select(0, batch.token_programs)
        return self.projection(self.local_norm(self._referenced(base, batch) + programs))

    def from_tables(self, batch):
        """`forward` of a sampling pass's `TableBatch`: the tokens and programs new to the
        tables are encoded and written to them, then every token's encoding and program
        are read from them. The new tokens and the nodes of every new program go through
        one `fields` pass: there are few of them, and every pass is a round of launches."""
        tables = batch.tables
        parts = [batch.tokens, *(group.fields for group in batch.programs)]
        sizes = [len(part.ids) for part in parts]
        if sum(sizes):
            encoded = self.fields(Fields.cat(parts)).split(sizes)
            if sizes[0]:
                tables.tokens.index_copy_(0, batch.new_token_rows, encoded[0])
            for group, nodes in zip(batch.programs, encoded[1:]):
                # A bound node reads its entity from the table, written above if new.
                tables.programs.index_copy_(0, group.slots, self._program(group, nodes, tables.tokens))
        base = tables.tokens.index_select(0, batch.token_rows)
        programs = tables.programs.index_select(0, batch.token_programs)
        return self.projection(self.local_norm(self._referenced(base, batch) + programs))


class EncoderTables:
    """The entity encoder's output for the tokens and programs a round's sampling
    processes have sent (`main.play.wire`), on the model's device: one row of `tokens`
    or `programs` each, every process owning a range of rows. Rows hold what the weights
    of their pass computed: a round makes tables of its own."""

    def __init__(self, model, token_rows, program_rows):
        width, device = model.config.local_size, model.device
        self.tokens = torch.zeros((token_rows, width), device=device)
        self.programs = torch.zeros((program_rows, width), device=device)


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


# The shapes a graphed pass pads its batch to: decisions (rows) and tokens of a decision
# (pad). One CUDA graph is captured for each pair that comes up; a batch beyond the
# largest runs the layers eagerly.
ROW_BUCKETS = (8, 16, 32, 64, 128, 256)
PAD_BUCKETS = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def bucket(count, buckets):
    """The smallest of `buckets` that holds `count`, or None beyond them."""
    return next((size for size in buckets if size >= count), None)


class GraphedStack:
    """The global layers as CUDA graphs, one per padded shape, for sampling.

    A pass of a handful of decisions is bound by the host: the layers launch a few
    hundred small kernels, and the interpreter and dispatcher take longer to launch
    them than the GPU takes to run them. A graph replays the kernels that one pass of
    a shape launched, the same kernels in the same order, so the numbers are those of
    the eager pass on the padded batch: the layers see a few rows and tokens more, all
    masked. Inputs are copied into the graph's own buffers before each replay; the
    output buffer is read before the next replay overwrites it."""

    def __init__(self, layers):
        self.layers = layers  # layers(hidden, valid, biases) -> hidden, the eager pass
        self.graphs = {}

    def __call__(self, hidden, valid, biases):
        # A graph replays the kernels it captured, at their precision: a pass under
        # autocast and one without are two graphs of a shape.
        key = (tuple(valid.shape), torch.is_autocast_enabled())
        entry = self.graphs.get(key)
        if entry is None:
            entry = self.graphs[key] = self._capture(hidden, valid, biases)
        graph, (static_hidden, static_valid, static_biases), out = entry
        static_hidden.copy_(hidden)
        static_valid.copy_(valid)
        for static, bias in zip(static_biases, biases):
            static.copy_(bias)
        graph.replay()
        return out

    def _capture(self, hidden, valid, biases):
        inputs = (hidden.clone(), valid.clone(), [bias.clone() for bias in biases])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Warm the kernels of this shape (the compiled blocks, the attention's
            # workspaces) outside the capture.
            for _ in range(2):
                self.layers(*inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.layers(*inputs)
        return graph, inputs, out


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
        self._graphs = None  # the global layers as CUDA graphs, for sampling (`graphed`)
        if config.compile_blocks:
            # Module.compile keeps checkpoint state_dict names unchanged.
            for block in self.blocks:
                block.compile(dynamic=True)

    @property
    def device(self):
        return self.query.weight.device

    def graphed(self, enabled=True):
        """Run the global layers of a sampling pass (no gradient, evaluation mode, on
        the GPU) as CUDA graphs (`GraphedStack`), the batch padded to bucketed shapes.
        Training and replay are untouched: they take the eager layers as before."""
        if not enabled:
            self._graphs = None
        elif self._graphs is None:
            self._graphs = GraphedStack(self._layers)
        return self

    def _graphing(self, tokens):
        return self._graphs is not None and tokens.is_cuda and not self.training and not torch.is_grad_enabled()

    def _layers(self, hidden, valid, biases):
        for block, bias in zip(self.blocks, biases):
            hidden = block(hidden, valid, bias)
        return self.final_norm(hidden)

    def _relation_biases(self, edges, valid):
        """Every global layer's relation bias, from as few scatters as the batch's
        pairs allow (`relation_biases`), in the attention dtype."""
        device = valid.device.type
        dtype = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else None
        return relation_biases(list(self.relation), edges, valid, dtype=dtype)

    def _entities(self, batch):
        """Run the entity encoder in FP32 and return its tokens.

        Its batches are small and vary in size with what a decision is batched
        with, so FP32 kernels differ in the last bits between batches, and under
        BF16 the backbone turns that into small log-probability differences per
        decision. The log-probabilities a sampled run records carry this noise,
        and so does the ratio the clipped objective forms with them: its first
        batch starts near 1 rather than at it. An FP64 encoder would remove it
        at several times the backbone's cost.
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            if isinstance(batch, TableBatch):
                return self.encoder.from_tables(batch)
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
        # The host's time at each stage (`clock`): what a sampling pass, bound by the
        # host, spends launching each part of the network.
        laps = Laps()
        tokens = self._entities(batch)
        laps("network_encoder")
        rows, pad, total = len(batch.lengths), batch.pad, tokens.shape[0]
        lengths = batch.lengths
        # A graphed pass pads the batch to a bucketed shape: the rows added hold one
        # valid token each, so that their attention stays finite; they are dropped after.
        graphed = self._graphing(tokens)
        if graphed:
            padded_rows, padded_pad = bucket(rows, ROW_BUCKETS), bucket(pad, PAD_BUCKETS)
            graphed = padded_rows is not None and padded_pad is not None
        if graphed:
            lengths = F.pad(lengths, (0, padded_rows - rows), value=1)
            rows_, pad_ = padded_rows, padded_pad
        else:
            rows_, pad_ = rows, pad
        valid = torch.arange(pad_, device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        # Place every token by a computed slot: a boolean index would make the
        # host wait for the token count before the backbone can be queued.
        row = torch.repeat_interleave(
            torch.arange(rows, device=tokens.device), batch.lengths, output_size=total
        )
        starts = batch.lengths.cumsum(0) - batch.lengths
        slots = row * pad_ + torch.arange(total, device=tokens.device) - starts[row]
        hidden = tokens.new_zeros((rows_ * pad_, tokens.shape[-1])).index_copy_(0, slots, tokens).view(rows_, pad_, -1)
        biases = self._relation_biases(batch.relations, valid)
        laps("network_layout")
        if graphed:
            hidden = self._graphs(hidden, valid, biases)[:rows, :pad]
            laps("network_layers")
            return hidden
        for block, bias in zip(self.blocks, biases):
            if (
                self.config.checkpoint_layers
                and self.training
                and torch.is_grad_enabled()
            ):
                hidden = checkpoint(block, hidden, valid, bias, use_reentrant=False)
            else:
                hidden = block(hidden, valid, bias)
        hidden = self.final_norm(hidden)
        laps("network_layers")
        return hidden

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

    def choose(self, decisions, *, sample=True, temperature=1.0, distributions=False):
        """One action for every decision of a packed batch, in a single pass.

        Returns the chosen slot, its log-probability and its log-probability at
        temperature 1, each [decisions], and with `distributions` every slot's
        log-probability at the decision's temperature [decisions, slots] (−inf off its
        legal slots): drawn from the scores divided by
        `temperature` (one number, or one per decision), or the most likely slot
        when not sampling. A decision whose `explore` row marks slots is drawn among
        those alone, at their relative odds; one with a `rank` of 0 or more takes the
        legal slot of that place in the policy's order, with no draw. Either way its
        log-probability is still the whole distribution's for the slot taken (what the
        policy itself would have drawn it with), so that what is explored is the action
        and not the record.
        """
        batch = decisions.batch
        hidden = self.hidden(batch)
        laps = Laps()
        self.decoder_calls += len(batch.lengths)
        state = hidden[:, 0]
        keys = self._keys(self._actions(hidden, batch))
        scores = self._score(self._query(state), keys, decisions.masks)
        temperature = torch.as_tensor(temperature, dtype=scores.dtype, device=scores.device)
        log_probs = (scores / temperature.view(-1, 1)).log_softmax(-1)
        if sample:
            weights = log_probs.exp()
            explore = decisions.explore & decisions.masks
            weights = torch.where(explore.any(-1, keepdim=True), weights.masked_fill(~explore, 0), weights)
            slots = torch.multinomial(weights, 1).squeeze(1)
            rank = decisions.rank.to(slots.device)
            # Illegal slots score -inf and sort last; a rank names a legal slot (`prepare_decisions`).
            ranked = log_probs.argsort(-1, descending=True).gather(1, rank.clamp_min(0).unsqueeze(1)).squeeze(1)
            slots = torch.where(rank >= 0, ranked, slots)
        else:
            slots = log_probs.argmax(-1)
        at = slots.unsqueeze(1)
        chosen = slots, log_probs.gather(1, at).squeeze(1), scores.log_softmax(-1).gather(1, at).squeeze(1)
        laps("network_heads")
        return (*chosen, log_probs) if distributions else chosen

    def decode(self, encoded, legal_mask):
        return self.decode_batch([encoded], [legal_mask])[0]

    def parameter_report(self):
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "backbone": sum(p.numel() for b in self.blocks for p in b.parameters())
            + sum(p.numel() for p in self.final_norm.parameters()),
            "full_layers": len(self.blocks),
        }
