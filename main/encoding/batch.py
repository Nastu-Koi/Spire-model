"""Observations packed into tensors on the CPU; the network consumes only these.

Packing needs the vocabulary but no weights, so it can run in data-loading
worker processes while the GPU is busy with the previous batch.
"""

from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from main.encoding.fields import field_tensors
from main.encoding.vocabulary import REFERENCE_ROLES


def _move(value, device):
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and torch.device(device).type == "cuda" and not value.is_pinned():
            # A copy from pageable memory makes the host wait for everything queued on the
            # stream; one from pinned memory is queued behind it.
            value = value.pin_memory()
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, Packed):
        return value.to(device)
    return value


def given(values, device, dtype=torch.float32):
    """A few host numbers as a tensor on `device`, queued rather than waited for."""
    return _move(torch.tensor(values, dtype=dtype), device)


class Packed:
    def to(self, device):
        return type(self)(**{f.name: _move(getattr(self, f.name), device) for f in fields(self)})


@dataclass
class Fields(Packed):
    ids: torch.Tensor  # [rows, width] symbol indices
    kinds: torch.Tensor  # [rows, width] field-name indices
    numbers: torch.Tensor  # [rows, width, 4]
    mask: torch.Tensor  # [rows, width]

    @classmethod
    def pack(cls, rows, vocabulary):
        return cls(*field_tensors(rows, vocabulary))

    @classmethod
    def cat(cls, parts):
        """The rows of several `Fields` in one, each padded to the widest."""
        width = max(part.mask.shape[1] for part in parts)

        def padded(tensor):
            extra = width - tensor.shape[1]
            return F.pad(tensor, (0, 0) * (tensor.dim() - 2) + (0, extra)) if extra else tensor

        return cls(*(torch.cat([padded(getattr(part, name)) for part in parts])
                     for name in ("ids", "kinds", "numbers", "mask")))


@dataclass
class Rows:
    """Field rows not yet made tensors, and the width `Fields.pack` would give them. A
    sampling process sends the fields of only the rows new to the collecting process's
    tables (`main.play.wire`): it makes tensors of those alone (`fields`), at the width
    all the rows would have had, so that what it sends is the same."""

    rows: list
    width: int
    vocabulary: object

    @classmethod
    def of(cls, rows, vocabulary):
        return cls(rows, max(map(len, rows), default=1), vocabulary)

    def fields(self, keep):
        """`Fields` of the rows `keep` marks."""
        return Fields(*field_tensors([row for row, kept in zip(self.rows, keep) if kept], self.vocabulary,
                                     width=self.width))


@dataclass
class ProgramGroup(Packed):
    """Local programs padded to one length. The fields are those of the real nodes
    only: `nodes` says where each stands among the padded `programs * length` rows,
    so that the encoder computes nothing for the padding and scatters the nodes
    into place (`SharedEncoder.programs`)."""

    fields: Fields  # [nodes, width]
    length: int
    lengths: torch.Tensor  # [programs] nodes per program
    edges: torch.Tensor  # [edges, 4]: program, source node, target node, relation
    binding_targets: torch.Tensor  # flat node row receiving an entity
    binding_sources: torch.Tensor  # flat token row of that entity
    binding_roles: torch.Tensor  # field-name index of the binding role
    slots: torch.Tensor  # [programs] row in the batch's program table
    nodes: torch.Tensor  # [nodes] row of each node in the padded [programs * length] layout


@dataclass
class Batch(Packed):
    tokens: Fields  # every token of every observation, concatenated
    lengths: torch.Tensor  # [observations] tokens per observation
    pad: int  # padded token length of the global sequence
    reference_targets: torch.Tensor
    reference_sources: torch.Tensor
    reference_roles: torch.Tensor
    programs: list[ProgramGroup]
    program_count: int
    token_programs: torch.Tensor  # [tokens] program row of each token
    relations: torch.Tensor  # [edges, 4]: observation, query token, key token, relation
    actions: torch.Tensor  # [observations, max actions] token index, 0 where padded
    unregistered: list[str]  # names met here which the vocabulary never registered


@dataclass
class TableBatch(Batch):
    """A sampling pass's batch whose entities and programs are rows of the collecting
    process's encoder tables (`main.play.wire`, `EncoderTables`). `tokens` holds the
    fields of the tokens new to the tables only; `token_rows` is the table row of every
    token's encoding and `new_token_rows` the row each new one goes to. `programs` holds
    the programs new to the tables: their `slots` and every `token_programs` entry are
    rows of the program table, their `binding_sources` rows of the token table.
    `program_count` is 0. The rest is a `Batch`'s."""

    token_rows: torch.Tensor  # [tokens]
    new_token_rows: torch.Tensor  # [new tokens]
    tables: object  # the EncoderTables the rows are of


def _long(values, width=None):
    tensor = torch.tensor(values, dtype=torch.long)
    return tensor if width is None else tensor.reshape(-1, width)


def program_items(observations):
    """The distinct programs of observations batched together, in the order a batch
    numbers them: each an (effect, bindings) pair whose bindings are (node, token row,
    role) with the token rows counted across the observations; and the program of
    every token."""
    # A program is shared by every token holding the same parsed program with
    # the same entity bindings: repeated cards, and the map behind a frontier
    # node across the decisions of a batch.
    index, items, token_programs, offset = {}, [], [], 0
    for obs in observations:
        for effect in obs.effects:
            bindings = []
            for node, ref, role in effect.bindings:
                if ref not in obs.refs:
                    raise ValueError("Unresolved effect-program entity reference")
                bindings.append((node, offset + obs.refs[ref], role))
            key = (id(effect), tuple(bindings))
            if key not in index:
                index[key] = len(items)
                items.append((effect, bindings))
            token_programs.append(index[key])
        offset += len(obs.tokens)
    return items, token_programs


def collate(observations, vocabulary, pad_to=None, *, tensors=True):
    """The batch of `observations`. Without `tensors`, the token and program fields are
    `Rows`, left for `main.play.wire.pack` to make tensors of what it sends."""
    field, relation = vocabulary.fields.encode, vocabulary.relations.encode
    tabled = (lambda rows: Fields.pack(rows, vocabulary)) if tensors else (lambda rows: Rows.of(rows, vocabulary))
    before = set(vocabulary.unregistered())
    offsets, token_rows = [], []
    for obs in observations:
        offsets.append(len(token_rows))
        token_rows.extend(obs.tokens)

    targets, sources, roles, relations = [], [], [], []
    for index, (obs, offset) in enumerate(zip(observations, offsets)):
        for query, key, role in obs.edges:
            relations.append((index, query, key, relation(role)))
            if role in REFERENCE_ROLES:
                targets.append(offset + query)
                sources.append(offset + key)
                roles.append(field("reference." + role))

    items, token_programs = program_items(observations)

    # Programs of similar size share a group, so a long map program does not
    # pad every single-node program to its length.
    by_length = {}
    for slot, (effect, _) in enumerate(items):
        by_length.setdefault(1 << (len(effect.nodes) - 1).bit_length(), []).append(slot)
    groups = []
    for length, slots in sorted(by_length.items()):
        rows, lengths, edges, nodes = [], [], [], []
        binding_targets, binding_sources, binding_roles = [], [], []
        for row, slot in enumerate(slots):
            effect, bindings = items[slot]
            rows.extend(effect.nodes)
            nodes.extend(row * length + node for node in range(len(effect.nodes)))
            lengths.append(len(effect.nodes))
            table = effect.edge_table(vocabulary)
            if len(table):
                edges.append(F.pad(table, (1, 0), value=row))
            for node, source, role in bindings:
                binding_targets.append(row * length + node)
                binding_sources.append(source)
                binding_roles.append(field("binding." + role))
        groups.append(
            ProgramGroup(
                tabled(rows),
                length,
                _long(lengths),
                torch.cat(edges) if edges else _long([], 4),
                _long(binding_targets),
                _long(binding_sources),
                _long(binding_roles),
                _long(slots),
                _long(nodes),
            )
        )

    width = max(len(obs.action_indices) for obs in observations)
    actions = [
        obs.action_indices + [0] * (width - len(obs.action_indices))
        for obs in observations
    ]
    return Batch(
        tabled(token_rows),
        _long([len(obs.tokens) for obs in observations]),
        max(max(len(obs.tokens) for obs in observations), pad_to or 0),
        _long(targets),
        _long(sources),
        _long(roles),
        groups,
        len(items),
        _long(token_programs),
        _long(relations, 4),
        _long(actions, width),
        sorted(set(vocabulary.unregistered()) - before),
    )
