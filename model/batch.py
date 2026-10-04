"""Observations packed into tensors on the CPU; the network consumes only these.

Packing needs the vocabulary but no weights, so it can run in data-loading
worker processes while the GPU is busy with the previous batch.
"""

from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from .representation import REFERENCE_ROLES, field_tensors


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, Packed):
        return value.to(device)
    return value


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


@dataclass
class ProgramGroup(Packed):
    """Local programs padded to one length."""

    fields: Fields  # [programs * length, width]
    length: int
    lengths: torch.Tensor  # [programs] nodes per program
    edges: torch.Tensor  # [edges, 4]: program, source node, target node, relation
    binding_targets: torch.Tensor  # flat node row receiving an entity
    binding_sources: torch.Tensor  # flat token row of that entity
    binding_roles: torch.Tensor  # field-name index of the binding role
    slots: torch.Tensor  # [programs] row in the batch's program table


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


def _long(values, width=None):
    tensor = torch.tensor(values, dtype=torch.long)
    return tensor if width is None else tensor.reshape(-1, width)


def collate(observations, vocabulary, pad_to=None):
    field, relation = vocabulary.fields.encode, vocabulary.relations.encode
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

    # A program is shared by every token holding the same parsed program with
    # the same entity bindings: repeated cards, and the map behind a frontier
    # node across the decisions of a batch.
    index, items, token_programs = {}, [], []
    for obs, offset in zip(observations, offsets):
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

    # Programs of similar size share a group, so a long map program does not
    # pad every single-node program to its length.
    by_length = {}
    for slot, (effect, _) in enumerate(items):
        by_length.setdefault(1 << (len(effect.nodes) - 1).bit_length(), []).append(slot)
    groups = []
    for length, slots in sorted(by_length.items()):
        rows, lengths, edges = [], [], []
        binding_targets, binding_sources, binding_roles = [], [], []
        for row, slot in enumerate(slots):
            effect, bindings = items[slot]
            rows.extend(effect.nodes + [[]] * (length - len(effect.nodes)))
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
                Fields.pack(rows, vocabulary),
                length,
                _long(lengths),
                torch.cat(edges) if edges else _long([], 4),
                _long(binding_targets),
                _long(binding_sources),
                _long(binding_roles),
                _long(slots),
            )
        )

    width = max(len(obs.action_indices) for obs in observations)
    actions = [
        obs.action_indices + [0] * (width - len(obs.action_indices))
        for obs in observations
    ]
    return Batch(
        Fields.pack(token_rows, vocabulary),
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
