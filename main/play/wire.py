"""Packed decisions as three flat arrays, to cross from sampling processes to the model.

A pack as the network reads it is some eighty padded tensors. The process that runs
the network is bound by its own interpreter: what it does for every tensor of every
message it receives is time between two passes. Here a pack travels without its
padding as a header, one array of integers and one of numbers, and the packs of any
number of processes are joined on the device by a fixed number of array operations.

`unpack(map(pack, packs))` holds the decisions of the packs in order. It is the pack
`prepare_decisions` gives them together, except for the programs: one that two packs
hold is encoded once for each, and programs are padded into fewer groups.
"""

import numpy as np
import torch

from main.encoding.batch import Batch, Fields, ProgramGroup
from main.network.policy import Decisions

# Program groups of 1, 2, 4, ... nodes.
BUCKETS = 12
# Integer fields of a pack, then those of each program group. A field of padded
# rows travels as the length of every row and the cells its mask keeps.
GLOBAL = ("rows", "ids", "kinds", "lengths", "reference_targets", "reference_sources",
          "reference_roles", "token_programs", "relation_observation", "relation_query",
          "relation_key", "relation_role", "actions", "legal")
GROUP = ("rows", "ids", "kinds", "lengths", "edge_program", "edge_source", "edge_target",
         "edge_relation", "binding_targets", "binding_sources", "binding_roles", "slots", "nodes")
INTEGERS = len(GLOBAL) + BUCKETS * len(GROUP)
NUMBERS = 1 + BUCKETS  # the numbers of the tokens, then of each group's nodes
# What a pack counts: the rows that the indices of later packs move by.
TOKENS, OBSERVATIONS, PROGRAMS, GROUPS = 0, 1, 2, 3
COUNTS = GROUPS + BUCKETS
# Header: the length of every field, the counts, then the widths the padding had:
# token fields, each group's node fields, actions, and the padded token length.
WIDTHS = 1 + BUCKETS + 2
HEADER = INTEGERS + NUMBERS + COUNTS + WIDTHS


def _group(bucket, name):
    return len(GLOBAL) + bucket * len(GROUP) + GROUP.index(name)


def _moves():
    """For every integer field, the count its values move by and how many times."""
    count, times = np.full(INTEGERS, -1), np.ones(INTEGERS, dtype=np.int64)
    for name, moved in (("reference_targets", TOKENS), ("reference_sources", TOKENS),
                        ("token_programs", PROGRAMS), ("relation_observation", OBSERVATIONS)):
        count[GLOBAL.index(name)] = moved
    for bucket in range(BUCKETS):
        count[_group(bucket, "edge_program")] = GROUPS + bucket
        # A binding names a node by its place among the padded nodes of the group, as
        # does the row of each node.
        for name in ("binding_targets", "nodes"):
            count[_group(bucket, name)] = GROUPS + bucket
            times[_group(bucket, name)] = 1 << bucket
        count[_group(bucket, "binding_sources")] = TOKENS
        count[_group(bucket, "slots")] = PROGRAMS
    return count, times


MOVED_BY, MOVED_TIMES = _moves()
EMPTY = np.zeros(0, dtype=np.int32)
# The buckets `unpack` pads into one program group each. Every group is another round
# of the entity encoder's loop, whatever it holds; the fields of a group are computed
# for its real nodes only, so padding programs further costs the local blocks alone.
GROUPED = ((0, 1, 2), tuple(range(3, BUCKETS)))


def pack(decisions):
    """(header, integers, numbers, unregistered names) of packed decisions."""
    batch = decisions.batch
    if decisions.masks.shape != batch.actions.shape:
        raise ValueError("Legal masks must cover the action slots")
    integers, numbers = [EMPTY] * INTEGERS, [np.zeros(0, dtype=np.float32)] * NUMBERS
    counts, widths = np.zeros(COUNTS, dtype=np.int64), np.zeros(WIDTHS, dtype=np.int64)

    def rows(first, number, fields):
        mask = fields.mask.numpy()
        integers[first] = mask.sum(1)
        integers[first + 1] = fields.ids.numpy()[mask]
        integers[first + 2] = fields.kinds.numpy()[mask]
        numbers[number] = fields.numbers.numpy()[mask].ravel()
        return mask.shape[1]

    widths[0] = rows(0, 0, batch.tokens)
    relations = batch.relations.numpy()
    for name, values in (
        ("lengths", batch.lengths), ("reference_targets", batch.reference_targets),
        ("reference_sources", batch.reference_sources), ("reference_roles", batch.reference_roles),
        ("token_programs", batch.token_programs), ("actions", batch.actions),
        ("legal", decisions.masks),
    ):
        integers[GLOBAL.index(name)] = values.numpy().ravel()
    for column, name in enumerate(GLOBAL[8:12]):
        integers[GLOBAL.index(name)] = relations[:, column]
    counts[[TOKENS, OBSERVATIONS, PROGRAMS]] = len(batch.tokens.ids), len(batch.lengths), batch.program_count
    for group in batch.programs:
        bucket = group.length.bit_length() - 1
        if group.length != 1 << bucket or bucket >= BUCKETS:
            raise ValueError("Program group length is not a supported power of two")
        widths[1 + bucket] = rows(_group(bucket, "rows"), 1 + bucket, group.fields)
        edges = group.edges.numpy()
        for column, name in enumerate(GROUP[4:8]):
            integers[_group(bucket, name)] = edges[:, column]
        for name in ("lengths", "binding_targets", "binding_sources", "binding_roles", "slots", "nodes"):
            integers[_group(bucket, name)] = getattr(group, name).numpy()
        counts[GROUPS + bucket] = len(group.lengths)
    widths[-2:] = batch.actions.shape[1], batch.pad
    header = np.concatenate((
        [len(field) for field in integers], [len(field) for field in numbers], counts, widths))
    return (header, np.concatenate(integers, dtype=np.int32, casting="unsafe"),
            np.concatenate(numbers), batch.unregistered)


def size(message):
    """How many decisions a packed message holds."""
    return int(message[0][INTEGERS + NUMBERS + OBSERVATIONS])


def _joined(buffers, lengths, device, moves=None):
    """The fields of every message, each field's values together: the tensor and the
    bounds of the fields in it. `lengths` is [messages, fields]; `moves` what to add
    to the values of each message's field."""
    buffer = torch.from_numpy(np.concatenate(buffers)).to(device)
    by_message, by_field = lengths.ravel(), lengths.T.ravel()
    bounds = np.concatenate(([0], np.cumsum(lengths.sum(0))))
    if len(buffers) > 1:
        # Messages arrive one after another; place the pieces field after field.
        source = (np.cumsum(by_message) - by_message).reshape(lengths.shape).T.ravel()
        shift = source - (np.cumsum(by_field) - by_field)
        repeats = torch.from_numpy(by_field).to(device)
        index = torch.arange(len(buffer), device=device) + torch.repeat_interleave(
            torch.from_numpy(shift).to(device), repeats, output_size=len(buffer))
        buffer = buffer[index]
        if moves is not None:
            buffer = buffer.long() + torch.repeat_interleave(
                torch.from_numpy(moves.T.ravel()).to(device), repeats, output_size=len(buffer))
    return (buffer.long() if moves is not None else buffer), bounds


def _mask(rows, width):
    """Which cells of rows padded to `width` hold one of their `rows` values."""
    return torch.arange(width, device=rows.device).unsqueeze(0) < rows.unsqueeze(1)


def _padded(mask, cells, *, shape=()):
    """The padded tensor whose `mask` cells are `cells`, in order."""
    place = mask.view(*mask.shape, *(1,) * len(shape))
    return cells.new_zeros((*mask.shape, *shape)).masked_scatter_(place, cells)


def unpack(messages, device):
    """The decisions of `pack`ed messages on `device`, in their order, and the names
    they met which the vocabulary never registered."""
    headers = np.stack([message[0] for message in messages])
    counts = headers[:, INTEGERS + NUMBERS:][:, :COUNTS]
    widths = headers[:, -WIDTHS:].max(0)
    starts = np.cumsum(counts, 0) - counts
    moves = np.where(MOVED_BY >= 0, starts[:, MOVED_BY] * MOVED_TIMES, 0)
    integers, bounds = _joined([m[1] for m in messages], headers[:, :INTEGERS], device, moves)
    numbers, number_bounds = _joined(
        [m[2] for m in messages], headers[:, INTEGERS:INTEGERS + NUMBERS], device)

    def field(index):
        return integers[bounds[index]:bounds[index + 1]]

    tokens = _mask(field(0), int(widths[0]))
    groups = []
    for buckets in GROUPED:
        present = [bucket for bucket in buckets if counts[:, GROUPS + bucket].any()]
        if not present:
            continue
        length = 1 << present[-1]
        first = np.cumsum([0] + [int(counts[:, GROUPS + bucket].sum()) for bucket in present])

        def together(name):
            return torch.cat([field(_group(bucket, name)) for bucket in present])

        # The fields of the real nodes, bucket after bucket; the padded layout they
        # stand in is told by `nodes`.
        mask = _mask(together("rows"), int(max(widths[1 + bucket] for bucket in present)))

        def placed(name):
            # A place among the padded nodes: program and node, both move.
            return torch.cat([
                (value // (1 << bucket) + start) * length + value % (1 << bucket)
                for bucket, start in zip(present, first)
                for value in [field(_group(bucket, name))]
            ])
        groups.append(ProgramGroup(
            Fields(
                _padded(mask, together("ids")),
                _padded(mask, together("kinds")),
                _padded(mask, torch.cat([
                    numbers[number_bounds[1 + bucket]:number_bounds[2 + bucket]] for bucket in present
                ]), shape=(4,)),
                mask,
            ),
            length,
            together("lengths"),
            torch.stack([
                torch.cat([field(_group(bucket, "edge_program")) + start
                           for bucket, start in zip(present, first)]),
                *(together(name) for name in GROUP[5:8]),
            ], dim=1),
            placed("binding_targets"),
            together("binding_sources"),
            together("binding_roles"),
            together("slots"),
            placed("nodes"),
        ))
    # Every message padded its actions to a width of its own.
    slots = _mask(
        torch.from_numpy(np.repeat(headers[:, -2], counts[:, OBSERVATIONS])).to(device),
        int(widths[-2]))
    batch = Batch(
        Fields(
            _padded(tokens, field(1)),
            _padded(tokens, field(2)),
            _padded(tokens, numbers[:number_bounds[1]], shape=(4,)),
            tokens,
        ),
        field(GLOBAL.index("lengths")),
        int(widths[-1]),
        field(GLOBAL.index("reference_targets")),
        field(GLOBAL.index("reference_sources")),
        field(GLOBAL.index("reference_roles")),
        groups,
        int(counts[:, PROGRAMS].sum()),
        field(GLOBAL.index("token_programs")),
        torch.stack([field(GLOBAL.index(name)) for name in GLOBAL[8:12]], dim=1),
        _padded(slots, field(GLOBAL.index("actions"))),
        [],
    )
    legal = _padded(slots, field(GLOBAL.index("legal")).bool())
    return Decisions(batch, legal), sorted({name for message in messages for name in message[3]})
