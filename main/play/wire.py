"""Packed decisions as three flat arrays, to cross from sampling processes to the model.

A pack as the network reads it is some eighty padded tensors. The process that runs
the network is bound by its own interpreter: what it does for every tensor of every
message it receives is time between two passes. Here a pack travels without its
padding as a header, one array of integers and one of numbers, and the packs of any
number of processes are joined on the device by a fixed number of array operations.

Most of what a decision shows the encoder, a decision before it of the same run showed
already: the deck, the relics, the map. The collecting process keeps the entity
encoder's output for every token and program of a round in tables on the device
(`EncoderTables`), each sampling process owning a range of their rows. A sampling
process remembers what it has sent (`Sent`): a token or program travels with its
fields the first time and as its row after that. `unpack` gives a `TableBatch` whose
fields are those of the new tokens and programs, and whose rows say where every one is.
"""

import numpy as np
import torch

from main.encoding.batch import Fields, ProgramGroup, Rows, TableBatch, program_items
from main.encoding.fields import row_digest
from main.network.model import EncoderTables
from main.network.policy import Decisions

# Program groups of 1, 2, 4, ... nodes.
BUCKETS = 12
# The program groups `unpack` makes of these buckets: up to 64 nodes, and longer. The
# new programs of a pass are few: padding them to the longest of their group costs
# little, and every group fewer is a round of the encoder's loop fewer.
GROUPED = (tuple(range(7)), tuple(range(7, BUCKETS)))
# Integer fields of a pack, then those of each program group. A field of padded
# rows travels as the length of every row and the cells its mask keeps. "rows", "ids"
# and "kinds" are the fields of the pack's new tokens.
GLOBAL = ("rows", "ids", "kinds", "lengths", "reference_targets", "reference_sources",
          "reference_roles", "token_programs", "relation_observation", "relation_query",
          "relation_key", "relation_role", "actions", "legal", "explore", "rank",
          "token_rows", "new_token_rows")
GROUP = ("rows", "ids", "kinds", "lengths", "edge_program", "edge_source", "edge_target",
         "edge_relation", "binding_targets", "binding_sources", "binding_roles", "slots", "nodes")
INTEGERS = len(GLOBAL) + BUCKETS * len(GROUP)
NUMBERS = 1 + BUCKETS  # the numbers of the new tokens, then of each group's nodes
# What a pack counts: the rows that the indices of later packs move by.
TOKENS, OBSERVATIONS, GROUPS = 0, 1, 2
COUNTS = GROUPS + BUCKETS
# Header: the length of every field, the counts, then the widths the padding had:
# token fields, each group's node fields, actions, and the padded token length.
WIDTHS = 1 + BUCKETS + 2
HEADER = INTEGERS + NUMBERS + COUNTS + WIDTHS
# Rows of the encoder tables each sampling process owns: tokens and programs. A run
# shows some two thousand distinct tokens and four hundred distinct programs.
TOKEN_ROWS, PROGRAM_ROWS = 16384, 4096


def _group(bucket, name):
    return len(GLOBAL) + bucket * len(GROUP) + GROUP.index(name)


def _moves():
    """For every integer field, the count its values move by and how many times. Table
    rows (token rows, programs, a binding's entity) name the same row in every pack."""
    count, times = np.full(INTEGERS, -1), np.ones(INTEGERS, dtype=np.int64)
    for name, moved in (("reference_targets", TOKENS), ("reference_sources", TOKENS),
                        ("relation_observation", OBSERVATIONS)):
        count[GLOBAL.index(name)] = moved
    for bucket in range(BUCKETS):
        count[_group(bucket, "edge_program")] = GROUPS + bucket
        # A binding names a node by its place among the padded nodes of the group, as
        # does the row of each node.
        for name in ("binding_targets", "nodes"):
            count[_group(bucket, name)] = GROUPS + bucket
            times[_group(bucket, name)] = 1 << bucket
    return count, times


MOVED_BY, MOVED_TIMES = _moves()
EMPTY = np.zeros(0, dtype=np.int32)


class _Rows:
    """The rows of one table a sampling process owns, by the digest of what they hold.
    They are filled half by half: when the half in use is full the process forgets it
    and fills the other, so that rows named by packs already sent keep what they hold
    while the collecting process reads them."""

    def __init__(self, first, count):
        self.first, self.half = first, count // 2
        self.side, self.rows = 0, {}

    def place(self, digests):
        """The row of every digest and whether its content must be sent."""
        if len(digests) > self.half:
            raise ValueError("A pack holds more than half a process's table rows")
        if len(self.rows) + len(digests) > self.half:
            self.side, self.rows = 1 - self.side, {}
        start = self.first + self.side * self.half
        rows, new = np.empty(len(digests), dtype=np.int64), np.zeros(len(digests), dtype=bool)
        for index, digest in enumerate(digests):
            row = self.rows.get(digest)
            if row is None:
                row = self.rows[digest] = start + len(self.rows)
                new[index] = True
            rows[index] = row
        return rows, new


class Sent:
    """What sampling process `process` has sent this round: the table row of every token
    and program content (`main.encoding.fields.row_digest`, `Effect.digest`). A round
    starts with nothing sent: its tables are new."""

    def __init__(self, process):
        self.tokens = _Rows(process * TOKEN_ROWS, TOKEN_ROWS)
        self.programs = _Rows(process * PROGRAM_ROWS, PROGRAM_ROWS)


def tables(model, processes):
    """The encoder tables of a round sampled by `processes` processes."""
    return EncoderTables(model, processes * TOKEN_ROWS, processes * PROGRAM_ROWS)


def pack(decisions, observations, sent):
    """(header, integers, numbers, unregistered names) of the decisions of `observations`
    (`prepare_decisions`), with the fields of only what `sent` has not sent. Fields
    left as rows (`prepare_decisions(tensors=False)`) are made tensors here, those sent
    alone."""
    batch = decisions.batch
    unregistered = set(batch.unregistered)
    deferred = isinstance(batch.tokens, Rows)
    if deferred:
        vocabulary = batch.tokens.vocabulary
        before = set(vocabulary.unregistered())
    integers, numbers = [EMPTY] * INTEGERS, [np.zeros(0, dtype=np.float32)] * NUMBERS
    counts, widths = np.zeros(COUNTS, dtype=np.int64), np.zeros(WIDTHS, dtype=np.int64)
    if decisions.masks.shape != batch.actions.shape or decisions.explore.shape != batch.actions.shape:
        raise ValueError("Legal and exploration masks must cover the action slots")

    def rows(first, number, fields, keep):
        if isinstance(fields, Rows):
            fields, keep = fields.fields(keep), slice(None)
        mask = fields.mask.numpy()[keep]
        integers[first] = mask.sum(1)
        integers[first + 1] = fields.ids.numpy()[keep][mask]
        integers[first + 2] = fields.kinds.numpy()[keep][mask]
        numbers[number] = fields.numbers.numpy()[keep][mask].ravel()
        return mask.shape[1]

    digests = [row_digest(row) for obs in observations for row in obs.tokens]
    if len(digests) != int(batch.lengths.sum()):
        raise ValueError("The observations are not those of the decisions")
    token_rows, new_tokens = sent.tokens.place(digests)
    widths[0] = rows(0, 0, batch.tokens, new_tokens)
    # A bound program is told apart by the content of the entities it binds.
    items, _ = program_items(observations)
    program_rows, new_programs = sent.programs.place([
        effect.digest if not bindings else
        hash((effect.digest, tuple((node, digests[source], role) for node, source, role in bindings)))
        for effect, bindings in items])

    relations = batch.relations.numpy()
    for name, values in (
        ("lengths", batch.lengths), ("reference_targets", batch.reference_targets),
        ("reference_sources", batch.reference_sources), ("reference_roles", batch.reference_roles),
        ("actions", batch.actions), ("legal", decisions.masks), ("explore", decisions.explore),
        ("rank", decisions.rank),
    ):
        integers[GLOBAL.index(name)] = values.numpy().ravel()
    for column, name in enumerate(GLOBAL[8:12]):
        integers[GLOBAL.index(name)] = relations[:, column]
    integers[GLOBAL.index("token_programs")] = program_rows[batch.token_programs.numpy()]
    integers[GLOBAL.index("token_rows")] = token_rows
    integers[GLOBAL.index("new_token_rows")] = token_rows[new_tokens]
    counts[[TOKENS, OBSERVATIONS]] = len(digests), len(batch.lengths)
    for group in batch.programs:
        bucket = group.length.bit_length() - 1
        if group.length != 1 << bucket or bucket >= BUCKETS:
            raise ValueError("Program group length is not a supported power of two")
        slots = group.slots.numpy()
        keep = new_programs[slots]
        if not keep.any():
            continue
        # The group's new programs, numbered again among themselves.
        lengths = group.lengths.numpy()
        kept_nodes = keep[np.repeat(np.arange(len(slots)), lengths)]
        widths[1 + bucket] = rows(_group(bucket, "rows"), 1 + bucket, group.fields, kept_nodes)
        renumber = np.cumsum(keep) - 1
        edges = group.edges.numpy()
        edges = edges[keep[edges[:, 0]]]
        for column, name in enumerate(GROUP[5:8]):
            integers[_group(bucket, name)] = edges[:, column + 1]
        integers[_group(bucket, "edge_program")] = renumber[edges[:, 0]]

        def placed(values):
            # A place among the group's padded nodes: program and node.
            return renumber[values // group.length] * group.length + values % group.length

        nodes = group.nodes.numpy()[kept_nodes]
        integers[_group(bucket, "nodes")] = placed(nodes)
        targets = group.binding_targets.numpy()
        bound = keep[targets // group.length]
        integers[_group(bucket, "binding_targets")] = placed(targets[bound])
        integers[_group(bucket, "binding_sources")] = token_rows[group.binding_sources.numpy()[bound]]
        integers[_group(bucket, "binding_roles")] = group.binding_roles.numpy()[bound]
        integers[_group(bucket, "lengths")] = lengths[keep]
        integers[_group(bucket, "slots")] = program_rows[slots[keep]]
        counts[GROUPS + bucket] = int(keep.sum())
    widths[-2:] = batch.actions.shape[1], batch.pad
    header = np.concatenate((
        [len(field) for field in integers], [len(field) for field in numbers], counts, widths))
    if deferred:
        # Names met in the fields made here, which `collate` did not see.
        unregistered |= set(vocabulary.unregistered()) - before
    return (header, np.concatenate(integers, dtype=np.int32, casting="unsafe"),
            np.concatenate(numbers), sorted(unregistered))


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


def unpack(messages, tables):
    """The decisions of `pack`ed messages on the device of `tables` (`tables`), in their
    order, and the names they met which the vocabulary never registered."""
    device = tables.tokens.device
    headers = np.stack([message[0] for message in messages])
    counts = headers[:, INTEGERS + NUMBERS:][:, :COUNTS]
    widths = headers[:, -WIDTHS:].max(0)
    starts = np.cumsum(counts, 0) - counts
    moves = np.where(MOVED_BY >= 0, starts[:, np.maximum(MOVED_BY, 0)] * MOVED_TIMES, 0)
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
    batch = TableBatch(
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
        0,
        field(GLOBAL.index("token_programs")),
        torch.stack([field(GLOBAL.index(name)) for name in GLOBAL[8:12]], dim=1),
        _padded(slots, field(GLOBAL.index("actions"))),
        [],
        field(GLOBAL.index("token_rows")),
        field(GLOBAL.index("new_token_rows")),
        tables,
    )
    legal = _padded(slots, field(GLOBAL.index("legal")).bool())
    explore = _padded(slots, field(GLOBAL.index("explore")).bool())
    return Decisions(batch, legal, explore, field(GLOBAL.index("rank"))), sorted(
        {name for message in messages for name in message[3]})
