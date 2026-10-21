"""The fields of an entity as the network reads them: a symbol under a field name,
numbers with their known/applicable flags (`fields_of`, `scalar_field`), the rows of an
entity cached by content (`entity_fields`) with a digest of that content (`row_digest`),
and rows packed into tensors (`field_tensors`)."""

import math
from dataclasses import dataclass

from main.env.protocol import ProtocolError
from main.encoding.frames import REFERENCE_FIELDS


@dataclass
class Field:
    name: str
    symbol: str
    # (value, known, applicable, is_numeric); the model derives its own scales.
    number: tuple[float, float, float, float]


def fields_of(value, prefix=""):
    fields = []
    known = value.get("known_masks", {}) if isinstance(value, dict) else {}
    applicable = value.get("applicable_masks", {}) if isinstance(value, dict) else {}
    if isinstance(value, dict):
        for key in sorted(value):
            if key in REFERENCE_FIELDS | {
                "decoder_slot_ref",
                "candidate_ref",
                "semantic_program",
                "program",
                "cards",
                "known_masks",
                "applicable_masks",
                # Names an option displays, bound for its rule text; the variable itself is the token.
                "named",
            }:
                continue
            item = value[key]
            name = f"{prefix}.{key}" if prefix else key
            if known.get(key) is False:
                item = {"known": False, "applicable": applicable.get(key, True)}
            if applicable.get(key) is False:
                item = {"known": False, "applicable": False}
            if isinstance(item, dict) and (
                "value" in item
                or item.get("known") is False
                or item.get("applicable") is False
            ):
                fields.append(
                    scalar_field(
                        name,
                        item.get("value"),
                        item.get("known", True),
                        item.get("applicable", True),
                    )
                )
            elif isinstance(item, (dict, list)):
                fields.extend(fields_of(item, name))
            else:
                fields.append(scalar_field(name, item))
        for key in sorted((known.keys() | applicable.keys()) - value.keys()):
            # Unknown references must differ from roles which do not apply.
            name = f"{prefix}.{key}" if prefix else key
            fields.append(
                scalar_field(
                    name,
                    True if known.get(key) else None,
                    known.get(key, False),
                    applicable.get(key, True),
                )
            )
    elif isinstance(value, list):
        for item in (
            value
        ):  # Unordered repeated fields retain multiplicity, no index embedding.
            fields.extend(
                fields_of(item, prefix)
                if isinstance(item, (dict, list))
                else [scalar_field(prefix, item)]
            )
    return fields or [scalar_field(prefix or "empty", None)]


def scalar_field(name, value, known=True, applicable=True):
    known = bool(known and value is not None and applicable)
    if not known:
        return Field(name, "<unknown>", (0, 0, float(applicable), 0))
    if name == "content_id" and isinstance(value, str):
        from main.encoding.rules import canonical_content_id

        value = canonical_content_id(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ProtocolError("Non-finite public number")
        return Field(name, name + "=<number>", (value, 1, 1, 1))
    return Field(name, f"{name}={value}", (0, 1, 1, 0))


class FieldRow(list):
    """The fields of one entity (`entity_fields`), with the digest of their content kept
    beside them (`row_digest`): a row is shared by every frame of a run that shows the
    entity, so its digest is taken once."""

    __slots__ = ("digest",)


def content_digest(rows):
    """A 64-bit digest of what the network reads of rows of fields: their names, symbols
    and numbers. Python's own hash, so digests compare within one process (and the
    processes it forks) only."""
    return hash(tuple(tuple((f.name, f.symbol, tuple(f.number)) for f in row) for row in rows))


def row_digest(row):
    """`content_digest` of one row; taken once for a `FieldRow`."""
    digest = getattr(row, "digest", None)
    if digest is None:
        digest = content_digest([row])
        if isinstance(row, FieldRow):
            row.digest = digest
    return digest


_FIELD_ROWS = {}


_FIELD_ROWS_LIMIT = 16384


def entity_fields(entity):
    """`fields_of` an entity, shared between equal entities: the cards, relics and
    powers of a run repeat across its frames and across runs. Rows are read only.
    Keyed by the entity as Python writes it, keys in their order: `fields_of` sorts
    them, so an equal entity written in another order only misses the cache."""
    key = repr(entity)
    rows = _FIELD_ROWS.get(key)
    if rows is None:
        rows = FieldRow(fields_of(entity))
        if len(_FIELD_ROWS) >= _FIELD_ROWS_LIMIT:
            _FIELD_ROWS.pop(next(iter(_FIELD_ROWS)))
        _FIELD_ROWS[key] = rows
    return rows


def field_tensors(rows, vocabulary, width=None):
    """Pack field rows into (symbol ids, field-name ids, numbers, mask) on the CPU,
    padded to `width` (the longest row's, unless told)."""
    import torch

    width = max(map(len, rows), default=1) if width is None else width
    raw_ids, raw_kinds, raw_nums, raw_mask = [], [], [], []
    symbol, name = vocabulary.encode, vocabulary.fields.encode
    for row in rows:
        pad = width - len(row)
        raw_ids.append([symbol(f.symbol) for f in row] + [0] * pad)
        raw_kinds.append([name(f.name) for f in row] + [0] * pad)
        raw_nums.append([f.number for f in row] + [(0,) * 4] * pad)
        raw_mask.append([True] * len(row) + [False] * pad)
    n = len(rows)
    return (
        torch.tensor(raw_ids, dtype=torch.long).reshape(n, width),
        torch.tensor(raw_kinds, dtype=torch.long).reshape(n, width),
        torch.tensor(raw_nums, dtype=torch.float32).reshape(n, width, 4),
        torch.tensor(raw_mask, dtype=torch.bool).reshape(n, width),
    )


def forget():
    """Drop the cached field rows."""
    _FIELD_ROWS.clear()
