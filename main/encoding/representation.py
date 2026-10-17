"""Typed public fields, local programs and reference-only entity handles: a decision
frame as an `Observation` (`observation`).

Packing indices never become features. Piles remain full unordered multisets.
Symbols are a frozen, checkpointed vocabulary (`vocabulary.py`); unknown content is
explicit. The act map is compressed (`maps.py`): only the current node and its
frontier stay entities, and each frontier node carries the map reachable from it as a
local program. The field whitelists and the cleaning are `frames.py`, the field
encoding `fields.py`, the local programs `programs.py`."""

from dataclasses import dataclass, field

from main.env.protocol import ProtocolError, fingerprint
from main.encoding.fields import Field, entity_fields, fields_of
from main.encoding.frames import clean_entity, clean_public
from main.encoding.maps import compress_map
from main.encoding.programs import Effect, entity_program


ENCODING_VERSION = "public-encoding-v3"


@dataclass
class Observation:
    tokens: list[list[Field]]
    effects: list[Effect]
    refs: dict[str, int]
    action_indices: list[int]
    slot_refs: list[str]
    edges: list[tuple[int, int, str]]
    context: dict
    # (entities, edges, map keys): the content the digest is taken over.
    _source: tuple = field(default=None, repr=False, compare=False)
    _digest: str | None = field(default=None, repr=False, compare=False)

    @property
    def digest(self):
        """Content key of the observation, for the session cache of an unchanged
        frame; computed when first asked, as training never asks."""
        if self._digest is None:
            entities, edges, map_keys = self._source
            # Dynamic prefix/mask never contaminates cached Transformer input.
            self._digest = fingerprint({"entities": entities, "edges": edges, "map": map_keys})
        return self._digest


def forget():
    """Drop what parsing has cached of entities, programs, maps and rule text.

    The caches pay within a run, whose frames repeat its entities and its maps. A
    process that parses one run after another needs nothing of the run before; kept,
    the caches grow to their limits in every such process."""
    from main.encoding import fields, maps, programs, rules

    for module in (fields, programs, maps, rules):
        module.forget()


def _bind_displayed_variables(entities, by_ref):
    """Render an option's text with the public numbers linked to that option."""
    for item in entities:
        if item.get("entity_type") != "displayed_variable":
            continue
        owner = by_ref.get(item.get("owner_ref"))
        if owner is None or owner.get("known") is False or owner.get("revealed") is False:
            continue
        if isinstance(item.get("names"), str) and isinstance(item.get("content_id"), str):
            owner.setdefault("named", {})[item["content_id"]] = item["names"]
            continue
        fields = {f.name: f for f in fields_of(item)}
        name, amount = fields.get("content_id"), fields.get("amount")
        if name is None or not name.number[1] or amount is None:
            continue
        value, known, applicable, numeric = amount.number
        owner.setdefault("stats", {})[item["content_id"]] = {
            "value": value,
            "known": bool(known and numeric),
            "applicable": bool(applicable and item.get("applicable", True)),
        }


def observation(frame):
    public = clean_public(frame["public"])
    from main.env.public_history import HISTORY_KINDS, unknown_memory

    if not any(e.get("entity_type") in HISTORY_KINDS for e in public["memory"]):
        # Old BC summaries/recordings have no public prefix. Keep those samples,
        # but explicitly mask history instead of assigning a new-run prior.
        public["memory"].extend(unknown_memory(public["entities"]))
    context = public.get("selection_context") or {}
    # Selection progress lives in the input: the global token carries
    # bounds and counts, selected entities are marked (with their order when it
    # matters). No decoder state survives between decisions.
    progress = {k: v for k, v in context.items() if k != "selected_refs"}
    entities = [dict(progress, entity_type="global", phase=public["phase"])]
    entities += public["entities"] + public["memory"]
    selected = context.get("selected_refs", [])
    order_known = (context.get("known_masks") or {}).get("order_matters", True)
    ordered = bool(context.get("order_matters")) and order_known
    by_ref = {item.get("ref"): item for item in entities if item.get("ref")}
    _bind_displayed_variables(entities, by_ref)
    for index, ref in enumerate(selected):
        if ref not in by_ref:
            raise ProtocolError("Selected reference is not a public entity")
        by_ref[ref]["selected"] = True
        if ordered:
            by_ref[ref]["selected_order"] = index + 1
    if public.get("event_context"):
        entities.append(dict(public["event_context"], entity_type="event_context"))
    # Bundle contents stay separate entities, linked to their containing bundle.
    for item in list(entities):
        for i, card in enumerate(item.pop("cards", [])):
            ref = f"bundle-child:{len(entities)}:{i}"
            entities.append(dict(card, ref=ref, owner_ref=item["ref"]))
    bank = public.get("decoder_bank") or [
        clean_entity(c, action=True) for c in frame["legal"]["candidates"]
    ]
    entities += [dict(c, entity_type="action") for c in bank]
    effects = [entity_program(x) for x in entities]
    entities, effects, relations, map_keys = compress_map(
        entities, effects, public["relations"], selected
    )
    action_start = len(entities) - len(bank)
    refs = {}
    for i, item in enumerate(entities):
        ref = item.get("ref")
        if ref:
            if ref in refs:
                raise ProtocolError("Duplicate public entity reference")
            refs[ref] = i
    edges = []
    for i, item in enumerate(entities):
        for role in ("source", "target", "option", "owner"):
            bindings = item.get(role + "_refs", [])
            if item.get(role + "_ref"):
                bindings = bindings + [item[role + "_ref"]]
            for ref in bindings:
                if ref not in refs:
                    raise ProtocolError(f"Unresolved public {role} reference")
                edges += [(i, refs[ref], role), (refs[ref], i, "reverse_" + role)]
    for edge in relations:
        if edge.get("source") not in refs or edge.get("target") not in refs:
            raise ProtocolError("Unresolved relation")
        source, target = refs[edge["source"]], refs[edge["target"]]
        edges += [
            (source, target, edge["role"]),
            (target, source, "reverse_" + edge["role"]),
        ]
    return Observation(
        [entity_fields(x) for x in entities],
        effects,
        refs,
        list(range(action_start, len(entities))),
        [c["decoder_slot_ref"] for c in bank],
        edges,
        context,
        (entities, edges, map_keys),
    )
