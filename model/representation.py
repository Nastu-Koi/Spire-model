"""Typed public fields, local programs and reference-only entity handles.

Packing indices never become features. Piles remain full unordered multisets.
Symbols are a frozen, checkpointed vocabulary; unknown content is explicit.
The act map is compressed: only the current node and its frontier stay
entities, and each frontier node carries the map reachable from it as a local
program.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

from .protocol import ProtocolError, fingerprint

ENCODING_VERSION = "public-encoding-v3"

ENTITY_FIELDS = {
    "entity_type",
    "content_id",
    "zone",
    "character",
    "phase",
    "act",
    "floor",
    "hp",
    "max_hp",
    "block",
    "gold",
    "energy",
    "max_energy",
    "stars",
    "turn",
    "round",
    "lost_hp_this_turn",
    "cost",
    "base_cost",
    "star_cost",
    "base_star_cost",
    "x_cost",
    "upgraded",
    "upgrade_level",
    "card_type",
    "rarity",
    "enchantment",
    "affliction",
    "stacks",
    "count",
    "capacity",
    "slot",
    "position",
    "known",
    "applicable",
    "exhausted",
    "used",
    "remaining",
    "charges",
    "target_type",
    "intent",
    "damage",
    "hits",
    "total_damage",
    "current",
    "visited",
    "selectable",
    "complete",
    "revealed",
    "operation",
    "destination",
    "scope",
    "effect_coverage",
    "quantity",
    "max_slots",
    "order_known",
    "valid_until",
    "event_type",
    "stage",
    "mode",
    "min_total",
    "max_total",
    "selected_count",
    "remaining_required",
    "remaining_steps",
    "can_finish",
    "can_skip",
    "can_cancel",
    "order_matters",
    "repetition_allowed",
}
ENTITY_FIELDS.update(
    [
        "x",
        "y",
        "col",
        "row",
        "price",
        "sold_out",
        "rarity",
        "enchantment_amount",
        "affliction_amount",
        "visibility",
        "remaining_uses",
        "step_index",
        "elapsed_turns",
        "trigger_period",
        "remaining_turns",
        "incoming_damage",
        "nominal_block_margin",
        "energy_margin",
        "gold_margin",
        "stars_margin",
    ]
)
ENTITY_FIELDS.update(
    ["enabled", "base_star_cost", "current_star_cost", "source", "amount"]
)
ENTITY_FIELDS.update(["rider_effect", "retain", "stars_x", "exhaust_on_next_play"])
ENTITY_FIELDS.update(["counter", "free_travel", "ascension", "encounter"])
ENTITY_FIELDS.add("probability")
# The content a displayed variable shows by name (an option's card, relic, potion, enchantment).
ENTITY_FIELDS.add("names")
CONTAINERS = {
    "stats",
    "keywords",
    "modifiers",
    "powers",
    "intents",
    "public_previews",
    "public_costs",
    "known_masks",
    "applicable_masks",
    "after_upgrade",
    "fields",
    "properties",
    "numeric",
}
STAT_FIELDS = {
    "damage",
    "block",
    "draw",
    "cards",
    "energy",
    "stars",
    "hp",
    "maxhp",
    "heal",
    "strength",
    "dexterity",
    "vulnerable",
    "weak",
    "frail",
    "poison",
    "exhaust",
    "discard",
    "summon",
    "repeat",
    "times",
    "amount",
    "count",
    "turns",
    "cost",
    "max_select",
    "min_select",
}
STAT_FIELDS.update({"passive", "evoke"})
# Native card DynamicVars use these public display names, rather than the
# shorter aliases above. Preserve health costs and status magnitudes too.
STAT_FIELDS.update(
    {
        "hploss",
        "strengthpower",
        "dexteritypower",
        "weakpower",
        "vulnerablepower",
        "calculationbase",
    }
)
# Mad Science declares all of these public parameters on every variant. The
# separate rider_effect and card_type identify which ones the card uses.
STAT_FIELDS.update(
    [
        "sappingweak",
        "sappingvulnerable",
        "violencehits",
        "chokingdamage",
        "energizedenergy",
        "wisdomcards",
        "expertisestrength",
        "expertisedexterity",
        "curiousreduction",
    ]
)
REFERENCE_FIELDS = {
    "ref",
    "source_ref",
    "owner_ref",
    "source_refs",
    "target_refs",
    "option_refs",
    "selected_refs",
}
ACTION_FIELDS = (
    ENTITY_FIELDS
    | CONTAINERS
    | REFERENCE_FIELDS
    | {"verb", "decoder_slot_ref", "candidate_ref", "semantic_program", "program"}
)
PROGRAM_FIELDS = {
    "kind",
    "op",
    "args",
    "bindings",
    "role",
    "timing",
    "reason",
    "children",
    "steps",
    "body",
    "condition",
    "then",
    "else",
    "cases",
    "repeat",
    "probability",
    "count",
    "amount",
    "value",
    "known",
    "applicable",
    "unit",
    "content_id",
    "object_kind",
    "field",
    "read_at",
    "domain",
    "distribution",
    "alternatives",
    "weights",
    "min",
    "max",
    "ordered",
    "order_matters",
    "replacement",
    "options",
    "choice",
    "trigger",
    "duration",
    "left",
    "right",
    "operand",
    "recipient",
    "subject",
    "actor",
    "source",
    "target",
    "destination",
    "cost",
    "consequence",
    "selector",
    "scope",
    "name",
    "entity",
    "ref",
    "program",
    "on_use_program",
    "definition_id",
}
PROGRAM_FIELDS.update(
    [
        "acquisition_method",
        "bind",
        "board",
        "bound_roles",
        "cancelable",
        "cause",
        "center",
        "clip_to_board",
        "collection",
        "completion",
        "completion_rules",
        "context",
        "context_ref",
        "continuation",
        "controller",
        "coverage",
        "damage_flags",
        "decision_kind",
        "definition_ref",
        "delta",
        "enchantment",
        "evaluate_at",
        "event",
        "filters",
        "generator",
        "groups",
        "identity",
        "insufficient_rule",
        "iteration",
        "limit",
        "mode",
        "operation",
        "owner",
        "per_event",
        "predicate",
        "prospective_effect",
        "public_generator_rule",
        "public_preview",
        "public_rule",
        "radius",
        "rarity",
        "requested_count",
        "resource",
        "result_identity",
        "reveal_at",
        "reveal_semantics_ref",
        "selected_member",
        "selected_refs",
        "selection_effect_ref",
        "shape",
        "stacks",
        "target_type",
        "times",
        "usage",
        "uses_cost",
        "refs",
    ]
)
BINDING_ROLES = {
    "actor",
    "recipient",
    "subject",
    "source",
    "target",
    "owner",
    "destination",
    "option",
    "payer",
    "beneficiary",
    "card",
    "potion",
    "relic",
    "enemy",
    "player",
    "self",
    "other",
}


def _program(value):
    if isinstance(value, list):
        return [_program(x) for x in value]
    if not isinstance(value, dict):
        return value
    result = {
        k: _program(v)
        for k, v in value.items()
        if k in PROGRAM_FIELDS or k in BINDING_ROLES
    }
    if result.get("known") is False:
        result = {
            k: v
            for k, v in result.items()
            if k in {"kind", "known", "applicable", "unit"}
        }
    return result


def clean_entity(value, *, action=False):
    from . import rules

    allowed = (
        ACTION_FIELDS
        if action
        else ENTITY_FIELDS
        | CONTAINERS
        | REFERENCE_FIELDS
        | {"semantic_program", "program", "cards"}
    )
    result = {}
    for key, item in value.items():
        if key not in allowed:
            continue
        if key in {"semantic_program", "program"}:
            result[key] = _program(item)
        elif key == "cards":
            result[key] = [clean_entity(x) for x in item]
        elif key == "stats":
            # A value the content's own description displays is public with it.
            displayed = rules.variables(value.get("content_id"))
            stats = {
                k: _public_value(v)
                for k, v in item.items()
                if k.lower().replace("_", "") in STAT_FIELDS or k in displayed
            }
            if stats:
                result[key] = stats
        elif key in CONTAINERS and isinstance(item, dict):
            result[key] = {
                k: _public_value(v)
                for k, v in item.items()
                if k in ENTITY_FIELDS or k.lower() in STAT_FIELDS or k in BINDING_ROLES
            }
        elif key in CONTAINERS and isinstance(item, list):
            result[key] = [clean_entity(x) if isinstance(x, dict) else x for x in item]
        else:
            result[key] = _public_value(item)
    if result.get("zone") in {"draw_pile", "discard_pile", "exhaust_pile"}:
        # Publicly revealed order has a separate relation, never a raw array index.
        for key in ("position", "slot", "row", "col"):
            result.pop(key, None)
    if result.get("revealed") is False or result.get("known") is False:
        public_shape = {
            "ref",
            "entity_type",
            "zone",
            "x",
            "y",
            "row",
            "col",
            "position",
            "count",
            "capacity",
            "known",
            "applicable",
            "revealed",
        }
        if result.get("entity_type") == "previous_intent":
            public_shape.add("owner_ref")
        result = {k: v for k, v in result.items() if k in public_shape}
    return result


def _public_value(item):
    if isinstance(item, dict):
        # Numeric fields may carry known/applicable masks alongside their value.
        return {
            k: _public_value(v)
            for k, v in item.items()
            if k in ENTITY_FIELDS | {"value", "unit"}
        }
    if isinstance(item, list):
        return [_public_value(x) for x in item]
    return item


def clean_public(public):
    result = {
        "phase": public.get("phase", "unknown"),
        "entities": [clean_entity(x) for x in public.get("entities", [])],
        "memory": [clean_entity(x) for x in public.get("memory", [])],
        "relations": [
            {k: v for k, v in x.items() if k in {"source", "target", "role"}}
            for x in public.get("relations", [])
        ],
    }
    for key in ("selection_context", "event_context"):
        if public.get(key):
            result[key] = clean_entity(public[key])
    if public.get("decoder_bank") is not None:
        result["decoder_bank"] = [
            clean_entity(x, action=True) for x in public["decoder_bank"]
        ]
    return result


class Table:
    """Indices for a frozen set of names; row 0 is padding.

    Registered names receive exact, collision-free rows. A name which was not
    registered when the table was frozen hashes into the spare rows, so it
    stays distinguishable from the registered ones and is recorded in
    ``unregistered`` instead of silently sharing a row with one of them.
    """

    def __init__(self, names, capacity):
        self.names = tuple(sorted(set(names)))
        self.capacity = capacity
        self.spare = capacity - 1 - len(self.names)
        if self.spare < 1:
            raise ValueError(
                "Name table exceeds configured capacity; increase capacity explicitly"
            )
        self.lookup = {name: i + 1 for i, name in enumerate(self.names)}
        self.unregistered = {}

    def encode(self, name):
        index = self.lookup.get(name)
        if index is None:
            index = self.unregistered.get(name)
            if index is None:
                digest = hashlib.blake2s(name.encode(), digest_size=4).digest()
                index = 1 + len(self.names) + int.from_bytes(digest, "big") % self.spare
                self.unregistered[name] = index
        return index


REFERENCE_ROLES = ("source", "target", "option", "owner")
# Roles of the relations the engine publishes between entities.
ENGINE_RELATIONS = ("map_edge", "offers", "known_draw_before")
MAP_RELATIONS = (
    "self",
    "forward_direct",
    "reverse_direct",
    "forward_indirect",
    "reverse_indirect",
    "free_direct",
    "free_reverse",
    "unreachable",
)
PROGRAM_RELATIONS = (
    "program_child",
    "program_parent",
    "binds_variable",
    "variable_from",
    "same_variable",
)


class Vocabulary:
    """Frozen, checkpointed tables: field symbols, field names and relation roles."""

    def __init__(self, symbols=None, capacity=16384, *, fields=(), relations=(),
                 field_capacity=512, relation_capacity=128, ordered=False):
        self.symbols = tuple(symbols) if ordered else tuple(
            ["<pad>", "<unknown>"] + sorted(set(symbols or []) - {"<pad>", "<unknown>"}))
        if self.symbols[:2] != ("<pad>", "<unknown>") or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Invalid frozen symbol table")
        if len(self.symbols) > capacity:
            raise ValueError(
                "Vocabulary exceeds configured capacity; increase capacity explicitly"
            )
        self.capacity = capacity
        self.lookup = {s: i for i, s in enumerate(self.symbols)}
        self.fields = Table(fields, field_capacity)
        self.relations = Table(relations, relation_capacity)
        self._reported = set()
        self._unknown_symbols = set()
        self._digest = fingerprint(
            [self.symbols, self.fields.names, field_capacity, self.relations.names, relation_capacity]
        )

    @classmethod
    def from_frames(cls, frames, config):
        symbols, fields, relations = names_from_frames(frames)
        return cls(symbols, config.vocabulary_size, fields=fields, relations=relations,
                   field_capacity=config.field_size, relation_capacity=config.relation_size)

    @classmethod
    def from_state(cls, state, config):
        return cls(state["symbols"], config.vocabulary_size, fields=state["fields"],
                   relations=state["relations"], field_capacity=config.field_size,
                   relation_capacity=config.relation_size, ordered=True)

    def expanded(self):
        """Add current public schema symbols in spare rows, preserving every old ID."""
        required, _, _ = names_from_frames(())
        return Vocabulary(list(self.symbols) + sorted(required - self.lookup.keys()), self.capacity,
                          fields=self.fields.names, relations=self.relations.names,
                          field_capacity=self.fields.capacity, relation_capacity=self.relations.capacity,
                          ordered=True)

    def state(self):
        return {"symbols": list(self.symbols), "fields": list(self.fields.names),
                "relations": list(self.relations.names)}

    def encode(self, symbol):
        result = self.lookup.get(symbol)
        if result is None:
            self._unknown_symbols.add("symbol:" + symbol)
            return 1
        return result

    def unregistered(self):
        """Missing symbols (prefixed symbol:), field names and relation roles."""
        return sorted(
            self._reported | self._unknown_symbols | self.fields.unregistered.keys() | self.relations.unregistered.keys()
        )

    def report(self, names):
        """Record unregistered names met in another process."""
        self._reported.update(names)

    @property
    def digest(self):
        return self._digest


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
        from .rules import canonical_content_id

        value = canonical_content_id(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(value):
            raise ProtocolError("Non-finite public number")
        return Field(name, name + "=<number>", (value, 1, 1, 1))
    return Field(name, f"{name}={value}", (0, 1, 1, 0))


@dataclass
class Effect:
    """A local program: field rows, relations between them and entity bindings."""

    nodes: list[list[Field]] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)
    bindings: list[tuple[int, str, str]] = field(default_factory=list)
    _tables: dict = field(default_factory=dict, repr=False, compare=False)

    def edge_table(self, vocabulary):
        """Rows of (source, target, relation index), built once per vocabulary."""
        table = self._tables.get(vocabulary.digest)
        if table is None:
            import torch

            encode = vocabulary.relations.encode
            table = torch.tensor(
                [(i, j, encode(role)) for i, j, role in self.edges], dtype=torch.long
            ).reshape(-1, 3)
            self._tables[vocabulary.digest] = table
        return table


def effect_tree(program):
    """Return a bounded cached parse tree; callers treat Effect as immutable."""
    encoded = json.dumps(
        program, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _effect_tree_cached(encoded)


@lru_cache(maxsize=8192)
def _effect_tree_cached(encoded):
    program = json.loads(encoded)
    effect = Effect()

    def visit(obj, path, parent=None, scope=None):
        scope = dict(scope or {})
        if isinstance(obj, list):
            for i, child in enumerate(obj):
                # Order belongs to the effect program, never to entity packing.
                visit(child, f"{path}[{i}]", parent, scope)
            return
        if not isinstance(obj, dict):
            obj = {"value": obj}
        index = len(effect.nodes)
        own = {
            k: v
            for k, v in obj.items()
            if not isinstance(v, (dict, list))
            and k not in {"ref", "bind"}
            and not (k == "name" and obj.get("kind") == "variable")
        }
        if obj.get("kind") == "variable":
            # Variable names are resolved as program-local identities, not vocab tokens.
            own["variable"] = True
        own["program_role"] = path
        effect.nodes.append(fields_of(own))
        if parent is not None:
            effect.edges.append((parent, index, "program_child"))
            effect.edges.append((index, parent, "program_parent"))
        if obj.get("kind") == "entity" and obj.get("ref"):
            effect.bindings.append((index, obj["ref"], path.split(".")[-1]))
        if obj.get("bind"):
            scope[obj["bind"]] = index
        if obj.get("kind") == "variable" and obj.get("name"):
            binder = scope.get(obj["name"])
            if binder is not None:
                effect.edges.extend(
                    [
                        (binder, index, "binds_variable"),
                        (index, binder, "variable_from"),
                    ]
                )
            else:
                variables.setdefault(obj["name"], []).append(index)
        for key, child in obj.items():
            if key == "refs" and obj.get("kind") == "set":
                for ref in child:
                    effect.bindings.append((index, ref, "set_member"))
                continue
            if isinstance(child, (dict, list)):
                visit(child, key, index, scope)

    variables = {}
    if program:
        visit(program, "root")
    else:
        visit({"kind": "unknown", "known": False}, "root")
    for refs in variables.values():
        for i in refs:
            for j in refs:
                effect.edges.append((i, j, "same_variable"))
    return effect


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


_FIELD_ROWS = {}
_FIELD_ROWS_LIMIT = 16384


def entity_fields(entity):
    """`fields_of` an entity, shared between equal entities: the cards, relics and
    powers of a run repeat across its frames and across runs. Rows are read only."""
    try:
        key = json.dumps(entity, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return fields_of(entity)
    rows = _FIELD_ROWS.get(key)
    if rows is None:
        rows = fields_of(entity)
        if len(_FIELD_ROWS) >= _FIELD_ROWS_LIMIT:
            _FIELD_ROWS.pop(next(iter(_FIELD_ROWS)))
        _FIELD_ROWS[key] = rows
    return rows


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
    from .public_history import HISTORY_KINDS, unknown_memory

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
    entities, effects, relations, map_keys = _compress_map(
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


def entity_program(entity):
    """The engine's program, or the public rule text where the engine has none."""
    from . import rules

    program = entity.get("semantic_program") or entity.get("program")
    if not program or program.get("op") == "OPAQUE_RULE":
        text = rules.program(entity)
        if text is not None:
            return text
    return effect_tree(program)


MAP_DYNAMIC_FIELDS = {"ref", "current", "visited", "selectable"}


def _compress_map(entities, effects, relations, selected):
    """Keep the current map node and its frontier; fold the rest into programs.

    The frontier is every successor of the current node (the entry nodes when
    no node is current) plus any map node another entity still refers to. Each
    frontier node receives the map reachable from it as its local program, so
    nodes which can no longer be reached, and the path already taken, leave
    the observation. While the player may ignore paths (``free_travel``), every
    node of the next floor is a successor.
    """
    nodes = {
        x["ref"]: x
        for x in entities
        if x.get("entity_type") == "map_node" and x.get("ref")
    }
    if not nodes:
        return entities, effects, relations, {}
    forward = {ref: [] for ref in nodes}
    entered = set()
    referenced = set(selected)
    for edge in relations:
        source, target = edge.get("source"), edge.get("target")
        if edge.get("role") == "map_edge" and source in nodes and target in nodes:
            forward[source].append(target)
            entered.add(target)
        else:
            referenced.update((source, target))
    for item, effect in zip(entities, effects):
        if item.get("entity_type") == "map_node":
            continue
        for role in ("source", "target", "option", "owner"):
            referenced.update(item.get(role + "_refs", []))
            referenced.add(item.get(role + "_ref"))
        referenced.update(ref for _, ref, _ in effect.bindings)
    free = any(x.get("free_travel") is True for x in entities)
    if free:
        floors = {}
        for ref, x in nodes.items():
            floors.setdefault(x.get("floor"), []).append(ref)
        steps = {
            ref: sorted(set(forward[ref]) | set(floors.get(x["floor"] + 1, ())))
            if type(x.get("floor")) is int
            else forward[ref]
            for ref, x in nodes.items()
        }
    else:
        steps = forward
    current = [ref for ref, x in nodes.items() if x.get("current") is True]
    frontier = (
        {target for ref in current for target in steps[ref]}
        if current
        else set(nodes) - entered
    )
    keep = set(current) | frontier | (referenced & nodes.keys())
    keys = {
        ref: _map_key(ref, nodes, forward, steps)
        for ref in sorted(keep - set(current))
    }
    kept_entities, kept_effects = [], []
    for item, effect in zip(entities, effects):
        if item.get("entity_type") == "map_node":
            ref = item.get("ref")
            if ref not in keep:
                continue
            if ref in keys:
                effect = _map_program(keys[ref])
        kept_entities.append(item)
        kept_effects.append(effect)
    kept_relations = [
        edge
        for edge in relations
        if not (
            edge.get("role") == "map_edge"
            and edge.get("source") in nodes
            and edge.get("target") in nodes
            and not (edge["source"] in keep and edge["target"] in keep)
        )
    ]
    return kept_entities, kept_effects, kept_relations, keys


def _map_key(root, nodes, forward, steps):
    """Canonical description of the map reachable from root; root is node 0.

    ``forward`` holds the drawn paths and ``steps`` the moves actually allowed,
    which also include every next-floor node during free travel.
    """
    reachable, stack = {root}, [root]
    while stack:
        for target in steps[stack.pop()]:
            if target not in reachable:
                reachable.add(target)
                stack.append(target)
    order = [root] + sorted(reachable - {root})
    index = {ref: i for i, ref in enumerate(order)}
    static = [
        {k: v for k, v in nodes[ref].items() if k not in MAP_DYNAMIC_FIELDS}
        for ref in order
    ]
    links = sorted(
        (index[ref], index[target])
        for ref in order
        for target in forward[ref]
        if target in index
    )
    free = sorted(
        (index[ref], index[target])
        for ref in order
        for target in steps[ref]
        if target not in forward[ref]
    )
    return json.dumps(
        [static, links, free], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


@lru_cache(maxsize=4096)
def _map_program(encoded):
    """The reachable map as a program: a map is fixed for an act, so this is cached."""
    static, links, free = json.loads(encoded)
    effect = Effect()
    base = static[0].get("floor")
    for node in static:
        floor = node.get("floor")
        if all(type(x) in (int, float) for x in (base, floor)):
            # Distance ahead of the frontier node, in floors.
            node = dict(node, depth=floor - base)
        effect.nodes.append(fields_of(node))
    effect.edges.extend(
        map_relations(len(static), [tuple(x) for x in links], [tuple(x) for x in free])
    )
    return effect


def map_relations(count, links, free=()):
    """Classify every ordered node pair of a map by reachability along its paths.

    ``free`` pairs are single moves which no path provides; they are labelled
    as such instead of unreachable.
    """
    direct, free = set(links), set(free)
    adjacency = {i: [j for a, j in links if a == i] for i in range(count)}
    reachable = {}

    def reach(i, stack):
        if i in stack:
            raise ProtocolError("Map must be a DAG")
        if i not in reachable:
            reachable[i] = set()
            for j in adjacency[i]:
                reachable[i].add(j)
                reachable[i].update(reach(j, stack | {i}))
        return reachable[i]

    for i in range(count):
        reach(i, set())
    result = []
    for i in range(count):
        for j in range(count):
            role = (
                "self"
                if i == j
                else "forward_direct"
                if (i, j) in direct
                else "reverse_direct"
                if (j, i) in direct
                else "forward_indirect"
                if j in reachable[i]
                else "reverse_indirect"
                if i in reachable[j]
                else "free_direct"
                if (i, j) in free
                else "free_reverse"
                if (j, i) in free
                else "unreachable"
            )
            result.append((i, j, "map_" + role))
    return result


def names_from_frames(frames):
    """Field symbols, field names and relation roles to freeze into a vocabulary."""
    # Schema categories are public constants. Initializing from only a Neow
    # snapshot must not collapse combat/map/shop phases to the same unknown ID.
    categories = {
        "phase": "combat map event rest_site shop treasure rewards card_reward card_select bundle_select crystal_sphere",
        "entity_type": "global action player card pile_summary enemy summon intent previous_intent history_probability history_rule known_draw_position orb_slots orb power relic potion map_node displayed_variable event event_option rest_option shop_item reward reward_alternative bundle event_state tool board_cell event_context",
        "zone": "deck hand draw_pile discard_pile exhaust_pile selection reward treasure shop",
        "operation": "unknown transform exhaust remove enchant discard upgrade copy obtain",
        "source": "unknown hand deck draw_pile discard_pile exhaust_pile",
        "destination": "unknown hand deck draw_pile discard_pile exhaust_pile removed",
        "mode": "buffered sequential Small Big",
        "effect_coverage": "opaque partial complete unknown",
        "character": "IRONCLAD SILENT DEFECT REGENT NECROBINDER Ironclad Silent Defect Regent Necrobinder",
        "card_type": "None Attack Skill Power Status Curse Quest",
        "rarity": "None Basic Common Uncommon Rare Ancient Special Token Event Status Curse Quest",
        "target_type": "None Self AnyEnemy AllEnemies RandomEnemy AnyPlayer AnyAlly AllAllies TargetedNoCreature Osty",
        "intent": "Attack Buff Debuff DebuffStrong Defend Escape Heal Hidden Sleep StatusCard CardDebuff DeathBlow Stun Summon Unknown",
        "content_id": "Small Big CrystalSphereGold CrystalSphereRelic CrystalSpherePotion CrystalSphereCurse CrystalSphereCardReward empty CRYSTAL_SPHERE potion_drop question_room question_previous_shop",
        "rider_effect": "None Violence Sapping Choking Energized Wisdom Chaos Expertise Curious Improvement",
        "op": "OPAQUE_RULE REVEAL",
        "shape": "cell square",
    }
    from .public_history import PROBABILITY_SCOPES

    categories["scope"] = " ".join(PROBABILITY_SCOPES)
    symbols = {
        f"{name}={value}"
        for name, values in categories.items()
        for value in values.split()
    }
    for name in [
        "known",
        "applicable",
        "x_cost",
        "upgraded",
        "exhausted",
        "used",
        "current",
        "visited",
        "selectable",
        "complete",
        "revealed",
        "order_known",
        "enabled",
        "sold_out",
        "can_finish",
        "can_skip",
        "can_cancel",
        "order_matters",
        "repetition_allowed",
        "selected",
    ]:
        symbols.update({f"{name}=True", f"{name}=False"})
    # Selection progress on the global token and selected entities.
    for name in ["selected_order", "min_total", "max_total", "selected_count", "remaining_required"]:
        symbols.add(f"{name}=<number>")
    from . import rules

    symbols |= rules.symbols()
    # The alternatives of a card reward have no model of their own; the reward
    # screen's text table names them.
    symbols.update("content_id=" + name for name in rules.reward_alternatives())
    fields = set(ENTITY_FIELDS) | CONTAINERS | {"empty", "selected", "selected_order"}
    fields.update(rules.TEXT_FIELDS)
    # Every public field may be a number or a flag, whichever frames were seen.
    for name in ENTITY_FIELDS:
        symbols.update((f"{name}=<number>", f"{name}=True", f"{name}=False"))
    # Values a description displays are exported under the same name in `stats`.
    for name in rules.displayed() | {"passive", "evoke"}:
        fields.add("stats." + name)
        symbols.add(f"stats.{name}=<number>")
        # Event/rest options export these as owned displayed_variable entities.
        symbols.add(f"content_id={name}")
    # The reveal program of a Crystal Sphere tool.
    for name in ("shape", "radius", "clip_to_board"):
        fields.add(name)
        symbols.update((f"{name}=<number>", f"{name}=True", f"{name}=False"))
    fields.update("reference." + role for role in REFERENCE_ROLES)
    fields.update("binding." + role for role in BINDING_ROLES)
    relations = {"map_" + role for role in MAP_RELATIONS} | set(PROGRAM_RELATIONS)
    relations.update(rules.TEXT_RELATIONS)
    for role in REFERENCE_ROLES + ENGINE_RELATIONS:
        relations.update((role, "reverse_" + role))
    for frame in frames:
        obs = observation(frame)
        rows = (
            obs.tokens
            + [n for e in obs.effects for n in e.nodes]
            + [fields_of(obs.context)]
        )
        for row in rows:
            symbols.update(f.symbol for f in row)
            fields.update(f.name for f in row)
        relations.update(role for _, _, role in obs.edges)
        for effect in obs.effects:
            relations.update(role for _, _, role in effect.edges)
            fields.update("binding." + role for _, _, role in effect.bindings)
    # A card names its enchantment or affliction, and a boss node of the map its
    # encounter, by the content id of that model.
    for symbol in list(symbols):
        for kind in ("enchantment", "affliction", "encounter"):
            if symbol.startswith(f"content_id={kind.upper()}."):
                symbols.add(kind + symbol[len("content_id"):])
        # An option names a card, relic, potion or enchantment it gives, takes or applies.
        if symbol.startswith(("content_id=CARD.", "content_id=RELIC.", "content_id=POTION.", "content_id=ENCHANTMENT.")):
            symbols.add("names" + symbol[len("content_id"):])
    return symbols, fields, relations


def field_tensors(rows, vocabulary):
    """Pack field rows into (symbol ids, field-name ids, numbers, mask) on the CPU."""
    import torch

    width = max(map(len, rows), default=1)
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
