"""The public fields of a decision frame: the whitelists of entity fields, stats,
references and program fields, and the cleaning that lets only them through to the
model (`clean_frame`, `clean_public`, `clean_entity`)."""

from copy import deepcopy


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
    from main.encoding import rules

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


def clean_frame(frame):
    """Retain replay/routing metadata, but only whitelisted model input fields."""
    # The public state and the candidates, most of a frame, are built anew below: only
    # the rest is copied, in the frame's order.
    result = {key: value if key in ("public", "legal") else deepcopy(value) for key, value in frame.items()}
    result["public"] = clean_public(frame["public"])
    # Public candidate semantics are cleaned by the same representation path.
    result["legal"] = {
        "candidates": [
            clean_public({"decoder_bank": [c]})["decoder_bank"][0]
            for c in frame["legal"]["candidates"]
        ]
    }
    return result
