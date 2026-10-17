"""The frozen name tables of a checkpoint (`Table`, `Vocabulary`): field symbols,
field names and relation roles, registered from the engine's catalog and the
representation's own enumerations (`names_from_frames`)."""

import hashlib

from main.env.protocol import fingerprint
from main.encoding.fields import fields_of
from main.encoding.frames import BINDING_ROLES, CONTAINERS, ENTITY_FIELDS
from main.encoding.representation import observation


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
    from main.env.public_history import PROBABILITY_SCOPES

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
    from main.encoding import rules

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
