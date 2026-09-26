"""Inspectable public combat transitions used by the live search planner.

This is a bounded approximation. Predictions describe supported effects, never
claim to reproduce arbitrary native hooks or reveal future draws.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field

from .astar import player


def number(value, default=0.0):
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return float(default)


def stat(entity, name, default=0.0):
    return next(
        (
            number(v, default)
            for k, v in (entity.get("stats") or {}).items()
            if k.lower().replace("_", "") == name.lower()
        ),
        float(default),
    )


def content(entity):
    return entity.get("content_id", "").split(".")[-1].upper()


@dataclass
class Combat:
    energy: float
    hp: float
    block: float
    enemies: dict
    strength: float
    vulnerable: dict
    used: frozenset = frozenset()
    hand: frozenset = frozenset()
    discard: frozenset = frozenset()
    exhaust: frozenset = frozenset()
    lost_hp_this_turn: bool | None = None
    incoming: dict = field(default_factory=dict)
    weak: dict = field(default_factory=dict)
    potions: frozenset = frozenset()


@dataclass
class EffectPrediction:
    state: Combat | None
    coverage: str
    reason: str | None
    damage: float = 0.0
    block: float = 0.0

    def to_dict(self):
        data = asdict(self)
        if data["state"] is not None:
            for key in ("used", "hand", "discard", "exhaust", "potions"):
                data["state"][key] = sorted(data["state"][key])
        return data


class CombatModel:
    profiles = ("legacy", "second_wind", "spite", "cards", "potions")
    cards = frozenset(
        {
            "STRIKE_IRONCLAD",
            "DEFEND_IRONCLAD",
            "ANGER",
            "BLUDGEON",
            "TWIN_STRIKE",
            "IRON_WAVE",
            "HEMOKINESIS",
            "BLOOD_WALL",
            "BREAKTHROUGH",
            "WHIRLWIND",
            "RAMPAGE",
            "BASH",
            "TREMBLE",
            "MOLTEN_FIST",
            "SETUP_STRIKE",
            "INFLAME",
            "BODY_SLAM",
        }
    )

    safe_powers = frozenset(
        {"STRENGTH", "DEXTERITY", "WEAK", "FRAIL", "VULNERABLE", "SETUP_STRIKE"}
    )
    # Native hooks only run on acquisition or after combat. Their already
    # resolved rewards are in the public snapshot; combat-end is a boundary.
    safe_relics = frozenset(
        {"BURNING_BLOOD", "FISHING_ROD", "LOST_COFFER", "KALEIDOSCOPE"}
    )
    potion_models = frozenset(
        {
            "FIRE_POTION",
            "EXPLOSIVE_AMPOULE",
            "BLOCK_POTION",
            "WEAK_POTION",
            "STRENGTH_POTION",
            "ENERGY_POTION",
        }
    )

    def __init__(self, frame, effect_profile="potions", dynamic_legality=False):
        if effect_profile not in self.profiles:
            raise ValueError("Unknown effect profile")
        self.profile = effect_profile
        self.dynamic_legality = dynamic_legality
        self.strict = self.profiles.index(effect_profile) >= 3
        self.entities = frame["public"]["entities"]
        self.refs = {e["ref"]: e for e in self.entities if e.get("ref")}
        p = player(frame)
        enemies = {
            e["ref"]: [number(e.get("hp")), number(e.get("block"))]
            for e in self.entities
            if e.get("entity_type") == "enemy"
        }
        self.incoming = dict.fromkeys(enemies, 0.0)
        self.powers = {}
        for e in self.entities:
            if e.get("entity_type") == "intent" and e.get("owner_ref") in self.incoming:
                self.incoming[e["owner_ref"]] += max(
                    0,
                    number(
                        e.get("total_damage"),
                        number(e.get("damage")) * max(1, number(e.get("hits"), 1)),
                    ),
                )
            if e.get("entity_type") == "power":
                self.powers[(e.get("owner_ref"), content(e))] = number(e.get("stacks"))
        self.unsupported_hooks = any(
            (
                e.get("entity_type") == "power"
                and content(e).removesuffix("_POWER") not in self.safe_powers
            )
            or (e.get("entity_type") == "relic" and content(e) not in self.safe_relics)
            for e in self.entities
        )
        self.root = Combat(
            number(p.get("energy")),
            number(p.get("hp")),
            number(p.get("block")),
            enemies,
            self.amount("player", "STRENGTH"),
            {ref: self.amount(ref, "VULNERABLE") for ref in enemies},
        )

        fact = p.get("lost_hp_this_turn")
        self.root.lost_hp_this_turn = fact if type(fact) is bool else None
        self.root.incoming = dict(self.incoming)
        self.root.weak = {ref: self.amount(ref, "WEAK") for ref in enemies}
        self.root.potions = frozenset(
            e["ref"]
            for e in self.entities
            if e.get("entity_type") == "potion" and e.get("ref")
        )
        for zone in ("hand", "discard", "exhaust"):
            native_zone = {"discard": "discard_pile", "exhaust": "exhaust_pile"}.get(
                zone, zone
            )
            setattr(
                self.root,
                zone,
                frozenset(
                    e["ref"]
                    for e in self.entities
                    if e.get("entity_type") == "card" and e.get("zone") == native_zone
                ),
            )

    def amount(self, owner, name):
        return self.powers.get(
            (owner, name), self.powers.get((owner, name + "_POWER"), 0)
        )

    def predict(self, candidate):
        """Predict one root action; the planner uses this same transition in branches."""
        return self.transition(self.root, candidate)

    def transition(self, state, candidate):
        verb = candidate["verb"]
        if verb not in {"PLAY_CARD", "USE_POTION"}:
            return EffectPrediction(None, "unknown", "boundary")
        source_ref = (candidate.get("source_refs") or [None])[0]
        source = self.refs.get(source_ref, {})
        cost = number(source.get("cost")) if verb == "PLAY_CARD" else 0
        if source.get("x_cost"):
            cost = state.energy
        targets = list(candidate.get("target_refs", []))
        if (
            source_ref in state.used
            or cost < 0
            or cost > state.energy
            or any(t in state.enemies and state.enemies[t][0] <= 0 for t in targets)
        ):
            return EffectPrediction(None, "unknown", "unavailable")
        name = content(source)
        modeled_potion = (
            verb == "USE_POTION"
            and self.profile == "potions"
            and name in self.potion_models
            and bool(source.get("stats"))
        )
        if self.strict:
            supported = (
                verb == "PLAY_CARD"
                and name
                in (
                    self.cards
                    | {"SECOND_WIND", "SPITE"}
                    | ({"BLOODLETTING"} if self.dynamic_legality else set())
                )
            ) or modeled_potion
            unsupported = (
                self.unsupported_hooks
                or source.get("enchantment")
                or source.get("affliction")
            )
            if not supported or unsupported:
                return EffectPrediction(
                    None,
                    "unknown",
                    "unsupported_hooks" if unsupported else "unsupported_effect",
                    stat(source, "Damage", stat(source, "CalculationBase")),
                    stat(source, "Block"),
                )
        nxt = deepcopy(state)
        nxt.energy -= cost
        nxt.used = state.used | {source_ref}
        if verb == "USE_POTION":
            nxt.potions = state.potions - {source_ref}
        if verb == "PLAY_CARD":
            nxt.hand = state.hand - {source_ref}
            if source.get("exhaust_on_next_play") or any(
                str(k).lower() == "exhaust" for k in source.get("keywords", [])
            ):
                nxt.exhaust = state.exhaust | {source_ref}
            elif source.get("card_type", "").lower() != "power":
                nxt.discard = state.discard | {source_ref}
        damage = stat(source, "Damage", stat(source, "CalculationBase"))
        block = stat(source, "Block")
        repeat = max(1, min(20, int(stat(source, "Repeat", 1))))
        name = content(source)
        unknown_spite_condition = False
        if name == "SPITE" and self.profiles.index(self.profile) >= 2:
            unknown_spite_condition = state.lost_hp_this_turn is None
            repeat = repeat if state.lost_hp_this_turn is True else 1
        if name == "TWIN_STRIKE":
            repeat = 2
        if name == "BODY_SLAM":
            damage = state.block
        if name == "WHIRLWIND":
            repeat = max(0, int(cost) if self.strict else min(20, int(cost)))
        hp_loss = max(0, stat(source, "HpLoss"))
        nxt.hp -= hp_loss
        if hp_loss > 0:
            nxt.lost_hp_this_turn = True
        if nxt.hp <= 0:
            return EffectPrediction(nxt, "approximate", "estimated_self_lethal")
        if damage > 0 or (self.strict and name == "BODY_SLAM"):
            damage = max(0, damage + (0 if modeled_potion else state.strength))
            if not modeled_potion and self.amount("player", "WEAK") > 0:
                damage = damage * 0.75 if self.strict else math.floor(damage * 0.75)
            if source.get("target_type") == "AllEnemies":
                targets = [r for r, (hp, _) in nxt.enemies.items() if hp > 0]
            for target in targets:
                if target not in nxt.enemies:
                    continue
                hit = math.floor(
                    damage
                    * (
                        1.5
                        if not modeled_potion and state.vulnerable[target] > 0
                        else 1
                    )
                )
                for _ in range(repeat):
                    hp, shield = nxt.enemies[target]
                    nxt.enemies[target] = [
                        max(0, hp - max(0, hit - shield)),
                        max(0, shield - hit),
                    ]
        block_count = 1
        if name == "SECOND_WIND" and self.profile != "legacy":
            consumed = frozenset(
                ref
                for ref in nxt.hand
                if self.refs[ref].get("card_type", "").lower() != "attack"
            )
            nxt.hand -= consumed
            nxt.exhaust |= consumed
            nxt.used |= consumed
            block_count = len(consumed)
        if block > 0:
            block = max(
                0, block + (0 if modeled_potion else self.amount("player", "DEXTERITY"))
            )
            if not modeled_potion and self.amount("player", "FRAIL") > 0:
                block = math.floor(block * 0.75)
            nxt.block += block * block_count
        if name in {"BASH", "TREMBLE", "MOLTEN_FIST"}:
            for target in targets:
                if target in nxt.vulnerable:
                    if name == "MOLTEN_FIST":
                        if not self.strict or nxt.enemies[target][0] > 0:
                            nxt.vulnerable[target] *= 2
                    else:
                        nxt.vulnerable[target] += stat(source, "VulnerablePower")
        if name in {"INFLAME", "SETUP_STRIKE"}:
            nxt.strength += stat(source, "StrengthPower")
        if name == "BLOODLETTING" and verb == "PLAY_CARD" and self.dynamic_legality:
            nxt.energy += max(0, stat(source, "Energy"))
        potion_reason = None
        if modeled_potion:
            if name == "STRENGTH_POTION":
                nxt.strength += max(0, stat(source, "StrengthPower"))
            elif name == "ENERGY_POTION":
                nxt.energy += max(0, stat(source, "Energy"))
                potion_reason = "new_legality_requires_native_observation"
            elif name == "WEAK_POTION":
                for target in targets:
                    if target not in nxt.weak:
                        continue
                    if nxt.weak[target] <= 0 and stat(source, "WeakPower") > 0:
                        hits = [
                            e
                            for e in self.entities
                            if e.get("entity_type") == "intent"
                            and e.get("owner_ref") == target
                        ]
                        uncertain = self.amount("player", "VULNERABLE") > 0
                        round_hit = math.ceil if uncertain else math.floor
                        nxt.incoming[target] = sum(
                            round_hit(number(e.get("damage")) * 0.75)
                            * max(1, number(e.get("hits"), 1))
                            for e in hits
                        )
                        if uncertain:
                            potion_reason = "intent_rounding_uncertain"
                    nxt.weak[target] += max(0, stat(source, "WeakPower"))
        simple = (
            verb == "PLAY_CARD"
            and (
                name in self.cards
                or (name == "BLOODLETTING" and self.dynamic_legality)
                or (name == "SECOND_WIND" and self.profile != "legacy")
                or (
                    name == "SPITE"
                    and self.profiles.index(self.profile) >= 2
                    and not unknown_spite_condition
                )
            )
            and not source.get("enchantment")
            and not source.get("affliction")
        )
        if modeled_potion:
            return EffectPrediction(
                nxt,
                "modeled" if potion_reason is None else "approximate",
                potion_reason,
                damage,
                block,
            )
        coverage = "modeled" if self.strict and simple else "approximate"
        reason = None if simple else "opaque_or_reveal"
        if (
            self.strict
            and self.dynamic_legality
            and name == "BLOODLETTING"
            and stat(source, "Energy") > 0
        ):
            coverage, reason = "approximate", "new_legality_requires_native_observation"
        if self.strict and name in {"ANGER", "RAMPAGE"}:
            coverage, reason = "approximate", "pile_change_unmodeled"
        return EffectPrediction(nxt, coverage, reason, damage, block)
