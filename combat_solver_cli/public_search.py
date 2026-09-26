"""Bounded public-state planning, without an engine, seed, or hidden-state handle.

This is an approximate rule planner, not a game simulator. It searches sequences
of modeled, currently playable cards within this turn. Draws, opaque effects and
end-turn are leaves: their unknown continuations are never read from the real run.
Every native root candidate is scored, including actions outside model coverage.
"""

import math

from model.protocol import ProtocolError
from model.representation import clean_entity, clean_public

from .astar import card_value, player, preference
from .public_effects import CombatModel, content, number, stat
from .public_future import FutureSampler
from .public_resources import ResourcePolicy
from .public_rewards import RewardPolicy


def public_view(frame):
    """Drop transport/debug fields before any rule can inspect an observation."""
    return {
        "public": clean_public(frame["public"]),
        "legal": {
            "candidates": [
                clean_entity(c, action=True) for c in frame["legal"]["candidates"]
            ]
        },
    }


class PublicPlanner:
    """Replan after every real action; only the first native action is executed."""

    version = "public-rules-beam-v5"

    def __init__(
        self,
        depth=3,
        beam_width=16,
        sampling_seed=0,
        effect_profile="potions",
        route_resources=False,
        deck_rewards=False,
        dynamic_legality=False,
        draw_samples=0,
        future_value=False,
    ):
        if type(depth) is not int or not 1 <= depth <= 12:
            raise ValueError("depth must be an integer from 1 to 12")
        if type(beam_width) is not int or not 1 <= beam_width <= 256:
            raise ValueError("beam_width must be an integer from 1 to 256")
        if type(sampling_seed) is not int:
            raise ValueError(
                "sampling_seed must be an integer independent of the game seed"
            )
        if effect_profile not in CombatModel.profiles:
            raise ValueError("Unknown effect profile")
        if type(route_resources) is not bool:
            raise ValueError("route_resources must be a boolean")
        if type(deck_rewards) is not bool:
            raise ValueError("deck_rewards must be a boolean")
        if type(dynamic_legality) is not bool:
            raise ValueError("dynamic_legality must be a boolean")
        if type(draw_samples) is not int or not 0 <= draw_samples <= 8:
            raise ValueError("draw_samples must be an integer from 0 to 8")
        if type(future_value) is not bool:
            raise ValueError("future_value must be a boolean")
        self.route_resources = route_resources
        self.deck_rewards = deck_rewards
        self.dynamic_legality = dynamic_legality
        self.draw_samples = draw_samples
        self.future_value = future_value
        self.sampling_seed = sampling_seed
        self.effect_profile = effect_profile
        self.depth = depth
        self.beam_width = beam_width

    def choose(self, frame):
        view = public_view(frame)
        candidates = view["legal"]["candidates"]
        if not candidates:
            raise ProtocolError("Public planner requires legal candidates")
        if len({c["candidate_ref"] for c in candidates}) != len(candidates):
            raise ProtocolError("Ambiguous candidate references")
        if view["public"]["phase"] == "combat":
            scores, expanded, leaves = self._combat(view)
        else:
            resources = ResourcePolicy(view) if self.route_resources else None
            rewards = RewardPolicy(view) if self.deck_rewards else None
            scores = []
            for c in candidates:
                score = rewards.score(c) if rewards else None
                if score is None:
                    score = resources.score(c) if resources else None
                scores.append(self._outside(view, c) if score is None else score)
            expanded, leaves = 0, len(candidates)
        if not all(math.isfinite(s) for s in scores):
            raise ProtocolError("Non-finite public search score")
        best = max(range(len(candidates)), key=lambda i: scores[i])
        return frame["legal"]["candidates"][best], {
            "planner": self.version,
            "evaluation": "approximate_rules_not_win_probability",
            "depth": self.depth,
            "sampling_seed": self.sampling_seed,
            "effect_profile": self.effect_profile,
            "route_resources": self.route_resources,
            "deck_rewards": self.deck_rewards,
            "dynamic_legality": self.dynamic_legality,
            "draw_samples": self.draw_samples,
            "future_value": self.future_value,
            "future_model": (
                "one_turn_repeated_visible_intent" if self.draw_samples else "disabled"
            ),
            "future_boundaries": (
                [
                    "no_draw_replacement",
                    "no_start_turn_hooks",
                    "retained_or_ethereal_truncated",
                    "star_or_changed_cost_not_played",
                    "player_weak_or_frail_truncated",
                ]
                if self.draw_samples
                else []
            ),
            "future_terms": (
                [
                    "strength",
                    "effective_draw",
                    "persistent_defense_prior",
                    "retained_potions",
                    "visible_survival",
                ]
                if self.future_value
                else []
            ),
            "beam_width": self.beam_width,
            "expanded": expanded,
            "boundary_leaves": leaves,
            "candidate_scores": [
                {"candidate_ref": c["candidate_ref"], "score": s}
                for c, s in zip(candidates, scores)
            ],
        }

    def abandon_reason(self, frame):
        view = public_view(frame)
        # Never infer inevitable death from an approximate combat model. Only
        # abandon between rooms when all visible next rooms are combat rooms.
        if view["public"]["phase"] != "map":
            return None
        p = player(view)
        if number(p.get("hp")) > 0.08 * max(1, number(p.get("max_hp"))):
            return None
        entities = view["public"]["entities"]
        if any(e.get("entity_type") == "potion" for e in entities):
            return None
        refs = {e.get("ref"): e for e in entities if e.get("ref")}
        moves = [c for c in view["legal"]["candidates"] if c["verb"] == "MOVE_TO_NODE"]
        kinds = [
            content(refs.get((c.get("source_refs") or [None])[0], {})) for c in moves
        ]
        if kinds and all(k in {"MONSTER", "ELITE", "BOSS"} for k in kinds):
            return "low_hp_before_forced_combat_without_potion"
        return None

    def _outside(self, frame, candidate):
        score = preference(frame, candidate)
        entities = frame["public"]["entities"]
        refs = set(candidate.get("source_refs", []))
        source = next((e for e in entities if e.get("ref") in refs), {})
        # The old history search could revisit a skipped chest. This online
        # teacher cannot: taking the visible relic must beat leaving the room.
        if candidate["verb"] in {"OPEN_CHEST", "TAKE_TREASURE_RELIC"}:
            return 6.0
        if candidate["verb"] == "SELECT_BUNDLE":
            deck = [e for e in entities if e.get("zone") == "deck"]
            score += sum(card_value(c, deck) for c in source.get("cards", []))
        # Public numeric event outcomes are not a complete event interpreter.
        # Unknown events retain the existing neutral prior and native legality.
        return float(score)

    def _combat(self, frame):
        entities = frame["public"]["entities"]
        refs = {e["ref"]: e for e in entities if e.get("ref")}
        model = CombatModel(
            frame,
            effect_profile=self.effect_profile,
            dynamic_legality=self.dynamic_legality,
        )
        root, incoming = model.root, model.incoming
        future = (
            FutureSampler(model, self.sampling_seed, self.draw_samples)
            if self.draw_samples
            else None
        )
        # A pure one-turn HP score can defend forever against scaling enemies.
        # Under visible pressure, value reducing the encounter's remaining HP
        # more highly. This is a rule prior, not knowledge of the next intent.
        damage_weight = 1 + min(2.0, sum(incoming.values()) / 10)

        def value(state):
            threat = sum(
                state.incoming[r] for r, (hp, _) in state.enemies.items() if hp > 0
            )
            loss = max(0, threat - state.block)
            remaining_hp = state.hp - loss
            dealt = sum(
                root.enemies[r][0] - max(0, hp) for r, (hp, _) in state.enemies.items()
            )
            kills = sum(
                root.enemies[r][0] > 0 and hp <= 0
                for r, (hp, _) in state.enemies.items()
            )
            # Block only contributes when it prevents visible incoming damage.
            # Death risk dominates damage greed; this is not a calibrated value.
            result = (
                (remaining_hp - root.hp) * 3
                + dealt * damage_weight
                + kills * 8
                + (state.strength - root.strength) * 3
                + sum(
                    min(3, state.vulnerable[r] - root.vulnerable[r]) * 2
                    for r, (hp, _) in state.enemies.items()
                    if hp > 0
                )
                - (500 if remaining_hp <= 0 else 0)
                - (
                    (6 if self.future_value else 4) * len(root.potions - state.potions)
                    if self.effect_profile == "potions"
                    else 0
                )
            )
            if (
                self.future_value
                and remaining_hp > 0
                and not model.amount("player", "SETUP_STRIKE")
                and not any(
                    content(refs.get(r, {})) == "SETUP_STRIKE" for r in state.used
                )
            ):
                attacks = sum(
                    e.get("entity_type") == "card"
                    and e.get("card_type", "").lower() == "attack"
                    and e.get("zone") in {"hand", "draw_pile", "discard_pile"}
                    for e in entities
                )
                result += max(0, state.strength - root.strength) * min(3, attacks) * 1.5
            return result

        candidates = frame["legal"]["candidates"]

        def future_draw_value(state, source, *, cost_paid):
            draw = max(0, stat(source, "Cards", stat(source, "Draw")))
            available = sum(
                e.get("entity_type") == "card"
                and e.get("zone") == "draw_pile"
                and e.get("ref") not in state.hand | state.discard | state.exhaust
                for e in entities
            ) + len(state.discard - {source.get("ref")})
            energy = state.energy - (
                0 if cost_paid else max(0, number(source.get("cost"), 1))
            )
            return min(6, min(draw, available) * min(2, max(0, energy)))

        def transition(state, candidate):
            verb = candidate["verb"]
            if verb == "END_TURN":
                return None, value(state), "end_turn"
            prediction = model.transition(state, candidate)
            if prediction.reason == "unavailable":
                return None, -math.inf, "unavailable"
            if prediction.reason == "estimated_self_lethal":
                return None, -1_000_000.0, prediction.reason
            if prediction.state is None:
                # A displayed magnitude can inform a leaf prior without
                # inventing a successor or a kill under unsupported hooks.
                threat = sum(
                    incoming[r] for r, (hp, _) in state.enemies.items() if hp > 0
                )
                bonus = 0.5 * max(0, prediction.damage) + min(
                    max(0, prediction.block), max(0, threat - state.block)
                )
                source = refs.get((candidate.get("source_refs") or [None])[0], {})
                hand_attacks = sum(
                    e.get("zone") == "hand"
                    and e.get("card_type", "").lower() == "attack"
                    for e in entities
                )
                bonus += min(hand_attacks, max(0, stat(source, "Energy"))) * 5
                if stat(source, "HpLoss") >= state.hp:
                    bonus = -1_000_000
                if self.future_value and state.hp > threat - state.block:
                    if content(source) == "METALLICIZE":
                        bonus += min(8, max(0, stat(source, "Block")))
                    elif content(source) == "BARRICADE":
                        bonus += 5 if state.block > 0 else 2
                    bonus += future_draw_value(state, source, cost_paid=False)
                return (
                    None,
                    value(state) + bonus + preference(frame, candidate),
                    prediction.reason,
                )
            nxt = prediction.state
            source_ref = (candidate.get("source_refs") or [None])[0]
            source = refs.get(source_ref, {})
            damage, block = prediction.damage, prediction.block
            if prediction.reason is None:
                return nxt, value(nxt), None
            if (
                self.dynamic_legality
                and prediction.reason == "new_legality_requires_native_observation"
            ):
                return nxt, value(nxt), None
            # Do not treat an opaque action as a simulated no-op and keep
            # searching past it. Give it an explicit leaf estimate instead.
            bonus = 0.0
            if verb == "USE_POTION":
                threat = sum(
                    incoming[r] for r, (hp, _) in state.enemies.items() if hp > 0
                )
                bonus = 6.0 if threat - state.block >= state.hp * 0.5 else -5.0
            elif source.get("card_type", "").lower() == "power":
                bonus = 4.0 if state.hp > sum(incoming.values()) else -4.0
            else:
                bonus = min(4, max(0, stat(source, "Cards", stat(source, "Draw"))) * 2)
                if damage == 0 and block == 0:
                    bonus += 0.25
            # Public printed magnitudes inform leaf values, without claiming
            # to simulate the status/energy effects or their continuations.
            hand_attacks = sum(
                e.get("zone") == "hand"
                and e.get("card_type", "").lower() == "attack"
                and e.get("ref") not in nxt.used
                for e in entities
            )
            if content(source) == "INFLAME":
                bonus += max(0, stat(source, "StrengthPower")) * (hand_attacks + 3)
            elif content(source) == "DEMON_FORM":
                bonus += max(0, stat(source, "StrengthPower")) * 5
            elif content(source) == "RUPTURE":
                self_damage_cards = sum(
                    e.get("zone") == "deck" and stat(e, "HpLoss") > 0 for e in entities
                )
                bonus += max(0, stat(source, "StrengthPower")) * min(
                    5, self_damage_cards
                )
            bonus += min(3, max(0, stat(source, "VulnerablePower"))) * 2
            bonus += min(3, max(0, stat(source, "WeakPower"))) * 2
            if stat(source, "Energy") > 0 and hand_attacks:
                bonus += min(hand_attacks, stat(source, "Energy")) * 5
            if self.future_value:
                bonus += future_draw_value(nxt, source, cost_paid=True)
                if content(source) == "METALLICIZE":
                    bonus += min(8, max(0, stat(source, "Block")))
                elif content(source) == "BARRICADE":
                    bonus += 5 if nxt.block > 0 else 2
            # Opportunity estimate for the still-visible unused basic hand;
            # this does not invent cards revealed by a draw or other effects.
            potential = self._hand_potential(nxt, refs, candidates, incoming)
            return None, value(nxt) + bonus + potential, "opaque_or_reveal"

        base = value(root)
        scores, expanded, leaves = [], 0, 0
        for first in candidates:
            state, score, reason = transition(root, first)
            if reason == "unavailable":
                # Native root legality wins over an approximate cost model.
                scores.append(base - 1)
                leaves += 1
                continue
            expanded += 1
            leaves += reason is not None
            beam = [] if state is None else [state]
            best = score
            for _ in range(1, self.depth):
                children = []
                for parent in beam:
                    branch_candidates = (
                        self._hypothetical_candidates(parent, model, candidates)
                        if self.dynamic_legality
                        else candidates
                    )
                    for candidate in branch_candidates:
                        child, score, reason = transition(parent, candidate)
                        if reason == "unavailable":
                            continue
                        expanded += 1
                        leaves += reason is not None
                        best = max(best, score)
                        if child is not None:
                            children.append((score, child))
                if not children:
                    break
                # Root actions are never pruned; beam width limits only their
                # hypothetical continuations. All root scores remain exported.
                children.sort(key=lambda pair: pair[0], reverse=True)
                beam = [child for _, child in children[: self.beam_width]]
            if future is not None:
                # Keep the immediate successor even if its current-turn value
                # is low: growth can reverse that ordering next turn. At most
                # one later beam leaf is sampled for each native root.
                sampled_states = [state] if state is not None else []
                if beam:
                    best_leaf = max(beam, key=value)
                    if best_leaf is not state:
                        sampled_states.append(best_leaf)
                if first["verb"] == "END_TURN":
                    sampled_states = [root]
                for leaf in sampled_states:
                    if (
                        leaf.hp
                        - max(
                            0,
                            sum(
                                leaf.incoming.get(r, 0)
                                for r, (hp, _) in leaf.enemies.items()
                                if hp > 0
                            )
                            - leaf.block,
                        )
                        > 0
                    ):
                        best = max(best, value(leaf) + 1.2 * future.estimate(leaf))
            scores.append(best)
        return scores, expanded, leaves

    @staticmethod
    def _hypothetical_candidates(state, model, roots):
        """Generate modeled branch actions; these refs never leave the search."""
        templates = {
            (c.get("source_refs") or [None])[0]: []
            for c in roots
            if c["verb"] == "PLAY_CARD"
        }
        for c in roots:
            if c["verb"] == "PLAY_CARD":
                templates[(c.get("source_refs") or [None])[0]].append(c)
        options = []
        for ref in sorted(state.hand):
            card = model.refs.get(ref, {})
            if content(card) not in model.cards | {
                "SECOND_WIND",
                "SPITE",
                "BLOODLETTING",
            }:
                continue
            if card.get("enchantment") or card.get("affliction"):
                continue
            if number(card.get("star_cost"), -1) >= 0:
                continue  # Star-resource legality is outside this state model.
            if "cost" not in card:
                continue
            cost = number(card["cost"], -1)
            if cost < 0 or (not card.get("x_cost") and cost > state.energy):
                continue
            if ref in templates:
                options.extend(templates[ref])
                continue
            target_type = card.get("target_type")
            if target_type == "AnyEnemy":
                targets = [
                    [enemy]
                    for enemy, (hp, _) in sorted(state.enemies.items())
                    if hp > 0
                ]
            elif target_type in {"Self", "AnyPlayer"}:
                targets = [["player"]]
            elif target_type in {"None", "AllEnemies"}:
                targets = [[]]
            else:
                continue  # Unknown target constraints need a native observation.
            for target_refs in targets:
                options.append(
                    {
                        "candidate_ref": "hypothetical:"
                        + ref
                        + ":"
                        + ",".join(target_refs),
                        "verb": "PLAY_CARD",
                        "source_refs": [ref],
                        "target_refs": target_refs,
                    }
                )
        options.extend(
            c
            for c in roots
            if c["verb"] == "USE_POTION"
            and (c.get("source_refs") or [None])[0] in state.potions
        )
        return options

    @staticmethod
    def _hand_potential(state, refs, candidates, incoming):
        options, seen = [], set()
        for c in candidates:
            ref = (c.get("source_refs") or [None])[0]
            card = refs.get(ref, {})
            if c["verb"] != "PLAY_CARD" or ref in state.used or ref in seen:
                continue
            seen.add(ref)
            if content(card) not in {"STRIKE_IRONCLAD", "DEFEND_IRONCLAD"}:
                continue
            cost = max(0, number(card.get("cost")))
            if cost > state.energy:
                continue
            damage = stat(card, "Damage")
            gap = max(
                0,
                sum(incoming[r] for r, (hp, _) in state.enemies.items() if hp > 0)
                - state.block,
            )
            benefit = damage + 3 * min(stat(card, "Block"), gap)
            options.append((benefit / max(1, cost), cost, benefit))
        energy, result = state.energy, 0.0
        for _, cost, benefit in sorted(options, reverse=True):
            if cost <= energy:
                energy -= cost
                result += benefit * 0.5
        return result
