"""Bounded next-turn samples from public unordered card piles.

The next enemy intent is approximated by repeating the currently visible intent.
Only modeled cards are played, at most three in one sampled next turn. No native
deck order, game seed, or engine random state is consulted. Enemy block is
carried as a conservative prior; unknown defensive intents are not simulated.
"""

import hashlib
import json
import random
from copy import deepcopy

from .public_effects import content, number

# Native printed costs for the narrow next-turn rollout set. A different
# current cost may be a temporary discount whose expiry is not observable here.
STABLE_COSTS = {
    "STRIKE_IRONCLAD": 1,
    "DEFEND_IRONCLAD": 1,
    "INFLAME": 1,
    "BLUDGEON": 3,
    "BASH": 2,
}


class FutureSampler:
    def __init__(self, model, seed, samples):
        self.model = model
        self.seed = seed
        self.samples = samples
        self.draw_pile = frozenset(
            e["ref"]
            for e in model.entities
            if e.get("entity_type") == "card"
            and e.get("zone") == "draw_pile"
            and e.get("ref")
        )
        self.pile_boundary = any(
            e.get("entity_type") == "pile_summary"
            and e.get("zone") in {"draw_pile", "discard_pile"}
            and (
                e.get("order_known") is True
                or number(e.get("count"), -1)
                != sum(
                    card.get("entity_type") == "card"
                    and card.get("zone") == e.get("zone")
                    for card in model.entities
                )
            )
            for e in model.entities
        )
        self.max_energy = number(
            next(
                (
                    e.get("max_energy")
                    for e in model.entities
                    if e.get("entity_type") == "player"
                ),
                3,
            ),
            3,
        )

    def _key(self, ref):
        card = self.model.refs[ref]
        fields = {
            k: card.get(k)
            for k in (
                "content_id",
                "cost",
                "base_cost",
                "star_cost",
                "enchantment",
                "affliction",
                "upgraded",
                "upgrade_level",
                "retain",
                "x_cost",
                "card_type",
                "target_type",
                "stats",
                "keywords",
                "exhaust_on_next_play",
            )
        }
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))

    def draws(self, state):
        """Draw up to five cards, reshuffling the branch's discard if needed."""
        available = self.draw_pile - state.exhaust - state.hand - state.discard
        draw = sorted(available, key=lambda ref: (self._key(ref), ref))
        discard = sorted(
            (state.discard | state.hand) - state.exhaust,
            key=lambda ref: (self._key(ref), ref),
        )
        # Cards played this turn already enter state.discard; retained hand is
        # assumed discarded here. Retain and start-of-turn hooks are unsupported.
        signature = json.dumps(
            ([self._key(r) for r in draw], [self._key(r) for r in discard]),
            separators=(",", ":"),
        ).encode()
        salt = int.from_bytes(hashlib.sha256(signature).digest()[:8], "big")
        result = []
        for index in range(self.samples):
            rng = random.Random(self.seed ^ salt ^ index)
            first = draw.copy()
            rng.shuffle(first)
            chosen = first[:5]
            if len(chosen) < 5:
                second = discard.copy()
                rng.shuffle(second)
                chosen.extend(second[: 5 - len(chosen)])
            result.append(tuple(chosen))
        return result

    def estimate(self, state):
        if not self.samples or self.pile_boundary:
            return 0.0
        if self.model.unsupported_hooks:
            return 0.0
        if self.model.amount("player", "SETUP_STRIKE") or any(
            content(self.model.refs.get(ref, {})) == "SETUP_STRIKE"
            for ref in state.used
        ):
            return 0.0
        if any(
            self.model.refs.get(ref, {}).get("retain")
            or "ethereal"
            in {
                str(k).lower() for k in self.model.refs.get(ref, {}).get("keywords", [])
            }
            for ref in state.hand
        ):
            return 0.0
        if self.model.amount("player", "WEAK") or self.model.amount("player", "FRAIL"):
            return 0.0
        threat = sum(
            state.incoming.get(ref, 0)
            for ref, (hp, _) in state.enemies.items()
            if hp > 0
        )
        remaining = state.hp - max(0, threat - state.block)
        if remaining <= 0 or all(hp <= 0 for hp, _ in state.enemies.values()):
            return 0.0
        total = 0.0
        for drawn in self.draws(state):
            nxt = deepcopy(state)
            nxt.hp = remaining
            nxt.block = 0
            nxt.energy = self.max_energy
            nxt.hand = frozenset(drawn)
            nxt.used = frozenset()
            nxt.discard = (state.discard | state.hand) - frozenset(drawn)
            nxt.lost_hp_this_turn = False
            nxt.vulnerable = {
                ref: max(0, amount - 1) for ref, amount in state.vulnerable.items()
            }
            nxt.weak = {ref: max(0, amount - 1) for ref, amount in state.weak.items()}
            nxt.incoming = dict(self.model.incoming)
            start = deepcopy(nxt)
            for _ in range(3):
                best = None
                for ref in sorted(nxt.hand, key=lambda r: (self._key(r), r)):
                    card = self.model.refs.get(ref, {})
                    name = content(card)
                    if name not in STABLE_COSTS:
                        continue
                    if card.get("enchantment") or card.get("affliction"):
                        continue
                    if number(card.get("star_cost"), -1) >= 0:
                        continue
                    if "base_cost" in card and number(card["base_cost"], -1) != number(
                        card.get("cost"), -1
                    ):
                        continue  # Current discount need not survive turn boundary.
                    cost = number(card.get("cost"), -1)
                    if cost != STABLE_COSTS[name] or cost > nxt.energy:
                        continue
                    kind = card.get("target_type")
                    if kind == "AnyEnemy":
                        targets = [
                            [r] for r, (hp, _) in sorted(nxt.enemies.items()) if hp > 0
                        ]
                    elif kind in {"Self", "AnyPlayer"}:
                        targets = [["player"]]
                    elif kind in {"None", "AllEnemies"}:
                        targets = [[]]
                    else:
                        continue
                    for target in targets:
                        prediction = self.model.transition(
                            nxt,
                            {
                                "verb": "PLAY_CARD",
                                "source_refs": [ref],
                                "target_refs": target,
                            },
                        )
                        if prediction.state is None or prediction.reason is not None:
                            continue
                        score = self._benefit(start, prediction.state)
                        if best is None or score > best[0]:
                            best = (score, prediction.state)
                if best is None or best[0] <= self._benefit(start, nxt):
                    break
                nxt = best[1]
            total += self._benefit(start, nxt)
        return total / self.samples

    @staticmethod
    def _benefit(start, state):
        dealt = sum(
            before[0] - after[0]
            for ref, before in start.enemies.items()
            if (after := state.enemies[ref])[0] >= 0
        )
        threat = sum(
            state.incoming.get(ref, 0)
            for ref, (hp, _) in state.enemies.items()
            if hp > 0
        )
        saved = min(state.block, threat)
        return dealt + saved * 2 + (state.strength - start.strength) * 3
