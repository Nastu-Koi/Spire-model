"""Experimental public route and supply priors, never native future simulation.

Room losses/growth and potion protection are deliberately coarse estimates.
Only visible topology and current resources enter the route calculation; shops
on the map provide an opportunity prior, not assumed inventory or healing.
"""

from functools import cache

from .astar import card_value, player
from .public_effects import content, number


class ResourcePolicy:
    def __init__(self, frame):
        self.public = frame["public"]
        self.entities = self.public["entities"]
        self.refs = {e["ref"]: e for e in self.entities if e.get("ref")}
        self.player = player(frame)
        self.max_hp = max(1, number(self.player.get("max_hp")))
        self.hp = number(self.player.get("hp"))
        self.gold = max(0, number(self.player.get("gold")))
        self.potions = sum(e.get("entity_type") == "potion" for e in self.entities)
        self.deck = [
            e
            for e in self.entities
            if e.get("entity_type") == "card" and e.get("zone") == "deck"
        ]
        self.nodes = {
            ref: e for ref, e in self.refs.items() if e.get("entity_type") == "map_node"
        }
        self.children = {ref: set() for ref in self.nodes}
        for relation in self.public.get("relations", []):
            if (
                relation.get("role") == "map_edge"
                and relation.get("source") in self.nodes
                and relation.get("target") in self.nodes
            ):
                self.children[relation["source"]].add(relation["target"])

    def score(self, candidate):
        """Return None outside this policy's scope to preserve the frozen baseline."""
        source = self.refs.get((candidate.get("source_refs") or [None])[0], {})
        if candidate["verb"] == "MOVE_TO_NODE":
            return self._route(source.get("ref"))
        if candidate["verb"] == "CHOOSE_REST_OPTION":
            kind = content(source)
            if kind in {"HEAL", "REST"}:
                healed = min(self.max_hp - self.hp, int(self.max_hp * 0.3))
                return healed * (0.6 if self.hp < self.max_hp * 0.5 else 0.15)
            if kind == "SMITH":
                return 4.0
        if candidate["verb"] == "BUY_ITEM":
            return self._purchase(source)
        if candidate["verb"] == "LEAVE_ROOM" and self.public["phase"] == "shop":
            return 0.0
        return None

    def _purchase(self, source):
        price = number(source.get("price"))
        ident = source.get("content_id", "").upper()
        if ident.startswith("CARD."):
            offered = next(
                (
                    self.refs.get(r.get("target"), {})
                    for r in self.public.get("relations", [])
                    if r.get("role") == "offers"
                    and r.get("source") == source.get("ref")
                ),
                source,
            )
            benefit = max(-4, card_value(offered, self.deck))
        elif ident.startswith("RELIC."):
            benefit = 5.0  # opaque relic opportunity, not simulated effects
        elif ident.startswith("POTION."):
            benefit = 5.0 if self.hp < self.max_hp * 0.5 else 2.0
            benefit -= self.potions
        elif "REMOVAL" in ident:
            benefit = max((-card_value(c, self.deck) for c in self.deck), default=0)
        else:
            benefit = 0.0
        # Saving gold has opportunity value; no unseen future stock is assumed.
        return benefit - price / 25 - (1 if self.gold - price < 50 else 0)

    def _route(self, start):
        if start not in self.nodes:
            return 0.0
        visiting = set()

        @cache
        def visit(ref, hp, potions):
            if ref in visiting:
                raise ValueError("Public map contains a cycle")
            visiting.add(ref)
            kind = content(self.nodes[ref])
            if kind not in {
                "MONSTER",
                "ELITE",
                "BOSS",
                "UNKNOWN",
                "REST",
                "RESTSITE",
                "SHOP",
                "MERCHANT",
                "TREASURE",
                "ANCIENT",
            }:
                kind = "UNKNOWN"
            loss = {"MONSTER": 8, "ELITE": 18, "BOSS": 24, "UNKNOWN": 5}.get(kind, 0)
            growth = {"MONSTER": 3, "ELITE": 12, "TREASURE": 7, "UNKNOWN": 2}.get(
                kind, 0
            )
            if loss and potions and hp - loss < self.max_hp * 0.4:
                loss = max(0, loss - 6)
                potions -= 1
            remaining = hp - loss
            value = growth - 0.15 * loss
            if remaining <= 0:
                # Stop here: a future rest cannot rescue a lethal prefix.
                value -= 200
            else:
                value -= 0.5 * max(0, self.max_hp * 0.4 - remaining)
                if kind in {"REST", "RESTSITE"}:
                    if remaining < self.max_hp * 0.65:
                        healing = min(self.max_hp - remaining, int(self.max_hp * 0.3))
                        remaining += healing
                        value += healing * 0.25
                    else:
                        value += 4  # possible upgrade, not a known future card
                elif kind in {"SHOP", "MERCHANT"}:
                    value += min(5, self.gold / 40) - 1
                value += 0.9 * max(
                    (
                        visit(child, int(remaining), potions)
                        for child in sorted(self.children[ref])
                    ),
                    default=0,
                )
            visiting.remove(ref)
            return value

        return float(visit(start, int(self.hp), self.potions))
