"""Experimental deck marginal-value priors; printed magnitudes are not effects.

Coverage ratios describe the current public deck. These heuristic roles and
synergies neither simulate conditional effects nor claim a win probability.
"""

from math import fsum

from .public_effects import number, stat


def _roles(card):
    cost = max(1, number(card.get("cost"), 1))
    return {
        "damage": min(2, max(0, stat(card, "Damage")) / (8 * cost)),
        "block": min(2, max(0, stat(card, "Block")) / (6 * cost)),
        "draw": min(2, max(0, stat(card, "Cards", stat(card, "Draw")) - 1)),
        "growth": min(2, max(0, stat(card, "StrengthPower")) / 2),
    }


class RewardPolicy:
    def __init__(self, frame):
        self.public = frame["public"]
        self.refs = {e["ref"]: e for e in self.public["entities"] if e.get("ref")}
        self.deck = [
            e
            for e in self.public["entities"]
            if e.get("entity_type") == "card" and e.get("zone") == "deck"
        ]

    def _marginal(self, card, deck):
        if card.get("card_type", "").lower() in {"curse", "status"}:
            return -8.0
        roles = _roles(card)
        size = max(1, len(deck))
        coverage = {role: fsum(_roles(c)[role] for c in deck) / size for role in roles}
        desired = {"damage": 0.5, "block": 0.4, "draw": 0.2, "growth": 0.15}
        value = sum(
            amount * max(-1, (desired[role] - coverage[role]) / desired[role]) * 3
            for role, amount in roles.items()
        )
        # Strength helps attack-rich decks; effective draw benefits an existing
        # mix, but repeated copies still dilute a finite hand/energy budget.
        value += roles["growth"] * min(1, coverage["damage"])
        value += roles["damage"] * min(0.75, coverage["growth"])
        value -= sum(c.get("content_id") == card.get("content_id") for c in deck) * 0.65
        value -= 0.5 + max(0, len(deck) - 20) * 0.15
        if not any(roles.values()):
            # Unknown special effects keep a neutral prior and legal candidacy.
            value += 0.5
        return float(value)

    def score(self, candidate):
        verb = candidate["verb"]
        source = self.refs.get((candidate.get("source_refs") or [None])[0], {})
        if verb == "TAKE_CARD_REWARD":
            return self._marginal(source, self.deck)
        if self.public["phase"] == "card_reward" and verb in {
            "SKIP",
            "CHOOSE_REWARD_ALTERNATIVE",
        }:
            return 0.0
        if verb == "SELECT_ONE":
            operation = (
                (self.public.get("selection_context") or {})
                .get("operation", "")
                .lower()
            )
            deck = [c for c in self.deck if c.get("ref") != source.get("ref")]
            if "remove" in operation:
                return -self._marginal(source, deck)
            if "upgrade" in operation:
                if source.get("upgraded"):
                    return -1.0
                # The protocol may not expose a modeled upgrade delta. Use the
                # same deck need direction, without inventing exact new stats.
                return self._marginal(source, deck) + 1.0
        return None
