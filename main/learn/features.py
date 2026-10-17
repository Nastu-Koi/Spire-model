"""The state summary of a decision frame: a few numbers a baseline is fitted on.

Where the player stands (act, floor), who they are, and the player's state as the
frame shows it: HP, maximum HP, gold, how many cards the deck holds, how many relics
and potions. The group baselines (`main/learn/groups.py`) regress on it.
"""

from main.env.protocol import CHARACTERS

STATE_COUNTS = {"deck": lambda e: e.get("entity_type") == "card" and e.get("zone") == "deck",
                "relics": lambda e: e.get("entity_type") == "relic",
                "potions": lambda e: e.get("entity_type") == "potion"}
ACTS = (1, 2, 3)
NAMES = ([f"act{a}" for a in ACTS] + ["floor"] + [f"act{a}_floor" for a in ACTS] + list(CHARACTERS)
         + ["hp_ratio", "max_hp", "gold", "deck", "relics", "potions", "in_fight"])


def where(frame):
    """The player's act, floor and the state summary of a frame."""
    entities = frame["public"]["entities"]
    player = next((e for e in entities if e.get("entity_type") == "player"), None)
    if player is None or type(player.get("act")) is not int or type(player.get("floor")) is not int:
        raise ValueError("A decision frame needs a player entity with an act and a floor")
    state = {key: player.get(key) for key in ("hp", "max_hp", "gold")}
    state.update({name: sum(1 for e in entities if test(e)) for name, test in STATE_COUNTS.items()})
    return player["act"], player["floor"], state


def summary(character, act, floor, state, in_fight=False):
    """The features of a decision, in the order of `NAMES`, each of the order of one."""
    acts = [float(act == a) for a in ACTS]
    scaled = floor / 50.0
    max_hp = max(1.0, float(state.get("max_hp") or 0))
    return acts + [scaled] + [a * scaled for a in acts] + [float(character == c) for c in CHARACTERS] + [
        float(state.get("hp") or 0) / max_hp, max_hp / 100.0, float(state.get("gold") or 0) / 100.0,
        (state.get("deck") or 0) / 20.0, (state.get("relics") or 0) / 5.0, (state.get("potions") or 0) / 3.0,
        float(in_fight)]
