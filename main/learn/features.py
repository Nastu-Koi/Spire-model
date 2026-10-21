"""The state summary of a decision frame: a few numbers a baseline is fitted on.

Where the player stands (act, floor), who they are, and the player's state as the
frame shows it: HP, maximum HP, gold, how many cards the deck holds, how many relics
and potions. The group baselines (`main/learn/groups.py`) regress on it. What a frame
shows of them is read where the frames are, as segments are handed over
(`play.digests.where`).
"""

from main.env.protocol import CHARACTERS

ACTS = (1, 2, 3)
NAMES = ([f"act{a}" for a in ACTS] + ["floor"] + [f"act{a}_floor" for a in ACTS] + list(CHARACTERS)
         + ["hp_ratio", "max_hp", "gold", "deck", "relics", "potions", "in_fight"])


def summary(character, act, floor, state, in_fight=False):
    """The features of a decision, in the order of `NAMES`, each of the order of one."""
    acts = [float(act == a) for a in ACTS]
    scaled = floor / 50.0
    max_hp = max(1.0, float(state.get("max_hp") or 0))
    return acts + [scaled] + [a * scaled for a in acts] + [float(character == c) for c in CHARACTERS] + [
        float(state.get("hp") or 0) / max_hp, max_hp / 100.0, float(state.get("gold") or 0) / 100.0,
        (state.get("deck") or 0) / 20.0, (state.get("relics") or 0) / 5.0, (state.get("potions") or 0) / 3.0,
        float(in_fight)]
