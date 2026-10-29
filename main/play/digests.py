"""What the advantages of a round of act groups need of each decision, taken as a
segment is handed over (`main/learn/groups.py`).

A wave's digest (`FILE`, beside its shards) holds a line for each segment the wave
wrote: its run id, group, character, outcome and ascension, and for each of its
policy macro-actions what the advantages compare and regress on (`FIELDS`): the
macro-action's place in the segment, its frame as the segments compare it, the
candidates chosen, where the player stands and what they have, whether it was
explored or forked, and the frame's phase (a digest written before it was kept has
none: its decisions read as of no phase). The sampling process takes it from the frames it decodes to
write the segment's shard line anyway (`rollout.packed_line`); the advantages read
it instead of decoding every frame of the round a second time.
"""

import gzip
import json

from main.env.protocol import fingerprint

FILE = "decisions.jsonl.gz"
# A decision's entries in its digest, in order: the state summary's raw numbers, so
# that the features fitted on them (`learn.features.summary`) can change without a
# round sampled again.
FIELDS = ("index", "key", "action", "act", "floor", "hp", "max_hp", "gold", "deck", "relics", "potions",
          "fight", "explored", "forked", "phase")
STATE = ("hp", "max_hp", "gold", "deck", "relics", "potions")
# What the state counts, by entity type: the cards count only in the deck.
STATE_COUNTS = {"card": "deck", "relic": "relics", "potion": "potions"}


def frame_key(frame):
    """What makes two frames the same decision: the public state and the candidates."""
    return fingerprint({"public": frame["public"], "legal": frame["legal"]})


def where(frame):
    """The player's act, floor and the state summary of a frame."""
    player, counts = None, dict.fromkeys(STATE_COUNTS.values(), 0)
    # One pass over the entities: a frame holds a hundred or so, and a round a million frames.
    for entity in frame["public"]["entities"]:
        kind = entity.get("entity_type")
        if kind in STATE_COUNTS:
            if kind != "card" or entity.get("zone") == "deck":
                counts[STATE_COUNTS[kind]] += 1
        elif kind == "player" and player is None:
            player = entity
    if player is None or type(player.get("act")) is not int or type(player.get("floor")) is not int:
        raise ValueError("A decision frame needs a player entity with an act and a floor")
    state = {key: player.get(key) for key in ("hp", "max_hp", "gold")}
    state.update(counts)
    return player["act"], player["floor"], state


def decision(index, macro, frame):
    """The digest of a policy macro-action (`FIELDS`), the `index`-th of its segment,
    `frame` its first step's frame."""
    act, floor, state = where(frame)
    steps = macro["steps"]
    return [index, frame_key(frame), [step["candidate_ref"] for step in steps], act, floor,
            *(state[key] for key in STATE), macro.get("fight") is not None,
            explored_mark(steps), any(bool(step.get("forked")) for step in steps), frame["public"].get("phase")]


def explored_mark(steps):
    """A macro-action's `explored` field: false, true, or the name of who proposed its
    explored candidate (`RolloutRunner.explore_prior`: "prior")."""
    if not any(step.get("explored") for step in steps):
        return False
    return next((step["explored_by"] for step in steps if step.get("explored_by")), True)


def frameless(run):
    """How many of a run's macro-actions kept no frames: the fights of a segment played
    with greedy fights (`RolloutRunner.greedy_fights`), which have no line of their own
    and no advantage."""
    return sum("steps" not in macro for macro in run["macros"])


def line(run, decisions):
    """A segment's line of its wave's digest, as a gzip member: the members of a wave's
    segments, written one after another, read back as one gzip file. `frameless` counts
    the macro-actions it has no line for (`frameless`)."""
    text = json.dumps(dict(run_id=run["run_id"], group=run["group"], character=run["character"],
                           outcome=run["outcome"], ascension=run.get("ascension"), decisions=decisions,
                           frameless=frameless(run)),
                      ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return gzip.compress((text + "\n").encode("utf-8"), compresslevel=6, mtime=0)


def entries(path):
    """The lines of a wave's digest, one at a time. A segment written twice (a wave gone
    on with after it was cut off) has two lines: the later is the one that stands."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        yield from map(json.loads, stream)
