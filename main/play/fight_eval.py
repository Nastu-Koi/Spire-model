"""The fight evaluation set: fixed fights, each played from its entry save to its end.

A set (`build_set`) is a few hundred fights the policy walked into, normal, elite and
boss 3:4:4 (`fight_groups.quotas`), drawn evenly within each kind from a directory of
entry saves and copied into the set's own directory: the saves (`SAVES`), the list of
the fights (`MANIFEST`, a selection as `spire_codex_data.entry_battles` reads one, so
that the solver fights the same saves) and the seeds of their runs (`SEEDS`, a seed
file for `iterate --exclude-seeds`). An entry save names its run's character, seed
and ascension beside the save itself, as the solver reads it.

`play_set` plays every fight once by the policy's most likely action (`sample=False`),
in the engines of sampling processes, from its entry save to the first decision after
it or the run's end inside it (`RolloutRunner.run(until_fight_end=True)`), and makes a
row of each (`ROW`): what the fight was, how it went (the fight's record,
`fights.FightLog`), how it was played (`behavior`) and a fingerprint of its actions.
The decisions where the player was in danger (below `DANGER` of their max HP, with a
potion they could use) are scored again after the round (`score_danger`): the
policy's probability of using a potion there, at temperature 1 and at the fight
temperature. `solver_rows` makes the same rows of the solver's fights of the set
(`entry_battles run`), won and lost. What a fight cost is for the learning side to
say (`main/learn/fight_report.py`).
"""

import gzip
import json
import random
import shutil
import zlib
from collections import Counter
from pathlib import Path

import orjson
import torch

from main.env.protocol import CHARACTERS, fingerprint
from main.network.policy import SessionPolicy, choose_batch
from main.play.fight_groups import SHARES, quotas
from main.play.fights import FILE, PLAYS
from main.play.rollout import SEGMENTS, precision_context

FORMAT = "fight-eval-v1"
MANIFEST = "fights.json"  # the set's fights, a selection of `entry_battles`
SEEDS = "seeds.json"  # the seeds of the set's runs by character
SAVES = "saves"  # the entry saves, `<fight id>.json.gz`
SOLVER = "solver"  # where the solver's fights of the set go by default (`entry_battles run --output`)
COUNT = 440
DANGER = 0.3  # a decision below this share of the player's max HP is in danger
# What names a fight of a set.
FIELDS = ("id", "character", "seed", "ascension", "kind", "act", "floor", "encounter", "second_boss")
# A fight's record as a row keeps it (`fights.ROW`, less where it was played).
RECORD = ("entry_hp", "entry_max_hp", "exit_hp", "exit_max_hp", "last_hp", "last_max_hp", "potions_used",
          "opening_selection", "died", "decisions")
# The choices that set a fight up before its first play, by the relic that offers them
# and what the selection looks like (`selection_context`).
SETUP_SOURCES = (
    ("RELIC.GAMBLING_CHIP", lambda c: c.get("source") == "hand" and not c.get("min_total")
     and (c.get("max_total") or 0) > 1),
    ("RELIC.TOASTY_MITTENS", lambda c: c.get("operation") == "exhaust" and c.get("max_total") == 1),
    ("RELIC.TOOLBOX", lambda c: c.get("source") == "unknown" and c.get("max_total") == 1),
)


def read_entry(path):
    """An entry save as its file holds it, gzip-compressed or not."""
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def fight_id(path):
    name = Path(path).name
    return name.removesuffix(".gz").removesuffix(".json")


def entry_fights(directory):
    """The fights of the entry saves in `directory`, a file each named by the fight's
    id: what names each fight (`FIELDS`), whether an event began it, and its file. A
    save that does not name its run's character, seed and ascension, as a round's
    (`main/play/fights.py`), takes them from its wave's records (`FILE` beside
    `directory`), the row of the same file."""
    directory = Path(directory)
    records = directory.parent / FILE
    rows = {}
    if records.is_file():
        for line in records.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("path"):
                rows[Path(row["path"]).name] = row
    found = []
    for path in sorted(directory.glob("*.json*")):
        entry = read_entry(path)
        fight = entry["fight"]
        named = {key: entry.get(key) for key in ("character", "seed", "ascension")}
        if any(value is None for value in named.values()) and path.name in rows:
            named = {key: rows[path.name].get(key) for key in named}
        if any(value is None for value in named.values()):
            raise ValueError(f"{path} does not name its run's character, seed and ascension")
        found.append(dict(id=fight_id(path), character=named["character"], seed=str(named["seed"]),
                          ascension=named["ascension"], kind=fight["kind"], act=fight["act"], floor=fight["floor"],
                          encounter=fight["encounter"], second_boss=bool(fight.get("second_boss")),
                          from_event=bool(fight.get("from_event")), path=str(path)))
    return found


def selected(files):
    """The ids of the fights selections name (`fights` and `extra`)."""
    ids = set()
    for path in files:
        chosen = json.loads(Path(path).read_text(encoding="utf-8"))
        ids |= {fight["id"] for role in ("fights", "extra") for fight in chosen.get(role, ())}
    return ids


def draw(fights, count, root, shares=SHARES):
    """`count` of `fights` shared among the kinds by `shares` (`fight_groups.quotas`),
    each kind's drawn evenly from its fights in the order of their ids, seeded by
    `root` and the kind: the same fights whenever they are drawn from the same ones."""
    chosen = []
    for kind, quota in quotas(count, shares).items():
        pool = sorted((fight for fight in fights if fight["kind"] == kind), key=lambda fight: fight["id"])
        if len(pool) < quota:
            raise ValueError(f"{len(pool)} {kind} fights to draw {quota} from")
        chosen += sorted(random.Random(f"{root}:{kind}").sample(pool, quota), key=lambda fight: fight["id"])
    return chosen


def build_set(sources, output, *, count=COUNT, root, exclude=(), shares=SHARES):
    """A fight evaluation set in `output`: `count` fights drawn (`draw`, shared among
    the kinds by `shares`, seeded by `root`) from the entry saves of the `sources`
    directories, leaving out the fights begun by an event and those the selections
    `exclude` name. Their saves are copied into `SAVES`; the list of them is
    `MANIFEST`, the seeds of their runs `SEEDS`."""
    output = Path(output)
    if (output / MANIFEST).exists():
        raise FileExistsError(f"{output} holds a fight evaluation set already")
    taken = selected(exclude)
    found = [fight for directory in sources for fight in entry_fights(directory)]
    if len({fight["id"] for fight in found}) != len(found):
        raise ValueError("Two entry saves of the sources share a fight id")
    pool = [fight for fight in found if not fight["from_event"] and fight["id"] not in taken]
    chosen = draw(pool, count, root, shares)
    (output / SAVES).mkdir(parents=True, exist_ok=True)
    fights = []
    for fight in chosen:
        name = f"{fight['id']}.json.gz"
        shutil.copyfile(fight["path"], output / SAVES / name)
        fights.append(dict({key: fight[key] for key in FIELDS}, path=f"{SAVES}/{name}"))
    seeds = {character: sorted({fight["seed"] for fight in fights if fight["character"] == character})
             for character in CHARACTERS}
    _write(output / SEEDS, seeds)
    account = dict(sources=[str(directory) for directory in sources], saves=len(found),
                   excluded=dict(selections=[str(path) for path in exclude],
                                 fights=sum(fight["id"] in taken for fight in found),
                                 from_event=sum(fight["from_event"] for fight in found)),
                   pool=dict(sorted(Counter(fight["kind"] for fight in pool).items())))
    # Last: a set without its list is no set.
    _write(output / MANIFEST, dict(format=FORMAT, root=root, count=count, quota=quotas(count, shares), **account,
                                   fights=fights, extra=[]))
    return dict(directory=str(output), fights=len(fights), kinds=dict(Counter(fight["kind"] for fight in fights)),
                seeds={character: len(values) for character, values in seeds.items()}, **account)


def _write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_set(directory):
    """The fights of a set, each with the path of its save, and the seeds of their runs."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"{directory} is not a fight evaluation set of {FORMAT}")
    fights = [dict(fight, path=str(directory / fight["path"])) for fight in manifest["fights"]]
    return fights, json.loads((directory / SEEDS).read_text(encoding="utf-8"))


def jobs(fights):
    """The segments that play the fights: each from its entry save, at the ascension
    the save was written at, to the fight's end; named by the fight's id."""
    return [(fight["character"], dict(seed=fight["seed"], run_id=fight["id"], group=fight["id"], resume=fight["path"],
                                      until_fight_end=True, ascension=fight["ascension"]), None)
            for fight in fights]


def player(public):
    return next((entity for entity in public["entities"] if entity.get("entity_type") == "player"), {})


def content(public, ref):
    """The content id of the entity a reference names in a public state."""
    return next((entity.get("content_id") for entity in public["entities"] if entity.get("ref") == ref), None)


def run_decisions(run):
    """The decisions of a played segment in the order played, each its public state,
    its candidates (`options`), the one taken (`chosen`), its routing and its frame.
    A segment of decisions none of which had a choice is not in the run
    (`RolloutRunner.run`); neither does `behavior` count one."""
    taken = [(step["frame"], step["candidate_ref"]) for macro in run["macros"] for step in macro.get("steps", ())]
    taken += [(action["frame"], action["candidate_ref"]) for action in run.get("environment_actions", ())]
    taken.sort(key=lambda item: item[0]["routing"]["state_version"])
    decisions = []
    for frame, ref in taken:
        options = frame["legal"]["candidates"]
        chosen = next(index for index, option in enumerate(options) if option["candidate_ref"] == ref)
        decisions.append(dict(public=frame["public"], options=options, chosen=chosen, routing=frame["routing"],
                              frame=frame))
    return decisions


def solver_decisions(rows):
    """The decisions of a fight the solver fought, from the rows `entry_battles` wrote
    of it, in `run_decisions`' form (without frames)."""
    return [dict(public=row["observation"], options=row["options"], chosen=row["options"].index(row["label"]),
                 routing=row["metadata"]["routing"]) for row in rows]


def setup_source(public, context):
    """The relic that offers a selection made before a fight's first play, or None."""
    held = {entity.get("content_id") for entity in public["entities"] if entity.get("entity_type") == "relic"}
    return next((relic for relic, offers in SETUP_SOURCES if relic in held and offers(context)), None)


def behavior(decisions):
    """How a fight was played, from its decisions (`run_decisions`, `solver_decisions`):
    - `held`: the content ids of the potions in the belt at its first decision;
    - `used`: those of the potions used, in order;
    - `setups`: the selections made before its first play (a card, a potion or the end
      of a turn), each with the relic that offers it (`SETUP_SOURCES`, None if none
      does), the selection's bounds, how many it picked and the content ids picked;
    - `selections`: the other selections, each with what opened it (`trigger`: the card
      or potion played just before, END_TURN, or START at the fight's start), its
      operation and bounds and how many it picked;
    - `danger`: the decisions with a choice, a potion among it, and the player below
      `DANGER` of their max HP: where in the fight (`index`), the HP, and whether a
      potion was used there.
    Only a selection with at least one decision with a choice counts: a run does not
    keep a selection with none."""
    used, processes, danger = [], [], []
    acted, trigger, process, last = False, "START", None, None
    for index, decision in enumerate(decisions):
        public, options = decision["public"], decision["options"]
        option = options[decision["chosen"]]
        verb = option.get("verb")
        context = public.get("selection_context")
        if context:
            key = (decision.get("routing") or {}).get("selection_id")
            if process is None or key != process["key"] or key is None and last != "SELECT_ONE":
                process = dict(key=key, setup=not acted, trigger=trigger, context=context, public=public, choice=False,
                               picked=[])
                processes.append(process)
            process["choice"] |= len(options) > 1
            if verb == "SELECT_ONE":
                process["picked"].append(content(public, (option.get("source_refs") or [None])[0]))
        else:
            process = None
        source = content(public, (option.get("source_refs") or [None])[0])
        if verb == "USE_POTION":
            used.append(source)
        hero = player(public)
        if (len(options) > 1 and any(other.get("verb") == "USE_POTION" for other in options)
                and type(hero.get("hp")) is int and hero.get("max_hp") and hero["hp"] < DANGER * hero["max_hp"]):
            danger.append(dict(index=index, hp=hero["hp"], max_hp=hero["max_hp"], used=verb == "USE_POTION"))
        if verb in PLAYS:
            acted = True
            trigger = "END_TURN" if verb == "END_TURN" else source or verb
        last = verb
    setups, selections = [], []
    for process in processes:
        if not process["choice"]:
            continue
        context = process["context"]
        bounds = dict(operation=context.get("operation"), source=context.get("source"),
                      min=context.get("min_total"), max=context.get("max_total"))
        if process["setup"]:
            setups.append(dict(relic=setup_source(process["public"], context), **bounds, picks=len(process["picked"]),
                               picked=process["picked"]))
        else:
            selections.append(dict(trigger=process["trigger"], **bounds, picks=len(process["picked"])))
    held = sorted(entity["content_id"] for entity in (decisions[0]["public"]["entities"] if decisions else ())
                  if entity.get("entity_type") == "potion" and entity.get("content_id"))
    return dict(held=held, used=used, setups=setups, selections=selections, danger=danger)


def opening_selection(decisions):
    """Whether a choice among several candidates that is no play (`fights.PLAYS`) was
    made before the fight's first play, as `fights.FightLog` records it of the policy's."""
    for decision in decisions:
        if decision["options"][decision["chosen"]].get("verb") in PLAYS:
            return False
        if len(decision["options"]) > 1:
            return True
    return False


def actions(decisions):
    """A fingerprint of the actions a fight was played with, for telling two plays of it apart."""
    return fingerprint([[decision["routing"].get("state_version"), decision["options"][decision["chosen"]].get(
        "decoder_slot_ref")] for decision in decisions])


def fight_row(fight, run, frames=None):
    """The row of a fight the policy played (`ROW`): `run` as a sampling process hands
    it over (`sampling._handed`: its summary, the shard line of its frames, the record
    of its fight). With `frames` a list, the frames of its danger decisions are added
    to it, compressed, as (fight id, place among the fight's danger decisions, frame)."""
    decisions = run_decisions(orjson.loads(gzip.decompress(run["member"]))) if run.get("member") else []
    records = run.get("fight_records") or []
    found = behavior(decisions)
    if frames is not None:
        for place, entry in enumerate(found["danger"]):
            frames.append((fight["id"], place, zlib.compress(orjson.dumps(decisions[entry["index"]]["frame"]), 1)))
    return dict({key: fight[key] for key in FIELDS}, source="policy", status=run["status"], outcome=run.get("outcome"),
                error=run.get("error"), record={key: records[0].get(key) for key in RECORD} if records else None,
                behavior=found, actions=actions(decisions),
                choices=sum(len(decision["options"]) > 1 for decision in decisions))


# What `fight_row` makes of a fight (`policy` or `solver`), in order.
ROW = (*FIELDS, "source", "status", "outcome", "error", "record", "behavior", "actions", "choices")


def score_danger(model, vocabulary, precision, frames, temperature, batch=64):
    """The policy's probability of using a potion at each of `frames` (decision frames
    with a choice), summed over its USE_POTION candidates: (at temperature 1, at
    `temperature`). The tempered one is the first's distribution raised to 1/T and
    normalized, as the fight temperature draws from it."""
    found = []
    model.eval()
    with torch.no_grad(), precision_context(model, precision):
        for start in range(0, len(frames), batch):
            chunk = frames[start:start + batch]
            widest = max(len(frame["legal"]["candidates"]) for frame in chunk)
            choices = choose_batch([SessionPolicy(model, vocabulary) for _ in chunk], chunk, sample=False,
                                   top_k=widest)
            for frame, choice in zip(chunk, choices):
                odds = {entry["candidate_ref"]: entry["probability"] for entry in choice.ranking}
                potions = {option["candidate_ref"] for option in frame["legal"]["candidates"]
                           if option.get("verb") == "USE_POTION"}
                tempered = {ref: probability ** (1.0 / temperature) for ref, probability in odds.items()}
                total = sum(tempered.values())
                found.append((sum(odds.get(ref, 0.0) for ref in potions),
                              sum(tempered.get(ref, 0.0) for ref in potions) / total if total else 0.0))
    return found


def play_set(samplers, runner, fights, *, temperature):
    """Play every fight of `fights` (`load_set`) once by the policy's most likely action,
    in the engines of `samplers` (`sampling.Samplers`) with `runner`, from its entry
    save to its end; then score its danger decisions (`score_danger`, at temperature 1
    and at `temperature`). Returns the rows (`fight_row`), in the order of `fights`."""
    rows, frames = [None] * len(fights), []

    def arrived(index, run):
        rows[index] = fight_row(fights[index], run, frames)

    samplers.collect(runner, jobs(fights), sample=False, on_run=arrived, layout=SEGMENTS)
    places = {row["id"]: row for row in rows}
    built = [orjson.loads(zlib.decompress(frame)) for _, _, frame in frames]
    for (fight, place, _), (probability, tempered) in zip(
            frames, score_danger(runner.model, runner.vocabulary, runner.precision, built, temperature)):
        places[fight]["behavior"]["danger"][place].update(probability=probability, tempered=tempered)
    return rows


def solver_rows(directory, fights):
    """The rows (`fight_row`'s, `source` "solver") of the solver's fights of a set:
    what `entry_battles run --selection <set>/fights.json --output <directory>` wrote,
    each fight's report (`runs/<run>.json`, won or lost) and its decisions
    (`samples/<run>.jsonl.gz`). Only fights of `fights` count; a fight the solver could
    not play is a row with `status` "error". The report has no max HP after the fight:
    the last decision's stands for it."""
    directory = Path(directory)
    named = {fight["id"]: fight for fight in fights}
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    rows = []
    for run_id in selection["runs"]:
        report = json.loads((directory / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))
        decided = {}
        samples = directory / "samples" / f"{run_id}.jsonl.gz"
        if samples.is_file():
            with gzip.open(samples, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    decided.setdefault(row["metadata"]["sample_group"], []).append(row)
        for fought in report["fights"]:
            fight = named.get(fought["id"])
            if fight is None:
                continue
            base = dict({key: fight[key] for key in FIELDS}, source="solver", seconds=fought.get("seconds"))
            if fought["status"] != "verified":
                rows.append(dict(base, status="error", outcome=None, error=fought.get("error"), record=None,
                                 behavior=None, actions=None, choices=None))
                continue
            decisions = solver_decisions(decided.get(fought["id"], []))
            if len(decisions) != fought["rows"]:
                raise ValueError(f"The solver's decisions of {fought['id']} are missing")
            found = behavior(decisions)
            first = player(decisions[0]["public"]) if decisions else {}
            last = player(decisions[-1]["public"]) if decisions else {}
            died = fought["outcome"] == "loss"
            record = dict(entry_hp=fought["entry_hp"], entry_max_hp=first.get("max_hp"),
                          exit_hp=0 if died else fought["exit_hp"], exit_max_hp=last.get("max_hp"),
                          last_hp=last.get("hp"), last_max_hp=last.get("max_hp"), potions_used=found["used"],
                          opening_selection=opening_selection(decisions), died=died, decisions=len(decisions))
            rows.append(dict(base, status="complete", outcome=fought["outcome"], error=None, record=record,
                             behavior=found, actions=actions(decisions),
                             choices=sum(len(decision["options"]) > 1 for decision in decisions)))
    return rows


def read_rows(path):
    """The rows of a file of them, one JSON object a line."""
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
