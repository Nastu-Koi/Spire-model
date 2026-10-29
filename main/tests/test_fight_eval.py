"""The fight evaluation set (`main.play.fight_eval`) and its report
(`main.learn.fight_report`): the draw of the set and the seeds it keeps apart, how a
fight was played, the solver's fights, the costs by kind, the danger decisions, the
gap to the solver, the table of behavior and two plays set against each other."""

import gzip
import json
import math
from pathlib import Path

import pytest

from main import commands
from main.cli import parser
from main.env.protocol import CHARACTERS
from main.env.seeds import excluded_seeds, round_seeds
from main.learn.costs import FIXED
from main.learn.fight_report import behavior as behavior_table, gap, paired_plays, priced, report, summary, text
from main.network.policy import SessionPolicy, choose_batch
from main.play.fight_eval import (MANIFEST, SAVES, SEEDS, behavior, build_set, draw, jobs, load_set,
                                  opening_selection, play_set, read_rows, score_danger, solver_rows)
from main.play.rollout import RolloutRunner
from main.play.sampling import Samplers
from main.tests.journeys import TEMPERATURES, Journey, checkpoint, learner


# ---- The set ----

def write_entry(directory, name, character, seed, kind, *, room=0, ascension=0, from_event=False, act=3):
    """An entry save as the probe and `entry_battles` read it: the save, the fight, and
    the run's character, seed and ascension beside them."""
    path = Path(directory) / f"{name}.json.gz"
    entry = dict(at="fight", save=dict(seed=seed, room=room, character=character, ascension=ascension),
                 fight=dict(fight=room + 1, kind=kind, act=act, encounter="ENCOUNTER.SYNTHETIC", from_event=from_event,
                            second_boss=False, map_point_type="monster", floor=room + 1),
                 character=character, seed=seed, ascension=ascension)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(entry, stream)
    return path


def saves(directory, per_kind=6):
    """Entry saves of every kind for every character, named `<character>-<seed>-<NNN>`."""
    directory.mkdir(parents=True, exist_ok=True)
    number = 0
    for kind in ("normal", "elite", "boss"):
        for index in range(per_kind):
            character = CHARACTERS[index % len(CHARACTERS)]
            seed = f"SEED{index:02d}{character[:3].upper()}"
            write_entry(directory, f"{character}-{seed}-{number:03d}", character, seed, kind, room=number % 3)
            number += 1
    return directory


def test_the_set_is_drawn_three_four_four_evenly_and_keeps_its_seeds_apart(tmp_path):
    source = saves(tmp_path / "saves")
    write_entry(source, "Ironclad-EVENT-099", "Ironclad", "EVENT", "normal", from_event=True)
    taken = sorted(path.name.removesuffix(".json.gz") for path in source.glob("*.json.gz"))[:2]
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(dict(fights=[dict(id=taken[0])], extra=[dict(id=taken[1])])))
    found = build_set([source], tmp_path / "set", count=11, root="test", exclude=[selection])
    assert found["kinds"] == dict(normal=3, elite=4, boss=4) and found["excluded"]["fights"] == 2
    fights, seeds = load_set(tmp_path / "set")
    ids = [fight["id"] for fight in fights]
    assert len(ids) == 11 and not set(ids) & set(taken) and "Ironclad-EVENT-099" not in ids
    # Every save is the set's own copy, as `entry_battles` resolves it from the list.
    manifest = json.loads((tmp_path / "set" / MANIFEST).read_text())
    assert manifest["extra"] == [] and all(fight["path"] == f"{SAVES}/{fight['id']}.json.gz"
                                           for fight in manifest["fights"])
    assert all(Path(fight["path"]).is_file() and Path(fight["path"]).parent == tmp_path / "set" / SAVES
               for fight in fights)
    # The same draw from the same saves, wherever it is made.
    again = build_set([source], tmp_path / "again", count=11, root="test", exclude=[selection])
    assert again["kinds"] == found["kinds"] and [fight["id"] for fight in load_set(tmp_path / "again")[0]] == ids
    other = build_set([source], tmp_path / "other", count=11, root="other", exclude=[selection])
    assert [fight["id"] for fight in load_set(tmp_path / "other")[0]] != ids and other["fights"] == 11
    # The seeds of the set's runs, by character, as an evaluation's seed file.
    assert seeds == json.loads((tmp_path / "set" / SEEDS).read_text())
    assert {seed for values in seeds.values() for seed in values} == {fight["seed"] for fight in fights}
    assert all(fight["seed"] in seeds[fight["character"]] for fight in fights)
    # Rounds that exclude the seed file are never played on them.
    kept = excluded_seeds([tmp_path / "set" / SEEDS])
    assert kept == {fight["seed"] for fight in fights}
    assert not {seed for values in round_seeds("test", 0, 3, kept).values() for seed in values} & kept
    with pytest.raises(FileExistsError):
        build_set([source], tmp_path / "set", count=11, root="test")
    with pytest.raises(ValueError):
        build_set([source], tmp_path / "short", count=30, root="test")


def test_a_rounds_saves_are_named_by_their_waves_records(tmp_path):
    """A round's entry saves (`main/play/fights.py`) do not name their run: the wave's
    records beside them do, by the save's file."""
    from main.play.fights import FILE

    wave = tmp_path / "act1"
    source = saves(wave / "fights", per_kind=2)
    rows = []
    for path in sorted(source.glob("*.json.gz")):
        entry = json.loads(gzip.decompress(path.read_bytes()))
        rows.append(dict(character=entry.pop("character"), seed=entry.pop("seed"), ascension=entry.pop("ascension"),
                         path=f"elsewhere/{path.name}"))
        path.write_bytes(gzip.compress(json.dumps(entry).encode()))
    (wave / FILE).write_text("".join(json.dumps(row) + "\n" for row in rows))
    found = build_set([source], tmp_path / "set", count=6, root="test", shares=dict(normal=1, elite=1, boss=1))
    assert found["fights"] == 6
    fights, _ = load_set(tmp_path / "set")
    named = {Path(row["path"]).name.removesuffix(".json.gz"): row for row in rows}
    assert all((fight["character"], fight["seed"], fight["ascension"])
               == (named[fight["id"]]["character"], named[fight["id"]]["seed"], named[fight["id"]]["ascension"])
               for fight in fights)
    # Without the records a save that names no run is refused.
    (wave / FILE).unlink()
    with pytest.raises(ValueError, match="does not name"):
        build_set([source], tmp_path / "again", count=6, root="test", shares=dict(normal=1, elite=1, boss=1))


def test_a_kind_is_drawn_evenly_from_its_fights():
    fights = [dict(id=f"f{index:03d}", kind="normal") for index in range(200)]
    counts = [0] * 200
    for root in range(400):
        for fight in draw(fights, 3, root=str(root), shares=dict(normal=1)):
            counts[int(fight["id"][1:])] += 1
    # 1,200 draws over 200 fights: 6 each on average, none far off.
    assert sum(counts) == 1200 and max(counts) < 20 and min(counts) >= 0


def test_a_fight_is_played_from_its_save_to_its_end_at_its_ascension():
    fights = [dict(id="Silent-S-001", character="Silent", seed="S", ascension=4, path="/set/saves/Silent-S-001.json.gz")]
    assert jobs(fights) == [("Silent", dict(seed="S", run_id="Silent-S-001", group="Silent-S-001",
                                            resume="/set/saves/Silent-S-001.json.gz", until_fight_end=True,
                                            ascension=4), None)]


# ---- How a fight was played ----

def public(hp=60, max_hp=80, potions=(), relics=(), cards=(), context=None):
    entities = [dict(ref="player", entity_type="player", hp=hp, max_hp=max_hp)]
    entities += [dict(ref=f"potion:{index}", entity_type="potion", content_id=potion)
                 for index, potion in enumerate(potions)]
    entities += [dict(entity_type="relic", content_id=relic) for relic in relics]
    entities += [dict(ref=f"card:{index}", entity_type="card", content_id=card) for index, card in enumerate(cards)]
    found = dict(phase="card_select" if context else "combat", entities=entities)
    if context:
        found["selection_context"] = context
    return found


def option(verb, ref=None):
    return dict(verb=verb, decoder_slot_ref=f"{verb}:{ref}", source_refs=[ref] if ref else [], target_refs=[])


def step(state, options, chosen, version, selection=None):
    return dict(public=state, options=options, chosen=chosen, routing=dict(state_version=version, selection_id=selection))


CHIP = dict(source="hand", operation="unknown", min_total=0, max_total=999999999, selected_count=0)
DISCARD = dict(source="hand", operation="discard", min_total=1, max_total=1, selected_count=0)
CARDS = ("CARD.STRIKE", "CARD.SURVIVOR", "CARD.DEFEND", "CARD.NEUTRALIZE")
POTIONS = ("POTION.FIRE_POTION", "POTION.BLOCK_POTION")


def fought(chip_picks=0, *, use_at_danger=True, relics=("RELIC.GAMBLING_CHIP",)):
    """A fight's decisions: Gambling Chip at its start (picking `chip_picks` cards), a
    card that opens a discard, a potion used (or not) below 30% HP, the end of a turn
    and a selection with no choice."""
    hand = public(potions=POTIONS, relics=relics, cards=CARDS, context=CHIP)
    decisions = []
    for pick in range(chip_picks):
        decisions.append(step(hand, [option("SELECT_ONE", f"card:{index}") for index in range(pick, 4)]
                              + [option("FINISH_SELECTION")], 0, len(decisions), "chip"))
    decisions.append(step(hand, [option("SELECT_ONE", "card:3"), option("FINISH_SELECTION")], 1, len(decisions), "chip"))
    plays = [option("PLAY_CARD", "card:1"), option("USE_POTION", "potion:0"), option("END_TURN", "player")]
    decisions.append(step(public(potions=POTIONS, cards=CARDS), plays, 0, len(decisions)))
    decisions.append(step(public(potions=POTIONS, cards=CARDS, context=DISCARD),
                          [option("SELECT_ONE", "card:2"), option("SELECT_ONE", "card:3")], 0, len(decisions), "discard"))
    decisions.append(step(public(potions=POTIONS, cards=CARDS, context=dict(DISCARD, selected_count=1)),
                          [option("FINISH_SELECTION")], 0, len(decisions), "discard"))
    low = public(hp=20, potions=POTIONS, cards=CARDS)
    decisions.append(step(low, plays, 1 if use_at_danger else 0, len(decisions)))
    decisions.append(step(public(hp=20, cards=CARDS), [option("PLAY_CARD", "card:0"), option("END_TURN", "player")],
                          1, len(decisions)))
    decisions.append(step(public(hp=30, potions=POTIONS, cards=CARDS), plays, 2, len(decisions)))
    decisions.append(step(public(hp=20, potions=POTIONS, cards=CARDS),
                          [option("PLAY_CARD", "card:0"), option("USE_POTION", "potion:1"), option("END_TURN", "player")],
                          2, len(decisions)))
    decisions.append(step(public(cards=CARDS, context=DISCARD), [option("SELECT_ONE", "card:0")], 0,
                          len(decisions), "forced"))
    return decisions


def test_how_a_fight_was_played_potions_setups_selections_and_danger():
    found = behavior(fought(chip_picks=2))
    assert found["held"] == sorted(POTIONS) and found["used"] == ["POTION.FIRE_POTION"]
    # Gambling Chip at the start: two cards picked, the strike and the survivor.
    assert found["setups"] == [dict(relic="RELIC.GAMBLING_CHIP", operation="unknown", source="hand", min=0,
                                    max=999999999, picks=2, picked=["CARD.STRIKE", "CARD.SURVIVOR"])]
    # The discard the survivor opened; the selection with no choice does not count.
    assert found["selections"] == [dict(trigger="CARD.SURVIVOR", operation="discard", source="hand", min=1, max=1,
                                        picks=1)]
    # Below 30% of max HP with a potion to use: twice, one used. The decision without a
    # potion and the one at 30 HP (37.5%) are not in danger.
    assert [(entry["index"], entry["hp"], entry["used"]) for entry in found["danger"]] == [(6, 20, True), (9, 20, False)]
    kept = behavior(fought(use_at_danger=False))
    assert kept["used"] == [] and [entry["used"] for entry in kept["danger"]] == [False, False]
    assert kept["setups"][0]["picks"] == 0 and kept["setups"][0]["picked"] == []
    # A selection at the start that no relic of the list offers.
    other = behavior(fought(relics=("RELIC.ANCHOR",)))
    assert other["setups"][0]["relic"] is None
    # A choice before the first play sets the fight up, as the policy's record says it.
    decisions = fought()
    assert opening_selection(decisions) is True
    assert opening_selection(decisions[1:]) is False  # begun with a play
    forced = dict(decisions[0], options=[option("FINISH_SELECTION")], chosen=0)
    assert opening_selection([forced, *decisions[1:]]) is False


def test_toasty_mittens_and_toolbox_set_a_fight_up():
    mittens = dict(source="hand", operation="exhaust", min_total=1, max_total=1, selected_count=0)
    toolbox = dict(source="unknown", operation="unknown", min_total=1, max_total=1, selected_count=0)
    held = ("RELIC.TOASTY_MITTENS", "RELIC.TOOLBOX")
    decisions = [
        step(public(relics=held, cards=CARDS, context=mittens), [option("SELECT_ONE", "card:0"),
                                                                 option("SELECT_ONE", "card:2")], 1, 0, "m"),
        step(public(relics=held, cards=CARDS, context=toolbox), [option("SELECT_ONE", "card:3"),
                                                                 option("SELECT_ONE", "card:1")], 0, 1, "t"),
        step(public(relics=held, cards=CARDS), [option("END_TURN", "player"), option("PLAY_CARD", "card:0")], 0, 2)]
    setups = behavior(decisions)["setups"]
    assert [(setup["relic"], setup["picked"]) for setup in setups] == [
        ("RELIC.TOASTY_MITTENS", ["CARD.DEFEND"]), ("RELIC.TOOLBOX", ["CARD.NEUTRALIZE"])]


# ---- The solver's fights ----

def solver_output(directory, fights):
    """What `entry_battles run` writes: the selection, a report per run and the rows of
    every fight played. The first fight is won with a potion, the second lost, the
    third failed."""
    directory.mkdir(parents=True)
    (directory / "selection.json").write_text(json.dumps(dict(runs=["Regent-S"])))
    won = fought(chip_picks=3)
    lost = fought(use_at_danger=False)
    reports = [
        dict(id=fights[0]["id"], status="verified", outcome="win", entry_hp=60, exit_hp=58, rows=len(won), seconds=9.5),
        dict(id=fights[1]["id"], status="verified", outcome="loss", entry_hp=60, exit_hp=None, rows=len(lost),
             seconds=30.0),
        dict(id=fights[2]["id"], status="execution_failed", error="ValueError:solver_failed:x", seconds=1.0)]
    (directory / "runs").mkdir()
    (directory / "runs" / "Regent-S.json").write_text(json.dumps(dict(identity={}, run_id="Regent-S", fights=reports)))
    (directory / "samples").mkdir()
    with gzip.open(directory / "samples" / "Regent-S.jsonl.gz", "wt", encoding="utf-8") as stream:
        for fight, decisions in ((fights[0], won), (fights[1], lost)):
            for decision in decisions:
                stream.write(json.dumps(dict(observation=decision["public"], options=decision["options"],
                                             label=decision["options"][decision["chosen"]], actor="combat_solver",
                                             metadata=dict(sample_group=fight["id"], routing=decision["routing"])))
                             + "\n")
    return directory


def set_fight(name, kind="normal", act=1, **changes):
    return dict(dict(id=name, character="Regent", seed="S", ascension=0, kind=kind, act=act, floor=5,
                     encounter="ENCOUNTER.X", second_boss=False), **changes)


def test_the_solvers_fights_won_and_lost_are_rows_of_their_own(tmp_path):
    fights = [set_fight("Regent-S-001"), set_fight("Regent-S-002", "elite"), set_fight("Regent-S-003", "boss")]
    rows = {row["id"]: row for row in solver_rows(solver_output(tmp_path / "solver", fights), fights)}
    won, lost, failed = (rows[fight["id"]] for fight in fights)
    assert won["source"] == "solver" and won["status"] == "complete" and won["outcome"] == "win"
    assert won["record"] == dict(entry_hp=60, entry_max_hp=80, exit_hp=58, exit_max_hp=80, last_hp=60, last_max_hp=80,
                                 potions_used=["POTION.FIRE_POTION"], opening_selection=True, died=False, decisions=12)
    assert won["behavior"]["setups"][0]["picks"] == 3 and won["seconds"] == 9.5
    assert lost["record"]["died"] is True and lost["record"]["exit_hp"] == 0 and lost["record"]["potions_used"] == []
    assert failed["status"] == "error" and failed["record"] is None and "solver_failed" in failed["error"]
    # Priced as the policy's fights are.
    assert priced(won)["cost"]["total"] == 2 + 7 and priced(lost)["cost"]["death"] == 100


# ---- The report ----

def played(name, kind="normal", *, act=1, entry=60, exit=50, died=False, potions=(), max_hp=(80, 80), source="policy",
           status="complete", actions="a", behaviour=None, **fight):
    record = None if status != "complete" else dict(
        entry_hp=entry, entry_max_hp=max_hp[0], exit_hp=0 if died else exit, exit_max_hp=max_hp[1], last_hp=exit,
        last_max_hp=max_hp[1], potions_used=list(potions), opening_selection=False, died=died, decisions=10)
    return dict(set_fight(name, kind, act, **fight), source=source, status=status, outcome="reached", error=None,
                record=record, actions=actions, choices=5,
                behavior=behaviour or dict(held=[], used=list(potions), setups=[], selections=[], danger=[]))


def test_the_summary_prices_each_fight_and_counts_deaths_potions_and_danger():
    danger = [dict(index=3, hp=10, max_hp=80, used=False, probability=0.1, tempered=0.01),
              dict(index=4, hp=9, max_hp=80, used=True, probability=0.5, tempered=0.6)]
    rows = [
        played("n1", exit=50, potions=["POTION.FIRE_POTION"]),  # 10 + 7
        played("n2", exit=60, max_hp=(80, 83)),  # 0 − 3
        played("e1", "elite", died=True, behaviour=dict(held=[], used=[], setups=[], selections=[], danger=danger)),
        played("b1", "boss", act=1, entry=60, exit=35, potions=["POTION.GAMBLERS_BREW"]),  # 25/5 + 11
        played("b3", "boss", act=3, entry=60, exit=None, last=5, status="error"),
    ]
    found = summary(rows)
    assert found["normal"]["cost"] == pytest.approx(7.0) and found["normal"]["deaths"] == 0
    assert found["normal"]["cost_error"] == pytest.approx(math.sqrt(200) / math.sqrt(2))
    assert found["normal"]["items"] == dict(hp=5.0, max_hp=-1.5, potions=3.5, death=0.0, growth=0.0)
    assert found["normal"]["potions_per_fight"] == 0.5 and found["normal"]["fights_with_potion"] == 1
    assert found["elite"]["cost"] == 160.0 and found["elite"]["death_rate"] == 1.0
    assert found["elite"]["danger"] == dict(decisions=2, fights=1, used=1, use_rate=0.5, probability=0.3,
                                            tempered=0.305)
    assert found["boss"]["cost"] == 16.0 and found["boss"]["fights"] == 2 and found["boss"]["unfinished"] == 1
    assert found["all"]["finished"] == 4 and found["all"]["deaths"] == 1
    assert found["all"]["cost"] == pytest.approx((17 - 3 + 160 + 16) / 4)
    # Other prices, the same fights: the records are what is priced.
    from main.learn.costs import Prices

    assert summary(rows, Prices(tiers=(1, 1, 1)))["normal"]["cost"] == pytest.approx(4.0)


def test_the_deaths_with_a_potion_left_to_drink_and_the_fights_that_used_one():
    def held(*potions):
        return dict(held=list(potions), used=[], setups=[], selections=[], danger=[])

    rows = [
        played("e1", "elite", died=True, behaviour=held("POTION.FIRE_POTION")),  # held one, used none
        played("e2", "elite", died=True, potions=["POTION.BLOCK_POTION"]),  # drank before dying
        played("e3", "elite", died=True, behaviour=held("POTION.FAIRY_IN_A_BOTTLE")),  # a fairy is not drunk
        played("e4", "elite", potions=["POTION.FIRE_POTION"]),
        played("b1", "boss", act=1, died=True, behaviour=held("POTION.SWIFT_POTION")),
        # The run's last fight has no price to fit: its deaths are left out.
        played("b3", "boss", act=3, died=True, behaviour=held("POTION.FIRE_POTION")),
    ]
    found = summary(rows)
    assert found["elite"]["held_at_death"] == dict(deaths=3, holding=1, share=0.333)
    assert found["elite"]["fights_with_potion_share"] == 0.5
    assert found["boss"]["held_at_death"] == dict(deaths=1, holding=1, share=1.0)
    assert found["all"]["held_at_death"] == dict(deaths=4, holding=2, share=0.5)
    assert found["normal"]["held_at_death"] == dict(deaths=0, holding=0, share=None)
    assert "| elite | 4 | 3 |" in text(dict(summary=found)) and "| 1/3 |" in text(dict(summary=found))


def test_a_fight_that_ended_the_run_is_priced_by_its_last_decision_on_both_sides():
    policy = played("b3", "boss", act=3, entry=60, exit=None, last=None)
    policy["record"].update(exit_hp=None, exit_max_hp=None, last_hp=40)
    solver = played("b3", "boss", act=3, entry=60, exit=44, source="solver")
    solver["record"]["last_hp"] = 40
    assert priced(policy)["cost"]["hp"] == priced(solver)["cost"]["hp"] == 20.0


def chip(picks):
    return dict(relic="RELIC.GAMBLING_CHIP", operation="unknown", source="hand", min=0, max=999999999, picks=picks,
                picked=["CARD.STRIKE"] * picks)


def test_the_gap_and_the_table_of_behavior_against_the_solver():
    held = ["POTION.FIRE_POTION"]
    ours = [
        played("n1", exit=50, behaviour=dict(held=held, used=[], setups=[chip(0)], danger=[],
                                             selections=[dict(trigger="CARD.SURVIVOR", operation="discard",
                                                              source="hand", min=1, max=1, picks=1)])),
        played("e1", "elite", died=True, behaviour=dict(held=held, used=[], setups=[chip(0)], selections=[],
                                                        danger=[])),
        played("b1", "boss", exit=40, behaviour=dict(held=[], used=[], setups=[], selections=[], danger=[])),
        played("x1", "normal", status="error"),
    ]
    theirs = [
        played("n1", exit=58, source="solver", potions=["POTION.FIRE_POTION"],
               behaviour=dict(held=held, used=held, setups=[chip(2)], selections=[], danger=[])),
        played("e1", "elite", exit=40, source="solver", behaviour=dict(held=held, used=[], setups=[chip(3)],
                                                                       selections=[], danger=[])),
        played("b1", "boss", source="solver", status="error"),
    ]
    found = gap(ours, theirs)
    assert found["normal"] == dict(fights=1, policy=10.0, solver=9.0, difference=1.0, difference_error=None,
                                   policy_deaths=0, solver_deaths=0, policy_potions=0.0, solver_potions=1.0,
                                   costlier=1, cheaper=0)
    assert found["elite"]["difference"] == 160.0 - 20.0 and found["elite"]["policy_deaths"] == 1
    assert found["boss"]["fights"] == 0 and found["all"]["fights"] == 2
    assert found["unpaired"] == dict(policy_unfinished=1, solver_unfinished=1, solver_missing=1)
    rows = {(row["kind"], row["key"], row["content"]): row for row in behavior_table(ours, theirs)["rows"]}
    # The policy never discards with Gambling Chip; the solver does.
    assert rows["setup", "RELIC.GAMBLING_CHIP", "picked 0"]["policy"] == dict(count=2, of=2, share=1.0)
    assert rows["setup", "RELIC.GAMBLING_CHIP", "picked 0"]["solver"] == dict(count=0, of=2, share=0.0)
    assert rows["setup", "RELIC.GAMBLING_CHIP", "picked 2"]["solver"]["share"] == 0.5
    assert rows["potion", "POTION.FIRE_POTION", "used"]["policy"] == dict(count=0, of=2, share=0.0, uses=0)
    assert rows["potion", "POTION.FIRE_POTION", "used"]["solver"] == dict(count=1, of=2, share=0.5, uses=1)
    assert rows["selection", "CARD.SURVIVOR discard 1-1", "picked 1"]["solver"] == dict(count=0, of=0, share=None)
    # The tables as text, with every kind.
    tables = text(dict(report(ours, theirs), paired=paired_plays(ours, ours)))
    assert "RELIC.GAMBLING_CHIP" in tables and "## policy − solver" in tables and "## paired plays" in tables
    assert report(ours)["summary"]["all"]["finished"] == 3 and "gap" not in report(ours)


def test_two_plays_of_the_set_are_set_against_each_other_fight_by_fight():
    first = [played("n1", exit=50, actions="a"), played("n2", exit=55, actions="b"),
             played("e1", "elite", exit=40, actions="c"), played("b1", "boss", status="error")]
    second = [played("n1", exit=50, actions="a"), played("n2", exit=55, actions="other"),
              played("e1", "elite", died=True, actions="also"), played("b1", "boss", exit=40)]
    found = paired_plays(first, second)
    assert found["normal"] == dict(fights=2, other_actions=1, other_records=0, other_deaths=0, difference=0.0,
                                   difference_error=0.0)
    assert found["elite"]["other_records"] == 1 and found["elite"]["other_deaths"] == 1
    assert found["elite"]["difference"] == 160.0 - 20.0
    assert found["all"]["fights"] == 3 and found["differing"] == ["n2", "e1"]
    assert found["unpaired"] == dict(first=1, second=1)


# ---- Danger: the policy's probability of using a potion ----

def danger_frame():
    """A synthetic decision frame with a potion to use among its candidates."""
    frame = Journey(4, 2, rooms=3).reset("Regent", "S", 0)
    frame["public"]["entities"].append(dict(ref="potion:0", entity_type="potion", content_id="POTION.FIRE_POTION",
                                            slot=0))
    potion = dict(verb="USE_POTION", decoder_slot_ref="potion:0", source_refs=["potion:0"], target_refs=[],
                  effect_coverage="opaque")
    frame["public"]["decoder_bank"].append(potion)
    frame["legal"]["candidates"].append(dict(potion, candidate_ref="c-potion"))
    return frame


def test_the_probability_of_using_a_potion_is_the_policys_at_both_temperatures():
    policy = learner()
    frame = danger_frame()
    (probability, tempered), = score_danger(policy.model, policy.vocabulary, "no", [frame], 0.5)
    ranking = choose_batch([SessionPolicy(policy.model, policy.vocabulary)], [frame], sample=False,
                           top_k=10)[0].ranking
    odds = {entry["candidate_ref"]: entry["probability"] for entry in ranking}
    assert probability == pytest.approx(odds["c-potion"])
    assert tempered == pytest.approx(odds["c-potion"] ** 2 / sum(value ** 2 for value in odds.values()))
    assert score_danger(policy.model, policy.vocabulary, "no", [frame], 1.0)[0][1] == pytest.approx(probability)
    # Several at once score as each alone.
    both = score_danger(policy.model, policy.vocabulary, "no", [frame, danger_frame()], 0.5)
    assert both[1] == pytest.approx((probability, tempered))


# ---- Playing the set ----

class Fights(Journey):
    """Rooms whose HP drops by 5 a room and by 3 more once its fight is over."""

    def frame(self):
        frame = super().frame()
        frame["public"]["entities"][0]["hp"] = 70 - 5 * self.room - (3 if self.stage == "combat_play" else 0)
        return frame


def fights_engine(root=None):
    return Fights(4, 2, rooms=4)


def normal_saves(directory):
    """The entry saves of the normal fights of rooms 0–2 of a run of every character:
    the fights `Fights` can enter again."""
    directory.mkdir(parents=True)
    for index, character in enumerate(CHARACTERS):
        for room in range(3):
            write_entry(directory, f"{character}-S{index}-{room:03d}", character, f"S{index}", "normal", room=room)
    return directory


def runner(policy):
    return RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no", ascension=0,
                         temperatures=TEMPERATURES)


def test_every_fight_is_played_greedily_from_its_save_to_its_end(tmp_path):
    build_set([normal_saves(tmp_path / "saves")], tmp_path / "set", count=15, root="test", shares=dict(normal=1))
    fights = load_set(tmp_path / "set")[0]
    policy = learner()
    with Samplers(fights_engine, policy.model, policy.vocabulary, engines=3, processes=2) as pool:
        rows = play_set(pool, runner(policy), fights, temperature=TEMPERATURES["combat"])
        again = play_set(pool, runner(policy), fights, temperature=TEMPERATURES["combat"])
    assert [row["id"] for row in rows] == [fight["id"] for fight in fights]
    for fight, row in zip(fights, rows):
        room = int(fight["id"][-3:])
        assert row["source"] == "policy" and row["status"] == "complete" and row["outcome"] == "reached"
        # The fight's record: in at the room's HP, out 3 lower at the decision after it.
        assert row["record"]["entry_hp"] == 70 - 5 * room and row["record"]["exit_hp"] == 67 - 5 * room
        assert row["record"]["died"] is False and row["record"]["decisions"] == 3
        # The fight's one selection, at its start: two cards picked, then finished.
        (setup,) = row["behavior"]["setups"]
        assert (setup["relic"], setup["operation"], setup["min"], setup["max"], setup["picks"]) == (
            None, "exhaust", 2, 2, 2)
        assert row["behavior"]["danger"] == [] and row["choices"] == 2
        assert priced(row)["cost"]["total"] == 3.0
    # Greedy and resumed from the same saves: the same actions.
    assert [row["actions"] for row in again] == [row["actions"] for row in rows]


def test_the_commands_draw_a_set_play_it_and_report_it_again(tmp_path, monkeypatch):
    # The set of a mixed directory of saves, by the command.
    args = parser().parse_args(["fight-set", "--sources", str(saves(tmp_path / "mixed")), "--count", "11",
                                "--root", "t", "--output", str(tmp_path / "mixed-set")])
    assert args.exclude == [] and commands.COMMANDS["fight-set"](args, {}) is None
    assert len(load_set(tmp_path / "mixed-set")[0]) == 11
    # A set the synthetic engine can play: normal fights only.
    build_set([normal_saves(tmp_path / "saves")], tmp_path / "set", count=15, root="test", shares=dict(normal=1))
    monkeypatch.setattr(commands, "CliEngine", fights_engine)
    path = checkpoint(tmp_path / "policy")
    first, second = tmp_path / "eval-a", tmp_path / "eval-b"
    for output in (first, second):
        arguments = parser().parse_args(["fight-eval", "--checkpoint", str(path), "--fights", str(tmp_path / "set"),
                                         "--output", str(output), "--workers", "2", "--precision", "fp32"])
        assert arguments.per_kind is None
        commands.COMMANDS["fight-eval"](arguments, {})
    rows = read_rows(first / "fights.jsonl")
    assert len(rows) == 15 and all(row["cost"]["total"] == 3.0 for row in rows)
    found = json.loads((first / "report.json").read_text())
    assert found["summary"]["normal"]["finished"] == 15 and found["solver_dir"] is None and "gap" not in found
    assert json.loads((first / "metadata.json").read_text())["precision"] == "fp32"
    assert "## policy" in (first / "report.txt").read_text()
    with pytest.raises(FileExistsError):
        commands.COMMANDS["fight-eval"](parser().parse_args(
            ["fight-eval", "--checkpoint", str(path), "--fights", str(tmp_path / "set"), "--output", str(first)]), {})
    # A trial on the first fight of each kind.
    from main.learn import costs, potion_prices

    cheap = tmp_path / "cheap.json"
    potion_prices.save(costs.Prices(tiers=(1.0, 1.0, 1.0)), cheap)
    trial = parser().parse_args(["fight-eval", "--checkpoint", str(path), "--fights", str(tmp_path / "set"),
                                 "--output", str(tmp_path / "trial"), "--workers", "1", "--per-kind", "2",
                                 "--prices", str(cheap)])
    commands.COMMANDS["fight-eval"](trial, {})
    assert len(read_rows(tmp_path / "trial" / "fights.jsonl")) == 2
    # Beside the report at the fixed prices, one at the prices asked for.
    again = json.loads((tmp_path / "trial" / "report-prices.json").read_text())
    assert again["prices"] == str(cheap) and again["summary"]["all"]["finished"] == 2 and "gap" not in again
    # Reported again: against the solver's fights of the set, found in the set, and
    # the two plays fight by fight.
    solver_output(tmp_path / "set" / "solver", load_set(tmp_path / "set")[0])
    shown = []
    monkeypatch.setattr(commands, "emit", shown.append)
    commands.COMMANDS["fight-compare"](parser().parse_args(["fight-compare", str(first), str(second)]), {})
    (compared,) = shown
    assert compared["solver_dir"] == str(tmp_path / "set" / "solver")
    assert compared["gap"]["all"]["fights"] == 2 and compared["gap"]["unpaired"]["solver_unfinished"] == 1
    assert compared["paired"]["all"] == dict(fights=15, other_actions=0, other_records=0, other_deaths=0,
                                             difference=0.0, difference_error=0.0)
    assert compared["second"]["all"]["finished"] == 15
    # At other prices, the policy's fights alone: no potion was used, so they cost the same.
    commands.COMMANDS["fight-compare"](parser().parse_args(["fight-compare", str(first), str(second),
                                                            "--prices", str(cheap)]), {})
    priced_again = shown[-1]
    assert "gap" not in priced_again and priced_again["paired"]["all"]["difference"] == 0.0
    assert priced_again["summary"]["all"]["cost"] == compared["summary"]["all"]["cost"]
