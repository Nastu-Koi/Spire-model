"""Each fight a segment plays: its record and its entry save (`main.play.fights`), kept
by the run (`RolloutRunner.run(fight_entries=True)`) and written by every wave of a
round (`sample_groups(fight_entries=True)`)."""

import json
from pathlib import Path

import pytest

from main.cli import parser
from main.data.shards import RunShards
from main.env.protocol import CHARACTERS
from main.env.saves import FightTracker
from main.play.fights import FILE, ROW, SAVES, FightLog, fight_rows, state
from main.play.groups import groups_summary, read_save, sample_groups
from main.play.rollout import RolloutRunner, play, stand
from main.play.sampling import Samplers
from main.tests.journeys import TEMPERATURES, Journey, learner
from main.tests.test_iterate import driver, settings, world  # noqa: F401  (`world` is a fixture)

ROOMS, ACTS = 6, 2  # three acts of two rooms


class Saving:
    """An engine that only writes continue saves, each naming how many it wrote."""

    def __init__(self):
        self.saves = 0

    def send(self, command):
        assert command["cmd"] == "write_continue_save"
        self.saves += 1
        Path(command["path"]).write_text(json.dumps({"number": self.saves}))
        return {"type": "save_result", "success": True}


def frame(hp, max_hp=80, *, potions=(), candidates=None, cards=(), relics=0, rows=0):
    entities = [dict(ref="player", entity_type="player", hp=hp, max_hp=max_hp, act=1, floor=3, capacity=3)]
    entities += [dict(ref=ref, entity_type="potion", content_id=content, owner_ref="player", slot=slot)
                 for slot, (ref, content) in enumerate(potions)]
    entities += [dict(ref=f"card:{i}", entity_type="card", zone=zone, card_type=kind, upgraded=upgraded)
                 for i, (zone, kind, upgraded) in enumerate(cards)]
    entities += [dict(ref=f"relic:{i}", entity_type="relic") for i in range(relics)]
    entities += [dict(ref=f"node:{row}", entity_type="map_node", floor=row) for row in range(rows)]
    if candidates is None:
        candidates = [dict(verb="END_TURN", candidate_ref="c0", source_refs=[])]
    return dict(public=dict(phase="combat", entities=entities), legal=dict(candidates=candidates))


def started(number, kind="normal", from_event=False):
    return dict(type="encounter_started", encounter_id=f"e{number}", encounter=f"ENCOUNTER.{number}", act=1,
                kind=kind, from_event=from_event)


def won(number):
    return dict(type="encounter_completed", encounter_id=f"e{number}", result="victory")


def test_a_fights_record_runs_from_its_first_decision_to_the_first_after_it():
    engine, tracker, log = Saving(), FightTracker(), FightLog()

    def decide(seen, events=(), choice="c0", floor=3):
        tracker.events(list(events))
        fight, entry = tracker.decision()
        log.passed(seen, fight)
        log.entered(engine, seen, fight, entry, floor)
        log.chose(seen, fight, choice)

    decide(frame(60))  # outside any fight
    decide(frame(50, potions=[("potion:0", "POTION.FIRE")]), [started(1)])
    use = [dict(verb="USE_POTION", candidate_ref="c1", source_refs=["potion:0"]),
           dict(verb="END_TURN", candidate_ref="c0", source_refs=[])]
    decide(frame(44, potions=[("potion:0", "POTION.FIRE")], candidates=use), choice="c1")
    decide(frame(40))
    decide(frame(45, 82), [won(1)])  # the first decision after the fight: the relics' healing counts
    decide(frame(45, 82), [started(2, "elite", from_event=True)], floor=4)
    decide(frame(30, 82))
    log.ended(dict(public=dict(phase="terminal", outcome=dict(victory=False))), False)

    first, second = log.records(dict(run_id="r", group="g", character="Silent", seed="S", ascension=2))
    assert first["save"] == dict(at="fight", save={"number": 1}, fight=dict(
        fight=1, encounter_id="e1", encounter="ENCOUNTER.1", kind="normal", act=1, from_event=False,
        map_point_type="monster", second_boss=False, floor=3))
    assert {key: first[key] for key in ROW if key != "path"} == dict(
        fight=1, kind="normal", act=1, floor=3, encounter="ENCOUNTER.1", second_boss=False, from_event=False,
        entry_hp=50, entry_max_hp=80, exit_hp=45, exit_max_hp=82, last_hp=40, last_max_hp=80,
        potions_used=["POTION.FIRE"], opening_selection=False, died=False, decisions=3,
        entry_state=dict(act=1, floor=3, act_floors=None, hp=50, max_hp=80, potions=["POTION.FIRE"], capacity=3,
                         deck=dict(cards=0, attacks=0, skills=0, powers=0, curses=0, statuses=0, upgraded=0),
                         relics=0),
        growth={}, relic_counters={}, run_id="r", group="g", character="Silent", seed="S", ascension=2)
    # A fight an event began has no entry save; died in it, the run's terminal frame has
    # no player: no HP left, the max HP of its last decision.
    assert second["save"] is None and engine.saves == 1
    assert (second["from_event"], second["kind"], second["floor"], second["decisions"]) == (True, "elite", 4, 2)
    assert (second["died"], second["exit_hp"], second["exit_max_hp"], second["last_hp"]) == (True, 0, 82, 30)


def test_the_state_a_fight_is_entered_in_counts_the_belt_the_deck_and_the_relics():
    """What a potion's price may depend on: the deck counts only the deck's cards (not
    the hand's or the piles'), by the engine's card types; the belt in slot order; the
    act's floors are its map's rows and one."""
    cards = [("deck", "Attack", True), ("deck", "Attack", False), ("deck", "Skill", False), ("deck", "Power", True),
             ("deck", "Curse", False), ("deck", "Status", False), ("hand", "Attack", True), ("draw_pile", "Skill", False)]
    seen = frame(41, 70, potions=[("potion:0", "POTION.FIRE"), ("potion:1", "POTION.BLOCK")], cards=cards, relics=4,
                 rows=17)
    seen["public"]["entities"].append(dict(ref="potion:x", entity_type="potion", content_id="POTION.SHOP", owner_ref="shop"))
    assert state(seen) == dict(act=1, floor=3, act_floors=17, hp=41, max_hp=70, potions=["POTION.FIRE", "POTION.BLOCK"],
                               capacity=3, deck=dict(cards=6, attacks=2, skills=1, powers=1, curses=1, statuses=1,
                                                     upgraded=2), relics=4)
    assert state(dict(public=dict(entities=[]))) is None


def test_a_fight_set_up_by_a_selection_before_its_first_play():
    """A choice among several that is no play, before the fight's first play (a gambling
    chip's discard, a card of several to keep): the fight is set up by a selection. One
    after the first play, or one with nothing to choose between, is not."""
    select = [dict(verb="SELECT_ONE", candidate_ref="c1", source_refs=[]),
              dict(verb="FINISH_SELECTION", candidate_ref="c0", source_refs=[])]
    finish = [dict(verb="FINISH_SELECTION", candidate_ref="c0", source_refs=[])]

    def opened(*decisions):
        engine, tracker, log = Saving(), FightTracker(), FightLog()
        tracker.events([started(1)])
        for number, (candidates, choice) in enumerate(decisions):
            fight, entry = tracker.decision()
            seen = frame(50, candidates=candidates)
            log.passed(seen, fight)
            log.entered(engine, seen, fight, entry, 3)
            log.chose(seen, fight, choice)
        (record,) = log.records({})
        assert "played" not in record
        return record["opening_selection"]

    assert opened((select, "c1"), (select, "c0"), (None, "c0")) is True
    assert opened((finish, "c0"), (None, "c0")) is False
    assert opened((None, "c0"), (select, "c1")) is False


def test_the_last_boss_won_ends_its_record_with_what_the_terminal_frame_says():
    engine, tracker, log = Saving(), FightTracker(), FightLog()
    tracker.events([started(1, "boss")])
    fight, entry = tracker.decision()
    log.passed(frame(50), fight)
    log.entered(engine, frame(50), fight, entry, 17)
    tracker.events([won(1)])
    log.ended(dict(public=dict(phase="terminal", outcome=dict(victory=True))), True)
    (record,) = log.records({})
    assert (record["died"], record["exit_hp"], record["exit_max_hp"], record["last_hp"]) == (False, None, None, 50)
    # A potion the candidate names but the frame lacks is still counted, unnamed.
    other = FightLog()
    tracker = FightTracker()
    tracker.events([started(1)])
    fight, entry = tracker.decision()
    other.entered(engine, frame(50), fight, entry, 3)
    other.chose(frame(50, candidates=[dict(verb="USE_POTION", candidate_ref="c0", source_refs=["potion:9"])]),
                fight, "c0")
    assert other.records({})[0]["potions_used"] == [None]


class Evented(Journey):
    """Rooms whose second fight an event began."""

    def enter_room(self):
        super().enter_room()
        if self.room == 1:
            self.events[0]["from_event"] = True


class Doomed(Journey):
    """The last room's fight is lost: once its selection is made, the run ends in defeat."""

    def send(self, command):
        reply = super().send(command)
        if self.stage == "combat_play" and self.room == self.rooms - 1:
            self.stage, self.won = "terminal", False
            self.events = [dict(type="run_completed", victory=False, act=3, final_boss_defeated=False)]
            return self.frame()
        return reply


def runner(policy):
    return RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                         ascension=0, temperatures=TEMPERATURES, iteration=0)


def test_a_run_keeps_its_fights_records_and_plays_as_it_does_without():
    policy = learner()
    played = runner(policy)
    kept = played.run(Evented(4, 2, rooms=ROOMS, acts=ACTS), "Regent", "seed", sample=False, fight_entries=True,
                      run_id="r", group="g")
    plain = played.run(Evented(4, 2, rooms=ROOMS, acts=ACTS), "Regent", "seed", sample=False, run_id="r", group="g")
    records = kept.pop("fight_records")
    assert kept == plain and "fight_records" not in plain
    assert [record["fight"] for record in records] == list(range(1, ROOMS + 1))
    for room, record in enumerate(records):
        assert set(record) == {*ROW, "save"} - {"path"}
        # Each fight is the room's selection: two choices and the forced finish; the
        # decision after it, outside the fight, closes the record.
        assert (record["act"], record["floor"], record["kind"]) == (
            room // ACTS + 1, room % ACTS + 1, "boss" if (room + 1) % ACTS == 0 else "normal")
        assert (record["entry_hp"], record["exit_hp"], record["decisions"], record["died"]) == (70, 70, 3, False)
        assert (record["run_id"], record["group"], record["character"], record["seed"], record["ascension"]) == (
            "r", "g", "Regent", "seed", 0)
        assert (record["save"] is None) == (room == 1) and record["from_event"] == (room == 1)
    # An entry save enters its fight again: the room, on the recorded floor.
    third = records[2]["save"]
    assert third["at"] == "fight" and third["fight"]["floor"] == records[2]["floor"]
    again = played.run(Journey(4, 2, rooms=ROOMS, acts=ACTS), "Regent", "seed", resume=third, until_act=3,
                       run_id="r:f3", group="r:f3", fight_entries=True)
    first = again["fight_records"][0]
    assert again["outcome"] == "reached" and (first["act"], first["floor"], first["kind"], first["encounter"]) == (
        records[2]["act"], records[2]["floor"], records[2]["kind"], records[2]["encounter"])
    frames = [step["frame"] for macro in again["macros"] for step in macro["steps"]]
    assert stand(frames[0])[:2] == (2, 1)
    # Stopped at an act with its records kept, a segment says where it stood there.
    assert again["cut_state"]["act"] == 3 and again["cut_state"]["hp"] == 70
    # Without entry saves the records are kept alone.
    alone = played.run(Journey(4, 2, rooms=ROOMS, acts=ACTS), "Regent", "seed", resume=third, until_act=3,
                       run_id="r:f3", group="r:f3", fight_entries=True, entry_saves=False)
    assert [record["save"] for record in alone["fight_records"]] == [None] * len(again["fight_records"])
    assert [record["entry_hp"] for record in alone["fight_records"]] == [r["entry_hp"] for r in again["fight_records"]]
    assert "cut_state" not in played.run(Journey(4, 2, rooms=ROOMS, acts=ACTS), "Regent", "seed", resume=third,
                                         until_act=3, run_id="r:f3", group="r:f3")
    # Lost in its last fight, a run's record says so.
    lost = played.run(Doomed(4, 2, rooms=3), "Regent", "seed", fight_entries=True)
    assert lost["status"] == "complete" and lost["victory"] is False
    assert [record["died"] for record in lost["fight_records"]] == [False, False, True]
    assert (lost["fight_records"][-1]["exit_hp"], lost["fight_records"][-1]["exit_max_hp"]) == (0, 80)
    # Only the segments of act groups keep them.
    with pytest.raises(ValueError, match="segments"):
        play(Journey(4, 2), played, ("Regent", "seed", Path("x.json")), fight_entries=True)


def chain():
    return Evented(4, 2, rooms=ROOMS, acts=ACTS)


def test_a_round_with_fight_entries_writes_each_waves_fights_and_their_saves(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    with Samplers(chain, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        summary = sample_groups(pool, runner(policy), seeds, tmp_path / "kept", n=2, m=[1, 1], fight_entries=True)
        plain = sample_groups(pool, runner(policy), seeds, tmp_path / "plain", n=2, m=[1, 1])
    assert summary["fight_entries"] is True and "fight_entries" not in plain
    assert groups_summary(tmp_path / "kept") == summary
    for wave in summary["waves"]:
        directory = tmp_path / "kept" / wave
        rows = fight_rows(tmp_path / "kept", wave)
        written = summary[wave]["written"]
        # Every segment fights in its act's two rooms; the first act's second fight an event began.
        assert len(rows) == 2 * written and all(list(row) == list(ROW) for row in rows)
        events = written if wave == "act1" else 0
        assert summary[wave]["fights"] == dict(
            records=2 * written, kinds=dict(boss=written, normal=written), died=0, saves=2 * written - events,
            save_bytes=sum(path.stat().st_size for path in (directory / SAVES).iterdir()),
            write_seconds=summary[wave]["fights"]["write_seconds"])
        assert len(list((directory / SAVES).iterdir())) == 2 * written - events
        segments = {run["run_id"]: run for window in RunShards(directory).windows(shards=1) for run in window}
        for row in rows:
            assert row["run_id"] in segments and row["group"] == segments[row["run_id"]]["group"]
            assert (row["path"] is None) == row["from_event"]
            if row["path"] is not None:
                save = read_save(row["path"])
                assert save["at"] == "fight" and save["fight"]["floor"] == row["floor"]
                assert save["fight"]["encounter"] == row["encounter"] and save["fight"]["fight"] == row["fight"]
        # The shards hold the segments as they are: no records, no saves.
        assert not any("fight_records" in run for run in segments.values())
    # Without fight entries a wave writes none.
    for wave in plain["waves"]:
        assert "fights" not in plain[wave]
        assert not (tmp_path / "plain" / wave / FILE).exists() and not (tmp_path / "plain" / wave / SAVES).exists()


def test_every_wave_of_a_run_records_its_fights_and_they_go_at_its_cleanup(world):
    state = driver(world, eval_seeds=None).run(1)
    entry = state["iterations"]["0"]
    # Every segment of these rooms fights twice in its act.
    written = {wave: sum(outcomes.values()) for wave, outcomes in entry["sample"]["outcomes"].items()}
    assert {wave: fights["records"] for wave, fights in entry["sample"]["fights"].items()} == {
        wave: 2 * count for wave, count in written.items()}
    assert all(fights["saves"] == fights["records"] for fights in entry["sample"]["fights"].values())
    # Read by nothing after the update, the records and saves go with the round's shards.
    for wave in written:
        directory = world.output / "groups/round-0" / wave
        assert not (directory / SAVES).exists() and not (directory / FILE).exists()
        assert "fights" in json.loads((directory / "summary.json").read_text())
