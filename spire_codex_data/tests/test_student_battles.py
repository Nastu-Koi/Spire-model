"""Student-state battles: which fights of a sampled round are labelled, how the solver
enters them, and what the export and import make of them."""

import gzip
import json
from pathlib import Path

import pytest

from combat_outcome.data import fights as outcome_fights, labels as outcome_labels
from model.data import RunShards, samples
from model.dataset_index import HEADER, INDEX, STUDENT_BATTLE
from model.independent import import_samples
from model.protocol import execution_command
from model.saves import SCHEMA as SAVES_SCHEMA, anchor_commands
from model.testing import SyntheticEngine
from spire_codex_data import student_battles
from spire_codex_data.anchor import BUDGETS, row as sample_of
from spire_codex_data.sources import split_group
from spire_codex_data.student_battles import (
    BY_DEATH, BY_FRACTION, anchor_for, belt_full, export, in_fraction, process_run, relic_differences, run_round,
    select,
)

POLICY = dict(policy_version=3, policy_digest="b" * 64, iteration=2)


def named(inside, fraction=0.5, taken=()):
    """A run id on the wanted side of the fraction, not used yet."""
    for i in range(1000):
        candidate = f"run-{i:03d}"
        if candidate not in taken and in_fraction(candidate, fraction) == inside:
            return candidate
    raise AssertionError("no run id on that side")


def save(potions=(), slots=3):
    """A save as the game writes it: the belt is objects, or null when empty."""
    belt = [dict(id=potion, slot_index=i) for i, potion in enumerate(potions)] or None
    return dict(players=[dict(current_hp=60, max_hp=80, gold=50, potions=belt, max_potion_slot_count=slots,
                              deck=[], relics=[])], acts=[])


def fight(number, kind="normal", result="win", from_event=False, saved=True, potions=(), act=1, floor=3):
    return dict(fight=number, encounter_id=f"e{number}", encounter="ENCOUNTER.SYNTHETIC", kind=kind, act=act,
                floor=floor, map_point_type={"normal": "monster"}.get(kind, kind), second_boss=False,
                from_event=from_event, combat_index=number, result=result, save=save(potions) if saved else None,
                **({} if saved else dict(error="refused")))


def write_round(directory, runs, quarantined=()):
    """A sampled round with the files the selection reads: index, summary, saves and
    combat labels (entities only) of every run."""
    directory = Path(directory)
    (directory / "accepted").mkdir(parents=True)
    with gzip.open(directory / INDEX, "wt") as index:
        index.write(json.dumps(HEADER) + "\n")
        for line, run in enumerate(runs):
            if run["run_id"] in quarantined:
                continue
            index.write(json.dumps(dict(run_id=run["run_id"], seed=run["seed"], character=run["character"],
                                        phases={"map": 1}, contract="c", source="rollout", **POLICY,
                                        victory=run["victory"], shard="accepted/00000.jsonl.gz", line=line)) + "\n")
    (directory / "summary.json").write_text(json.dumps(dict(kind="rollout_round", **POLICY)))
    for i, run in enumerate(runs):
        name = f"{run['character']}-{i}"
        with gzip.open(directory / (name + ".saves.jsonl.gz"), "wt") as stream:
            for row in run["fights"]:
                stream.write(json.dumps(dict(schema=SAVES_SCHEMA, seed=run["seed"], run_id=run["run_id"],
                                             character=run["character"], ascension=0, **row)) + "\n")
        with gzip.open(directory / (name + ".combats.jsonl.gz"), "wt") as stream:
            for row in run["fights"]:
                stream.write(json.dumps(dict(combat_index=row["combat_index"],
                                             entities=run.get("entities") or [dict(entity_type="player")])) + "\n")
    return directory


def a_round(tmp_path):
    taken = []
    ids = {}
    for key, inside in (("whole", True), ("lost", False), ("quiet", False), ("refused", True), ("dead", True),
                        ("gone", True)):
        ids[key] = named(inside, taken=taken)
        taken.append(ids[key])
    runs = [
        dict(run_id=ids["whole"], seed="S1", character="Ironclad", victory=True,
             fights=[fight(1), fight(2, "elite", from_event=True), fight(3, "boss", potions=["POTION.A"] * 3)]),
        dict(run_id=ids["lost"], seed="S2", character="Silent", victory=False, fights=[fight(1), fight(2, result="loss")]),
        dict(run_id=ids["quiet"], seed="S3", character="Defect", victory=True, fights=[fight(1), fight(2)]),
        dict(run_id=ids["refused"], seed="S4", character="Regent", victory=True,
             fights=[fight(1, saved=False), fight(2, "elite")]),
        dict(run_id=ids["dead"], seed="S5", character="Necrobinder", victory=False,
             fights=[fight(1), fight(2, "elite", result="loss", potions=["POTION.A", "POTION.B", "POTION.C"])]),
        dict(run_id=ids["gone"], seed="S6", character="Ironclad", victory=None, fights=[fight(1)]),
    ]
    return write_round(tmp_path / "round-0", runs, quarantined={ids["gone"]}), ids


def test_selection_takes_the_fraction_whole_and_the_fight_each_lost_run_died_in(tmp_path):
    directory, ids = a_round(tmp_path)
    runs, counts = select(directory, 0.5)
    chosen = {run["run_id"]: run for run in runs}
    assert set(chosen) == {ids["whole"], ids["lost"], ids["refused"], ids["dead"]}
    assert [f["fight"] for f in chosen[ids["whole"]]["fights"]] == [1, 3]
    assert all(f["selected_by"] == [BY_FRACTION] for f in chosen[ids["whole"]]["fights"])
    assert chosen[ids["lost"]]["fights"] == [dict(fight=2, selected_by=[BY_DEATH])]
    assert [f["fight"] for f in chosen[ids["refused"]]["fights"]] == [2]
    # A lost run in the fraction: its last fight is selected once, under both reasons.
    assert [f["selected_by"] for f in chosen[ids["dead"]]["fights"]] == [[BY_FRACTION], [BY_FRACTION, BY_DEATH]]
    # The reasons add up to the fights: one fight is selected for the death alone, two are deaths.
    assert counts == dict(fights=12, runs=6, runs_in_fraction=3, runs_not_indexed=1, runs_selected=4, selected=6,
                          selected_boss=1, selected_by_death=1, selected_by_fraction=5, selected_deaths=2,
                          selected_elite=2, selected_normal=3, skipped_from_event=1, skipped_no_save=1)
    # What a run carries to its worker: its files, its policy and its outcome.
    whole = chosen[ids["whole"]]
    assert (whole["saves"], whole["combats"]) == ("Ironclad-0.saves.jsonl.gz", "Ironclad-0.combats.jsonl.gz")
    assert (whole["seed"], whole["character"], whole["victory"], whole["round"]) == ("S1", "Ironclad", True, str(directory))
    assert {k: whole[k] for k in POLICY} == POLICY
    # The fraction picks runs by their id alone: a larger fraction keeps the same runs and adds.
    more, _ = select(directory, 1.0)
    assert set(chosen) <= {run["run_id"] for run in more}
    fewer, counts = select(directory, 0.0)
    assert {run["run_id"] for run in fewer} == {ids["lost"], ids["dead"]} and counts["selected"] == 2
    # A trial takes the first runs of the selection, and counts those alone.
    limited, counts = select(directory, 0.5, max_runs=2)
    assert [run["run_id"] for run in limited] == [run["run_id"] for run in runs[:2]]
    assert (counts["runs_selected"], counts["runs_beyond_limit"]) == (2, 2)
    assert counts["selected"] == sum(len(r["fights"]) for r in limited) == counts["selected_by_fraction"] + counts.get(
        "selected_by_death", 0)
    with pytest.raises(ValueError):
        select(directory, 1.5)


def test_the_anchor_enters_as_the_saves_row_does_and_asks_for_a_potion_only_with_a_full_belt():
    row = dict(fight(2, "elite", potions=["POTION.A", "POTION.B", "POTION.C"]), run_id="r")
    spot = anchor_for(row)
    enter, room = anchor_commands(row)
    assert (spot["floor"], spot["map_point_type"], spot["second_boss"]) == (
        enter["act_floor"], enter["map_point_type"], enter["second_boss"])
    assert spot["room"] == room["room"] == dict(type="combat", encounter="SYNTHETIC")
    assert (spot["id"], spot["room_type"], spot["route"], spot["kind"], spot["belt"]) == (
        "r:fight-002", "elite", None, "elite", 3)
    assert spot["potion_policy"] == "RequireAtLeastOne"
    assert anchor_for(dict(fight(3, "boss", potions=["POTION.A"] * 3), run_id="r"))["potion_policy"] == "RequireAtLeastOne"
    # A normal fight, a belt with room, and a belt of potions the game uses by itself ask for nothing.
    assert "potion_policy" not in anchor_for(dict(fight(1, potions=["POTION.A"] * 3), run_id="r"))
    assert "potion_policy" not in anchor_for(dict(fight(1, "elite", potions=["POTION.A"] * 2), run_id="r"))
    assert "potion_policy" not in anchor_for(dict(fight(1, "boss", potions=["POTION.FAIRY_IN_A_BOTTLE"] * 3), run_id="r"))
    assert anchor_for(dict(fight(1, "boss"), run_id="r"))["room_type"] == "boss"
    assert belt_full(dict(potions=[dict(id="POTION.A", slot_index=0), dict(id="POTION.FAIRY_IN_A_BOTTLE", slot_index=1)],
                          max_potion_slot_count=2))
    assert belt_full(dict(potions=["POTION.A", "POTION.B"], max_potion_slot_count=2))
    assert not belt_full(dict(potions=None, max_potion_slot_count=0)) and not belt_full(dict(max_potion_slot_count=3))
    assert anchor_for(dict(fight(1, "elite"), run_id="r"))["belt"] == 0


def test_relic_differences_name_the_fields_that_changed_on_entering_again():
    student = [dict(entity_type="player", hp=1), dict(ref="r1", entity_type="relic", content_id="RELIC.EMBER_TEA",
                                                      stats=dict(Combats=4)),
               dict(ref="r2", entity_type="relic", content_id="RELIC.ANCHOR")]
    observation = dict(phase="combat", entities=[
        dict(ref="x1", entity_type="relic", content_id="RELIC.EMBER_TEA", stats=dict(Combats=3)),
        dict(ref="x2", entity_type="relic", content_id="RELIC.ANCHOR"), dict(ref="c", entity_type="card", zone="hand")])
    assert relic_differences(student, dict(observation=observation)) == [
        dict(relic="RELIC.EMBER_TEA", field="stats", student=dict(Combats=4), solver=dict(Combats=3))]
    assert relic_differences(student[:2], dict(observation=observation))[0]["relic"] == "*"
    assert relic_differences(None, dict(observation=observation)) is None
    assert relic_differences([dict(entity_type="player")], dict(observation=dict(entities=[]))) == []


class Fighting(SyntheticEngine):
    """A synthetic fight: its selection stands in the combat phase."""

    def frame(self):
        frame = super().frame()
        if self.stage == "selection":
            frame["public"]["phase"] = "combat"
        return frame


class Process:
    """Stands in for a native process of a run."""

    def __init__(self, config):
        self.build_identity = dict(worker_dll_sha256="w", game_dll_sha256="g", solver_dll_sha256="s",
                                   adapter_dll_sha256="a")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def solved(fought):
    """A `play_battle` that fights every fight the synthetic way: lost when it is a boss
    fight. A fight asked for a potion reports the searches that found no route with one."""

    def play_battle(shared, second, save, spot, budgets):
        fought.append((spot["id"], spot.get("potion_policy"), budgets))
        required = spot.get("potion_policy") == "RequireAtLeastOne"
        engine = Fighting(4, 2)
        frame = engine.reset("Ironclad", spot["id"], 0)
        records = []
        while frame["boundary"] != "terminal":
            candidate = frame["legal"]["candidates"][0]
            records.append(sample_of(frame, candidate, "combat_solver"))
            frame = engine.send(execution_command(frame, candidate["candidate_ref"]))
        spot["map_position"] = "representative_point"
        lost = spot["kind"] == "boss"
        return dict(records=records, contract=engine.contract, outcome="loss" if lost else "win", entry_hp=70,
                    exit_hp=None if lost else 50, **({"potion_refusals": 2} if required else {}))
    return play_battle


def test_fights_are_solved_exported_and_imported_with_their_policy(tmp_path, monkeypatch):
    directory, ids = a_round(tmp_path)
    fought = []
    monkeypatch.setattr(student_battles, "RunProcess", Process)
    monkeypatch.setattr(student_battles, "play_battle", solved(fought))
    output = tmp_path / "student"
    summary = run_round(directory, output, fraction=0.5, workers=1, config="config.json", budgets=BUDGETS)
    assert (summary["runs"], summary["counts"]["fights"], summary["counts"]["verified"]) == (4, 6, 6)
    assert (summary["counts"]["outcome_win"], summary["counts"]["outcome_loss"]) == (5, 1)
    assert summary["counts"]["potion_policy_RequireAtLeastOne"] == 2 and summary["run_failures"] == {}
    assert summary["verified_rate"] == 1.0 and summary["win_rate"] == pytest.approx(5 / 6, abs=1e-4)
    assert summary["selection"]["selected"] == 6 and summary["fights_by_kind"] == dict(boss=1, elite=2, normal=3)
    assert summary["policies"] == [dict(POLICY, runs=4)] and (output / "selection.json").exists()
    # The two full-belt elite and boss fights were asked for a potion; the synthetic fight uses none.
    assert sorted((policy for _, policy, _ in fought), key=str) == [None] * 4 + ["RequireAtLeastOne"] * 2
    assert (summary["counts"]["potion_required_unmet"], summary["counts"]["potion_refusals"]) == (2, 4)
    assert not {"potion_required_met", "fights_using_potions", "potions_used"} & set(summary["counts"])
    assert all(budgets == BUDGETS for _, _, budgets in fought)
    # The report of a run: every fight with its outcome, rows, and what it was selected by.
    report = json.loads((output / "runs" / (ids["whole"] + ".json")).read_text())
    assert report["identity"] == dict(version="student-battle-v1", worker_dll_sha256="w", game_dll_sha256="g",
                                      solver_dll_sha256="s", adapter_dll_sha256="a", **BUDGETS)
    assert {k: report[k] for k in POLICY} == POLICY and report["victory"] is True
    assert [(f["fight"], f["status"], f["outcome"], f["potion_policy"], f.get("potion_refusals")) for f in report["fights"]] == [
        (1, "verified", "win", "Smart", None), (3, "verified", "loss", "RequireAtLeastOne", 2)]
    assert report["fights"][0]["rows"] > 0 and report["fights"][0]["relic_differences"] == []
    # Run again: finished reports stand, nothing is fought again.
    run_round(directory, output, fraction=0.5, workers=1, config="config.json", budgets=BUDGETS)
    assert len(fought) == 6

    exported = export(output)
    assert exported["anchor_kind"] == STUDENT_BATTLE and exported["counts"]["lost_fights"] == 1
    assert exported["counts"]["fights"] == exported["combat_outcomes"]["counts"]["elite/win"] + exported[
        "combat_outcomes"]["counts"]["regular/win"]
    with gzip.open(output / "independent-training.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    assert rows and {r["metadata"]["sample_group"] for r in rows} == {
        f["id"] for f in report["fights"] if f["outcome"] == "win"} | {
        f"{ids[k]}:fight-{n:03d}" for k, n in (("lost", 2), ("refused", 2), ("dead", 1), ("dead", 2))}
    first = next(r for r in rows if r["metadata"]["sample_group"] == f"{ids['whole']}:fight-001")
    meta = first["metadata"]
    assert (meta["anchor_kind"], meta["label_source"], meta["teacher_visibility"], meta["bc_only"]) == (
        STUDENT_BATTLE, "combat_solver", "privileged", True)
    assert (meta["native_replay_verified"], meta["verified_by"], meta["initialization"]) == (
        True, "native_replay", "summary-anchor-v1")
    assert meta["run_hash"] == ids["whole"] and meta["split_group"] == split_group("S1")
    assert {k: meta[k] for k in POLICY} == POLICY and meta["outcome"] == "win"
    assert (meta["kind"], meta["fight"], meta["student_result"], meta["map_position"]) == (
        "normal", 1, "win", "representative_point")
    assert first["actor"] == "combat_solver" and first["label"] in first["options"]
    # Every fought fight is an outcome label the combat-outcome reader accepts, lost ones too.
    labels = [row for _, _, row in outcome_labels(output / "combat-outcomes.jsonl.gz")]
    assert len(labels) == 6 and {row["source"] for row in labels} == {STUDENT_BATTLE}
    assert {row["actor"] for row in labels} == {"combat_solver"}
    lost = next(row for row in labels if row["result"] == "loss")
    assert (lost["kind"], lost["end_hp"], lost["hp_lost"], lost["student"]["fight"]) == ("boss", 0, 70, 3)
    assert lost["student"]["potion_policy"] == "RequireAtLeastOne" and lost["seed"] == split_group("S1")
    # The outcome rows point at their decision frames in the sample files.
    assert [len(frames) for _, frames in outcome_fights(output)] == [row["frames"]["count"] for row in labels]

    imported = import_samples(output / "independent-training.jsonl.gz", tmp_path / "imported")
    assert imported["accepted_runs"] == 5 and imported.get("quarantined_groups", 0) == 0
    shards = RunShards(tmp_path / "imported")
    assert all(row["source"] == STUDENT_BATTLE and {k: row[k] for k in POLICY} == POLICY for row in shards.rows)
    (window,) = shards.windows()
    items = samples(window, weights=shards.weights)
    assert {i.source for i in items} == {STUDENT_BATTLE} and sum(i.weight for i in items) == pytest.approx(1.0)
    # A larger fraction fights the runs it selects other fights of, and keeps the rest:
    # the lost run's first fight and the quiet run are new, three reports stand.
    grown = run_round(directory, output, fraction=1.0, workers=1, config="config.json", budgets=BUDGETS)
    assert sorted(name for name, _, _ in fought[6:]) == sorted(
        f"{ids[k]}:fight-{n:03d}" for k, n in (("lost", 1), ("lost", 2), ("quiet", 1), ("quiet", 2)))
    assert (grown["runs"], grown["counts"]["fights"]) == (5, 9)
    # A smaller one takes fewer runs: the reports of the others stay on disk for a
    # selection that takes them again, outside the summary and the export.
    shrunk = run_round(directory, output, fraction=0.0, workers=1, config="config.json", budgets=BUDGETS)
    assert (shrunk["runs"], shrunk["counts"]["fights"], shrunk["counts"]["selected_by_death"]) == (2, 2, 2)
    assert len(list((output / "runs").glob("*.json"))) == 5
    assert export(output)["combat_outcomes"]["counts"] == {"elite/win": 1, "regular/win": 1}
    # Other budgets are another identity: every selected run is fought again.
    before = len(fought)
    run_round(directory, output, fraction=0.0, workers=1, config="config.json", budgets=dict(BUDGETS, budget_ms=500))
    assert len(fought) == before + 2


CONTRACT = dict(adapter_version="duel-v1", observation_schema="public-state-v6", action_schema="candidate-v1",
                public_history_version="public-history-v1", fixed_ascension=0, training_ready=True,
                initialization="summary-anchor-v1")


class Duel:
    """A fight of `turns` turns with a belt, and a solver that does what it is asked: a
    search asked for a potion uses the first one of the belt, or refuses (while its
    state version is in `refuse`, or with an empty belt); a search left to itself ends
    the turn. `fail` are the state versions at which it finds nothing at all."""

    def __init__(self, turns=3, belt=("POTION.A", "POTION.B", "POTION.C"), refuse=(), fail=()):
        self.turns, self.start, self.refuse, self.fail = turns, list(belt), set(refuse), set(fail)
        self.asked = []

    def reset(self):
        self.version, self.turn, self.belt = 0, 0, list(self.start)
        return self.frame()

    def frame(self):
        fighting = self.turn < self.turns
        entities = [dict(ref="player", entity_type="player", hp=50 - self.turn, max_hp=80, round=self.turn + 1)]
        entities += [dict(ref=f"potion:{i}", entity_type="potion", content_id=potion, slot=i)
                     for i, potion in enumerate(self.belt)]
        legal = [dict(verb="USE_POTION", decoder_slot_ref=f"potion:{i}", source_refs=[f"potion:{i}"], target_refs=[])
                 for i in range(len(self.belt))] if fighting else []
        legal.append(dict(verb="END_TURN" if fighting else "SKIP_REWARDS", decoder_slot_ref="end", source_refs=[],
                          target_refs=[]))
        return dict(type="decision_frame", contract=dict(CONTRACT), boundary="decision", events=[],
                    routing=dict(episode_id="duel", decision_id=f"duel:{self.version}", state_version=self.version),
                    public=dict(phase="combat" if fighting else "rewards", entities=entities, relations=[], memory=[]),
                    legal=dict(candidates=[dict(c, candidate_ref=f"c{i}") for i, c in enumerate(legal)]))

    def send(self, command):
        if command["cmd"] == "execute_candidate":
            chosen = next(c for c in self.frame()["legal"]["candidates"] if c["candidate_ref"] == command["candidate_ref"])
            if chosen["verb"] == "USE_POTION":
                self.belt.pop(int(chosen["source_refs"][0].split(":")[1]))
            else:
                self.turn += 1
            self.version += 1
        return self.frame()

    def step(self, frame, **options):
        policy = options.get("potion_policy", "Smart")
        self.asked.append((self.version, policy))
        if self.version in self.fail:
            return dict(type="error", message="nothing found")
        candidates = frame["legal"]["candidates"]
        if policy == "RequireAtLeastOne":
            if self.version in self.refuse or not self.belt:
                return dict(type="error", message="no route with a potion")
            chosen = candidates[0]
        else:
            chosen = candidates[-1]
        return dict(type="solver_step", candidate_ref=chosen["candidate_ref"],
                    frame=self.send(execution_command(frame, chosen["candidate_ref"])))


class Seat:
    """A native process of a run, holding one scripted fight."""

    def __init__(self, **script):
        self.engine = Duel(**script)

    def enter(self, save, spot):
        return self.engine, self.engine.reset()

    def native_error(self):
        return None


def fought_as(spot, **script):
    from spire_codex_data.anchor import play_battle

    shared = Seat(**script)
    fight = play_battle(shared, Seat(**script), {}, dict(spot, room_type="elite"), BUDGETS)
    used = [r["label"]["source_refs"][0] for r in fight["records"] if r["label"]["verb"] == "USE_POTION"]
    return fight, used, shared.engine.asked


def test_a_fight_asked_for_a_potion_asks_every_search_until_one_is_used():
    asking = dict(potion_policy="RequireAtLeastOne")
    # Asked once: the solver uses one potion, and the rest of the fight is its own. Asked at
    # every search it would have emptied the belt and then refused.
    fight, used, asked = fought_as(asking)
    assert (fight["outcome"], fight["potion_refusals"], used) == ("win", 0, ["potion:0"])
    assert asked == [(0, "RequireAtLeastOne"), (1, "Smart"), (2, "Smart"), (3, "Smart")]
    assert (fight["entry_hp"], fight["exit_hp"], len(fight["records"])) == (50, 47, 4)
    # A search that finds no route with a potion is made again on the solver's own
    # policy, and the next search is asked again.
    fight, used, asked = fought_as(asking, refuse={0, 1})
    assert (fight["potion_refusals"], used) == (2, ["potion:0"])
    assert asked == [(0, "RequireAtLeastOne"), (0, "Smart"), (1, "RequireAtLeastOne"), (1, "Smart"),
                     (2, "RequireAtLeastOne"), (3, "Smart")]
    # No search finds one: the fight is the solver's own, and says how often it was refused.
    fight, used, asked = fought_as(asking, refuse={0, 1, 2})
    assert (fight["outcome"], fight["potion_refusals"], used) == ("win", 3, [])
    assert [policy for _, policy in asked] == ["RequireAtLeastOne", "Smart"] * 3
    # A fight that is not asked never mentions the policy, and reports no refusals.
    fight, used, asked = fought_as({})
    assert "potion_refusals" not in fight and used == [] and {policy for _, policy in asked} == {"Smart"}
    # A search that fails on the solver's own policy too fails the fight.
    with pytest.raises(ValueError, match="solver_failed:nothing found"):
        fought_as(asking, fail={1})
