"""The points and paired continuations the potions' prices are fitted on
(`main.play.potion_value`): a point at the map after each fight with a potion in the
belt, the points drawn by act, belt and HP, a continuation as it stood and one per potion
thrown away there, each come to again from its act's anchor and played to the next act's
start; then the fit on them (`potion_prices.fit`)."""

import gzip
import json
from functools import partial
from pathlib import Path

import pytest

from main.env.protocol import CHARACTERS, ProtocolError, execution_command
from main.learn import costs
from main.learn.potion_prices import determinism, fit, pairs
from main.play.potion_value import PointEngine, branch_saves, draw, play_branches, play_sources, read_point
from main.play.fights import state as fight_state
from main.play.rollout import RolloutRunner
from main.play.sampling import Samplers
from main.tests.journeys import TEMPERATURES, Journey, learner

BELT = ("POTION.FIRE_POTION", "POTION.BLOCK_POTION")


class Belted(Journey):
    """Three acts of two rooms with a belt of potions (capacity 3), full as each room is
    entered; the decision after each fight stands on the map and offers to throw each
    potion away. A continue save names the belt, and a run resumed from one holds what
    it names. As the engine's own saves do, one written at the map after a room resumes
    before that room, its fight to play again; one written at an act's first map
    decision resumes to that decision."""

    def __init__(self):
        super().__init__(4, 2, rooms=6, acts=2)
        self.belt = list(BELT)

    def enter_room(self):
        super().enter_room()
        self.belt = list(BELT)

    def frame(self):
        frame = super().frame()
        frame["public"]["entities"][0].update(capacity=3)
        frame["public"]["entities"] += [dict(ref=f"potion:{slot}", entity_type="potion", content_id=potion,
                                             owner_ref="player", slot=slot) for slot, potion in enumerate(self.belt)]
        if self.stage == "combat_play":
            frame["public"]["phase"] = "map"
            discards = [dict(verb="DISCARD_POTION", decoder_slot_ref=f"discard:potion:{slot}",
                             source_refs=[f"potion:{slot}"], target_refs=[], effect_coverage="opaque")
                        for slot in range(len(self.belt))]
            frame["public"]["decoder_bank"] = frame["public"]["decoder_bank"] + discards
            frame["legal"]["candidates"] += [dict(candidate, candidate_ref=f"d{slot}")
                                             for slot, candidate in enumerate(discards)]
        return frame

    def send(self, command):
        if command["cmd"] == "execute_candidate" and command["candidate_ref"].startswith("d"):
            before = self.frame()
            if command != execution_command(before, command["candidate_ref"]):
                raise ProtocolError("stale_decision")
            del self.belt[int(command["candidate_ref"][1:])]
            self.version += 1
            return self.frame()
        reply = super().send(command)
        if command["cmd"] == "write_continue_save":
            save = json.loads(Path(command["path"]).read_text())
            save["players"][0]["potions"] = [dict(id=potion, slot_index=slot) for slot, potion in enumerate(self.belt)]
            save["act_start"] = self.stage == "combat_play" and self.room > 0 and self.room % self.acts == 0
            Path(command["path"]).write_text(json.dumps(save))
        return reply

    def resume(self, save):
        self.character, self.seed, self.ascension = save["character"], str(save["seed"]), save["ascension"]
        self.commands, self.version, self.room = [], 0, save["room"]
        if save["act_start"]:
            self.stage, self.events = "combat_play", []
        else:
            self.enter_room()
        self.belt = [potion["id"] for potion in save["players"][0].get("potions") or []]
        return self.frame()


def pointed(directory, started):
    return PointEngine(Belted(), directory, started)


def test_points_their_continuations_and_the_fit_on_them(tmp_path):
    policy = learner()
    played = RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                           ascension=0, temperatures=TEMPERATURES, iteration=None)
    seeds = {character: [f"{character}-s"] for character in CHARACTERS}
    levels = {character: {f"{character}-s": 0} for character in CHARACTERS}
    started = {(character, f"{character}-s") for character in CHARACTERS}
    with Samplers(partial(pointed, tmp_path / "points", started), policy.model, policy.vocabulary,
                  engines=2, processes=1) as pool:
        source = play_sources(pool, played, seeds, levels, tmp_path)
        points = [read_point(path) for path in sorted((tmp_path / "points").glob("*.json.gz"))]
        # A point after each of the six fights of every run, the belt as it stands.
        assert (source["runs"], sum(source["outcomes"].values()), source["points"], len(points)) == (5, 5, 30, 30)
        assert all(point["state"]["potions"] == list(BELT) and point["state"]["capacity"] == 3 for point in points)
        assert sorted({point["state"]["act"] for point in points}) == [1, 2, 3]
        # Played once, the sources are not played again.
        assert play_sources(pool, played, seeds, levels, tmp_path) == source
        drawn, strata = draw(points, pairs_per_act=40)
        assert strata["1-room-2"] == dict(asked=5, found=6) and strata["1-full-0"] == dict(asked=10, found=0)
        branches = branch_saves(drawn, tmp_path / "branches")
        assert [branch["branch"] for branch in branches[:3]] == ["keep", "drop0", "drop1"]
        assert [branch["dropped"] for branch in branches[:3]] == [None, *BELT]
        results = play_branches(pool, played, drawn, branches, tmp_path, repeat=2)
        assert len(results) == len(branches) + 2 and all(row["status"] == "complete" for row in results)
        # Played once, the continuations are not played again.
        assert len(play_branches(pool, played, drawn, branches, tmp_path, repeat=2)) == len(results)
    named = {point["id"]: point for point in drawn}
    for row in results:
        where = named[row["point"]]["state"]
        assert all("save" not in record for record in row["records"])
        # Its fights are the rooms' after the point: the room the point follows is not
        # played again. After an act's last room no fight is left before the cut.
        assert all((record["act"], record["floor"]) > (where["act"], where["floor"]) for record in row["records"])
        assert bool(row["records"]) == (where["floor"] == 1)
        if where["act"] == 3:
            assert row["outcome"] in ("win", "loss") and row["cut_state"] is None
        else:
            # Stopped at the next act's first decision, the belt filled in its first room.
            assert row["outcome"] == "reached" and row["cut_state"]["act"] == where["act"] + 1
            assert row["cut_state"]["potions"] == list(BELT)
    assert determinism(results) == dict(points=2, same=2, cost_difference_sd=0.0)
    # Nothing a potion does here: the fights cost alike, so a pair of the first two acts
    # differs only by the potion held at the next act's start, which is worth what the
    # next act's prices say, nothing in the end.
    assert len(pairs(drawn, results)) == 2 * len(drawn)
    prices, report = fit(drawn, results, costs.FIXED, grid=(1e-2, 1.0))
    assert report["by_act"][3]["mean"] == pytest.approx(0.0)
    for act in (1, 2):
        assert report["by_act"][act]["mean"] == pytest.approx(0.0, abs=0.5)


def test_a_points_continuations_start_at_the_point(tmp_path):
    """A continuation resumes at its point's map decision, as the source run stood there,
    with the potion taken out thrown away; not before the room the point follows."""
    policy = learner()
    played = RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                           ascension=0, temperatures=TEMPERATURES, iteration=None)
    seeds = {"Ironclad": ["Ironclad-s"]}
    with Samplers(partial(pointed, tmp_path / "points", {("Ironclad", "Ironclad-s")}), policy.model,
                  policy.vocabulary, engines=1, processes=1) as pool:
        play_sources(pool, played, seeds, {"Ironclad": {"Ironclad-s": 0}}, tmp_path)
    points = [read_point(path) for path in sorted((tmp_path / "points").glob("*.json.gz"))]
    assert sorted({point["state"]["act"] for point in points}) == [1, 2, 3]
    for branch in branch_saves(points, tmp_path / "branches"):
        where = next(point for point in points if point["id"] == branch["point"])["state"]
        with gzip.open(branch["path"], "rt", encoding="utf-8") as stream:
            frame = pointed(tmp_path / "points", set()).resume(json.load(stream)["save"])
        resumed = fight_state(frame)
        held = list(where["potions"])
        if branch["dropped"] is not None:
            assert held.pop(int(branch["branch"][len("drop"):])) == branch["dropped"]
        assert frame["public"]["phase"] == "map"
        assert (resumed["act"], resumed["floor"], resumed["potions"]) == (where["act"], where["floor"], held)


def test_the_command_fits_the_prices_and_goes_on_where_it_stopped(tmp_path, monkeypatch):
    from main import commands
    from main.cli import parser
    from main.learn.potion_prices import Smoothed, load
    from main.play.potion_value import RESULTS, read_results
    from main.tests.journeys import checkpoint

    monkeypatch.setattr(commands, "CliEngine", lambda root=None: Belted())
    shown = []
    monkeypatch.setattr(commands, "emit", shown.append)
    path = checkpoint(tmp_path / "policy")
    output = tmp_path / "fit"
    arguments = ["fit-prices", "--checkpoint", str(path), "--output", str(output), "--seed-root", "t",
                 "--source-runs-per-character", "1", "--pairs", "60", "--repeat", "1", "--workers", "2",
                 "--processes", "1", "--memory-floor", "0", "--precision", "fp32"]
    commands.COMMANDS["fit-prices"](parser().parse_args(arguments), {})
    report = json.loads((output / "report.json").read_text())
    assert report["source"]["runs"] == 5 and report["failed"] == 0 and report["pairs"] > 0
    assert report["determinism"]["points"] == 1 and report["previous"] == "fixed"
    prices = load(output / "prices.json")
    assert isinstance(prices, Smoothed) and report["prices_digest"] == shown[0]["prices_digest"]
    # Asked again, it plays nothing more.
    played = len(read_results(output / RESULTS))
    commands.COMMANDS["fit-prices"](parser().parse_args(arguments), {})
    assert len(read_results(output / RESULTS)) == played
