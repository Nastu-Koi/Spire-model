"""Hindsight forks: a fight segment, replayed from its fight's entry save to a decision of
its own (`RolloutRunner.run(hindsight=)`), and there made to drink a potion, take a
candidate or left to draw; the decisions a segment parts at (`branch_points`), the groups
a round's fight groups give them and the wave they make
(`main.play.fight_groups.sample_hindsight`)."""

from copy import deepcopy

import pytest

from main.data.shards import RunShards
from main.env.protocol import CHARACTERS, ProtocolError, execution_command
from main.cli import parser
from main.iterate import Driver, fight_cost
from main.learn import costs, potion_prices
from main.learn.costs import FIXED, with_growth
from main.learn.batches import stream_shares
from main.learn.fight_groups import advantages, wave_advantages
from main.learn.segments import read_waves
from main.play import digests
from main.play.digests import frame_key
from main.play.fight_groups import (CONTROLS, DRAWS, FORCED, HINDSIGHT, PLACES, WAVE, arms, branch_points,
                                    combat_round, distinct_pair, drawn_decisions, drawn_steps, hp_ratio, pick_hindsight,
                                    quotas, read_picks, same_action, sample_fight_groups, sample_hindsight)
from main.play.fights import DANGER, drinks, fight_rows
from main.play.groups import sample_groups
from main.play.sampling import Samplers
from main.tests.journeys import learner
from main.tests.test_fight_groups import Idle, Rooms, entry, runner
from main.tests.test_iterate import keep_uniform, settings, world  # noqa: F401  (fixtures)

BLOCK, FIRE = "POTION.BLOCK_POTION", "POTION.FIRE_POTION"
ENEMY, CARD, BLOW = 60, 10, 30


class Brawl(Rooms):
    """Fights in rounds (`round` on the player): a card deals 10 of the enemy's 60 HP,
    two cards a round; the end of the turn takes 30 HP, or uses up the block of a block
    potion; a fire potion deals 20. The enemy dead, the room's decision after the fight
    follows; the player dead, the run is lost. Each fight begins with a block and a
    fire potion in the belt, at the HP the room gives."""

    def enter_room(self):
        super().enter_room()
        self.stage, self.round, self.plays, self.enemy, self.block = "brawl", 1, 0, ENEMY, False
        self.hp, self.belt = 70 - 5 * self.room, [BLOCK, FIRE]

    def frame(self):
        frame = super().frame()
        if self.stage != "brawl":
            return frame
        public = frame["public"]
        public["phase"] = "combat"
        public["entities"][0].update(hp=self.hp, round=self.round)
        public["entities"] += [dict(ref=f"potion:{slot}", entity_type="potion", owner_ref="player", slot=slot,
                                    content_id=potion) for slot, potion in enumerate(self.belt)]
        legal = [dict(verb="PLAY_CARD", decoder_slot_ref="PLAY_CARD", source_refs=["player"], target_refs=[],
                      effect_coverage="opaque", semantic_program=dict(kind="effect", op="PLAY_CARD"))] \
            if self.plays < 2 else []
        legal.append(dict(verb="END_TURN", decoder_slot_ref="END_TURN", source_refs=["player"], target_refs=[],
                          effect_coverage="opaque", semantic_program=dict(kind="effect", op="END_TURN")))
        legal += [dict(verb="USE_POTION", decoder_slot_ref=f"potion:{slot}", source_refs=[f"potion:{slot}"],
                       target_refs=[], effect_coverage="opaque", semantic_program=dict(kind="effect", op="USE_POTION"))
                  for slot in range(len(self.belt))]
        public["decoder_bank"] = legal
        frame["legal"] = dict(candidates=[dict(c, candidate_ref=f"c{i}") for i, c in enumerate(legal)])
        return frame

    def send(self, command):
        if self.stage != "brawl" or command["cmd"] != "execute_candidate":
            return super().send(command)
        before = self.frame()
        if command != execution_command(before, command["candidate_ref"]):
            raise ProtocolError("stale_decision")
        candidate = next(c for c in before["legal"]["candidates"] if c["candidate_ref"] == command["candidate_ref"])
        self.commands.append(deepcopy(command))
        self.version += 1
        verb = candidate["verb"]
        if verb == "PLAY_CARD":
            self.enemy, self.plays = self.enemy - CARD, self.plays + 1
        elif verb == "USE_POTION":
            potion = self.belt.pop(int(candidate["decoder_slot_ref"].split(":")[1]))
            if potion == BLOCK:
                self.block = True
            else:
                self.enemy -= 20
        else:
            if not self.block:
                self.hp -= BLOW
            self.block, self.round, self.plays = False, self.round + 1, 0
        if self.hp <= 0:
            self.stage, self.won = "terminal", False
            self.events = [dict(type="run_completed", victory=False, act=3, final_boss_defeated=False)]
        elif self.enemy <= 0:
            self.stage = "combat_play"
            self.events = [dict(type="encounter_completed", encounter_id=self.encounter, act=1, kind="normal",
                                result="victory")]
        return self.frame()


def drawn(run):
    """The decisions with a choice a run drew, in order."""
    return [step for macro in run["macros"] for step in macro["steps"] if not step["forced"]]


def played(room=1, **options):
    """A fight segment of room `room` of a brawl, drawn by an untrained policy."""
    return runner(learner()).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(room), until_fight_end=True,
                                 run_id="g-0", group="g", **options)


def fork_of(run, at, potion=None, force=None):
    """The hindsight of a fork at the `at`-th decision with a choice of `run`."""
    steps = drawn(run)
    return dict(replay=[step["candidate_ref"] for step in steps[:at]], key=frame_key(steps[at]["frame"]),
                potion=potion, force=force)


def longest(runs):
    return max(runs, key=lambda run: len(drawn(run)))


# ---- Replaying to a fork ----

def test_the_actions_of_a_segment_bring_its_fight_back_to_the_frame_of_any_decision():
    policy = learner()
    first = longest(played() for _ in range(4))
    steps = drawn(first)
    assert len(steps) >= 3 and first["status"] == "complete"
    for at in (0, 1, len(steps) - 1):
        again = runner(policy).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(1), until_fight_end=True,
                                   run_id="h-0", group="h", hindsight=fork_of(first, at))
        assert again["status"] == "complete", again.get("error")
        assert again["hindsight"] == dict(potion=None, force=None, replayed=at)
        # Nothing before the fork is kept: the segment's first decision is the fork's.
        assert frame_key(drawn(again)[0]["frame"]) == frame_key(steps[at]["frame"])
        assert again["macros"][0]["steps"][0]["frame"] == drawn(again)[0]["frame"]
        assert not any(step.get("explored") for step in drawn(again))
        # Its fight's record is the whole fight, from its entry.
        (record,) = again["fight_records"]
        assert (record["entry_hp"], record["fight"]) == (first["fight_records"][0]["entry_hp"], 1)


def test_a_fork_with_a_potion_drinks_it_as_an_explored_decision():
    first = longest(played() for _ in range(4))
    at = next(index for index, step in enumerate(drawn(first)) if FIRE in drinks(step["frame"]))
    again = runner(learner()).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(1), until_fight_end=True,
                                  run_id="h-0", group="h", hindsight=fork_of(first, at, potion=FIRE))
    assert again["status"] == "complete", again.get("error")
    fork = drawn(again)[0]
    candidate = next(c for c in fork["frame"]["legal"]["candidates"] if c["candidate_ref"] == fork["candidate_ref"])
    assert candidate["verb"] == "USE_POTION" and fork["candidate_ref"] in drinks(fork["frame"])[FIRE]
    assert fork["explored"] is True and again["hindsight"] == dict(potion=FIRE, force=None, replayed=at)
    assert FIRE in again["fight_records"][0]["potions_used"]


def test_a_fork_with_a_candidate_takes_it_as_a_forked_decision_and_draws_after_it():
    first = longest(played() for _ in range(4))
    steps = drawn(first)
    at = len(steps) // 2
    for candidate in steps[at]["frame"]["legal"]["candidates"]:
        again = runner(learner()).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(1), until_fight_end=True,
                                      run_id="h-0", group="h",
                                      hindsight=fork_of(first, at, force=candidate["candidate_ref"]))
        assert again["status"] == "complete", again.get("error")
        fork, *rest = drawn(again)
        assert fork["candidate_ref"] == candidate["candidate_ref"] and fork["forked"] is True
        assert not fork.get("explored") and not any(step.get("forked") or step.get("explored") for step in rest)
        assert again["hindsight"] == dict(potion=None, force=candidate["candidate_ref"], replayed=at)

def test_a_replay_that_misses_its_fork_is_an_error_of_the_segment():
    first = longest(played() for _ in range(4))
    steps = drawn(first)
    wrong = [dict(fork_of(first, 1), key="another frame"),
             dict(fork_of(first, 1), replay=["c9"]),
             fork_of(first, 1, potion="POTION.NOT_IN_THE_BELT"),
             fork_of(first, 1, force="c99"),
             dict(fork_of(first, 0), replay=[step["candidate_ref"] for step in steps] + ["c0"] * 40)]
    for hindsight in wrong:
        run = runner(learner()).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(1), until_fight_end=True,
                                    run_id="h-0", group="h", hindsight=hindsight)
        assert run["status"] == "error" and "error" in run


def test_only_a_sampled_fight_segment_replays_to_a_fork():
    hindsight = dict(replay=[], key="k", potion=None)
    with pytest.raises(ValueError):
        runner(learner()).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(1), hindsight=hindsight)
    with pytest.raises(ValueError):
        runner(learner(), greedy_fights=True).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(1),
                                                  until_fight_end=True, hindsight=hindsight)
    with pytest.raises(ValueError):
        runner(learner()).run(Brawl(4, 2, rooms=4), "Regent", "S", resume=entry(1), until_fight_end=True,
                              hindsight=dict(hindsight, potion=FIRE, force="c0"))


# ---- The forks of a round ----

def brawls():
    """Three acts of two rooms of brawls: a normal fight and the act's boss."""
    return Brawl(4, 2, rooms=6, acts=2)


class LateBrawl(Brawl):
    """Brawls in the last act alone: the fights before it are the rooms' selections,
    which every run comes through to the last act's wave."""

    def enter_room(self):
        if self.act_of(self.room) < 3:
            Rooms.enter_room(self)
        else:
            super().enter_room()


def late_brawls():
    return LateBrawl(4, 2, rooms=6, acts=2)


def with_distributions(run, gaps):
    """`run` with a distribution kept at each decision with a choice: the `at`-th's first
    two candidates `gaps[at]` apart (a large gap where none is given)."""
    run = deepcopy(run)
    for at, step in enumerate(drawn(run)):
        count = len(step["frame"]["legal"]["candidates"])
        step["log_probs"] = [0.0, -gaps.get(at, 9.0)] + [-20.0] * (count - 2)
    return run


def test_two_copies_of_a_card_are_the_same_action_and_two_targets_are_not():
    card = dict(entity_type="card", content_id="CARD.STRIKE", cost=1, upgraded=False, zone="hand")
    frame = dict(public=dict(entities=[dict(card, ref="card:1", position=0), dict(card, ref="card:2", position=1),
                                       dict(card, ref="card:3", position=2, upgraded=True)]),
                 legal=dict(candidates=[
                     dict(verb="PLAY_CARD", source_refs=["card:1"], target_refs=["creature:1"], candidate_ref="c0"),
                     dict(verb="PLAY_CARD", source_refs=["card:2"], target_refs=["creature:1"], candidate_ref="c1"),
                     dict(verb="PLAY_CARD", source_refs=["card:2"], target_refs=["creature:2"], candidate_ref="c2"),
                     dict(verb="PLAY_CARD", source_refs=["card:3"], target_refs=["creature:1"], candidate_ref="c3"),
                     dict(verb="END_TURN", source_refs=[], target_refs=[], candidate_ref="c4")]))
    c = frame["legal"]["candidates"]
    assert same_action(frame, c[0], c[1]) and not same_action(frame, c[0], c[2])
    assert not same_action(frame, c[0], c[3]) and not same_action(frame, c[0], c[4])
    # The second distinct action passes over the first's copies.
    assert distinct_pair(frame, [-0.1, -0.1, -2.0, -3.0, -4.0]) == ("c0", "c2", pytest.approx(1.9))
    one = dict(frame, legal=dict(candidates=c[:2]))
    assert distinct_pair(one, [-0.1, -0.2]) is None


def test_a_segment_parts_in_danger_where_it_is_least_sure_and_at_a_rounds_first_decision():
    run = longest(played() for _ in range(6))
    steps = drawn_decisions(run)
    rounds = [combat_round(step["frame"]) for step in steps]
    assert rounds[0] == 1 and rounds == sorted(rounds) and rounds[-1] >= 2
    later = [at for at, value in enumerate(rounds) if value >= 2]
    sure = later[-1]
    run = with_distributions(run, {sure: 0.5, 0: 0.1})
    points = {point["place"]: point for point in branch_points(run, "k")}
    for point in points.values():
        assert point["key"] == frame_key(steps[point["at"]]["frame"]) and point["round"] == rounds[point["at"]]
    danger = [at for at, step in enumerate(steps)
              if hp_ratio(step["frame"]) < DANGER and any(p is not None for p in drinks(step["frame"]))]
    if danger:
        at = danger[0]
        potions = sorted(p for p in drinks(steps[at]["frame"]) if p is not None)
        assert points["danger"]["at"] == at
        assert points["danger"]["arms"] == [{"potion": p} for p in potions] + [None] * CONTROLS
    else:
        assert "danger" not in points
    # The closest pair from the second round on, not the first round's closer one.
    if not danger or danger[0] != sure:
        uncertain = points["uncertain"]
        candidates = steps[sure]["frame"]["legal"]["candidates"]
        assert uncertain["at"] == sure
        assert uncertain["arms"] == ([{"force": candidates[0]["candidate_ref"]}] * FORCED
                                     + [{"force": candidates[1]["candidate_ref"]}] * FORCED)
    turn = points["turn"]
    assert turn["arms"] == [None] * DRAWS and turn["round"] >= 2
    assert turn["at"] == rounds.index(turn["round"]) and turn["at"] not in (
        point["at"] for place, point in points.items() if place != "turn")
    # Drawn by the key, and none where the segment kept no distribution.
    assert branch_points(run, "k") == branch_points(run, "k")
    bare = played()
    for macro in bare["macros"]:
        for step in macro["steps"]:
            step.pop("log_probs", None)
    assert "uncertain" not in {point["place"] for point in branch_points(bare, "k")}


def test_a_rounds_fight_segments_fork_into_groups_by_what_their_decisions_show(tmp_path):
    policy = learner()
    seeds = {character: [f"{character}-seed"] for character in CHARACTERS}
    greedy = runner(policy, greedy_fights=True)
    fights = greedy.variant(greedy_fights=False)
    directory = tmp_path / "round"
    with Samplers(brawls, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        sample_groups(pool, greedy, seeds, directory, n=2, m=[1, 1], fight_entries=True)
        sample_fight_groups(pool, fights, directory, 11, cost=fight_cost, key="test:0", size=3)
        summary = sample_hindsight(pool, fights, directory, 6, key="test:0")
    wave = summary["fight_hindsight"]
    stored = read_picks(directory / HINDSIGHT)
    forks, account = stored["forks"], stored["account"]
    records = {row["run_id"]: row for row in fight_rows(directory, WAVE)}
    # Trunks 2:2:2 by kind, died or not; a kind with fewer segments gives what it has.
    kinds = {kind: sum(row["kind"] == kind for row in records.values()) for kind in quotas(6)}
    assert account["trunks"] == sum(min(quota, kinds[kind]) for kind, quota in quotas(6).items())
    assert account["short"] == [kind for kind, quota in quotas(6).items() if kinds[kind] < quota]
    # The pick is the key's: the same forks again.
    assert pick_hindsight(directory, 6, "test:0") == (forks, account)
    # Shared otherwise, a kind at 0 is forked none of.
    _, bosses = pick_hindsight(directory, 6, "test:0", shares=dict(normal=0, elite=0, boss=1))
    assert bosses["trunks"] == min(6, kinds["boss"]) and bosses["kinds"]["normal"] == 0
    assert forks and set(account["places"]) == set(PLACES) and sum(account["places"].values()) == len(forks)
    for fork in forks:
        assert fork["group"] == f"{fork['source']}:{fork['place']}" and fork["source"] in records
        if fork["place"] == "danger":
            potions = [arm["potion"] for arm in fork["arms"] if arm is not None]
            assert potions == sorted(set(potions)) and fork["arms"][len(potions):] == [None] * CONTROLS
        elif fork["place"] == "uncertain":
            first, second = fork["arms"][0]["force"], fork["arms"][-1]["force"]
            assert first != second and fork["arms"] == [{"force": first}] * FORCED + [{"force": second}] * FORCED
        else:
            assert fork["arms"] == [None] * DRAWS
    # The sampled segments kept their distributions: the policy's least sure decisions are forked.
    assert account["places"]["uncertain"] > 0
    assert account["forks"] == len(forks) and account["segments"] == wave["segments"] == sum(
        len(fork["arms"]) for fork in forks)
    # Every arm came back to its fork: none failed.
    assert wave["failed"] == {} and wave["written"] == wave["segments"] and wave["groups"] == len(forks)
    assert wave["plan"] == dict(count=6, places=list(PLACES), arms=dict(controls=CONTROLS, forced=FORCED, draws=DRAWS),
                                key="test:0", temperature=dict(fights.temperatures), policy_digest=fights.digest)
    named = arms(forks)
    segments = {run["run_id"]: run for window in RunShards(directory / HINDSIGHT).windows(shards=1) for run in window}
    assert set(segments) == set(named)
    first = {entry["run_id"]: entry["decisions"][0] for entry in digests.entries(directory / HINDSIGHT / digests.FILE)}
    hindsight_records = {row["run_id"]: row for row in fight_rows(directory, HINDSIGHT)}
    for run_id, (fork, arm) in named.items():
        run = segments[run_id]
        arm = arm or {}
        assert run["group"] == fork["group"] and run["hindsight"] == dict(
            potion=arm.get("potion"), force=arm.get("force"), replayed=len(fork["replay"]))
        # The first decision it keeps is the fork, the same frame for every arm.
        assert first[run_id][1] == fork["key"]
        ref, frame = drawn_steps(run)[0]
        fork_step = run["macros"][0]["steps"][0]
        if "potion" in arm:
            assert fork_step["explored"] is True and ref in drinks(frame)[arm["potion"]]
            assert arm["potion"] in hindsight_records[run_id]["potions_used"]
        elif "force" in arm:
            assert fork_step["forked"] is True and ref == arm["force"] and not fork_step.get("explored")
        else:
            assert not fork_step.get("explored") and not fork_step.get("forked")
        assert hindsight_records[run_id]["entry_hp"] == records[fork["source"]]["entry_hp"]
    # Priced as fight groups, each fork a group: at the fork every arm stands on the same
    # frame, and each action taken there has the mean of the arms that took it against the
    # fork's: of the arms that did not drink by force there (`segments.tree`).
    rows = fight_rows(directory, HINDSIGHT)
    returns = {row["run_id"]: costs.fight_return(row, costs.FIXED) for row in rows}
    segments = read_waves(directory, [HINDSIGHT])[HINDSIGHT]
    runs, _ = wave_advantages(segments, rows, costs.FIXED)
    for fork in forks:
        members = [segment for segment in segments if segment.group == fork["group"]]
        plain = [s for s in members if not s.decisions[0].explored] or members
        mean = sum(returns[s.run_id] for s in plain) / len(plain)
        if len({returns[s.run_id] for s in members}) == 1:
            continue
        for segment in members:
            index, value, advantage = runs[segment.run_id][0]
            alike = [returns[s.run_id] for s in members if s.decisions[0].action == segment.decisions[0].action]
            assert index == segment.decisions[0].index and value == pytest.approx(mean)
            assert advantage == pytest.approx(sum(alike) / len(alike) - mean)
    head = advantages(directory, costs.FIXED, wave=HINDSIGHT)
    assert head["wave"] == HINDSIGHT and head["runs"] == wave["written"]
    told = head["stats"]["hindsight"]
    assert set(told["places"]) <= set(PLACES) and sum(cell["groups"] for cell in told["places"].values()) == len(forks)
    assert all(cell["split"] <= cell["groups"] for cell in told["places"].values())
    assert set(told["potions"]) <= {BLOCK, FIRE} and set(told["drinks"]) <= {"danger"}
    assert sum(cell["groups"] for cell in told["uncertain"].values()) <= account["places"]["uncertain"]
    # Played again on the same forks, the finished wave stands.
    assert sample_hindsight(Idle(), fights, directory, 6, key="test:0")["fight_hindsight"] == wave
    with pytest.raises(ValueError):
        sample_hindsight(Idle(), greedy, directory, 6, key="test:0")


# ---- Learned beside the fight groups ----

def test_the_fight_groups_and_their_hindsight_forks_share_their_kinds_part_of_the_batch_by_decisions():
    shares = stream_shares({"act1": 30, "act2": 10}, {"fight-groups": {"fight-groups": 30, HINDSIGHT: 10}},
                           sources={"act-groups": 1, "fight-groups": 1})
    assert shares == pytest.approx({"act1": 0.375, "act2": 0.125, "fight-groups": 0.375, HINDSIGHT: 0.125})
    # One stream of a kind is named by the kind, as before.
    assert stream_shares({"act1": 30}, {"fight-groups": 10}) == pytest.approx({"act1": 0.75, "fight-groups": 0.25})


def test_hindsight_forks_are_sampled_after_the_fight_groups_priced_and_learned_beside_them(world, keep_uniform):
    """With a win's HP bonus beside them, as the fifth stretch runs: the act waves' wins priced with it;
    and a growth reward, the fight groups picked and priced with it."""
    # The tiny policy's batch of eight has a sample of every stream at these shares: forked
    # mid-fight, the arms are shorter than the fight groups' segments, hence many trunks.
    on = dict(eval_seeds=None, fight_groups=11, fight_group_size=3,
              wave_shares=[1, 1, 1], source_shares={"act-groups": 1, "fight-groups": 1})
    state = Driver(world.output, settings(world, fight_hindsight=11, win_hp_bonus=0.1, growth_bonus=0.5, **on),
                   checkpoints=world.checkpoints, engine=late_brawls, say=lambda _: None).run(1)
    entry = state["iterations"]["0"]
    assert entry["settings"]["fight_hindsight"] == 11 and entry["settings"]["win_hp_bonus"] == 0.1
    assert entry["sample"]["prices"] == potion_prices.digest(with_growth(FIXED, 0.5))
    assert "growth_per_segment" in entry["advantages"][WAVE]["kinds"]["boss"]
    assert all(entry["advantages"][wave]["win_bonus"]["missing"] == 0 for wave in ("act1", "act2", "act3"))
    sampled = entry["sample"]["fight_hindsight"]
    assert sampled["segments"] == sampled["written"] > 0 and sampled["failed"] == 0
    assert 0 < sampled["picks"]["trunks"] <= 11 and sampled["picks"]["segments"] == sampled["segments"]
    priced, update = entry["advantages"], entry["update"]
    assert set(priced) == {"act1", "act2", "act3", WAVE, HINDSIGHT} and priced[HINDSIGHT]["runs"] == sampled["written"]
    assert "hindsight" in priced[HINDSIGHT] and "hindsight" not in priced[WAVE]
    assert priced[WAVE]["explored_by_hp"] is not None
    # Two streams of the fight groups' kind, which share its part of the batch by their decisions.
    assert set(update["shares"]) == {"act1", "act2", "act3", WAVE, HINDSIGHT}
    assert update["shares"][WAVE] / update["shares"][HINDSIGHT] == pytest.approx(
        priced[WAVE]["decisions"] / priced[HINDSIGHT]["decisions"])
    assert update["fight_streams"][HINDSIGHT]["samples"] > 0 and update["fights"]["samples"] > 0
    assert set(update["mix"]) == {"act-groups", "fight-groups"}
    # Read last by the update, the wave goes at the iteration's cleanup.
    directory = world.output / "groups/round-0"
    for name in ("accepted", "index.jsonl.gz", digests.FILE, "fights.jsonl", "picks.json", "advantages.json.gz"):
        assert not (directory / HINDSIGHT / name).exists()
    # It needs fight groups; the CLI names it.
    for wrong in (dict(fight_hindsight=-1, **on), dict(on, fight_groups=0, fight_hindsight=4),
                  dict(on, win_hp_bonus=1.0), dict(on, win_hp_bonus=-0.1),
                  dict(on, growth_bonus=-0.5), dict(eval_seeds=None, growth_bonus=0.5)):
        with pytest.raises(ValueError):
            settings(world, **wrong)
    common = ["iterate", "--checkpoint", "c", "--output", "o", "--iterations", "1"]
    parsed = parser().parse_args([*common, "--fight-hindsight", "300", "--win-hp-bonus", "0.1"])
    assert (parsed.fight_hindsight, parsed.win_hp_bonus) == (300, 0.1)
    assert (parser().parse_args(common).fight_hindsight, parser().parse_args(common).win_hp_bonus) == (0, 0.0)
