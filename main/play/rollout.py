import hashlib
import json
import math
import random
import re
import time
import zlib
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import orjson
import torch

from main.network.policy import Choice, SessionPolicy
from main.data.index import COMPRESSION
from main.env.clock import timed
from main.env.control import CONTROL_VERSION, REWARD_RULE, environment_action, free_reward, is_policy_frame
from main.env.protocol import CHARACTERS, SCHEMA, ProtocolError, execution_command, fingerprint, segment_key, validate_frame
from main.encoding.frames import clean_frame
from main.env.rewards import MilestoneLedger
from main.env.public_history import require_history_version
from main.env.saves import FightTracker, continue_save
from main.play.digests import frame_key
from main.play.fights import FightLog, drinks
from main.play.fights import state as fight_state


# The precisions a run may be played at: bf16 autocasts the global layers, fp32 (or
# a profile's `no`) runs them as they are. The entity encoder and the head are FP32 either way.
PRECISIONS = ("bf16", "fp32")


def precision_context(model, precision):
    return (
        torch.autocast(model.device.type, dtype=torch.bfloat16)
        if precision == "bf16"
        else nullcontext()
    )


def numeric_backend(model):
    backend = {
        "device": model.device.type,
        "attention": model.config.backend,
        "torch": str(torch.__version__),
        "tf32": torch.backends.cuda.matmul.allow_tf32,
        "entity_encoder": "fp32",
    }
    # The fingerprint names the encoder precision only when it is not inherited.
    if model.config.encoder_precision != "inherit":
        backend["encoder_precision"] = model.config.encoder_precision
    return backend


def policy_digest(model):
    """The weights themselves: a version number does not tell two checkpoints apart."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def stand(frame):
    """Where a frame stands: (act, floor, floors of the act). The map counts its rows
    from zero and the player's floor from one, so an act is one floor longer than its
    last row. Each is None where the frame does not say."""
    entities = frame["public"]["entities"]
    hero = next((e for e in entities if e.get("entity_type") == "player"), {})
    act, floor = hero.get("act"), hero.get("floor")
    if type(act) is not int or type(floor) is not int:
        return None, None, None
    rows = [e["floor"] for e in entities
            if e.get("entity_type") == "map_node" and type(e.get("floor")) is int]
    return act, floor, max(rows) + 1 if rows else None


def climbed(stands):
    """The furthest floor of each act a run entered and how many floors those acts
    have, over where its recorded frames stood."""
    reached, lengths = {}, {}
    for act, floor, length in stands:
        if act is None:
            continue
        reached[act] = max(reached.get(act, 0), floor)
        if act not in lengths and length:
            lengths[act] = length
    return reached, lengths


def _pack(frame):
    """A frame as a run holds it while it is played: compressed. A run is hundreds of
    frames of tens of kilobytes that differ little; packed, a sampling process holds
    a run in a few hundred kilobytes instead of tens of megabytes. The text is
    orjson's, as the frame's line of a shard holds it (`packed_line`)."""
    with timed("frame_packing"):
        return zlib.compress(orjson.dumps(frame), 1)


def _unpack(packed):
    return orjson.loads(zlib.decompress(packed))


# Stands in a run's text for a packed frame: a string JSON writes as "\u0000<n>",
# which no frame or field of a run holds.
_FRAME = re.compile(r'"\\u0000(\d+)"')


def packed_line(run, on_policy=None):
    """A run whose frames are still packed (`run(..., packed=True)`) as the line it makes
    in a shard (`index.member`), with the count of its policy macro-actions by phase.
    `on_policy(index, macro, frame)` is told of every policy macro-action, the
    `index`-th of the run, with its first step's frame built (`play.digests`).

    Built whole, a run's frames are tens of megabytes of objects for a line of under
    a megabyte, and a sampling process keeps the pages of its largest run for as long
    as it lives. Here each frame goes into the line as the text it was packed from,
    one at a time; the line reads back as `member(run)` of the built run does."""
    frames = []

    def stand_in(frame):
        frames.append(frame)
        return f"\0{len(frames) - 1}"

    phases = Counter()
    macros = []
    for index, macro in enumerate(run["macros"]):
        if "steps" not in macro:
            macros.append(macro)
            continue
        # One frame built at a time, as `control.policy_macros` reads them.
        first = _unpack(macro["steps"][0]["frame"]) if macro["steps"] else None
        if macro["steps"] and (is_policy_frame(first) or any(is_policy_frame(_unpack(step["frame"]))
                                                             for step in macro["steps"][1:])):
            phases[macro["phase"]] += 1
            if on_policy is not None:
                on_policy(index, macro, first)
        macros.append(dict(macro, steps=[dict(step, frame=stand_in(step["frame"])) for step in macro["steps"]]))
    skeleton = dict(run, macros=macros)
    if "environment_actions" in run:
        skeleton["environment_actions"] = [dict(action, frame=stand_in(action["frame"]))
                                           for action in run["environment_actions"]]
    pieces = _FRAME.split(json.dumps(skeleton, ensure_ascii=False, allow_nan=False) + "\n")
    if [int(number) for number in pieces[1::2]] != list(range(len(frames))):
        raise ValueError("A run's text holds a frame's stand-in of its own")
    compressor = zlib.compressobj(COMPRESSION, zlib.DEFLATED, 31)  # a gzip member
    line = []
    for position, piece in enumerate(pieces):
        line.append(compressor.compress(
            zlib.decompress(frames[int(piece)]) if position % 2 else piece.encode("utf-8")))
    line.append(compressor.flush())
    return b"".join(line), dict(phases)


# What a round writes of each run. An evaluation of the command line keeps
# everything: the run with every frame, its journal and its fights.
FULL_RUNS = dict(trajectory=True, journal=True, combat_frames=True)
# What an evaluation of the iterations keeps: each run's outcome, ledger and climb,
# no frames and no journal; enough for its report and for `compare`.
SUMMARY_RUNS = dict(trajectory="summary", journal=False, combat_frames=False)
# Segments of runs played in act groups (`main/play/groups.py`): every frame, fights
# included (a round with greedy fights keeps none of theirs), no run file of its own
# (the segment makes a line in a shard), and the continue save where a segment stops.
SEGMENTS = dict(trajectory=False, journal=False, combat_frames=True, segment=True)


class RolloutRunner:
    def __init__(
        self,
        model,
        vocabulary,
        *,
        precision="no",
        version=0,
        max_steps=10000,
        timeout=30,
        ascension=0,
        claim_rewards=False,
        temperatures=None,
        explore_outside=0.0,
        explore_prior=0.0,
        prior=None,
        greedy_fights=False,
        iteration=None,
    ):
        if max_steps < 1 or timeout <= 0:
            raise ValueError("Rollout limits must be positive")
        explore_outside = outside_rates(explore_outside)
        explore_prior = outside_rates(explore_prior)
        temperatures = {"combat": 1.0, "other": 1.0} if temperatures is None else dict(temperatures)
        if set(temperatures) != {"combat", "other"} or not all(t > 0 for t in temperatures.values()):
            raise ValueError("The sampling temperatures must be positive, one for fights and one for the rest")
        if type(greedy_fights) is not bool:
            raise ValueError("Greedy fights are on or off")
        if type(ascension) is not int or not 0 <= ascension <= 10:
            raise ValueError("Ascension must be an integer from 0 to 10")
        if iteration is not None and (type(iteration) is not int or iteration < 0):
            raise ValueError("The sampling iteration must be a non-negative integer")
        self.model, self.vocabulary = model, vocabulary
        self.precision, self.version = precision, version
        self.max_steps, self.timeout = max_steps, timeout
        self.ascension = ascension
        # Gold, relics and potions the belt has room for are taken for the policy.
        self.claim_rewards = claim_rewards
        # Sampled choices are drawn from the scores divided by the temperature of the
        # decision's phase: under 1 the policy plays closer to its most likely actions,
        # which are the same at every temperature. The temperatures are part of the
        # policy's definition: a run records them.
        self.temperatures = temperatures
        # Exploration outside fights, a probability per phase (`outside_rates`): with it a
        # sampled decision of that phase takes a candidate other than the policy's most
        # likely, half the time the second most likely and otherwise one of the rest at
        # random, never a potion's discard. The policy grows sure of its choices outside
        # fights as it trains, and at the sampling temperature a candidate under a few
        # percent is never drawn: no group then holds a segment that took it, and no
        # update can find it better. The decision is recorded with the policy's own
        # log-probability of the candidate it took, as if the policy had drawn it; the
        # update tells it apart (`explored`) and learns it by its score.
        self.explore_outside = explore_outside
        # Proposals of a prior outside fights, a probability per phase (`outside_rates`):
        # with it a sampled decision of that phase asks the prior (another network of the
        # same vocabulary, `prior`, or the policy handed in's `prior`) for its most likely
        # candidate and, where that is not the policy's most likely, takes it in place of
        # the draw, recorded as an explored decision of the prior's (`explored_by`) with
        # the policy's own log-probabilities. Where they agree the decision goes on to
        # `explore_outside` as any other. The prior proposes what the policy has stopped
        # trying; the update alone says whether it pays (ticket 33 of .scratch/training-plan).
        self.explore_prior = explore_prior
        self.prior = prior
        # Greedy fights: a sampled run takes every decision inside a fight by the
        # policy's most likely action, as a greedy run does, and draws only the rest at
        # their temperature. The fights are then the policy's fixed play, and whatever
        # sets the segments of a group apart is the decisions outside them; a round of
        # act groups keeps no frames of its fights (`groups.sample_groups`), so that no
        # update learns from them. They explore nothing.
        self.greedy_fights = greedy_fights
        self._random = random.Random()
        # The sampling iteration the runs belong to.
        self.iteration = iteration
        self._digest = None

    @property
    def digest(self):
        if self._digest is None:
            self._digest = policy_digest(self.model)
        return self._digest

    def variant(self, **changes):
        """A runner of the same policy (model, vocabulary, digest) with some of its
        settings changed: a round's fight groups are played by its runner without
        greedy fights (`main.play.fight_groups`)."""
        settings = dict(precision=self.precision, version=self.version, max_steps=self.max_steps,
                        timeout=self.timeout, ascension=self.ascension, claim_rewards=self.claim_rewards,
                        temperatures=self.temperatures, explore_outside=self.explore_outside, explore_prior=self.explore_prior,
                        prior=self.prior, greedy_fights=self.greedy_fights, iteration=self.iteration)
        if not set(changes) <= set(settings):
            raise ValueError(f"A runner's settings are {sorted(settings)}")
        runner = RolloutRunner(self.model, self.vocabulary, **{**settings, **changes})
        # The same weights: hashed once.
        runner._digest = self.digest
        return runner

    def temperature_for(self, frame, sample=True, fight=None):
        """The temperature a decision of `frame` is drawn at: a fight's or the rest's.
        `fight` is the fight the decision belongs to (`saves.FightTracker`): a fight is
        what the engine's encounter events bound, so a selection made inside one (a
        discard, an exhaust) is drawn at the fight's temperature whatever its phase.
        The most likely action, which a greedy run takes, is scored at 1. With
        `greedy_fights` a sampled decision inside a fight is not drawn: None, it takes
        the most likely action scored at 1 (`run`'s `choose`). The fight is the
        tracker's alone, as it is for the frames a round keeps (`combat_frames`)."""
        if not sample:
            return 1.0
        if self.greedy_fights and fight is not None:
            return None
        inside = fight is not None or frame["public"]["phase"] == "combat"
        return self.temperatures["combat" if inside else "other"]

    def explore_by_prior(self, frame, choice, temperature, prior):
        """`choice`, the policy's draw at a sampled decision outside fights, or in its
        place the most likely candidate of `prior` (a policy of the prior network), at the
        rate of the frame's phase (`explore_prior`), where that is neither the policy's
        most likely nor a potion's discard. The candidate is recorded as
        `explore_outside_fights` records its own, marked as the prior's."""
        rate = self.explore_prior.get(frame["public"]["phase"], 0.0)
        candidates = frame["legal"]["candidates"]
        if not rate or prior is None or len(candidates) < 2 or self._random.random() >= rate:
            return choice
        drawn = getattr(choice, "log_probs", None)
        if drawn is None:
            raise ValueError("Exploring outside fights needs the distribution the decision was drawn from")
        proposed = prior.choose(frame, sample=False, temperature=1.0).candidate_ref
        top = max(range(len(candidates)), key=lambda index: drawn[index])
        taken = next(index for index, candidate in enumerate(candidates) if candidate["candidate_ref"] == proposed)
        if taken == top or candidates[taken]["verb"] == "DISCARD_POTION":
            return choice
        return _explored(candidates, drawn, taken, temperature, by="prior")

    def explore_outside_fights(self, frame, choice, temperature):
        """`choice`, the policy's draw at a sampled decision outside fights, or in its
        place a candidate other than the policy's most likely, at the rate of the frame's
        phase (`explore_outside`): the second most likely half the time, otherwise one of
        the others at random; a potion's discard is never taken. The candidate is recorded
        with the policy's own log-probabilities of it, at `temperature` and at 1, from the
        distribution the draw came with."""
        rate = self.explore_outside.get(frame["public"]["phase"], 0.0)
        candidates = frame["legal"]["candidates"]
        if not rate or len(candidates) < 2 or self._random.random() >= rate:
            return choice
        drawn = getattr(choice, "log_probs", None)
        if drawn is None:
            raise ValueError("Exploring outside fights needs the distribution the decision was drawn from")
        top = max(range(len(candidates)), key=lambda index: drawn[index])
        others = [index for index, candidate in enumerate(candidates)
                  if index != top and candidate["verb"] != "DISCARD_POTION"]
        if not others:
            return choice
        if self._random.random() < 0.5:
            taken = max(others, key=lambda index: drawn[index])
        else:
            taken = self._random.choice(others)
        return _explored(candidates, drawn, taken, temperature)

    def run(self, engine, character, seed, *, sample=True, journal=None,
            combat_frames=True, resume=None, until_act=None, until_fight_end=False,
            run_id=None, group=None, ascension=None, packed=False, fight_entries=False, entry_saves=True,
            hindsight=None, _policy=None):
        """Play one run, or a segment of one.

        With `resume` (a save a segment stopped at, as `save` below) the run goes on
        from that save instead of starting: at the map (`engine.resume`) or at the
        fight it was written before (`engine.resume_fight`); `character` and `seed`
        are the saved run's. With `until_act` the run stops at the first decision of
        that act; the decision is not part of this segment. The engine's continue save
        there is returned as `save`, `{"at": "act", "save": …}`, and the segment's
        `outcome` is `reached`; a run that ended first has `outcome` `win` or `loss`.
        `run_id` and `group` name the segment and the group of segments it belongs to
        (`main.play.groups`). `ascension` is the run's
        (the runner's own without one); a save goes on at the ascension it was
        written at, which must be the one asked for.

        With `hindsight` (`replay`: the candidates another segment of the same fight
        took at its decisions with a choice, in order; `key`: the frame they led it to,
        `digests.frame_key`; `potion`: a content id or None; `force`: a candidate or
        None, not both) a segment resumed from a
        fight's entry save to the fight's end takes those actions again, and the
        decisions without a choice between them, without drawing or keeping them: the
        same actions bring the fight back to the same frame (`hindsight` in the trace
        says how many it replayed). Its first decision with a choice after them is the
        fork, which must be at `key`: there it drinks `potion`, among that potion's
        candidates by the policy, as an explored decision, or takes `force` with the
        policy's own log-probability, as a forked decision; with neither it draws as it
        would. It goes on drawing to the fight's end. A replay that does not come back
        to its key, a potion the fork cannot drink or a candidate it does not offer is
        an error of the segment.

        Without `combat_frames` the macro-actions played inside fights keep their
        counts and log-probability but not their frames, which are most of a run.
        With `packed` the frames stay as the run held them while it was played
        (`_pack`), for `packed_line` to write without building them again.

        With `fight_entries` the run keeps a record of every fight it plays and writes
        the entry save of every fight not begun by an event (`fights.FightLog`), returned
        as `fight_records`; the decisions and the frames are what they are without.
        Without `entry_saves` it keeps the records alone. A run that keeps records and
        stops at an act also returns where the player stands there and what they have
        (`fights.state` of the decision it stops at) as `cut_state`.

        With `until_fight_end` a segment resumed from a fight's entry save (`resume` at
        "fight") plays that fight alone: it stops at the first decision that does not
        belong to the fight, which is not part of the segment (`outcome` `reached`, no
        save), or with the run's end inside it (`win` or `loss`). The first decision of
        another fight is such a decision: the last act's two bosses may follow each
        other with no decision between them. The segment keeps its fight's record, the
        HP after it taken at the decision it stopped at, as `fight_records`, and writes
        no entry save: it was resumed from one."""
        if character not in CHARACTERS:
            raise ValueError("Unknown character")
        if until_fight_end and (resume is None or resume.get("at") != "fight"):
            raise ValueError("A segment plays to its fight's end from the fight's entry save")
        if until_fight_end and until_act is not None:
            raise ValueError("A segment stops at its fight's end or at an act, one of them")
        if hindsight is not None and (not until_fight_end or not sample or self.greedy_fights):
            raise ValueError("A hindsight segment replays a sampled fight segment from its entry save, drawing "
                             "after its fork")
        if hindsight is not None and hindsight.get("potion") is not None and hindsight.get("force") is not None:
            raise ValueError("A hindsight fork drinks a potion or takes a candidate, not both")
        ascension = self.ascension if ascension is None else ascension
        if type(ascension) is not int or not 0 <= ascension <= 10:
            raise ValueError("Ascension must be an integer from 0 to 10")
        trace = {
            "schema": SCHEMA,
            "character": character,
            "seed": str(seed),
            "ascension": ascension,
            "policy_version": self.version,
            "policy_digest": self.digest,
            "precision": self.precision,
            "vocabulary_hash": self.vocabulary.digest,
            "numeric_backend": numeric_backend(self.model),
            "source": "rollout" if sample else "evaluation",
            "sampling": "full_distribution" if sample else "argmax",
            "temperature": dict(self.temperatures) if sample else None,
            "iteration": self.iteration,
            "status": "unresolved",
            "macros": [],
            "automatic_steps": 0,
            "initial_reward": 0.0,
            "reward_version": MilestoneLedger().version,
            "control_version": CONTROL_VERSION,
            "reward_claim_rule": REWARD_RULE if self.claim_rewards else None,
            "environment_actions": [],
        }
        if sample and self.explore_outside:
            trace["explore_outside"] = dict(self.explore_outside)
        if sample and self.explore_prior:
            trace["explore_prior"] = dict(self.explore_prior)
        if group is not None:
            trace["group"] = group
        if hindsight is not None:
            trace["hindsight"] = dict(potion=hindsight.get("potion"), force=hindsight.get("force"), replayed=0)
        if sample and self.greedy_fights:
            # Its fights' macro-actions record no temperature: they were not drawn.
            trace["greedy_fights"] = True
        if resume is not None or until_act is not None:
            trace["segment"] = {"resumed": resume is not None, "until_act": until_act}
            if until_fight_end:
                trace["segment"]["until_fight_end"] = True
            trace["outcome"] = None
        ledger = MilestoneLedger()
        policy = _policy or SessionPolicy(
            self.model, self.vocabulary, version=self.version
        )
        # The prior of `explore_prior`: the policy handed in's, or a session of `prior` here.
        if _policy is not None:
            prior = getattr(_policy, "prior", None)
        else:
            prior = None if self.prior is None else SessionPolicy(self.prior, self.vocabulary)

        # A policy handed in draws its choices where the network runs; either is told
        # the temperature of each decision here, where the fight it belongs to is known.
        def choose(frame, fight, *, sample, restrict=None):
            temperature = self.temperature_for(frame, sample, fight)
            if temperature is None:
                # A fight's decision with greedy fights: the most likely action, as a
                # greedy run takes it, scored at 1; it explores nothing.
                return policy.choose(frame, sample=False, temperature=1.0)
            # A hindsight fork drinks the potion it was given (`restrict`). A policy that
            # does not explore (a test double) is asked nothing it does not know.
            among = {} if restrict is None else {"restrict": restrict}
            chosen = policy.choose(frame, sample=sample, temperature=temperature, **among)
            if sample and fight is None and restrict is None and frame["public"]["phase"] != "combat":
                if self.explore_prior:
                    chosen = self.explore_by_prior(frame, chosen, temperature, prior)
                if self.explore_outside and not getattr(chosen, "explored", False):
                    chosen = self.explore_outside_fights(frame, chosen, temperature)
            return chosen

        pending_steps, current_segment, active_macro = [], None, None
        waiting_since = None
        fights, stands = FightTracker(), []
        # A segment played to its fight's end keeps the fight's record, not its entry save again.
        log = FightLog(saves=entry_saves and not until_fight_end) if fight_entries or until_fight_end else None
        started = None  # with `until_fight_end`, the fight the segment plays
        # With `hindsight`, the actions to take again before the fork and how many were.
        replay, replayed = (None, 0) if hindsight is None else (list(hindsight["replay"]), 0)
        # A hindsight fork's candidate given (`force`): its step is kept as a forked one.
        forced = False
        sink = Path(journal).open("w") if journal else nullcontext()
        self.model.eval()
        try:
            with (
                sink as stream,
                torch.no_grad(),
                # A policy handed in runs the network elsewhere, under its own context;
                # a sampling process must not even ask what its device supports.
                nullcontext() if _policy else precision_context(self.model, self.precision),
            ):
                frame = engine.reset(character, seed, ascension) if resume is None else _resumed(engine, resume)
                require_history_version(frame.get("contract", {}).get("public_history_version"))
                if frame.get("contract", {}).get("fixed_ascension") != ascension:
                    raise ProtocolError("Engine started a different ascension than requested")
                trace["run_id"] = run_id or frame.get("routing", {}).get(
                    "episode_id", fingerprint([character, seed])
                )
                trace["contract"] = deepcopy(frame.get("contract", {}))
                steps = 0
                while True:
                    validate_frame(frame)
                    if frame["contract"] != trace["contract"]:
                        raise ProtocolError("Engine contract changed during run")
                    reward = ledger.apply(frame.get("events", []))
                    boundary = frame["boundary"]
                    if boundary == "terminal":
                        victory = bool(
                            frame.get("public", {}).get("outcome", {}).get("victory")
                        )
                        reward += ledger.finish(victory)
                    if trace["macros"]:
                        trace["macros"][-1]["reward"] += reward
                    else:
                        trace["initial_reward"] += reward
                    if stream:
                        stream.write(
                            json.dumps(
                                {"frame": frame, "reward": reward}, allow_nan=False
                            )
                            + "\n"
                        )
                        stream.flush()
                    fights.events(frame.get("events", []))
                    if boundary == "terminal":
                        fights.finish(victory)
                        if log is not None:
                            log.ended(frame, victory)
                        trace.update(status="complete", victory=victory)
                        if "segment" in trace:
                            trace["outcome"] = "win" if victory else "loss"
                        break
                    if boundary == "waiting":
                        waiting_since = waiting_since or time.monotonic()
                        if time.monotonic() - waiting_since > self.timeout:
                            raise TimeoutError("unresolved_wait_timeout")
                        time.sleep(0.01)
                        frame = engine.send({"cmd": "advance_to_boundary"})
                        continue
                    waiting_since = None
                    if steps >= self.max_steps:
                        trace["error"] = "unresolved_step_limit"
                        break
                    with timed("cleaning"):
                        frame = clean_frame(frame)
                    act, floor, length = stand(frame)
                    if until_act is not None and act is not None and act >= until_act:
                        # The first decision of the act the segment stops at belongs to
                        # the next segment; the save there is where it goes on from.
                        trace["save"] = dict(at="act", save=continue_save(engine))
                        if log is not None:
                            log.passed(frame, None)
                            trace["cut_state"] = fight_state(frame)
                        trace.update(status="complete", victory=None, outcome="reached", reached_act=act)
                        break
                    fight, entry = fights.decision()
                    if entry is not None and fight == 1 and resume is not None and resume.get("at") == "fight":
                        # The fight a run resumed at is its save's: begun there, the tracker
                        # has not seen the act's first boss, which the save says it follows.
                        entry = dict(entry, second_boss=bool(resume["fight"].get("second_boss", entry["second_boss"])))
                    if log is not None:
                        log.passed(frame, fight)
                    if until_fight_end:
                        if started is None:
                            if entry is None:
                                raise ProtocolError("A segment resumed at a fight began off its first decision")
                            started = fight
                        elif fight != started:
                            # The first decision after the fight, or another fight's first:
                            # its record took the HP after the fight here (`log.passed`).
                            trace.update(status="complete", victory=None, outcome="reached")
                            break
                    if log is not None:
                        log.entered(engine, frame, fight, entry, floor)
                    stands.append((act, floor, length))
                    automatic = environment_action(frame)
                    if automatic is None and self.claim_rewards:
                        automatic = free_reward(frame)
                    if automatic is not None:
                        policy.reset()
                        current_segment, pending_steps, active_macro = None, [], None
                        trace["environment_actions"].append(dict(frame=_pack(frame), **automatic))
                        trace["automatic_steps"] += 1
                        steps += 1
                        frame = engine.send(execution_command(frame, automatic["candidate_ref"]))
                        continue
                    branching = len(frame["legal"]["candidates"]) > 1
                    drink = None
                    if replay is not None:
                        if replayed < len(replay) or not branching:
                            # Before the fork: the actions given, neither drawn nor kept.
                            ref = replay[replayed] if branching else frame["legal"]["candidates"][0]["candidate_ref"]
                            if branching and all(c["candidate_ref"] != ref for c in frame["legal"]["candidates"]):
                                raise ProtocolError(f"Hindsight replay diverged at its decision {replayed}")
                            replayed += branching
                            trace["hindsight"]["replayed"] = replayed
                            if log is not None:
                                log.chose(frame, fight, ref)
                            steps += 1
                            frame = engine.send(execution_command(frame, ref))
                            continue
                        if frame_key(frame) != hindsight["key"]:
                            raise ProtocolError("Hindsight replay came to another frame than its fork's")
                        replay = None
                        if hindsight.get("potion") is not None:
                            drink = drinks(frame).get(hindsight["potion"])
                            if not drink:
                                raise ProtocolError(f"The fork cannot drink {hindsight['potion']}")
                        if hindsight.get("force") is not None:
                            if all(c["candidate_ref"] != hindsight["force"] for c in frame["legal"]["candidates"]):
                                raise ProtocolError("The fork does not offer its arm's candidate")
                            drink, forced = [hindsight["force"]], True
                    key = segment_key(frame)
                    if key != current_segment:
                        current_segment, pending_steps, active_macro = key, [], None
                    choice = choose(frame, fight, sample=sample, restrict=drink)
                    if log is not None:
                        log.chose(frame, fight, choice.candidate_ref)
                    # A frame played inside a fight is kept only where every frame is.
                    step = {
                        "frame": _pack(frame) if combat_frames or fight is None else None,
                        "candidate_ref": choice.candidate_ref,
                        "forced": not branching,
                    }
                    if forced:
                        # The fork's candidate given, not drawn.
                        step["forked"], forced = True, False
                    elif getattr(choice, "explored", False):
                        step["explored"] = True
                        if getattr(choice, "explored_by", None):
                            step["explored_by"] = choice.explored_by
                    if branching and getattr(choice, "log_probs", None) is not None:
                        # The distribution it was drawn from: the update's KL may take it for its reference's.
                        step["log_probs"] = choice.log_probs
                    if branching:
                        if active_macro is None:
                            active_macro = {
                                "phase": frame["public"]["phase"],
                                "fight": fight,
                                "temperature": self.temperature_for(frame, sample, fight),
                                "old_log_prob": 0.0,
                                "base_log_prob": 0.0,
                                "reward": 0.0,
                                "steps": pending_steps,
                            }
                            trace["macros"].append(active_macro)
                        active_macro["old_log_prob"] += float(choice.log_prob)
                        _add_base(active_macro, choice)
                    else:
                        trace["automatic_steps"] += 1
                    if active_macro is None:
                        pending_steps.append(step)
                    else:
                        active_macro["steps"].append(step)
                    steps += 1
                    frame = engine.send(execution_command(frame, choice.candidate_ref))
                if replay is not None and trace["status"] == "complete":
                    raise ProtocolError("Hindsight replay ended before its fork")
        except (ProtocolError, TimeoutError, OSError, ValueError) as exc:
            trace["status"] = "unresolved" if isinstance(exc, TimeoutError) else "error"
            trace["error"] = str(exc)
        trace["ledger"] = ledger.state_dict()
        reached, lengths = climbed(stands)
        trace["climb"] = {"reached": reached, "lengths": lengths}
        trace["fights"] = fights.tally()
        if log is not None:
            trace["fight_records"] = log.records(trace)
        for macro in trace["macros"]:
            steps = macro.pop("steps")
            if macro["fight"] is not None and not combat_frames:
                macro["branches"] = sum(not step["forced"] for step in steps)
                macro["decisions"] = len(steps)
            elif packed:
                macro["steps"] = steps
            else:
                macro["steps"] = [dict(step, frame=_unpack(step["frame"])) for step in steps]
        if not packed:
            for action in trace["environment_actions"]:
                action["frame"] = _unpack(action["frame"])
        return trace


def _explored(candidates, drawn, taken, temperature, by=None):
    """The explored choice of candidate `taken`, with the policy's own log-probabilities
    of it from `drawn` (at `temperature`) and at 1, marked as `by`'s proposal."""
    # At temperature 1 the scores are the tempered log-probabilities times the temperature.
    scaled = [temperature * value for value in drawn]
    peak = max(scaled)
    total = peak + math.log(sum(math.exp(value - peak) for value in scaled))
    return Choice(candidates[taken]["candidate_ref"], drawn[taken], base_log_prob=scaled[taken] - total,
                  explored=True, log_probs=list(drawn), explored_by=by)


# The phases of the decisions outside fights (`explore_outside`).
OUTSIDE_PHASES = ("map", "rewards", "card_reward", "event", "rest_site", "shop", "treasure", "card_select",
                  "bundle_select", "crystal_sphere")


def outside_rates(rates):
    """The exploration rate outside fights of each phase that explores (`OUTSIDE_PHASES`),
    from one probability for them all or a mapping by phase; phases at 0 are left out."""
    if isinstance(rates, dict):
        if not set(rates) <= set(OUTSIDE_PHASES):
            raise ValueError(f"Exploration outside fights is given by phase {OUTSIDE_PHASES}")
    else:
        rates = {phase: rates for phase in OUTSIDE_PHASES}
    for value in rates.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("The exploration rate outside fights is a probability")
    return {phase: float(value) for phase, value in rates.items() if value}


def _resumed(engine, resume):
    """The first frame of a run resumed from the save a segment stopped at."""
    if resume.get("at") == "fight":
        return engine.resume_fight(dict(resume["fight"], save=resume["save"]))
    if resume.get("at") == "act":
        return engine.resume(resume["save"])
    raise ProtocolError("A segment resumes from a save written at an act or at a fight")


def _add_base(macro, choice):
    """Add a choice's log-probability at temperature 1 to its macro-action's; a
    choice that did not say it leaves the macro-action without one."""
    base = getattr(choice, "base_log_prob", None)
    if base is None or macro.get("base_log_prob") is None:
        macro["base_log_prob"] = None
    else:
        macro["base_log_prob"] += float(base)


def write_run(path, trace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(trace, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def play(engine, runner, job, *, sample=True, policy=None, trajectory=True, journal=True,
         combat_frames=True, segment=False, fight_entries=False, entry_saves=True):
    """One job of a round: a run and, as the layout says (`FULL_RUNS`,
    `SUMMARY_RUNS`), its journal and its run file. With `segment` the job is
    (character, spec, None): a segment of a run (`main.play.groups`), `spec` naming
    its seed, `run_id`, `group`, `until_act` or `until_fight_end`, its
    `ascension` if not the runner's and, to go on from a save, the file the save is in
    (`resume`); with `until_fight_end` true it plays from a fight's entry save to the
    fight's end and keeps the fight's record (`RolloutRunner.run`), with `hindsight`
    replaying another segment's actions to a fork first. With `fight_entries`
    it keeps its fights' records and, unless not `entry_saves`, their entry saves
    (`fights.FightLog`).
    Returns the run; one that no file of its own is written for keeps its frames packed, for the line it
    makes in a shard (`packed_line`)."""
    if fight_entries and not segment:
        raise ValueError("Fight entries are kept for the segments of act groups")
    if segment:
        from main.play.groups import read_save

        character, spec, _ = job
        return runner.run(
            engine, character, spec["seed"], sample=sample, combat_frames=combat_frames,
            resume=read_save(spec["resume"]) if spec.get("resume") else None,
            until_act=spec.get("until_act"), until_fight_end=spec.get("until_fight_end", False),
            run_id=spec["run_id"], group=spec["group"], ascension=spec.get("ascension"),
            packed=not trajectory, fight_entries=fight_entries, entry_saves=entry_saves,
            hindsight=spec.get("hindsight"), _policy=policy)
    character, seed, path = job
    trace = runner.run(
        engine,
        character,
        seed,
        sample=sample,
        journal=path.with_suffix(".jsonl") if journal else None,
        combat_frames=combat_frames,
        packed=not trajectory,
        _policy=policy,
    )
    if trajectory:
        write_run(path, trace if trajectory is True else {
            key: value for key, value in trace.items() if key not in ("macros", "environment_actions")})
    return trace


def assign(seeds, directory):
    """The jobs of a round, in a directory of its own: the assignment is saved before
    any engine starts, including queues which later fail."""
    counts = {c: len(seeds.get(c, [])) for c in CHARACTERS}
    if len(set(counts.values())) != 1 or not all(counts.values()):
        raise ValueError("Allocate equal nonzero run counts to all five characters")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob("*.json")) or (directory / "index.jsonl.gz").exists():
        raise FileExistsError(
            "Rollout directory already contains runs; choose a new round directory"
        )
    # Keep metadata outside the trajectory JSON glob.
    assignment = directory / "metadata" / "seeds.json"
    if assignment.exists():
        raise FileExistsError(
            "Rollout seed assignment already exists; choose a new round directory"
        )
    write_run(assignment, seeds)
    return [
        (character, seed, directory / f"{character}-{i}.json")
        for character in CHARACTERS
        for i, seed in enumerate(seeds[character])
    ]


def collect_round(samplers, runner, seeds, directory, on_run=None, *, sample=True, layout=None):
    """Freeze the assigned queue before running it; any failure blocks what follows.

    The round is played by the engines of `samplers` (`sampling.Samplers`), each run
    in an engine of its own, while this process only runs the network. Returns where
    the runs were written, in the order of the jobs: every run with every frame,
    beside its journal. The segments of act groups are written by
    `groups.sample_groups` instead.
    """
    jobs = assign(seeds, directory)
    played = samplers.collect(runner, jobs, sample=sample, on_run=on_run, layout=layout)
    failures = [str(path) for (_, _, path), run in zip(jobs, played) if run["status"] != "complete"]
    if failures and sample:
        raise ProtocolError(
            f"Round incomplete; a sampled round must finish every run. Inspect: {failures}"
        )
    return [path for _, _, path in jobs]
