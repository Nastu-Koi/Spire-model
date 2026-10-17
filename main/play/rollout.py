import hashlib
import json
import random
import re
import time
import zlib
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from main.network.policy import SessionPolicy
from main.data.index import COMPRESSION
from main.env.control import CONTROL_VERSION, REWARD_RULE, environment_action, free_reward, is_policy_frame
from main.env.protocol import CHARACTERS, SCHEMA, ProtocolError, execution_command, fingerprint, segment_key, validate_frame
from main.encoding.frames import clean_frame
from main.env.rewards import MilestoneLedger
from main.env.public_history import require_history_version
from main.env.saves import FightTracker, continue_save


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
    a run in a few hundred kilobytes instead of tens of megabytes."""
    return zlib.compress(json.dumps(frame, separators=(",", ":"), allow_nan=False).encode(), 1)


def _unpack(packed):
    return json.loads(zlib.decompress(packed))


# Stands in a run's text for a packed frame: a string JSON writes as "\u0000<n>",
# which no frame or field of a run holds.
_FRAME = re.compile(r'"\\u0000(\d+)"')


def packed_line(run):
    """A run whose frames are still packed (`run(..., packed=True)`) as the line it makes
    in a shard (`index.member`), with the count of its policy macro-actions by phase.

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
    for macro in run["macros"]:
        if "steps" not in macro:
            macros.append(macro)
            continue
        # One frame built at a time, as `control.policy_macros` reads them.
        if any(is_policy_frame(_unpack(step["frame"])) for step in macro["steps"]):
            phases[macro["phase"]] += 1
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
# included, no run file of its own (the segment makes a line in a shard), and the
# continue save where a segment stops.
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
        explore_potions=0.0,
        iteration=None,
    ):
        if max_steps < 1 or timeout <= 0:
            raise ValueError("Rollout limits must be positive")
        if not isinstance(explore_potions, (int, float)) or not 0 <= explore_potions <= 1:
            raise ValueError("The potion exploration rate is a probability")
        temperatures = {"combat": 1.0, "other": 1.0} if temperatures is None else dict(temperatures)
        if set(temperatures) != {"combat", "other"} or not all(t > 0 for t in temperatures.values()):
            raise ValueError("The sampling temperatures must be positive, one for fights and one for the rest")
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
        # Exploration of potion use: with this probability a sampled fight decision
        # that could use a potion draws one of its USE_POTION candidates instead of a
        # sample from the tempered policy. The policy puts too little on using potions
        # for the sampling temperature ever to draw one, so without it no group ever
        # holds a segment that used one to compare against. An explored decision is
        # recorded with the policy's own log-probability of the action, as if the
        # policy had drawn it: the update then weighs it by its advantage in full.
        self.explore_potions = float(explore_potions)
        self._random = random.Random()
        # The sampling iteration the runs belong to.
        self.iteration = iteration
        self._digest = None

    @property
    def digest(self):
        if self._digest is None:
            self._digest = policy_digest(self.model)
        return self._digest

    def temperature_for(self, frame, sample=True, fight=None):
        """The temperature a decision of `frame` is drawn at: a fight's or the rest's.
        `fight` is the fight the decision belongs to (`saves.FightTracker`): a fight is
        what the engine's encounter events bound, so a selection made inside one (a
        discard, an exhaust) is drawn at the fight's temperature whatever its phase.
        The most likely action, which a greedy run takes, is scored at 1."""
        if not sample:
            return 1.0
        inside = fight is not None or frame["public"]["phase"] == "combat"
        return self.temperatures["combat" if inside else "other"]

    def exploration_for(self, frame, fight, sample=True):
        """The candidates a decision of `frame` is drawn among when it explores potion
        use (`explore_potions`), or None for a decision drawn from the policy: a sampled
        decision inside a fight that could use a potion, with the exploration rate."""
        if not sample or fight is None or not self.explore_potions:
            return None
        potions = [c["candidate_ref"] for c in frame["legal"]["candidates"] if c["verb"] == "USE_POTION"]
        if not potions or len(frame["legal"]["candidates"]) < 2 or self._random.random() >= self.explore_potions:
            return None
        return potions

    def run(self, engine, character, seed, *, sample=True, journal=None,
            combat_frames=True, resume=None, until_act=None, until_boss=None, run_id=None, group=None,
            packed=False, _policy=None):
        """Play one run, or a segment of one.

        With `resume` (a save a segment stopped at, as `save` below) the run goes on
        from that save instead of starting: at the map (`engine.resume`) or at the
        fight it was written before (`engine.resume_fight`); `character` and `seed`
        are the saved run's. With `until_act` the run stops at the first decision of
        that act, with `until_boss` at the first decision of that act's (first) boss
        fight; the decision is not part of this segment. The engine's continue save
        there is returned as `save`, `{"at": "act", "save": …}` or `{"at": "fight",
        "save": …, "fight": …}` (the fight as `saves.FightTracker` describes it, with
        its floor), and the segment's `outcome` is `reached`; a run that ended first
        has `outcome` `win` or `loss`. `run_id` and `group` name the segment and the
        group of segments it belongs to (`main.play.groups`).

        Without `combat_frames` the macro-actions played inside fights keep their
        counts and log-probability but not their frames, which are most of a run.
        With `packed` the frames stay as the run held them while it was played
        (`_pack`), for `packed_line` to write without building them again."""
        if character not in CHARACTERS:
            raise ValueError("Unknown character")
        trace = {
            "schema": SCHEMA,
            "character": character,
            "seed": str(seed),
            "ascension": self.ascension,
            "policy_version": self.version,
            "policy_digest": self.digest,
            "precision": self.precision,
            "vocabulary_hash": self.vocabulary.digest,
            "numeric_backend": numeric_backend(self.model),
            "source": "rollout" if sample else "evaluation",
            "sampling": "full_distribution" if sample else "argmax",
            "temperature": dict(self.temperatures) if sample else None,
            "explore_potions": self.explore_potions if sample else None,
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
        if group is not None:
            trace["group"] = group
        if until_act is not None and until_boss is not None:
            raise ValueError("A segment stops at an act or at a boss fight, not both")
        if resume is not None or until_act is not None or until_boss is not None:
            trace["segment"] = {"resumed": resume is not None, "until_act": until_act, "until_boss": until_boss}
            trace["outcome"] = None
        ledger = MilestoneLedger()
        policy = _policy or SessionPolicy(
            self.model, self.vocabulary, version=self.version
        )

        # A policy handed in draws its choices where the network runs; either is told
        # the temperature of each decision here, where the fight it belongs to is known.
        def choose(frame, fight, *, sample):
            restrict = self.exploration_for(frame, fight, sample)
            # A policy that does not explore (a test double) is asked nothing it does not know.
            among = {} if restrict is None else {"restrict": restrict}
            return policy.choose(frame, sample=sample, temperature=self.temperature_for(frame, sample, fight), **among)

        pending_steps, current_segment, active_macro = [], None, None
        waiting_since = None
        fights, stands = FightTracker(), []
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
                frame = engine.reset(character, seed, self.ascension) if resume is None else _resumed(engine, resume)
                require_history_version(frame.get("contract", {}).get("public_history_version"))
                if frame.get("contract", {}).get("fixed_ascension") != self.ascension:
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
                    frame = clean_frame(frame)
                    act, floor, length = stand(frame)
                    if until_act is not None and act is not None and act >= until_act:
                        # The first decision of the act the segment stops at belongs to
                        # the next segment; the save there is where it goes on from.
                        trace["save"] = dict(at="act", save=continue_save(engine))
                        trace.update(status="complete", victory=None, outcome="reached", reached_act=act)
                        break
                    fight, entry = fights.decision()
                    if entry and until_boss is not None and entry["kind"] == "boss" and not entry["second_boss"] \
                            and entry["act"] is not None and entry["act"] >= until_boss:
                        # The first decision of the boss fight the segment stops at: the
                        # engine still stands at the fight's entry and saves the run as it
                        # was before the room; the next segment enters the fight again.
                        trace["save"] = dict(at="fight", save=continue_save(engine), fight=dict(entry, floor=floor))
                        trace.update(status="complete", victory=None, outcome="reached", reached_boss=entry["act"])
                        break
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
                    key = segment_key(frame)
                    if key != current_segment:
                        current_segment, pending_steps, active_macro = key, [], None
                    branching = len(frame["legal"]["candidates"]) > 1
                    choice = choose(frame, fight, sample=sample)
                    # A frame played inside a fight is kept only where every frame is.
                    step = {
                        "frame": _pack(frame) if combat_frames or fight is None else None,
                        "candidate_ref": choice.candidate_ref,
                        "forced": not branching,
                    }
                    if getattr(choice, "explored", False):
                        step["explored"] = True
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
        except (ProtocolError, TimeoutError, OSError, ValueError) as exc:
            trace["status"] = "unresolved" if isinstance(exc, TimeoutError) else "error"
            trace["error"] = str(exc)
        trace["ledger"] = ledger.state_dict()
        reached, lengths = climbed(stands)
        trace["climb"] = {"reached": reached, "lengths": lengths}
        trace["fights"] = fights.tally()
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
         combat_frames=True, segment=False):
    """One job of a round: a run and, as the layout says (`FULL_RUNS`,
    `SUMMARY_RUNS`), its journal and its run file. With `segment` the job is
    (character, spec, None): a segment of a run (`main.play.groups`), `spec` naming
    its seed, `run_id`, `group`, `until_act` or `until_boss` and, to go on from a save,
    the file the save is in (`resume`). Returns the run; one that no file of its own is written for
    keeps its frames packed, for the line it makes in a shard (`packed_line`)."""
    if segment:
        from main.play.groups import read_save

        character, spec, _ = job
        return runner.run(
            engine, character, spec["seed"], sample=sample, combat_frames=combat_frames,
            resume=read_save(spec["resume"]) if spec.get("resume") else None,
            until_act=spec.get("until_act"), until_boss=spec.get("until_boss"), run_id=spec["run_id"],
            group=spec["group"],
            packed=not trajectory, _policy=policy)
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
