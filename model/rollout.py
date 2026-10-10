import hashlib
import json
import time
import zlib
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from combat_outcome.online import CombatRecorder, act_results

from .policy import SessionPolicy
from .control import CONTROL_VERSION, REWARD_RULE, environment_action, free_reward
from .protocol import (
    CHARACTERS,
    SCHEMA,
    ProtocolError,
    clean_frame,
    execution_command,
    fingerprint,
    segment_key,
    validate_frame,
)
from .rewards import MilestoneLedger
from .public_history import require_history_version
from .saves import EntrySaves, FightTracker


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


def final_hp(engine):
    """HP after a fight that ended the run: no frame follows it, the run state still
    holds it. None when the engine cannot say; the run's outcome stands either way."""
    try:
        reply = engine.send({"cmd": "anchor_state"})
    except (ProtocolError, TimeoutError, OSError):
        return None
    if reply.get("type") != "anchor_state":
        return None
    return reply["save"]["players"][0]["current_hp"]


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


# What a round writes of each run. Evaluation keeps everything: the run with every
# frame, its journal and its fights. Sampling keeps what training reads (see
# `model/buffer.py`): the frames of the decisions outside fights, the entry save of
# each fight, the fights, and no run file of its own.
FULL_RUNS = dict(trajectory=True, journal=True, combat_frames=True, saves=False)
OUTSIDE_FRAMES = dict(trajectory=False, journal=False, combat_frames=False, saves=True)


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
        iteration=None,
    ):
        if max_steps < 1 or timeout <= 0:
            raise ValueError("Rollout limits must be positive")
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
        # The sampling iteration the runs belong to: the buffer keeps the last few.
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

    def run(self, engine, character, seed, *, sample=True, journal=None, combats=None,
            saves=None, combat_frames=True, _policy=None):
        """Play one run. `combats` is where the fights of the run are written as
        combat-outcome labels of this policy (combat_outcome.online); `saves` where the
        engine's save at the entry of each fight goes (`model.saves`).

        Without `combat_frames` the macro-actions played inside fights keep their
        counts and log-probability but not their frames: a fight is taught by the
        solver from its entry save, not by the policy's own play, and its frames are
        most of a run."""
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
        ledger = MilestoneLedger()
        policy = _policy or SessionPolicy(
            self.model, self.vocabulary, version=self.version
        )

        # A policy handed in draws its choices where the network runs; either is told
        # the temperature of each decision here, where the fight it belongs to is known.
        def choose(frame, fight, *, sample):
            return policy.choose(frame, sample=sample, temperature=self.temperature_for(frame, sample, fight))

        pending_steps, current_segment, active_macro = [], None, None
        waiting_since = None
        recorder = entries = None
        fights, stands, last_act = FightTracker(), [], None
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
                frame = engine.reset(character, seed, self.ascension)
                require_history_version(frame.get("contract", {}).get("public_history_version"))
                if frame.get("contract", {}).get("fixed_ascension") != self.ascension:
                    raise ProtocolError("Engine started a different ascension than requested")
                trace["run_id"] = frame.get("routing", {}).get(
                    "episode_id", fingerprint([character, seed])
                )
                trace["contract"] = deepcopy(frame.get("contract", {}))
                if combats:
                    recorder = CombatRecorder(
                        combats, seed=seed, character=character, run_id=trace["run_id"],
                        contract=trace["contract"],
                        policy=dict(
                            version=self.version, digest=self.digest,
                            **{k: trace[k] for k in (
                                "vocabulary_hash", "precision", "sampling", "temperature",
                                "numeric_backend")}),
                    )
                if saves:
                    entries = EntrySaves(saves, seed=seed, character=character, run_id=trace["run_id"],
                                         ascension=self.ascension)
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
                    if recorder:
                        recorder.events(frame.get("events", []))
                    if boundary == "terminal":
                        if recorder:
                            recorder.terminal(victory, lambda: final_hp(engine))
                        trace.update(status="complete", victory=victory)
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
                    stands.append((act, floor, length))
                    last_act = act if act is not None else last_act
                    fight, entry = fights.decision()
                    if entry and entries:
                        # The first decision of a fight: the engine still stands at its
                        # entry, and saves the run as it was before the room.
                        entries.take(engine, entry, floor)
                    if recorder:
                        recorder.decision(frame)
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
                    if recorder:
                        recorder.step(
                            frame,
                            next(
                                c for c in frame["legal"]["candidates"]
                                if c["candidate_ref"] == choice.candidate_ref
                            ),
                            steps,
                        )
                    # A frame played inside a fight is kept only where every frame is.
                    step = {
                        "frame": _pack(frame) if combat_frames or fight is None else None,
                        "candidate_ref": choice.candidate_ref,
                        "forced": not branching,
                    }
                    if branching:
                        if active_macro is None:
                            active_macro = {
                                "phase": frame["public"]["phase"],
                                "fight": fight,
                                "old_log_prob": 0.0,
                                "reward": 0.0,
                                "steps": pending_steps,
                            }
                            trace["macros"].append(active_macro)
                        active_macro["old_log_prob"] += float(choice.log_prob)
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
        # The acts whose boss fell; the last act is passed by winning the run.
        passed = set(ledger.bosses) | ({3} if ledger.victory_paid else set())
        trace["acts"] = {str(act): result for act, result in sorted(
            act_results(trace["status"], trace.get("victory"), passed, last_act).items())}
        for macro in trace["macros"]:
            steps = macro.pop("steps")
            if macro["fight"] is not None and not combat_frames:
                macro["branches"] = sum(not step["forced"] for step in steps)
                macro["decisions"] = len(steps)
            else:
                macro["steps"] = [dict(step, frame=_unpack(step["frame"])) for step in steps]
        for action in trace["environment_actions"]:
            action["frame"] = _unpack(action["frame"])
        if recorder:
            trace["combats"] = recorder.close(trace["status"], trace.get("victory"), passed)
        if entries:
            trace["saves"] = entries.close(
                {row["encounter_id"]: (row["combat_index"], row["result"]) for row in recorder.rows}
                if recorder else {})
        return trace


def write_run(path, trace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(trace, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def play(engine, runner, job, *, sample=True, policy=None, trajectory=True, journal=True,
         combat_frames=True, saves=False):
    """One job of a round: a run, with the fights it recorded beside it and, as the
    layout says (`FULL_RUNS`, `OUTSIDE_FRAMES`), its journal, its run file and the entry
    saves of its fights. Returns the run."""
    character, seed, path = job
    trace = runner.run(
        engine,
        character,
        seed,
        sample=sample,
        journal=path.with_suffix(".jsonl") if journal else None,
        combats=path.with_suffix(".combats.jsonl.gz"),
        saves=path.with_suffix(".saves.jsonl.gz") if saves else None,
        combat_frames=combat_frames,
        _policy=policy,
    )
    if trajectory:
        write_run(path, trace)
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


def collect_round(samplers, runner, seeds, directory, on_run=None, *, sample=True):
    """Freeze the assigned queue before running it; any failure blocks what follows.

    The round is played by the engines of `samplers` (`sampling.Samplers`), each run
    in an engine of its own, while this process only runs the network. Returns where
    the runs were written, in the order of the jobs: every run with every frame,
    beside its journal and the fights it recorded. A sampled round that training
    reads is written by `buffer.sample_round` instead.
    """
    jobs = assign(seeds, directory)
    played = samplers.collect(runner, jobs, sample=sample, on_run=on_run)
    failures = [str(path) for (_, _, path), run in zip(jobs, played) if run["status"] != "complete"]
    if failures and sample:
        raise ProtocolError(
            f"Round incomplete; a sampled round must finish every run. Inspect: {failures}"
        )
    return [path for _, _, path in jobs]
