import hashlib
import json
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from combat_outcome.online import CombatRecorder

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
    ):
        if max_steps < 1 or timeout <= 0:
            raise ValueError("Rollout limits must be positive")
        temperatures = {"combat": 1.0, "other": 1.0} if temperatures is None else dict(temperatures)
        if set(temperatures) != {"combat", "other"} or not all(t > 0 for t in temperatures.values()):
            raise ValueError("The sampling temperatures must be positive, one for fights and one for the rest")
        if type(ascension) is not int or not 0 <= ascension <= 10:
            raise ValueError("Ascension must be an integer from 0 to 10")
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
        self._digest = None

    @property
    def digest(self):
        if self._digest is None:
            self._digest = policy_digest(self.model)
        return self._digest

    def temperature_for(self, frame, sample=True):
        """The temperature a decision of `frame` is drawn at: a fight's or the rest's.
        The most likely action, which a greedy run takes, is scored at 1."""
        if not sample:
            return 1.0
        return self.temperatures["combat" if frame["public"]["phase"] == "combat" else "other"]

    def run(self, engine, character, seed, *, sample=True, journal=None, combats=None,
            _policy=None):
        """Play one run. `combats` is where the fights of the run are written as
        combat-outcome labels of this policy (combat_outcome.online)."""
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
            "source": "on_policy" if sample else "evaluation",
            "sampling": "full_distribution" if sample else "argmax",
            "temperature": dict(self.temperatures) if sample else None,
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

        # A policy handed in draws its choices where the network runs, at the
        # temperatures it was told; this runner's own is told them here.
        def choose(frame, *, sample):
            if _policy:
                return policy.choose(frame, sample=sample)
            return policy.choose(frame, sample=sample, temperature=self.temperature_for(frame, sample))

        pending_steps, current_segment, active_macro = [], None, None
        waiting_since = None
        recorder = None
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
                    if recorder:
                        recorder.decision(frame)
                    automatic = environment_action(frame)
                    if automatic is None and self.claim_rewards:
                        automatic = free_reward(frame)
                    if automatic is not None:
                        policy.reset()
                        current_segment, pending_steps, active_macro = None, [], None
                        trace["environment_actions"].append(dict(frame=deepcopy(frame), **automatic))
                        trace["automatic_steps"] += 1
                        steps += 1
                        frame = engine.send(execution_command(frame, automatic["candidate_ref"]))
                        continue
                    key = segment_key(frame)
                    if key != current_segment:
                        current_segment, pending_steps, active_macro = key, [], None
                    branching = len(frame["legal"]["candidates"]) > 1
                    choice = choose(frame, sample=sample)
                    if recorder:
                        recorder.step(
                            frame,
                            next(
                                c for c in frame["legal"]["candidates"]
                                if c["candidate_ref"] == choice.candidate_ref
                            ),
                            steps,
                        )
                    step = {
                        "frame": frame,
                        "candidate_ref": choice.candidate_ref,
                        "forced": not branching,
                    }
                    if branching:
                        if active_macro is None:
                            active_macro = {
                                "steps": pending_steps,
                                "old_log_prob": 0.0,
                                "reward": 0.0,
                                "phase": frame["public"]["phase"],
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
        if recorder:
            # The acts whose boss fell; the last act is passed by winning the run.
            passed = set(ledger.bosses) | ({3} if ledger.victory_paid else set())
            trace["combats"] = recorder.close(trace["status"], trace.get("victory"), passed)
        return trace


def write_run(path, trace):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(trace, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def play(engine, runner, job, *, sample=True, policy=None):
    """One job of a round: a run, written beside its journal and the fights it
    recorded. Returns the run."""
    character, seed, path = job
    trace = runner.run(
        engine,
        character,
        seed,
        sample=sample,
        journal=path.with_suffix(".jsonl"),
        combats=path.with_suffix(".combats.jsonl.gz"),
        _policy=policy,
    )
    write_run(path, trace)
    return trace


def collect_round(samplers, runner, seeds, directory, on_run=None, *, sample=True):
    """Freeze the assigned queue before running it; any failure blocks what follows.

    The round is played by the engines of `samplers` (`sampling.Samplers`), each run
    in an engine of its own, while this process only runs the network. Returns where
    the runs were written, in the order of the jobs: every run beside its journal and
    the fights it recorded.
    """
    counts = {c: len(seeds.get(c, [])) for c in CHARACTERS}
    if len(set(counts.values())) != 1 or not all(counts.values()):
        raise ValueError("Allocate equal nonzero run counts to all five characters")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob("*.json")):
        raise FileExistsError(
            "Rollout directory already contains runs; choose a new round directory"
        )
    # Save the full assignment before starting any engine, including queues
    # which later fail. Keep metadata outside the trajectory JSON glob.
    assignment = directory / "metadata" / "seeds.json"
    if assignment.exists():
        raise FileExistsError(
            "Rollout seed assignment already exists; choose a new round directory"
        )
    write_run(assignment, seeds)
    jobs = [
        (character, seed, directory / f"{character}-{i}.json")
        for character in CHARACTERS
        for i, seed in enumerate(seeds[character])
    ]
    played = samplers.collect(runner, jobs, sample=sample, on_run=on_run)
    failures = [str(path) for (_, _, path), run in zip(jobs, played) if run["status"] != "complete"]
    if failures and sample:
        raise ProtocolError(
            f"Round incomplete; a sampled round must finish every run. Inspect: {failures}"
        )
    return [path for _, _, path in jobs]
