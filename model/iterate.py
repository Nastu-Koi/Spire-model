"""The training iteration: sample, label, value, weigh, update, evaluate.

One iteration of policy improvement, repeated (`python -m model iterate`):

1. **Sample** a round with the current policy at its temperatures, on seeds of the
   iteration's own, into the run buffer (`buffer/round-N`, `model/buffer.py`).
2. **Label** the fights of a share of its runs with the solver, from their entry saves
   (`spire_codex_data.student_battles`), and import the labels as supervised data.
   The solver searches on a time budget: it runs as a command of its own, alone or,
   when told to, beside the steps that follow, never beside the engines.
3. **Value**: continue the critic for a pass over the buffer's rounds and hold its
   held-out report to the gate (`critic/gate.py`).
4. **Weigh**: every decision outside a fight of the buffer's rounds gets its advantage
   from the critic (`critic/advantage.py`), kept beside its round.
5. **Update** the policy by advantage-weighted regression (`model/awr.py`): half of
   each update the buffer's decisions, half the demonstrations with every round's
   labels so far. A critic that fails the gate leaves the policy as it is.
6. **Evaluate** the policy greedily on fixed seeds every few iterations.

Everything is written under one output directory, and every step is skipped when its
result is already there: an interrupted run goes on from the step it was in. What
an interrupted step left is moved under `interrupted/`, never deleted.

    buffer/round-N/            the sampled rounds, each with its advantages.json.gz
    student/round-N/           the solver's fights of round N; student-import/round-N/ as training data
    critic/iteration-NNNN/     the critic after iteration N
    policy/current, policy/iteration-NNNN/   the policy after its update of iteration N
    evaluation/iteration-NNNN/ the evaluation after iteration N, with report.json
    seeds/, logs/              the seeds of each round; the output of the commands run
    state.json                 what every iteration's steps came to
    status.json, history.jsonl, updates.jsonl   what the monitor reads
"""

import contextlib
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from spire_codex_data.sources import split_group

from . import awr
from .advantage import CRITIC, LAMBDA, RETURN, write_returns
from .buffer import buffer_rounds, round_summary, sample_round
from .checkpoint import archive_checkpoint
from .data import RunShards
from .evaluation import evaluate_runs
from .history import TrainingHistory
from .protocol import CHARACTERS, ProtocolError
from .rollout import RolloutRunner, collect_round
from .sampling import Samplers
from .seeds import validate_evaluation_seeds

FORMAT = "iterate-v1"
CODE = Path(__file__).resolve().parents[1]
# The exit status of `import-independent` when the export held no sample to import.
NOTHING_IMPORTED = 2


@dataclass
class Settings:
    """What an invocation asks for. An iteration records the settings it ran under;
    all but the checkpoint and the seed root may differ from one invocation to the
    next, and take effect from the steps still to be made."""

    checkpoint: str  # the policy the first iteration starts from, resumed strictly
    demonstrations: list  # data directories of the supervised half, the largest first
    runs_per_character: int = 200  # a round is five times this
    seed_root: str = "iterate"  # what every round's seeds are derived from
    exclude_seeds: list = field(default_factory=list)  # seed files a round's seeds must avoid
    buffer_iterations: int = 10  # the buffer: the rounds of the last so many iterations
    workers: int = 1  # engines of a round
    processes: int | None = None  # sampling processes; one per engine by default
    max_steps: int = 10000
    engine_root: str | None = None
    quarantine: float = 0.02  # the share of a round's runs that may fail to finish
    labels: bool = True  # label fights with the solver
    label_fraction: float = 0.25  # the share of a round's runs whose every fight is labelled
    label_workers: int = 8  # runs the solver fights at once
    label_failures: float = 0.05  # the share of the selected runs whose labelling may fail
    labels_in_background: bool = False  # let the solver run beside the critic and the update
    solver_config: str | None = None
    advantage: str = CRITIC  # or `return`: the control without a critic
    lam: float = LAMBDA
    critic: str | None = None  # a saved critic to continue from, instead of a new one
    critic_config: str | None = None  # the profile whose model and critic sections make the critic
    critic_gate: bool = True  # hold the critic to the gate before it weighs anything
    gate: list = field(default_factory=list)  # `name=value` overrides of the gate's floors
    beta: str = "auto"  # `auto`, a number, or `inf` for no weighting
    ess_share: float = awr.ESS_SHARE
    beta_min: float = awr.BETA_MIN
    beta_max: float = awr.BETA_MAX
    updates: str = "round"  # `round`, `buffer`, or a number of updates per iteration
    weight_cap: float = awr.WEIGHT_CAP  # what a decision of a small group may weigh, in even shares; 0 for no cap
    window_shards: int = 8
    evaluate_every: int = 5
    eval_seeds: str | None = None  # the seed file of the evaluations; without it none is made
    eval_runs_per_character: int | None = None  # the leading seeds of each character to play
    eval_workers: int | None = None

    def __post_init__(self):
        if self.runs_per_character < 1 or self.buffer_iterations < 1 or self.evaluate_every < 1:
            raise ValueError("Runs per character, the buffer's depth and the evaluation interval are positive")
        if self.advantage not in (CRITIC, RETURN):
            raise ValueError("The advantage comes from the critic or from the return alone")
        if self.updates not in ("round", "buffer") and not (str(self.updates).isdigit() and int(self.updates) > 0):
            raise ValueError("Updates per iteration: `round`, `buffer` or a positive number")
        if self.weight_cap and self.weight_cap < 1:
            raise ValueError("The cap on a sampled decision's weight is in even shares: at least one, or 0 for none")
        if not 0 <= self.label_fraction <= 1:
            raise ValueError("The labelled share of a round's runs is between 0 and 1")
        self.temperature()

    def temperature(self):
        return awr.Temperature.parse(self.beta, share=self.ess_share, low=self.beta_min, high=self.beta_max)


def canonical(seed):
    """A seed as the game reads it: it takes O for 0 and I for 1."""
    return str(seed).upper().replace("O", "0").replace("I", "1").strip()


def seeds_of(path):
    """The seeds a file names: a mapping of characters to seeds, as rounds and
    evaluations are assigned them, or a plain list."""
    data = json.loads(Path(path).read_text())
    lists = data.values() if isinstance(data, dict) else [data]
    return {str(seed) for seeds in lists for seed in seeds}


def round_seeds(root, iteration, count, excluded=()):
    """`count` seeds for each character, derived from the root and the iteration
    number alone: the same round whenever it is asked for, a different one for every
    iteration. A seed among `excluded` (as itself, in the game's reading, or as the
    split group demonstrations are recorded under) is passed over."""
    excluded = set(excluded) | {canonical(seed) for seed in excluded}
    seeds, taken = {}, set()
    for character in CHARACTERS:
        chosen, number = [], 0
        while len(chosen) < count:
            seed = hashlib.sha256(f"{root}:{iteration}:{character}:{number}".encode()).hexdigest()[:16].upper()
            number += 1
            if seed not in taken and seed not in excluded and split_group(seed) not in excluded:
                taken.add(seed)
                chosen.append(seed)
        seeds[character] = chosen
    return seeds


class Command:
    """A command of the pipeline running in a session of its own, its output appended
    to a log. The session lets it be ended with everything it started."""

    def __init__(self, command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        # The command runs this code, wherever the interpreter would find the packages.
        path = os.pathsep.join(filter(None, [str(CODE), os.environ.get("PYTHONPATH")]))
        with log.open("a") as stream:
            stream.write("$ " + " ".join(map(str, command)) + "\n")
            stream.flush()
            self.process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                                            env=dict(os.environ, PYTHONPATH=path), start_new_session=True)

    def wait(self):
        try:
            return self.process.wait()
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait()


def _stop(*_):
    raise KeyboardInterrupt


# How many compiled variants of one function the driver's process may hold.
COMPILED_VARIANTS = 64


def allow_compiled_variants(limit=COMPILED_VARIANTS):
    """Let the compiler keep more variants of a function than its default of eight.

    The global layers of the policy and of the critic are one compiled function. A
    process that only trains, or only samples, runs it in one or two ways; this one
    samples, trains the critic, scores with it and trains the policy in turn, each
    with its own gradient mode, precision and module, and past the limit the compiler
    leaves whatever comes next uncompiled for as long as the process lives."""
    import torch._dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if isinstance(getattr(torch._dynamo.config, name, None), int):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), limit))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


class Driver:
    """The iterations of one output directory. `engine` makes an engine for a sampling
    process; `launch(command, log)` starts a command and returns what `wait()`s for its
    exit status and can `stop()` it. Both are the real ones unless a test hands in
    others."""

    def __init__(self, output, settings, device="cpu", *, engine=None, launch=Command, runtime=None, say=None):
        self.output, self.settings, self.device = Path(output), settings, device
        self.launch, self.runtime = launch, runtime or {}
        self.say = say or (lambda value: print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True))
        if engine is None:
            from .engine import CliEngine

            engine = lambda: CliEngine(root=settings.engine_root)  # noqa: E731
        self.engine = engine
        self.buffer, self.policy = self.output / "buffer", self.output / "policy"
        self.learner = self.history = self.background = None
        self.at = {}  # the iteration and the step under way, for the status
        self.output.mkdir(parents=True, exist_ok=True)
        identity = dict(format=FORMAT, checkpoint=str(Path(settings.checkpoint).resolve()), seed_root=settings.seed_root)
        path = self.output / "iterate.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise ValueError(f"{self.output} holds the iterations of another checkpoint or seed root; "
                                 "use a new output directory")
        else:
            write_json(path, identity)
        path = self.output / "state.json"
        self.state = json.loads(path.read_text()) if path.exists() else dict(format=FORMAT, iterations={})
        if self.state.get("format") != FORMAT:
            raise ValueError("Unsupported iteration state")

    # ---- State ----

    def entry(self, iteration):
        return self.state["iterations"].setdefault(str(iteration), {})

    def record(self, iteration, step, value):
        self.entry(iteration)[step] = value
        write_json(self.output / "state.json", self.state)
        if step != "settings":
            self.say(dict(iteration=iteration, step=step, **(value if isinstance(value, dict) else dict(value=value))))
        return value

    def begin(self, iteration, name):
        """Say which step of which iteration is under way."""
        self.at = dict(round=iteration, step=name)
        self.history.status("running", **self.at)

    def step(self, iteration, name, function):
        """A step's recorded result, made now if it is not there yet."""
        done = self.entry(iteration).get(name)
        if done is not None:
            return done
        self.begin(iteration, name)
        return self.record(iteration, name, function(iteration))

    def set_aside(self, path):
        """Move what an interrupted step left out of the way of its second attempt."""
        if not path.exists():
            return
        aside = self.output / "interrupted"
        aside.mkdir(exist_ok=True)
        number = 0
        while (aside / f"{path.parent.name}-{path.name}.{number}").exists():
            number += 1
        shutil.move(str(path), str(aside / f"{path.parent.name}-{path.name}.{number}"))

    def release(self):
        """Give back what a step held on the GPU and in the heap before the next one."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Steps ----

    def load(self):
        """The policy with its optimizer: the last update's, or the starting checkpoint's."""
        from .cli import _learner

        current = self.policy / "current"
        learner, _ = _learner(current if current.exists() else self.settings.checkpoint, self.device)
        return learner

    def excluded(self):
        """Seeds no round may be played on: the evaluations', the listed files', and
        what the policy was trained on."""
        settings = self.settings
        seeds = set(map(str, getattr(self.learner, "checkpoint_metadata", {}).get("training_seeds", [])))
        for path in [*settings.exclude_seeds, *([settings.eval_seeds] if settings.eval_seeds else [])]:
            seeds |= seeds_of(path)
        return seeds

    def rounds(self):
        return buffer_rounds(self.buffer, self.settings.buffer_iterations)

    def sampler(self, iteration=None):
        learner, settings = self.learner, self.settings
        return RolloutRunner(
            learner.model, learner.vocabulary, precision=learner.config.precision, version=learner.policy_version,
            max_steps=settings.max_steps, ascension=learner.config.ascension,
            temperatures=learner.config.temperatures, iteration=iteration)

    def samplers(self, engines):
        return Samplers(self.engine, self.learner.model, self.learner.vocabulary, engines=engines,
                        processes=self.settings.processes)

    def sample(self, iteration):
        settings, directory = self.settings, self.buffer / f"round-{iteration}"
        summary = round_summary(directory)
        if summary is None:
            self.set_aside(directory)
            excluded = self.excluded()
            seeds = round_seeds(settings.seed_root, iteration, settings.runs_per_character, excluded)
            if excluded & {seed for listed in seeds.values() for seed in listed}:
                raise ProtocolError("A round's seeds overlap the seeds kept apart from sampling")
            write_json(self.output / "seeds" / f"iteration-{iteration:04d}.json", seeds)
            with self.samplers(settings.workers) as samplers:
                try:
                    summary = sample_round(samplers, self.sampler(iteration), seeds, directory)
                except ProtocolError:
                    # Runs that did not finish fail the round after it is written; a
                    # few of them leave it usable.
                    summary = round_summary(directory)
                    if summary is None:
                        raise
            self.release()
        if not summary["accepted_runs"] or summary["quarantined_runs"] > settings.quarantine * summary["runs"]:
            raise ProtocolError(
                f"Round {directory} finished {summary['accepted_runs']} of {summary['runs']} runs; "
                f"inspect its quarantine: {summary['quarantine']}")
        report = summary["report"]
        return dict(
            round=directory.name, runs=summary["accepted_runs"], quarantined=summary["quarantined_runs"],
            wins=summary["wins"], win_rate=report["equal_character_win_rate"],
            mean_floor=report["equal_character_mean_floor"],
            act_pass_rates=report["equal_character_act_pass_rates"], decisions=summary["macros"],
            fights=summary["fights"], policy_version=summary["policy_version"],
            policy_digest=summary["policy_digest"], temperature=summary["temperature"])

    def label_command(self, iteration):
        settings = self.settings
        command = [sys.executable, "-m", "spire_codex_data.student_battles", "run",
                   "--round", self.buffer / f"round-{iteration}", "--output", self.output / "student" / f"round-{iteration}",
                   "--fraction", settings.label_fraction, "--workers", settings.label_workers]
        return command + (["--solver-config", settings.solver_config] if settings.solver_config else [])

    def label(self, iteration, started=None):
        """The solver's labels of a round as training data: fight, export, import.
        `started` is the fighting begun earlier in the background, if it was."""
        settings = self.settings
        work, target = self.output / "student" / f"round-{iteration}", self.output / "student-import" / f"round-{iteration}"
        log = self.output / "logs" / f"labels-round-{iteration}.log"
        status = (started or self.launch(self.label_command(iteration), log)).wait()
        if status:
            # A finished run's report stands: a second attempt fights only what failed.
            status = self.launch(self.label_command(iteration), log).wait()
        summary = json.loads((work / "summary.json").read_text()) if (work / "summary.json").is_file() else {}
        failures = len(summary.get("run_failures") or {})
        selected = summary.get("selection", {}).get("runs_selected", 0)
        # The command fails when runs did; anything else that fails it left no such count.
        if not summary or (status and not 0 < failures <= settings.label_failures * selected):
            raise ProtocolError(f"The solver's labelling of round {iteration} failed "
                                f"({failures} of {selected} runs); see {log}")
        if self.launch([sys.executable, "-m", "spire_codex_data.student_battles", "export", "--output", work], log).wait():
            raise ProtocolError(f"The export of round {iteration}'s labels failed; see {log}")
        self.set_aside(target)
        status = self.launch([sys.executable, "-m", "model", "import-independent", work / "independent-training.jsonl.gz",
                              "--output", target, "--shard-samples", 250], log).wait()
        if status not in (0, NOTHING_IMPORTED):
            raise ProtocolError(f"The import of round {iteration}'s labels failed; see {log}")
        imported = json.loads((target / "summary.json").read_text())
        counts = summary["counts"]
        return dict(
            fraction=settings.label_fraction, selected_runs=selected, failed_runs=failures,
            fights=counts.get("fights", 0), verified=counts.get("verified", 0), wins=counts.get("outcome_win", 0),
            groups=imported["accepted_runs"], macros=imported.get("macros", 0),
            data=str(target.relative_to(self.output)) if imported["accepted_runs"] else None)

    def finish_labels(self):
        """Have the labels of every sampled round before engines run again: the
        solver's searches are timed, and a round's engines would take their time."""
        if self.background:
            iteration, started = self.background
            self.background = None
            self.begin(iteration, "labels")
            self.record(iteration, "labels", self.label(iteration, started))
        # A round whose labelling was cut off; one sampled without labels stays without.
        for iteration in sorted(map(int, self.state["iterations"])):
            entry = self.entry(iteration)
            if "sample" in entry and entry.get("settings", {}).get("labels"):
                self.step(iteration, "labels", self.label)

    def student_data(self):
        """The imports of every round's labels so far: they accumulate, the buffer's
        rounds do not."""
        return [str(self.output / entry["labels"]["data"]) for _, entry in sorted(
            self.state["iterations"].items(), key=lambda item: int(item[0])) if (entry.get("labels") or {}).get("data")]

    def latest_critic(self, before):
        found = [entry["critic"]["directory"] for number, entry in sorted(
            self.state["iterations"].items(), key=lambda item: int(item[0])) if int(number) < before and entry.get("critic")]
        return str(self.output / found[-1]) if found else self.settings.critic

    def critic(self, iteration):
        from critic import train

        settings, target = self.settings, self.output / "critic" / f"iteration-{iteration:04d}"
        rounds = self.rounds()
        if not (target / "manifest.json").is_file():
            self.set_aside(target)
            previous = self.latest_critic(iteration)
            arguments = [*map(str, rounds), "--device", self.device, "--save", str(target)]
            arguments += ["--config", settings.critic_config] if settings.critic_config else []
            arguments += ["--pretrained", previous] if previous else ["--checkpoint", settings.checkpoint]
            log = self.output / "logs" / f"critic-iteration-{iteration:04d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            # The critic's training seeds the generators for itself; the policy's
            # sampling goes on from where its own stood.
            generators = torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                with log.open("a") as stream, contextlib.redirect_stdout(stream):
                    train.main(arguments)
            finally:
                torch.set_rng_state(generators[0])
                if generators[1] is not None:
                    torch.cuda.set_rng_state_all(generators[1])
            self.release()
        manifest = json.loads((target / "manifest.json").read_text())
        report, split = manifest["report"], manifest["data"]["split"]
        baseline = (report.get("baselines") or {}).get("logistic_act_floor") or {}
        return dict(
            directory=str(target.relative_to(self.output)), version=manifest["version"], pretrained=manifest["pretrained"],
            rounds=[path.name for path in rounds], runs=manifest["data"]["runs"], decisions=manifest["data"]["decisions"],
            heldout_seeds=split["test_seeds"], train_loss=report["history"][-1]["train_loss"],
            **{f"{head}_{key}": report["critic"][head][key] for head in ("win", "act")
               for key in ("explained_variance", "brier", "auc", "rate", "predicted")},
            **{f"baseline_{head}_explained_variance": (baseline.get(head) or {}).get("explained_variance")
               for head in ("win", "act")})

    def gate(self, critic):
        """The gate's verdict on a saved critic, under the settings of this invocation."""
        from critic.gate import Gate, check

        manifest = json.loads((self.output / critic["directory"] / "manifest.json").read_text())
        result = check(dict(manifest["report"], data=manifest["data"]), Gate.of(self.settings.gate))
        return dict(enforced=self.settings.critic_gate, passed=result["passed"], failed=result["failed"],
                    checks=result["checks"])

    def weigh(self, iteration):
        """The advantages of every round of the buffer, by the iteration's critic or by
        the return alone. A round that holds them already is left alone."""
        settings, rounds = self.settings, self.rounds()
        if settings.advantage == RETURN:
            headers = write_returns(rounds)
        else:
            from critic.advantage import score_rounds

            headers = score_rounds(rounds, self.output / self.entry(iteration)["critic"]["directory"], self.device,
                                   lam=settings.lam)
            self.release()
        return [dict(round=path.name, **{key: header[key] for key in ("kind", "runs", "decisions", "advantage")})
                for path, header in zip(rounds, headers)]

    def update(self, iteration):
        from .cli import _save

        settings, learner, rounds = self.settings, self.learner, self.rounds()
        current, archived = self.policy / "current", self.policy / f"iteration-{iteration:04d}"
        applied = getattr(learner, "checkpoint_metadata", {})
        if applied.get("iteration") == iteration and applied.get("update"):
            # The update was saved before the run was cut off: the policy in hand is its result.
            if not archived.exists():
                archive_checkpoint(current, archived)
            return applied["update"]
        weighed = self.weigh(iteration)
        supervised = [*settings.demonstrations, *self.student_data()]
        newest = [path for path in rounds if round_summary(path)["iteration"] == iteration]
        updates = (awr.pass_updates(newest or rounds, learner.config) if settings.updates == "round"
                   else None if settings.updates == "buffer" else int(settings.updates))
        started = time.monotonic()
        metrics = awr.update(
            learner, rounds, supervised, temperature=settings.temperature(), updates=updates,
            window_shards=settings.window_shards, weight_cap=settings.weight_cap,
            on_update=lambda update: self.history.append_update(
                update, round=iteration, stage="awr", policy_version=learner.policy_version))
        self.release()
        done = metrics["awr"]
        result = dict(
            policy=str(archived.relative_to(self.output)), policy_version=learner.policy_version,
            updates=learner.updates, pass_updates=done["batches"], rounds=[path.name for path in rounds],
            supervised=supervised, advantages=weighed, loss=metrics["loss"], accuracy=metrics["accuracy"],
            awr_loss=done["awr_loss"], supervised_loss=done["supervised_loss"], beta=done["beta"],
            beta_min=done["beta_min"], beta_max=done["beta_max"], effective_share=done["effective_share"],
            effective_share_min=done["effective_share_min"], temperature=done["temperature"],
            capped_groups=done["capped_groups"],
            sampled=done["sampled"], labelled=done["supervised"], sources=metrics["sources"],
            grad_norm=metrics.get("grad_norm"), seconds=round(time.monotonic() - started, 1))
        critic = self.entry(iteration).get("critic")
        seeds = RunShards(rounds).seeds
        for directory in self.student_data():
            seeds |= RunShards(directory).seeds
        _save(current, learner, dict(
            runtime=self.runtime, iteration=iteration, update=result, advantage=settings.advantage,
            critic=dict(version=critic["version"], directory=critic["directory"]) if critic else None,
            training_seeds=sorted(seeds)))
        self.set_aside(archived)
        archive_checkpoint(current, archived)
        return result

    def evaluate(self, iteration):
        settings, directory = self.settings, self.output / "evaluation" / f"iteration-{iteration:04d}"
        report = directory / "report.json"
        if not report.is_file():
            self.set_aside(directory)
            listed = json.loads(Path(settings.eval_seeds).read_text())
            seeds = {character: listed[character][:settings.eval_runs_per_character] for character in CHARACTERS}
            validate_evaluation_seeds(seeds, getattr(self.learner, "checkpoint_metadata", {}))
            with self.samplers(settings.eval_workers or settings.workers) as samplers:
                paths = collect_round(samplers, self.sampler(), seeds, directory, sample=False)
            self.release()
            # One run in memory at a time.
            write_json(report, dict(policy_version=self.learner.policy_version, iteration=iteration,
                                    **evaluate_runs(json.loads(path.read_text()) for path in paths)))
        result = json.loads(report.read_text())
        return dict(directory=str(directory.relative_to(self.output)), policy_version=result["policy_version"],
                    runs=result["pooled"]["attempts"], wins=result["pooled"]["wins"],
                    win_rate=result["equal_character_win_rate"], mean_floor=result["equal_character_mean_floor"],
                    act_pass_rates=result["equal_character_act_pass_rates"], comparable=result["comparable"])

    # ---- The loop ----

    def evaluation_due(self, iteration):
        """After every few iterations, of a policy that has not been evaluated yet."""
        if not self.settings.eval_seeds or (iteration + 1) % self.settings.evaluate_every:
            return False
        versions = {entry["evaluation"]["policy_version"] for entry in self.state["iterations"].values()
                    if entry.get("evaluation")}
        return self.learner.policy_version not in versions

    def iterate(self, iteration):
        settings, started = self.settings, time.monotonic()
        self.record(iteration, "settings", asdict(settings))
        if "sample" not in self.entry(iteration):
            self.finish_labels()
        sampled = self.step(iteration, "sample", self.sample)
        if settings.labels and "labels" not in self.entry(iteration):
            if settings.labels_in_background:
                log = self.output / "logs" / f"labels-round-{iteration}.log"
                self.background = iteration, self.launch(self.label_command(iteration), log)
            else:
                self.step(iteration, "labels", self.label)
        critic = gate = None
        if settings.advantage == CRITIC:
            critic = self.step(iteration, "critic", self.critic)
        if "update" not in self.entry(iteration):
            gate = self.gate(critic) if critic else None
            if gate and settings.critic_gate and not gate["passed"]:
                self.record(iteration, "update", dict(skipped="critic_gate", failed=gate["failed"], gate=gate))
            else:
                self.begin(iteration, "update")
                self.record(iteration, "update", dict(self.update(iteration), gate=gate))
        update = self.entry(iteration)["update"]
        if "evaluation" not in self.entry(iteration) and self.evaluation_due(iteration):
            self.finish_labels()
            self.step(iteration, "evaluation", self.evaluate)
        evaluation, labels = self.entry(iteration).get("evaluation"), self.entry(iteration).get("labels")
        gate = update.get("gate")
        metrics = dict(
            updates=self.learner.updates, sampled_runs=sampled["runs"], sampled_win_rate=sampled["win_rate"],
            sampled_mean_floor=sampled["mean_floor"],
            **{f"sampled_act{act}_pass_rate": rate for act, rate in enumerate(sampled["act_pass_rates"] or [], 1)},
            buffer_rounds=len(self.rounds()), policy_updated="skipped" not in update)
        if labels:
            metrics.update(label_fights=labels["fights"], label_macros=labels["macros"])
        if critic:
            metrics.update(
                critic_version=critic["version"], critic_win_ev=critic["win_explained_variance"],
                critic_act_ev=critic["act_explained_variance"],
                critic_baseline_win_ev=critic["baseline_win_explained_variance"],
                critic_heldout_seeds=critic["heldout_seeds"], critic_gate_passed=gate["passed"] if gate else None)
        if "skipped" not in update:
            metrics.update(
                awr_updates=update["pass_updates"], loss=update["loss"], accuracy=update["accuracy"],
                awr_loss=update["awr_loss"], supervised_loss=update["supervised_loss"], beta=update["beta"],
                effective_share=update["effective_share"], effective_share_min=update["effective_share_min"],
                student_macros=update["labelled"]["student_macros"])
        if evaluation:
            metrics.update(eval_runs=evaluation["runs"], eval_win_rate=evaluation["win_rate"],
                           eval_mean_floor=evaluation["mean_floor"])
        self.history.append(iteration, metrics, policy_version=self.learner.policy_version,
                            duration_seconds=time.monotonic() - started)
        self.record(iteration, "complete", True)

    def run(self, iterations):
        """Bring the output to `iterations` finished iterations. Returns the state."""
        if iterations < 1:
            raise ValueError("At least one iteration")
        self.history = TrainingHistory(self.output, "iterate", asdict(self.settings))
        # A request to stop ends the step under way as an interruption does: the
        # engines, the loaders and the solver's command go with it.
        handled = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGTERM, _stop) if handled else None
        try:
            allow_compiled_variants()
            self.learner = self.load()
            self.history.status("running", updates=self.learner.updates)
            for iteration in range(iterations):
                if not self.entry(iteration).get("complete"):
                    self.iterate(iteration)
            self.finish_labels()
            self.history.status("completed", round=iterations - 1)
        except KeyboardInterrupt:
            self.history.status("interrupted", **self.at)
            raise
        except BaseException as error:
            self.history.status("failed", error=str(error), **self.at)
            raise
        finally:
            if self.background:
                self.background[1].stop()
            self.history.close()
            if handled:
                signal.signal(signal.SIGTERM, previous)
        return self.state
