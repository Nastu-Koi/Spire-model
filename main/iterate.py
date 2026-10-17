"""The training iteration of act groups: sample, value, update, evaluate, clean up.

One iteration, repeated (`python -m main iterate`):

1. **Sample** a round of act groups with the current policy at its temperatures, on
   seeds of the iteration's own (`main/play/groups.py`): segments of runs cut at the
   boundary acts, several from each start, one wave per stretch, written under
   `groups/round-N`.
2. **Value**: every wave's decisions get their advantages from their own groups
   (`main/learn/groups.py`), returns passed back up the chain of waves.
3. **Update**: one pass over the round, every logical batch drawing from each wave
   by its share (`batches.wave_shares`), with the clipped ratio and a KL to the
   starting checkpoint (`grpo.group_update`).
4. **Evaluate** the policy greedily on fixed seeds every few iterations; the
   evaluation keeps each run's outcome, not its frames.
5. **Clean up** what the next iterations do not read: the round's frames, policy
   archives no evaluation was made of.

Everything is written under one output directory, and every step is skipped when its
result is already there: an interrupted run goes on from the step it was in. What
an interrupted step left is moved under `interrupted/`, never deleted.

    groups/round-N/            the rounds: a directory per wave, each with its advantages.json.gz
    policy/current, policy/iteration-NNNN/   the policy after its update of iteration N
    evaluation/iteration-NNNN/ the evaluation after iteration N, with report.json
    seeds/, logs/              the seeds of each round; the output of the commands run
    state.json                 what every iteration's steps came to
    status.json, history.jsonl, updates.jsonl   the status, one line per iteration, one per update
"""

import gc
import hashlib
import json
import shutil
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from main.env.seeds import split_group

from main.learn import batches, grpo
from main.learn.groups import KIND as GROUPS, advantages as group_advantages
from main.play import groups as act_groups
from main.play.groups import group_rounds, groups_summary, sample_groups
from main.network.checkpoint import archive_checkpoint
from main.network.config import LearningRates
from main.data.shards import RunShards
from main.play.evaluation import evaluate_runs
from main.learn.history import TrainingHistory
from main.env.protocol import CHARACTERS, ProtocolError
from main.play.rollout import SUMMARY_RUNS, RolloutRunner, collect_round
from main.play.sampling import Samplers
from main.env.seeds import validate_evaluation_seeds

FORMAT = "iterate-v1"


@dataclass
class Settings:
    """What an invocation asks for. An iteration records the settings it ran under;
    all but the checkpoint and the seed root may differ from one invocation to the
    next, and take effect from the steps still to be made."""

    checkpoint: str  # the policy the first iteration starts from, resumed strictly
    config: str | None = None  # a profile whose sampling temperatures replace the checkpoint's
    runs_per_character: int = 200  # seeds of each character a round starts from
    seed_root: str = "iterate"  # what every round's seeds are derived from
    exclude_seeds: list = field(default_factory=list)  # seed files a round's seeds must avoid
    workers: int = 1  # engines of a round
    processes: int | None = None  # sampling processes; one per engine by default
    max_steps: int = 10000
    engine_root: str | None = None
    quarantine: float = 0.02  # the share of a round's segments that may fail to finish
    boundaries: list = field(default_factory=lambda: list(act_groups.BOUNDARIES))  # the acts act groups cut at
    bosses: list = field(default_factory=list)  # the acts whose boss fight act groups also cut at
    group_n: int = act_groups.N  # segments from each seed in the first wave
    group_m: list = field(default_factory=lambda: [act_groups.M])  # segments from each save, one or one per wave
    updates: str = "round"  # `round` (one pass over the round) or a number of updates per iteration
    weight_cap: float = batches.WEIGHT_CAP  # what a decision of a small group may weigh, in even shares; 0 for no cap
    phase_weights: str = "group"  # how a wave's decisions share its weight: by (character, phase) group, or even
    window_shards: int = 8
    evaluate_every: int = 5
    eval_seeds: str | None = None  # the seed file of the evaluations; without it none is made
    eval_runs_per_character: int | None = None  # the leading seeds of each character to play
    eval_workers: int | None = None
    # The waves' relative shares of a batch, one per wave in order; by their decisions when None.
    wave_shares: list | None = None
    # Stop once an update's mean policy entropy on its decisions falls below this.
    min_entropy: float | None = None
    # Every this many iterations the KL's reference becomes the policy of that moment
    # (the archive of the iteration before); 0 keeps the starting checkpoint throughout.
    reference_every: int = 0
    # A JSON file of the rates this run's updates train at (`LearningRates`); the checkpoint's without it.
    learning_rates: str | None = None
    clip: float = grpo.CLIP
    kl: float = grpo.KL
    # Sample the next round (and evaluate) on a sampling side of its own while the
    # policy is updated: round N+1 is played by the policy before update N.
    overlap: bool = False

    def __post_init__(self):
        if self.runs_per_character < 1 or self.evaluate_every < 1:
            raise ValueError("Runs per character and the evaluation interval are positive")
        if self.learning_rates is not None:
            LearningRates.load(self.learning_rates)
        if not 0 < self.clip < 1 or self.kl < 0:
            raise ValueError("The clip ε is in (0, 1) and the KL coefficient not negative")
        if self.updates != "round" and not (str(self.updates).isdigit() and int(self.updates) > 0):
            raise ValueError("Updates per iteration: `round` or a positive number")
        if self.weight_cap and self.weight_cap < 1:
            raise ValueError("The cap on a sampled decision's weight is in even shares: at least one, or 0 for none")
        if self.phase_weights not in batches.PHASE_WEIGHTS:
            raise ValueError("Phase weights are `group` or `even`")
        if type(self.reference_every) is not int or self.reference_every < 0:
            raise ValueError("The reference is renewed every so many iterations, or never with 0")
        names = act_groups.waves(self.boundaries, self.bosses)
        self.group_m = act_groups.per_wave(self.group_m, len(names) - 1)
        if self.group_n < 1:
            raise ValueError("Act groups: at least one segment from each seed")
        if self.wave_shares is not None:
            self.wave_shares = [float(share) for share in self.wave_shares]
            batches.wave_shares(dict.fromkeys(names, 1), self.wave_shares)

    def group_shares(self, macros):
        """The shares of an update's batch, `macros` being each wave's priced
        decisions: the waves by their decisions, or at `wave_shares` (`batches.wave_shares`)."""
        return batches.wave_shares(macros, self.wave_shares)


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


def _stop(*_):
    raise KeyboardInterrupt


class Halt(Exception):
    """The run stops on purpose after a complete iteration: a guard was crossed."""


# How many compiled variants of one function the driver's process may hold.
COMPILED_VARIANTS = 64


def allow_compiled_variants(limit=COMPILED_VARIANTS):
    """Let the compiler keep more variants of a function than its default of eight.

    The policy's global layers are one compiled function. A process that only trains,
    or only samples, runs it in one or two ways; this one samples, trains the policy
    and runs the frozen reference of a clipped update in turn, each with its own
    gradient mode, precision and module, and past the limit the compiler leaves
    whatever comes next uncompiled for as long as the process lives."""
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


def set_aside(output, path):
    """Move what an interrupted step left out of the way of its second attempt."""
    path = Path(path)
    if not path.exists():
        return
    aside = Path(output) / "interrupted"
    aside.mkdir(exist_ok=True)
    number = 0
    while (aside / f"{path.parent.name}-{path.name}.{number}").exists():
        number += 1
    shutil.move(str(path), str(aside / f"{path.parent.name}-{path.name}.{number}"))


def samplers_for(pool, engines, settings, engine_factory, model, vocabulary):
    """The sampling processes of a round of `engines` engines: those of the run
    (`pool`), whose engines rest after the round, or for another number of engines
    processes of the round's own."""
    if engines == settings.workers and pool is not None:
        return _Resting(pool)
    return Samplers(engine_factory, model, vocabulary, engines=engines, processes=settings.processes)


def play_round(samplers, runner, seeds, directory, settings, output):
    """A round of act groups on `seeds` into `directory` (`main.play.groups`), by the
    sampling processes `samplers` (a context manager). A round cut off on these seeds
    goes on from the wave it was in; one on other seeds is set aside. Returns the summary."""
    directory = Path(directory)
    listed = directory / "metadata" / "seeds.json"
    if directory.exists() and not (listed.is_file() and json.loads(listed.read_text()) == seeds):
        set_aside(output, directory)
    with samplers as pool:
        return sample_groups(pool, runner, seeds, directory, n=settings.group_n, m=settings.group_m,
                             boundaries=settings.boundaries, bosses=settings.bosses, resume=True)


def round_result(summary, directory, settings):
    """What the `sample` step of an iteration records of its round's summary."""
    directory = Path(directory)
    waves = summary["waves"]
    segments = sum(summary[wave]["segments"] for wave in waves)
    written = sum(summary[wave]["written"] for wave in waves)
    failed = sum(sum(summary[wave]["failed"].values()) for wave in waves)
    if not summary[waves[-1]]["written"] or failed > settings.quarantine * segments:
        raise ProtocolError(f"Round {directory} finished {written} of {segments} segments; inspect the "
                            "quarantine of its waves")
    last = summary[waves[-1]]
    return dict(
        round=directory.name, groups=summary["groups"], waves=waves, segments=segments, written=written,
        failed=failed, saves=summary["saves"], outcomes={wave: summary[wave]["outcomes"] for wave in waves},
        reach_rates={wave: round(summary["saves"][wave] / summary[wave]["written"], 4)
                     for wave in waves[:-1] if summary[wave]["written"]},
        win_rate=round(last["outcomes"].get("win", 0) / last["written"], 4),
        decisions={wave: sum(summary[wave]["phases"].values()) for wave in waves},
        collector={wave: summary[wave].get("collector") for wave in waves},
        policy_version=summary["policy_version"],
        policy_digest=summary["policy_digest"], temperature=summary["temperature"], seconds=summary["seconds"])


def evaluation_seeds(settings):
    """The seeds of an evaluation: each character's leading `eval_runs_per_character`."""
    listed = json.loads(Path(settings.eval_seeds).read_text())
    return {character: listed[character][:settings.eval_runs_per_character] for character in CHARACTERS}


def play_evaluation(samplers, runner, seeds, directory, iteration, metadata, output):
    """The greedy evaluation of the runner's policy on `seeds` into `directory`, by the
    sampling processes `samplers` (a context manager), unless its report is there
    already; `metadata` is the policy checkpoint's progress, whose training seeds the
    evaluation's must avoid. Returns what the `evaluation` step records."""
    directory, report = Path(directory), Path(directory) / "report.json"
    if not report.is_file():
        set_aside(output, directory)
        validate_evaluation_seeds(seeds, metadata)
        with samplers as pool:
            paths = collect_round(pool, runner, seeds, directory, sample=False, layout=SUMMARY_RUNS)
        # One run in memory at a time.
        write_json(report, dict(policy_version=runner.version, iteration=iteration,
                                **evaluate_runs(json.loads(path.read_text()) for path in paths)))
    result = json.loads(report.read_text())
    return dict(directory=str(directory.relative_to(output)), policy_version=result["policy_version"],
                runs=result["pooled"]["attempts"], wins=result["pooled"]["wins"],
                win_rate=result["equal_character_win_rate"], mean_floor=result["equal_character_mean_floor"],
                act_pass_rates=result["equal_character_act_pass_rates"], comparable=result["comparable"])


class _Resting:
    """The run's sampling processes for one round: their engines rest after it."""

    def __init__(self, pool):
        self.pool = pool

    def __enter__(self):
        return self.pool

    def __exit__(self, kind, *_):
        # A round that failed has ended the processes already (`Samplers.collect`).
        if kind is None:
            self.pool.rest()


class Driver:
    """The iterations of one output directory. `engine` makes an engine for a sampling
    process: the real one unless a test hands in another."""

    def __init__(self, output, settings, device="cpu", *, engine=None, runtime=None, say=None):
        self.output, self.settings, self.device = Path(output), settings, device
        self.runtime = runtime or {}
        self.say = say or (lambda value: print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True))
        if engine is None:
            from main.env.engine import CliEngine

            engine = lambda: CliEngine(root=settings.engine_root)  # noqa: E731
        self.engine = engine
        self.policy, self.groups = self.output / "policy", self.output / "groups"
        self.learner = self.history = None
        # The sampling processes of every round, forked before anything is trained; or
        # with `overlap` the sampling side that holds them (`main/overlap.py`).
        self.pool = self.side = None
        self.iterations = 0  # how many the run brings the output to
        self.dispatched = None  # the job the sampling side is on during an update
        self.resumed = False  # the policy in hand is this run's own (`load`)
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
        set_aside(self.output, path)

    def release(self):
        """Give back what a step held on the GPU and in the heap before the next one."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Steps ----

    def load(self):
        """The policy with its optimizer: the last update's, or the starting checkpoint's.
        `resumed` says which: only a policy of this run's own may be the update of an
        iteration that was cut off after it was saved (`update`)."""
        from main.cli import _learner, _profile

        current = self.policy / "current"
        self.resumed = current.exists()
        rates = self.settings.learning_rates
        learner, _ = _learner(current if self.resumed else self.settings.checkpoint, self.device,
                              training=_profile(self.settings.config),
                              rates=LearningRates.load(rates) if rates else None)
        return learner

    def excluded(self):
        """Seeds no round may be played on: the evaluations', the listed files', and
        what the policy was trained on."""
        settings = self.settings
        seeds = set(map(str, getattr(self.learner, "checkpoint_metadata", {}).get("training_seeds", [])))
        for path in [*settings.exclude_seeds, *([settings.eval_seeds] if settings.eval_seeds else [])]:
            seeds |= seeds_of(path)
        return seeds

    def sampler(self, iteration=None):
        learner, settings = self.learner, self.settings
        return RolloutRunner(
            learner.model, learner.vocabulary, precision=learner.config.precision, version=learner.policy_version,
            max_steps=settings.max_steps, ascension=learner.config.ascension,
            temperatures=learner.config.temperatures, iteration=iteration)

    def samplers(self, engines):
        """The sampling processes of a round of `engines` engines (`samplers_for`)."""
        return samplers_for(self.pool, engines, self.settings, self.engine, self.learner.model,
                            self.learner.vocabulary)

    def round_seeds(self, iteration):
        """The seeds of an iteration's round, written under `seeds/`."""
        settings = self.settings
        excluded = self.excluded()
        seeds = round_seeds(settings.seed_root, iteration, settings.runs_per_character, excluded)
        if excluded & {seed for listed in seeds.values() for seed in listed}:
            raise ProtocolError("A round's seeds overlap the seeds kept apart from sampling")
        write_json(self.output / "seeds" / f"iteration-{iteration:04d}.json", seeds)
        return seeds

    def sample(self, iteration):
        """A round of act groups (`main.play.groups`) on seeds of the iteration's own:
        `runs_per_character` seeds of each character, `group_n` segments from each,
        `group_m` from each save."""
        settings, directory = self.settings, self.groups / f"round-{iteration}"
        summary = groups_summary(directory)
        if summary is None:
            seeds = self.round_seeds(iteration)
            if self.side is not None:
                # The sampling side plays it now, with the policy the round is due.
                return self.side.run(dict(policy=str(self.sampling_policy(iteration)),
                                          round=self.round_job(iteration, seeds)))["round"]
            summary = play_round(self.samplers(settings.workers), self.sampler(iteration), seeds, directory,
                                 settings, self.output)
            self.release()
        return round_result(summary, directory, settings)

    # ---- The sampling side of an overlapped run ----

    def round_job(self, iteration, seeds):
        return dict(iteration=iteration, seeds=seeds, directory=str(self.groups / f"round-{iteration}"))

    def evaluation_job(self, iteration):
        return dict(iteration=iteration, seeds=evaluation_seeds(self.settings),
                    directory=str(self.output / "evaluation" / f"iteration-{iteration:04d}"))

    def sampling_policy(self, iteration):
        """The policy that samples round `iteration` of an overlapped run: the one in
        hand when the round's sampling begins, during the update before, that is the
        archive of iteration − 2; the starting checkpoint for the first two rounds. An
        archive that is gone (a run that was not overlapped before) gives way to the
        current policy."""
        if iteration < 2:
            return Path(self.settings.checkpoint)
        archive = self.policy / f"iteration-{iteration - 2:04d}"
        return archive if archive.exists() else self.policy / "current"

    def evaluation_pending(self, iteration):
        """Whether the policy of `iteration`'s update is due an evaluation not made yet."""
        entry = self.entry(iteration) if iteration >= 0 else {}
        return bool(self.settings.eval_seeds) and (iteration + 1) % self.settings.evaluate_every == 0 \
            and "update" in entry and "evaluation" not in entry

    def dispatch(self, iteration):
        """Hand the sampling side what it does during this iteration's update: the next
        round, by the policy of the iteration before this one, and that policy's
        evaluation when due."""
        job, following = dict(policy=str(self.sampling_policy(iteration + 1))), iteration + 1
        if following < self.iterations and "sample" not in self.entry(following) \
                and groups_summary(self.groups / f"round-{following}") is None:
            job["round"] = self.round_job(following, self.round_seeds(following))
        if self.evaluation_pending(iteration - 1):
            job["evaluation"] = self.evaluation_job(iteration - 1)
        if "round" in job or "evaluation" in job:
            self.side.dispatch(job)
            self.dispatched = job

    def collect(self, iteration):
        """Record what the sampling side did during the update. Returns it by kind."""
        if self.dispatched is None:
            return {}
        job, self.dispatched = self.dispatched, None
        results = self.side.collect()
        if "round" in results:
            self.record(job["round"]["iteration"], "sample", results["round"])
        if "evaluation" in results:
            self.record(job["evaluation"]["iteration"], "evaluation", results["evaluation"])
        return results

    def advantages(self, iteration):
        """The advantages of every wave of the iteration's round of act groups, from
        the groups themselves (`main.learn.groups`)."""
        written = group_advantages(self.groups / f"round-{iteration}")
        return {wave: dict(
            runs=head["runs"], decisions=head["decisions"], advantage=head["advantage"],
            **{key: head["stats"][key] for key in ("groups", "return_mean", "between_share", "uniform_groups")},
            tree_share=head["stats"]["tree"]["share"], branching_share=head["stats"]["tree"]["branching_share"],
            r2_within=head["stats"]["baseline"]["r2_within"]) for wave, head in written.items()}

    @staticmethod
    def remove(path):
        """Remove a file or a directory; returns the bytes it held."""
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            shutil.rmtree(path)
            return size
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            return size
        return 0

    def reference(self, iteration):
        """The checkpoint the KL of `iteration`'s update is to: the starting checkpoint,
        or with `reference_every` k the archive of iteration k·⌊N/k⌋ − 1, the policy as
        it stood when the current stretch of k iterations began."""
        every = self.settings.reference_every
        if not every or iteration < every:
            return Path(self.settings.checkpoint)
        return self.policy / f"iteration-{(iteration // every) * every - 1:04d}"

    def is_reference(self, number):
        """Whether the archive of iteration `number` serves as a KL reference later."""
        every = self.settings.reference_every
        return bool(every) and (number + 1) % every == 0

    def prune_archives(self, iteration):
        """Remove the policy archives of earlier iterations nobody evaluated, except
        those an evaluation is due of and those a later update's KL refers to."""
        evaluated = {int(number) for number, entry in self.state["iterations"].items() if entry.get("evaluation")}
        # The sampling side of an overlapped run still reads the archive before this iteration's.
        latest = iteration - 1 if self.settings.overlap else iteration
        freed = 0
        for archive in self.policy.glob("iteration-*"):
            number = int(archive.name.split("-")[1])
            if (number < latest and number not in evaluated and (number + 1) % self.settings.evaluate_every
                    and not self.is_reference(number)):
                freed += self.remove(archive)
        return freed

    def cleanup(self, iteration):
        """Remove the frames of the rounds of act groups up to this iteration (their
        shards and saves; the summaries, indexes and advantages stay), and policy
        archives of iterations not evaluated."""
        freed = 0
        for directory in group_rounds(self.groups):
            summary = groups_summary(directory)
            if summary["iteration"] > iteration:
                continue
            for wave in summary["waves"]:
                for name in ("accepted", act_groups.SAVES):
                    freed += self.remove(directory / wave / name)
        freed += self.prune_archives(iteration)
        return dict(freed_gb=round(freed / 2**30, 3))

    def update(self, iteration):
        """One pass over the iteration's round with the clipped objective
        (`grpo.group_update`), the KL to the policy the iterations started from."""
        from main.cli import _save

        settings, learner, directory = self.settings, self.learner, self.groups / f"round-{iteration}"
        current, archived = self.policy / "current", self.policy / f"iteration-{iteration:04d}"
        applied = getattr(learner, "checkpoint_metadata", {}) if self.resumed else {}
        if applied.get("iteration") == iteration and applied.get("update"):
            # The update was saved before the run was cut off: the policy in hand is its result.
            if not archived.exists():
                archive_checkpoint(current, archived)
            return applied["update"]
        waves = {wave: directory / wave for wave in groups_summary(directory)["waves"]}
        macros = batches.priced_macros(waves)
        updates = (batches.pass_updates(macros, learner.config.logical_batch_size) if settings.updates == "round"
                   else int(settings.updates))
        shares = settings.group_shares(macros)
        started = time.monotonic()
        told = lambda update: self.history.append_update(  # noqa: E731
            update, round=iteration, stage="groups", policy_version=learner.policy_version)
        # The KL is to the policy the iterations started from, or the one the current
        # stretch of `reference_every` iterations started from.
        anchored = self.reference(iteration)
        reference = grpo.Reference.load(anchored, self.device)
        metrics = grpo.group_update(
            learner, waves, reference, clip=settings.clip, kl=settings.kl, updates=updates,
            window_shards=settings.window_shards, weight_cap=settings.weight_cap, shares=shares,
            phase_weights=settings.phase_weights, on_update=told)
        del reference
        self.release()
        done = metrics["groups"]
        result = dict(
            policy=str(archived.relative_to(self.output)), policy_version=learner.policy_version,
            updates=learner.updates, pass_updates=max(wave["batches"] for wave in done["waves"].values()),
            round=directory.name, advantages=self.entry(iteration)["advantages"],
            loss=metrics["loss"], accuracy=metrics["accuracy"], entropy=metrics.get("entropy"),
            losses=done["losses"], shares=done["shares"], sizes=done["sizes"], decisions=done["decisions"],
            waves=done["waves"], grad_norm=metrics.get("grad_norm"), seconds=round(time.monotonic() - started, 1),
            wait_seconds=round(metrics["wait_seconds"], 1),
            wait_share=round(metrics["wait_seconds"] / max(metrics["duration_seconds"], 1e-9), 4),
            clip=done["clip"], kl=done["kl"], phase_weights=done["phase_weights"], seen=done["seen"],
            reference=str(anchored.relative_to(self.output) if anchored.is_relative_to(self.output) else anchored))
        seeds = set().union(*(RunShards(path).seeds for path in waves.values()))
        _save(current, learner, dict(
            runtime=self.runtime, iteration=iteration, update=result, advantage=GROUPS,
            training_seeds=sorted(seeds)))
        self.set_aside(archived)
        archive_checkpoint(current, archived)
        return result

    def evaluate(self, iteration):
        settings, directory = self.settings, self.output / "evaluation" / f"iteration-{iteration:04d}"
        if self.side is not None and not (directory / "report.json").is_file():
            # The sampling side evaluates the policy this iteration's update archived.
            return self.side.run(dict(policy=str(self.policy / f"iteration-{iteration:04d}"),
                                      evaluation=self.evaluation_job(iteration)))["evaluation"]
        result = play_evaluation(self.samplers(settings.eval_workers or settings.workers), self.sampler(),
                                 evaluation_seeds(settings), directory, iteration,
                                 getattr(self.learner, "checkpoint_metadata", {}), self.output)
        self.release()
        return result

    # ---- The loop ----

    def evaluation_due(self, iteration):
        """After every few iterations, of a policy that has not been evaluated yet."""
        if not self.settings.eval_seeds or (iteration + 1) % self.settings.evaluate_every:
            return False
        versions = {entry["evaluation"]["policy_version"] for entry in self.state["iterations"].values()
                    if entry.get("evaluation")}
        return self.learner.policy_version not in versions

    def iterate(self, iteration):
        """One iteration: sample, advantages, update, evaluation, cleanup. An update
        that leaves the policy's entropy under `min_entropy` ends the run once its
        iteration is complete (`Halt`)."""
        settings, started = self.settings, time.monotonic()
        self.record(iteration, "settings", asdict(settings))
        sampled = self.step(iteration, "sample", self.sample)
        advantages = self.step(iteration, "advantages", self.advantages)
        if self.side is not None:
            self.dispatch(iteration)
        if "update" not in self.entry(iteration):
            self.begin(iteration, "update")
            self.record(iteration, "update", self.update(iteration))
        update = self.entry(iteration)["update"]
        arrived = self.collect(iteration)
        # Overlapped, a policy's evaluation comes with the next iteration's sampling,
        # except the last iteration's, which nothing follows.
        if "evaluation" not in self.entry(iteration) and self.evaluation_due(iteration) \
                and (self.side is None or iteration + 1 >= self.iterations):
            self.step(iteration, "evaluation", self.evaluate)
        self.step(iteration, "cleanup", self.cleanup)
        evaluation, evaluated = self.entry(iteration).get("evaluation"), iteration
        if evaluation is None and "evaluation" in arrived:
            evaluation, evaluated = arrived["evaluation"], iteration - 1
        waves = sampled["waves"]
        seen = update["seen"]
        metrics = dict(
            updates=self.learner.updates, sampled_groups=sampled["groups"], sampled_segments=sampled["written"],
            sampled_win_rate=sampled["win_rate"],
            **{f"reach_rate_{wave}": rate for wave, rate in sampled["reach_rates"].items()},
            **{f"between_share_{wave}": advantages[wave]["between_share"] for wave in waves},
            **{f"r2_within_{wave}": advantages[wave]["r2_within"] for wave in waves},
            policy_updated=True, pass_updates=update["pass_updates"], loss=update["loss"],
            accuracy=update["accuracy"], entropy=update.get("entropy"),
            ratio_mean=seen["ratio_mean"], clip_fraction=seen["clip_fraction"], kl=seen["kl"],
            zero_share=seen["zero_share"], update_wait_share=update["wait_share"],
            **{f"{wave}_loss": update["losses"].get(wave) for wave in waves},
            **{f"{wave}_zero_share": update["waves"][wave]["zero_share"] for wave in waves})
        if evaluation:
            metrics.update(eval_runs=evaluation["runs"], eval_win_rate=evaluation["win_rate"],
                           eval_mean_floor=evaluation["mean_floor"], eval_iteration=evaluated)
        self.history.append(iteration, metrics, policy_version=self.learner.policy_version,
                            duration_seconds=time.monotonic() - started)
        self.record(iteration, "complete", True)
        entropy = update.get("entropy")
        if settings.min_entropy is not None and entropy is not None and entropy < settings.min_entropy:
            raise Halt(f"Iteration {iteration}'s update left the policy's entropy at {entropy:.4f}, "
                       f"below {settings.min_entropy}")

    def run(self, iterations):
        """Bring the output to `iterations` finished iterations. Returns the state."""
        if iterations < 1:
            raise ValueError("At least one iteration")
        self.iterations = iterations
        if self.settings.overlap:
            # Forked before this process touches the GPU, so that it may have a CUDA
            # context of its own, and before the history takes the output's lock.
            from main.overlap import SamplingSide

            self.side = SamplingSide(self.engine, self.settings, self.device)
        self.history = TrainingHistory(self.output, "iterate", asdict(self.settings))
        # A request to stop ends the step under way as an interruption does: the
        # engines and the loaders go with it.
        handled = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGTERM, _stop) if handled else None
        try:
            allow_compiled_variants()
            self.learner = self.load()
            if self.side is None:
                # Forked now, the processes share the heap of a policy just loaded; forked
                # after an update, each copies page by page the trainer's heap it allocates in.
                self.pool = Samplers(self.engine, self.learner.model, self.learner.vocabulary,
                                     engines=self.settings.workers, processes=self.settings.processes)
            self.history.status("running", updates=self.learner.updates)
            for iteration in range(iterations):
                if not self.entry(iteration).get("complete"):
                    self.iterate(iteration)
            self.history.status("completed", round=iterations - 1)
        except Halt as reason:
            self.state["halted"] = str(reason)
            write_json(self.output / "state.json", self.state)
            self.history.status("halted", reason=str(reason), **self.at)
            self.say(dict(halted=str(reason)))
        except KeyboardInterrupt:
            self.history.status("interrupted", **self.at)
            raise
        except BaseException as error:
            self.history.status("failed", error=str(error), **self.at)
            raise
        finally:
            if self.pool is not None:
                self.pool.close()
                self.pool = None
            if self.side is not None:
                self.side.close()
                self.side = None
            self.history.close()
            if handled:
                signal.signal(signal.SIGTERM, previous)
        return self.state
