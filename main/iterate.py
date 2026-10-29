"""The training iteration of act groups: sample, value, update, evaluate, clean up.

One iteration, repeated (`python -m main iterate`):

1. **Sample** a round of act groups with the current policy at its temperatures, on
   seeds of the iteration's own (`main/play/groups.py`): segments of runs cut at the
   boundary acts, several from each start, one wave per stretch, written under
   `groups/round-N`. The fights are played by argmax and kept without frames: the
   round's decisions, and what the update learns from, are those outside them; every
   wave records its fights and their entry saves (`main/play/fights.py`). With
   `fight_groups` some of those fights are then played again from their entry saves,
   each several times to the fight's end, drawn at the fight temperature
   (`main/play/fight_groups.py`): a wave of their own beside the act waves. With
   `fight_hindsight` some of their segments, died or not, are then forked
   (`fight_groups.sample_hindsight`): replayed to a decision in danger, the one the
   policy is least sure of and the first of a round, there made to drink each potion,
   take each of two actions or left to draw, in a wave of their own.
   With `round_demos` (the BC phase) some of the fights the waves recorded are handed
   to the solver after the advantages (`main/play/demo_picks.py`, `main/demos/rounds.py`):
   its won fights, imported, are the round's combat demonstrations.
2. **Value**: every wave's decisions get their advantages from their own groups
   (`main/learn/groups.py`), returns passed back up the chain of waves, a win raised
   by `win_hp_bonus` times the share of its maximum HP it ended with; the groups whose
   segments all came to the same return are left out of the update. The fight groups'
   decisions get theirs from their groups too, each segment's return its fight's cost
   at the run's prices (`fight_prices`: the fixed ones, or a fit's with
   `potion_prices`; `main/learn/fight_groups.py`), with the same leaving out; so do
   the hindsight forks', each fork a group.
3. **Update**: one pass over the round, every logical batch drawing from each wave
   by its share (`batches.wave_shares`), with the clipped ratio and a KL to the
   reference policy (`grpo.group_update`): the starting checkpoint, renewed every
   `evaluate_every` iterations to the policy as it then stood (`Driver.reference`).
   With `round_demos` the same batches also hold the fights of the last `demo_buffer`
   rounds' demonstrations, learned by their NLL with no KL (`main/learn/mixed.py`,
   `Driver.demos`), at the kinds' shares (`source_shares`, `batches.stream_shares`);
   the pass still sees every wave once.
4. **Evaluate** the policy greedily on fixed seeds every few iterations; the
   evaluation keeps each run's outcome, not its frames.
5. **Clean up** what nothing reads any more: the round's frames, indexes and
   advantages, policy archives that were not evaluated and no later update refers to,
   what interrupted steps left behind.

The records are written under one output directory, the policies under a checkpoints
directory of the run's own (`checkpoints_for`: by default `checkpoints/` followed by
the output's path under `runs/`), and every step is skipped when its result is already
there: an interrupted run goes on from the step it was in. What an interrupted step
left is moved under `interrupted/`, out of the way of its second attempt, and removed
at the iteration's cleanup.

    groups/round-N/            the rounds: a directory per wave; after the update only the summaries
    groups/round-N/demos/      the fights handed to the solver, its reports and the import of its
                               won fights, kept while a later update may read them (`demo_buffer`)
    evaluation/iteration-NNNN/ the evaluation after iteration N, with report.json
    seeds/                     the seeds of each round
    state.json                 what every iteration's steps came to
    status.json, history.jsonl, updates.jsonl   the status, one line per iteration, one per update

and in the checkpoints directory

    current/, iteration-NNNN/  the policy after its update of iteration N
"""

import gc
import json
import shutil
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import torch

from main.env.seeds import excluded_seeds, round_ascensions, round_seeds

from main.demos import rounds as demo_rounds
from main.learn import batches, clipped, combat_demos, costs, grpo, mixed, potion_prices, streams
from main.learn.advantage import FILE as ADVANTAGES
from main.learn.fight_groups import advantages as fight_advantages
from main.learn.groups import KIND as GROUPS, advantages as group_advantages
from main.play import demo_picks, digests, fight_groups as fight_waves, fights as fight_files, groups as act_groups
from main.play.groups import group_rounds, groups_summary, sample_groups
from main.network.checkpoint import archive_checkpoint
from main.network.config import LearningRates, TrainConfig
from main.data.shards import RunShards
from main.play.evaluation import evaluate_runs
from main.learn.history import TrainingHistory
from main.learn.trainer import Learner
from main.env.protocol import CHARACTERS, ProtocolError
from main.play.rollout import PRECISIONS, SUMMARY_RUNS, RolloutRunner, collect_round, outside_rates, policy_digest
from main.play.sampling import Samplers
from main.env.seeds import validate_evaluation_seeds

FORMAT = "iterate-v1"
# The groups whose segments all came to one return are left out of the update
# (`groups.wave_advantages`): they have no signal, and the shares and the pass are then
# of the decisions kept. Off only where a test's tiny groups would leave nothing.
DROP_UNIFORM = True


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
    group_n: int = act_groups.N  # segments from each seed in the first wave
    group_m: list = field(default_factory=lambda: [act_groups.M])  # segments from each save, one or one per wave
    # Macro-actions of one optimizer step: a multiple of the profile's logical batch, whose
    # batches are accumulated into the step (`trainer.Learner.fit`); the logical batch when None.
    batch: int | None = None
    window_shards: int = 8
    evaluate_every: int = 5
    eval_seeds: str | None = None  # the seed file of the evaluations; without it none is made
    eval_runs_per_character: int | None = None  # the leading seeds of each character to play
    eval_workers: int | None = None
    # The precision the evaluations play at (`PRECISIONS`); the checkpoint's when None.
    # Sampling keeps the checkpoint's: the precision is part of what the samples are.
    eval_precision: str | None = None
    # The waves' relative shares of a batch, one per wave in order; by their decisions when None.
    wave_shares: list | None = None
    # A won run's return is 1 plus this times the share of its maximum HP it ended with
    # (`groups.win_bonuses`, from the fights' records): a tie-break among wins that parts
    # the groups whose segments all won. 0 adds nothing.
    win_hp_bonus: float = 0.0
    # Stop once an update's mean policy entropy on its act groups' decisions falls below this
    # (`guarded_entropy`: the demonstrations a batch also holds are not the policy's choices).
    min_entropy: float | None = None
    # A JSON file of the rates this run's updates train at (`LearningRates`); the checkpoint's without it.
    learning_rates: str | None = None
    clip: float = clipped.CLIP
    kl: float = clipped.KL
    # The KL coefficient of the fight groups' samples (and their hindsight forks'); `kl` when None.
    fight_kl: float | None = None
    # Passes an update makes over its round, every one at the round's sampled log-probabilities
    # (the clip holds the drift of the passes before it), the KL to the same reference.
    passes: int = 1
    # The act waves teach nothing of their own: their advantages are 0 and only their KL to the
    # reference pulls (`grpo.group_update(frozen=)`); what keeps the outside where it was while
    # the fights learn.
    freeze_outside: bool = False
    # The probability, by phase, that a sampled decision outside fights of the act groups
    # takes a candidate other than the policy's most likely (`RolloutRunner.explore_outside`,
    # `rollout.outside_rates`): a number for every phase or a mapping by phase.
    explore_outside: object = 0.0
    # The probability, by phase, that a sampled decision outside fights of the act groups
    # asks the prior (`prior`, a checkpoint of the policy's vocabulary) for its most likely
    # candidate and takes it where the policy's most likely differs
    # (`RolloutRunner.explore_prior`, `rollout.outside_rates`). The prior only proposes:
    # nothing pulls the policy toward it.
    explore_prior: object = 0.0
    prior: str | None = None
    # Only the explored decisions outside fights that the tree of their group prices
    # learn (`groups.wave_advantages(explored_on_tree=)`); the others get no advantage.
    explored_on_tree: bool = False
    # The explored decisions outside fights that learn have their advantages centered by
    # phase in each wave (`groups.wave_advantages(explored_centered=)`).
    explored_centered: bool = False
    # The ascension window a round's groups are played in, [lowest, highest]: each
    # (character, seed) group at one of its ascensions (`round_ascensions`); at the
    # checkpoint's when None. The evaluations stay at the checkpoint's.
    ascensions: list | None = None
    # After its act waves a round plays this many of the fights they recorded again, each
    # `fight_group_size` times from its entry save to the fight's end (`fight_groups`):
    # picked normal, elite and boss 3:4:4, a `fight_group_priority` share of each kind by
    # deaths and cost and the rest evenly. Drawn at the fight temperature, every frame
    # kept. 0 plays none.
    fight_groups: int = 0
    fight_group_size: int = fight_waves.SIZE
    fight_group_priority: float = fight_waves.PRIORITY
    # After its fight groups a round forks this many of their segments, died or not
    # (`fight_groups.sample_hindsight`): each replayed to the decisions it parts at
    # (`fight_groups.branch_points`: in danger, least sure, a round's first), there
    # arms drink each potion, take each of two actions or draw. The update learns them
    # beside the fight groups, a stream of theirs. 0 forks none; it needs `fight_groups`.
    fight_hindsight: int = 0
    # How the fight groups share their fights among the kinds ({kind: share}; None is
    # `fight_groups.SHARES`), the kinds whose groups are of a size of their own
    # ({kind: segments}), the fight evaluation sets whose entry saves give a
    # `fight_pool_share` of them (`fight_groups.pool_rows`), and how the hindsight
    # trunks are shared among the kinds.
    fight_group_shares: object = None
    fight_group_kind_size: object = None
    fight_pool: list = field(default_factory=list)
    fight_pool_share: float = 0.0
    fight_hindsight_shares: object = None
    # The round's own fights are picked at their natural weights (`fight_groups.pick(natural=)`):
    # every run once in each act, not the act groups' fan-out that crowds the last act.
    fight_group_natural: bool = False
    # The prices file of a fit (`fit-prices`' prices.json) the fight groups are picked
    # and priced at (`fight_prices`); the fixed prices without. An iteration's fight
    # groups are priced at the prices they were picked at.
    potion_prices: str | None = None
    # The growth reward: the HP each growth event of a fight takes off its cost
    # (`costs.Growing`: a card grown for the run, a relic's count left near its trigger),
    # where its fight groups are picked and priced. 0 counts none; it needs fight groups.
    growth_bonus: float = 0.0
    # The kinds of source's relative shares of an update's batch ({kind: number}, the
    # kinds of `mixed.KINDS` the update has, every one of them): by their decisions
    # when None (`batches.stream_shares`).
    source_shares: dict | None = None
    # The BC phase: after its advantages a round hands this many of the fights its waves
    # recorded to the solver (`demo_picks.pick`: normal, elite and boss 3:4:4 at their
    # natural frequency, the fights set up by a selection first up to `demo_opening` of
    # them; `demos.rounds.solve_round` on `demo_workers` runs at once), and its update
    # learns the solver's won fights of the last `demo_buffer` rounds beside the act
    # groups, by their NLL (`combat_demos.fights`, `grpo.group_update`). 0 hands none.
    round_demos: int = 0
    demo_buffer: int = 3
    demo_opening: float = demo_picks.OPENING
    demo_workers: int = demo_rounds.WORKERS

    def __post_init__(self):
        if self.runs_per_character < 1 or self.evaluate_every < 1:
            raise ValueError("Runs per character and the evaluation interval are positive")
        self.explore_outside = outside_rates(self.explore_outside)
        self.explore_prior = outside_rates(self.explore_prior)
        if self.explore_prior and not self.prior:
            raise ValueError("Proposals of a prior need the prior's checkpoint")
        if type(self.explored_on_tree) is not bool:
            raise ValueError("Learning only the tree's explored decisions is on or off")
        if type(self.explored_centered) is not bool:
            raise ValueError("Centering the explored decisions' advantages is on or off")
        if type(self.fight_groups) is not int or self.fight_groups < 0:
            raise ValueError("Fight groups: a number of fights, 0 or more")
        bonus = self.win_hp_bonus
        if isinstance(bonus, bool) or not isinstance(bonus, (int, float)) or not 0 <= bonus < 1:
            raise ValueError("A win's HP bonus is a weight in [0, 1): a win stays worth more than a loss by most")
        for name in ("fight_group_shares", "fight_hindsight_shares"):
            shares = getattr(self, name)
            if shares is not None and (set(shares) != set(fight_waves.SHARES) or not sum(shares.values()) > 0
                                       or any(v < 0 for v in shares.values())):
                raise ValueError(f"{name}: a share for each of normal, elite and boss, not all 0")
        sizes = self.fight_group_kind_size
        if sizes is not None and (not set(sizes) <= set(fight_waves.SHARES)
                                  or any(type(n) is not int or n < 2 for n in sizes.values())):
            raise ValueError("fight_group_kind_size: kinds of fight, each 2 segments or more")
        if not 0 <= self.fight_pool_share <= 1 or bool(self.fight_pool_share) != bool(self.fight_pool):
            raise ValueError("A fight pool and its share in (0, 1] go together")
        if type(self.fight_group_natural) is not bool or (self.fight_group_natural and not self.fight_groups):
            raise ValueError("A natural pick of the fight groups is on or off, and needs fight groups")
        if self.fight_pool and not self.fight_groups:
            raise ValueError("A fight pool gives fights to the fight groups: it needs fight_groups")
        if type(self.fight_hindsight) is not int or self.fight_hindsight < 0:
            raise ValueError("Hindsight forks: a number of segments, 0 or more")
        if self.fight_hindsight and not self.fight_groups:
            raise ValueError("Hindsight forks are of the fight groups' segments: they need fight groups")
        if self.potion_prices is not None:
            if not self.fight_groups:
                raise ValueError("Potion prices price the fight groups: they need fight groups")
            potion_prices.load(self.potion_prices)  # a file of prices, read now rather than mid-round
        growth = self.growth_bonus
        if isinstance(growth, bool) or not isinstance(growth, (int, float)) or growth < 0:
            raise ValueError("The growth reward is a number of HP, 0 or more")
        if growth and not self.fight_groups:
            raise ValueError("The growth reward prices the fight groups: it needs fight groups")
        if type(self.fight_group_size) is not int or self.fight_group_size < 2:
            raise ValueError("A fight group has at least two segments")
        priority = self.fight_group_priority
        if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not 0 <= priority <= 1:
            raise ValueError("The share of the fights picked by priority is in [0, 1]")
        if self.learning_rates is not None:
            LearningRates.load(self.learning_rates)
        if not 0 < self.clip < 1 or self.kl < 0:
            raise ValueError("The clip ε is in (0, 1) and the KL coefficient not negative")
        if self.fight_kl is not None and (self.fight_kl < 0 or not self.fight_groups):
            raise ValueError("The fight groups' KL coefficient is not negative, and needs fight groups")
        if type(self.passes) is not int or self.passes < 1:
            raise ValueError("An update makes one pass or more")
        if type(self.freeze_outside) is not bool:
            raise ValueError("Freezing the outside is on or off")
        if self.batch is not None and (type(self.batch) is not int or self.batch < 1):
            raise ValueError("An optimizer step's batch is a positive number of macro-actions")
        if self.eval_precision is not None and self.eval_precision not in PRECISIONS:
            raise ValueError(f"The evaluation precision is one of {sorted(PRECISIONS)}")
        names = act_groups.waves(self.boundaries)
        self.group_m = act_groups.per_wave(self.group_m, len(names) - 1)
        if self.group_n < 1:
            raise ValueError("Act groups: at least one segment from each seed")
        if self.wave_shares is not None:
            self.wave_shares = [float(share) for share in self.wave_shares]
            batches.wave_shares(dict.fromkeys(names, 1), self.wave_shares)
        if type(self.round_demos) is not int or self.round_demos < 0:
            raise ValueError("Combat demonstrations by the round: a number of fights, 0 or more")
        if type(self.demo_buffer) is not int or self.demo_buffer < 1:
            raise ValueError("An update learns the demonstrations of one round or more")
        opening = self.demo_opening
        if isinstance(opening, bool) or not isinstance(opening, (int, float)) or not 0 <= opening <= 1:
            raise ValueError("The share of the fights set up by a selection is in [0, 1]")
        if type(self.demo_workers) is not int or self.demo_workers < 1:
            raise ValueError("The solver fights at least one run at a time")
        if self.source_shares is not None:
            shares = dict(self.source_shares)
            unknown = set(shares) - set(mixed.KINDS)
            if unknown or not shares:
                raise ValueError(f"The shares of the sources name kinds of {list(mixed.KINDS)}, not {sorted(unknown)}")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0
                   for value in shares.values()):
                raise ValueError("A source's share is a positive number")
            self.source_shares = {kind: float(shares[kind]) for kind in mixed.KINDS if kind in shares}
        if self.ascensions is not None:
            window = list(self.ascensions)
            if len(window) != 2 or not all(type(a) is int for a in window) or not 0 <= window[0] <= window[1] <= 10:
                raise ValueError("The ascension window is two integers, lowest and highest, from 0 to 10")
            self.ascensions = window

    def group_shares(self, macros, others=None):
        """The shares of an update's batch, `macros` being each wave's priced
        decisions and `others` the decisions of the update's other kinds of source:
        the kinds by their decisions or at `source_shares`, the waves by their
        decisions or at `wave_shares` (`batches.stream_shares`)."""
        return batches.stream_shares(macros, others, waves=self.wave_shares, sources=self.source_shares)


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


def samplers_for(pool, engines, settings, engine_factory, model, vocabulary, prior=None):
    """The sampling processes of a round of `engines` engines: those of the run
    (`pool`), whose engines rest after the round, or for another number of engines
    processes of the round's own."""
    if engines == settings.workers and pool is not None:
        return _Resting(pool)
    return Samplers(engine_factory, model, vocabulary, engines=engines, processes=settings.processes, prior=prior)


def accumulation(settings, config):
    """How many of the profile's logical batches make one optimizer step of the
    settings' `batch` macro-actions."""
    if settings.batch is None:
        return 1
    if settings.batch % config.logical_batch_size:
        raise ValueError(f"An optimizer step's batch ({settings.batch}) is a multiple of the profile's "
                         f"logical batch ({config.logical_batch_size}), whose batches it accumulates")
    return settings.batch // config.logical_batch_size


def fight_prices(settings):
    """The prices a fight's cost is counted at, what a round's fight groups are picked
    by and their segments' returns: the solver's fixed prices (`costs.FIXED`), or with
    `potion_prices` a fit's (`main/learn/potion_prices.py`, `fit-prices`); with
    `growth_bonus` they carry the growth reward (`costs.with_growth`)."""
    return costs.with_growth(potion_prices.load(settings.potion_prices), settings.growth_bonus)


def fight_cost(fight, prices=costs.FIXED):
    """What a fight a round recorded cost, in HP, at `prices`: the order its fight
    groups are picked by (`fight_groups.pick`)."""
    return costs.cost(fight, prices).total


def play_round(samplers, runner, seeds, directory, settings, output, ascensions=None):
    """A round of act groups on `seeds` into `directory` (`main.play.groups`), by the
    sampling processes `samplers` (a context manager), each group at its ascension in
    `ascensions` when given, and with `settings.fight_groups` its fight groups after it
    (`fight_groups.sample_fight_groups`), played by the runner without greedy fights.
    A round cut off on these seeds and ascensions goes on from the wave it was in; one
    on others is set aside. Returns the summary."""
    directory = Path(directory)
    listed, levels = directory / "metadata" / "seeds.json", directory / "metadata" / "ascensions.json"
    same = (listed.is_file() and json.loads(listed.read_text()) == seeds
            and (json.loads(levels.read_text()) if levels.is_file() else None) == ascensions)
    if directory.exists() and not same:
        set_aside(output, directory)
    with samplers as pool:
        summary = groups_summary(directory)
        if summary is None:
            summary = sample_groups(pool, runner, seeds, directory, n=settings.group_n, m=settings.group_m,
                                    boundaries=settings.boundaries, ascensions=ascensions, resume=True,
                                    fight_entries=True)
        if settings.fight_groups:
            # Their fights drawn at the fight temperature, not greedy: that is what a group compares.
            fights = runner.variant(greedy_fights=False, explore_outside=0.0, explore_prior=0.0)
            summary = fight_waves.sample_fight_groups(
                pool, fights, directory, settings.fight_groups, cost=partial(fight_cost, prices=fight_prices(settings)),
                key=f"{settings.seed_root}:{runner.iteration}", size=fight_group_sizes(settings),
                priority=settings.fight_group_priority, shares=settings.fight_group_shares or fight_waves.SHARES,
                pool=fight_waves.pool_rows(settings.fight_pool) if settings.fight_pool else None,
                pool_share=settings.fight_pool_share, natural=settings.fight_group_natural)
        if settings.fight_hindsight:
            # Drawn as the fight groups are, forced only at their forks.
            forks = runner.variant(greedy_fights=False, explore_outside=0.0, explore_prior=0.0)
            summary = fight_waves.sample_hindsight(pool, forks, directory, settings.fight_hindsight,
                                                   key=f"{settings.seed_root}:{runner.iteration}",
                                                   shares=settings.fight_hindsight_shares or fight_waves.SHARES)
        return summary


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
        explored={wave: summary[wave].get("explored", 0) for wave in waves},
        forked={wave: summary[wave].get("forked", 0) for wave in waves},
        by_ascension={wave: summary[wave].get("by_ascension") for wave in waves},
        collector={wave: summary[wave].get("collector") for wave in waves},
        policy_version=summary["policy_version"],
        policy_digest=summary["policy_digest"], temperature=summary["temperature"],
        greedy_fights=summary.get("greedy_fights", False), seconds=summary["seconds"],
        # With fight entries, what each wave's fights wrote (`fights.FightWriter.summary`).
        **({"fights": {wave: summary[wave].get("fights") for wave in waves}} if summary.get("fight_entries") else {}),
        **({"fight_groups": fight_groups_result(summary["fight_groups"], directory, settings)}
           if summary.get("fight_groups") else {}),
        **({"fight_hindsight": fight_groups_result(summary["fight_hindsight"], directory, settings,
                                                   fight_waves.HINDSIGHT)}
           if summary.get("fight_hindsight") else {}))


def fight_group_sizes(settings):
    """The fight groups' size (`fight_groups.group_size`): one for every kind, or by kind."""
    if not settings.fight_group_kind_size:
        return settings.fight_group_size
    return {kind: settings.fight_group_kind_size.get(kind, settings.fight_group_size) for kind in fight_waves.SHARES}


def fight_groups_result(wave, directory, settings, name=fight_waves.WAVE):
    """What the `sample` step records of a round's fight groups (`fight_groups`), or of
    its hindsight forks: the pick (each kind's fights by priority and drawn evenly, the
    kinds short of their quota; the forks) and how the segments came out."""
    failed = sum(wave["failed"].values())
    if failed > settings.quarantine * wave["segments"]:
        raise ProtocolError(f"The {name} of round {directory} finished {wave['written']} of "
                            f"{wave['segments']} segments; inspect the quarantine of {name}")
    return dict(groups=wave["groups"], segments=wave["segments"], written=wave["written"], failed=failed,
                outcomes=wave["outcomes"], picks=wave["picks"], decisions=sum(wave["phases"].values()),
                explored=wave["explored"], forked=wave["forked"], fights=wave["fights"], seconds=wave["seconds"],
                collector=wave["collector"])


def fight_groups_advantages(head):
    """What the `advantages` step records of a round's fight groups: the advantage
    header (`fight_groups.advantages`) without its tree's counts, by kind of fight
    the diagnostics of the groups."""
    stats = head["stats"]
    return dict(runs=head["runs"], decisions=head["decisions"], advantage=head["advantage"], prices=head["prices"],
                **{key: stats[key] for key in ("groups", "return_mean", "between_share", "within_variance",
                                               "uniform_groups")},
                dropped=stats.get("dropped"), tree_share=stats["tree"]["share"],
                branching_share=stats["tree"]["branching_share"], explored=stats["explored"], forked=stats["forked"],
                explored_by_hp=stats.get("explored_by_hp"), kinds=stats["kinds"],
                **({"hindsight": stats["hindsight"]} if "hindsight" in stats else {}))


# What an iteration's history keeps of every kind of source's report (`mixed.Mixed.means`).
MIX_METRICS = ("samples", "loss", "entropy", "accuracy", "kl", "clip_fraction", "ratio_mean", "zero_share", "nll",
               *(f"ratio_{entry}_{kind}" for kind in clipped.RATIO_KINDS for entry in ("max", "over")))


def mix_metrics(update):
    """The flat history entries of an update's report by kind of source, `<kind>_<entry>`
    (`act_groups_kl`, `combat_demos_loss`, ...), and the repeats of its demonstrations."""
    metrics = {f"{kind.replace('-', '_')}_{key}": report[key]
               for kind, report in update.get("mix", {}).items() for key in MIX_METRICS if key in report}
    if update.get("demos"):
        metrics["combat_demos_repeats"] = update["demos"]["repeats"]
    return metrics


def guarded_entropy(update):
    """The entropy `min_entropy` guards: the policy's on the act groups' decisions of an
    update (`mix`), its entropy on the whole batch when the update has no kinds."""
    return update.get("mix", {}).get(mixed.ACT, {}).get("entropy", update.get("entropy"))


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
    """The iterations of one output directory, their policies in `checkpoints`.
    `engine` makes an engine for a sampling process: the real one unless a test hands
    in another."""

    def __init__(self, output, settings, device="cpu", *, checkpoints, engine=None, runtime=None, say=None):
        self.output, self.settings, self.device = Path(output), settings, device
        self.runtime = runtime or {}
        self.say = say or (lambda value: print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True))
        if engine is None:
            from main.env.engine import CliEngine

            engine = lambda: CliEngine(root=settings.engine_root)  # noqa: E731
        self.engine = engine
        self.policy, self.groups = Path(checkpoints), self.output / "groups"
        self.learner = self.history = self.prior = None
        # The sampling processes of every round, forked before anything is trained.
        self.pool = None
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
        if (self.output / "policy").exists():
            raise ValueError(f"{self.output / 'policy'} holds this run's policies; they are kept in "
                             f"{self.policy}: move them there")
        kept = self.state.setdefault("checkpoints", str(self.policy))
        if Path(kept).resolve() != self.policy.resolve():
            raise ValueError(f"The policies of {self.output} are in {kept}, not {self.policy}")
        write_json(path, self.state)

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
        current = self.policy / "current"
        self.resumed = current.exists()
        if not self.resumed and any(entry.get("update") for entry in self.state["iterations"].values()):
            raise ValueError(f"{self.output} has updated its policy, but {current} is missing")
        rates = self.settings.learning_rates
        learner, _ = Learner.load(current if self.resumed else self.settings.checkpoint, self.device,
                                  training=TrainConfig.from_profile(self.settings.config)
                                  if self.settings.config else None,
                              rates=LearningRates.load(rates) if rates else None)
        accumulation(self.settings, learner.config)
        return learner

    def load_prior(self):
        """The prior's network of `explore_prior`, for inference, or None; its vocabulary
        must be the policy's: the sampling processes encode for both alike."""
        if not self.settings.explore_prior:
            return None
        prior, _ = Learner.load(self.settings.prior, self.device, inference=True)
        if prior.vocabulary.digest != self.learner.vocabulary.digest:
            raise ValueError("The prior's vocabulary differs from the policy's")
        return prior.model

    def excluded(self):
        """Seeds no round may be played on: the evaluations', the listed files', and
        what the policy was trained on."""
        settings = self.settings
        return excluded_seeds([*settings.exclude_seeds, *([settings.eval_seeds] if settings.eval_seeds else [])],
                              getattr(self.learner, "checkpoint_metadata", {}).get("training_seeds", []))

    def sampler(self, iteration=None, precision=None):
        """The runner of a round, or of an evaluation at `precision`."""
        learner, settings = self.learner, self.settings
        return RolloutRunner(
            learner.model, learner.vocabulary, precision=precision or learner.config.precision,
            version=learner.policy_version,
            max_steps=settings.max_steps, ascension=learner.config.ascension,
            temperatures=learner.config.temperatures,
            explore_outside=settings.explore_outside, explore_prior=settings.explore_prior, prior=self.prior,
            greedy_fights=True, iteration=iteration)

    def samplers(self, engines):
        """The sampling processes of a round of `engines` engines (`samplers_for`)."""
        return samplers_for(self.pool, engines, self.settings, self.engine, self.learner.model,
                            self.learner.vocabulary, self.prior)

    def round_seeds(self, iteration):
        """The seeds of an iteration's round, written under `seeds/`."""
        settings = self.settings
        excluded = self.excluded()
        seeds = round_seeds(settings.seed_root, iteration, settings.runs_per_character, excluded)
        if excluded & {seed for listed in seeds.values() for seed in listed}:
            raise ProtocolError("A round's seeds overlap the seeds kept apart from sampling")
        write_json(self.output / "seeds" / f"iteration-{iteration:04d}.json", seeds)
        return seeds

    def round_ascensions(self, iteration, seeds):
        """The ascension of each group of an iteration's round (`round_ascensions`),
        written under `seeds/`; None without a window."""
        settings = self.settings
        if settings.ascensions is None:
            return None
        levels = round_ascensions(settings.seed_root, iteration, seeds, settings.ascensions)
        write_json(self.output / "seeds" / f"iteration-{iteration:04d}-ascensions.json", levels)
        return levels

    def sample(self, iteration):
        """A round of act groups (`main.play.groups`) on seeds of the iteration's own:
        `runs_per_character` seeds of each character, `group_n` segments from each,
        `group_m` from each save; and its fight groups (`fight_groups`)."""
        settings, directory = self.settings, self.groups / f"round-{iteration}"
        summary = groups_summary(directory)
        if summary is None or (settings.fight_groups and not summary.get("fight_groups")) \
                or (settings.fight_hindsight and not summary.get("fight_hindsight")):
            seeds = self.round_seeds(iteration)
            summary = play_round(self.samplers(settings.workers), self.sampler(iteration), seeds, directory,
                                 settings, self.output, self.round_ascensions(iteration, seeds))
            self.release()
        found = round_result(summary, directory, settings)
        if settings.fight_groups:
            # The prices the fight groups were picked at: their returns are counted at the same.
            found["prices"] = potion_prices.digest(fight_prices(settings))
        return found

    def advantages(self, iteration):
        """The advantages of every wave of the iteration's round of act groups, from
        the groups themselves (`main.learn.groups`), and of its fight groups, if it
        has them (`main.learn.fight_groups`), under their wave's name."""
        directory = self.groups / f"round-{iteration}"
        written = group_advantages(directory, drop_uniform=DROP_UNIFORM, win_hp_bonus=self.settings.win_hp_bonus,
                                   explored_on_tree=self.settings.explored_on_tree,
                                   explored_centered=self.settings.explored_centered)
        found = {wave: dict(
            runs=head["runs"], decisions=head["decisions"], advantage=head["advantage"],
            **{key: head["stats"][key] for key in ("groups", "return_mean", "between_share", "uniform_groups")},
            dropped=head["stats"].get("dropped"),
            tree_share=head["stats"]["tree"]["share"], branching_share=head["stats"]["tree"]["branching_share"],
            r2_within=head["stats"]["baseline"]["r2_within"], explored=head["stats"].get("explored"),
            forked=head["stats"].get("forked"), fights=head["stats"].get("fights"),
            by_ascension=head["stats"].get("by_ascension"),
            **({"explored_outside": head["stats"]["explored_outside"]}
               if (head["stats"].get("explored_outside") or {}).get("decisions") else {}),
            **({"win_bonus": head["stats"]["win_bonus"]} if "win_bonus" in head["stats"] else {}))
            for wave, head in written.items()}
        if groups_summary(directory).get("fight_groups"):
            prices = fight_prices(self.settings)
            picked = self.entry(iteration).get("sample", {}).get("prices")
            if picked is not None and picked != potion_prices.digest(prices):
                raise ValueError(f"Iteration {iteration}'s fight groups were picked at other prices: finish it with "
                                 "the potion prices and growth reward it was sampled with")
            found[fight_waves.WAVE] = fight_groups_advantages(fight_advantages(directory, prices, drop_uniform=DROP_UNIFORM))
            if groups_summary(directory).get("fight_hindsight"):
                found[fight_waves.HINDSIGHT] = fight_groups_advantages(fight_advantages(
                    directory, prices, drop_uniform=DROP_UNIFORM, wave=fight_waves.HINDSIGHT))
        return found

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
        """The checkpoint the KL of `iteration`'s update is to: with `evaluate_every` k
        the archive of iteration k·⌊N/k⌋ − 1, the policy as it stood when the current
        stretch of k iterations began (an evaluated one), the starting checkpoint
        before the first."""
        every = self.settings.evaluate_every
        if iteration < every:
            return Path(self.settings.checkpoint)
        return self.policy / f"iteration-{(iteration // every) * every - 1:04d}"

    def prune_archives(self, iteration):
        """Remove the policy archives of earlier iterations that nothing reads any more.
        An evaluated archive is kept, as are one an evaluation is due of, the one the
        next update's KL refers to, and this iteration's own."""
        evaluated = {int(number) for number, entry in self.state["iterations"].items() if entry.get("evaluation")}
        due = {number for number in range(iteration) if self.settings.eval_seeds
               and (number + 1) % self.settings.evaluate_every == 0}
        reference = self.reference(iteration + 1)
        freed = 0
        for archive in self.policy.glob("iteration-*"):
            number = int(archive.name.split("-")[1])
            if number < iteration and number not in evaluated and number not in due and archive != reference:
                freed += self.remove(archive)
        return freed

    def cleanup(self, iteration):
        """Remove what nothing reads after this iteration's update: the shards, saves,
        indexes and advantages of the rounds of act groups up to it, their fights'
        records and entry saves, and their fight groups' shards, records, pick and
        advantages (the summaries and quarantines stay), the policy archives nothing refers to any more
        (`prune_archives`), and what interrupted steps left under `interrupted/`. Of a
        round's demonstrations the solver's samples and export go once they are imported,
        and the import once no later update takes it (`demo_buffer`); the pick and the
        solver's reports stay."""
        freed = 0
        for directory in group_rounds(self.groups):
            summary = groups_summary(directory)
            if summary["iteration"] > iteration:
                continue
            demos = directory / demo_picks.DIRECTORY
            for name in ("samples", demo_rounds.EXPORT):
                freed += self.remove(demos / demo_rounds.SOLVED / name)
            if summary["iteration"] <= iteration - self.settings.demo_buffer + 1:
                freed += self.remove(demos / demo_rounds.IMPORTED)
            # A round's fight groups are read last by its advantages step, before the cleanup.
            for wave in [*summary["waves"], fight_waves.WAVE, fight_waves.HINDSIGHT]:
                for name in ("accepted", act_groups.SAVES, "index.jsonl.gz", ADVANTAGES, digests.FILE,
                             fight_files.SAVES, fight_files.FILE, fight_waves.PICKS):
                    freed += self.remove(directory / wave / name)
        freed += self.prune_archives(iteration)
        freed += self.remove(self.output / "interrupted")
        return dict(freed_gb=round(freed / 2**30, 3))

    def demos(self, iteration):
        """The directories of combat demonstrations `iteration`'s update learns beside its
        round: with `round_demos` the imports of the solver's fights of this round and the
        `demo_buffer` − 1 before it; None for none. An update draws its share of them with
        a key of its own (`demo_key`)."""
        listed = []
        if self.settings.round_demos:
            for number in range(max(0, iteration - self.settings.demo_buffer + 1), iteration + 1):
                solved = self.state["iterations"].get(str(number), {}).get("demos")
                if solved:
                    listed.append(solved["data"])
        return listed or None

    def solve_demos(self, iteration):
        """Hand `round_demos` of the fights the iteration's round recorded to the solver
        (`demo_picks.pick_round`, seeded by `demo_key`) and import its won fights
        (`demo_rounds.solve_round`): the pick's account, what the solver and the import
        came to, and the data directory. A pick already written stands."""
        settings, directory = self.settings, self.groups / f"round-{iteration}"
        target = directory / demo_picks.DIRECTORY
        selection = target / demo_picks.SELECTION
        if selection.is_file():
            account = json.loads(selection.read_text(encoding="utf-8"))["account"]
        else:
            selection, account = demo_picks.pick_round(directory, groups_summary(directory)["waves"],
                                                       settings.round_demos, self.demo_key(iteration),
                                                       opening=settings.demo_opening)
        started = time.monotonic()
        found = demo_rounds.solve_round(target, workers=settings.demo_workers,
                                        log=lambda line: self.say(dict(iteration=iteration, solver=line)))
        return dict(found, selection=str(selection), account=account, seconds=round(time.monotonic() - started, 1))

    def demo_key(self, iteration):
        """What an iteration's draw of combat demonstrations is seeded by (`combat_demos.draw`)."""
        return f"{self.settings.seed_root}:{iteration}"

    def update(self, iteration):
        """One pass over the iteration's round with the clipped objective
        (`grpo.group_update`), the KL to the reference of `Driver.reference`: its act
        waves, its fight groups and their hindsight forks if it has them (learned as its
        act waves are, two streams of one kind), and with
        combat demonstrations (`demos`) their fights in the same batches, by their NLL."""
        settings, learner, directory = self.settings, self.learner, self.groups / f"round-{iteration}"
        current, archived = self.policy / "current", self.policy / f"iteration-{iteration:04d}"
        applied = getattr(learner, "checkpoint_metadata", {}) if self.resumed else {}
        if applied.get("iteration") == iteration and applied.get("update"):
            # The update was saved before the run was cut off: the policy in hand is its result.
            if not archived.exists():
                archive_checkpoint(current, archived)
            return applied["update"]
        summary = groups_summary(directory)
        waves = {wave: directory / wave for wave in summary["waves"]}
        macros = batches.priced_macros(waves)
        fights = {name: directory / name for name, part in ((fight_waves.WAVE, "fight_groups"),
                                                            (fight_waves.HINDSIGHT, "fight_hindsight"))
                  if summary.get(part)} or None
        listed = self.demos(iteration)
        demos = combat_demos.fights(listed) if listed else None
        accumulate = accumulation(settings, learner.config)
        others = {**({mixed.FIGHTS: batches.priced_macros(fights)} if fights is not None else {}),
                  **({mixed.DEMOS: demos.macros} if demos is not None else {})}
        shares = settings.group_shares(macros, others or None)
        # The pass in logical batches: every act wave seen once (their part of every batch
        # is the whole of it without other sources).
        act = sum(streams.sizes(learner.config.logical_batch_size, shares)[wave] for wave in waves)
        updates = batches.pass_updates(macros, act)
        started = time.monotonic()
        told = lambda update: self.history.append_update(  # noqa: E731
            update, round=iteration, stage="groups", policy_version=learner.policy_version)
        # The KL is to the policy the iterations started from, or the one the current
        # stretch of `evaluate_every` iterations started from.
        anchored = self.reference(iteration)
        reference = clipped.Reference.load(anchored, self.device)
        # The reference played the round: its distributions are the ones the decisions
        # were drawn from, kept as they were sampled, and it need not run again.
        recorded = (bool(summary.get("distributions")) and policy_digest(reference.model) == summary["policy_digest"]
                    and all(bool(summary[part].get("distributions"))
                            for part in ("fight_groups", "fight_hindsight") if summary.get(part)))
        if recorded:
            reference = clipped.Reference.recorded()
            self.release()
        kl = settings.kl if settings.fight_kl is None else {mixed.ACT: settings.kl, mixed.FIGHTS: settings.fight_kl}
        passes = []
        for number in range(settings.passes):
            # Every pass at the round's sampled log-probabilities: the clip holds the drift of the ones before.
            metrics = grpo.group_update(
                learner, waves, reference, fights=fights, demos=demos,
                demo_key=self.demo_key(iteration) if not number else f"{self.demo_key(iteration)}:pass-{number}",
                clip=settings.clip, kl=kl, updates=updates, window_shards=settings.window_shards, shares=shares,
                on_update=told, accumulate=accumulate, frozen={mixed.ACT} if settings.freeze_outside else None)
            seen = metrics["groups"]["seen"]
            passes.append(dict(kl=seen["kl"], clip_fraction=seen["clip_fraction"], ratio_mean=seen["ratio_mean"],
                               loss=metrics["loss"], mix={kind: {key: part.get(key) for key in ("kl", "clip_fraction")}
                                                          for kind, part in metrics["groups"]["mix"].items()}))
        del reference
        self.release()
        done = metrics["groups"]
        result = dict(
            policy=str(archived), policy_version=learner.policy_version,
            updates=learner.updates,
            pass_updates=-(-max(wave["batches"] for wave in done["waves"].values()) // accumulate),
            batch=done["batch"], accumulate=accumulate,
            round=directory.name, advantages=self.entry(iteration)["advantages"],
            loss=metrics["loss"], accuracy=metrics["accuracy"], entropy=metrics.get("entropy"),
            losses=done["losses"], shares=done["shares"], sizes=done["sizes"], decisions=done["decisions"],
            waves=done["waves"], grad_norm=metrics.get("grad_norm"), seconds=round(time.monotonic() - started, 1),
            wait_seconds=round(metrics["wait_seconds"], 1),
            wait_share=round(metrics["wait_seconds"] / max(metrics["duration_seconds"], 1e-9), 4),
            shapes=metrics["shapes"],
            clip=done["clip"], kl=done["kl"], phase_weights=done["phase_weights"], seen=done["seen"],
            reference=str(anchored), reference_recorded=recorded,
            # The round's fights were greedy: nothing of them was learned (`RolloutRunner.greedy_fights`).
            greedy_fights=summary.get("greedy_fights", False),
            # By kind of source (`mixed.KINDS`): its decisions, share, samples, loss, accuracy and KL.
            mix=done["mix"], passes=passes, **({"fights": done["fights"]} if "fights" in done else {}),
            **({"fight_streams": done["fight_streams"]} if "fight_streams" in done else {}),
            **({"demos": done["demos"]} if "demos" in done else {}))
        seeds = set().union(*(RunShards(path).seeds for path in waves.values()))
        learner.save(current, dict(
            runtime=self.runtime, iteration=iteration, update=result, advantage=GROUPS,
            training_seeds=sorted(seeds)))
        self.set_aside(archived)
        archive_checkpoint(current, archived)
        return result

    def evaluate(self, iteration):
        settings, directory = self.settings, self.output / "evaluation" / f"iteration-{iteration:04d}"
        result = play_evaluation(self.samplers(settings.eval_workers or settings.workers),
                                 self.sampler(precision=settings.eval_precision),
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
        if settings.round_demos:
            self.step(iteration, "demos", self.solve_demos)
        if "update" not in self.entry(iteration):
            self.begin(iteration, "update")
            self.record(iteration, "update", self.update(iteration))
        update = self.entry(iteration)["update"]
        if "evaluation" not in self.entry(iteration) and self.evaluation_due(iteration):
            self.step(iteration, "evaluation", self.evaluate)
        self.step(iteration, "cleanup", self.cleanup)
        evaluation = self.entry(iteration).get("evaluation")
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
            # Where the pass's drift had taken the explored decisions when each was seen (`clipped.Clipped`).
            **{key: seen.get(key) for key in ("explored", "explored_up_log_ratio", "explored_down_log_ratio",
                                              "explored_outside_clip", "explored_beyond")},
            **{f"{wave}_loss": update["losses"].get(wave) for wave in waves},
            **{f"{wave}_zero_share": update["waves"][wave]["zero_share"] for wave in waves},
            **mix_metrics(update))
        solved = self.entry(iteration).get("demos")
        if solved:
            metrics.update(demo_fights=solved["account"]["picked"],
                           demo_opening=sum(kind["opening"] for kind in solved["account"]["kinds"].values()),
                           demo_runs=solved["imported"]["accepted_runs"], demo_macros=solved["imported"]["macros"],
                           demo_seconds=solved["seconds"])
        if evaluation:
            metrics.update(eval_runs=evaluation["runs"], eval_win_rate=evaluation["win_rate"],
                           eval_mean_floor=evaluation["mean_floor"])
        self.history.append(iteration, metrics, policy_version=self.learner.policy_version,
                            duration_seconds=time.monotonic() - started)
        self.record(iteration, "complete", True)
        entropy = guarded_entropy(update)
        if settings.min_entropy is not None and entropy is not None and entropy < settings.min_entropy:
            raise Halt(f"Iteration {iteration}'s update left the policy's entropy at {entropy:.4f}, "
                       f"below {settings.min_entropy}")

    def run(self, iterations):
        """Bring the output to `iterations` finished iterations. Returns the state."""
        if iterations < 1:
            raise ValueError("At least one iteration")
        self.history = TrainingHistory(self.output, "iterate", asdict(self.settings))
        # A request to stop ends the step under way as an interruption does: the
        # engines and the loaders go with it.
        handled = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGTERM, _stop) if handled else None
        try:
            allow_compiled_variants()
            self.learner = self.load()
            self.prior = self.load_prior()
            # Forked now, the processes share the heap of a policy just loaded; forked
            # after an update, each copies page by page the trainer's heap it allocates in.
            self.pool = Samplers(self.engine, self.learner.model, self.learner.vocabulary,
                                 engines=self.settings.workers, processes=self.settings.processes, prior=self.prior)
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
            self.history.close()
            if handled:
                signal.signal(signal.SIGTERM, previous)
        return self.state
