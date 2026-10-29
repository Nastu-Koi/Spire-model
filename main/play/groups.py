"""Act groups: segments of runs played several times from one state.

A run is cut at the first decision of each boundary act (`BOUNDARIES`: the second
and the third) into as many waves as there are stretches between the cuts, each named
by where it starts: `act1` from the seed's start, `act2` from the save at the second
act's first decision. A later wave plays from the continue save a segment of the wave
before wrote at the cut, the run going on from the map (`rollout.RolloutRunner.run`).
A group is the segments played from one start, `n`
from each (character, seed) in the first wave and `m` from each save in the later
ones; its mean return is the baseline each segment is measured against. A group is
at one ascension: a first-wave group at its seed's when the round is given ascensions
(`sample_groups(..., ascensions=)`), a later one at its save's. A segment of
a wave with a cut plays to the cut, where the engine's save is taken, or to the run's
end before it; one of the last wave plays to the end. The groups of a wave are those
of every segment of the wave before that reached the cut, so that segment's return
can be the win rate of its own group (`main/learn/groups.py`); nothing prices a
state here.

A round of groups is the waves in turn, each a pass of the sampling processes, the
saves written as they arrive. A round that was cut off goes on from where it stood
(`sample_groups(..., resume=True)`): the waves that finished stand, the wave under
way keeps the segments it finished and plays the rest. Each wave is written as a data directory of its own
(`act1/`, `act2/`, ...): shards and an index of the finished segments, every frame
kept (fights included, unless they are greedy), a `summary.json`; with fight entries, the records of the
segments' fights and their entry saves (`main/play/fights.py`). A segment that did not finish is counted and
set aside under `quarantine/`; its group is smaller, the round goes on. Segments are
named by their group: `<character>:<seed>:act1` and `<parent segment>:<wave>`. A
round's fight groups (`main/play/fight_groups.py`) are written beside its waves in a
directory of their own, by `WaveWriter` as well; they are not one of its `waves`.
"""

import gzip
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from main.data.index import INDEX, ShardWriter, index_row
from main.data.shards import RunShards
from main.env.protocol import CHARACTERS
from main.play import digests
from main.play.fights import FightWriter
from main.play.rollout import SEGMENTS, write_run

FORMAT = "act-groups-v2"
BOUNDARIES = (2, 3)
N, M = 8, 8
SAVES = "saves"


def cuts(boundaries=BOUNDARIES):
    """The cuts of a round in the order a run meets them: ("act", k) at the first
    decision of act k (k from 2)."""
    boundaries = [int(act) for act in boundaries]
    if not boundaries:
        raise ValueError("A round of act groups has at least one cut")
    if boundaries != sorted(set(boundaries)) or boundaries[0] < 2:
        raise ValueError("Boundary acts rise from the second act on")
    return [("act", act) for act in boundaries]


def wave_name(cut):
    return "act1" if cut is None else f"{cut[0]}{cut[1]}"


def waves(boundaries=BOUNDARIES):
    """The waves of a round cut at `boundaries` (acts), named by where each starts:
    `act1`, `act2`, `act3`."""
    return [wave_name(None), *(wave_name(cut) for cut in cuts(boundaries))]


def until(cut):
    """Where a segment stops, as `rollout.RolloutRunner.run` takes it."""
    return dict(until_act=None if cut is None else cut[1])


def per_wave(m, count):
    """`m` segments per save for each wave after the first: one number for all of
    them, or one for each."""
    listed = [int(value) for value in (m if isinstance(m, (list, tuple)) else [m])]
    if len(listed) == 1:
        listed = listed * count
    if len(listed) != count or min(listed, default=1) < 1:
        raise ValueError(f"Segments per save: one number, or one for each of the {count} later waves, each at least 1")
    return listed


def start_jobs(seeds, n, wave, stop, ascensions=None):
    """The jobs of the first wave: each seed of each character `n` times, to the cut
    `stop` (`cuts`; None plays to the end). `ascensions`
    (character → seed → ascension) gives every segment of a seed's group its seed's
    ascension; without it the segments are at the runner's."""
    jobs = []
    for character in CHARACTERS:
        for seed in seeds.get(character, []):
            group = f"{character}:{seed}:{wave}"
            level = {} if ascensions is None else dict(ascension=ascensions[character][str(seed)])
            for number in range(n):
                jobs.append((character, dict(seed=str(seed), run_id=f"{group}-{number}", group=group,
                                             **until(stop), **level), None))
    return jobs


def resume_jobs(rows, m, wave, stop):
    """The jobs of a later wave: `m` segments from each save row (`save_rows`) of the
    wave before, each to the cut `stop` or, with None, to the run's end, at the
    ascension the save was written at."""
    jobs = []
    for row in rows:
        group = f"{row['run_id']}:{wave}"
        level = {} if row.get("ascension") is None else dict(ascension=row["ascension"])
        for number in range(m):
            jobs.append((row["character"], dict(seed=row["seed"], run_id=f"{group}-{number}", group=group,
                                                resume=row["path"], **until(stop), **level), None))
    return jobs


def save_path(directory, run_id):
    return Path(directory) / SAVES / (run_id.replace(":", "-") + ".json.gz")


def write_save(directory, run, save):
    """A segment's save (`{"at": …, "save": …}`, as `rollout.RolloutRunner.run` returns
    it), in a file of its own; returns its row."""
    path = save_path(directory, run["run_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(save, stream, separators=(",", ":"))
    return dict(run_id=run["run_id"], group=run["group"], character=run["character"], seed=run["seed"],
                ascension=run.get("ascension"), at=save["at"], act=run.get("reached_act"), path=str(path))


def read_save(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def save_rows(directory, wave):
    """The rows of the saves a wave wrote (`<wave>/saves.jsonl`), oldest first."""
    path = Path(directory) / wave / "saves.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class WaveWriter:
    """The files of one wave: the shards and index of its finished segments, their
    digest (`digests`), the quarantine of the rest, and the counts of its summary;
    with `fight_entries`, the records and entry saves of their fights (`fights`)."""

    def __init__(self, directory, shard_samples=2000, *, resume=False, fight_entries=False):
        """With `resume`, the segments a cut-off wave finished stand (`done` names
        them) and the rest are written after them."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.shards = ShardWriter(self.directory, shard_samples, flush=True, resume=resume)
        # Gone on with, a wave's digest keeps the lines of the segments that stand.
        self.digests = (self.directory / digests.FILE).open("ab" if resume else "wb")
        self.outcomes, self.failed, self.phases = Counter(), Counter(), Counter()
        self.groups = set()
        # Each ascension's finished segments and their outcomes.
        self.ascensions = defaultdict(Counter)
        self.written = self.stored_steps = self.explored = self.forked = 0
        # The segments whose decisions kept the distributions they were drawn from.
        self.recorded = 0
        self.done = {row["run_id"] for row in self.shards.rows}
        self.resumed = len(self.shards.rows)
        for row in self.shards.rows:
            self.outcomes[row["outcome"]] += 1
            self.phases.update(row["phases"])
            self.groups.add(row["group"])
            self.ascensions[row.get("ascension")][row["outcome"]] += 1
            self.written += 1
            self.explored += row.get("explored") or 0
            self.forked += row.get("forked") or 0
            self.recorded += bool(row.get("distributions"))
        # The contracts of the rows found, from one of their runs each.
        for row in self.shards.rows:
            if row["contract"] not in self.shards.contracts:
                run = RunShards(self.directory, [row]).read([row["shard"]])[0]
                self.shards.contracts[row["contract"]] = run["contract"]
        self.fights = FightWriter(self.directory, self.done if resume else None) if fight_entries else None

    def add(self, run):
        """A segment as its sampling process handed it over (`sampling._handed`). Its
        fights' records are written once the segment is: a finished segment's only."""
        records = run.pop("fight_records", None)
        if run["status"] != "complete":
            directory = self.directory / "quarantine"
            directory.mkdir(exist_ok=True)
            (directory / (run["run_id"].replace(":", "-") + ".json.gz")).write_bytes(run["member"])
            self.failed[run.get("error") or run["status"]] += 1
            return False
        self.shards.write(run["member"], index_row(run, phases=run["phases"]), run["stored_steps"], run["contract"])
        # After the shard and its index: a digest line is never of a segment the index lacks.
        if run.get("digest") is not None:
            self.digests.write(run["digest"])
            self.digests.flush()
        if self.fights is not None and records is not None:
            self.fights.add(records)
        self.outcomes[run["outcome"]] += 1
        self.phases.update(run["phases"])
        self.groups.add(run["group"])
        self.ascensions[run.get("ascension")][run["outcome"]] += 1
        self.written += 1
        self.stored_steps += run["stored_steps"]
        self.explored += run.get("explored") or 0
        self.forked += run.get("forked") or 0
        self.recorded += bool(run.get("distributions"))
        return True

    def close_files(self):
        self.shards.close()
        self.digests.close()
        if self.fights is not None:
            self.fights.close()

    def tally(self):
        """What a wave's summary counts of the segments it wrote."""
        return dict(written=self.written, resumed=self.resumed, groups=len(self.groups),
                    outcomes=dict(self.outcomes), failed=dict(self.failed), phases=dict(sorted(self.phases.items())),
                    explored=self.explored, forked=self.forked, distributions=self.recorded == self.written,
                    by_ascension={str(level): dict(written=sum(counts.values()), outcomes=dict(counts))
                                  for level, counts in sorted(self.ascensions.items(), key=lambda x: str(x[0]))},
                    stored_steps=self.stored_steps, shards=len(self.shards.shards), contracts=self.shards.contracts)

    def close(self, wave, stop, jobs, seconds, collector=None):
        """`stop` is the wave's cut (`cuts`; None for the last wave); `collector` the
        collecting process's account of the wave (`Samplers.timing`), if any."""
        self.close_files()
        summary = dict(kind="act_groups_wave", format=FORMAT, wave=wave,
                       until_act=None if stop is None else stop[1],
                       until=None if stop is None else list(stop), segments=len(jobs),
                       **self.tally(), seconds=round(seconds, 1), collector=collector)
        if self.fights is not None:
            summary["fights"] = self.fights.summary()
        write_summary(self.directory, summary)
        return summary


def write_summary(directory, summary):
    path = Path(directory) / "summary.json"
    stage = path.with_suffix(".tmp")
    stage.write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=1) + "\n", encoding="utf-8")
    stage.replace(path)


def wave_summary(directory):
    """The summary of a finished wave, or None."""
    path = Path(directory) / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary if summary.get("kind") == "act_groups_wave" and summary.get("format") == FORMAT else None


def set_aside(directory):
    """Move a wave's directory under the round's `interrupted/`, out of the way of a
    second attempt."""
    directory = Path(directory)
    aside = directory.parent / "interrupted"
    aside.mkdir(exist_ok=True)
    number = 0
    while (aside / f"{directory.name}.{number}").exists():
        number += 1
    shutil.move(str(directory), str(aside / f"{directory.name}.{number}"))


def sample_groups(samplers, runner, seeds, directory, *, n=N, m=M, boundaries=BOUNDARIES, ascensions=None,
                  shard_samples=2000, on_run=None, resume=False, fight_entries=False):
    """Sample one round of act groups into `directory`: the first wave over `seeds`
    (per character), then a wave over the saves of each wave before; the cuts are the
    first decisions of the `boundaries` acts (`cuts`). The round is
    played by the engines of `samplers` (`sampling.Samplers`) at the runner's
    temperatures; with its greedy fights (`RolloutRunner.greedy_fights`) the fights are
    played by argmax and kept without frames.
    Returns the summary: the waves in order, each wave's own summary
    under its name, and the saves each wave wrote. A segment that did not finish is
    set aside, not written. With `resume`, a round cut off in `directory` goes on:
    finished waves stand, the wave under way keeps what it finished and plays the
    rest; a wave left without an index is set aside and played again. With `ascensions` (character → seed →
    ascension) every group is played at its seed's ascension, its later waves at the
    ascension their saves were written at. With `fight_entries` every wave also writes
    the records of its segments' fights and the entry saves of those not begun by an
    event (`fights`: `<wave>/fights.jsonl`, `<wave>/fights/`); the segments are played
    as they are without.
    """
    stops = [*cuts(boundaries), None]
    names = [wave_name(None), *(wave_name(cut) for cut in stops[:-1])]
    later = per_wave(m, len(names) - 1)
    if n < 1:
        raise ValueError("Act groups: at least one segment from each seed")
    # With greedy fights a round keeps no frames of its fights: they are the policy's
    # fixed play, and no update learns from them (`RolloutRunner.greedy_fights`).
    segments = dict(SEGMENTS, combat_frames=False) if runner.greedy_fights else SEGMENTS
    if fight_entries:
        segments = dict(segments, fight_entries=True)
    directory = Path(directory)
    if groups_summary(directory) is not None or (
            not resume and directory.exists() and any(directory.iterdir())):
        raise FileExistsError("Act groups directory already holds a round; choose a new one")
    directory.mkdir(parents=True, exist_ok=True)
    write_run(directory / "metadata" / "seeds.json", seeds)
    if ascensions is not None:
        write_run(directory / "metadata" / "ascensions.json", ascensions)
    started = time.monotonic()

    summaries, saves, rows = {}, {}, []
    for level, (wave, stop) in enumerate(zip(names, stops)):
        jobs = (start_jobs(seeds, n, wave, stop, ascensions) if level == 0
                else resume_jobs(rows, later[level - 1], wave, stop))
        finished = wave_summary(directory / wave) if resume else None
        if finished is not None and finished["wave"] == wave and finished["segments"] == len(jobs) \
                and finished["written"] + sum(finished["failed"].values()) == len(jobs):
            summaries[wave] = finished
            rows = save_rows(directory, wave)
            if stop is not None:
                saves[wave] = len(rows)
            continue
        partial = resume and (directory / wave / INDEX).is_file()
        if (directory / wave).exists() and not partial:
            set_aside(directory / wave)
        writer = WaveWriter(directory / wave, shard_samples, resume=partial, fight_entries=fight_entries)
        begun, rows = time.monotonic(), (save_rows(directory, wave) if partial else [])
        rows = [row for row in rows if row["run_id"] in writer.done]
        pending = [job for job in jobs if job[1]["run_id"] not in writer.done]
        listing = (directory / wave / "saves.jsonl").open("a", encoding="utf-8") if stop is not None else None

        def arrived(index, run, wave=wave, writer=writer, listing=listing, rows=rows):
            save = run.pop("save", None)
            added = writer.add(run)
            if added and run["outcome"] == "reached":
                if save is None or listing is None:
                    raise ValueError("A segment that reached the cut has no save, or its wave takes none")
                row = write_save(directory / wave, run, save)
                listing.write(json.dumps(row) + "\n")
                listing.flush()
                rows.append(row)
            if on_run:
                on_run(wave, index, run)

        try:
            if pending:
                samplers.collect(runner, pending, sample=True, on_run=arrived, layout=segments)
        finally:
            # A wave cut off keeps readable shards and index, and no summary: it is
            # not finished.
            if listing is not None:
                listing.close()
            writer.close_files()
        collector = getattr(samplers, "timing", None) if pending else None
        summaries[wave] = writer.close(wave, stop, jobs, time.monotonic() - begun, collector)
        if stop is not None:
            saves[wave] = len(rows)

    summary = dict(
        kind="act_groups", format=FORMAT, iteration=runner.iteration, policy_version=runner.version,
        policy_digest=runner.digest, temperature=dict(runner.temperatures), greedy_fights=runner.greedy_fights,
        ascension=runner.ascension,
        n=n, m=later, boundaries=[int(act) for act in boundaries],
        cuts=[list(cut) for cut in stops[:-1]], waves=names, seeds="metadata/seeds.json",
        ascensions=None if ascensions is None else "metadata/ascensions.json",
        groups=sum(len(seeds.get(c, [])) for c in CHARACTERS), saves=saves,
        # Every decision of the round kept the distribution it was drawn from (`WaveWriter`).
        distributions=all(summaries[wave].get("distributions") for wave in names),
        **summaries, seconds=round(time.monotonic() - started, 1))
    if fight_entries:
        # Each wave's summary says what its fights wrote (`fights`).
        summary["fight_entries"] = True
    write_summary(directory, summary)
    return summary


def groups_summary(directory):
    """The summary of a finished round of act groups, or None."""
    path = Path(directory) / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary if summary.get("kind") == "act_groups" and summary.get("format") == FORMAT else None


def group_rounds(directory):
    """The finished rounds of act groups under `directory` (`round-N/`), oldest
    iteration first."""
    found = []
    for path in Path(directory).glob("round-*"):
        summary = groups_summary(path) if path.is_dir() else None
        if summary and type(summary.get("iteration")) is int:
            found.append((summary["iteration"], path.name, path))
    return [path for _, _, path in sorted(found)]
