"""One split identity for raw game seeds and imported independent samples; the seeds
of sampled rounds, derived from a root, and the seeds kept apart from them."""

import hashlib
import json
from pathlib import Path

from main.env.protocol import CHARACTERS

SPLIT_VERSION = "seed-group-v1"


def split_group(seed):
    """Runs of one seed share their map and offers: they must stay on one side of a split."""
    value = str(seed).upper().replace("O", "0").replace("I", "1").strip()
    return hashlib.sha256(value.encode()).hexdigest()


def seed_group(run):
    """The split group of a run or index row: named outright (`split_group`), or
    derived from its seed. An independent import's `seed` is the group's digest
    already (its index rows, imported without `source`, are read as such); a
    sampled run carries the game seed, which is hashed."""
    if "split_group" in run:
        return str(run["split_group"])
    seed = str(run.get("seed", run.get("run_id")))
    return seed if run.get("source") in {"independent"} else split_group(seed)


def validate_evaluation_seeds(seeds, progress):
    if set(seeds) != set(CHARACTERS) or any(
        not isinstance(seeds[c], list) or not seeds[c] for c in CHARACTERS
    ):
        raise ValueError("Evaluation requires a nonempty seed list for each character")
    values = [str(seed) for c in CHARACTERS for seed in seeds[c]]
    groups = set(map(split_group, values))
    if len(values) != len(groups):
        raise ValueError("Evaluation seeds must be unique across characters")
    known = set(map(str, progress.get("training_seeds", [])))
    # Rollouts are recorded under the game seed itself, independent supervised
    # samples under the split group of the seed they were rebuilt from.
    # Legacy checkpoints mix raw seeds and group digests; checking both forms is
    # conservative without guessing whether a string is itself a digest.
    if groups.intersection(known | set(map(split_group, known))):
        raise ValueError("Evaluation seed overlaps checkpoint training seeds")
    return seeds


def canonical(seed):
    """A seed as the game reads it: it takes O for 0 and I for 1."""
    return str(seed).upper().replace("O", "0").replace("I", "1").strip()


def seeds_of(path):
    """The seeds a file names: a mapping of characters to seeds, as rounds and
    evaluations are assigned them, or a plain list."""
    data = json.loads(Path(path).read_text())
    lists = data.values() if isinstance(data, dict) else [data]
    return {str(seed) for seeds in lists for seed in seeds}


def excluded_seeds(files, training_seeds=()):
    """The seeds no sampled round may be played on: those the files name (`seeds_of`:
    evaluation and test sets) and those a policy was trained on."""
    seeds = set(map(str, training_seeds))
    for path in files:
        seeds |= seeds_of(path)
    return seeds


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


def round_ascensions(root, iteration, seeds, window):
    """The ascension of each (character, seed) group of a round, in `window` [lowest,
    highest]: each character's seeds take the window's ascensions in turn from a start
    derived from the root, the iteration and the character, so that every ascension
    has as many of a character's groups as another, give or take one, the same each
    time the round is asked for."""
    lowest, highest = window
    width = highest - lowest + 1
    levels = {}
    for character in CHARACTERS:
        start = int(hashlib.sha256(f"{root}:{iteration}:{character}:ascension".encode()).hexdigest(), 16) % width
        levels[character] = {str(seed): lowest + (start + place) % width
                             for place, seed in enumerate(seeds.get(character, []))}
    return levels
