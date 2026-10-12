"""One split identity for raw game seeds and imported independent samples."""

from spire_codex_data.sources import split_group

from .protocol import CHARACTERS

SPLIT_VERSION = "seed-group-v1"


def seed_group(run):
    """A canonical split group, including the two legacy seed representations.

    Independent imports already stored the group's digest as `seed`; do not hash
    it again. Other runs carry the game seed. New index rows name the group explicitly.
    """
    if "split_group" in run:
        return str(run["split_group"])
    seed = str(run.get("seed", run.get("run_id")))
    return seed if run.get("source") in {"independent", "student_battle"} else split_group(seed)


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
