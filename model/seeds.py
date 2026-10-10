"""Evaluation seeds, kept apart from everything a checkpoint was trained on."""

from spire_codex_data.sources import split_group

from .protocol import CHARACTERS


def validate_evaluation_seeds(seeds, progress):
    if set(seeds) != set(CHARACTERS) or any(
        not isinstance(seeds[c], list) or not seeds[c] for c in CHARACTERS
    ):
        raise ValueError("Evaluation requires a nonempty seed list for each character")
    values = [str(seed) for c in CHARACTERS for seed in seeds[c]]
    if len(values) != len(set(values)):
        raise ValueError("Evaluation seeds must be unique across characters")
    known = set(map(str, progress.get("training_seeds", [])))
    # Rollouts are recorded under the game seed itself, independent supervised
    # samples under the split group of the seed they were rebuilt from.
    if known.intersection(values) or known.intersection(map(split_group, values)):
        raise ValueError("Evaluation seed overlaps checkpoint training seeds")
    return seeds
