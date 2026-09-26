"""Random game-seed queues with a checkpointed, independent generation stream."""

import random
import secrets

from .protocol import CHARACTERS, fingerprint


def validate_evaluation_seeds(seeds, progress):
    if set(seeds) != set(CHARACTERS) or any(
        not isinstance(seeds[c], list) or not seeds[c] for c in CHARACTERS
    ):
        raise ValueError("Evaluation requires a nonempty seed list for each character")
    values = [str(seed) for c in CHARACTERS for seed in seeds[c]]
    if len(values) != len(set(values)):
        raise ValueError("Evaluation seeds must be unique across characters")
    known = set(map(str, progress.get("training_seeds", [])))
    known.update(
        map(str, (progress.get("random_seed_schedule") or {}).get("used_seeds", []))
    )
    if known.intersection(values):
        raise ValueError("Evaluation seed overlaps checkpoint training seeds")
    return seeds


class RandomSeedSchedule:
    def __init__(self, runs_per_character=None, *, state=None):
        if runs_per_character is None:
            runs_per_character = (state or {}).get("runs_per_character", 4)
        if type(runs_per_character) is not int or runs_per_character < 1:
            raise ValueError("Runs per character must be a positive integer")
        self.runs_per_character = runs_per_character
        if state is None:
            self.root = secrets.token_hex(16)
            self.next_round = 0
            self.used = set()
        else:
            if state.get("version") != "random-game-seeds-v1":
                raise ValueError("Unsupported random seed schedule")
            self.root = state["root"]
            self.next_round = state["next_round"]
            self.used = set(state["used_seeds"])
            if (
                not isinstance(self.root, str)
                or not self.root
                or type(self.next_round) is not int
                or self.next_round < 0
            ):
                raise ValueError("Invalid random seed schedule")

    def next(self):
        # A separate PRNG keeps the environment queue independent of PPO's
        # shuffling and Torch sampling. The initial root uses fresh OS entropy.
        rng = random.Random(fingerprint([self.root, self.next_round]))
        queue = {}
        for character in CHARACTERS:
            queue[character] = []
            while len(queue[character]) < self.runs_per_character:
                seed = f"{rng.getrandbits(64):016X}"
                if seed in self.used:
                    continue
                self.used.add(seed)
                queue[character].append(seed)
        self.next_round += 1
        return queue

    def state_dict(self):
        return {
            "version": "random-game-seeds-v1",
            "root": self.root,
            "next_round": self.next_round,
            "runs_per_character": self.runs_per_character,
            "used_seeds": sorted(self.used),
        }
