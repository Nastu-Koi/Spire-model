"""A small world for the tests of the iteration: runs through several rooms
(`model.testing.Journey`), a policy checkpoint whose vocabulary covers them,
demonstrations as a sharded data directory and a critic profile. No test of its own."""

import json
import random
from dataclasses import asdict

import torch

from model.buffer import sample_round
from model.cli import _save
from model.config import ModelConfig, TrainConfig
from model.dataset_index import ShardWriter, index_row, member
from model.model import PolicyNetwork
from model.protocol import CHARACTERS, clean_frame, execution_command
from model.representation import Vocabulary
from model.rollout import RolloutRunner
from model.sampling import Samplers
from model.testing import Journey, demonstration

TEMPERATURES = {"combat": 0.5, "other": 0.8}
ROOMS = 3


def engine():
    return Journey(4, 2, rooms=ROOMS)


def walked(character):
    """The frames of one run through the rooms, taking the first candidate throughout."""
    played = engine()
    frame, frames = played.reset(character, "vocabulary", 0), []
    while frame["boundary"] != "terminal":
        frames.append(clean_frame(frame))
        frame = played.send(execution_command(frame, frame["legal"]["candidates"][0]["candidate_ref"]))
    return frames


def learner(seed=0, **training):
    """A tiny policy whose vocabulary covers the rooms and the demonstrations."""
    from model.trainer import Learner

    config = ModelConfig.tiny()
    frames = [frame for character in CHARACTERS for frame in walked(character)]
    frames += [step["frame"] for character in CHARACTERS
               for macro in demonstration(character, "vocabulary", 4, 2)["macros"] for step in macro["steps"]]
    settings = dict(optimizer="adamw", precision="no", loader_workers=0, logical_batch_size=8, microbatch_size=4,
                    bootstrap_warmup_updates=0, ascension=0, combat_temperature=TEMPERATURES["combat"],
                    other_temperature=TEMPERATURES["other"])
    torch.manual_seed(seed)
    return Learner(PolicyNetwork(config), Vocabulary.from_frames(frames, config), TrainConfig(**{**settings, **training}))


def checkpoint(path, **training):
    """A saved policy that resumes strictly, as the iterations start from one. The
    generators it is saved with are seeded: what resumes from it shuffles the same way."""
    policy = learner(**training)
    random.seed(0)
    _save(path, policy)
    return path


def demonstrations(directory, seeds=4):
    """Demonstrations of every character as a sharded data directory."""
    directory.mkdir(parents=True)
    writer = ShardWriter(directory, 30)
    for character in CHARACTERS:
        for number in range(seeds):
            run = demonstration(character, f"demonstration-{number}", 4, 2)
            writer.write(member(run), index_row(run), sum(len(macro["steps"]) for macro in run["macros"]),
                         run["contract"])
    writer.close()
    (directory / "summary.json").write_text(json.dumps(dict(contracts=writer.contracts, shards=len(writer.shards))))
    return directory


def profile(path, **critic):
    """A critic profile of the tiny architecture, packed and trained in this process."""
    settings = dict(epochs=1, loader_workers=0, precision="no", logical_batch_size=4, microbatch_size=3,
                    warmup_updates=1, token_buckets=[32, 64, 128], action_buckets=[8, 16], holdout=0.5)
    path.write_text(json.dumps(dict(model=asdict(ModelConfig.tiny()), critic={**settings, **critic})))
    return path


def sampled_round(policy, directory, iteration=0, runs=2, name="round"):
    """A round of `runs` runs per character played by `policy`, written as a buffer
    reads it; returns its summary."""
    runner = RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                           ascension=0, temperatures=TEMPERATURES, iteration=iteration)
    seeds = {character: [f"{name}-{iteration}-{character}-{index}" for index in range(runs)]
             for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        return sample_round(pool, runner, seeds, directory)
