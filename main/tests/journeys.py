"""A small world for the tests of the iteration: runs through several rooms
(`main.play.testing.Journey`), a policy checkpoint whose vocabulary covers them and
sampled rounds. No test of its own."""

import random

import torch

from main.play.buffer import sample_round
from main.cli import _save
from main.network.config import ModelConfig, TrainConfig
from main.network.model import PolicyNetwork
from main.env.protocol import CHARACTERS, execution_command
from main.encoding.representation import clean_frame
from main.encoding.representation import Vocabulary
from main.play.rollout import RolloutRunner
from main.play.sampling import Samplers
from main.play.testing import Journey, demonstration

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
    from main.learn.trainer import Learner

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


def sampled_round(policy, directory, iteration=0, runs=2, name="round"):
    """A round of `runs` runs per character played by `policy`, written as a buffer
    reads it; returns its summary."""
    runner = RolloutRunner(policy.model, policy.vocabulary, version=policy.policy_version, precision="no",
                           ascension=0, temperatures=TEMPERATURES, iteration=iteration)
    seeds = {character: [f"{name}-{iteration}-{character}-{index}" for index in range(runs)]
             for character in CHARACTERS}
    with Samplers(engine, policy.model, policy.vocabulary, engines=4, processes=2) as pool:
        return sample_round(pool, runner, seeds, directory)
