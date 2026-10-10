"""Synthetic complete runs with out-of-combat decisions, and a tiny policy checkpoint."""

import json
from copy import deepcopy
from dataclasses import asdict

import pytest

from model.checkpoint import save_checkpoint
from model.config import ModelConfig, TrainConfig
from model.data import held_out
from model.model import PolicyNetwork
from model.protocol import SCHEMA
from model.representation import Vocabulary
from model.testing import demonstration

COMBAT = ("combat", 3, 99)


def synthetic_run(character, seed, *, victory, bosses, stands, forced_first=False):
    """A complete run with one out-of-combat macro-action per (phase, act, floor) of
    `stands`, followed by a combat macro-action. Each macro-action holds two choices
    and a forced finish; with `forced_first` the first one starts with a forced step."""
    base = demonstration(character, seed, 4, 2)
    selection, run_id = base["macros"][0], base["run_id"]
    macros = []
    for index, (phase, act, floor) in enumerate(list(stands) + [COMBAT]):
        macro = deepcopy(selection)
        macro["phase"] = phase
        if index == 0 and forced_first:
            macro["steps"].insert(0, deepcopy(macro["steps"][-1]))
        for number, step in enumerate(macro["steps"]):
            frame = step["frame"]
            frame["public"]["phase"] = phase
            frame["routing"]["decision_id"] = f"{run_id}:{index}:{number}"
            frame["routing"]["selection_id"] = f"{run_id}:selection:{index}"
            player = next(e for e in frame["public"]["entities"] if e["entity_type"] == "player")
            player.update(act=act, floor=floor, gold=10 * floor)
        macros.append(macro)
    return dict(schema=SCHEMA, source="evaluation", sampling="argmax", status="complete", victory=victory,
                ascension=10, character=character, seed=str(seed), run_id=run_id, contract=base["contract"],
                policy_version=3, policy_digest="synthetic", macros=macros,
                ledger=dict(version="outcome-v1", bosses=sorted(bosses)))


def seeds_on(side, count, holdout, prefix="seed"):
    """`count` seeds that the policy's split rule puts on `side` (True: held out)."""
    found, number = [], 0
    while len(found) < count:
        seed = f"{prefix}-{number}"
        if held_out({"seed": seed, "run_id": seed}, holdout) == side:
            found.append(seed)
        number += 1
    return found


STANDS = (("map", 1, 1), ("rewards", 1, 5), ("card_select", 2, 8), ("shop", 3, 12))


def frames_of(run):
    return [step["frame"] for macro in run["macros"] for step in macro["steps"]]


@pytest.fixture
def runs():
    """Six runs on three held-out and three training seeds (holdout 0.5), two of them won."""
    characters = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder", "Ironclad")
    seeds = seeds_on(True, 3, 0.5) + seeds_on(False, 3, 0.5)
    result = []
    for index, (character, seed) in enumerate(zip(characters, seeds)):
        victory = index % 3 == 0
        bosses = {1, 2, 3} if victory else {1} if index % 3 == 1 else set()
        result.append(synthetic_run(character, seed, victory=victory, bosses=bosses, stands=STANDS,
                                    forced_first=index == 1))
    return result


@pytest.fixture
def runs_dir(tmp_path, runs):
    directory = tmp_path / "runs"
    directory.mkdir()
    for run in runs:
        (directory / f"{run['character']}-{run['seed']}.json").write_text(json.dumps(run))
    return directory


@pytest.fixture
def profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(dict(
        model=asdict(ModelConfig.tiny()),
        # The policy's training section is not the critic's and must be ignored.
        training=dict(optimizer="muon_adamw", ascension=0),
        critic=dict(epochs=2, loader_workers=0, precision="no", logical_batch_size=4, microbatch_size=3,
                    warmup_updates=1, token_buckets=[32, 64, 128], action_buckets=[8, 16]))))
    return path


@pytest.fixture
def checkpoint(tmp_path, runs):
    """A policy checkpoint whose frozen vocabulary covers the runs' frames."""
    config = ModelConfig.tiny()
    vocabulary = Vocabulary.from_frames((f for run in runs for f in frames_of(run)), config)
    path = tmp_path / "policy"
    save_checkpoint(path, PolicyNetwork(config), vocabulary, training=TrainConfig(precision="no"),
                    progress=dict(policy_version=3))
    return path
