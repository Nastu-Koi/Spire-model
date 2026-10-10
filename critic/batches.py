"""Decision frames packed as the policy's trainer packs them.

One row per frame; the batch of a set of frames is split under the policy's
budgets (`model.data.microbatches`: shortest first, padded to the token buckets) and
each part is collated with `model.batch.collate`. Packing needs the vocabulary and no
weights, so it runs in loader workers.
"""

from dataclasses import dataclass

import torch

from model.batch import collate
from model.data import batch_pad_size, microbatches
from model.representation import Observation, observation


@dataclass
class Item:
    """What the policy's microbatch split reads of a sample: its parsed lengths."""

    observation: Observation
    _lengths: tuple
    macro: None = None

    @classmethod
    def of(cls, obs):
        return cls(obs, ((len(obs.tokens), len(obs.slot_refs)),))


def pack(observations, vocabulary, budgets):
    """The parts of one batch of observations: (positions, Batch) each, in the order
    the forward passes run. Every observation is in exactly one part."""
    items = [Item.of(obs) for obs in observations]
    position = {id(item): index for index, item in enumerate(items)}
    return [
        ([position[id(item)] for item in micro],
         collate([item.observation for item in micro], vocabulary, batch_pad_size(micro, budgets)))
        for micro in microbatches(items, budgets)
    ]


def parse(frames):
    return [observation(frame) for frame in frames]


def labels(rows, positions):
    """[rows, 2] targets of the rows at `positions`: run won, act passed."""
    return torch.tensor([[float(rows[i].win), float(rows[i].act_passed)] for i in positions])
