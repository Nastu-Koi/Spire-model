"""Replay microbatches prepared ahead of the GPU, optionally in worker processes.

Workers are forked, so they share the samples with the trainer without copying
them, and only build CPU tensors: the parent's CUDA state is never touched.
"""

from torch.utils.data import DataLoader, Dataset

from .data import batch_pad_size, sample_lengths
from .policy import prepare_replay

MEASURE_CHUNK = 64


class _Replays(Dataset):
    def __init__(self, micros, vocabulary, config):
        self.micros, self.vocabulary, self.config = micros, vocabulary, config

    def __len__(self):
        return len(self.micros)

    def __getitem__(self, index):
        micro = self.micros[index]
        return prepare_replay(
            self.vocabulary,
            [item.macro["steps"] for item in micro],
            pad_to=batch_pad_size(micro, self.config),
        )


class _Lengths(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return -(-len(self.items) // MEASURE_CHUNK)

    def __getitem__(self, index):
        start = index * MEASURE_CHUNK
        return [sample_lengths(item) for item in self.items[start : start + MEASURE_CHUNK]]


def _same(value):
    return value


def _stream(dataset, workers):
    workers = min(workers, len(dataset))
    if not workers:
        for index in range(len(dataset)):
            yield dataset[index]
        return
    iterator = iter(
        DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=workers,
            collate_fn=_same,
            multiprocessing_context="fork",
        )
    )
    try:
        yield from iterator
    finally:
        # Stop the workers as soon as the consumer leaves, including an early stop.
        del iterator


def measure(items, config):
    """Fill in the token and action counts of samples which have none yet."""
    pending = [item for item in items if item._lengths is None]
    chunks = _stream(_Lengths(pending), config.loader_workers)
    for index, lengths in enumerate(chunks):
        start = index * MEASURE_CHUNK
        for item, value in zip(pending[start : start + MEASURE_CHUNK], lengths):
            item._lengths = value


def replays(micros, vocabulary, config):
    """Yield one prepared replay batch per microbatch, in order."""
    for prepared in _stream(_Replays(micros, vocabulary, config), config.loader_workers):
        vocabulary.report(prepared.batch.unregistered)
        yield prepared
