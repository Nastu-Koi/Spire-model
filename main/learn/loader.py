"""Replay batches prepared ahead of the GPU, in worker processes.

Workers are forked, so they share the samples with the process that holds them
without copying, and only build CPU tensors: the trainer's CUDA state is never
touched. A worker prepares one logical batch at a time: it parses every frame
once, splits the batch under the budgets and packs each part.

Feeder processes read the sharded windows, each with workers of its own, and
hand over packed batches: while one feeder decodes its next shards, the workers of
the others pack. Only their bounded final remainders enter the trainer.

A feeder forks its next workers once per window, and a thread's lock copied
mid-operation into a child can never be released there, so a feeder holds no
thread when it forks. Packed tensors therefore cross process boundaries as named
shared-memory files (torch's `file_system` strategy): the default descriptor
passing serves every descriptor from a background thread of the sending process.
Each feeder writes its messages to a pipe of its own, read by the trainer; a
`multiprocessing.Queue` would write them from a thread too.

A named shared file is unlinked when every process that holds its tensor has let go.
A producer that is ended (a pass that stops by its count of updates, as a round's does,
ends the feeders with one or two prefetched results per packer) never lets go: the
files of what it still held stay until this process exits, some gigabytes over a run,
whether nobody received them or the trainer did and has since let go. The consumer
that ended the producers removes their files once it holds none of them itself
(`_release_shared`): at the end of the stream, and again when the next one starts.
"""

import gc
import os
import random
import sys
import traceback
from dataclasses import replace
from multiprocessing.connection import wait

import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader, Dataset, Subset

from main.data.shards import batch_pad_size, microbatches, sample_lengths, samples
from main.network.policy import prepare_replay
from main.env.processes import end, end_with
from main.encoding.representation import observation

def pack(logical, start, vocabulary, config):
    """The parts of one logical batch, each as (positions of its samples, Replay)."""
    seen = {}

    def observe(frame):
        # Lengths and tensors come from the same parse of a frame.
        if id(frame) not in seen:
            seen[id(frame)] = observation(frame)
        return seen[id(frame)]

    position = {id(item): start + index for index, item in enumerate(logical)}
    for item in logical:
        if item._lengths is None:
            item._lengths = sample_lengths(item, observe)
    return [
        (
            [position[id(item)] for item in micro],
            prepare_replay(
                vocabulary,
                [item.macro["steps"] for item in micro],
                pad_to=batch_pad_size(micro, config),
                observe=observe,
            ),
        )
        for micro in microbatches(logical, config)
    ]


class _Logicals(Dataset):
    def __init__(self, items, vocabulary, config):
        self.items, self.vocabulary, self.config = items, vocabulary, config

    def __len__(self):
        return -(-len(self.items) // self.config.logical_batch_size)

    def __getitem__(self, index):
        start = index * self.config.logical_batch_size
        logical = self.items[start : start + self.config.logical_batch_size]
        return pack(logical, start, self.vocabulary, self.config)


def _share_by_name():
    """Tensors sent to another process are shared as named files, not descriptors."""
    if multiprocessing.get_sharing_strategy() != "file_system":
        multiprocessing.set_sharing_strategy("file_system")


def _same(value):
    return value


SHARED = "/dev/shm"
# Producers this process ended, whose files it may still hold tensors of.
_ENDED = set()


def _processes(members):
    """The pids of the processes `members` keeps: those whose process group a pid of
    `leaders` leads (a feeder's packers are in its group), or this process's children."""
    found = set()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as stream:
                stat = stream.read()
        except OSError:
            continue
        # After the command name in parentheses: state, parent, process group, session.
        fields = stat[stat.rindex(")") + 2:].split()
        if members(int(fields[1]), int(fields[2])):
            found.add(int(entry))
    return found


def _group_members(leaders):
    leaders = set(leaders)
    return leaders | _processes(lambda parent, group: group in leaders)


def _children():
    me = os.getpid()
    return _processes(lambda parent, group: parent == me)


def _release_shared(pids=()):
    """Unlink the shared-memory files of tensors created by `pids` and by every producer
    ended before, except those this process has mapped. An ended producer never lets go
    of what it held, so the sharing will not unlink these files on its own
    (`file_system` sharing names a file after the process that created it); a file this
    process still holds a tensor of must wait, as the sharing aborts when it finds the
    name gone at the tensor's release. Returns how many files were removed."""
    _ENDED.update(str(pid) for pid in pids)
    try:
        names = os.listdir(SHARED)
        with open("/proc/self/maps") as stream:
            mapped = {line.split()[-1] for line in stream if SHARED in line}
    except OSError:
        return 0
    removed = 0
    for name in names:
        parts = name.split("_")
        path = os.path.join(SHARED, name)
        if len(parts) >= 3 and parts[0] == "torch" and parts[1] in _ENDED and path not in mapped:
            try:
                os.unlink(path)
                removed += 1
            except OSError:
                pass
    return removed


# A logical batch packs in seconds. The workers of a window have been seen to sit idle
# with a result still owed and nothing left in any pipe: waiting longer than this for
# the next batch means it will not come.
STALL_SECONDS = 180


def _stream(dataset, workers, release=False, report=None):
    """The dataset's items in order, packed by `workers` processes. With `release` the
    consumer is this process: the shared files of results left unread when the
    workers end are removed. A feeder streams to the trainer and must not: a result
    it has sent on may still be in the pipe; it tells the trainer its packers' pids
    through `report` instead, for the trainer to release what they left."""
    workers = min(workers, len(dataset))
    if not workers:
        for index in range(len(dataset)):
            yield dataset[index]
        return
    _share_by_name()
    if release:
        _release_shared()
    # A worker shares the samples only while it writes to none of them, and a garbage
    # collection writes to every object it visits: what exists now is set aside from
    # the collector, here and in the workers forked next. Nothing is collected first:
    # walking everything a window or a round holds takes over a second each time, and
    # what it would free is not what a worker reads.
    gc.freeze()
    done = 0
    try:
        while done < len(dataset):
            # Batches arrive in order, so the ones still owed are the rest of the range.
            iterator = iter(DataLoader(
                Subset(dataset, range(done, len(dataset))),
                batch_size=None,
                shuffle=False,
                num_workers=min(workers, len(dataset) - done),
                collate_fn=_same,
                multiprocessing_context="fork",
                timeout=STALL_SECONDS,
            ))
            try:
                if report is not None:
                    report(sorted(_children()))
                for value in iterator:
                    done += 1
                    yield value
            except RuntimeError as error:
                if "timed out after" not in str(error):
                    raise
                print(f"Loader workers stalled at batch {done} of {len(dataset)}; packing the rest again",
                      file=sys.stderr, flush=True)
            finally:
                # Stop the workers as soon as the consumer leaves, including an early stop.
                packers = _children() if release else ()
                del iterator
                _release_shared(packers)
    finally:
        gc.unfreeze()


def batches(items, vocabulary, config):
    """The logical batches of `items`, in order: each a list of (samples, Replay), one
    entry per part the budgets split it into."""
    for parts in _stream(_Logicals(items, vocabulary, config), config.loader_workers, release=True):
        for _, prepared in parts:
            vocabulary.report(prepared.batch.unregistered)
        yield [([items[index] for index in positions], prepared) for positions, prepared in parts]


# A feeder sends something every few seconds while it packs. Silence this long is
# checked against the feeder processes being alive.
FEEDER_SILENCE_SECONDS = 60


def _send(connection, message):
    try:
        connection.send(message)
    except OSError:
        # The trainer has gone: there is nobody left to tell.
        pass


# Niceness the feeders (and the workers they fork) add to the trainer's.
FEEDER_NICE = 5


def _feed(connection, shards, windows, weights, vocabulary, config, shuffle, seed):
    """Feeder process: the packed logical batches of its windows, then its number,
    written to its own pipe. It holds no thread when it forks its workers; what it
    sent stays mapped for the trainer by the shared files' own reference counts."""
    feeder, windows = windows
    try:
        # The feeder leads a process group with its workers, so that the trainer can
        # end them together when it stops early; a worker whose feeder is gone would
        # otherwise wait forever to finish writing a result to it.
        trainer = os.getppid()
        os.setsid()
        end_with(trainer)
        # The parent's intra-op pool does not survive the fork; packing needs none.
        torch.set_num_threads(1)
        # The trainer's thread launches the kernels of every step; the feeders and their
        # workers yield to it, or dozens of them compete with it for the cores.
        os.nice(FEEDER_NICE)
        random.seed(seed)
        carry = []
        for names in windows:
            items = carry + samples(shards.read(names), weights=weights)
            if shuffle:
                random.shuffle(items)
            boundary = len(items) // config.logical_batch_size * config.logical_batch_size
            carry, items = items[boundary:], items[:boundary]
            for parts in _stream(_Logicals(items, vocabulary, config), config.loader_workers,
                                 report=lambda pids: connection.send(("packers", pids))):
                # What the trainer reads of a sample fits in a few fields; its frames stay here.
                connection.send(("batch", [
                    ([replace(items[index], macro=None, _lengths=None) for index in positions], prepared)
                    for positions, prepared in parts
                ]))
            del items
        # At most B-1 raw samples per feeder. The parent combines these once,
        # after all complete batches, so the epoch has only one short update.
        if carry:
            connection.send(("tail", (feeder, carry)))
    except BaseException:
        _send(connection, ("error", traceback.format_exc()))
    finally:
        _send(connection, ("done", feeder))
        connection.close()


def stream(shards, vocabulary, config, *, window_shards, shuffle, weights=None):
    """The logical batches of sharded runs, as `batches` yields them, with samples that
    carry no frames. `window_shards` shards are in memory at a time. With workers
    they are split between `loader_feeders` feeders: the samples of a feeder's shards
    are shuffled together when `shuffle`, and the batches of the feeders interleave.
    Sample weights are the whole data set's (`RunShards.weights`), not the window's."""
    if type(window_shards) is not int or window_shards < 1:
        raise ValueError("Window shard count must be positive")
    weights = shards.weights if weights is None else weights
    if not config.loader_workers:
        carry = []
        for names in shards.window_names(window_shards, shuffle):
            items = carry + samples(shards.read(names), weights=weights)
            if shuffle:
                random.shuffle(items)
            boundary = len(items) // config.logical_batch_size * config.logical_batch_size
            carry, items = items[boundary:], items[:boundary]
            yield from batches(items, vocabulary, config)
        yield from batches(carry, vocabulary, config)
        return
    _share_by_name()
    _release_shared()
    context = multiprocessing.get_context("fork")
    windows = shards.window_names(max(1, window_shards // config.loader_feeders), shuffle)
    count = min(config.loader_feeders, len(windows))
    feeders, receivers = [], []
    tails = {}
    # Every process that packed for this stream: a packer that has exited may still have
    # a result in flight, and only its pid names the result's files.
    producers = set()
    try:
        for index in range(count):
            # One pipe per feeder: a write blocks until the trainer reads, which bounds
            # what is in flight to the workers' prefetch behind each feeder.
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=_feed, args=(
                sender, shards, (index, windows[index::count]), weights, vocabulary, config,
                shuffle, random.getrandbits(64)))
            process.start()
            sender.close()
            feeders.append(process)
            receivers.append(receiver)
        active = dict(enumerate(receivers))
        while active:
            ready = wait(list(active.values()), timeout=FEEDER_SILENCE_SECONDS)
            if not ready:
                # A feeder that died without its last word would be waited for forever.
                dead = [index for index in active if not feeders[index].is_alive()]
                if dead:
                    raise RuntimeError(f"Loader feeder {dead[0]} exited without finishing")
                continue
            for receiver in ready:
                index = next(i for i, c in active.items() if c is receiver)
                try:
                    kind, value = receiver.recv()
                except EOFError:
                    raise RuntimeError(f"Loader feeder {index} exited without finishing") from None
                if kind == "error":
                    raise RuntimeError("Loader feeder failed:\n" + value)
                if kind == "done":
                    del active[index]
                    receiver.close()
                    continue
                if kind == "tail":
                    feeder, items = value
                    tails[feeder] = items
                    continue
                if kind == "packers":
                    producers.update(value)
                    continue
                for _, prepared in value:
                    vocabulary.report(prepared.batch.unregistered)
                yield value
        remainder = [item for index in sorted(tails) for item in tails[index]]
        # Pack this bounded remainder locally, then keep the same sample-summary
        # contract as complete batches from the feeders.
        for logical in batches(remainder, vocabulary, replace(config, loader_workers=0)):
            yield [([replace(item, macro=None, _lengths=None) for item in items], prepared)
                   for items, prepared in logical]
    finally:
        for receiver in receivers:
            receiver.close()
        # The feeders and their packers go together; what they still had in flight is
        # released with them.
        producers |= _group_members(process.pid for process in feeders)
        end(feeders)
        _release_shared(producers)
