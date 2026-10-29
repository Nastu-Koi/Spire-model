"""The same stateless evaluator serves rollout, deployment and differentiable replay.

Every decision is scored from its own observation;
multi-select progress is part of that observation, so sessions only cache the
encoding of an unchanged frame and enforce fresh state versions.
"""

import math
from dataclasses import dataclass

import torch

from main.encoding.batch import Batch, Packed, collate
from main.env.protocol import ProtocolError, segment_key
from main.env.control import is_policy_frame
from main.encoding.representation import observation


@dataclass
class Choice:
    candidate_ref: str
    log_prob: torch.Tensor | None = None  # at the temperature it was drawn at
    entropy: torch.Tensor | None = None
    ranking: list[dict] | None = None
    base_log_prob: torch.Tensor | float | None = None  # at temperature 1
    explored: bool = False  # drawn among the candidates the caller restricted it to, not from the policy
    explored_by: str | None = None  # who proposed an explored candidate: "prior", or None for the policy's own rule
    forked: bool = False  # the candidate of the rank the caller named, not a draw from the policy
    # Every legal candidate's log-probability at the temperature it was drawn at, in the
    # frame's order: the distribution it was drawn from (or would have been, explored or forked).
    log_probs: list[float] | None = None


class SessionPolicy:
    def __init__(self, model, vocabulary, *, version=0):
        self.model, self.vocabulary, self.version = model, vocabulary, version
        self.reset()

    def reset(self):
        self.segment = None
        self.cache_key = None
        self.encoded = None
        self.precision = None
        self.last_state_version = None

    def advance(self, frame):
        """Enter the next decision of the session; returns its legal candidates."""
        segment = segment_key(frame)
        if self.segment != segment:
            self.reset()
            self.segment = segment
        state_version = frame["routing"]["state_version"]
        if self.last_state_version is not None and state_version <= self.last_state_version:
            raise ProtocolError("Decisions require a fresh state/prefix version after each action")
        self.last_state_version = state_version
        return frame["legal"]["candidates"]

    def choose(self, frame, *, teacher=None, sample=True, top_k=0, pad_to=None, temperature=1.0, restrict=None,
               rank=None):
        return choose_batch(
            [self],
            [frame],
            teachers=[teacher],
            sample=sample,
            top_k=top_k,
            pad_to=pad_to,
            temperature=temperature,
            restrict=[restrict],
            rank=[rank],
        )[0]


def _precision(model):
    return (
        model.device.type,
        torch.is_autocast_enabled(model.device.type),
        str(torch.get_autocast_dtype(model.device.type)),
        model.config.backend,
        model.config.encoder_precision,
    )


def _mask(model, obs, candidates):
    slots = obs.slot_refs
    legal = {candidate["decoder_slot_ref"]: candidate for candidate in candidates}
    if not set(legal) <= set(slots):
        raise ProtocolError("Unrepresented legal candidate")
    return legal, torch.tensor([slot in legal for slot in slots], dtype=torch.bool, device=model.device)


def choose_batch(policies, frames, *, teachers=None, sample=True, top_k=0, pad_to=None,
                 temperature=1.0, restrict=None, rank=None):
    """Choose one action for each independent session with one model batch.

    Forced actions bypass the model. Probabilities are those of the scores
    divided by `temperature`. `restrict`, one entry per frame, names the candidates
    an exploring decision is drawn among (at their relative odds under the
    policy) instead of from the whole distribution; None draws from it. `rank`, one
    entry per frame, makes a forking decision take the candidate of that place in the
    policy's order (0 the most likely), with no draw. The choice keeps the policy's
    own log-probability either way: what is explored is the action, not what the
    policy is held to have said.
    """
    policies, frames = list(policies), list(frames)
    if len(policies) != len(frames):
        raise ValueError("Policies and frames must have the same length")
    if len({id(policy) for policy in policies}) != len(policies):
        raise ValueError("A SessionPolicy may occur only once in a batch")
    teachers = [None] * len(policies) if teachers is None else list(teachers)
    if len(teachers) != len(policies):
        raise ValueError("Teachers and policies must have the same length")
    restrict = [None] * len(policies) if restrict is None else list(restrict)
    if len(restrict) != len(policies):
        raise ValueError("Restrictions and policies must have the same length")
    rank = [None] * len(policies) if rank is None else list(rank)
    if len(rank) != len(policies):
        raise ValueError("Ranks and policies must have the same length")
    if not policies:
        return []
    model, vocabulary = policies[0].model, policies[0].vocabulary
    if any(policy.model is not model or policy.vocabulary is not vocabulary for policy in policies[1:]):
        raise ValueError("A choice batch must share one model and vocabulary")

    results = [None] * len(policies)
    pending, observations, cache_keys, metadata = [], [], [], {}
    precision = _precision(model)
    for batch_index, (policy, frame, teacher) in enumerate(zip(policies, frames, teachers)):
        candidates = policy.advance(frame)
        segment = policy.segment
        refs = {candidate["candidate_ref"]: candidate for candidate in candidates}
        if teacher is not None and teacher not in refs:
            raise ProtocolError("Recorded label is absent from the engine legal set")
        if restrict[batch_index] is not None and not set(restrict[batch_index]) <= set(refs):
            raise ProtocolError("A restriction names candidates the engine did not offer")
        _check_rank(rank[batch_index], len(candidates), restrict[batch_index])
        if len(candidates) == 1:
            results[batch_index] = Choice(candidates[0]["candidate_ref"])
            continue
        if policy.precision is not None and policy.precision != precision:
            raise ProtocolError("Cannot change numerical backend during a session")
        policy.precision = precision
        obs = observation(frame)
        cache_key = (segment, obs.digest, policy.version, vocabulary.digest, precision)
        if policy.cache_key != cache_key:
            pending.append(batch_index)
            observations.append(obs)
            cache_keys.append(cache_key)
        metadata[batch_index] = (candidates, teacher, restrict[batch_index], rank[batch_index])

    if observations:
        encoded = model.encode(observations, vocabulary, pad_to=pad_to)
        for batch_index, item, cache_key in zip(pending, encoded, cache_keys):
            policies[batch_index].encoded = item
            policies[batch_index].cache_key = cache_key

    active = list(metadata)
    if not active:
        return results
    legal_sets, masks = [], []
    for batch_index in active:
        legal, mask = _mask(model, policies[batch_index].encoded.observation, metadata[batch_index][0])
        legal_sets.append(legal)
        masks.append(mask)
    outputs = model.decode_batch([policies[i].encoded for i in active], masks)
    for batch_index, legal, output in zip(active, legal_sets, outputs):
        policy = policies[batch_index]
        candidates, teacher, allowed, place = metadata[batch_index]
        slots = policy.encoded.observation.slot_refs
        refs = {candidate["candidate_ref"]: candidate for candidate in candidates}
        distribution = output.distribution(temperature)
        if teacher is not None:
            index = slots.index(refs[teacher]["decoder_slot_ref"])
        elif not sample:
            index = int(output.logits.argmax())
        elif place is not None:
            index = int(output.logits.argsort(descending=True)[place])
        elif allowed is None:
            index = int(distribution.sample())
        else:
            among = {refs[ref]["decoder_slot_ref"] for ref in allowed}
            keep = torch.tensor([slot in among for slot in slots], device=distribution.probs.device)
            index = int(torch.multinomial(distribution.probs.masked_fill(~keep, 0), 1))
        ranking = None
        if top_k:
            indices, _ = output.topk(top_k)
            ranking = [
                {"candidate_ref": legal[slots[i]]["candidate_ref"], "probability": float(distribution.probs[i].detach())}
                for i in indices.tolist()
            ]
        index_tensor = torch.tensor(index, device=model.device)
        drawn = None
        if sample:
            # Each candidate's log-probability at the temperature, in the frame's order,
            # as a sampling process's choice carries it (`main/play/sampling.py`).
            logs = distribution.logits.detach().log_softmax(-1)
            drawn = [float(logs[slots.index(candidate["decoder_slot_ref"])]) for candidate in candidates]
        results[batch_index] = Choice(
            legal[slots[index]]["candidate_ref"],
            distribution.log_prob(index_tensor),
            distribution.entropy(),
            ranking,
            output.distribution(1.0).log_prob(index_tensor),
            explored=sample and allowed is not None,
            forked=sample and place is not None and teacher is None,
            log_probs=drawn,
        )
    return results


def _check_rank(rank, candidates, restrict):
    """A forking decision's rank names one of its candidates, and nothing restricts it."""
    if rank is None:
        return
    if type(rank) is not int or not 0 <= rank < candidates:
        raise ProtocolError("A fork's rank names no candidate of the decision")
    if restrict is not None:
        raise ProtocolError("A decision forks by rank or explores a restriction, not both")


@dataclass
class Decisions(Packed):
    """Independent decisions packed for one sampling pass; built without the model."""

    batch: Batch
    masks: torch.Tensor  # [decisions, actions] legal slots
    explore: torch.Tensor  # [decisions, actions] the slots an exploring decision is drawn among; no row set otherwise
    # [decisions] the place, in the policy's order of its legal slots, of the slot a
    # forking decision takes (0 the most likely); -1 for a decision that is drawn.
    rank: torch.Tensor | None = None

    def __post_init__(self):
        if self.rank is None:
            self.rank = torch.full((len(self.masks),), -1, dtype=torch.long)


def prepare_decisions(vocabulary, frames, restrict=None, rank=None, *, observe=observation, tensors=True):
    """Pack one decision of each frame; returns their observations beside the pack.
    Every frame must offer a choice: a forced action has nothing to score. `restrict`,
    one entry per frame, names the candidates an exploring decision is drawn among
    (`choose_batch`); None for a decision drawn from the whole distribution. `rank`,
    one entry per frame, the place of the candidate a forking decision takes.
    `observe` parses a frame, as `prepare_replay`'s does. Without `tensors` the batch's
    fields are left as rows (`collate`) for a sampling process's `wire.pack`."""
    restrict = [None] * len(frames) if restrict is None else list(restrict)
    rank = [None] * len(frames) if rank is None else list(rank)
    observations, masks, explore = [], [], []
    for frame, allowed, place in zip(frames, restrict, rank):
        obs = observe(frame)
        candidates = {candidate["candidate_ref"]: candidate["decoder_slot_ref"] for candidate in frame["legal"]["candidates"]}
        legal = set(candidates.values())
        if not legal <= set(obs.slot_refs):
            raise ProtocolError("Unrepresented legal candidate")
        if allowed is not None and not set(allowed) <= set(candidates):
            raise ProtocolError("A restriction names candidates the engine did not offer")
        _check_rank(place, len(legal), allowed)
        among = set() if allowed is None else {candidates[ref] for ref in allowed}
        observations.append(obs)
        masks.append([slot in legal for slot in obs.slot_refs])
        explore.append([slot in among for slot in obs.slot_refs])
    width = max(map(len, masks))
    return observations, Decisions(
        collate(observations, vocabulary, tensors=tensors),
        torch.tensor([mask + [False] * (width - len(mask)) for mask in masks]),
        torch.tensor([mask + [False] * (width - len(mask)) for mask in explore]),
        torch.tensor([-1 if place is None else place for place in rank], dtype=torch.long),
    )


@dataclass
class Replay(Packed):
    """Macro-actions packed for one differentiable pass; built without the model."""

    batch: Batch
    masks: torch.Tensor  # [decisions, actions] legal slots
    labels: torch.Tensor  # [decisions] chosen slot
    owners: torch.Tensor  # [decisions] macro-action of each decision
    first: torch.Tensor  # [macros] first decision of each macro-action
    # [decisions, actions] the log-probabilities each decision was drawn with, −inf off
    # its legal slots, where every decision kept them (`Choice.log_probs`); else None.
    reference: torch.Tensor | None = None


def prepare_replay(vocabulary, list_of_steps, *, pad_to=None, observe=observation):
    """Pack policy decisions; forced and controller steps carry no label.
    `observe` parses a frame: a caller that has parsed the frames passes its results.
    Where every decision kept the distribution it was drawn from, it is packed too."""
    owners, observations, masks, labels, first, drawn = [], [], [], [], [], []
    for macro, steps in enumerate(list_of_steps):
        segment = None
        for step in steps:
            frame = step["frame"]
            current = segment_key(frame)
            if segment is None:
                segment = current
            elif segment != current:
                raise ProtocolError("Macro crosses a decoder reset/reveal boundary")
            candidates = frame["legal"]["candidates"]
            refs = {candidate["candidate_ref"]: candidate for candidate in candidates}
            if step["candidate_ref"] not in refs:
                raise ProtocolError("Recorded label is absent from the engine legal set")
            if not is_policy_frame(frame):
                continue
            obs = observe(frame)
            legal = {candidate["decoder_slot_ref"] for candidate in candidates}
            if not legal <= set(obs.slot_refs):
                raise ProtocolError("Unrepresented legal candidate")
            if len(first) == macro:
                first.append(len(observations))
            owners.append(macro)
            observations.append(obs)
            masks.append([slot in legal for slot in obs.slot_refs])
            labels.append(obs.slot_refs.index(refs[step["candidate_ref"]]["decoder_slot_ref"]))
            if drawn is not None and "log_probs" in step:
                row = [-math.inf] * len(obs.slot_refs)
                for candidate, log_prob in zip(candidates, step["log_probs"], strict=True):
                    row[obs.slot_refs.index(candidate["decoder_slot_ref"])] = log_prob
                drawn.append(row)
            else:
                drawn = None
        if len(first) == macro:
            raise ProtocolError("Macros without policy decisions are not training samples")
    width = max(map(len, masks))
    return Replay(
        collate(observations, vocabulary, pad_to),
        torch.tensor([mask + [False] * (width - len(mask)) for mask in masks]),
        torch.tensor(labels),
        torch.tensor(owners),
        torch.tensor(first),
        None if drawn is None else torch.tensor([row + [-math.inf] * (width - len(row)) for row in drawn]),
    )


def replay_batch(model, vocabulary, list_of_steps, *, pad_to=None):
    """Differentiably replay macro-actions; all decision steps form one batch.

    Returns (summed log-prob, summed entropy) per macro, the log-probability
    being the one the macro was sampled with at temperature 1.
    """
    prepared = prepare_replay(vocabulary, list_of_steps, pad_to=pad_to)
    return list(zip(*(x.unbind(0) for x in model.replay(prepared.to(model.device)))))


def replay(model, vocabulary, steps):
    return replay_batch(model, vocabulary, [steps])[0]
