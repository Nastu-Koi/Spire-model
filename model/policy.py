"""The same stateless evaluator serves rollout, deployment and differentiable replay.

Every decision is scored from its own observation;
multi-select progress is part of that observation, so sessions only cache the
encoding of an unchanged frame and enforce fresh state versions.
"""

from dataclasses import dataclass

import torch

from .batch import Batch, Packed, collate
from .protocol import ProtocolError, segment_key
from .representation import observation


@dataclass
class Choice:
    candidate_ref: str
    log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    value: torch.Tensor | None = None
    ranking: list[dict] | None = None


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

    def choose(self, frame, *, teacher=None, sample=True, top_k=0, pad_to=None):
        return choose_batch(
            [self],
            [frame],
            teachers=[teacher],
            sample=sample,
            top_k=top_k,
            pad_to=pad_to,
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


def choose_batch(policies, frames, *, teachers=None, sample=True, top_k=0, pad_to=None):
    """Choose one action for each independent session with one model batch.

    Forced actions bypass both model heads.
    """
    policies, frames = list(policies), list(frames)
    if len(policies) != len(frames):
        raise ValueError("Policies and frames must have the same length")
    if len({id(policy) for policy in policies}) != len(policies):
        raise ValueError("A SessionPolicy may occur only once in a batch")
    teachers = [None] * len(policies) if teachers is None else list(teachers)
    if len(teachers) != len(policies):
        raise ValueError("Teachers and policies must have the same length")
    if not policies:
        return []
    model, vocabulary = policies[0].model, policies[0].vocabulary
    if any(policy.model is not model or policy.vocabulary is not vocabulary for policy in policies[1:]):
        raise ValueError("A choice batch must share one model and vocabulary")

    results = [None] * len(policies)
    pending, observations, cache_keys, metadata = [], [], [], {}
    precision = _precision(model)
    for batch_index, (policy, frame, teacher) in enumerate(zip(policies, frames, teachers)):
        candidates = frame["legal"]["candidates"]
        segment = segment_key(frame)
        if policy.segment != segment:
            policy.reset()
            policy.segment = segment
        state_version = frame["routing"]["state_version"]
        if policy.last_state_version is not None and state_version <= policy.last_state_version:
            raise ProtocolError("Decisions require a fresh state/prefix version after each action")
        policy.last_state_version = state_version
        refs = {candidate["candidate_ref"]: candidate for candidate in candidates}
        if teacher is not None and teacher not in refs:
            raise ProtocolError("Recorded label is absent from the engine legal set")
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
        metadata[batch_index] = (candidates, teacher)

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
        candidates, teacher = metadata[batch_index]
        slots = policy.encoded.observation.slot_refs
        refs = {candidate["candidate_ref"]: candidate for candidate in candidates}
        distribution = output.distribution()
        index = (
            slots.index(refs[teacher]["decoder_slot_ref"])
            if teacher is not None
            else int(distribution.sample())
            if sample
            else int(output.logits.argmax())
        )
        ranking = None
        if top_k:
            indices, _ = output.topk(top_k)
            ranking = [
                {"candidate_ref": legal[slots[i]]["candidate_ref"], "probability": float(distribution.probs[i].detach())}
                for i in indices.tolist()
            ]
        index_tensor = torch.tensor(index, device=model.device)
        results[batch_index] = Choice(
            legal[slots[index]]["candidate_ref"],
            distribution.log_prob(index_tensor),
            distribution.entropy(),
            output.value,
            ranking,
        )
    return results


@dataclass
class Replay(Packed):
    """Macro-actions packed for one differentiable pass; built without the model."""

    batch: Batch
    masks: torch.Tensor  # [decisions, actions] legal slots
    labels: torch.Tensor  # [decisions] chosen slot
    owners: torch.Tensor  # [decisions] macro-action of each decision
    first: torch.Tensor  # [macros] first decision of each macro-action


def prepare_replay(vocabulary, list_of_steps, *, pad_to=None):
    """Pack the branching decisions of macro-actions; forced steps carry no label."""
    owners, observations, masks, labels, first = [], [], [], [], []
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
            if len(candidates) == 1:
                continue
            obs = observation(frame)
            legal = {candidate["decoder_slot_ref"] for candidate in candidates}
            if not legal <= set(obs.slot_refs):
                raise ProtocolError("Unrepresented legal candidate")
            if len(first) == macro:
                first.append(len(observations))
            owners.append(macro)
            observations.append(obs)
            masks.append([slot in legal for slot in obs.slot_refs])
            labels.append(obs.slot_refs.index(refs[step["candidate_ref"]]["decoder_slot_ref"]))
        if len(first) == macro:
            raise ProtocolError("All-forced macros are not training samples")
    width = max(map(len, masks))
    return Replay(
        collate(observations, vocabulary, pad_to),
        torch.tensor([mask + [False] * (width - len(mask)) for mask in masks]),
        torch.tensor(labels),
        torch.tensor(owners),
        torch.tensor(first),
    )


def replay_batch(model, vocabulary, list_of_steps, *, pad_to=None):
    """Differentiably replay macro-actions; all decision steps form one batch.

    Returns (summed log-prob, value at the first decision step, summed entropy)
    per macro, matching the semantics used when the macro was sampled.
    """
    prepared = prepare_replay(vocabulary, list_of_steps, pad_to=pad_to)
    return list(zip(*(x.unbind(0) for x in model.replay(prepared.to(model.device)))))


def replay(model, vocabulary, steps):
    return replay_batch(model, vocabulary, [steps])[0]
