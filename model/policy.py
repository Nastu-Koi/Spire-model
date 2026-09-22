"""The same session evaluator serves rollout, deployment and differentiable replay."""
from dataclasses import dataclass

import torch

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
        self.hidden = None
        self.previous_slot = None
        self.previous_embedding = None
        self.started = False
        self.precision = None
        self.last_state_version = None

    def choose(self, frame, *, teacher=None, sample=True, top_k=0, pad_to=None):
        return choose_batch([self], [frame], teachers=[teacher], sample=sample,
                            top_k=top_k, pad_to=pad_to)[0]


def choose_batch(policies, frames, *, teachers=None, sample=True, top_k=0, pad_to=None):
    """Choose one action for each independent session with one model batch.

    Forced actions still bypass both model heads. Cached encodings and recurrent
    decoder state remain owned by their SessionPolicy, so batching cannot join
    the state of two selection segments.
    """
    policies, frames = list(policies), list(frames)
    if len(policies) != len(frames):
        raise ValueError("Policies and frames must have the same length")
    if len({id(policy) for policy in policies}) != len(policies):
        raise ValueError("A SessionPolicy may occur only once in a batch")
    if teachers is None:
        teachers = [None] * len(policies)
    else:
        teachers = list(teachers)
        if len(teachers) != len(policies):
            raise ValueError("Teachers and policies must have the same length")
    if not policies:
        return []
    model, vocabulary = policies[0].model, policies[0].vocabulary
    if any(policy.model is not model or policy.vocabulary is not vocabulary for policy in policies[1:]):
        raise ValueError("A choice batch must share one model and vocabulary")

    results = [None] * len(policies)
    pending = []
    observations = []
    cache_keys = []
    metadata = {}
    vocabulary_digest = vocabulary.digest
    for batch_index, (policy, frame, teacher) in enumerate(zip(policies, frames, teachers)):
        candidates = frame["legal"]["candidates"]
        segment = segment_key(frame)
        if policy.segment != segment:
            policy.reset()
            policy.segment = segment
        state_version = frame["routing"]["state_version"]
        if policy.last_state_version is not None and state_version <= policy.last_state_version:
            raise ProtocolError("Decoder requires a fresh state/prefix version after each action")
        policy.last_state_version = state_version
        refs = {candidate["candidate_ref"]: candidate for candidate in candidates}
        if teacher is not None and teacher not in refs:
            raise ProtocolError("Recorded label is absent from the engine legal set")
        if len(candidates) == 1:
            chosen = candidates[0]
            policy.previous_slot = chosen["decoder_slot_ref"]
            if policy.encoded and policy.previous_slot in policy.encoded.observation.slot_refs:
                index = policy.encoded.observation.slot_refs.index(policy.previous_slot)
                policy.previous_embedding = policy.encoded.actions[index]
            else:
                policy.previous_embedding = None
            results[batch_index] = Choice(chosen["candidate_ref"])
            continue

        obs = observation(frame)
        precision = (model.device.type, torch.is_autocast_enabled(model.device.type),
                     str(torch.get_autocast_dtype(model.device.type)), model.config.backend)
        if policy.precision is not None and policy.precision != precision:
            raise ProtocolError("Cannot change numerical backend during a buffered session")
        policy.precision = precision
        routing = frame["routing"]
        cache_key = (segment, routing.get("base_public_version"), routing.get("action_bank_version"),
                     obs.digest, policy.version, vocabulary_digest, precision)
        if policy.cache_key != cache_key:
            pending.append(batch_index)
            observations.append(obs)
            cache_keys.append(cache_key)
        metadata[batch_index] = (obs, candidates, refs, teacher)

    if observations:
        encoded = model.encode(observations, vocabulary, pad_to=pad_to)
        for batch_index, item, cache_key in zip(pending, encoded, cache_keys):
            policies[batch_index].encoded = item
            policies[batch_index].cache_key = cache_key

    active, masks, contexts, hidden, previous, needs_value = [], [], [], [], [], []
    for batch_index in metadata:
        policy = policies[batch_index]
        obs, candidates, _, _ = metadata[batch_index]
        slots = policy.encoded.observation.slot_refs
        legal = {candidate["decoder_slot_ref"]: candidate for candidate in candidates}
        if not set(legal) <= set(slots):
            raise ProtocolError("Unrepresented legal candidate")
        mask = torch.tensor([slot in legal for slot in slots], dtype=torch.bool, device=model.device)
        prior = slots.index(policy.previous_slot) if policy.previous_slot in slots else policy.previous_embedding
        if policy.previous_slot is not None and prior is None:
            raise ProtocolError("Previous forced action semantics disappeared from the buffered bank")
        active.append(batch_index)
        masks.append(mask)
        contexts.append(obs.context)
        hidden.append(policy.hidden)
        previous.append(prior)
        needs_value.append(not policy.started)

    if active:
        outputs = model.decode_batch([policies[i].encoded for i in active], masks, contexts, vocabulary,
                                     hidden=hidden, previous=previous, value=needs_value)
        if len(outputs) != len(active):
            raise RuntimeError("decode_batch returned the wrong number of sessions")
        for batch_index, output in zip(active, outputs):
            policy = policies[batch_index]
            _, candidates, refs, teacher = metadata[batch_index]
            slots = policy.encoded.observation.slot_refs
            legal = {candidate["decoder_slot_ref"]: candidate for candidate in candidates}
            policy.hidden, policy.started = output.hidden, True
            distribution = output.distribution()
            index = (slots.index(refs[teacher]["decoder_slot_ref"]) if teacher is not None else
                     int(distribution.sample()) if sample else int(output.logits.argmax()))
            chosen = legal[slots[index]]
            ranking = None
            if top_k:
                indices, _ = output.topk(top_k)
                ranking = [{"candidate_ref": legal[slots[i]]["candidate_ref"],
                            "probability": float(distribution.probs[i].detach())}
                           for i in indices.tolist()]
            policy.previous_slot = slots[index]
            policy.previous_embedding = policy.encoded.actions[index]
            index_tensor = torch.tensor(index, device=model.device)
            results[batch_index] = Choice(chosen["candidate_ref"], distribution.log_prob(index_tensor),
                                          distribution.entropy(), output.value, ranking)
    return results


def replay(model, vocabulary, steps):
    return replay_batch(model, vocabulary, [steps])[0]


def replay_batch(model, vocabulary, list_of_steps, *, pad_to=None):
    """Differentiably replay independent macro-actions in lockstep."""
    sequences = [list(steps) for steps in list_of_steps]
    policies = [SessionPolicy(model, vocabulary) for _ in sequences]
    log_probs, entropies = [[] for _ in sequences], [[] for _ in sequences]
    values = [None] * len(sequences)
    segments = [None] * len(sequences)
    for offset in range(max(map(len, sequences), default=0)):
        active = [i for i, steps in enumerate(sequences) if offset < len(steps)]
        frames, teachers = [], []
        for i in active:
            step = sequences[i][offset]
            frame = step["frame"]
            current = segment_key(frame)
            if segments[i] is None:
                segments[i] = current
            elif segments[i] != current:
                raise ProtocolError("Macro crosses a decoder reset/reveal boundary")
            frames.append(frame)
            teachers.append(step["candidate_ref"])
        choices = choose_batch([policies[i] for i in active], frames, teachers=teachers, pad_to=pad_to)
        for i, choice in zip(active, choices):
            if choice.log_prob is not None:
                log_probs[i].append(choice.log_prob)
                entropies[i].append(choice.entropy)
                if values[i] is None:
                    values[i] = choice.value
    result = []
    for probabilities, value, entropy in zip(log_probs, values, entropies):
        if not probabilities:
            raise ProtocolError("All-forced macros are not training samples")
        result.append((torch.stack(probabilities).sum(), value, torch.stack(entropy).sum()))
    return result
