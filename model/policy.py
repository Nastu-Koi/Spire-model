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

    def choose(self, frame, *, teacher=None, sample=True, top_k=0):
        candidates = frame["legal"]["candidates"]
        segment = segment_key(frame)
        if self.segment != segment:
            self.reset()
            self.segment = segment
        state_version = frame["routing"]["state_version"]
        if self.last_state_version is not None and state_version <= self.last_state_version:
            raise ProtocolError("Decoder requires a fresh state/prefix version after each action")
        self.last_state_version = state_version
        refs = {c["candidate_ref"]: c for c in candidates}
        if teacher is not None and teacher not in refs:
            raise ProtocolError("Recorded label is absent from the engine legal set")
        if len(candidates) == 1:
            chosen = candidates[0]
            self.previous_slot = chosen["decoder_slot_ref"]
            if self.encoded and self.previous_slot in self.encoded.observation.slot_refs:
                self.previous_embedding = self.encoded.actions[self.encoded.observation.slot_refs.index(self.previous_slot)]
            else:
                self.previous_embedding = None
            return Choice(chosen["candidate_ref"])
        obs = observation(frame)
        precision = (self.model.device.type, torch.is_autocast_enabled(self.model.device.type),
                     str(torch.get_autocast_dtype(self.model.device.type)), self.model.config.backend)
        if self.precision is not None and self.precision != precision:
            raise ProtocolError("Cannot change numerical backend during a buffered session")
        self.precision = precision
        routing = frame["routing"]
        cache_key = (segment, routing.get("base_public_version"), routing.get("action_bank_version"),
                     obs.digest, self.version, self.vocabulary.digest, precision)
        if self.cache_key != cache_key:
            self.encoded = self.model.encode([obs], self.vocabulary)[0]
            self.cache_key = cache_key
        slots = self.encoded.observation.slot_refs
        legal = {c["decoder_slot_ref"]: c for c in candidates}
        if not set(legal) <= set(slots):
            raise ProtocolError("Unrepresented legal candidate")
        mask = torch.tensor([s in legal for s in slots], dtype=torch.bool, device=self.model.device)
        previous = slots.index(self.previous_slot) if self.previous_slot in slots else self.previous_embedding
        if self.previous_slot is not None and previous is None:
            raise ProtocolError("Previous forced action semantics disappeared from the buffered bank")
        output = self.model.decode(self.encoded, mask, obs.context, self.vocabulary,
                                   hidden=self.hidden, previous=previous, value=not self.started)
        self.hidden, self.started = output.hidden, True
        dist = output.distribution()
        index = (slots.index(refs[teacher]["decoder_slot_ref"]) if teacher is not None else
                 int(dist.sample()) if sample else int(output.logits.argmax()))
        chosen = legal[slots[index]]
        ranking = None
        if top_k:
            indices, _ = output.topk(top_k)
            ranking = [{"candidate_ref": legal[slots[i]]["candidate_ref"], "probability": float(dist.probs[i].detach())}
                       for i in indices.tolist()]
        self.previous_slot = slots[index]
        self.previous_embedding = self.encoded.actions[index]
        return Choice(chosen["candidate_ref"], dist.log_prob(torch.tensor(index, device=self.model.device)),
                      dist.entropy(), output.value, ranking)


def replay(model, vocabulary, steps):
    policy = SessionPolicy(model, vocabulary)
    log_probs, entropies, value = [], [], None
    segment = None
    for step in steps:
        frame = step["frame"]
        if segment is None:
            segment = segment_key(frame)
        elif segment != segment_key(frame):
            raise ProtocolError("Macro crosses a decoder reset/reveal boundary")
        choice = policy.choose(frame, teacher=step["candidate_ref"])
        if choice.log_prob is not None:
            log_probs.append(choice.log_prob)
            entropies.append(choice.entropy)
            if value is None:
                value = choice.value
    if not log_probs:
        raise ProtocolError("All-forced macros are not training samples")
    return torch.stack(log_probs).sum(), value, torch.stack(entropies).sum()
