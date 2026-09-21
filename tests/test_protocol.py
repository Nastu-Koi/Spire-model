from copy import deepcopy

import pytest
import torch

from model.protocol import ProtocolError, execution_command, validate_frame
from model.representation import clean_public, observation
from model.policy import SessionPolicy
from model.testing import SyntheticEngine


def test_stale_and_nonlegal_commands_do_not_mutate():
    engine = SyntheticEngine(5, 3)
    frame = engine.reset()
    command = execution_command(frame, "c0")
    engine.send(command)
    selected = list(engine.selected)
    with pytest.raises(ProtocolError):
        engine.send(command)
    assert engine.selected == selected
    with pytest.raises(ProtocolError, match="non-legal"):
        execution_command(frame, "bogus")


def test_contract_mask_validation(demo):
    frame = deepcopy(demo["macros"][0]["steps"][0]["frame"])
    validate_frame(frame)
    frame["contract"]["training_ready"] = False
    with pytest.raises(ProtocolError, match="training_ready"):
        validate_frame(frame)
    validate_frame(frame, allow_prototype=True)
    frame["contract"]["training_ready"] = True
    frame["public"]["decoder_bank"] = frame["public"]["decoder_bank"][:1]
    with pytest.raises(ProtocolError, match="cover"):
        validate_frame(frame)


def test_candidate_semantics_must_match_bank(demo):
    frame = deepcopy(demo["macros"][0]["steps"][0]["frame"])
    frame["legal"]["candidates"][0]["operation"] = "upgrade"
    with pytest.raises(ProtocolError, match="semantics"):
        validate_frame(frame)


def test_hidden_fields_and_handle_renumbering_do_not_change_policy(setup, demo):
    model, vocab = setup
    frame = demo["macros"][0]["steps"][0]["frame"]
    altered = deepcopy(frame)
    altered["public"]["seed"] = "secret"
    altered["public"]["rng"] = 12
    altered["public"]["future_reward"] = {"card": "ANSWER"}
    for i, entity in enumerate(altered["public"]["entities"]):
        entity["instance_id"] = 900 + i
        entity["saved_properties"] = {"hidden": 1}
        entity["stats"] = dict(entity.get("stats", {}), secret_future=991)
    assert observation(frame).tokens == observation(altered).tokens
    refs = {e["ref"]: f"arbitrary-{i * 197}" for i, e in enumerate(altered["public"]["entities"])}

    def rename(value, key=""):
        if isinstance(value, dict):
            return {k: rename(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(x, key) for x in value]
        reference = key.endswith("_ref") or key.endswith("_refs") or key in {"ref", "source", "target"}
        return refs.get(value, value) if reference and isinstance(value, str) else value

    altered = rename(altered)
    with torch.no_grad():
        a = SessionPolicy(model, vocab).choose(frame, sample=False, top_k=20)
        b = SessionPolicy(model, vocab).choose(altered, sample=False, top_k=20)
    assert a.candidate_ref == b.candidate_ref
    for x, y in zip(a.ranking, b.ranking):
        assert x["probability"] == pytest.approx(y["probability"], abs=1e-6)


def test_piles_preserve_duplicates_but_not_packing_order(demo):
    frame = deepcopy(demo["macros"][0]["steps"][0]["frame"])
    for zone in ("draw_pile", "discard_pile", "exhaust_pile"):
        frame["public"]["entities"] += [{"ref": zone + str(i), "entity_type": "card", "zone": zone, "content_id": "STRIKE"} for i in range(3)]
    obs = observation(frame)
    assert len(obs.tokens) == len(frame["public"]["entities"]) + 1 + len(frame["public"]["decoder_bank"])


def test_map_direction_floor_and_dag(demo):
    frame = deepcopy(demo["macros"][1]["steps"][0]["frame"])
    frame["public"]["entities"] += [{"ref": f"n{i}", "entity_type": "map_node", "floor": i} for i in range(4)]
    frame["public"]["relations"] = [{"source": "n0", "target": "n1", "role": "map_edge"},
                                      {"source": "n1", "target": "n2", "role": "map_edge"}]
    obs = observation(frame)
    a, b = obs.refs["n0"], obs.refs["n2"]
    assert (a, b, "map_forward_indirect") in obs.edges
    assert (b, a, "map_reverse_indirect") in obs.edges
    assert (a, obs.refs["n3"], "map_unreachable") in obs.edges
    frame["public"]["relations"].append({"source": "n2", "target": "n0", "role": "map_edge"})
    with pytest.raises(ProtocolError, match="DAG"):
        observation(frame)
