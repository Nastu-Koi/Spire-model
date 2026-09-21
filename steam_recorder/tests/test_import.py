from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from model.data import load_runs, validate_run
from model.protocol import ProtocolError, fingerprint
from model.recorder import convert_journal, decision_macro, import_recordings, PublicSnapshot


def card(identity, name="STRIKE_IRONCLAD"):
    return {"instance_id": identity, "id": name, "current_cost": 1, "cost": 1,
            "card_type": "Attack", "rarity": "Basic", "keywords": [], "vars": {"Damage": {"base_value": 6}},
            "saved_properties": {"hidden": "forbidden"}}


def state(scope="combat_play"):
    return {"run": {"ascension": 10, "seed": "test-recorder-seed", "act": 1, "floor": 1, "visited": []},
            "players": [{"character": "IRONCLAD", "creature": {"instance_id": 10, "hp": 20, "max_hp": 80,
                        "block": 0, "powers": []}, "gold": 99, "max_potions": 2, "deck": [card(1), card(2)],
                        "combat": {"energy": 3, "max_energy": 3, "stars": 0, "hand": [card(3)],
                                   "draw": [card(4), card(5, "DEFEND_IRONCLAD")], "discard": [], "exhaust": []},
                        "potions": [], "relics": []}],
            "combat": {"in_progress": True, "round": 1, "enemies": [{"instance_id": 20, "id": "JAW_WORM",
                       "hp": 30, "max_hp": 40, "block": 0, "move": "hidden_ai", "powers": [],
                       "intents": [{"type": "Attack", "damage": 6, "repeats": 2}]}]},
            "map": [], "ui": [{"answer": "secret"}], "visibility": {"draw_pile_order": "engine_internal"},
            "legal": {"schema": 1, "status": "complete", "scope": scope, "selection": None, "context": None,
                      "actions": [{"command": "play_card", "args": {"card_instance_id": 3, "target_instance_id": 20}},
                                  {"command": "end_turn", "args": {}}]}}


def decision(order=1):
    s = state()
    return {"action_id": f"decision-{order}", "order": order, "status": "completed", "actor": "human",
            "actor_evidence": "game_ui_call_stack", "state_boundary": "execution_or_semantic_entry",
            "state": s, "choice_key": s["legal"]["actions"][0], "legal_match": {"status": "matched"},
            "next_state": {"secret": "future"}, "choices": []}


def journal(tmp_path, decisions=None):
    ds = decisions if decisions is not None else [decision()]
    start = {"begins_at_run_start": True, "recorder_version": "0.4.0", "game_version": "v0.111.0",
             "seed": "test-recorder-seed", "state": state(),
             "loaded_assemblies": [{"name": "sts2", "mvid": "test-game-assembly"}]}
    records = [("segment_start", start)] + [("decision_committed", d) for d in ds] + [
        ("run_ended", {"victory": False, "state": {"run": {"ended": True, "victory": False, "ascension": 10}}}),
        ("segment_end", {"run_ended": True, "uncommitted_operations": 0, "state_capture_errors": 0})]
    rows = [{"schema": "sts2-run-recorder/v2", "run_id": "fixture-run", "segment_id": "fixture-segment",
             "seq": i, "kind": kind, "data": data} for i, (kind, data) in enumerate(records, 1)]
    path = tmp_path / "recording.jsonl"
    write_rows(path, rows)
    return path, rows


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("bc_only", [False, True])
def test_recording_to_training_checkpoint(tmp_path, bc_only):
    import torch
    from model.cli import vocabulary_for
    from model.config import ModelConfig, TrainConfig
    from model.model import PolicyValue
    from model.trainer import Learner
    from model.checkpoint import save_checkpoint, load_model
    torch.set_num_threads(2)
    raw, _ = journal(tmp_path)
    output = tmp_path / "import"
    result = import_recordings([raw], output, bc_only=bc_only)
    assert result["accepted_runs"] == 1
    runs = load_runs(output / "accepted.jsonl")
    cfg = ModelConfig.tiny()
    vocabulary = vocabulary_for(runs, cfg.vocabulary_size)
    model = PolicyValue(cfg)
    learner = Learner(model, vocabulary, TrainConfig(precision="no"))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    learner.bootstrap(runs)
    assert learner.updates == 1
    assert any(not torch.equal(before[name], value) for name, value in model.named_parameters())
    save_checkpoint(tmp_path / "checkpoint", model, vocabulary, training=learner.config,
                    progress={"policy_version": learner.policy_version})
    loaded, _, _ = load_model(tmp_path / "checkpoint")
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), loaded.parameters()))
    with pytest.raises(FileExistsError):
        import_recordings([raw], output)


def test_public_boundary_removes_hidden_fields_and_pile_order(tmp_path):
    path, rows = journal(tmp_path)
    first = convert_journal(path)["macros"][0]["steps"][0]["frame"]
    raw = rows[1]["data"]
    raw["next_state"] = {"seed": "different-future"}
    raw["state"]["players"][0]["combat"]["draw"].reverse()
    raw["state"]["combat"]["enemies"][0]["move"] = "other_hidden_move"
    raw["state"]["ui"] = [{"answer": "different"}]
    write_rows(path, rows)
    second = convert_journal(path)["macros"][0]["steps"][0]["frame"]
    assert first == second
    # Seed lives only in audit metadata and is never in the public snapshot.
    a = deepcopy(raw["state"])
    a["run"]["seed"] = "different-seed"
    assert PublicSnapshot(a).entities == PublicSnapshot(raw["state"]).entities
    text = json.dumps(first["public"])
    for hidden in ("instance_id", "saved_properties", "seed", "hidden_ai", "future", "secret"):
        assert hidden not in text


def test_hidden_rewards_do_not_enter_observation():
    s = state("reward_choice")
    s["legal"].update(context={"rewards": [{"instance_id": 50, "reward_type": "Card", "cards": [card(100)]}]},
                      actions=[{"command": "take_reward", "args": {"reward_instance_id": 50}},
                               {"command": "skip_rewards", "args": {"reward_set_id": 0}}])
    a = PublicSnapshot(s).entities
    s["legal"]["context"]["rewards"][0]["cards"] = [card(200, "SECRET_REWARD")]
    assert a == PublicSnapshot(s).entities


def test_nested_decisions_follow_execution_order_and_inherit_teacher(tmp_path):
    outer, child = decision(1), decision(2)
    child["actor_evidence"] = "parent_action:decision-1"
    outer["choices"] = [child]
    path, _ = journal(tmp_path, [outer, decision(3)])
    run = convert_journal(path)
    assert [m["steps"][0]["frame"]["routing"]["decision_id"] for m in run["macros"]] == ["decision-1", "decision-2", "decision-3"]


@pytest.mark.parametrize("damage,reason", [
    (lambda r: r.pop(), "no segment end"),
    (lambda r: r[-2]["data"].pop("victory"), "terminal outcome"),
    (lambda r: r[0]["data"].update(begins_at_run_start=False), "Resumed fragment"),
    (lambda r: r[-1]["data"].update(uncommitted_operations=1), "uncommitted"),
    (lambda r: r[-1]["data"].update(state_capture_errors=1), "failed state captures"),
    (lambda r: r[1].update(seq=99), "sequence gap"),
    (lambda r: r[1]["data"].update(actor="combat_solver"), "Teacher visibility"),
    (lambda r: r[1]["data"]["state"]["legal"].update(status="unavailable"), "complete recorded legal set"),
    (lambda r: r[1]["data"].update(choice_key={"command": "nonexistent", "args": {}}), "not uniquely"),
    (lambda r: r[1]["data"].update(status="failed"), "not completed"),
    (lambda r: r[0]["data"]["state"]["run"].update(ascension=0), "not A10"),
])
def test_rejects_bad_runs_without_partial_training_data(tmp_path, damage, reason):
    path, rows = journal(tmp_path)
    damage(rows)
    write_rows(path, rows)
    with pytest.raises(ProtocolError, match=reason):
        convert_journal(path)
    report = import_recordings([path], tmp_path / "import")
    assert report["accepted_runs"] == 0 and report["quarantined_files"] == 1
    assert (tmp_path / "import" / "accepted.jsonl").read_text() == ""


def test_truncated_last_line_and_duplicate_run(tmp_path):
    path, _ = journal(tmp_path)
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    with pytest.raises(ProtocolError, match="Truncated"):
        convert_journal(path)
    path, _ = journal(tmp_path)
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(path.read_bytes())
    result = import_recordings([tmp_path], tmp_path / "import")
    assert result["accepted_runs"] == result["quarantined_files"] == 1


@pytest.fixture(scope="module")
def recorder_exports(tmp_path_factory):
    directory = tmp_path_factory.mktemp("recorder-checks")
    path = directory / "selection.json"
    root = Path(__file__).resolve().parents[2]
    game = Path("/mnt/d/SteamLibrary/steamapps/common/Slay the Spire 2/data_sts2_windows_x86_64")
    if not game.exists():
        pytest.skip("Local Steam game assemblies required for recorder runtime checks")
    headless = root / "sts2-cli/src/Sts2Headless/bin/Release/net9.0/Sts2Headless.dll"
    subprocess.run(["dotnet", "run", "--project", str(root / "steam_recorder/tests/RecorderChecks"),
                    "-c", "Release", f"-p:GameDataDir={game}", "--", str(game), str(path), str(directory / "states.json")]
                   + ([str(headless)] if headless.exists() else []),
                   check=True, capture_output=True, text=True)
    return json.loads(path.read_text()), json.loads((directory / "states.json").read_text())


@pytest.fixture(scope="module")
def recorded_selection_path(recorder_exports):
    return recorder_exports[0]


def test_real_dark_orbs_survive_import_and_model_fields(recorder_exports, tmp_path):
    from model.representation import observation
    d = decision()
    combat = d["state"]["players"][0]["combat"]
    combat.update(orbs=recorder_exports[1]["orbs"], orb_slots=5)
    path, _ = journal(tmp_path, [d])
    frame = convert_journal(path)["macros"][0]["steps"][0]["frame"]
    orbs = [e for e in frame["public"]["entities"] if e["entity_type"] == "orb"]
    assert [(o["position"], o["stats"]) for o in orbs] == [
        (0, {"passive": 6, "evoke": 36}), (1, {"passive": 6, "evoke": 18})]
    before = observation(frame)
    combat["orbs"][0] = recorder_exports[1]["after_orb"]
    path, _ = journal(tmp_path, [d])
    after = observation(convert_journal(path)["macros"][0]["steps"][0]["frame"])
    assert before.digest != after.digest
    assert any(f.name == "stats.evoke" and f.number[0] == 4.2 for token in after.tokens for f in token)


def test_real_mad_science_variants_reach_model(recorder_exports):
    from model.representation import clean_entity, fields_of
    from model.recorder import card_fields
    signatures = set()
    for raw in recorder_exports[1]["cards"]:
        public = clean_entity(card_fields(raw))
        assert public["rider_effect"] == raw["mad_science"]["rider_effect"]
        assert public["card_type"] == raw["mad_science"]["card_type"]
        assert len(public["stats"]) == 11
        fields = fields_of(public)
        assert any(f.name == "rider_effect" and f.symbol.endswith("=" + public["rider_effect"]) for f in fields)
        assert any(f.name == "stats.ViolenceHits" and f.number[0] == .3 for f in fields)
        signatures.add(fingerprint(public))
        assert "saved_properties" not in public
    assert len(signatures) == len(recorder_exports[1]["cards"])


def test_mad_science_missing_variant_is_rejected():
    from model.recorder import card_fields
    with pytest.raises(ProtocolError, match="type/rider"):
        card_fields(card(1, "MAD_SCIENCE"))


def test_mad_science_steam_and_headless_parity(recorder_exports):
    from model.recorder import card_fields
    from model.representation import clean_entity
    exports = recorder_exports[1]
    if not exports["headless_cards"]:
        pytest.skip("Build the Release headless adapter to check live card parity")
    assert len(exports["cards"]) == len(exports["headless_cards"])
    for raw, headless in zip(exports["cards"], exports["headless_cards"]):
        steam, engine = clean_entity(card_fields(raw)), clean_entity(headless)
        for key in ("content_id", "card_type", "rider_effect", "stats", "target_type", "keywords",
                    "cost", "stars_x", "retain", "exhaust_on_next_play"):
            assert steam[key] == engine[key], key


def test_public_card_flags_are_preserved():
    from model.recorder import card_fields
    from model.representation import clean_entity
    flags = dict(retain=True, stars_x=True, exhaust_on_next_play=True)
    public = clean_entity(card_fields({**card(1), **flags}))
    assert all(public[k] is v for k, v in flags.items())


def test_csharp_selection_path_import_and_forced_finish(tmp_path, recorded_selection_path):
    d = decision()
    s = d["state"]
    s["legal"].update(scope="card_select", context=[card(1), card(2), card(3)],
                      selection={"min": 2, "max": 2, "cancelable": False, "ordered": True},
                      actions=[{"command": "select_card", "args": {"card_instance_id": i}} for i in (1, 2, 3)])
    d.update(choice_key={"command": "select_cards", "args": {"card_instance_ids": [3, 1]}},
             selection_path=recorded_selection_path, state_boundary="selection_offer")
    path, rows = journal(tmp_path, [d])
    run = convert_journal(path)
    validate_run(run)
    steps = run["macros"][0]["steps"]
    assert len(steps) == 3 and [s["forced"] for s in steps] == [False, False, True]
    assert [len(s["frame"]["legal"]["candidates"]) for s in steps] == [3, 2, 1]
    rows[1]["data"]["selection_path"]["steps"][1]["actions"].pop()
    write_rows(path, rows)
    with pytest.raises(ProtocolError, match="prefix mask"):
        convert_journal(path)


def test_internal_id_renumbering_is_feature_invariant(tmp_path):
    path, rows = journal(tmp_path)
    before = convert_journal(path)["macros"]
    def renumber(value):
        if isinstance(value, list):
            return [renumber(x) for x in value]
        if isinstance(value, dict):
            return {k: (v + 5000 if k.endswith("instance_id") and isinstance(v, int) else renumber(v))
                    for k, v in value.items()}
        return value
    write_rows(path, renumber(rows))
    assert convert_journal(path)["macros"] == before


def test_crystal_fog_hides_item_geometry_and_reveals_only_public_kind():
    s = state("crystal_sphere")
    s["legal"].update(context={"remaining": 2, "cells": [
        {"x": 0, "y": 0, "hidden": True, "item": {"type": "SECRET", "width": 3}},
        {"x": 1, "y": 0, "hidden": False, "item": {"type": "game.RelicItem", "width": 4}}]},
        actions=[{"command": "crystal_sphere_cell", "args": {"x": 0, "y": 0, "tool": t}} for t in ("Small", "Big")])
    snap = PublicSnapshot(s)
    cells = [e for e in snap.entities if e["entity_type"] == "board_cell"]
    assert "content_id" not in cells[0] and "width" not in cells[1]
    assert cells[1]["content_id"] == "RelicItem"
    assert snap.action(s["legal"]["actions"][0], 0, "crystal_sphere")["target_refs"] == ["cell:0:0"]


def test_shop_model_and_reward_card_binding():
    s = state("shop")
    s["legal"]["context"] = [{"instance_id": 50, "value": {"type": "game.MerchantRelicEntry",
        "Model": {"instance_id": 51, "id": "ANCHOR"}, "Cost": 150, "IsStocked": True}}]
    snap = PublicSnapshot(s)
    item = next(e for e in snap.entities if e.get("ref") == "shop:0")
    assert item["content_id"] == "RELIC.ANCHOR" and item["price"] == 150
    s["legal"].update(scope="card_reward", context={"cards": [card(99)], "alternatives": [{"OptionId": "SKIP"}]})
    snap = PublicSnapshot(s)
    assert snap.action({"command": "select_reward_option", "args": {"index": 0}}, 0, "card_reward")["verb"] == "TAKE_CARD_REWARD"
    assert snap.action({"command": "select_reward_option", "args": {"index": 1}}, 1, "card_reward")["verb"] == "CHOOSE_REWARD_ALTERNATIVE"
    assert snap.action({"command": "select_reward_option", "args": {"index": None}}, 2, "card_reward")["verb"] == "SKIP"


def test_import_cli_exit_status(tmp_path, capsys):
    from model.cli import main
    path, rows = journal(tmp_path)
    assert main(["import-recorder", str(path), "--output", str(tmp_path / "good")]) == 0
    assert json.loads(capsys.readouterr().out)["accepted_runs"] == 1
    rows[1]["data"]["actor"] = "combat_solver"
    write_rows(path, rows)
    assert main(["import-recorder", str(path), "--output", str(tmp_path / "bad")]) == 2
    assert json.loads(capsys.readouterr().out)["accepted_runs"] == 0


def test_bootstrap_rejects_empty_training_before_model_allocation(tmp_path, capsys, monkeypatch):
    from model import cli
    data = tmp_path / "accepted.jsonl"
    data.write_text("")
    (tmp_path / "summary.json").write_text('{"accepted_runs": 0}')
    def unexpected_model(*args, **kwargs):
        pytest.fail("Empty Bootstrap data must be rejected before allocating a model")
    monkeypatch.setattr(cli, "PolicyValue", unexpected_model)
    assert cli.main(["bootstrap", "--data", str(data), "--output", str(tmp_path / "checkpoint")]) == 1
    assert "summary.json" in json.loads(capsys.readouterr().err)["error"]
    assert not (tmp_path / "checkpoint").exists()


def test_bootstrap_only_accepts_solver_and_partial_runs_without_faking_outcomes(tmp_path):
    d = decision()
    d.update(actor="combat_solver", actor_evidence="call_stack:CombatSolver")
    path, rows = journal(tmp_path, [d])
    rows = rows[:-2]
    rows[0]["data"].update(begins_at_run_start=False)
    write_rows(path, rows)
    run = convert_journal(path, bc_only=True, recorder_version="0.4.0")
    assert run["source"] == "recorder_bc" and run["teacher_visibility"] == "unverified"
    assert run["status"] == "partial" and run["victory"] is None
    assert run["provenance"]["actors"] == {"combat_solver": 1}
    assert len(run["macros"]) == 1
    validate_run(run)
    with pytest.raises(ProtocolError, match="supervised-only"):
        validate_run(run, on_policy=True)
    with pytest.raises(ProtocolError):
        convert_journal(path)


def test_bootstrap_only_filters_version_and_audits_bad_decisions(tmp_path):
    good, bad = decision(), decision(2)
    bad["state"]["legal"]["status"] = "unavailable"
    path, _ = journal(tmp_path, [good, bad])
    summary = import_recordings([path], tmp_path / "bootstrap", bc_only=True, recorder_version="0.4.0")
    assert summary["accepted_runs"] == 1 and summary["rejected_decisions"] == 1
    run = load_runs(tmp_path / "bootstrap/accepted.jsonl")[0]
    assert len(run["macros"]) == 1
    assert run["provenance"]["rejected_decisions"][0]["action_id"] == "decision-2"
    summary = import_recordings([path], tmp_path / "filtered", bc_only=True, recorder_version="0.4.1")
    assert summary["accepted_runs"] == 0 and summary["version_filtered_files"] == 1


def test_bootstrap_all_training_data_trains_without_holdout(tmp_path, capsys, monkeypatch):
    from model import cli
    raw, _ = journal(tmp_path)
    imported = tmp_path / "import"
    import_recordings([raw], imported, bc_only=True)
    def unexpected_split(*args, **kwargs):
        pytest.fail("--all-training-data must not split off demonstrations")
    monkeypatch.setattr(cli, "split_runs", unexpected_split)
    assert cli.main(["bootstrap", "--data", str(imported / "accepted.jsonl"), "--all-training-data",
                     "--tiny", "--precision", "no", "--output", str(tmp_path / "checkpoint")]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["validation_runs"] == 0
    assert metrics["metrics"]["updates"] > 0


def test_sold_out_shop_slot_needs_no_missing_model():
    s = state("shop")
    s["legal"].update(context=[{"instance_id": 50, "value": {
        "type": "game.MerchantRelicEntry", "Cost": 53, "IsStocked": False}}],
        actions=[{"command": "leave_shop", "args": {}}])
    public = PublicSnapshot(s)
    slot = next(e for e in public.entities if e.get("ref") == "shop:0")
    assert slot["sold_out"] and slot["content_id"] is None
    s["legal"]["context"][0]["value"]["IsStocked"] = True
    with pytest.raises(ProtocolError, match="Missing visible shop item"):
        PublicSnapshot(s)


def test_live_steam_input_matches_bc_public_features():
    from model.steam import live_frame
    d = decision()
    message = dict(state=d["state"], token="live", episode="game")
    live, actions = live_frame(message)
    recorded = decision_macro(d, "game", live["contract"])["steps"][0]["frame"]
    assert live["public"] == recorded["public"]
    assert live["legal"] == recorded["legal"]
    assert actions["c0"] == d["choice_key"]
    assert "hidden_ai" not in json.dumps(live)
    assert "secret" not in json.dumps(live)


def test_live_buffered_selection_masks_and_commit():
    from model.steam import choose_action, live_frame
    from model.policy import Choice
    s = state("card_select")
    s["legal"].update(context=[card(3), card(4), card(5)],
                      selection=dict(min=2, max=2, cancelable=False, ordered=True),
                      actions=[dict(command="select_card", args=dict(card_instance_id=i)) for i in (3, 4, 5)])
    message = dict(state=s, token="offer", episode="game")
    first, _ = live_frame(message)
    second, _ = live_frame(message, [3])
    final, _ = live_frame(message, [3, 4])
    assert len(first["legal"]["candidates"]) == 3
    assert len(second["legal"]["candidates"]) == 2
    assert [c["verb"] for c in final["legal"]["candidates"]] == ["FINISH_SELECTION"]
    assert first["public"]["decoder_bank"] == final["public"]["decoder_bank"]
    class FirstPolicy:
        def choose(self, frame, **_):
            return Choice(frame["legal"]["candidates"][0]["candidate_ref"])
    assert choose_action(FirstPolicy(), message) == dict(command="select_cards", args=dict(card_instance_ids=[3, 4]))
