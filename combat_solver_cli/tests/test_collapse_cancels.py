"""Prefix records drop non-combat open -> select -> cancel detours."""

import pytest

from combat_solver_cli.search_support import ReplayMismatch, collapse_noncombat_cancels


def record(phase, verb, actor="a0_policy"):
    return {"phase": phase, "action": {"verb": verb}, "actor": actor}


def test_drops_opening_action_selections_and_cancel():
    records = [
        record("map", "MOVE_TO_NODE", "forced"),
        record("rest_site", "CHOOSE_REST_OPTION"),
        record("card_select", "SELECT_ONE"),
        record("card_select", "CANCEL"),
        record("rest_site", "CHOOSE_REST_OPTION"),
    ]
    assert collapse_noncombat_cancels(records) == [records[0], records[4]]


def test_keeps_solver_cancels_in_combat():
    records = [
        record("combat", "PLAY_CARD", "combat_solver"),
        record("card_select", "CANCEL", "combat_solver"),
    ]
    assert collapse_noncombat_cancels(records) == records


def test_rejects_cancel_without_opening_action():
    with pytest.raises(ReplayMismatch):
        collapse_noncombat_cancels([record("card_select", "SELECT_ONE"), record("card_select", "CANCEL")])
