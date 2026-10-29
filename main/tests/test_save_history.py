"""A continue save keeps the public history its run had observed, and a run resumed
from it is told it (`main.env.saves`: `continue_save`, `resume_fields`)."""

import json
from pathlib import Path

from main.env.engine import CliEngine
from main.env.saves import PUBLIC_HISTORY, continue_save, fight_resume_command, resume_fields

HISTORY = dict(version="public-history-v1", act=1, potion=0.3, rooms=dict(Monster=0.2, Elite=-1.0, Treasure=0.04,
               Shop=0.06), tutorial_rolls=None, observed_tutorial_rolls=0, previous_point_had_shop=False)


class Saving:
    """An engine that writes one continue save and answers with `history` beside it."""

    def __init__(self, history):
        self.history = history

    def send(self, command):
        assert command["cmd"] == "write_continue_save"
        Path(command["path"]).write_text(json.dumps({"players": [{"current_hp": 50}]}))
        reply = {"type": "save_result", "success": True}
        if self.history is not None:
            reply["public_history"] = self.history
        return reply


def test_a_continue_save_keeps_the_history_the_engine_returned_beside_it():
    save = continue_save(Saving(HISTORY))
    assert save == {"players": [{"current_hp": 50}], PUBLIC_HISTORY: HISTORY}
    # An engine that returns none (before it did) leaves the save as it wrote it.
    assert continue_save(Saving(None)) == {"players": [{"current_hp": 50}]}


def test_resuming_hands_the_history_back_and_leaves_the_save_as_it_was():
    save = {"players": [], PUBLIC_HISTORY: HISTORY}
    assert resume_fields(save) == dict(save={"players": []}, public_history=HISTORY)
    assert save == {"players": [], PUBLIC_HISTORY: HISTORY}
    assert resume_fields({"players": []}) == dict(save={"players": []})
    command = fight_resume_command(dict(save=save, encounter="ENCOUNTER.X", floor=3, map_point_type="monster"))
    assert command["save"] == {"players": []} and command["public_history"] == HISTORY
    assert "public_history" not in fight_resume_command(dict(save={"players": []}, encounter="ENCOUNTER.X", floor=3,
                                                             map_point_type="monster"))


class Recording(CliEngine):
    """A `CliEngine` without a process, recording what it would send."""

    def __init__(self):
        self.sent, self.closed, self.runs = [], False, 0
        self.proc = type("Running", (), {"poll": staticmethod(lambda: None)})()

    def send(self, command):
        self.sent.append(command)
        return {"type": "resume_installed"} if command["cmd"] == "resume_save" else {"type": "decision_frame"}


def test_the_engine_resumes_a_save_with_its_history():
    engine = Recording()
    engine.resume({"players": [], PUBLIC_HISTORY: HISTORY})
    assert engine.sent[0] == {"cmd": "resume_save", "save": {"players": []}, "public_history": HISTORY}
    engine.resume({"players": []})
    assert engine.sent[2] == {"cmd": "resume_save", "save": {"players": []}}
