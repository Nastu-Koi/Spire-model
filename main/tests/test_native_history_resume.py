"""A run resumed from a continue save goes on with the public history its own run had
observed, against the native engine (`saves.continue_save`, `CliEngine.resume`,
`CliEngine.resume_fight`): the odds of a potion drop and of the question rooms, and
whether the point before had a shop. Resumed without it, the history is unknown."""

import pytest

from combat_solver_cli.client import SolverEngine
from main.env.saves import PUBLIC_HISTORY, continue_save
from main.tests.test_native_fight_entry import into_monster_room, native_config, resume_row, settle, to_map


def history(frame):
    """The frame's public history: each probability and the previous-shop rule."""
    return {(e["content_id"], e.get("scope")): e.get("probability", e.get("enabled"))
            for e in frame["public"]["memory"] if e["entity_type"] in ("history_probability", "history_rule")}


@pytest.mark.engine
def test_native_resume_goes_on_with_the_public_history_of_its_run():
    with SolverEngine(native_config()) as engine:
        frame = to_map(engine, settle(engine, engine.reset("Ironclad", "history-resume", 0)))
        # After a fight its reward has rolled for a potion: the odds are no longer the prior.
        frame = to_map(engine, into_monster_room(engine, frame)[0])
        seen = history(frame)
        assert seen and all(value["known"] for value in seen.values())
        save = continue_save(engine)
        assert save[PUBLIC_HISTORY]["potion"] is not None and save[PUBLIC_HISTORY]["potion"] != 0.4

        blind = settle(engine, engine.resume({k: v for k, v in save.items() if k != PUBLIC_HISTORY}))
        assert not any(value["known"] for (kind, _), value in history(blind).items() if kind == "potion_drop")

        resumed = settle(engine, engine.resume(save))
        assert history(resumed) == seen
        # A fight's entry save written in the resumed run: entered again, the fight's
        # first decision knows what it knew.
        entered, started = into_monster_room(engine, resumed)
        again = engine.resume_fight(resume_row(engine, entered, started))
        assert history(again) == history(entered)
        assert all(value["known"] for value in history(again).values())
