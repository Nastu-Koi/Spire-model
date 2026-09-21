import json
from pathlib import Path
import sys

import pytest
import torch

from model.config import ModelConfig
from model.engine import CliEngine
from model.model import PolicyValue
from model.policy import SessionPolicy
from model.protocol import ProtocolError, execution_command, validate_frame
from model.representation import Vocabulary, symbols_from_frames


def test_json_lines_transport_and_nonjson_diagnostics(tmp_path):
    script = tmp_path / "engine.py"
    script.write_text('import sys,json\nprint("diagnostic", flush=True)\nprint(json.dumps({"type":"ready"}),flush=True)\nfor line in sys.stdin:\n print(json.dumps({"echo":json.loads(line)}),flush=True)\n')
    with CliEngine([sys.executable, "-u", str(script)], root=tmp_path, timeout=2) as engine:
        assert engine.send({"cmd": "hello"}) == {"echo": {"cmd": "hello"}}
    assert engine.proc.poll() is not None


def test_timeout_closes_stream(tmp_path):
    script = tmp_path / "engine.py"
    script.write_text('import time\nprint(\'{"type":"ready"}\',flush=True)\ntime.sleep(10)\n')
    engine = CliEngine([sys.executable, "-u", str(script)], root=tmp_path, timeout=.2)
    with pytest.raises(TimeoutError):
        engine.send({"cmd": "wait"})
    assert engine.closed
    assert engine.proc.poll() is not None


@pytest.mark.engine
def test_real_cli_selection_model_and_native_cancelable_commit():
    root = Path(__file__).resolve().parents[1] / "sts2-cli"
    dll = root / "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll"
    if not dll.exists() or not (root / "lib/sts2.dll").exists():
        pytest.skip("Build the local engine and supply game assemblies")
    with CliEngine(root=root) as engine:
        state = engine.send({"cmd": "start_run", "character": "Ironclad", "ascension": 10, "seed": "model-integration", "lang": "en"})
        # Explicit fixture setup only; none of these debug operations enter a policy candidate set.
        state = engine.send({"cmd": "set_player", "hp": 70, "max_hp": 80, "deck": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"]})
        state = engine.send({"cmd": "enter_room", "type": "rest_site"})
        smith = next(o for o in state["options"] if o["option_id"] == "SMITH")
        state = engine.send({"cmd": "action", "action": "choose_option", "args": {"option_index": smith["index"]}})
        assert state["decision"] == "card_select"
        frame = engine.send({"cmd": "advance_to_boundary"})
        validate_frame(frame, allow_prototype=True)
        with pytest.raises(ProtocolError, match="training_ready"):
            validate_frame(frame)
        config = ModelConfig.tiny()
        vocab = Vocabulary(symbols_from_frames([frame]), config.vocabulary_size)
        model = PolicyValue(config)
        policy = SessionPolicy(model, vocab)
        with torch.no_grad():
            # Smith requires manual confirmation and permits cancellation even
            # after selecting the card. Replay this concrete valid sequence.
            pick = next(c for c in frame["legal"]["candidates"] if c["verb"] == "SELECT_ONE")
            choice = policy.choose(frame, teacher=pick["candidate_ref"], top_k=5)
            next_frame = engine.send(execution_command(frame, choice.candidate_ref))
            validate_frame(next_frame, allow_prototype=True)
            stop = next(c for c in next_frame["legal"]["candidates"] if c["verb"] == "FINISH_SELECTION")
            committed = policy.choose(next_frame, teacher=stop["candidate_ref"])
        assert model.encoder_calls == 1
        assert model.decoder_calls == 2
        assert committed.log_prob is not None
        stale = engine.send(execution_command(frame, choice.candidate_ref))
        assert stale["error"]["code"] == "stale_decision"
        final = engine.send(execution_command(next_frame, committed.candidate_ref))
        validate_frame(final, allow_prototype=True)
        assert final["public"]["phase"] == "map"
        state = engine.send({"cmd": "set_player"})
        assert sum(c["upgraded"] for c in state["player"]["deck"]) == 1
