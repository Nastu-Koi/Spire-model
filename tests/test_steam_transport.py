import json
import threading
import time

import pytest

from model.protocol import ProtocolError
from model.steam import SteamBridge


def test_transport_matches_request_id_and_locks_controller(tmp_path):
    bridge = SteamBridge(tmp_path, timeout=2)
    with pytest.raises(ProtocolError, match="Another model controller"):
        SteamBridge(tmp_path)
    (tmp_path / "response.json").write_text('{"id":"old","status":"executed"}')
    def game():
        path = tmp_path / "request.json"
        deadline = time.monotonic() + 2
        while not path.exists():
            if time.monotonic() > deadline:
                return
            time.sleep(.01)
        request = json.loads(path.read_text())
        temp = tmp_path / "response.tmp"
        temp.write_text(json.dumps(dict(id=request["id"], status="waiting")))
        temp.replace(tmp_path / "response.json")
    thread = threading.Thread(target=game)
    thread.start()
    try:
        assert bridge.request("observe")["status"] == "waiting"
    finally:
        bridge.close()
        thread.join()
    assert not (tmp_path / "request.json").exists()


def test_transport_timeout_removes_undelivered_command(tmp_path):
    bridge = SteamBridge(tmp_path, timeout=.02)
    try:
        with pytest.raises(TimeoutError):
            bridge.request("execute", token="expired")
        assert not (tmp_path / "request.json").exists()
    finally:
        bridge.close()
