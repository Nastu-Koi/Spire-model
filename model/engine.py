"""One serial JSON-lines connection per engine process, with bounded I/O."""
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading

from .protocol import ProtocolError


class CliEngine:
    def __init__(self, command=None, *, root=None, timeout=30.0):
        self.root = Path(root or Path(__file__).resolve().parents[1] / "sts2-cli").resolve()
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        if command is None:
            dll = self.root / "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll"
            if not dll.exists():
                raise FileNotFoundError(f"Build the engine first: dotnet build {self.root}/src/Sts2Headless")
            command = [shutil.which("dotnet") or "dotnet", str(dll)]
        env = dict(os.environ, STS2_GAME_DIR=str(self.root / "lib"))
        self.stderr = tempfile.TemporaryFile(mode="w+t")
        self.proc = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)
        self.responses = queue.Queue(maxsize=32)
        self.closed = False
        self.lock = threading.Lock()

        def reader():
            try:
                for line in self.proc.stdout:
                    if line.lstrip().startswith("{"):
                        self.responses.put(line)
            finally:
                self.responses.put(None)

        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            if self._read().get("type") != "ready":
                raise ProtocolError("Engine did not send ready")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            line = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()  # A timed-out stream cannot safely associate the next response.
            raise TimeoutError("Engine response timeout; connection closed") from None
        if line is None:
            raise ProtocolError(f"Engine exited ({self.proc.poll()})")
        try:
            return json.loads(line)
        except ValueError as exc:
            raise ProtocolError("Malformed engine JSON") from exc

    def send(self, command):
        with self.lock:
            if self.closed:
                raise ProtocolError("Engine connection is closed")
            self.proc.stdin.write(json.dumps(command, allow_nan=False) + "\n")
            self.proc.stdin.flush()
            return self._read()

    def reset(self, character, seed):
        response = self.send({"cmd": "start_run", "character": character, "seed": str(seed),
                              "ascension": 10, "lang": "en", "decision_protocol": True})
        if response.get("type") == "error":
            raise ProtocolError(str(response))
        return response if response.get("type") == "decision_frame" else self.send({"cmd": "advance_to_boundary"})

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.proc.terminate()
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=3)
        self.proc.stdin.close()
        self.proc.stdout.close()
        if threading.current_thread() is not self.reader:
            self.reader.join(timeout=1)
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
