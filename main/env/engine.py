"""One serial JSON-lines connection per engine process, with bounded I/O."""

import json
import os
import queue
import subprocess
import tempfile
import threading
from pathlib import Path

from main.env.dotnet_runtime import runtime_command
from main.env.protocol import ProtocolError
from main.env.saves import fight_resume_command


# An engine process grows by a few megabytes with every run it plays.
RUNS_PER_PROCESS = 16


class CliEngine:
    def __init__(self, command=None, *, root=None, timeout=30.0):
        self.root = Path(
            root or Path(__file__).resolve().parents[2] / "sts2-cli"
        ).resolve()
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        if command is None:
            dll = self.root / "src/Sts2Headless/bin/Debug/net9.0/Sts2Headless.dll"
            if not dll.exists():
                raise FileNotFoundError(
                    f"Build the engine first: dotnet build {self.root}/src/Sts2Headless"
                )
            command = runtime_command(dll)
        self.command = command
        self.lock = threading.Lock()
        self._start()

    def _start(self):
        """Begin a process and wait until it is ready."""
        self.stderr = tempfile.TemporaryFile(mode="w+t")
        self.proc = proc = subprocess.Popen(
            self.command,
            cwd=self.root,
            env=dict(os.environ, STS2_GAME_DIR=str(self.root / "lib")),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        self.responses = responses = queue.Queue(maxsize=32)
        self.closed = False
        self.runs = 0

        def reader():
            try:
                for line in proc.stdout:
                    if line.lstrip().startswith("{"):
                        responses.put(line)
            finally:
                responses.put(None)

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

    def reset(self, character, seed, ascension):
        """Begin a run. The process that played the run before begins this one too,
        without loading the game again; a new process does where there is none left,
        where it has played its share of runs, or where the run before cannot be
        ended (it stopped in a fight, or failed)."""
        command = {
            "cmd": "start_run",
            "character": character,
            "seed": str(seed),
            "ascension": ascension,
            "lang": "en",
            "decision_protocol": True,
        }
        if self.closed or self.proc.poll() is not None or self.runs >= RUNS_PER_PROCESS:
            self.close()
            self._start()
        response = self.send(command)
        if response.get("code") == "run_not_ended":
            self.close()
            self._start()
            response = self.send(command)
        self.runs += 1
        if response.get("type") == "error":
            raise ProtocolError(str(response))
        return (
            response
            if response.get("type") == "decision_frame"
            else self.send({"cmd": "advance_to_boundary"})
        )

    def resume(self, save):
        """Resume a run from a continue save written at a map decision
        (`saves.continue_save`): the same state, played on from the map, with the
        engine's `resume_save`. Returns the map decision's frame, as `reset` returns a
        run's first; the process is reused or replaced as `reset` does."""
        command = {"cmd": "resume_save", "save": save}
        if self.closed or self.proc.poll() is not None or self.runs >= RUNS_PER_PROCESS:
            self.close()
            self._start()
        reply = self.send(command)
        if reply.get("code") == "run_not_ended":
            self.close()
            self._start()
            reply = self.send(command)
        self.runs += 1
        if reply.get("type") != "resume_installed":
            raise ProtocolError("resume_save: " + str(reply.get("message") or reply))
        return self.send({"cmd": "advance_to_boundary"})

    def resume_fight(self, row):
        """Resume a run at one of its fights from the save written at the fight's first
        decision (`saves.fight_resume_command`): the engine's `resume_fight` installs
        the save at the fight's map point and enters the encounter again. Returns the
        fight's first frame; the process is reused or replaced as `reset` does."""
        command = fight_resume_command(row)
        if self.closed or self.proc.poll() is not None or self.runs >= RUNS_PER_PROCESS:
            self.close()
            self._start()
        reply = self.send(command)
        if reply.get("code") == "run_not_ended":
            self.close()
            self._start()
            reply = self.send(command)
        self.runs += 1
        if reply.get("type") == "error":
            raise ProtocolError("resume_fight: " + str(reply.get("message") or reply))
        return reply if reply.get("type") == "decision_frame" else self.send({"cmd": "advance_to_boundary"})

    def enter_fight(self, commands):
        """Enter a fight again from its entry save: `commands` are the two that
        `saves.anchor_commands` gives. Returns the fight's first frame, as `reset` returns
        a run's; the process is reused or replaced as `reset` does."""
        install, room = commands
        if self.closed or self.proc.poll() is not None or self.runs >= RUNS_PER_PROCESS:
            self.close()
            self._start()
        reply = self.send(install)
        if reply.get("code") == "run_not_ended":
            self.close()
            self._start()
            reply = self.send(install)
        self.runs += 1
        if reply.get("type") != "anchor_installed":
            raise ProtocolError("enter_anchor: " + str(reply.get("message") or reply))
        frame = self.send(room)
        if frame.get("type") == "error":
            raise ProtocolError("anchor_room: " + str(frame.get("message") or frame))
        return frame if frame.get("type") == "decision_frame" else self.send({"cmd": "advance_to_boundary"})

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
