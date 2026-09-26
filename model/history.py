"""Durable per-round records shared by trainers and the monitoring page."""

import fcntl
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path


class TrainingHistory:
    def __init__(self, directory, kind, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".training.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise ValueError(
                f"Another trainer is writing to {self.directory}"
            ) from None
        # A killed append may leave an incomplete final JSON line. Preserve all
        # complete records, discard only that torn tail before appending again.
        history = self.directory / "history.jsonl"
        if history.exists():
            with history.open("rb+") as stream:
                raw = stream.read()
                if raw and not raw.endswith(b"\n"):
                    tail = raw.rsplit(b"\n", 1)[-1]
                    try:
                        json.loads(tail)
                    except (ValueError, UnicodeDecodeError):
                        stream.truncate(len(raw) - len(tail))
                    else:
                        stream.write(b"\n")
        self.kind = kind
        self.session = uuid.uuid4().hex
        self.started = time.monotonic()
        self.config = config
        self.status("starting")

    def status(self, state, **details):
        value = dict(
            kind=self.kind,
            session=self.session,
            pid=os.getpid(),
            state=state,
            time=datetime.now(UTC).isoformat(),
            **details,
        )
        target = self.directory / "status.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
        temporary.replace(target)
        if state in {"completed", "failed", "interrupted"}:
            self.close()

    def close(self):
        if not self.lock.closed:
            self.lock.close()

    def append(self, round_number, metrics, **details):
        record = dict(
            kind=self.kind,
            session=self.session,
            round=round_number,
            time=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - self.started,
            config=self.config,
            metrics=metrics,
            **details,
        )
        with (self.directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.status("running", round=round_number)
        return record
