"""Per-update telemetry and durable per-round training summaries."""

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
        for name in ("history.jsonl", "updates.jsonl"):
            history = self.directory / name
            if history.exists():
                with history.open("rb+") as stream:
                    size = stream.seek(0, os.SEEK_END)
                    start, tail = size, b""
                    while start and b"\n" not in tail:
                        length = min(start, 4096)
                        start -= length
                        stream.seek(start)
                        tail = stream.read(length) + tail
                    tail = tail.rsplit(b"\n", 1)[-1]
                    if tail:
                        try:
                            json.loads(tail)
                        except (ValueError, UnicodeDecodeError):
                            stream.truncate(size - len(tail))
                        else:
                            stream.seek(0, os.SEEK_END)
                            stream.write(b"\n")
        self.kind = kind
        self.session = uuid.uuid4().hex
        self.started = time.monotonic()
        self.config = config
        self.updates = None
        self.status("starting")

    def status(self, state, **details):
        if "updates" in details:
            self.updates = details["updates"]
        elif self.updates is not None:
            details["updates"] = self.updates
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
        record = self._record(metrics, round=round_number, **details)
        self._write("history.jsonl", record, durable=True)
        last = metrics[-1] if isinstance(metrics, list) and metrics else metrics
        if isinstance(last, dict):
            self.updates = last.get("updates", self.updates)
        self.status("running", round=round_number)
        return record

    def append_update(self, metrics, **details):
        """Flush each successful optimizer step without fsyncing the GPU loop."""
        self.updates = metrics["updates"]
        record = self._record(metrics, record_type="update", **details)
        self._write("updates.jsonl", record, durable=False)
        self.status("updating", **{key: details[key] for key in ("round", "stage") if key in details})
        return record

    def _record(self, metrics, **details):
        return dict(
            kind=self.kind,
            session=self.session,
            time=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - self.started,
            config=self.config,
            metrics=metrics,
            **details,
        )

    def _write(self, name, record, *, durable):
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
