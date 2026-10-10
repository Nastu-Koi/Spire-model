"""Read-only training dashboard; deliberately independent of Torch.

It reads what `history.TrainingHistory` writes: the logs of a behavior-cloning run
(`bootstrap`) and of the iterations of policy improvement (`iterate`, whose rounds are
iterations and whose update records are those of its advantage-weighted updates)."""

import argparse
from array import array
from collections import deque
from datetime import UTC, datetime
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
import json
import math
import os
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid


WEB = Path(__file__).with_name("monitor_web")
KINDS = {"bootstrap", "iterate"}
# Complete records kept per update log: the page tabulates only this many.
RECENT = 200
# Dictionaries of numbers that become one curve per entry.
NESTED = {"learning_rates", "grad_norm"}
META = ("update", "round", "policy_version", "time", "stage", "session")
INTEGRAL = {"update", "round", "policy_version", "stage", "session"}
NAN = float("nan")


def finite_json(value):
    """Old logs can contain NaN; browsers need strict JSON and explicit gaps."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def validation_average(validation, a0=False):
    groups = [
        value for key, value in validation.items()
        if key.startswith("A0/") == a0 and isinstance(value, dict)
        and number(value.get("nll")) and number(value.get("branches"))
        and value["branches"] > 0
    ]
    branches = sum(value["branches"] for value in groups)
    return sum(value["nll"] for value in groups) / branches if branches else None


def normalize_record(raw, *, strict=False):
    """A point at the recorded optimizer update; old summaries stay sparse.

    `strict` says the caller already parsed NaN and Infinity away.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("kind"), str) or raw["kind"] not in KINDS:
        raise ValueError("Not a training record of a known kind")
    if not strict:
        raw = finite_json(raw)
    epochs = raw.get("metrics", {})
    if isinstance(epochs, dict):
        epochs = [epochs]
    if not isinstance(epochs, list) or any(not isinstance(item, dict) for item in epochs):
        raise ValueError("Metrics must be an object or a list of objects")
    validation = raw.get("validation") or {}
    config = raw.get("config") or {}
    if not all(isinstance(value, dict) for value in (validation, config)):
        raise ValueError("Invalid training metadata")
    metrics = dict(epochs[-1]) if epochs else {}
    metrics["validation_nll"] = validation_average(validation)
    metrics["a0_validation_nll"] = validation_average(validation, a0=True)
    metrics["duration_seconds"] = raw.get("duration_seconds", metrics.get("duration_seconds"))
    update = metrics.get("updates")
    update = update if number(update) and update >= 0 and int(update) == update else None
    return dict(
        kind=raw["kind"], session=raw.get("session"), round=raw.get("round"),
        update=update, record_type="update" if raw.get("record_type") == "update" else "summary",
        time=raw.get("time"), stage=raw.get("stage", raw["kind"]),
        policy_version=raw.get("policy_version"), metrics=metrics,
        optimization_epochs=len(epochs),
        validation=validation, config=config, raw=raw,
    )


def timestamp(value):
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def summary_row(record):
    """A round as the page tabulates it: no raw JSON, no repeated configuration."""
    return {
        **{key: value for key, value in record.items()
           if key not in {"raw", "config", "validation", "metrics"}},
        "metrics": {key: value for key, value in record["metrics"].items()
                    if not isinstance(value, (dict, list))},
    }


class Series:
    """Every point of a log as columns of doubles; NaN where a record has no value.

    A run is hundreds of thousands of updates. One double per value keeps them all
    in memory, and a client that says how many points it holds is sent only the rest.
    """

    def __init__(self, limit=None):
        self.limit = limit
        self.clear()

    def clear(self):
        self.meta = {name: array("d") for name in META}
        self.metrics = {}
        self.labels = {"stage": [], "session": []}
        self.length = 0  # retained points
        self.first = 0  # position of the oldest retained point in the whole log

    def label(self, name, value):
        table = self.labels[name]
        value = value if isinstance(value, str) else None
        if value not in table:
            table.append(value)
        return table.index(value)

    def append(self, record):
        point = dict(
            update=record["update"], round=record["round"],
            policy_version=record["policy_version"], time=timestamp(record["time"]),
            stage=self.label("stage", record["stage"]),
            session=self.label("session", record["session"]),
        )
        for name, column in self.meta.items():
            column.append(point[name] if number(point[name]) else NAN)
        scalars = {}
        for name, value in record["metrics"].items():
            # Nearly every value of a long log is a plain float: decide those without a call.
            if (value - value == 0 if type(value) is float else number(value)):
                scalars[name] = value
            elif name in NESTED and isinstance(value, dict):
                scalars.update((f"{name}.{part}", item) for part, item in value.items() if number(item))
        for name, column in self.metrics.items():
            column.append(scalars.pop(name, NAN))
        for name, value in scalars.items():
            # First appearance of a metric: every earlier point lacks it.
            self.metrics[name] = array("d", [NAN]) * self.length + array("d", [value])
        self.length += 1

    def trim(self):
        excess = self.length - self.limit if self.limit else 0
        if excess > 0:
            for column in (*self.meta.values(), *self.metrics.values()):
                del column[:excess]
            self.length -= excess
            self.first += excess

    def since(self, position):
        """Copies of the columns from `position` of the whole log onward."""
        start = min(max(position, self.first), self.first + self.length)
        offset = start - self.first
        return dict(
            start=start,
            meta={name: column[offset:] for name, column in self.meta.items()},
            metrics={name: column[offset:] for name, column in self.metrics.items()},
            labels={name: list(table) for name, table in self.labels.items()},
        )


def encode_series(window):
    """Columns as strict JSON: null for a missing value, integers where they are counts."""
    def values(name, column):
        if name in INTEGRAL:
            return [None if value != value else int(value) for value in column]
        if name == "time":
            return [None if value != value else round(value, 3) for value in column]
        return [None if value != value else value for value in column]

    return dict(
        window,
        meta={name: values(name, column) for name, column in window["meta"].items()},
        metrics={name: values(None, column) for name, column in window["metrics"].items()},
    )


class HistoryReader:
    """Incrementally consume complete JSONL lines; never repair the writer's file."""

    def __init__(self, limit=None, recent=None):
        # Every point stays as a row of the columns; `recent` bounds the complete records.
        self.records = deque(maxlen=min((n for n in (limit, recent) if n), default=None))
        self.series = Series(limit)
        self.offset = 0
        self.signature = None
        self.total = 0
        self.invalid = 0
        self.unplaced = 0
        self.pending = False
        self.anchor = b""
        self.generation = 0

    def reset(self):
        self.records.clear()
        self.series.clear()
        self.offset = self.total = self.invalid = self.unplaced = 0
        # Positions a client holds refer to the log that was replaced.
        self.generation += 1

    def update(self, path):
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if signature == self.signature:
            return
        # Replacement, truncation, or an in-place rewrite of the same length.
        if self.signature and (
            signature[:2] != self.signature[:2] or stat.st_size < self.signature[3]
            or stat.st_size == self.signature[3]
        ):
            self.reset()
        self.pending = False
        with path.open("rb") as stream:
            # A truncate-and-regrow can already be larger than the previous stat.
            # Check the consumed boundary before trusting the stored offset.
            stream.seek(max(0, self.offset - len(self.anchor)))
            if self.offset and stream.read(len(self.anchor)) != self.anchor:
                self.reset()
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    self.pending = True
                    break
                self.offset = stream.tell()
                if not line.strip():
                    continue
                try:
                    # Old logs wrote NaN; read as null it needs no second pass over the record.
                    record = normalize_record(json.loads(line, parse_constant=lambda _: None), strict=True)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    self.invalid += 1
                    continue
                self.records.append(record)
                self.series.append(record)
                self.total += 1
                self.unplaced += record["update"] is None
            stream.seek(max(0, self.offset - 64))
            self.anchor = stream.read(min(self.offset, 64))
        self.series.trim()
        self.signature = signature


class MonitorStore:
    def __init__(self, root, limit=None):
        self.root = Path(root).resolve()
        self.limit = limit
        self.lock = threading.Lock()
        self.readers = {}
        self.update_readers = {}
        self.directories = {}
        self.discovery_time = None
        self.discovery_errors = 0
        # A restarted service counts its positions afresh.
        self.boot = uuid.uuid4().hex[:8]

    def safe_file(self, directory, name):
        path = directory / name
        # Do not follow a log symlink into a checkpoint or outside the selected root.
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Log path is outside the monitored directory")
        return path

    def discover(self):
        now = time.monotonic()
        if self.discovery_time is not None and now - self.discovery_time < 5:
            return
        directories = {}
        self.discovery_errors = 0
        pending = [self.root]
        while pending:
            directory = pending.pop()
            try:
                # DirEntry avoids a stat per trajectory in large data directories.
                with os.scandir(directory) as entries:
                    children = list(entries)
                names = {entry.name for entry in children if not entry.is_symlink()}
                if {"history.jsonl", "updates.jsonl", "status.json"} & names:
                    key = directory.relative_to(self.root).as_posix()
                    directories[key] = directory
                pending.extend(
                    Path(entry.path) for entry in children
                    if not entry.name.startswith(".") and entry.name not in {
                        "node_modules", "__pycache__", "current"
                    } and entry.is_dir(follow_symlinks=False)
                )
            except OSError:
                self.discovery_errors += 1
        self.directories = directories
        self.readers = {key: reader for key, reader in self.readers.items() if key in directories}
        self.update_readers = {key: reader for key, reader in self.update_readers.items() if key in directories}
        self.discovery_time = now

    def read_status(self, directory):
        try:
            path = self.safe_file(directory, "status.json")
            if not path.is_file():
                return {}, []
            status = finite_json(json.loads(path.read_text(encoding="utf-8")))
            if not isinstance(status, dict):
                raise ValueError("Status must be an object")
            return status, []
        except (OSError, ValueError, RecursionError):
            return {}, ["状态文件暂时无法读取。"]

    def load(self, key):
        """Bring both logs of a run up to date; the curve comes from its update log
        once that holds a record, from the round summaries of older runs otherwise."""
        directory = self.directories[key]
        warnings = []
        for readers, name, recent in ((self.readers, "history.jsonl", None),
                                      (self.update_readers, "updates.jsonl", RECENT)):
            reader = readers.setdefault(key, HistoryReader(self.limit, recent))
            try:
                path = self.safe_file(directory, name)
                if path.is_file():
                    reader.update(path)
                elif reader.signature:
                    # The log vanished: whatever replaces it is not the one a client read.
                    generation = reader.generation + 1
                    reader = readers[key] = HistoryReader(self.limit, recent)
                    reader.generation = generation
            except (OSError, ValueError):
                warnings.append(f"{name} 暂时无法读取，保留上次读取的记录。")
            if reader.invalid:
                warnings.append(f"{name} 跳过 {reader.invalid} 条损坏或不支持的记录。")
            if reader.pending:
                warnings.append(f"{name} 最后一条记录尚未写完，等待下一次刷新。")
        updates = self.update_readers[key]
        reader = updates if updates.total else self.readers[key]
        if reader.unplaced:
            warnings.append(f"{reader.unplaced} 条记录缺少更新次数，不在更新曲线上绘制。")
        return reader, warnings

    def describe(self, key, *, load):
        """What the run list shows. A status file names the kind, so the logs of a
        run nobody has opened stay unread; older runs say it only in their records."""
        directory = self.directories[key]
        status, warnings = self.read_status(directory)
        kind, reader = status.get("kind"), None
        if load or not isinstance(kind, str) or kind not in KINDS:
            reader, more = self.load(key)
            warnings += more
        latest = reader.records[-1] if reader and reader.records else None
        if not isinstance(kind, str) or kind not in KINDS:
            kind = (latest or {}).get("kind")
        if kind not in KINDS:
            return None, reader
        return dict(
            id=key, name=directory.name, path=str(directory), kind=kind, status=status,
            updated_at=status.get("time") or (latest or {}).get("time"), warnings=warnings,
        ), reader

    def list_runs(self):
        with self.lock:
            self.discover()
            runs = [run for key in self.directories if (run := self.describe(key, load=False)[0])]
            runs.sort(key=lambda run: str(run["updated_at"] or ""), reverse=True)
            return dict(
                root=str(self.root), runs=runs, limit=self.limit,
                discovery_errors=self.discovery_errors,
                server_time=datetime.now(UTC).isoformat(),
            )

    def detail(self, key, *, after=0, summaries=0, token=None):
        """A run, with the points and rounds the client does not hold yet.

        `after` and `summaries` count what it holds of the curve and of the round
        summaries; they are trusted only with the `token` of the logs they count.
        """
        with self.lock:
            self.discover()
            if key not in self.directories:
                raise KeyError(key)
            run, reader = self.describe(key, load=True)
            if run is None:
                raise KeyError(key)
            history, updates = self.readers[key], self.update_readers[key]
            source = "updates" if reader is updates else "history"
            current = f"{self.boot}.{source}.{updates.generation}.{history.generation}"
            if token != current:
                after = summaries = 0
            window = reader.series.since(after)
            fresh = min(len(reader.records), RECENT, reader.total - window["start"])
            oldest = history.total - len(history.records)
            begin = min(max(summaries, oldest), history.total)
            run.update(
                source=source, token=current,
                total_records=reader.total, retained_records=reader.series.length,
                latest=reader.records[-1] if reader.records else None,
                summary=history.records[-1] if history.records else None,
                # Complete records, for the table: the newest the client lacks.
                recent=list(reader.records)[len(reader.records) - fresh:],
                summaries=dict(start=begin, total=history.total, records=[
                    summary_row(record) for record in islice(history.records, begin - oldest, None)
                ]),
                server_time=datetime.now(UTC).isoformat(),
            )
        # Encoding a long curve is the slow part; it needs no lock on the store.
        return dict(finite_json(run), series=encode_series(window))


def create_server(root, host="127.0.0.1", port=8765, limit=None):
    store = MonitorStore(root, limit)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, body, content_type):
            compressed = len(body) > 4096 and any(
                coding.strip() == "gzip" for coding in self.headers.get("Accept-Encoding", "").split(",")
            )
            if compressed:
                body = gzip.compress(body, compresslevel=1, mtime=0)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Vary", "Accept-Encoding")
            if compressed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                             "style-src 'self'; script-src 'self'; connect-src 'self'; "
                             "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, status, value):
            # The store hands out strict JSON; a long curve is not walked a second time.
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.respond(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/api/runs":
                    self.json(200, store.list_runs())
                elif url.path == "/api/run":
                    query = parse_qs(url.query)

                    def held(name):
                        value = query.get(name, ["0"])[0]
                        return int(value) if value.isascii() and value.isdigit() else 0

                    self.json(200, store.detail(
                        query.get("id", [""])[0], after=held("after"), summaries=held("summaries"),
                        token=query.get("token", [None])[0],
                    ))
                elif url.path in {"/", "/index.html", "/app.js", "/style.css", "/favicon.svg"}:
                    name = "index.html" if url.path == "/" else url.path[1:]
                    types = {".html": "text/html", ".js": "text/javascript",
                             ".css": "text/css", ".svg": "image/svg+xml"}
                    self.respond(200, (WEB / name).read_bytes(), types[Path(name).suffix] + "; charset=utf-8")
                else:
                    self.json(404, {"error": "未找到页面。"})
            except KeyError:
                self.json(404, {"error": "训练目录不存在或尚未写入有效记录。"})
            except (OSError, ValueError):
                self.json(503, {"error": "暂时无法读取训练数据，请稍后刷新。"})

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only Bootstrap training dashboard")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Training root or a single output directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=0, help="Optional record limit per run (default: 0, all history)")
    args = parser.parse_args(argv)
    if not args.runs.is_dir():
        parser.error(f"Training directory does not exist: {args.runs}")
    if args.limit < 0:
        parser.error("--limit must be zero (all history) or positive")
    try:
        server = create_server(args.runs, args.host, args.port, args.limit or None)
    except OSError as error:
        parser.error(str(error))
    print(f"Spire training monitor: http://{args.host}:{server.server_port}", flush=True)
    print(f"Reading {args.runs.resolve()} (read only). Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
