"""Public replay downloader: bounded requests, resumable atomic cache, no auth."""
import gzip
import hashlib
from http.client import HTTPException
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://spire-codex.com'
FILTERS = dict(win='true', build_id='v0.111.0', players='single', game_mode='standard')
CHARACTERS = ('IRONCLAD', 'SILENT', 'DEFECT', 'NECROBINDER', 'REGENT')
ENDPOINTS = dict(replays='/api/replays', runs='/api/runs/list')
# Listings never report more than this many rows; a slice at the cap is truncated.
LISTING_CAP = 10000
# Transport failures worth another attempt. IncompleteRead and a truncated gzip
# stream are neither OSError nor ValueError.
TRANSIENT = (URLError, TimeoutError, ConnectionError, HTTPException, EOFError)
FAILURES = (OSError, ValueError, HTTPException, EOFError)


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def save_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def unpack(raw, limit=32 * 1024 * 1024):
    if raw.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Response exceeds decompressed size limit')
    return raw


# Requests a minute the site allows one client on each route (its X-RateLimit-Limit).
# The routes are counted separately; anything else gets the stricter value.
ROUTE_LIMITS = {'/api/runs/shared/': 60, '/api/runs/list': 120, '/api/replays': 120, '/replay': 120}
# Seconds to wait for a response. A listing is one database query over every run and
# can take most of a minute; a single run answers in about one second.
TIMEOUTS = {'/api/runs/list': 180, '/api/replays': 180}
# Pauses before asking for a listing page again once the client's own retries are spent.
# A slice has nothing else to do without its next page, so it waits the site out.
LISTING_WAITS = (60, 300, 900, 1800)


class Client:
    """Requests paced per route, shared by every thread that uses the client.

    Each route gets one slot every `60 / limit` seconds, stretched by `margin` so the
    site's own counter is never reached. Threads let slow responses overlap; they do
    not raise the rate. `interval` fixes one pace for every route instead.
    """

    def __init__(self, base=BASE, interval=None, retries=4, margin=1.05):
        self.base, self.interval, self.retries, self.margin = base.rstrip('/'), interval, retries, margin
        self.next_request = {}
        self.lock = threading.Lock()

    def route(self, path):
        return next((r for r in ROUTE_LIMITS if r in path.split('?')[0]), '')

    def reserve(self, route):
        pace = self.interval if self.interval is not None else 60 / ROUTE_LIMITS.get(route, 60) * self.margin
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_request.get(route, 0.0))
            self.next_request[route] = start + pace
        time.sleep(start - now)

    def hold(self, route, seconds):
        """Keep every thread off a route until `seconds` from now."""
        with self.lock:
            self.next_request[route] = max(self.next_request.get(route, 0.0), time.monotonic() + seconds)

    def get(self, path):
        route = self.route(path)
        for attempt in range(self.retries + 1):
            self.reserve(route)
            try:
                request = Request(self.base + path, headers={
                    'User-Agent': 'Spire-model-dataset/1.0', 'Accept-Encoding': 'gzip'})
                with urlopen(request, timeout=TIMEOUTS.get(route, 45)) as response:
                    raw = response.read(32 * 1024 * 1024 + 1)
                # Inside the retry: a cut-off body only shows up while unpacking.
                return unpack(raw)
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
                retry = exc.headers.get('Retry-After', '')
                self.hold(route, min(float(retry) if retry.isdigit() else 2 ** (attempt + 1), 300))
            except TRANSIENT:
                if attempt == self.retries:
                    raise
                self.hold(route, 2 ** (attempt + 1))
        raise AssertionError('unreachable')


def ident(value):
    return value.split('.', 1)[-1] if isinstance(value, str) else value


def check_run(run, row=None, character=None):
    """Reject a run document that contradicts the request or its listing row."""
    if not isinstance(run, dict):
        raise ValueError('Invalid run document')
    players = run.get('players')
    actual = dict(build_id=str(run.get('build_id', '')).split('+')[0], win=run.get('win') is True,
                  game_mode=run.get('game_mode'), players=len(players) if isinstance(players, list) else None,
                  character=ident(players[0].get('character')) if players and isinstance(players[0], dict) else None,
                  ascension=run.get('ascension'))
    expected = dict(build_id=FILTERS['build_id'], win=True, game_mode=FILTERS['game_mode'], players=1)
    if character:
        expected['character'] = character
    for name in ('character', 'ascension'):
        if (row or {}).get(name) is not None:
            if expected.setdefault(name, row[name]) != row[name]:
                raise ValueError(f'Listing row contradicts request: {name}')
    wrong = sorted(name for name, value in expected.items() if actual[name] != value)
    if wrong:
        raise ValueError('Run document contradicts listing: ' + ','.join(wrong))


def check_journal(data):
    records = [json.loads(line) for line in data.splitlines() if line.strip()]
    if not records or records[0].get('t') != 'header':
        raise ValueError('Missing replay header')


def fetch_asset(client, raw_dir, key, suffix, *, row=None, character=None, adopt=False):
    """Return (sha256, requested). Content is validated before it enters the cache.

    adopt: accept an existing file that has no receipt (caches written before
    receipts existed) instead of downloading it again; it is still validated.
    """
    target = Path(raw_dir) / f'{key}.{suffix}'
    receipt = target.with_suffix(target.suffix + '.sha256')
    data = target.read_bytes() if target.exists() else None
    if data is not None and receipt.exists() and hashlib.sha256(data).hexdigest() == receipt.read_text().strip():
        return hashlib.sha256(data).hexdigest(), False
    requested = not (adopt and data is not None and not receipt.exists())
    if requested:
        data = client.get(f'/api/runs/{key}/replay' if suffix == 'replay.jsonl' else f'/api/runs/shared/{key}')
    if suffix == 'replay.jsonl':
        check_journal(data)
    else:
        check_run(json.loads(data), row, character)
    digest = hashlib.sha256(data).hexdigest()
    if requested:
        atomic(target, data)
    atomic(receipt, digest.encode())
    return digest, requested


def list_page(client, source, page, character=None):
    query = dict(FILTERS, limit=100, page=page)
    if character:
        query['character'] = character
    listing = json.loads(client.get(ENDPOINTS[source] + '?' + urlencode(query)))
    if not isinstance(listing, dict) or not isinstance(listing.get('runs'), list):
        raise ValueError('Invalid listing page')
    if listing.get('total', 0) >= LISTING_CAP:
        raise ValueError(f'Listing {source}/{character or "all"} reports {listing["total"]} rows: '
                         f'at the {LISTING_CAP}-row cap, so it is truncated; narrow the slice')
    for row in listing['runs']:
        if not re.fullmatch('[a-f0-9]{16,64}', str(row.get('run_hash'))):
            raise ValueError('Unsafe run_hash')
    return listing


def download(output, *, max_runs=0, start_page=1, source="replays", client=None, characters=None, resume=False,
             parallel=1):
    """Walk every listing slice; return (rows, errors) handled by this call.

    Run summaries are sliced by character because the unsliced listing is
    capped. max_runs bounds the runs that needed a request in this call, split
    evenly over the slices; cached runs cost no request. resume restarts each
    slice at its last recorded page: listings are newest-first, so rows only
    drift to later pages, and submissions made since need a full rescan.
    parallel walks that many slices at once on one shared request budget per route:
    the site caps requests per minute, so this overlaps slow responses and gains no more.
    """
    root = Path(output)
    raw_dir = root / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    if characters is None:
        characters = CHARACTERS if source == 'runs' else (None,)
    manifest_path = root / f'download-{source}.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    merged = {r['run_hash']: r for r in previous.get('runs', [])}
    # Failures stay on record across calls until that run downloads cleanly.
    failed = {e['run_hash']: e for e in previous.get('errors', [])}
    slices = dict(previous.get('slices', {}))
    rows, errors, seen = [], [], set()
    lock = threading.Lock()  # one manifest, written by whichever slice finishes a page
    share, extra = divmod(max_runs, len(characters))
    quotas = [share + (i < extra) for i in range(len(characters))]

    def persist():
        save_json(manifest_path, dict(filters=FILTERS, slices=slices, runs=list(merged.values()),
                                      errors=list(failed.values())))

    def walk(character, quota, client):
        name = character or 'all'
        used, page = 0, max(start_page, slices.get(name, {}).get('last_page', 1) if resume else 1)
        while not (max_runs and used >= quota):
            for pause in LISTING_WAITS:
                try:
                    listing = list_page(client, source, page, character)
                    break
                except TRANSIENT:
                    time.sleep(pause)
            else:
                listing = list_page(client, source, page, character)
            save_json(root / 'pages' / source / (character or '') / f'{page:05}.json', listing)
            with lock:
                slices[name] = dict(total=listing['total'], total_pages=listing.get('total_pages'), last_page=page)
            for row in listing['runs']:
                key = row['run_hash']
                with lock:
                    if key in seen:
                        continue
                    seen.add(key)
                asked = False
                try:
                    checksums = {}
                    assets = ['run.json'] + (['replay.jsonl'] if row.get('has_replay') or source == 'replays' else [])
                    for suffix in assets:
                        checksums[suffix], fresh = fetch_asset(client, raw_dir, key, suffix, row=row, character=character)
                        asked |= fresh
                    record = dict(run_hash=key, sha256=checksums, listing=row)
                    with lock:
                        rows.append(record)
                        merged[key] = record
                        failed.pop(key, None)
                except FAILURES as exc:
                    asked = True
                    with lock:
                        failed[key] = dict(run_hash=key, error=f'{type(exc).__name__}: {exc}', slice=name)
                        errors.append(failed[key])
                used += asked
                if max_runs and used >= quota:
                    break
            with lock:
                persist()
            if page >= listing.get('total_pages', page) or not listing['runs']:
                break
            page += 1

    client = client or Client()
    try:
        if parallel > 1 and len(characters) > 1:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                jobs = [pool.submit(walk, character, quota, client) for character, quota in zip(characters, quotas)]
            for job in jobs:
                job.result()  # a slice that could not go on stops the call, after the others end
        else:
            for character, quota in zip(characters, quotas):
                walk(character, quota, client)
    finally:
        with lock:
            persist()
    return rows, errors
