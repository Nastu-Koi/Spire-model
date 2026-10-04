import gzip
import json

import pytest

from spire_codex_data.fetch import download, unpack


def run():
    return dict(seed='SEED', build_id='v0.111.0', ascension=0, start_time=10,
                win=True, was_abandoned=False, game_mode='standard', modifiers=[],
                players=[dict(character='CHARACTER.IRONCLAD', deck=[dict(id='FUTURE_CARD')])],
                map_point_history=[[dict(rooms=[dict(room_type='shop')], player_stats=[dict(
                    current_gold=99, current_hp=50, card_choices=[dict(card=dict(id='CARD.A'), was_picked=False)])])]])


def journal():
    return [dict(t='header', s=0, seed='SEED', ascension=0, character='IRONCLAD', start_time=10,
                 build_id='v0.111.0', replay_version=7, starting_deck=[], rng_state={'secret': 42},mods=[],harmony_owners=[]),
            dict(t='decision', s=1, act=1, floor=1, decision_id=1, decision_type='event',
                 options=[dict(option_index=0, option_id='A', presented=True, selectable=True),
                          dict(option_index=1, option_id='B', presented=True, selectable=True)],
                 n_presented=2, n_selectable=2),
            dict(t='outcome', s=2, decision_id=1, option_index=1),
            dict(t='gold', s=3, gold=999), dict(t='end', s=4, capture_status='complete')]


def test_bounded_gzip_and_plain_transport():
    assert unpack(b'{}') == unpack(gzip.compress(b'{}')) == b'{}'
    with pytest.raises(ValueError):
        unpack(gzip.compress(b'a' * 100), limit=10)


def test_both_sources_merge_cache_without_duplicate_download(tmp_path):
    class Client:
        def __init__(self):
            self.calls = []
        def get(self, path):
            self.calls.append(path)
            if '?' in path:
                return json.dumps(dict(runs=[dict(run_hash='a'*16, has_replay=True)], total=1, total_pages=1)).encode()
            if path.endswith('/replay'):
                return b'{"t":"header"}\n'
            return json.dumps(run()).encode()
    c = Client()
    download(tmp_path, source='replays', client=c)
    download(tmp_path, source='runs', client=c)
    assert sum(p.endswith('/replay') for p in c.calls) == 1
    cached = tmp_path / 'raw' / (('a'*16) + '.replay.jsonl')
    cached.write_bytes(b'corrupt')
    download(tmp_path, source='runs', client=c)
    assert sum(p.endswith('/replay') for p in c.calls) == 2


class FakeSite:
    """Listing rows per character; assets fail by run_hash until told otherwise."""
    def __init__(self, rows, total=None):
        self.rows, self.total, self.calls, self.broken = rows, total, [], {}
    def get(self, path):
        self.calls.append(path)
        if '?' in path:
            from urllib.parse import parse_qs
            query = {k: v[0] for k, v in parse_qs(path.split('?', 1)[1]).items()}
            rows = [r for r in self.rows if query.get('character') in (None, r['character'])]
            page = int(query['page'])
            return json.dumps(dict(runs=rows[(page-1)*2:page*2], total=self.total or len(rows),
                                   total_pages=max(1, -(-len(rows)//2)))).encode()
        key = path.split('/')[-1]
        if key in self.broken:
            raise self.broken[key]
        row = next(r for r in self.rows if r['run_hash'] == key)
        return json.dumps(dict(run(), ascension=row['ascension'],
                               players=[dict(character='CHARACTER.' + row.get('actual', row['character']))])).encode()


def site_rows():
    return [dict(run_hash=f'{i:016x}', character=c, ascension=i % 3)
            for i, c in enumerate(['IRONCLAD'] * 3 + ['SILENT'] * 3 + ['DEFECT'])]


def test_run_summaries_are_sliced_by_character(tmp_path):
    site = FakeSite(site_rows())
    rows, errors = download(tmp_path, source='runs', client=site)
    assert len(rows) == 7 and not errors
    listings = [p for p in site.calls if '?' in p]
    assert all('character=' in p for p in listings) and len(listings) == 2 + 2 + 1 + 1 + 1
    manifest = json.loads((tmp_path / 'download-runs.json').read_text())
    assert manifest['slices']['IRONCLAD'] == dict(total=3, total_pages=2, last_page=2)
    assert (tmp_path / 'pages' / 'runs' / 'SILENT' / '00002.json').exists()


def test_slices_walked_in_parallel_share_one_manifest(tmp_path):
    site = FakeSite(site_rows())
    rows, errors = download(tmp_path, source='runs', client=site, parallel=5)
    manifest = json.loads((tmp_path / 'download-runs.json').read_text())
    assert len(rows) == 7 == len(manifest['runs']) and not errors
    assert set(manifest['slices']) == {'IRONCLAD', 'SILENT', 'DEFECT', 'NECROBINDER', 'REGENT'}
    # A slice that fails outright ends the call, but only after the others have finished.
    site = FakeSite(site_rows())
    listing = site.get
    site.get = lambda path: (_ for _ in ()).throw(ValueError('boom')) if 'character=SILENT' in path else listing(path)
    with pytest.raises(ValueError, match='boom'):
        download(tmp_path / 'again', source='runs', client=site, parallel=5)
    saved = json.loads((tmp_path / 'again' / 'download-runs.json').read_text())
    assert {r['listing']['character'] for r in saved['runs']} == {'IRONCLAD', 'DEFECT'}


def test_capped_listing_is_refused_instead_of_silently_truncated(tmp_path):
    with pytest.raises(ValueError, match='cap'):
        download(tmp_path, source='runs', client=FakeSite(site_rows(), total=10000), characters=(None,))


def test_failures_persist_until_the_run_downloads_and_do_not_stop_the_walk(tmp_path):
    from http.client import IncompleteRead
    site = FakeSite(site_rows())
    site.broken = {f'{1:016x}': IncompleteRead(b''), f'{4:016x}': EOFError('truncated gzip')}
    rows, errors = download(tmp_path, source='runs', client=site)
    assert len(rows) == 5 and {e['run_hash'] for e in errors} == set(site.broken)
    del site.broken[f'{1:016x}']
    rows, errors = download(tmp_path, source='runs', client=site, characters=('IRONCLAD',))
    manifest = json.loads((tmp_path / 'download-runs.json').read_text())
    # The unvisited SILENT failure is still on record; the repaired one is gone.
    assert not errors and [e['run_hash'] for e in manifest['errors']] == [f'{4:016x}']
    assert len(manifest['runs']) == 6


def test_a_listing_that_stops_answering_is_waited_out(tmp_path, monkeypatch):
    from spire_codex_data import fetch
    site, waits, down = FakeSite(site_rows()), [], [3]
    answer = site.get
    def get(path):
        if '?' in path and 'page=2' in path and down[0]:
            down[0] -= 1
            raise TimeoutError('The read operation timed out')
        return answer(path)
    site.get = get
    monkeypatch.setattr(fetch.time, 'sleep', waits.append)
    rows, errors = download(tmp_path, source='runs', client=site, characters=('IRONCLAD',))
    assert len(rows) == 3 and not errors and waits == list(fetch.LISTING_WAITS[:3])
    # Longer than every wait: the slice ends with the failure instead of looping for ever.
    down[0] = 99
    with pytest.raises(TimeoutError):
        download(tmp_path / 'again', source='runs', client=site, characters=('IRONCLAD',))


def test_max_runs_counts_requests_and_spreads_over_slices(tmp_path):
    site = FakeSite(site_rows())
    download(tmp_path, source='runs', client=site, max_runs=3, characters=('IRONCLAD', 'SILENT', 'DEFECT'))
    cached = sorted(p.name[:16] for p in (tmp_path / 'raw').glob('*.run.json'))
    assert cached == [f'{i:016x}' for i in (0, 3, 6)]
    before = len(site.calls)
    download(tmp_path, source='runs', client=site, max_runs=2, characters=('IRONCLAD', 'SILENT', 'DEFECT'))
    # Cached runs are free: the second call reaches new ones.
    assert len(list((tmp_path / 'raw').glob('*.run.json'))) == 5
    assert sum('?' not in p for p in site.calls[before:]) == 2


def test_resume_restarts_each_slice_at_its_recorded_page(tmp_path):
    site = FakeSite(site_rows())
    download(tmp_path, source='runs', client=site, max_runs=3, characters=('IRONCLAD',))
    before = len(site.calls)
    download(tmp_path, source='runs', client=site, characters=('IRONCLAD',), resume=True)
    listings = [p for p in site.calls[before:] if '?' in p]
    assert len(listings) == 1 and 'page=2' in listings[0]
    assert len(list((tmp_path / 'raw').glob('*.run.json'))) == 3


def test_run_document_must_agree_with_its_listing_row(tmp_path):
    rows = site_rows()
    rows[0]['actual'] = 'REGENT'
    _, errors = download(tmp_path, source='runs', client=FakeSite(rows), characters=('IRONCLAD',))
    assert [e['run_hash'] for e in errors] == [rows[0]['run_hash']] and 'character' in errors[0]['error']
    assert not (tmp_path / 'raw' / f'{rows[0]["run_hash"]}.run.json').exists()


def test_transport_retries_cover_a_body_cut_off_mid_read(monkeypatch):
    from http.client import IncompleteRead
    from spire_codex_data import fetch
    attempts = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self, limit):
            attempts.append(limit)
            if len(attempts) == 1:
                raise IncompleteRead(b'')
            return gzip.compress(b'{}')[:-4] if len(attempts) == 2 else b'{}'
    monkeypatch.setattr(fetch, 'urlopen', lambda request, timeout: Response())
    monkeypatch.setattr(fetch.time, 'sleep', lambda seconds: None)
    assert fetch.Client(interval=0).get('/x') == b'{}' and len(attempts) == 3


def test_threads_share_one_request_budget(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from spire_codex_data import fetch
    starts = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self, limit): return b'{}'
    monkeypatch.setattr(fetch, 'urlopen', lambda request, timeout: starts.append(fetch.time.monotonic()) or Response())
    client = fetch.Client(interval=0.05)
    with ThreadPoolExecutor(5) as pool:
        list(pool.map(client.get, ['/x'] * 10))
    gaps = [b - a for a, b in zip(sorted(starts), sorted(starts)[1:])]
    # Ten requests from five threads still leave one interval between any two.
    assert len(starts) == 10 and min(gaps) >= 0.04
    # Routes are paced apart: the site counts each on its own.
    client = fetch.Client()
    assert client.route('/api/runs/shared/abc') != client.route('/api/runs/abc/replay') != client.route('/api/runs/list?page=1')
    started = fetch.time.monotonic()
    monkeypatch.setattr(fetch.time, 'sleep', lambda seconds: starts.append(('slept', seconds)))
    for path in ('/api/runs/shared/a', '/api/runs/a/replay', '/api/runs/list?page=1', '/api/runs/shared/b'):
        client.get(path)
    waits = [x[1] for x in starts if isinstance(x, tuple)]
    # Only the second summary waits, for a little over a second.
    assert [round(w) for w in waits] == [0, 0, 0, 1] and waits[3] > 1.0


def test_legacy_cache_without_receipt_is_adopted_only_after_validation(tmp_path):
    from spire_codex_data.fetch import fetch_asset
    key = 'b' * 16
    (tmp_path / f'{key}.run.json').write_text(json.dumps(run()))
    site = FakeSite([])
    digest, asked = fetch_asset(site, tmp_path, key, 'run.json', adopt=True)
    assert not asked and not site.calls and (tmp_path / f'{key}.run.json.sha256').read_text() == digest
    (tmp_path / ('c' * 16 + '.run.json')).write_text(json.dumps(dict(run(), win=False)))
    with pytest.raises(ValueError, match='win'):
        fetch_asset(site, tmp_path, 'c' * 16, 'run.json', adopt=True)
