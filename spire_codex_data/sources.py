"""Which cached runs may become summary anchors, and how far each is audited.

A run summary carries no mod list, console log or save history. Where a replay
of the run is cached, its headers and commands are checked; a summary without
one is usable but stays marked as unverified. Content outside the native game
is rejected either way.

python -m spire_codex_data.sources --existing data/spire-codex --output data/spire-codex/sources \
    --catalog data/spire-codex/catalog.json --per-character 20 --fetch
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .fetch import CHARACTERS, FAILURES, FILTERS, Client, fetch_asset, ident, list_page, save_json
from .integrity import inspect_journal

# Recording infrastructure only. Self-declared gameplay flags alone are not a
# trust boundary; an unknown UI/helper mod can still patch game logic.
ALLOWED_RECORDER_MODS = {'BaseLib', 'SpireCodex'}
ALLOWED_HARMONY_OWNERS = {'BaseLib', 'PostModInit', 'SpireCodex'}
# Gaps in what a source can prove. They leave it usable but unverified.
UNKNOWN = {'no_action_journal', 'mod_manifest_missing_or_invalid', 'patch_manifest_missing_or_invalid',
           'resume_requires_native_reconciliation'}
TYPES = {'ACT', 'CARD', 'RELIC', 'POTION', 'CHARACTER', 'ENCOUNTER', 'MONSTER', 'EVENT', 'ENCHANTMENT'}


def split_group(seed):
    """Runs of one seed share their map and offers: they must stay on one side of a split."""
    value = str(seed).upper().replace('O', '0').replace('I', '1').strip()
    return hashlib.sha256(value.encode()).hexdigest()


def run_reasons(run, header=None):
    reasons = []
    if str(run.get('build_id', '')).split('+')[0] != FILTERS['build_id']:
        reasons.append('wrong_build')
    if run.get('win') not in (True, 1) or run.get('was_abandoned'):
        reasons.append('not_victory')
    if run.get('game_mode') != 'standard' or run.get('modifiers'):
        reasons.append('not_standard')
    if len(run.get('players', [])) != 1:
        reasons.append('not_single_player')
    elif ident(run['players'][0].get('character')) not in CHARACTERS:
        reasons.append('unsupported_character')
    if header:
        for name in ('seed', 'ascension', 'start_time'):
            if str(run.get(name)) != str(header.get(name)):
                # Older journals recorded the hashed numeric seed; do not
                # silently treat that number as the original textual seed.
                reasons.append(f'header_mismatch:{name}')
        if ident(run['players'][0].get('character')) != ident(header.get('character')):
            reasons.append('header_mismatch:character')
        if str(header.get('build_id', '')).split('+')[0] != FILTERS['build_id']:
            reasons.append('header_mismatch:build')
        if header.get('replay_version') not in range(1, 8):
            reasons.append('unsupported_replay_version')
        mods = header.get('mods')
        if not isinstance(mods, list):
            reasons.append('mod_manifest_missing_or_invalid')
        else:
            if any(isinstance(m, dict) and m.get('affects_gameplay') is True for m in mods):
                reasons.append('gameplay_mods')
            if any(not isinstance(m, dict) or type(m.get('affects_gameplay')) is not bool for m in mods):
                reasons.append('mod_gameplay_effect_unknown')
            if any(not isinstance(m, dict) or m.get('id') not in ALLOWED_RECORDER_MODS for m in mods):
                reasons.append('unapproved_mods')
        owners = header.get('harmony_owners')
        if not isinstance(owners, list):
            reasons.append('patch_manifest_missing_or_invalid')
        elif any(not isinstance(owner, str) or owner not in ALLOWED_HARMONY_OWNERS for owner in owners):
            reasons.append('unapproved_harmony_patches')
    return reasons


def source_reasons(run, journal=None):
    """Everything the cached evidence holds against a run, or cannot show for it."""
    reasons = run_reasons(run, journal[0] if journal else None)
    if not journal:
        return reasons + ['no_action_journal']
    # A player can change mods between sessions: audit every segment's manifest.
    for segment in journal[1:]:
        if segment.get('t') == 'header':
            reasons.extend(r for r in run_reasons(run, segment) if r not in reasons)
    reasons.extend(inspect_journal(journal)['reasons'])
    order = [row.get('s') for row in journal]
    if any(type(s) is not int for s in order) or any(a >= b for a, b in zip(order, order[1:])):
        reasons.append('non_monotonic_sequence')
    if journal[-1].get('t') != 'end' or journal[-1].get('capture_status') not in ('complete', None):
        reasons.append('capture_incomplete')
    return reasons


def standard_acts(acts):
    # v0.111.0 ModelDb.ActsByIndex: Overgrowth/Underdocks, Hive, Glory.
    return (isinstance(acts, list) and len(acts) == 3 and acts[0] in ('ACT.OVERGROWTH', 'ACT.UNDERDOCKS')
            and acts[1:] == ['ACT.HIVE', 'ACT.GLORY'])


def audit(run, journal, catalog):
    """(reasons to reject, integrity). Integrity is `audited_recording` only when a
    replay exists and leaves nothing open; otherwise `unverified_source`."""
    reasons = source_reasons(run, journal)
    rejected = set(reasons) - UNKNOWN

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and value.split('.')[0] in TYPES and value not in catalog:
            rejected.add('unknown_native_content:' + value)
    visit(run)
    if type(run.get('ascension')) is not int or not 0 <= run['ascension'] <= 10:
        rejected.add('invalid_ascension')
    if not standard_acts(run.get('acts')):
        rejected.add('unsupported_acts')
    return sorted(rejected), 'unverified_source' if set(reasons) & UNKNOWN else 'audited_recording'


def collect(output, existing, catalog_path, per_character=20, fetch=False, client=None):
    """A manifest of accepted runs, one per seed and at most `per_character` per
    character (0: every accepted run).

    Runs the manifest already holds are considered first, so a manifest rebuilt
    while the cache grows only gains runs: what was generated from it stays valid.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    limit = per_character or float('inf')
    earlier = output / 'manifest.json'
    kept = [Path(s['path']) for s in json.loads(earlier.read_text())['selected']] if earlier.exists() else []
    catalog = {e.get('content_id') for e in json.loads(Path(catalog_path).read_text())['public']['entities']}
    chosen = {c: [] for c in CHARACTERS}
    seen, seeds, rejected, errors = set(), set(), {}, {}
    client = client or Client()

    def consider(path):
        key = path.name.removesuffix('.run.json')
        if key in seen:
            return
        run = json.loads(path.read_text())
        character = ident(run.get('players', [{}])[0].get('character'))
        if character not in CHARACTERS or len(chosen[character]) >= limit:
            return
        replay = path.with_name(key + '.replay.jsonl')
        # An advertised journal is evidence: don't silently ignore it.
        if run.get('has_replay') and not replay.exists():
            if not fetch:
                errors[key] = 'advertised_journal_not_cached'
                return
            replay = output / 'raw' / replay.name
            fetch_asset(client, replay.parent, key, 'replay.jsonl', adopt=True)
        journal = [json.loads(line) for line in replay.read_text().splitlines() if line.strip()] if replay.exists() else None
        if journal and journal[0].get('t') != 'header':
            raise ValueError('Invalid journal header')
        bad, integrity = audit(run, journal, catalog)
        seen.add(key)
        if bad:
            rejected[key] = bad
            return
        group = split_group(run['seed'])
        if (character, group) in seeds:
            rejected[key] = ['duplicate_character_seed']
            return
        seeds.add((character, group))
        evidence = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()}
        if replay.exists():
            evidence[str(replay.resolve())] = hashlib.sha256(replay.read_bytes()).hexdigest()
        chosen[character].append(dict(run_hash=key, character=character, path=str(path.resolve()),
            sha256=evidence[str(path.resolve())], integrity=integrity, evidence_sha256=evidence,
            ascension=run['ascension'], split_group=group))

    def persist():
        selected = [chosen[c][i] for i in range(max(map(len, chosen.values()))) for c in CHARACTERS
                    if i < len(chosen[c])]
        result = dict(method='summary-sources-v1', filters=FILTERS, requested_per_character=per_character,
            selected=selected, counts={c: len(v) for c, v in chosen.items()}, rejected=rejected, errors=errors,
            catalog_sha256=hashlib.sha256(Path(catalog_path).read_bytes()).hexdigest(),
            integrity_counts=dict(Counter(s['integrity'] for s in selected)))
        save_json(output / 'manifest.json', result)
        return result

    cached = [path for root in dict.fromkeys([Path(existing) / 'raw', output / 'raw'])
              for path in sorted(root.glob('*.run.json'))]
    for path in [p for p in kept if p.exists()] + cached:
        try:
            consider(path)
        except FAILURES as exc:
            errors[path.name.removesuffix('.run.json')] = f'{type(exc).__name__}: {exc}'
    if fetch:
        for character in CHARACTERS:
            page = 1
            while len(chosen[character]) < limit:
                listing = list_page(client, 'runs', page, character)
                save_json(output / 'pages' / character / f'{page:05}.json', listing)
                for row in listing['runs']:
                    key = row['run_hash']
                    if key in seen:
                        continue
                    path = output / 'raw' / f'{key}.run.json'
                    try:
                        # Same validation and receipts as the bulk downloader.
                        fetch_asset(client, path.parent, key, 'run.json', row=row, character=character, adopt=True)
                        consider(path)
                    except FAILURES as exc:
                        errors[key] = f'{type(exc).__name__}: {exc}'
                    if len(chosen[character]) >= limit:
                        break
                persist()
                if page >= listing.get('total_pages', page) or not listing['runs']:
                    break
                page += 1
    return persist()


def write_catalog(path, config=None):
    """The native content catalog of the configured game build."""
    from combat_solver_cli.client import DEFAULT_CONFIG, SolverEngine
    with SolverEngine(config or DEFAULT_CONFIG) as engine:
        catalog = engine.send(dict(cmd='public_catalog'))
    if not isinstance(catalog.get('public', {}).get('entities'), list):
        raise ValueError('public_catalog failed: ' + str(catalog.get('message')))
    save_json(path, catalog)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--existing', type=Path, required=True, help='Data directory whose raw/ holds cached runs')
    parser.add_argument('--catalog', type=Path, required=True, help='Native content catalog; written first if missing')
    parser.add_argument('--per-character', type=int, default=20, help='0 keeps every accepted run')
    parser.add_argument('--fetch', action='store_true', help='Download more runs until every character is filled')
    args = parser.parse_args()
    if args.per_character < 0:
        parser.error('per-character must not be negative')
    if not args.catalog.exists():
        write_catalog(args.catalog)
    result = collect(args.output, args.existing, args.catalog, args.per_character, args.fetch)
    print(json.dumps({k: result[k] for k in ('counts', 'integrity_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
