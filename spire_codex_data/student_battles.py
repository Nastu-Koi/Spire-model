"""Solver labels for the fights a policy walked into: student-state battles.

A sampled round (`python -m model collect`, `round-N/`) keeps the engine's save at the
entry of every fight (`<run>.saves.jsonl.gz`, `model/saves.py`). From such a save the
solver fights the same fight again, the way it fights a summary anchor's battle: it
enters with `enter_anchor` and `anchor_room` (same HP, gold, deck, relics, potions and
enemies; the enemies' random HP and the opening hand are drawn anew, it is another
fight from the same entry), plays it on one native process and another process repeats
its actions step by step (`anchor.play_battle`). The actions of a won fight are
independent supervised samples (`spire-independent-decisions-v1`,
`anchor_kind=student_battle`, `label_source=combat_solver`).

Which fights: every fight of a fixed fraction of the round's runs (drawn by the hash
of the run id, so the same runs whatever else is selected), and the fight each lost
run died in. Fights an event led into are skipped and counted, so are fights whose
save the engine refused. Potions: an elite or boss fight entered with a full belt is
asked for one potion of the solver's choosing (`RequireAtLeastOne`, held to the fight
by `anchor.play_battle`); elsewhere the solver's own policy.

python -m spire_codex_data.student_battles select --round runs/collect/round-3 --fraction 0.25
python -m spire_codex_data.student_battles run --round runs/collect/round-3 --output data/student/round-3 \\
    --fraction 0.25 --workers 8
python -m spire_codex_data.student_battles export --output data/student/round-3
python -m model import-independent data/student/round-3/independent-training.jsonl.gz \\
    --output data/student-import/round-3 --shard-samples 250
"""
import argparse
import gzip
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

from combat_solver_cli.client import DEFAULT_CONFIG, ROOT
from model.control import controller_for
from model.dataset_index import STUDENT_BATTLE, read_index
from model.protocol import fingerprint
from model.saves import read_saves
from . import anchor
from .anchor import AUTOMATIC_POTIONS, BUDGETS, INITIALIZATION, RunProcess, SCHEMA, play_battle
from .fetch import save_json
from .sources import split_group

VERSION = 'student-battle-v1'
FRACTION = 0.25
WORKERS = 8
# The solver worker built with `potion_directives` and `potion_policy`, when the
# workspace has it; the default configuration's worker may predate them.
POTION_BUILD_CONFIG = ROOT / 'runs/bootstrap-20261008/build-potions/config.json'
SOLVER_CONFIG = POTION_BUILD_CONFIG if POTION_BUILD_CONFIG.is_file() else DEFAULT_CONFIG
# What a fight is selected by.
BY_FRACTION, BY_DEATH = 'fraction', 'death'


def read(path):
    return json.loads(Path(path).read_text())


def in_fraction(run_id, fraction):
    """Whether a run is among the fraction whose every fight is labelled: the first 32
    bits of its id's fingerprint, as a fraction of 2**32, under `fraction`. A function
    of the id alone, so a run's side does not change as the fraction grows."""
    return int(fingerprint(run_id)[:8], 16) / 2**32 < fraction


def fight_id(run_id, fight):
    return f'{run_id}:fight-{fight:03d}'


def reason_counts(reasons):
    """How a selected fight is counted: under the first of its reasons, so that the
    reasons add up to the fights (`selected_by_death` are the fights selected for the
    death alone); `selected_deaths` are all the selected fights a run died in, of runs
    in the fraction and out of it."""
    return {'selected_by_' + reasons[0]: 1, 'selected_deaths': int(BY_DEATH in reasons)}


def select(round_dir, fraction=FRACTION, max_runs=0):
    """The fights of a round to label, by run: (runs, counts). A run is a dict with its
    identity, its policy and the fights selected of it, each with why (`selected_by`).
    With `max_runs` only the first so many runs with selected fights are taken."""
    if not 0 <= fraction <= 1:
        raise ValueError('The fraction of runs is between 0 and 1')
    round_dir = Path(round_dir)
    index = {row['run_id']: row for row in read_index(round_dir)}
    summary = read(round_dir / 'summary.json')
    runs, counts = [], Counter()
    for path in sorted(round_dir.glob('*.saves.jsonl.gz')):
        rows = list(read_saves(path))
        if not rows:
            counts['runs_without_fights'] += 1
            continue
        run_id = rows[0]['run_id']
        counts['runs'] += 1
        counts['fights'] += len(rows)
        indexed = index.get(run_id)
        if indexed is None:
            # A run the round did not accept (quarantined) is not data.
            counts['runs_not_indexed'] += 1
            continue
        whole = in_fraction(run_id, fraction)
        fights, taken = [], Counter(runs_in_fraction=int(whole))
        for row in rows:
            reasons = [reason for reason, holds in ((BY_FRACTION, whole), (BY_DEATH, row.get('result') == 'loss'))
                       if holds]
            if not reasons:
                continue
            if row.get('from_event'):
                taken['skipped_from_event'] += 1
            elif row.get('save') is None:
                taken['skipped_no_save'] += 1
            elif row.get('result') not in ('win', 'loss'):
                taken['skipped_unlabelled'] += 1
            else:
                taken.update(reason_counts(reasons), selected=1, **{'selected_' + row['kind']: 1})
                fights.append(dict(fight=row['fight'], selected_by=reasons))
        if fights and max_runs and len(runs) >= max_runs:
            counts['runs_beyond_limit'] += 1
            continue
        counts.update(taken)
        if fights:
            runs.append(dict(
                run_id=run_id, round=str(round_dir), saves=path.name,
                seed=rows[0]['seed'], character=rows[0]['character'], ascension=rows[0]['ascension'],
                victory=indexed.get('victory'),
                policy_version=indexed.get('policy_version', summary.get('policy_version')),
                policy_digest=indexed.get('policy_digest', summary.get('policy_digest')),
                iteration=indexed.get('iteration', summary.get('iteration')),
                fights=fights))
    return runs, dict({key: n for key, n in sorted(counts.items()) if n},
                      runs_selected=len(runs), selected=counts['selected'])


def belt(player):
    """The potion ids in the belt of a save's player. The game serializes the belt as
    objects (`id`, `slot_index`), and as null when it is empty."""
    return [p['id'] if isinstance(p, dict) else p for p in player.get('potions') or []]


def belt_full(player):
    """Whether the belt of a save's player is full, with a potion the solver can use."""
    potions = belt(player)
    usable = [p for p in potions if p not in AUTOMATIC_POTIONS]
    return bool(usable) and len(potions) >= player.get('max_potion_slot_count', 0)


def anchor_for(row):
    """The anchor `play_battle` fights a saves row as: the fight's floor, map point and
    encounter (as `model.saves.anchor_commands` enters it), its kind for the budget, and
    the solver's potion policy: at least one potion in an elite or boss fight entered
    with a full belt."""
    player = row['save']['players'][0]
    spot = dict(
        id=fight_id(row['run_id'], row['fight']), floor=row['floor'], map_point_type=row['map_point_type'],
        second_boss=row['second_boss'], route=None, room_type=row['map_point_type'],
        room=dict(type='combat', encounter=row['encounter'].removeprefix('ENCOUNTER.')),
        act=row['act'], encounter=row['encounter'], kind=row['kind'], belt=len(belt(player)))
    if row['kind'] in ('elite', 'boss') and belt_full(player):
        spot['potion_policy'] = 'RequireAtLeastOne'
    return spot


def relic_differences(student, solver):
    """The public fields of the relics that differ between the policy's entry frame
    (the `relics` of its saves row) and the solver's first frame: a relic that counts at
    the start of a fight or of a turn can stand one off when the fight is entered again."""
    if student is None:
        return None
    mine = student
    theirs = [e for e in solver['observation']['entities'] if e.get('entity_type') == 'relic']
    if [e.get('content_id') for e in mine] != [e.get('content_id') for e in theirs]:
        return [dict(relic='*', field='relics', student=[e.get('content_id') for e in mine],
                     solver=[e.get('content_id') for e in theirs])]
    differences = []
    for a, b in zip(mine, theirs):
        for field in sorted((set(a) | set(b)) - {'ref'}):
            # An empty value and a missing one are the same relic: engine builds differ
            # in writing `stats: {}` or no field at all.
            student_value, solver_value = (None if v in (None, {}, [], '') else v for v in (a.get(field), b.get(field)))
            if student_value != solver_value:
                differences.append(dict(relic=a.get('content_id'), field=field, student=student_value, solver=solver_value))
    return differences


def sample_row(record, run, row, spot, contract, outcome):
    """One exported decision of a fight: the row the solver's step made, with the
    metadata an independent sample carries and what names the policy and the fight."""
    item = {k: record[k] for k in ('schema', 'observation', 'options', 'label', 'coverage', 'actor')}
    item['metadata'] = dict(
        run_hash=run['run_id'], character=run['character'], ascension=run['ascension'], independent=True,
        bc_only=True, native_replay_verified=True, verified_by='native_replay', teacher_visibility='privileged',
        label_source='combat_solver', routing=record['routing'], contract=contract, sample_group=spot['id'],
        split_group=split_group(run['seed']), initialization=INITIALIZATION, anchor_kind=STUDENT_BATTLE,
        outcome=outcome, state_sources=[], map_position=spot.get('map_position'), source_integrity='sampled_rollout',
        policy_version=run['policy_version'], policy_digest=run['policy_digest'], iteration=run['iteration'],
        game_seed=run['seed'], fight=row['fight'], encounter=row['encounter'], kind=row['kind'], act=row['act'],
        floor=row['floor'], student_result=row['result'], potion_policy=spot.get('potion_policy', 'Smart'))
    return item


def process_run(run, output, config=SOLVER_CONFIG, budgets=BUDGETS):
    """The selected fights of one run, on two native processes of its own: the report
    goes to `runs/<run id>.json`, the rows of every fight (won or lost) to
    `samples/<run id>.jsonl.gz`.

    A finished report stands while it holds the same fights and was made by the same
    version, native build and solver budgets: another fraction fights again only the
    runs it selects other fights of."""
    output, round_dir = Path(output), Path(run['round'])
    shared, second = RunProcess(config), RunProcess(config)
    identity = dict(version=VERSION, **shared.build_identity, **budgets)
    target = output / 'runs' / f'{run["run_id"]}.json'
    if target.exists():
        done = read(target)
        if done['identity'] == identity and [f['fight'] for f in done['fights']] == [f['fight'] for f in run['fights']]:
            return done
    saves = {row['fight']: row for row in read_saves(round_dir / run['saves'])}
    samples = output / 'samples' / f'{run["run_id"]}.jsonl.gz'
    samples.parent.mkdir(parents=True, exist_ok=True)
    temp = samples.with_suffix('.tmp')
    reports = []
    with shared, second, gzip.open(temp, 'wt', encoding='utf-8') as stream:
        for selected in run['fights']:
            row = saves[selected['fight']]
            spot = anchor_for(row)
            report = dict(selected, id=spot['id'], encounter=row['encounter'], kind=row['kind'], act=row['act'],
                          floor=row['floor'], room_type=spot['room_type'], student_result=row['result'],
                          belt=spot['belt'], potion_policy=spot.get('potion_policy', 'Smart'))
            started = time.monotonic()
            try:
                result = play_battle(shared, second, row['save'], spot, budgets)
                records, contract = result.pop('records'), result.pop('contract')
                report.update(result, status='verified', rows=len(records), map_position=spot['map_position'],
                              potions_used=sum(r['label']['verb'] == 'USE_POTION' for r in records),
                              relic_differences=relic_differences(row.get('relics'), records[0]))
                for record in records:
                    stream.write(json.dumps(sample_row(record, run, row, spot, contract, result['outcome']),
                                            ensure_ascii=False) + '\n')
            except Exception as exc:
                report.update(status='execution_failed', error=f'{type(exc).__name__}:{exc}')
            report['seconds'] = round(time.monotonic() - started, 2)
            reports.append(report)
    temp.replace(samples)
    result = dict(identity=identity, **{k: run[k] for k in (
        'run_id', 'round', 'seed', 'character', 'ascension', 'victory', 'policy_version', 'policy_digest',
        'iteration')}, fights=reports)
    save_json(target, result)
    return result


def work(task):
    """One run in a worker process: (run id, its report, None) or (run id, None, why not)."""
    run, arguments = task[0], task[1:]
    try:
        return run['run_id'], process_run(run, *arguments), None
    except Exception as exc:
        return run['run_id'], None, f'{type(exc).__name__}:{exc}'


def error_class(error):
    """The kind of a failure, from its message: what failed, not where."""
    kind, _, detail = error.partition(':')
    if kind in ('ValueError', 'TimeoutError') and detail:
        return detail.split(':')[0]
    return kind


def selected_reports(output):
    """The reports of the runs of the last selection (`selection.json`), by name. A
    report left by a selection that took other runs is kept on disk for a selection
    that takes them again, and is not part of this one."""
    output = Path(output)
    selected = {run['run_id'] for run in read(output / 'selection.json')['runs']}
    return [path for path in sorted((output / 'runs').glob('*.json')) if path.stem in selected]


def summarize(output):
    """The counts of the selection's reports under `output`, written to its `summary.json`."""
    output = Path(output)
    counts, errors, seconds, by_kind, relics = Counter(), Counter(), Counter(), Counter(), Counter()
    policies = Counter()
    reports = selected_reports(output)
    for path in reports:
        report = read(path)
        policies[(report['policy_version'], report['policy_digest'], report['iteration'])] += 1
        for fight in report['fights']:
            counts['fights'] += 1
            counts[fight['status']] += 1
            seconds['total'] += fight.get('seconds', 0)
            seconds[fight['kind']] += fight.get('seconds', 0)
            by_kind[fight['kind']] += 1
            required = fight.get('potion_policy') == 'RequireAtLeastOne'
            counts['potion_policy_' + fight.get('potion_policy', 'Smart')] += 1
            counts.update(reason_counts(fight['selected_by']))
            if fight['status'] == 'verified':
                counts['outcome_' + fight['outcome']] += 1
                counts['rows'] += fight['rows']
                counts['potions_used'] += fight.get('potions_used', 0)
                counts['fights_using_potions'] += bool(fight.get('potions_used'))
                if required:
                    # Asked for a potion: whether one was used, and the searches that found no such route.
                    counts['potion_required_met' if fight.get('potions_used') else 'potion_required_unmet'] += 1
                    counts['potion_refusals'] += fight.get('potion_refusals', 0)
                if fight.get('relic_differences'):
                    relics[','.join(sorted({d['relic'] for d in fight['relic_differences']}))] += 1
                    counts['fights_with_relic_differences'] += 1
            else:
                errors[error_class(fight['error'])] += 1
    verified = counts['verified']
    result = dict(
        version=VERSION, runs=len(reports), counts={key: n for key, n in sorted(counts.items()) if n},
        errors=dict(errors.most_common()), relic_differences=dict(relics.most_common()),
        fights_by_kind=dict(sorted(by_kind.items())),
        seconds_per_fight={k: round(v / by_kind[k], 1) for k, v in seconds.items() if k in by_kind},
        seconds_total=round(seconds['total'], 1),
        verified_rate=round(verified / counts['fights'], 4) if counts['fights'] else None,
        win_rate=round(counts['outcome_win'] / verified, 4) if verified else None,
        policies=[dict(policy_version=v, policy_digest=d, iteration=i, runs=n)
                  for (v, d, i), n in sorted(policies.items(), key=str)])
    save_json(output / 'summary.json', result)
    return result


def run_round(round_dir, output, *, fraction=FRACTION, workers=WORKERS, config=SOLVER_CONFIG, budgets=BUDGETS,
              max_runs=0, log=None):
    """Select the fights of a round and fight them, `workers` runs at a time; the
    selection's counts and every run's outcome are written to `output/summary.json`."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs, selection = select(round_dir, fraction, max_runs)
    save_json(output / 'selection.json', dict(round=str(round_dir), fraction=fraction, counts=selection,
                                               runs=[dict(run_id=r['run_id'], fights=r['fights']) for r in runs]))
    failures = {}
    tasks = [(run, output, config, budgets) for run in runs]
    # Runs are spread over processes, as the summary anchors' are (`anchor.work`).
    with ProcessPoolExecutor(max_workers=workers) if workers > 1 and len(tasks) > 1 else nullcontext() as pool:
        for key, report, error in (pool.map(work, tasks) if pool else map(work, tasks)):
            if error:
                failures[key] = error
            if log:
                done = Counter(f['status'] for f in report['fights']) if report else {}
                log(json.dumps(dict(run=key, fights=dict(done), error=error)))
    result = dict(summarize(output), selection=selection, run_failures=failures,
                  solver_config=str(config), budgets=budgets, fraction=fraction)
    save_json(output / 'summary.json', result)
    return result


def decided(item):
    """Whether a row is a policy decision: more than one option and no controller."""
    return len(item['options']) > 1 and controller_for(item['observation'].get('phase'), item['options']) is None


def export(output, target=None):
    """The won fights' rows as independent supervised samples. Lost fights teach no
    actions and are counted; so are fights in which nothing was chosen (every step
    forced)."""
    output = Path(output)
    target = Path(target or output / 'independent-training.jsonl.gz')
    counts = Counter()
    temp = target.with_suffix('.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8', compresslevel=anchor.COMPRESSION) as training:
        for path in selected_reports(output):
            report = read(path)
            samples = output / 'samples' / (path.stem + '.jsonl.gz')
            verified = {f['id']: f for f in report['fights'] if f['status'] == 'verified'}
            if not verified:
                continue
            groups = {}
            with gzip.open(samples, 'rt', encoding='utf-8') as rows:
                for line in rows:
                    item = json.loads(line)
                    if item['schema'] != SCHEMA or item['label'] not in item['options']:
                        raise ValueError('Export label is not a legal option')
                    groups.setdefault(item['metadata']['sample_group'], []).append(item)
            for key, fight in verified.items():
                items = groups.get(key, [])
                if len(items) != fight['rows']:
                    raise ValueError(f'Fight rows are missing for {key}')
                if fight['outcome'] == 'loss':
                    counts['lost_fights'] += 1
                elif not any(decided(item) for item in items):
                    counts['fights_without_decision'] += 1
                else:
                    for item in items:
                        training.write(json.dumps(item, ensure_ascii=False) + '\n')
                        counts['rows_' + item['actor']] += 1
                    counts['fights'] += 1
                    counts['rows'] += len(items)
    temp.replace(target)
    summary = dict(path=str(target), schema=SCHEMA, anchor_kind=STUDENT_BATTLE, counts=dict(sorted(counts.items())),
                   continuous_winning_trajectory=False, initialization=INITIALIZATION)
    save_json(output / 'student-export-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['select', 'run', 'export'])
    parser.add_argument('--round', type=Path, help='A sampled round (round-N) of `model collect`')
    parser.add_argument('--output', type=Path, help='Student battle directory (reports, samples, exports)')
    parser.add_argument('--fraction', type=float, default=FRACTION,
                        help='The fraction of runs whose every fight is labelled; the fight each lost run '
                             'died in is labelled besides')
    parser.add_argument('--workers', type=int, default=WORKERS, help='Runs fought at once, two native processes each')
    parser.add_argument('--max-runs', type=int, default=0, help='Runs to take of the selection; 0 means all')
    parser.add_argument('--budget-ms', type=int, default=BUDGETS['budget_ms'],
                        help='Solver time per search in a normal fight')
    parser.add_argument('--elite-budget-ms', type=int, default=BUDGETS['elite_budget_ms'])
    parser.add_argument('--boss-budget-ms', type=int, default=BUDGETS['boss_budget_ms'])
    parser.add_argument('--solver-config', type=Path, default=SOLVER_CONFIG,
                        help='Solver worker configuration; the worker must know potion_policy')
    args = parser.parse_args()
    if args.mode == 'export':
        if not args.output:
            parser.error('export needs --output')
        result = export(args.output)
    elif args.mode == 'select':
        if not args.round:
            parser.error('select needs --round')
        runs, counts = select(args.round, args.fraction, args.max_runs)
        result = dict(round=str(args.round), fraction=args.fraction, counts=counts,
                      runs=[dict(run_id=r['run_id'], character=r['character'], victory=r['victory'],
                                 fights=[f['fight'] for f in r['fights']]) for r in runs])
    else:
        if not args.round or not args.output:
            parser.error('run needs --round and --output')
        if args.workers < 1:
            parser.error('workers must be positive')
        budgets = dict(budget_ms=args.budget_ms, elite_budget_ms=args.elite_budget_ms,
                       boss_budget_ms=args.boss_budget_ms)
        result = run_round(args.round, args.output, fraction=args.fraction, workers=args.workers,
                           config=args.solver_config, budgets=budgets, max_runs=args.max_runs,
                           log=lambda line: print(line, flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('run_failures'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
