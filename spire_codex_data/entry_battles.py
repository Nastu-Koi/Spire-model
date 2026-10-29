"""Solver demonstrations for fights the policy walked into, from their entry saves.

Group sampling keeps the engine's save at the first decision of a fight (an entry save,
`{"at": "fight", "save", "fight": {...}, "character", "seed", "ascension"}`). From such
a save the solver fights the same fight again, the way it fights a summary anchor's
battle (`anchor.play_battle`), except that both native processes enter it with
`resume_fight`, the save as it was written (relics keep their counters, which an anchor
rebuilt from fields loses). The save holds the run as the fight's room was entered, so the
fight starts as it did for the policy: what its start does to the run (Blood Vial's heal)
happens once, the enemies' random HP and the opening hand are drawn as they were. A save
an engine wrote before `RoomEntrySave` holds the run at the fight's first decision
instead: those effects apply once more, the enemies and the hand are drawn anew. One
process plays it, the other repeats its actions step by step. Its prices: a potion has to save 7, 9
or 11 HP by the solver's three tiers, and in the run's last fight 1 HP, the HP counted in
full there; with a potion in the belt, half of every search's budget is kept for the potion
layers; the choices of a turn's setup (Gambling Chip, Toasty Mittens, Toolbox) are the
solver's own (`docs/implementation/solver.md`). The actions of a won fight are independent
supervised samples (`spire-independent-decisions-v1`, `anchor_kind=entry_battle`,
`label_source=combat_solver`).

Which fights is a selection's (`selection.json`): `fights`, each with the `path` of its
entry save, and `extra` fights that are fought and reported but not exported.

python -m spire_codex_data.entry_battles run --selection <selection.json> --output <directory>
python -m spire_codex_data.entry_battles export --output <directory>
python -m main import-independent <directory>/independent-training.jsonl.gz --output <data directory>
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import shutil
import time

from combat_solver_cli.client import DEFAULT_CONFIG
from main.env.control import controller_for
from main.env.saves import resume_fields
from . import anchor
from .anchor import BUDGETS, RunProcess, SCHEMA, play_battle, settle
from .fetch import save_json
from .sources import split_group

VERSION = 'entry-battle-v1'
ENTRY_BATTLE = 'entry_battle'
# The engine's contract for a fight entered from an entry save.
INITIALIZATION = 'fight-resume-v1'
WORKERS = anchor.WORKERS['battle']
# Half of a search's budget is kept for the potion layers when the belt holds a potion.
SEARCH = dict(BUDGETS, potion_costs=[7, 9, 11], final_boss_potion_costs=[1, 1, 1], beam_portfolio=False,
              potion_reserve=0.5)
MAP_POINTS = {'normal': 'monster', 'elite': 'elite', 'boss': 'boss'}


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else open(path, encoding='utf-8')) as stream:
        return json.load(stream)


def resolve(selection, path):
    """A save path of a selection: as written, or relative to the selection's directory or
    to the first of its parents that holds it."""
    path = Path(path)
    if path.is_absolute():
        return path
    for base in [Path(selection).parent, *Path(selection).parent.parents]:
        if (base / path).exists():
            return base / path
    raise FileNotFoundError(path)


def runs_of(selection):
    """The selection's fights grouped by run (character and seed), in floor order, with
    the role each was selected for: `fights` or `extra`."""
    chosen = read(selection)
    runs = defaultdict(list)
    for role in ('fights', 'extra'):
        for fight in chosen.get(role, []):
            run_id = f"{fight['character']}-{fight['seed']}"
            runs[run_id].append(dict(fight, role=role, path=str(resolve(selection, fight['path']))))
    return [dict(run_id=run_id, fights=sorted(fights, key=lambda f: (f['act'], f['floor'])))
            for run_id, fights in sorted(runs.items())]


class EntryProcess(RunProcess):
    """A native process that enters each fight from its entry save as it was written."""

    def enter(self, save, spot):
        def start(engine):
            # What the process logged for earlier fights is not this fight's.
            self.logged = os.fstat(engine.stderr.fileno()).st_size
            reply = engine.send(dict(cmd='resume_fight', **resume_fields(save), encounter=spot['encounter'], act_floor=spot['floor'],
                                     map_point_type=spot['map_point_type'], second_boss=spot['second_boss']))
            if reply.get('type') == 'error':
                # A run the process cannot end (`run_not_ended`) is begun again in a new process.
                raise ValueError('resume_fight:' + str(reply.get('code') or reply.get('message')))
            frame = settle(engine, reply if reply.get('type') == 'decision_frame' else engine.send(dict(cmd='advance_to_boundary')))
            if frame['contract'].get('initialization') != INITIALIZATION:
                raise ValueError('resume_contract_missing')
            return frame
        return self.begin(start)


def spot_for(entry, fight_id):
    """The anchor `play_battle` fights an entry save as: the fight's floor in its act, map
    point and encounter (as `resume_fight` takes them), and its kind for the budget."""
    fight = entry['fight']
    point = fight.get('map_point_type') or MAP_POINTS[fight['kind']]
    return dict(id=fight_id, floor=fight['floor'], map_point_type=point, second_boss=bool(fight.get('second_boss')),
                route=None, map_position='entry_save', room_type=point, room=dict(type='combat', encounter=fight['encounter'].removeprefix('ENCOUNTER.')),
                act=fight['act'], encounter=fight['encounter'], kind=fight['kind'])


def setup_choice(records):
    """The cards the first selection of the fight took before it finished, and who chose
    them, when the fight begins with a selection (a turn's setup such as Gambling Chip)."""
    if not records or not records[0]['observation'].get('selection_context'):
        return None
    picks = 0
    for record in records:
        if record['label']['verb'] != 'SELECT_ONE':
            break
        picks += 1
    return dict(picks=picks, actor=records[0]['actor'],
                max_total=records[0]['observation']['selection_context'].get('max_total'))


def entry_of(fight):
    """A selected fight's entry save, with the run it belongs to: group sampling writes the
    save and the fight only, and the selection names the run."""
    run = {k: fight[k] for k in ('character', 'seed', 'ascension') if fight.get(k) is not None}
    return {**run, **read(fight['path'])}


def failed_report(fight, error, seconds=0):
    """The report of a selected fight that was not fought to its end."""
    return dict({k: fight.get(k) for k in ('id', 'role', 'kind', 'act', 'floor', 'encounter', 'ascension', 'path')},
                status='execution_failed', error=error, seconds=seconds)


def sample_row(record, entry, fight, spot, contract, outcome):
    """One exported decision of a fight: the row the solver's step made, with the metadata
    an independent sample carries and what names the fight."""
    item = {k: record[k] for k in ('schema', 'observation', 'options', 'label', 'coverage', 'actor')}
    item['metadata'] = dict(
        run_hash=f"{entry['character']}-{entry['seed']}", character=entry['character'], ascension=entry['ascension'],
        independent=True, bc_only=True, native_replay_verified=True, verified_by='native_replay',
        teacher_visibility='privileged', label_source='combat_solver', routing=record['routing'], contract=contract,
        sample_group=spot['id'], split_group=split_group(entry['seed']), initialization=INITIALIZATION,
        anchor_kind=ENTRY_BATTLE, outcome=outcome, state_sources=[], map_position=spot['map_position'],
        source_integrity='sampled_rollout', game_seed=entry['seed'], fight=entry['fight'].get('fight'),
        encounter=spot['encounter'], kind=spot['kind'], act=spot['act'], floor=spot['floor'])
    return item


def process_run(run, output, config=DEFAULT_CONFIG, budgets=SEARCH):
    """The selected fights of one run on two native processes of its own: the report goes to
    `runs/<run id>.json`, the rows of every fight (won or lost) to `samples/<run id>.jsonl.gz`.
    A report stands while the fights, version, native build and budgets agree."""
    output = Path(output)
    shared, second = EntryProcess(config), EntryProcess(config)
    identity = dict(version=VERSION, **shared.build_identity, **budgets)
    target = output / 'runs' / f"{run['run_id']}.json"
    samples = output / 'samples' / f"{run['run_id']}.jsonl.gz"
    if target.exists() and samples.exists():
        done = read(target)
        if done['identity'] == identity and [f['id'] for f in done['fights']] == [f['id'] for f in run['fights']]:
            return done
    samples.parent.mkdir(parents=True, exist_ok=True)
    temp = samples.with_suffix('.tmp')
    reports = []
    with shared, second, gzip.open(temp, 'wt', encoding='utf-8') as stream:
        for fight in run['fights']:
            started = time.monotonic()
            try:
                entry = entry_of(fight)
                spot = spot_for(entry, fight['id'])
                report = dict(id=fight['id'], role=fight['role'], kind=spot['kind'], act=spot['act'], floor=spot['floor'],
                              encounter=spot['encounter'], ascension=entry['ascension'], path=fight['path'])
                result = play_battle(shared, second, entry['save'], spot, budgets)
                records, contract = result.pop('records'), result.pop('contract')
                report.update(result, status='verified', rows=len(records), map_position=spot['map_position'],
                              potions_used=sum(r['label']['verb'] == 'USE_POTION' for r in records),
                              setup_choice=setup_choice(records))
                for record in records:
                    stream.write(json.dumps(sample_row(record, entry, fight, spot, contract, result['outcome']),
                                            ensure_ascii=False) + '\n')
            except Exception as exc:
                report = failed_report(fight, f'{type(exc).__name__}:{exc}')
                # Group sampling deletes its saves after the round; the copy can be fought again.
                if Path(fight['path']).is_file():
                    kept = output / 'failed' / Path(fight['path']).name
                    kept.parent.mkdir(exist_ok=True)
                    shutil.copyfile(fight['path'], kept)
                    report['kept'] = str(kept)
            report['seconds'] = round(time.monotonic() - started, 2)
            reports.append(report)
    temp.replace(samples)
    result = dict(identity=identity, run_id=run['run_id'], fights=reports)
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


def reports(output):
    """The fight reports of the selection last run under `output`."""
    output = Path(output)
    selected = read(output / 'selection.json')
    return [fight for run in selected['runs'] for fight in read(output / 'runs' / f"{run}.json")['fights']]


def summarize(output):
    """Counts by kind of fight, written to `summary.json`: fights, verified, failures by
    reason, wins and losses, seconds and potions per fight, and the cards the first selection took in
    fights that begin with one (Gambling Chip) and who chose them."""
    by_kind = defaultdict(list)
    for fight in reports(output):
        by_kind[fight['role'] if fight['role'] == 'extra' else fight['kind']].append(fight)
    summary = {}
    for kind, fights in sorted(by_kind.items()):
        verified = [f for f in fights if f['status'] == 'verified']
        setups = [f['setup_choice'] for f in verified if f.get('setup_choice')]
        summary[kind] = dict(
            fights=len(fights), verified=len(verified),
            failures=dict(Counter(error_class(f['error']) for f in fights if f['status'] != 'verified')),
            wins=sum(f['outcome'] == 'win' for f in verified), losses=sum(f['outcome'] == 'loss' for f in verified),
            seconds_per_fight=round(sum(f['seconds'] for f in fights) / max(1, len(fights)), 1),
            potions_per_fight=round(sum(f['potions_used'] for f in verified) / max(1, len(verified)), 3),
            fights_with_potion=sum(f['potions_used'] > 0 for f in verified),
            setup_selections=dict(fights=len(setups), by=dict(Counter(s['actor'] for s in setups)),
                                  picks=dict(sorted(Counter(s['picks'] for s in setups).items()))))
    save_json(Path(output) / 'summary.json', summary)
    return summary


def run(selection, output, *, workers=WORKERS, config=DEFAULT_CONFIG, budgets=SEARCH, log=print):
    """Fight every run of the selection, `workers` runs at once, then summarize."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = runs_of(selection)
    save_json(output / 'selection.json', dict(selection=str(selection), budgets=budgets, runs=[r['run_id'] for r in runs]))
    failures = {}
    by_id = {r['run_id']: r for r in runs}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for run_id, report, error in pool.map(work, [(r, output, config, budgets) for r in runs]):
            if error:
                failures[run_id] = error
                # Every fight of the run failed with it; no identity, so running again fights it again.
                save_json(output / 'runs' / f'{run_id}.json', dict(
                    identity=None, run_id=run_id, fights=[failed_report(f, error) for f in by_id[run_id]['fights']]))
            done = Counter(f['status'] for f in report['fights']) if report else {}
            log(json.dumps(dict(run=run_id, fights=dict(done), error=error)))
    return dict(summary=summarize(output), run_failures=failures)


def decided(item):
    """Whether a row is a policy decision: more than one option and no controller."""
    return len(item['options']) > 1 and controller_for(item['observation'].get('phase'), item['options']) is None


def export(output, target=None):
    """The won fights of the selection's `fights` as independent supervised samples, to
    `independent-training.jsonl.gz`. Lost fights teach no actions and are counted; so are
    fights in which nothing was chosen, and the `extra` fights, which are not exported."""
    output = Path(output)
    target = Path(target or output / 'independent-training.jsonl.gz')
    counts = Counter()
    temp = target.with_suffix('.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8', compresslevel=anchor.COMPRESSION) as training:
        for run_id in read(output / 'selection.json')['runs']:
            fights = {f['id']: f for f in read(output / 'runs' / f'{run_id}.json')['fights'] if f['status'] == 'verified'}
            if not fights:
                continue
            groups = defaultdict(list)
            with gzip.open(output / 'samples' / f'{run_id}.jsonl.gz', 'rt', encoding='utf-8') as rows:
                for line in rows:
                    item = json.loads(line)
                    if item['schema'] != SCHEMA or item['label'] not in item['options']:
                        raise ValueError('Export label is not a legal option')
                    groups[item['metadata']['sample_group']].append(item)
            for key, fight in fights.items():
                items = groups.get(key, [])
                if len(items) != fight['rows']:
                    raise ValueError(f'Fight rows are missing for {key}')
                if fight['role'] == 'extra':
                    counts['extra_fights'] += 1
                elif fight['outcome'] == 'loss':
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
    summary = dict(path=str(target), schema=SCHEMA, anchor_kind=ENTRY_BATTLE, counts=dict(sorted(counts.items())),
                   continuous_winning_trajectory=False, initialization=INITIALIZATION)
    save_json(output / 'export-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=['run', 'export'])
    parser.add_argument('--selection', type=Path, help='Selection of fights (fights, extra) with entry save paths')
    parser.add_argument('--output', type=Path, required=True, help='Directory of reports, samples and the export')
    parser.add_argument('--workers', type=int, default=WORKERS, help='Runs fought at once, two native processes each')
    parser.add_argument('--budget-ms', type=int, default=SEARCH['budget_ms'], help='Solver time per search in a normal fight')
    parser.add_argument('--elite-budget-ms', type=int, default=SEARCH['elite_budget_ms'])
    parser.add_argument('--boss-budget-ms', type=int, default=SEARCH['boss_budget_ms'])
    parser.add_argument('--potion-costs', type=int, nargs=3, default=SEARCH['potion_costs'],
                        metavar=('BASE', 'ELEVATED', 'HIGH'), help="HP a potion has to save, by the solver's three tiers")
    parser.add_argument('--final-boss-potion-costs', type=int, nargs=3, default=SEARCH['final_boss_potion_costs'],
                        metavar=('BASE', 'ELEVATED', 'HIGH'), help="The prices of the run's last fight")
    parser.add_argument('--beam-portfolio', action='store_true', help="The solver's set of beam widths (off by default)")
    parser.add_argument('--potion-reserve', type=float, default=SEARCH['potion_reserve'],
                        help="Share of a search's budget kept for the potion layers when the belt holds a potion")
    parser.add_argument('--solver-config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if args.mode == 'export':
        result = export(args.output)
    else:
        if not args.selection:
            parser.error('run needs --selection')
        if args.workers < 1:
            parser.error('workers must be positive')
        if not 0 <= args.potion_reserve <= 0.9:
            parser.error('potion reserve must be between 0 and 0.9')
        budgets = dict(budget_ms=args.budget_ms, elite_budget_ms=args.elite_budget_ms, boss_budget_ms=args.boss_budget_ms,
                       potion_costs=args.potion_costs, final_boss_potion_costs=args.final_boss_potion_costs,
                       beam_portfolio=args.beam_portfolio, potion_reserve=args.potion_reserve)
        result = run(args.selection, args.output, workers=args.workers, config=args.solver_config, budgets=budgets,
                     log=lambda line: print(line, flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('run_failures'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
