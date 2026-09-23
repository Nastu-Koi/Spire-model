"""Serially produce an exact, balanced target of replay-verified trajectories."""
from pathlib import Path
import json
import secrets
import shutil
import threading

from model.data import load_runs
from model.protocol import CHARACTERS, fingerprint
from .astar import search, write_json


def target_quotas(target, characters):
    characters = tuple(characters)
    if type(target) is not int or target < 1:
        raise ValueError('target_trajectories must be a positive integer')
    if not characters or len(set(characters)) != len(characters):
        raise ValueError('characters must be nonempty and unique')
    if any(character not in CHARACTERS for character in characters):
        raise ValueError('Unsupported character')
    base, remainder = divmod(target, len(characters))
    return {character: base + (index < remainder)
            for index, character in enumerate(characters)}


def target_schedule(quotas):
    return [character for index in range(max(quotas.values(), default=0))
            for character, quota in quotas.items() if index < quota]


def _new_seed(used):
    while True:
        seed = secrets.token_hex(8).upper()
        if seed not in used: return seed


def _write_aggregate(parts, destination):
    temporary = destination.with_suffix(destination.suffix+'.tmp')
    with temporary.open('wb') as output:
        for part in parts:
            with part.open('rb') as stream: shutil.copyfileobj(stream, output)
    temporary.replace(destination)


def _completed_state(manifest, parts):
    completed = {character: 0 for character in manifest['quotas']}
    results = []
    for part in parts:
        index = int(part.stem)
        game = next(game for game in manifest['games'] if game['index'] == index)
        completed[game['character']] += 1
        results.append(dict(character=game['character'], seed=game['seed'],
                            status='verified_victory', segments=len(game['segments'])))
    return completed, results


def _publish_verified(segment, part, character, ascension):
    runs = load_runs(segment/'accepted.jsonl')
    expected = f'A{ascension}_final_boss_victory'
    if (len(runs) != 1 or runs[0]['character'] != character
            or runs[0]['provenance'].get('verified_outcome') != expected):
        raise ValueError('Winning segment did not contain one matching verified run')
    temporary = part.with_suffix('.jsonl.tmp')
    shutil.copy2(segment/'accepted.jsonl', temporary)
    temporary.replace(part)


def _recover_segments(game_record, segments, manifest_path, manifest):
    """Reconcile segment directories created before a manifest write completed."""
    while True:
        index = len(game_record['segments'])
        segment = segments/f'{index:04d}'
        if not segment.is_dir(): break
        summary = segment/'summary.json'
        try:
            result = json.loads(summary.read_text()) if summary.is_file() else {'status': 'interrupted'}
        except (OSError, ValueError):
            result = {'status': 'interrupted'}
        game_record['segments'].append(dict(index=index, output=str(segment), result=result))
        write_json(manifest_path, manifest)


def generate_target(config, output, target_trajectories, *, characters=CHARACTERS,
                    workers=1, search_fn=search,
                    resume_existing=False, **options):
    if workers != 1: raise ValueError('Only one game may be searched at a time')
    if options.get('search_lanes', 2) not in (1, 2, 4):
        raise ValueError('search_lanes must be 1, 2 or 4')
    ascension = options.get('ascension', 0)
    quotas = target_quotas(target_trajectories, characters)
    schedule = target_schedule(quotas)
    output = Path(output)
    manifest_path = output/'manifest.json'
    if resume_existing:
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('mode'), manifest.get('target_trajectories'), manifest.get('quotas')) != (
                'target_verified_trajectories', target_trajectories, quotas):
            raise ValueError('Target manifest identity differs')
        (output/'STOP').unlink(missing_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'accepted-parts').mkdir(); (output/'games').mkdir()
        manifest = dict(mode='target_verified_trajectories', config=str(config),
                        target_trajectories=target_trajectories,
                        characters=list(characters), quotas=quotas, workers=1,
                        seed_source='system_random_per_trajectory',
                        options=options, games=[])
        write_json(manifest_path, manifest)
        (output/'accepted.jsonl').touch()
    parts_dir, games = output/'accepted-parts', output/'games'
    part_paths = sorted(parts_dir.glob('*.jsonl'))
    if resume_existing: _write_aggregate(part_paths, output/'accepted.jsonl')
    completed, results = _completed_state(manifest, part_paths)
    used = {game['seed'] for game in manifest['games']}

    for trajectory_index in range(len(part_paths), len(schedule)):
        if (output/'STOP').exists(): break
        character = schedule[trajectory_index]
        existing = next((g for g in manifest['games'] if g['index'] == trajectory_index), None)
        if existing:
            game_record = existing
            if existing['character'] != character:
                raise ValueError('Saved target schedule differs')
            seed = existing['seed']
        else:
            seed = _new_seed(used); used.add(seed)
            game_record = dict(index=trajectory_index, character=character, seed=seed,
                               target_for_character=quotas[character], segments=[])
            manifest['games'].append(game_record)
            write_json(manifest_path, manifest)
        game = games/f'{trajectory_index:05d}-{character}-{fingerprint(seed)[:12]}'
        segments = game/'segments'; segments.mkdir(parents=True, exist_ok=True)
        _recover_segments(game_record, segments, manifest_path, manifest)
        part = parts_dir/f'{trajectory_index:05d}.jsonl'
        recorded_win = next((entry for entry in reversed(game_record['segments'])
                             if entry['result'].get('status') == 'verified_victory'), None)
        if recorded_win:
            _publish_verified(Path(recorded_win['output']), part, character, ascension)
            part_paths.append(part)
            _write_aggregate(part_paths, output/'accepted.jsonl')
            completed[character] += 1
            results.append(dict(character=character, seed=seed, status='verified_victory',
                                segments=len(game_record['segments'])))
            continue
        prior = next((Path(entry['output'])/'frontier.json'
                      for entry in reversed(game_record['segments'])
                      if (Path(entry['output'])/'frontier.json').is_file()), None)
        if (game_record['segments'] and prior is None
                and any(entry['result'].get('status') != 'interrupted'
                        for entry in game_record['segments'])):
            summary = dict(status='blocked', target_trajectories=target_trajectories,
                           verified_trajectories=len(part_paths), quotas=quotas,
                           completed=completed, active_game=game_record,
                           error='Active seed ended without a resumable frontier')
            write_json(output/'summary.json', summary); return summary
        resume = prior
        segment_index = len(game_record['segments'])
        while True:
            segment = segments/f'{segment_index:04d}'
            stop_monitor = threading.Event()
            def propagate_stop():
                while not stop_monitor.wait(.25):
                    if (output/'STOP').exists():
                        if segment.is_dir(): (segment/'STOP').touch()
                        return
            monitor = threading.Thread(target=propagate_stop, daemon=True); monitor.start()
            try:
                result = search_fn(config, character, seed, segment, resume=resume, **options)
            finally:
                stop_monitor.set(); monitor.join(timeout=1)
            result_record = {k: v for k, v in result.items() if k != 'lanes'}
            game_record['segments'].append(dict(index=segment_index, output=str(segment),
                                                result=result_record))
            write_json(manifest_path, manifest)
            if result.get('status') == 'verified_victory':
                _publish_verified(segment, part, character, ascension)
                part_paths.append(part)
                _write_aggregate(part_paths, output/'accepted.jsonl')
                completed[character] += 1
                results.append(dict(character=character, seed=seed,
                                    status='verified_victory', segments=segment_index+1))
                break
            frontier = segment/'frontier.json'
            if (output/'STOP').exists(): break
            if not frontier.is_file():
                summary = dict(status='blocked', target_trajectories=target_trajectories,
                               verified_trajectories=len(part_paths), quotas=quotas,
                               completed=completed, active_game=game_record,
                               error='Active seed ended without a resumable frontier')
                write_json(output/'summary.json', summary); return summary
            resume = frontier; segment_index += 1
        if (output/'STOP').exists(): break

    complete = len(part_paths) == target_trajectories
    summary = dict(status='complete' if complete else 'stopped',
                   target_trajectories=target_trajectories,
                   verified_trajectories=len(part_paths), quotas=quotas,
                   completed=completed, workers=1,
                   search_lanes=options.get('search_lanes', 2), results=results)
    write_json(output/'summary.json', summary)
    return summary


def resume_target(output, search_fn=search):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest.get('mode') != 'target_verified_trajectories':
        raise ValueError('Not a target trajectory output')
    return generate_target(manifest['config'], output, manifest['target_trajectories'],
                           characters=manifest['characters'], workers=1,
                           search_fn=search_fn,
                           resume_existing=True, **manifest['options'])
