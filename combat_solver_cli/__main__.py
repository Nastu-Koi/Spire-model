import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .client import DEFAULT_CONFIG, ROOT, SolverEngine, environment
from .astar import search, write_json
from model.protocol import CHARACTERS, execution_command, validate_frame


def first_combat(config, character, seed, budget_ms, reuse_turn_plan=False, ascension=10):
    count = selection = 0
    with SolverEngine(config) as engine:
        frame = engine.reset(character, seed, ascension)
        for _ in range(1000):
            validate_frame(frame)
            if any(e.get('type') == 'encounter_completed' and e.get('result') == 'victory' for e in frame.get('events', [])):
                return dict(character=character, seed=seed, ascension=ascension, status='victory', solver_steps=count, selection_steps=selection, training_ready=frame['contract']['training_ready'])
            if frame['boundary'] == 'terminal': raise RuntimeError('Native run ended before first combat victory')
            if frame['boundary'] == 'waiting': frame = engine.send({'cmd': 'advance_to_boundary'}); continue
            phase = frame['public']['phase']
            in_combat = phase == 'combat' or (phase in ('card_select', 'card_reward') and engine.send({'cmd': 'solver_info'}).get('combat_in_progress'))
            if in_combat:
                result = engine.step(frame, budget_ms=budget_ms, potions=True, reuse_turn_plan=reuse_turn_plan)
                if result.get('type') != 'solver_step': raise RuntimeError(str(result))
                count += 1; selection += phase != 'combat'; frame = result['frame']
            else:
                candidate = next(c for c in frame['legal']['candidates'] if c['verb'] != 'ABANDON_RUN')
                frame = engine.send(execution_command(frame, candidate['candidate_ref']))
        raise RuntimeError('First combat smoke exceeded step limit')


def main():
    parser = argparse.ArgumentParser(description='Native CombatSolver teacher and full-run weighted A*')
    sub = parser.add_subparsers(dest='command', required=True)
    configure = sub.add_parser('configure')
    configure.add_argument('--solver', required=True, type=Path)
    configure.add_argument('--lib', type=Path, default=ROOT/'sts2-cli/lib')
    configure.add_argument('--dependency-dir', action='append', type=Path, default=[])
    configure.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    smoke = sub.add_parser('smoke')
    smoke.add_argument('--characters', nargs='+', choices=CHARACTERS, default=list(CHARACTERS))
    smoke.add_argument('--seed', default='combat-solver-cli-smoke')
    smoke.add_argument('--output', type=Path, default=ROOT/'combat_solver_cli/artifacts/smoke.json')
    smoke.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    smoke.add_argument('--budget-ms', type=int, default=1000)
    smoke.add_argument('--ascension', type=int, choices=range(11), default=10)
    smoke.add_argument('--reuse-turn-plan', action='store_true')
    for name in ('search', 'batch'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        cmd.add_argument('--output', type=Path, required=True)
        cmd.add_argument('--budget-ms', type=int, default=1000)
        cmd.add_argument('--ascension', type=int, choices=range(11), default=10)
        cmd.add_argument('--weight', type=float, default=5)
        cmd.add_argument('--max-expansions', type=int, default=2000)
        cmd.add_argument('--max-seconds', type=float, default=3600)
        cmd.add_argument('--max-steps', type=int, default=10000)
        cmd.add_argument('--reuse-turn-plan', action='store_true')
        cmd.add_argument('--rollout-decisions', type=int, default=0)
        cmd.add_argument('--search-lanes', type=int, choices=[1, 2, 4], default=2)
        if name == 'search':
            cmd.add_argument('--character', choices=CHARACTERS, default='Ironclad')
            cmd.add_argument('--seed', required=True)
            restore = cmd.add_mutually_exclusive_group()
            restore.add_argument('--resume', type=Path)
            restore.add_argument('--prefix-path', type=Path)
        else:
            cmd.add_argument('--jobs', type=Path, required=True)
            cmd.add_argument('--workers', type=int, choices=[1], default=1)
    verify = sub.add_parser('verify')
    verify.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    verify.add_argument('--prefix', type=Path, required=True)
    verify.add_argument('--output', type=Path, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'configure':
        solver, lib = args['solver'].resolve(), args['lib'].resolve()
        if not solver.is_file() or not (lib/'sts2.dll').is_file(): raise FileNotFoundError('Solver or game DLL missing')
        manifest = json.loads(solver.with_suffix('.json').read_text())
        if str(manifest.get('version')) != '0.44.0': raise ValueError('Adapter currently supports CombatSolver 0.44.0')
        dependencies = [p.resolve() for p in args['dependency_dir']]
        if any(not p.is_dir() for p in dependencies): raise FileNotFoundError('Dependency directory missing')
        subprocess.run(['dotnet', 'build', str(ROOT/'combat_solver_cli/CombatSolverCli.csproj'), '--nologo', '-v:q', '-m:1'], check=True)
        config = dict(solver_dll=str(solver), game_dll=str(lib/'sts2.dll'), dependency_dirs=list(map(str, dependencies)), worker_dll=str(ROOT/'combat_solver_cli/bin/Debug/net9.0/CombatSolverCli.dll'))
        for key in ('solver_dll', 'game_dll'): config[key+'_sha256'] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
        args['config'].parent.mkdir(parents=True, exist_ok=True)
        write_json(args['config'], config); print(args['config']); return
    if command == 'smoke':
        result = [first_combat(args['config'], c, args['seed'], args['budget_ms'], args['reuse_turn_plan'], args['ascension']) for c in args['characters']]
        args['output'].parent.mkdir(parents=True, exist_ok=True); write_json(args['output'], result)
    elif command == 'search':
        result = search(**args)
        print(json.dumps(result, ensure_ascii=False)); raise SystemExit(0 if result['status'] == 'verified_victory' else 2)
    elif command == 'batch':
        from .batch import generate
        args['jobs'] = json.loads(args['jobs'].read_text())
        result = generate(**args)
        print(json.dumps(result, ensure_ascii=False)); raise SystemExit(0 if result['verified_victories'] else 2)
    else:
        from .trajectory import verify_and_export
        result = verify_and_export(args['config'], args['prefix'], args['output'])
        result = dict(run_id=result['run_id'], provenance=result['provenance'])
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
