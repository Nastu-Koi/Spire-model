"""How far a summary's room-type sequence determines the route, measured on replays.

A run summary keeps only the type of every visited map node. Replays also keep
the map and the visited coordinates, so they show how often that type sequence
leaves a single route through the same map.

python -m spire_codex_data.map_ambiguity data/spire-codex --output data/anchors/map-ambiguity.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from .fetch import save_json


def consistent(nodes, start, kinds):
    """Per floor, the nodes on some route from start whose node types spell kinds."""
    forward = [{start} if nodes[start]['kind'] == kinds[0] else set()]
    for kind in kinds[1:]:
        forward.append({c for n in forward[-1] for c in nodes[n]['children'] if c in nodes and nodes[c]['kind'] == kind})
    alive = [set() for _ in kinds]
    alive[-1] = forward[-1]
    for i in range(len(kinds) - 2, -1, -1):
        alive[i] = {n for n in forward[i] if alive[i + 1] & set(nodes[n]['children'])}
    return alive


def act_report(nodes, path):
    """Route statistics of one act, or the reason the recorded path is not a route of this map."""
    if any(c not in nodes for c in path):
        return dict(status='coordinate_not_on_map')
    if any(b not in nodes[a]['children'] for a, b in zip(path, path[1:])):
        return dict(status='step_not_on_an_edge')
    alive = consistent(nodes, path[0], [nodes[c]['kind'] for c in path])
    choices = determined = position_known = 0
    for i in range(len(path) - 1):
        if len(nodes[path[i]]['children']) < 2:
            continue
        choices += 1
        position_known += len(alive[i]) == 1
        determined += len(alive[i]) == 1 and len(alive[i + 1]) == 1
    return dict(status='ok', unique_route=all(len(s) == 1 for s in alive), choices=choices,
                determined=determined, position_known=position_known,
                widest=max(len(s) for s in alive))


def analyze(journal):
    """One report per act of a replay; a replayed act uses the first map its path fits."""
    maps, paths = {}, {}
    for row in journal:
        if row.get('t') == 'map' and isinstance(row.get('nodes'), list):
            maps.setdefault(row['act'], []).append({n['coord']: n for n in row['nodes']})
        elif row.get('t') == 'room' and row.get('coord'):
            path = paths.setdefault(row['act'], [])
            if not path or path[-1] != row['coord']:
                path.append(row['coord'])
    reports = {}
    for act, path in paths.items():
        tried = [act_report(nodes, path) for nodes in maps.get(act, [])]
        reports[act] = next((r for r in tried if r['status'] == 'ok'), tried[-1] if tried else dict(status='no_map'))
    return reports


def summarize(roots):
    counts, widths, seen = Counter(), Counter(), set()
    for root in roots:
        for path in sorted((Path(root) / 'raw').glob('*.replay.jsonl')):
            if path.name in seen:
                continue
            seen.add(path.name)
            journal = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            # A resumed journal revisits coordinates on another save branch.
            if any(row.get('t') == 'resume' for row in journal):
                counts['replays_with_resume_skipped'] += 1
                continue
            counts['replays'] += 1
            for report in analyze(journal).values():
                counts['acts'] += 1
                counts['acts_' + report['status']] += 1
                if report['status'] != 'ok':
                    continue
                counts['acts_unique_route'] += report['unique_route']
                for name in ('choices', 'determined', 'position_known'):
                    counts[name] += report[name]
                widths[report['widest']] += 1
    ok, choices = counts['acts_ok'], counts['choices']
    return dict(counts=dict(counts), widest_floor_candidates=dict(sorted(widths.items())),
                unique_route_rate=counts['acts_unique_route'] / ok if ok else None,
                determined_choice_rate=counts['determined'] / choices if choices else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.roots)
    save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
