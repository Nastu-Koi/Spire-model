"""python -m spire_codex_data fetch ...

Downloads public run summaries and replays. Source selection, anchor generation
and the checks have their own entry points: spire_codex_data.sources, .anchor,
.anchor_audit and .map_ambiguity.
"""
import argparse
import json
from pathlib import Path

from .fetch import CHARACTERS, download


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch')
    fetch.add_argument('--output', default='data/spire-codex', type=Path)
    fetch.add_argument('--max-runs', type=int, default=0,
        help='0 means all; counts runs that needed a request, cached ones are free')
    fetch.add_argument('--source', choices=('runs', 'replays', 'both'), default='both')
    fetch.add_argument('--start-page', type=int, default=1)
    fetch.add_argument('--resume', action='store_true',
        help='Continue each slice from its last recorded page instead of rescanning from page 1')
    fetch.add_argument('--parallel', type=int, default=1,
        help='Character slices walked at once; they share one request budget per route (the site allows 60 run summaries a minute)')
    fetch.add_argument('--characters', nargs='+', choices=CHARACTERS, default=None,
        help='Run-summary slices to walk; default all five (the unsliced listing is capped)')
    args = parser.parse_args()
    if args.max_runs < 0 or args.start_page < 1 or args.parallel < 1:
        parser.error('max-runs must be nonnegative; start-page and parallel must be positive')
    failures = 0
    for source in ('replays', 'runs') if args.source == 'both' else (args.source,):
        rows, errors = download(args.output, max_runs=args.max_runs, source=source, start_page=args.start_page,
            characters=args.characters if source == 'runs' else None, resume=args.resume, parallel=args.parallel)
        print(json.dumps(dict(source=source, available=len(rows), errors=errors)))
        failures += len(errors)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
