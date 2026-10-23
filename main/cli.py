"""The command line of `python -m main`: the arguments of every command, and the
dispatch to `main/commands.py` with the errors turned into exit statuses."""

import argparse
import json
import os
import random
import sys

import torch

from main.env.protocol import CHARACTERS, ProtocolError
from main.env.runtime import configure_runtime
from main.commands import COMMANDS
from main.play.rollout import PRECISIONS
from main.steam import parse_duration


def source_share(text):
    """A `KIND=SHARE` argument of --source-shares, as (kind, share)."""
    kind, sep, share = text.partition("=")
    try:
        if not sep:
            raise ValueError
        return kind, float(share)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not KIND=SHARE") from None


def parser():
    p = argparse.ArgumentParser(
        description="STS2 Full Attention policy: data → Bootstrap → sampling and evaluation → Steam"
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser(
        "smoke", help="Synthetic Bootstrap/rollout/checkpoint end-to-end check"
    )
    x.add_argument("--output", required=True)
    x = sub.add_parser("parameters")
    x.add_argument("--config", help="Native model/training JSON profile")
    x = sub.add_parser(
        "init",
        help="Create random weights and a frozen vocabulary from public engine content",
    )
    x.add_argument("--config", required=True)
    x.add_argument("--engine-root")
    x.add_argument("--output", required=True)
    x = sub.add_parser("audit")
    x.add_argument("paths", nargs="+")
    x.add_argument("--accepted", required=True)
    x.add_argument("--quarantine", required=True)
    x = sub.add_parser(
        "import-recorder",
        help="Import Steam RunRecorder JSONL with complete-run and visibility checks",
    )
    x.add_argument(
        "--bootstrap-only",
        "--bc-only",
        dest="bc_only",
        action="store_true",
        help="Use valid recorded decisions for Bootstrap, including solver and partial runs; retain unverified provenance",
    )
    x.add_argument(
        "--recorder-version",
        help="Import only journals with this exact recorder version",
    )
    x.add_argument(
        "paths", nargs="+", help="Journal files, globs, or recorder directories"
    )
    x.add_argument(
        "--output",
        required=True,
        help="New directory for accepted data, quarantine and summary",
    )
    x = sub.add_parser(
        "import-independent",
        help="Import Spire Codex independent supervised samples as Bootstrap runs",
    )
    x.add_argument("samples", help="independent-training.jsonl(.gz) from spire_codex_data export")
    x.add_argument(
        "--output",
        required=True,
        help="New directory for the accepted shards, index, quarantine and summary",
    )
    x.add_argument("--shard-samples", type=int, default=2000,
                   help="Decision steps per gzip shard; Bootstrap holds a few shards in memory at a time")
    x.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="Processes that convert sample groups; default one for every two logical cores")
    x = sub.add_parser("reindex", help="Recount policy choices in existing demonstration shards")
    x.add_argument("--data", required=True)
    x.add_argument("--workers", type=int, default=4)
    x.add_argument("--listed-only", action="store_true",
                   help="Keep to the runs the present index lists: for a directory that shares "
                        "another's shards and holds a part of them")
    x = sub.add_parser("bootstrap")
    x.add_argument("--data", required=True, nargs="+",
                   help="Data directories read together as one data set (an import of "
                        "demonstrations, or rounds); a "
                        "single path may also be a small run file or directory")
    x.add_argument("--epochs", type=int,
                   help="Optional explicit epoch budget; otherwise continue until interrupted")
    x.add_argument("--window-shards", type=int, default=8,
                   help="Sharded data only: shards read into memory and shuffled together")
    x.add_argument("--tiny", action="store_true")
    x.add_argument("--config", help="Native model/training JSON profile; with a strict resume, "
                                    "its sampling temperatures replace the checkpoint's and it "
                                    "must agree with the checkpoint on the rest")
    x.add_argument("--checkpoint")
    x.add_argument("--learning-rates",
                   help="JSON file of the rates to train at (muon, backbone, embedding, head); "
                        "a resumed checkpoint keeps its own without it")
    x.add_argument("--weights-only", action="store_true",
                   help="Start new BC from checkpoint weights with fresh optimizer, warmup and RNG")
    x.add_argument("--decay-updates", type=int, default=0,
                   help="Finish BC from the given current checkpoint: decay every rate linearly "
                        "to zero over this many updates, then stop")
    x.add_argument("--precision", choices=["no", "bf16"], default="bf16")
    x.add_argument("--max-updates", type=int,
                   help="Stop after this many updates of this run and save the checkpoint, in the "
                        "middle of an epoch if it comes to that: a probe of a setting over a fixed "
                        "number of updates. What was done of the last epoch is saved as an epoch")
    x.add_argument("--include-heldout", action="store_true",
                   help="Train on the held-out seed groups as well; by default they are "
                        "skipped and kept for NLL comparisons (heldout-nll)")
    x.add_argument("--ascension", type=int,
                   help="Train on the runs of this ascension only: the second stage of BC fits "
                        "the A10 demonstrations after the first has fitted all of them")
    x.add_argument("--combat-from", nargs="+", metavar="DATA",
                   help="Take fights from these of the --data directories only; the others give "
                        "their decisions outside combat. The second stage of BC takes its fights "
                        "from the battles the solver fought on the larger budgets")
    x.add_argument("--output", required=True, help="The run's records: status, history, updates")
    x.add_argument("--checkpoints",
                   help="Directory of this run's checkpoints (current/ and the archives); by default "
                        "checkpoints/ followed by the output's path under runs/")
    x = sub.add_parser("heldout-nll",
                       help="Score a checkpoint on the held-out seed groups of a data set: "
                            "NLL per macro-action and per decision by character and phase")
    x.add_argument("--data", required=True, nargs="+",
                   help="Data directories read together, as for bootstrap")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--ascension", type=int, help="Score the held-out runs of this ascension only")
    x.add_argument("--combat-from", nargs="+", metavar="DATA",
                   help="Score the fights of these of the --data directories only, as bootstrap trains on them")
    x.add_argument("--all-groups", action="store_true",
                   help="Score every seed group, not only the held-out ones: for a data set kept out of training "
                        "whole, such as the solver's fights of the fight evaluation set")
    x.add_argument("--window-shards", type=int, default=8,
                   help="Sharded data only: shards read into memory at a time")
    x = sub.add_parser("compare",
                       help="Report one or two evaluation directories: every character, "
                            "the five together and the first seeds of every character; "
                            "two are paired seed by seed with sign tests")
    x.add_argument("directories", nargs="+", help="Output directories of `evaluate`")
    x.add_argument("--head", type=int, default=60,
                   help="Seeds per character of the leading subset reported beside the whole")
    x = sub.add_parser("evaluate")
    x.add_argument("--checkpoint", required=True)
    x.add_argument(
        "--seeds",
        required=True,
        help="JSON mapping all five characters to disjoint seeds",
    )
    x.add_argument(
        "--claim-rewards",
        action="store_true",
        help="Take gold, relics and potions the belt has room for from reward "
             "screens for the policy, as the demonstrations assume",
    )
    x.add_argument("--ascension", type=int,
                   help="Play the runs at this ascension; default the checkpoint's")
    x.add_argument("--precision", choices=sorted(PRECISIONS),
                   help="Precision of the network's global layers: bf16 (autocast) or fp32; default the checkpoint's")
    x.add_argument("--engine-root")
    x.add_argument("--output", required=True)
    x.add_argument("--max-steps", type=int, default=10000)
    x.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Concurrent engine processes sharing batched GPU inference; 64, the fastest "
             "on the training machine, by default (about 280 MB of memory each)",
    )
    x.add_argument(
        "--processes",
        type=int,
        help="Sampling processes that hold the engines, parse their frames and pack "
             "tensors; this process then only runs the network. Default one per "
             "engine, which is fastest",
    )
    x = sub.add_parser(
        "iterate",
        help="Improve a policy by iterations of act groups: sample segments cut at the boundary acts, "
             "several from each start; price each decision against its group; update; evaluate")
    x.add_argument("--checkpoint", required=True,
                   help="The policy to start from, resumed strictly; a run under way goes on "
                        "from its own last update")
    x.add_argument("--config", help="Profile whose sampling temperatures replace the checkpoint's")
    x.add_argument("--wave-shares", type=float, nargs="+",
                   help="The waves' relative shares of a batch, one per wave in the order of the acts; "
                        "a wave's share is its share of the gradient. By the waves' decisions when not "
                        "given, so that a pass sees every wave once")
    x.add_argument("--drop-uniform-groups", action="store_true",
                   help="Leave out of the update the groups whose segments all came to the same return: their "
                        "decisions get no advantage, the baseline is fitted on the other groups, and the shares "
                        "and the pass are of the decisions kept")
    x.add_argument("--output", required=True,
                   help="The records of the iterations (state, rounds' summaries, evaluations); "
                        "an existing one is continued")
    x.add_argument("--checkpoints",
                   help="Directory of this run's checkpoints (current/ and the archives); by default "
                        "checkpoints/ followed by the output's path under runs/")
    x.add_argument("--iterations", type=int, required=True, help="Iterations the output is brought to")
    x.add_argument("--runs-per-character", type=int, default=200, help="Seeds of each character a round starts from")
    x.add_argument("--seed-root", default="iterate", help="What every round's seeds are derived from")
    x.add_argument("--exclude-seeds", nargs="*", default=[],
                   help="Seed files (evaluation and test sets) no round may be played on")
    x.add_argument("--workers", type=int, default=64,
                   help="Engines of a sampled round; 64, the fastest on the training machine, by default")
    x.add_argument("--processes", type=int, help="Sampling processes; one per engine by default")
    x.add_argument("--max-steps", type=int, default=10000)
    x.add_argument("--engine-root")
    x.add_argument("--boundary-acts", type=int, nargs="+", default=[2, 3],
                   help="The acts a run is cut at, each cut a wave played from the save there")
    x.add_argument("--boundary-bosses", type=int, nargs="*", default=[],
                   help="The acts whose boss fight a run is also cut at: the wave from there enters the "
                        "fight again from the save at its entry")
    x.add_argument("--group-n", type=int, default=8, help="Segments from each seed in the first wave")
    x.add_argument("--group-m", type=int, nargs="+", default=[8],
                   help="Segments from each save, one number or one for each later wave")
    x.add_argument("--ascensions", type=int, nargs=2, metavar=("LOWEST", "HIGHEST"),
                   help="Play each (character, seed) group of a round at one ascension of this window, "
                        "the window's ascensions taking turns; the checkpoint's without it. Evaluations "
                        "stay at the checkpoint's")
    x.add_argument("--fork-depth", type=int, default=0,
                   help="Decisions with a choice at the start of every segment at which its group parts in "
                        "two, half taking the policy's most likely candidate and half its second; 0 forks none")
    x.add_argument("--fight-entries", action="store_true",
                   help="Every wave of a round also writes a record of each fight its segments played "
                        "(fights.jsonl) and the entry save of each fight not begun by an event (fights/), "
                        "kept until the iteration's cleanup")
    x.add_argument("--fight-groups", type=int, default=0,
                   help="After a round's act waves, play this many of the fights they recorded again from their "
                        "entry saves, each --fight-group-size times to the fight's end, the fights drawn at the "
                        "fight temperature: normal, elite and boss 3:4:4, part of each kind by deaths and cost "
                        "and the rest evenly. Needs --greedy-fights and --fight-entries; 0 plays none")
    x.add_argument("--fight-group-size", type=int, default=8, help="Segments from each fight of the fight groups")
    x.add_argument("--fight-group-priority", type=float, default=0.5,
                   help="The share of each kind of fight the fight groups pick by priority (deaths first, then "
                        "by cost); the rest is drawn evenly")
    x.add_argument("--fight-group-fork-depth", type=int, default=0,
                   help="Decisions with a choice at the start of every fight segment at which its group parts "
                        "in two, as --fork-depth does for the act groups; 0 forks none")
    x.add_argument("--fight-group-explore-potions", type=float, nargs="+", default=[0.0],
                   help="Potion exploration of the fight groups' segments, given as --explore-potions is")
    x.add_argument("--combat-demos", nargs="+", metavar="DIR",
                   help="Directories of combat demonstrations (imports of the solver's fights) whose fights every "
                        "update learns in the same batches as the act groups, every macro-action by its NLL with no "
                        "KL; held-out seeds included. A pass takes the demonstrations' share of its batches, the "
                        "fights repeated when that is more than they hold and a part of them drawn when less")
    x.add_argument("--round-demos", type=int, default=0,
                   help="The BC phase: after its advantages a round hands this many of the fights it recorded to the "
                        "solver (normal, elite and boss 3:4:4 at their natural frequency, the fights set up by a "
                        "selection first) and its update learns the solver's won fights of the last --demo-buffer "
                        "rounds. Needs --greedy-fights and --fight-entries; 0 hands none")
    x.add_argument("--demo-buffer", type=int, default=3,
                   help="The rounds whose demonstrations an update learns, this one and those before it")
    x.add_argument("--demo-opening", type=float, default=0.2,
                   help="The share of a round's demonstration fights the fights set up by a selection may take first")
    x.add_argument("--demo-workers", type=int, default=16,
                   help="Runs the solver fights at once (two native processes each; one per physical core)")
    x.add_argument("--source-shares", nargs="+", type=source_share, metavar="KIND=SHARE",
                   help="Relative shares of an update's batch by kind of source, naming every kind the update has: "
                        "act-groups, combat-demos (act-groups=1 combat-demos=1 for as many demonstrated "
                        "macro-actions as priced act decisions). By their decisions when not given")
    x.add_argument("--clip", type=float, default=0.2, help="The ratio's clip ε")
    x.add_argument("--kl", type=float, default=0.01, help="The coefficient of the KL to the reference")
    x.add_argument("--reference-every", type=int,
                   help="Renew the KL's reference every this many iterations with the policy of that moment; "
                        "every --evaluate-every iterations by default, 0 keeps --checkpoint as the reference throughout")
    x.add_argument("--learning-rates",
                   help="JSON file of the rates this run trains at (muon, backbone, embedding, head); "
                        "the checkpoint's without it")
    x.add_argument("--min-entropy", type=float,
                   help="Stop after an iteration whose update leaves the policy's mean entropy below this")
    x.add_argument("--explore-potions", type=float, nargs="+", default=[0.0],
                   help="Probability that a sampled fight decision which could use a potion draws one of its "
                        "USE_POTION candidates instead of a sample from the policy: one value for every fight, "
                        "two for normal fights and for elite and boss fights, three for each kind; 0 explores nothing")
    x.add_argument("--greedy-fights", action="store_true",
                   help="Play every decision inside a fight by the policy's most likely action and draw only the "
                        "rest: the fights are kept without frames, and the advantages and the update are of the "
                        "decisions outside them. Not with --explore-potions or --boundary-bosses")
    x.add_argument("--updates", default="round",
                   help="Updates per iteration: `round` (a pass over the round) or a number of optimizer steps")
    x.add_argument("--batch", type=int,
                   help="Macro-actions of one optimizer step, a multiple of the profile's logical batch whose "
                        "batches are accumulated into it; the logical batch by default. The rates stay as given")
    x.add_argument("--weight-cap", type=float, default=10.0,
                   help="What one sampled decision of a small character and phase group may weigh, "
                        "in even shares of its wave; 0 for no cap")
    x.add_argument("--phase-weights", choices=["group", "even"], default="group",
                   help="How a wave's decisions share its weight: every (character, phase) group alike, "
                        "or every decision alike")
    x.add_argument("--window-shards", type=int, default=8)
    x.add_argument("--evaluate-every", type=int, default=5, help="Iterations between evaluations")
    x.add_argument("--eval-seeds", help="Seed file of the evaluations; without it none is made")
    x.add_argument("--eval-runs-per-character", type=int,
                   help="The leading seeds of each character an evaluation plays; all by default")
    x.add_argument("--eval-workers", type=int, help="Engines of an evaluation; as a round's by default")
    x.add_argument("--eval-precision", choices=sorted(PRECISIONS),
                   help="Precision of the evaluations' network: bf16 (autocast) or fp32; the checkpoint's by "
                        "default. Sampling keeps the checkpoint's")
    x = sub.add_parser("fight-set",
                       help="Draw a fight evaluation set from a directory of entry saves: normal, elite and boss "
                            "3:4:4, evenly within each kind, seeded by --root")
    x.add_argument("--sources", required=True, nargs="+",
                   help="Directories of entry saves, each naming its run's character, seed and ascension")
    x.add_argument("--exclude", nargs="*", default=[],
                   help="Selections (selection.json of entry_battles) whose fights the set leaves out")
    x.add_argument("--count", type=int, default=440, help="Fights of the set")
    x.add_argument("--root", required=True, help="What the draws are seeded by")
    x.add_argument("--output", required=True, help="New directory of the set: fights.json, seeds.json and saves/")
    x = sub.add_parser("fight-eval",
                       help="Play every fight of a fight evaluation set greedily from its entry save to its end; "
                            "report the cost by kind of fight and, against the solver's fights, the gap and the "
                            "table of behavior")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--fights", required=True, help="Directory of the fight evaluation set (fight-set)")
    x.add_argument("--solver",
                   help="Output of `spire_codex_data.entry_battles run` on the set; the set's solver/ if it has one")
    x.add_argument("--precision", choices=sorted(PRECISIONS),
                   help="Precision of the network's global layers: bf16 (autocast) or fp32; default the checkpoint's")
    x.add_argument("--per-kind", type=int, help="Play only the first this many fights of each kind: a trial")
    x.add_argument("--engine-root")
    x.add_argument("--output", required=True, help="New directory: fights.jsonl, metadata.json, report.json, report.txt")
    x.add_argument("--max-steps", type=int, default=10000)
    x.add_argument("--workers", type=int, default=8,
                   help="Concurrent engine processes sharing batched GPU inference (about 280 MB of memory each); "
                        "8 by default, half the training machine's physical cores")
    x.add_argument("--processes", type=int, help="Sampling processes; one per engine by default")
    x = sub.add_parser("fight-compare",
                       help="Report fight evaluations again from their fights: one against the solver's fights, "
                            "two of the same set also fight by fight")
    x.add_argument("results", nargs="+", help="Output directories of fight-eval")
    x.add_argument("--fights", help="The set's directory; the first evaluation's by default")
    x.add_argument("--solver", help="Output of `entry_battles run` on the set; the set's solver/ if it has one")
    x.add_argument("--text", action="store_true", help="Print the tables instead of the JSON")
    x = sub.add_parser("infer")
    x.add_argument("--checkpoint", required=True)
    x.add_argument(
        "--frame",
        required=True,
        help="JSON decision frame, or JSONL frames to replay a session",
    )
    x.add_argument("--top-k", type=int, default=5)
    x = sub.add_parser(
        "play-steam", help="Control a running Steam game through steam_recorder"
    )
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--bridge-dir", required=True)
    x.add_argument("--run-for", type=parse_duration,
                   help="How long to stay beside the game, such as 8h, 45m or 90s; default until interrupted")
    x.add_argument("--request-timeout", type=float, default=120,
                   help="Seconds to wait for the game to answer one bridge request")
    x.add_argument("--max-decisions", type=int, help="Stop after this many decisions; default no limit")
    x = sub.add_parser("inspect")
    x.add_argument("--engine-root")
    x.add_argument("--character", choices=CHARACTERS, default="Ironclad")
    x.add_argument("--game-seed", default="protocol-inspect")
    x.add_argument("--ascension", type=int, default=0)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        runtime = configure_runtime(args.device, threads=args.threads)
        return COMMANDS[args.command](args, runtime) or 0
    except KeyboardInterrupt:
        return 130
    except (ProtocolError, ValueError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
