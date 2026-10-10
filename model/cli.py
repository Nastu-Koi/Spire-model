import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .buffer import sample_round
from .checkpoint import archive_checkpoint, load_model, restore_training, save_checkpoint
from .config import ModelConfig, TrainConfig
from .control import environment_action
from .data import HELDOUT_FRACTION, RunShards, audit_files, extent, load_runs, policy_macros, split_held_out
from .engine import CliEngine
from .evaluation import compare_evaluations, evaluate_runs, nll_report
from .history import TrainingHistory
from .model import PolicyNetwork
from .policy import SessionPolicy
from .protocol import (
    CHARACTERS,
    ProtocolError,
    execution_command,
    fingerprint,
    validate_frame,
)
from .representation import Vocabulary
from .rewards import MilestoneLedger
from .rollout import RolloutRunner, collect_round, precision_context, write_run
from .sampling import Samplers
from .runtime import configure_runtime
from .seeds import validate_evaluation_seeds
from .trainer import Learner


def vocabulary_for(runs, config):
    frames = (s["frame"] for r in runs for m in policy_macros(r) for s in m["steps"])
    return Vocabulary.from_frames(frames, config)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def smoke(output):
    from .testing import SyntheticEngine, demonstration

    config = ModelConfig.tiny()
    training = TrainConfig(
        precision="no",
        logical_batch_size=4,
        token_buckets=[32, 64],
        action_buckets=[8, 16],
    )
    runs = [demonstration(c, "smoke-demo", 4, 2) for c in CHARACTERS]
    vocabulary = vocabulary_for(runs, config)
    model = PolicyNetwork(config)
    learner = Learner(model, vocabulary, training)
    result = {
        "domain": "synthetic_test_only",
        "parameters": model.parameter_report(),
        "bootstrap": learner.bootstrap(runs),
    }
    rollout = RolloutRunner(model, vocabulary, version=learner.policy_version,
                            temperatures=training.temperatures)
    sampled = [
        rollout.run(SyntheticEngine(4, 2), c, "smoke-rollout") for c in CHARACTERS
    ]
    result["evaluation"] = evaluate_runs(sampled)
    result["policy_version"] = learner.policy_version
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    for run in sampled:
        write_run(path / (run["character"] + ".json"), run)
    save_checkpoint(
        path / "checkpoint",
        model,
        vocabulary,
        learner.optimizer,
        learner.scheduler,
        training=training,
        progress={"policy_version": learner.policy_version, "updates": learner.updates},
    )
    loaded, vocab, _ = load_model(path / "checkpoint")
    with torch.no_grad():
        from .policy import replay

        before = replay(model, vocabulary, runs[0]["macros"][0]["steps"])[0]
        after = replay(loaded, vocab, runs[0]["macros"][0]["steps"])[0]
    result["checkpoint_log_prob_error"] = abs(float(before - after))
    (path / "report.json").write_text(json.dumps(result, indent=2))
    return result


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
    x.add_argument(
        "--architecture", help="Read the model dimensions from architecture JSON"
    )
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
    x = sub.add_parser("bootstrap")
    x.add_argument("--data", required=True, nargs="+",
                   help="Data directories read together as one data set (an import of "
                        "demonstrations and imports of student battles, or rounds); a "
                        "single path may also be a small run file or directory")
    x.add_argument("--epochs", type=int,
                   help="Optional explicit epoch budget; otherwise continue until interrupted")
    x.add_argument("--window-shards", type=int, default=8,
                   help="Sharded data only: shards read into memory and shuffled together")
    x.add_argument("--tiny", action="store_true")
    x.add_argument("--architecture")
    x.add_argument("--config", help="Native model/training JSON profile; with a strict resume, "
                                    "its sampling temperatures replace the checkpoint's and it "
                                    "must agree with the checkpoint on the rest")
    x.add_argument("--checkpoint")
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
    x.add_argument("--output", required=True)
    x = sub.add_parser("heldout-nll",
                       help="Score a checkpoint on the held-out seed groups of a data set: "
                            "NLL per macro-action and per decision by character and phase")
    x.add_argument("--data", required=True, nargs="+",
                   help="Data directories read together, as for bootstrap")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--window-shards", type=int, default=8,
                   help="Sharded data only: shards read into memory at a time")
    x = sub.add_parser("compare",
                       help="Report one or two evaluation directories: every character, "
                            "the five together and the first seeds of every character; "
                            "two are paired seed by seed with sign tests")
    x.add_argument("directories", nargs="+", help="Output directories of `evaluate`")
    x.add_argument("--head", type=int, default=60,
                   help="Seeds per character of the leading subset reported beside the whole")
    for command in ("collect", "evaluate"):
        x = sub.add_parser(command)
        x.add_argument("--checkpoint", required=True)
        x.add_argument(
            "--seeds",
            required=True,
            help="JSON mapping all five characters to disjoint seeds",
        )
        if command == "evaluate":
            x.add_argument(
                "--claim-rewards",
                action="store_true",
                help="Take gold, relics and potions the belt has room for from reward "
                     "screens for the policy, as the demonstrations assume",
            )
        else:
            x.add_argument(
                "--config",
                help="Native profile whose sampling temperatures replace the checkpoint's "
                     "for this round; it must agree with the checkpoint on the rest",
            )
            x.add_argument("--iteration", type=int,
                           help="The sampling iteration the round belongs to, recorded in every run "
                                "and index row; default the round number")
            x.add_argument("--journals", action="store_true",
                           help="Also write each run's journal (every frame with its reward) beside "
                                "its fights; sampled rounds keep only what training reads otherwise")
            x.add_argument("--shard-samples", type=int, default=2000,
                           help="Stored decision steps per gzip shard of the round")
        x.add_argument("--engine-root")
        x.add_argument("--output", required=True)
        x.add_argument("--max-steps", type=int, default=10000)
        x.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Concurrent engine processes sharing batched GPU inference",
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
        help="Improve a policy by iterations: sample a round, label its fights with the "
             "solver, continue the critic, update by advantage-weighted regression, evaluate")
    x.add_argument("--checkpoint", required=True,
                   help="The policy to start from, resumed strictly; a run under way goes on "
                        "from its own last update")
    x.add_argument("--demonstrations", required=True, nargs="+",
                   help="Data directories of the supervised half of every update, the largest first")
    x.add_argument("--output", required=True, help="Everything the iterations write; an existing one is continued")
    x.add_argument("--iterations", type=int, required=True, help="Iterations the output is brought to")
    x.add_argument("--runs-per-character", type=int, default=200, help="A round is five times this many runs")
    x.add_argument("--seed-root", default="iterate", help="What every round's seeds are derived from")
    x.add_argument("--exclude-seeds", nargs="*", default=[],
                   help="Seed files (evaluation and test sets) no round may be played on")
    x.add_argument("--buffer-iterations", type=int, default=10,
                   help="The run buffer: the rounds of the last so many iterations")
    x.add_argument("--workers", type=int, default=1, help="Engines of a sampled round")
    x.add_argument("--processes", type=int, help="Sampling processes; one per engine by default")
    x.add_argument("--max-steps", type=int, default=10000)
    x.add_argument("--engine-root")
    x.add_argument("--no-labels", action="store_true", help="Do not label fights with the solver")
    x.add_argument("--label-fraction", type=float, default=0.25,
                   help="The share of a round's runs whose every fight the solver labels")
    x.add_argument("--label-workers", type=int, default=8, help="Runs the solver fights at once")
    x.add_argument("--solver-config", help="Solver worker configuration for the labels")
    x.add_argument("--labels-in-background", action="store_true",
                   help="Let the solver label a round beside the critic's training and the "
                        "update; its labels then enter the next iteration's update")
    x.add_argument("--advantage", choices=["critic", "return"], default="critic",
                   help="`return` is the control without a critic: the run's return minus the "
                        "mean return of its character's runs")
    x.add_argument("--lam", type=float, default=0.9, help="λ of the generalized advantage")
    x.add_argument("--critic", help="A saved critic to continue from, instead of starting a new one")
    x.add_argument("--critic-config", help="Profile whose model and critic sections make the critic")
    x.add_argument("--critic-gate", choices=["on", "off"], default="on",
                   help="Whether a critic that fails the gate leaves the policy as it is")
    x.add_argument("--gate", action="append", default=[], metavar="NAME=VALUE",
                   help="Override one of the gate's floors (critic/gate.py); may be repeated")
    x.add_argument("--beta", default="auto",
                   help="β of the weights exp(A/β): `auto` tunes it to every batch, `inf` is no "
                        "weighting, or a number")
    x.add_argument("--ess-share", type=float, default=0.3,
                   help="The effective sample size a tuned β keeps, as a share of the batch")
    x.add_argument("--beta-min", type=float, default=0.1)
    x.add_argument("--beta-max", type=float, default=100.0)
    x.add_argument("--updates", default="round",
                   help="Updates per iteration: `round` (as many as a pass over the newest round "
                        "makes, drawn from the whole buffer), `buffer` (a pass over the buffer) "
                        "or a number")
    x.add_argument("--weight-cap", type=float, default=10.0,
                   help="What one sampled decision of a small character and phase group may weigh "
                        "before its advantage, in even shares of the buffer; 0 for no cap")
    x.add_argument("--window-shards", type=int, default=8)
    x.add_argument("--evaluate-every", type=int, default=5, help="Iterations between evaluations")
    x.add_argument("--eval-seeds", help="Seed file of the evaluations; without it none is made")
    x.add_argument("--eval-runs-per-character", type=int,
                   help="The leading seeds of each character an evaluation plays; all by default")
    x.add_argument("--eval-workers", type=int, help="Engines of an evaluation; as a round's by default")
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
    x.add_argument("--timeout", type=float, default=120)
    x.add_argument("--max-decisions", type=int, default=10000)
    x = sub.add_parser("inspect")
    x.add_argument("--engine-root")
    x.add_argument("--character", choices=CHARACTERS, default="Ironclad")
    x.add_argument("--game-seed", default="protocol-inspect")
    x.add_argument("--ascension", type=int, default=0)
    return p


def _learner(checkpoint, device, *, weights_only=False, training=None, decay_updates=0, inference=False):
    """A checkpoint's policy with its training state. `inference` loads it to play only:
    nothing is resumed, so a checkpoint whose training has ended, a finished decay among
    them, is accepted."""
    model, vocab, manifest = load_model(checkpoint, device, allow_legacy_encoding=weights_only)
    progress = manifest.get("progress", {})
    saved_training = TrainConfig.from_dict(manifest["training"])
    if not weights_only and training is not None:
        # The sampling settings are the profile's; it may differ from the checkpoint in nothing else.
        training = saved_training.with_sampling(training)
    if weights_only:
        vocab = vocab.expanded()
    learner = Learner(
        model,
        vocab,
        training or saved_training,
        policy_version=0 if weights_only else progress.get("policy_version", 0),
        decay_updates=decay_updates,
    )
    if weights_only:
        learner.checkpoint_metadata = {
            "training_seeds": progress.get("training_seeds", []),
            "initialization": "weights-only",
            "source_checkpoint": str(Path(checkpoint).resolve()),
            "source_updates": progress.get("updates", 0),
        }
        return learner, manifest
    if inference:
        learner.updates = progress.get("updates", 0)
    else:
        # Whole-run PPO, which has been removed, left its rounds in the progress: such
        # a policy was not shaped by demonstrations alone, and its optimizer moments are
        # not those of a BC run.
        if "sampling_round" in progress or progress.get("completed_ppo_rounds"):
            raise ValueError("Checkpoint was trained with PPO, which has been removed; it cannot "
                             "be resumed. Start new training from its weights with --weights-only, "
                             "or play it as it is")
        if "optimizer" not in manifest:
            raise ValueError("Checkpoint has no optimizer state; use --weights-only")
        progress = restore_training(checkpoint, learner.optimizer, learner.scheduler)
        learner.updates = progress.get("updates", 0)
    learner.checkpoint_metadata = {
        k: v
        for k, v in manifest["progress"].items()
        if k not in {"policy_version", "updates"}
    }
    return learner, manifest


def _save(path, learner, extra=None):
    previous = getattr(learner, "checkpoint_metadata", {})
    metadata = {**previous, **(extra or {})}
    metadata["training_seeds"] = sorted(
        set(previous.get("training_seeds", []))
        | set(metadata.get("training_seeds", []))
    )
    save_checkpoint(
        path,
        learner.model,
        learner.vocabulary,
        learner.optimizer,
        learner.scheduler,
        overwrite=Path(path).name == "current",
        training=learner.config,
        progress={
            **metadata,
            "reward_version": MilestoneLedger().version,
            "policy_version": learner.policy_version,
            "updates": learner.updates,
        },
    )
    learner.checkpoint_metadata = metadata


def _data_metadata(runs):
    contracts = {fingerprint(r["contract"]): r["contract"] for r in runs}
    return {
        "training_engine_contracts": list(contracts.values()),
        "training_seeds": sorted({str(r["seed"]) for r in runs}),
        "training_seed_pool_hash": fingerprint(
            sorted((r["character"], str(r["seed"])) for r in runs)
        ),
    }


def _profile(path):
    """The training section of a native profile, or None without a path."""
    return TrainConfig.from_dict(json.loads(Path(path).read_text())["training"]) if path else None


def _dataset(paths):
    """The runs `--data` names: sharded directories are read together as one data set
    (RunShards); one path that is not sharded is a small run file or directory read
    into memory. Returns (runs, whether they are sharded)."""
    if all(RunShards.is_at(path) for path in paths):
        return RunShards(paths), True
    if len(paths) == 1:
        return load_runs(paths[0]), False
    raise ProtocolError("Several --data paths must all be sharded data directories (with an index)")


def main(argv=None):
    args = parser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    history = samplers = None
    try:
        runtime = configure_runtime(args.device, threads=args.threads)
        if args.command == "play-steam":
            from .steam import play

            emit(
                play(
                    args.checkpoint,
                    args.bridge_dir,
                    args.device,
                    args.timeout,
                    args.max_decisions,
                )
            )
        elif args.command == "parameters":
            config = (
                ModelConfig(**json.loads(Path(args.config).read_text())["model"])
                if args.config
                else ModelConfig.from_architecture(args.architecture)
                if args.architecture
                else ModelConfig()
            )
            with torch.device("meta"):
                model = PolicyNetwork(config)
            emit(model.parameter_report())
        elif args.command == "smoke":
            emit(smoke(args.output))
        elif args.command == "init":
            profile = json.loads(Path(args.config).read_text())
            config, training = (
                ModelConfig(**profile["model"]),
                TrainConfig.from_dict(profile["training"]),
            )
            frames = []
            for character in CHARACTERS:
                with CliEngine(root=args.engine_root) as engine:
                    frames.append(
                        validate_frame(
                            engine.reset(character, "vocabulary-" + character, training.ascension)
                        )
                    )
                    if len(frames) == 1:
                        catalog = engine.send({"cmd": "public_catalog"})
                        if catalog.get("type") != "public_catalog":
                            raise ProtocolError(
                                "Engine does not provide a static public vocabulary catalog"
                            )
                        frames.append(catalog)
            vocabulary = Vocabulary.from_frames(frames, config)
            with torch.device(args.device):
                model = PolicyNetwork(config)
            learner = Learner(model, vocabulary, training)
            _save(
                args.output,
                learner,
                {
                    "runtime": runtime,
                    "engine_contract": frames[0]["contract"],
                    "initialization": "random-public-catalog",
                },
            )
            emit(
                {
                    "checkpoint": args.output,
                    "vocabulary_symbols": len(vocabulary.symbols),
                    "parameters": model.parameter_report(),
                }
            )
        elif args.command == "audit":
            emit(audit_files(args.paths, args.accepted, args.quarantine))
        elif args.command == "import-recorder":
            from .recorder import import_recordings

            summary = import_recordings(
                args.paths,
                args.output,
                bc_only=args.bc_only,
                recorder_version=args.recorder_version,
            )
            emit(summary)
            if not summary["accepted_runs"]:
                return 2
        elif args.command == "import-independent":
            from .independent import import_samples

            summary = import_samples(args.samples, args.output, args.shard_samples, args.workers)
            emit(summary)
            if not summary["accepted_runs"]:
                return 2
        elif args.command == "reindex":
            from .dataset_index import rebuild_index

            emit(rebuild_index(args.data, workers=args.workers))
        elif args.command == "inspect":
            with CliEngine(root=args.engine_root) as engine:
                frame = engine.reset(args.character, args.game_seed, args.ascension)
            emit(frame)
        elif args.command == "bootstrap":
            if args.weights_only and not args.checkpoint:
                raise ValueError("--weights-only requires --checkpoint")
            if args.epochs is not None and args.epochs < 1:
                raise ValueError("Bootstrap needs at least one epoch")
            if args.decay_updates < 0:
                raise ValueError("Decay updates must be non-negative")
            if args.max_updates is not None and args.max_updates < 1:
                raise ValueError("--max-updates is at least one update")
            if args.decay_updates and (not args.checkpoint or args.weights_only):
                raise ValueError("--decay-updates continues a checkpoint's optimizer: "
                                 "pass its current checkpoint without --weights-only")
            output = Path(args.output)
            if ((output / "current").exists() or (output / "history.jsonl").exists()
                    or (output / "updates.jsonl").exists()
                    or any(output.glob("epoch-*"))):
                if (args.weights_only or not args.checkpoint
                        or Path(args.checkpoint).resolve() != (output / "current").resolve()):
                    raise ValueError("Existing BC output requires resume from its current checkpoint; use a new output directory")
            runs, streamed = _dataset(args.data)
            if streamed and not args.checkpoint:
                raise ProtocolError(
                    "Sharded data is read as a stream and cannot rebuild a vocabulary: "
                    "create a checkpoint with `init` and pass --checkpoint"
                )
            if not runs:
                report = Path(args.data[0]).parent / "summary.json"
                detail = f" See import report: {report}" if report.exists() else ""
                raise ProtocolError(
                    f"No accepted behavior-cloning samples: {args.data} contains zero runs."
                    " Import eligible recordings before starting Bootstrap." + detail
                )
            if not (runs.counts if streamed else any(next(policy_macros(r), None) for r in runs)):
                raise ProtocolError(
                    "No accepted behavior-cloning samples in the supplied data"
                )
            profile = json.loads(Path(args.config).read_text()) if args.config else None
            if args.checkpoint:
                learner, _ = _learner(args.checkpoint, args.device,
                                      weights_only=args.weights_only,
                                      training=TrainConfig.from_dict(profile["training"]) if profile else None,
                                      decay_updates=args.decay_updates)
                if profile and ModelConfig(**profile["model"]) != learner.model.config:
                    raise ValueError("A checkpoint's model architecture cannot be changed by --config")
            else:
                config = (
                    ModelConfig(**profile["model"])
                    if profile
                    else ModelConfig.tiny()
                    if args.tiny
                    else ModelConfig.from_architecture(args.architecture)
                    if args.architecture
                    else ModelConfig()
                )
                vocab = vocabulary_for(runs, config)
                with torch.device(args.device):
                    model = PolicyNetwork(config)
                learner = Learner(
                    model,
                    vocab,
                    TrainConfig.from_dict(profile["training"])
                    if profile
                    else TrainConfig(precision=args.precision),
                )
            history = TrainingHistory(args.output, "bootstrap", asdict(learner.config))
            start_epoch = getattr(learner, "checkpoint_metadata", {}).get(
                "completed_bootstrap_epochs", 0
            )
            metadata = {"runtime": runtime, **(runs.metadata() if streamed else _data_metadata(runs))}
            epoch_started = time.monotonic()
            completed = 0

            def update_done(metrics, *, epoch, stage):
                history.append_update(metrics, round=start_epoch + epoch, stage=stage,
                                      optimization_epoch=epoch, policy_version=learner.policy_version)

            def epoch_done(index, metrics):
                nonlocal epoch_started, completed
                epoch = start_epoch + index
                archived = output / f"epoch-{epoch:06d}"
                if archived.exists():
                    raise FileExistsError("Epoch checkpoint already exists; use a new output directory")
                record = history.append(
                    epoch,
                    metrics,
                    training_runs=metrics["training"]["groups"],
                    duration_seconds=time.monotonic() - epoch_started,
                    policy_version=learner.policy_version,
                )
                _save(
                    output / "current",
                    learner,
                    {
                        **metadata,
                        "completed_bootstrap_epochs": epoch,
                        "held_out": metrics["held_out"],
                        "last_training_record": record,
                    },
                )
                # Retain completed epochs for manual comparison. Hard links survive
                # the next atomic replacement of current without copying tensors.
                archive_checkpoint(output / "current", archived)
                emit(record)
                completed = index
                epoch_started = time.monotonic()

            history.status("updating", round=start_epoch + 1, updates=learner.updates)
            learner.bootstrap(runs, args.epochs, on_epoch=epoch_done, window_shards=args.window_shards,
                              on_update=update_done, include_held_out=args.include_heldout,
                              max_updates=args.max_updates)
            history.status("completed", round=start_epoch + completed)
        elif args.command == "heldout-nll":
            runs, _ = _dataset(args.data)
            learner, _ = _learner(args.checkpoint, args.device, inference=True)
            _, held = split_held_out(runs)
            if not extent(held)["macros"]:
                raise ProtocolError("The data set has no macro-actions in its held-out seed groups")
            groups = learner.evaluate_bootstrap(held, window_shards=args.window_shards, include_a0=True)
            emit({
                "data": [str(path) for path in args.data],
                "checkpoint": str(args.checkpoint),
                "policy_version": learner.policy_version,
                "updates": learner.updates,
                "held_out": {"fraction": HELDOUT_FRACTION, **extent(held), "of": extent(runs)},
                **nll_report(groups),
            })
        elif args.command == "compare":
            emit(compare_evaluations(args.directories, head=args.head))
        elif args.command == "iterate":
            from .iterate import Driver, Settings

            settings = Settings(
                checkpoint=args.checkpoint, demonstrations=args.demonstrations,
                runs_per_character=args.runs_per_character, seed_root=args.seed_root,
                exclude_seeds=args.exclude_seeds, buffer_iterations=args.buffer_iterations,
                workers=args.workers, processes=args.processes, max_steps=args.max_steps,
                engine_root=args.engine_root, labels=not args.no_labels, label_fraction=args.label_fraction,
                label_workers=args.label_workers, labels_in_background=args.labels_in_background,
                solver_config=args.solver_config, advantage=args.advantage, lam=args.lam, critic=args.critic,
                critic_config=args.critic_config, critic_gate=args.critic_gate == "on", gate=args.gate,
                beta=args.beta, ess_share=args.ess_share, beta_min=args.beta_min, beta_max=args.beta_max,
                updates=args.updates, weight_cap=args.weight_cap, window_shards=args.window_shards,
                evaluate_every=args.evaluate_every,
                eval_seeds=args.eval_seeds, eval_runs_per_character=args.eval_runs_per_character,
                eval_workers=args.eval_workers)
            state = Driver(args.output, settings, args.device, runtime=runtime).run(args.iterations)
            emit({"output": str(args.output), "iterations": sum(
                bool(entry.get("complete")) for entry in state["iterations"].values())})
        elif args.command == "infer":
            model, vocabulary, manifest = load_model(args.checkpoint, args.device)
            model.eval()
            raw = Path(args.frame).read_text()
            frames = (
                [json.loads(x) for x in raw.splitlines() if x.strip()]
                if args.frame.endswith(".jsonl")
                else [json.loads(raw)]
            )
            policy = SessionPolicy(model, vocabulary)
            with (
                torch.no_grad(),
                precision_context(model, manifest["training"]["precision"]),
            ):
                for frame in frames:
                    validate_frame(frame)
                    automatic = environment_action(frame)
                    if automatic is not None:
                        policy.reset()
                        emit(dict(command=execution_command(frame, automatic["candidate_ref"]),
                                  ranking=None, control=automatic))
                        continue
                    choice = policy.choose(frame, sample=False, top_k=args.top_k)
                    emit(
                        {
                            "command": execution_command(frame, choice.candidate_ref),
                            "ranking": choice.ranking,
                        }
                    )
        else:
            # collect and evaluate: the policy plays, in the engines of sampling processes.
            if args.workers < 1 or (args.processes is not None and args.processes < 1):
                raise ValueError("Sampling needs positive numbers of engines and processes")
            learner, manifest = _learner(
                args.checkpoint, args.device, inference=True,
                training=_profile(args.config) if args.command == "collect" else None,
            )
            seeds = json.loads(Path(args.seeds).read_text())
            if args.command == "evaluate":
                validate_evaluation_seeds(seeds, manifest["progress"])
            directory = Path(args.output)
            directory.mkdir(parents=True, exist_ok=True)
            # Never overwrite a round from an interrupted or previous invocation.
            existing = [
                int(p.name[6:])
                for p in directory.glob("round-*")
                if p.name[6:].isdigit()
            ]
            round_number = max(existing, default=-1) + 1
            runner = RolloutRunner(
                learner.model,
                learner.vocabulary,
                precision=learner.config.precision,
                version=learner.policy_version,
                max_steps=args.max_steps,
                ascension=learner.config.ascension,
                claim_rewards=args.command == "evaluate" and args.claim_rewards,
                temperatures=learner.config.temperatures,
                iteration=None if args.command == "evaluate"
                else round_number if args.iteration is None else args.iteration,
            )
            samplers = Samplers(lambda: CliEngine(root=args.engine_root), learner.model,
                                learner.vocabulary, engines=args.workers, processes=args.processes)
            if args.command == "evaluate":
                paths = collect_round(samplers, runner, seeds, directory, sample=False)
                # One run in memory at a time: together they are tens of gigabytes.
                emit(evaluate_runs(json.loads(p.read_text()) for p in paths))
            else:
                # The round as training reads it: shards and index under round-N.
                emit(sample_round(samplers, runner, seeds, directory / f"round-{round_number}",
                                  journals=args.journals, shard_samples=args.shard_samples))
    except KeyboardInterrupt:
        if history:
            history.status("interrupted")
        return 130
    except (ProtocolError, ValueError, OSError, RuntimeError) as exc:
        if history:
            history.status("failed", error=str(exc))
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if samplers:
            samplers.close()
        if history:
            history.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
