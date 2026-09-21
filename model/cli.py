import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

import torch

from .checkpoint import load_model, restore_training, save_checkpoint
from .config import ModelConfig, TrainConfig
from .data import audit_files, load_runs, split_runs
from .engine import CliEngine
from .history import TrainingHistory
from .model import PolicyValue
from .policy import SessionPolicy
from .protocol import CHARACTERS, ProtocolError, execution_command, fingerprint, validate_frame
from .rewards import MilestoneLedger
from .representation import Vocabulary, symbols_from_frames
from .rollout import RolloutRunner, collect_round, precision_context, write_run
from .runtime import configure_runtime
from .seeds import RandomSeedSchedule
from .trainer import Learner, evaluate_runs


def vocabulary_for(runs, capacity):
    frames = (s["frame"] for r in runs for m in r["macros"] for s in m["steps"])
    return Vocabulary(symbols_from_frames(frames), capacity)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def smoke(output):
    from .testing import SyntheticEngine, demonstration
    config = ModelConfig.tiny()
    training = TrainConfig(precision="no", logical_batch_size=4, token_buckets=[32, 64], action_buckets=[8, 16])
    runs = [demonstration(c, "smoke-demo", 4, 2) for c in CHARACTERS]
    vocabulary = vocabulary_for(runs, config.vocabulary_size)
    model = PolicyValue(config)
    learner = Learner(model, vocabulary, training)
    result = {"domain": "synthetic_test_only", "parameters": model.parameter_report(), "bootstrap": learner.bootstrap(runs)}
    rollout = RolloutRunner(model, vocabulary, version=learner.policy_version)
    sampled = [rollout.run(SyntheticEngine(4, 2), c, "smoke-rollout") for c in CHARACTERS]
    result["evaluation"] = evaluate_runs(sampled)
    result["ppo"] = learner.ppo(sampled)
    result["policy_version"] = learner.policy_version
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    for run in sampled:
        write_run(path / (run["character"] + ".json"), run)
    save_checkpoint(path / "checkpoint", model, vocabulary, learner.optimizer, learner.scheduler,
                    training=training, progress={"policy_version": learner.policy_version, "updates": learner.updates})
    loaded, vocab, _ = load_model(path / "checkpoint")
    with torch.no_grad():
        from .policy import replay
        before = replay(model, vocabulary, runs[0]["macros"][0]["steps"])[0]
        after = replay(loaded, vocab, runs[0]["macros"][0]["steps"])[0]
    result["checkpoint_log_prob_error"] = abs(float(before - after))
    (path / "report.json").write_text(json.dumps(result, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description="STS2 hybrid policy: data → Bootstrap → PPO → Steam")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("smoke", help="Synthetic Bootstrap/rollout/PPO/checkpoint end-to-end check")
    x.add_argument("--output", required=True)
    x = sub.add_parser("parameters")
    x.add_argument("--architecture", help="Read the model dimensions from architecture JSON")
    x = sub.add_parser("init", help="Create random weights and a frozen vocabulary from public engine content")
    x.add_argument("--config", required=True)
    x.add_argument("--engine-root")
    x.add_argument("--output", required=True)
    x = sub.add_parser("audit")
    x.add_argument("paths", nargs="+")
    x.add_argument("--accepted", required=True)
    x.add_argument("--quarantine", required=True)
    x = sub.add_parser("import-recorder", help="Import Steam RunRecorder JSONL with complete-run and visibility checks")
    x.add_argument("--bootstrap-only", "--bc-only", dest="bc_only", action="store_true", help="Use valid recorded decisions for Bootstrap, including solver and partial runs; retain unverified provenance")
    x.add_argument("--recorder-version", help="Import only journals with this exact recorder version")
    x.add_argument("paths", nargs="+", help="Journal files, globs, or recorder directories")
    x.add_argument("--output", required=True, help="New directory for accepted data, quarantine and summary")
    x = sub.add_parser("bootstrap")
    x.add_argument("--data", required=True)
    validation = x.add_mutually_exclusive_group()
    validation.add_argument("--validation")
    validation.add_argument("--all-training-data", action="store_true", help="Use all supplied runs for Bootstrap without an automatic validation holdout")
    x.add_argument("--epochs", type=int, default=1)
    x.add_argument("--tiny", action="store_true")
    x.add_argument("--architecture")
    x.add_argument("--config", help="Native model/training JSON profile")
    x.add_argument("--checkpoint")
    x.add_argument("--precision", choices=["no", "bf16"], default="bf16")
    x.add_argument("--output", required=True)
    for command in ("collect", "evaluate", "train"):
        x = sub.add_parser(command)
        x.add_argument("--checkpoint", required=True)
        if command == "train":
            source = x.add_mutually_exclusive_group()
            source.add_argument("--seeds", help="Use a fixed JSON seed queue instead of random generation")
            source.add_argument("--random-seeds", action="store_true", help="Generate fresh game seeds every round (default)")
            x.add_argument("--runs-per-character", type=int,
                           help="Random-mode games per character per round (default 4, or restored schedule)")
        else:
            x.add_argument("--seeds", required=True, help="JSON mapping all five characters to disjoint seeds")
        x.add_argument("--engine-root")
        x.add_argument("--output", required=True)
        x.add_argument("--max-steps", type=int, default=10000)
        if command == "train":
            x.add_argument("--rounds", type=int, default=160,
                           help="Additional sampling/update rounds in this invocation (default 160)")
            x.add_argument("--demonstrations")
            x.add_argument("--value-warmup", action="store_true")
    x = sub.add_parser("ppo")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--data", required=True)
    x.add_argument("--output", required=True)
    x = sub.add_parser("infer")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--frame", required=True, help="JSON decision frame, or JSONL frames to replay a session")
    x.add_argument("--top-k", type=int, default=5)
    x = sub.add_parser("monitor", help="Live Bootstrap/PPO training dashboard")
    x.add_argument("--root", default="runs")
    x.add_argument("--host", default="127.0.0.1")
    x.add_argument("--port", type=int, default=8765)
    x = sub.add_parser("play-steam", help="Control a running Steam game through steam_recorder")
    x.add_argument("--checkpoint", required=True)
    x.add_argument("--bridge-dir", required=True)
    x.add_argument("--timeout", type=float, default=120)
    x.add_argument("--max-decisions", type=int, default=10000)
    x = sub.add_parser("inspect")
    x.add_argument("--engine-root")
    x.add_argument("--character", choices=CHARACTERS, default="Ironclad")
    x.add_argument("--game-seed", default="protocol-inspect")
    return p


def _learner(checkpoint, device):
    model, vocab, manifest = load_model(checkpoint, device)
    learner = Learner(model, vocab, TrainConfig.from_dict(manifest["training"]),
                      policy_version=manifest["progress"].get("policy_version", 0))
    if "optimizer" in manifest:
        progress = restore_training(checkpoint, learner.optimizer, learner.scheduler)
        learner.updates = progress.get("updates", 0)
    learner.checkpoint_metadata = {k: v for k, v in manifest["progress"].items()
                                   if k not in {"policy_version", "updates"}}
    legacy_epochs = learner.checkpoint_metadata.pop("completed_bc_epochs", None)
    if legacy_epochs is not None:
        learner.checkpoint_metadata.setdefault("completed_bootstrap_epochs", legacy_epochs)
    return learner, manifest


def _save(path, learner, extra=None):
    save_checkpoint(path, learner.model, learner.vocabulary, learner.optimizer, learner.scheduler,
                    overwrite=Path(path).name == "current", training=learner.config, progress={**getattr(learner, "checkpoint_metadata", {}),
                                                       **(extra or {}), "reward_version": MilestoneLedger().version,
                                                       "policy_version": learner.policy_version, "updates": learner.updates})


def _data_metadata(runs):
    contracts = {fingerprint(r["contract"]): r["contract"] for r in runs}
    return {"training_engine_contracts": list(contracts.values()),
            "training_seed_pool_hash": fingerprint(sorted((r["character"], str(r["seed"])) for r in runs))}


def main(argv=None):
    args = parser().parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.command == "monitor":
        from .monitor import serve
        serve(args.root, args.host, args.port)
        return 0
    history = None
    try:
        runtime = configure_runtime(args.device, threads=args.threads)
        if args.command == "play-steam":
            from .steam import play
            emit(play(args.checkpoint, args.bridge_dir, args.device, args.timeout, args.max_decisions))
        elif args.command == "parameters":
            config = ModelConfig.from_architecture(args.architecture) if args.architecture else ModelConfig()
            with torch.device("meta"):
                model = PolicyValue(config)
            emit(model.parameter_report())
        elif args.command == "smoke":
            emit(smoke(args.output))
        elif args.command == "init":
            profile = json.loads(Path(args.config).read_text())
            config, training = ModelConfig(**profile["model"]), TrainConfig.from_dict(profile["training"])
            frames = []
            for character in CHARACTERS:
                with CliEngine(root=args.engine_root) as engine:
                    frames.append(validate_frame(engine.reset(character, "vocabulary-" + character)))
                    if len(frames) == 1:
                        catalog = engine.send({"cmd": "public_catalog"})
                        if catalog.get("type") != "public_catalog":
                            raise ProtocolError("Engine does not provide a static public vocabulary catalog")
                        frames.append(catalog)
            vocabulary = Vocabulary(symbols_from_frames(frames), config.vocabulary_size)
            with torch.device(args.device):
                model = PolicyValue(config)
            learner = Learner(model, vocabulary, training)
            _save(args.output, learner, {"runtime": runtime, "engine_contract": frames[0]["contract"],
                                       "initialization": "random-public-catalog"})
            emit({"checkpoint": args.output, "vocabulary_symbols": len(vocabulary.symbols),
                  "parameters": model.parameter_report()})
        elif args.command == "audit":
            emit(audit_files(args.paths, args.accepted, args.quarantine))
        elif args.command == "import-recorder":
            from .recorder import import_recordings
            summary = import_recordings(args.paths, args.output, bc_only=args.bc_only, recorder_version=args.recorder_version)
            emit(summary)
            if not summary["accepted_runs"]:
                return 2
        elif args.command == "inspect":
            with CliEngine(root=args.engine_root) as engine:
                frame = engine.reset(args.character, args.game_seed)
            emit(frame)
        elif args.command == "bootstrap":
            runs = load_runs(args.data)
            if not runs:
                report = Path(args.data).parent / "summary.json"
                detail = f" See import report: {report}" if report.exists() else ""
                raise ProtocolError(f"No accepted behavior-cloning samples: {args.data} contains zero runs."
                                    " Import eligible recordings before starting Bootstrap." + detail)
            if args.validation:
                validation = load_runs(args.validation)
                if {r["seed"] for r in runs} & {r["seed"] for r in validation} or {r["run_id"] for r in runs} & {r["run_id"] for r in validation}:
                    raise ProtocolError("Training/validation run or seed overlap")
            else:
                validation = []
            if not any(r["macros"] for r in runs):
                raise ProtocolError("No accepted behavior-cloning samples in the training split "
                                    f"({len(runs)} training runs, {len(validation)} validation runs). "
                                    "Add training demonstrations; validation runs are not used for updates.")
            if args.checkpoint:
                learner, _ = _learner(args.checkpoint, args.device)
            else:
                profile = json.loads(Path(args.config).read_text()) if args.config else None
                config = ModelConfig(**profile["model"]) if profile else ModelConfig.tiny() if args.tiny else ModelConfig.from_architecture(args.architecture) if args.architecture else ModelConfig()
                vocab = vocabulary_for(runs, config.vocabulary_size)
                with torch.device(args.device):
                    model = PolicyValue(config)
                learner = Learner(model, vocab, TrainConfig.from_dict(profile["training"]) if profile else TrainConfig(precision=args.precision))
            history = TrainingHistory(args.output, "bootstrap", asdict(learner.config))
            start_epoch = getattr(learner, "checkpoint_metadata", {}).get("completed_bootstrap_epochs", 0)
            metadata = {"runtime": runtime, **_data_metadata(runs)}
            epoch_started = time.monotonic()
            def epoch_done(index, metrics):
                nonlocal epoch_started
                epoch = start_epoch + index
                validation_metrics = learner.evaluate_bootstrap(validation) if validation else {}
                record = history.append(epoch, metrics, validation=validation_metrics,
                                        training_runs=len(runs), validation_runs=len(validation),
                                        duration_seconds=time.monotonic() - epoch_started,
                                        policy_version=learner.policy_version)
                _save(Path(args.output) / "current", learner,
                      {**metadata, "completed_bootstrap_epochs": epoch, "last_training_record": record})
                emit(record)
                epoch_started = time.monotonic()
            history.status("updating", round=start_epoch + 1)
            learner.bootstrap(runs, args.epochs, on_epoch=epoch_done)
            history.status("completed", round=start_epoch + args.epochs)
        elif args.command == "infer":
            model, vocabulary, manifest = load_model(args.checkpoint, args.device)
            model.eval()
            raw = Path(args.frame).read_text()
            frames = [json.loads(x) for x in raw.splitlines() if x.strip()] if args.frame.endswith(".jsonl") else [json.loads(raw)]
            policy = SessionPolicy(model, vocabulary)
            with torch.no_grad(), precision_context(model, manifest["training"]["precision"]):
                for frame in frames:
                    validate_frame(frame)
                    choice = policy.choose(frame, sample=False, top_k=args.top_k)
                    emit({"command": execution_command(frame, choice.candidate_ref), "ranking": choice.ranking})
        elif args.command == "ppo":
            learner, _ = _learner(args.checkpoint, args.device)
            runs = load_runs(args.data)
            history = TrainingHistory(args.output, "ppo", asdict(learner.config))
            number = learner.checkpoint_metadata.get("completed_ppo_rounds", 0) + 1
            history.status("updating", round=number)
            metrics = learner.ppo(runs)
            record = history.append(number, metrics, evaluation=evaluate_runs(runs), policy_version=learner.policy_version)
            _save(Path(args.output) / "current", learner, {"runtime": runtime, **_data_metadata(runs),
                  "completed_ppo_rounds": number, "last_training_record": record})
            history.status("completed", round=number)
            emit(record)
        else:
            if args.command == "train":
                if args.rounds < 1:
                    raise ValueError("Training needs at least one round")
                if args.runs_per_character is not None and (args.runs_per_character < 1 or args.seeds):
                    raise ValueError("--runs-per-character requires random seeds and a positive count")
            learner, manifest = _learner(args.checkpoint, args.device)
            seeds = json.loads(Path(args.seeds).read_text()) if args.seeds else None
            schedule = None
            if args.command == "train" and seeds is None:
                state = manifest["progress"].get("random_seed_schedule")
                count = args.runs_per_character if args.runs_per_character is not None else (state or {}).get("runs_per_character", 4)
                schedule = RandomSeedSchedule(count, state=state)
            completed_ppo_rounds = manifest["progress"].get("completed_ppo_rounds", 0)
            directory = Path(args.output)
            directory.mkdir(parents=True, exist_ok=True)
            factory = lambda: CliEngine(root=args.engine_root)
            if args.command == "evaluate":
                traces = []
                runner = RolloutRunner(learner.model, learner.vocabulary, precision=learner.config.precision,
                                       version=learner.policy_version, max_steps=args.max_steps)
                for character in CHARACTERS:
                    for i, seed in enumerate(seeds[character]):
                        with factory() as engine:
                            trace = runner.run(engine, character, seed, sample=False)
                        write_run(directory / f"{character}-{i}.json", trace)
                        traces.append(trace)
                emit(evaluate_runs(traces))
            else:
                rounds = args.rounds if args.command == "train" else 1
                history = TrainingHistory(directory, "ppo", asdict(learner.config)) if args.command == "train" else None
                start_round = manifest["progress"].get("sampling_round", -1) + 1
                # Never overwrite a trajectory from an interrupted or previous invocation.
                existing = [int(p.name[6:]) for p in directory.glob("round-*") if p.name[6:].isdigit()]
                start_round = max(start_round, max(existing, default=-1) + 1)
                for local_index in range(rounds):
                    round_index = start_round + local_index
                    if history:
                        history.status("collecting", round=round_index + 1)
                    round_started = time.monotonic()
                    runner = RolloutRunner(learner.model, learner.vocabulary, precision=learner.config.precision,
                                           version=learner.policy_version, max_steps=args.max_steps)
                    if schedule:
                        assigned = schedule.next()
                    elif rounds > 1:
                        assigned = {c: [f"{s}-round-{round_index}" for s in seeds[c]] for c in CHARACTERS}
                    else:
                        assigned = seeds
                    def game_done(character, completed, total):
                        if history:
                            history.status("collecting", round=round_index + 1, character=character,
                                           games_completed=completed, games_total=total)
                    paths = collect_round(factory, runner, assigned, directory / f"round-{round_index}", on_run=game_done)
                    if args.command == "train":
                        runs = [json.loads(p.read_text()) for p in paths]
                        if history:
                            history.status("updating", round=round_index + 1)
                        if args.value_warmup and local_index == 0:
                            metrics = learner.value_warmup(runs)
                        else:
                            metrics = learner.ppo(runs, load_runs(args.demonstrations) if args.demonstrations else None)
                            completed_ppo_rounds += 1
                        seed_metadata = {"seed_mode": "random" if schedule else "fixed"}
                        if schedule:
                            seed_metadata["random_seed_schedule"] = schedule.state_dict()
                        record = history.append(round_index + 1, metrics, completed_ppo_rounds=completed_ppo_rounds,
                                                stage="value" if args.value_warmup and local_index == 0 else "ppo",
                                                duration_seconds=time.monotonic() - round_started,
                                                evaluation=evaluate_runs(runs), policy_version=learner.policy_version,
                                                seed_file=str(directory / f"round-{round_index}" / "metadata" / "seeds.json"))
                        _save(directory / "current", learner,
                              {"runtime": runtime, **_data_metadata(runs), **seed_metadata,
                               "sampling_round": round_index, "completed_ppo_rounds": completed_ppo_rounds,
                               "last_training_record": record})
                        emit(record)
                    else:
                        emit({"runs": [str(p) for p in paths]})
                if history:
                    history.status("completed", round=round_index + 1)
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
        if history:
            history.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
