"""What each command of `python -m main` does, one function per command, called with
the parsed arguments and the runtime manifest (`main/cli.py` parses and dispatches).
A command returns its exit status, or None for success."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from main.network.checkpoint import archive_checkpoint, checkpoints_for, load_model, save_checkpoint
from main.network.config import LearningRates, ModelConfig, TrainConfig
from main.env.control import environment_action
from main.data.shards import HELDOUT_FRACTION, RunShards, at_ascension, audit_files, combat_from, extent, load_runs, policy_macros, split_held_out
from main.env.engine import CliEngine
from main.play.evaluation import compare_evaluations, evaluate_runs, nll_report
from main.learn.history import TrainingHistory
from main.network.model import PolicyNetwork
from main.network.policy import SessionPolicy
from main.env.protocol import CHARACTERS, ProtocolError, execution_command, fingerprint, validate_frame
from main.encoding.vocabulary import Vocabulary
from main.play.rollout import RolloutRunner, collect_round, precision_context, write_run
from main.play.sampling import Samplers
from main.env.seeds import validate_evaluation_seeds
from main.learn.trainer import Learner


def vocabulary_for(runs, config):
    frames = (s["frame"] for r in runs for m in policy_macros(r) for s in m["steps"])
    return Vocabulary.from_frames(frames, config)


def emit(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def smoke(output):
    from main.play.testing import SyntheticEngine, demonstration

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
        from main.network.policy import replay

        before = replay(model, vocabulary, runs[0]["macros"][0]["steps"])[0]
        after = replay(loaded, vocab, runs[0]["macros"][0]["steps"])[0]
    result["checkpoint_log_prob_error"] = abs(float(before - after))
    (path / "report.json").write_text(json.dumps(result, indent=2))
    return result


def _data_metadata(runs):
    contracts = {fingerprint(r["contract"]): r["contract"] for r in runs}
    return {
        "training_engine_contracts": list(contracts.values()),
        "training_seeds": sorted({str(r["seed"]) for r in runs}),
        "training_seed_pool_hash": fingerprint(
            sorted((r["character"], str(r["seed"])) for r in runs)
        ),
    }


def _dataset(paths):
    """The runs `--data` names: sharded directories are read together as one data set
    (RunShards); one path that is not sharded is a small run file or directory read
    into memory. Returns (runs, whether they are sharded)."""
    if all(RunShards.is_at(path) for path in paths):
        return RunShards(paths), True
    if len(paths) == 1:
        return load_runs(paths[0]), False
    raise ProtocolError("Several --data paths must all be sharded data directories (with an index)")


def run_play_steam(args, runtime):
    from main.steam import play

    emit(play(args.checkpoint, args.bridge_dir, args.device, request_timeout=args.request_timeout,
              run_for=args.run_for, max_decisions=args.max_decisions))


def run_parameters(args, runtime):
    config = (
        ModelConfig(**json.loads(Path(args.config).read_text())["model"])
        if args.config
        else ModelConfig()
    )
    with torch.device("meta"):
        model = PolicyNetwork(config)
    emit(model.parameter_report())


def run_smoke(args, runtime):
    emit(smoke(args.output))


def run_init(args, runtime):
    profile = json.loads(Path(args.config).read_text())
    config, training = (
        ModelConfig(**profile["model"]),
        TrainConfig(**profile["training"]),
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
    learner.save(
        args.output,
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


def run_audit(args, runtime):
    emit(audit_files(args.paths, args.accepted, args.quarantine))


def run_import_independent(args, runtime):
    from main.demos.independent import import_samples

    summary = import_samples(args.samples, args.output, args.shard_samples, args.workers)
    emit(summary)
    if not summary["accepted_runs"]:
        return 2


def run_reindex(args, runtime):
    from main.data.index import rebuild_index

    emit(rebuild_index(args.data, workers=args.workers, listed_only=args.listed_only))


def run_inspect(args, runtime):
    with CliEngine(root=args.engine_root) as engine:
        frame = engine.reset(args.character, args.game_seed, args.ascension)
    emit(frame)


def run_bootstrap(args, runtime):
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
    checkpoints = checkpoints_for(output, args.checkpoints)
    if (output / "current").exists() or any(output.glob("epoch-*")):
        raise ValueError(f"{output} holds checkpoints; they are kept in {checkpoints}: move them there")
    if ((checkpoints / "current").exists() or (output / "history.jsonl").exists()
            or (output / "updates.jsonl").exists()
            or any(checkpoints.glob("epoch-*"))):
        if (args.weights_only or not args.checkpoint
                or Path(args.checkpoint).resolve() != (checkpoints / "current").resolve()):
            raise ValueError("Existing BC output requires resume from its current checkpoint; use a new output directory")
    runs, streamed = _dataset(args.data)
    if args.ascension is not None:
        runs = at_ascension(runs, args.ascension)
    if args.combat_from:
        runs = combat_from(runs, args.combat_from)
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
    rates = LearningRates.load(args.learning_rates) if args.learning_rates else None
    if args.checkpoint:
        learner, _ = Learner.load(args.checkpoint, args.device,
                              weights_only=args.weights_only,
                              training=TrainConfig(**profile["training"]) if profile else None,
                              decay_updates=args.decay_updates, rates=rates)
        if profile and ModelConfig(**profile["model"]) != learner.model.config:
            raise ValueError("A checkpoint's model architecture cannot be changed by --config")
    else:
        config = (
            ModelConfig(**profile["model"])
            if profile
            else ModelConfig.tiny()
            if args.tiny
            else ModelConfig()
        )
        vocab = vocabulary_for(runs, config)
        with torch.device(args.device):
            model = PolicyNetwork(config)
        training = TrainConfig(**profile["training"]) if profile else TrainConfig(precision=args.precision)
        learner = Learner(model, vocab, training.with_rates(rates) if rates else training)
    history = TrainingHistory(args.output, "bootstrap", asdict(learner.config))
    try:
        start_epoch = getattr(learner, "checkpoint_metadata", {}).get(
            "completed_bootstrap_epochs", 0
        )
        metadata = {"runtime": runtime, **(runs.metadata() if streamed else _data_metadata(runs)),
                    "bootstrap_ascension": args.ascension,
                    "bootstrap_combat_from": args.combat_from and [str(path) for path in args.combat_from]}
        epoch_started = time.monotonic()
        completed = 0

        def update_done(metrics, *, epoch, stage):
            history.append_update(metrics, round=start_epoch + epoch, stage=stage,
                                  optimization_epoch=epoch, policy_version=learner.policy_version)

        def epoch_done(index, metrics):
            nonlocal epoch_started, completed
            epoch = start_epoch + index
            archived = checkpoints / f"epoch-{epoch:06d}"
            if archived.exists():
                raise FileExistsError("Epoch checkpoint already exists; use a new output directory")
            record = history.append(
                epoch,
                metrics,
                training_runs=metrics["training"]["groups"],
                duration_seconds=time.monotonic() - epoch_started,
                policy_version=learner.policy_version,
            )
            learner.save(
                checkpoints / "current",
                {
                    **metadata,
                    "completed_bootstrap_epochs": epoch,
                    "held_out": metrics["held_out"],
                    "last_training_record": record,
                },
            )
            # Retain completed epochs for manual comparison. Hard links survive
            # the next atomic replacement of current without copying tensors.
            archive_checkpoint(checkpoints / "current", archived)
            emit(record)
            completed = index
            epoch_started = time.monotonic()

        history.status("updating", round=start_epoch + 1, updates=learner.updates)
        learner.bootstrap(runs, args.epochs, on_epoch=epoch_done, window_shards=args.window_shards,
                          on_update=update_done, include_held_out=args.include_heldout,
                          max_updates=args.max_updates)
        history.status("completed", round=start_epoch + completed)
    except KeyboardInterrupt:
        history.status("interrupted")
        raise
    except (ProtocolError, ValueError, OSError, RuntimeError) as exc:
        history.status("failed", error=str(exc))
        raise
    finally:
        history.close()


def run_heldout_nll(args, runtime):
    runs, _ = _dataset(args.data)
    if args.ascension is not None:
        runs = at_ascension(runs, args.ascension)
    if args.combat_from:
        runs = combat_from(runs, args.combat_from)
    learner, _ = Learner.load(args.checkpoint, args.device, inference=True)
    # A data set kept out of training whole is scored on every seed group.
    held = runs if args.all_groups else split_held_out(runs)[1]
    if not extent(held)["macros"]:
        raise ProtocolError("The data set has no macro-actions in its held-out seed groups")
    groups = learner.evaluate_bootstrap(held, window_shards=args.window_shards, include_a0=True)
    emit({
        "data": [str(path) for path in args.data],
        "ascension": args.ascension,
        "combat_from": args.combat_from,
        "checkpoint": str(args.checkpoint),
        "policy_version": learner.policy_version,
        "updates": learner.updates,
        "held_out": {"fraction": 1.0 if args.all_groups else HELDOUT_FRACTION, **extent(held), "of": extent(runs)},
        **nll_report(groups),
    })


def run_compare(args, runtime):
    emit(compare_evaluations(args.directories, head=args.head))


def explore_outside(values):
    """--explore-outside's (or --explore-prior's) values as `Settings.explore_outside`
    takes them: one rate, or {phase: rate}."""
    if len(values) == 1 and not isinstance(values[0], tuple):
        return values[0]
    if not all(isinstance(value, tuple) for value in values):
        raise ValueError("--explore-outside and --explore-prior take one rate or PHASE=RATE pairs, not both")
    return dict(values)


def run_iterate(args, runtime):
    from main.iterate import Driver, Settings

    settings = Settings(
        checkpoint=args.checkpoint, runs_per_character=args.runs_per_character, seed_root=args.seed_root,
        exclude_seeds=args.exclude_seeds, workers=args.workers, processes=args.processes,
        max_steps=args.max_steps, engine_root=args.engine_root, config=args.config,
        wave_shares=args.wave_shares, win_hp_bonus=args.win_hp_bonus,
        boundaries=args.boundary_acts, group_n=args.group_n, group_m=args.group_m, ascensions=args.ascensions,
        min_entropy=args.min_entropy, clip=args.clip, kl=args.kl, learning_rates=args.learning_rates,
        fight_kl=args.fight_kl, passes=args.passes, freeze_outside=args.freeze_outside,
        explore_outside=explore_outside(args.explore_outside),
        explore_prior=explore_outside(args.explore_prior), prior=args.prior, explored_on_tree=args.explored_on_tree,
        explored_centered=args.explored_centered,
        fight_groups=args.fight_groups, fight_group_size=args.fight_group_size,
        fight_group_priority=args.fight_group_priority, fight_hindsight=args.fight_hindsight,
        fight_group_shares=dict(args.fight_group_shares) or None,
        fight_group_kind_size={kind: int(n) for kind, n in args.fight_group_kind_size} or None,
        fight_pool=list(args.fight_pool), fight_pool_share=args.fight_pool_share,
        fight_group_natural=args.fight_group_natural,
        fight_hindsight_shares=dict(args.fight_hindsight_shares) or None,
        potion_prices=args.potion_prices, growth_bonus=args.growth_bonus,
        source_shares=dict(args.source_shares) if args.source_shares else None,
        round_demos=args.round_demos, demo_buffer=args.demo_buffer, demo_opening=args.demo_opening,
        demo_workers=args.demo_workers,
        eval_precision=args.eval_precision, batch=args.batch, window_shards=args.window_shards,
        evaluate_every=args.evaluate_every,
        eval_seeds=args.eval_seeds, eval_runs_per_character=args.eval_runs_per_character,
        eval_workers=args.eval_workers)
    state = Driver(args.output, settings, args.device, runtime=runtime,
                   checkpoints=checkpoints_for(args.output, args.checkpoints)).run(args.iterations)
    emit({"output": str(args.output), "iterations": sum(
        bool(entry.get("complete")) for entry in state["iterations"].values())})


def run_infer(args, runtime):
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


def run_evaluate(args, runtime):
    """The policy plays greedily, in the engines of sampling processes."""
    if args.workers < 1 or (args.processes is not None and args.processes < 1):
        raise ValueError("Sampling needs positive numbers of engines and processes")
    learner, manifest = Learner.load(args.checkpoint, args.device, inference=True)
    seeds = json.loads(Path(args.seeds).read_text())
    validate_evaluation_seeds(seeds, manifest["progress"])
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    runner = RolloutRunner(
        learner.model,
        learner.vocabulary,
        precision=args.precision or learner.config.precision,
        version=learner.policy_version,
        max_steps=args.max_steps,
        ascension=learner.config.ascension if args.ascension is None else args.ascension,
        claim_rewards=args.claim_rewards,
        temperatures=learner.config.temperatures,
        iteration=None,
    )
    with Samplers(lambda: CliEngine(root=args.engine_root), learner.model,
                  learner.vocabulary, engines=args.workers, processes=args.processes) as samplers:
        paths = collect_round(samplers, runner, seeds, directory, sample=False)
    # One run in memory at a time: together they are tens of gigabytes.
    emit(evaluate_runs(json.loads(p.read_text()) for p in paths))


def run_fight_set(args, runtime):
    from main.play.fight_eval import build_set

    emit(build_set(args.sources, args.output, count=args.count, root=args.root, exclude=args.exclude))


def _fight_solver(solver, fights_dir):
    """The solver's fights to set against: `--solver`, or the set's own (`fight_eval.SOLVER`) if it has them."""
    from main.play.fight_eval import SOLVER

    if solver:
        return Path(solver)
    default = Path(fights_dir) / SOLVER
    return default if (default / "selection.json").is_file() else None


def run_fight_eval(args, runtime):
    """The policy plays every fight of a fight evaluation set greedily, from its entry
    save to its end, in the engines of sampling processes; each fight's row and the
    report go to the output."""
    from main.learn.costs import FIXED, with_growth
    from main.learn.fight_report import priced, report, text
    from main.play.fight_eval import load_set, play_set, solver_rows

    if args.workers < 1 or (args.processes is not None and args.processes < 1):
        raise ValueError("Sampling needs positive numbers of engines and processes")
    directory = Path(args.output)
    if (directory / "fights.jsonl").exists():
        raise FileExistsError(f"{directory} holds a fight evaluation already; choose a new output")
    learner, manifest = Learner.load(args.checkpoint, args.device, inference=True)
    fights, seeds = load_set(args.fights)
    validate_evaluation_seeds(seeds, manifest["progress"])
    if args.per_kind is not None:
        # A trial on part of the set: the first fights of each kind.
        fights = [fight for kind in ("normal", "elite", "boss")
                  for fight in [fight for fight in fights if fight["kind"] == kind][:args.per_kind]]
    precision = args.precision or learner.config.precision
    runner = RolloutRunner(learner.model, learner.vocabulary, precision=precision, version=learner.policy_version,
                           max_steps=args.max_steps, ascension=learner.config.ascension,
                           temperatures=learner.config.temperatures, iteration=None)
    temperature = learner.config.temperatures["combat"]
    began = time.monotonic()
    with Samplers(lambda: CliEngine(root=args.engine_root), learner.model,
                  learner.vocabulary, engines=args.workers, processes=args.processes) as samplers:
        rows = play_set(samplers, runner, fights, temperature=temperature)
        timing = samplers.timing
    seconds = time.monotonic() - began
    directory.mkdir(parents=True, exist_ok=True)
    # The fixed prices the rounds compare at, with the growth reward when one is given.
    prices = with_growth(FIXED, args.growth_bonus)
    with open(directory / "fights.jsonl", "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(priced(row, prices) if row["status"] == "complete" and row["record"] else row,
                                    ensure_ascii=False, allow_nan=False) + "\n")
    metadata = dict(checkpoint=str(args.checkpoint), policy_version=learner.policy_version,
                    policy_digest=runner.digest, precision=precision, fights=str(args.fights), played=len(rows),
                    workers=args.workers, processes=args.processes, danger_temperature=temperature,
                    growth_bonus=args.growth_bonus,
                    seconds=round(seconds, 1), collector=timing)
    (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    solver = _fight_solver(args.solver, args.fights)
    found = dict(report(rows, solver_rows(solver, fights) if solver else None, prices),
                 checkpoint=str(args.checkpoint), precision=precision, seconds=round(seconds, 1),
                 solver_dir=None if solver is None else str(solver))
    (directory / "report.json").write_text(json.dumps(found, ensure_ascii=False, indent=2) + "\n")
    (directory / "report.txt").write_text(text(found), encoding="utf-8")
    if args.prices:
        # The same fights at a fit's prices, beside the fixed ones the rounds compare at.
        again = _priced_report(rows, args.prices, growth=args.growth_bonus)
        (directory / "report-prices.json").write_text(json.dumps(again, ensure_ascii=False, indent=2) + "\n")
        (directory / "report-prices.txt").write_text(text(again), encoding="utf-8")
    emit(found)


def _priced_report(rows, spec, second=None, growth=0.0):
    """A fight evaluation's report at the prices `spec` names (`potion_prices.load`), the
    policy's fights alone: a fitted price needs the state a fight was entered in, which
    the solver's fights do not record. With `second`, another play of the set beside it;
    with `growth`, the growth reward (`costs.with_growth`)."""
    from main.learn import costs, potion_prices
    from main.learn.fight_report import paired_plays, report

    prices = costs.with_growth(potion_prices.load(spec), growth)
    found = dict(report(rows, None, prices), prices=spec, prices_digest=potion_prices.digest(prices))
    if second is not None:
        found["second"] = report(second, None, prices)["summary"]
        found["paired"] = paired_plays(rows, second, prices)
    return found


def run_fight_compare(args, runtime):
    """The report of one fight evaluation again, from its rows (at the fixed prices,
    against the solver's fights if there are any); of two, also fight by fight."""
    from main.learn.costs import FIXED, with_growth
    from main.learn.fight_report import paired_plays, report, text
    from main.play.fight_eval import load_set, read_rows, solver_rows

    if not 1 <= len(args.results) <= 2:
        raise ValueError("Compare one fight evaluation, or two of the same set")
    played = [read_rows(Path(directory) / "fights.jsonl") for directory in args.results]
    if args.prices:
        found = _priced_report(played[0], args.prices, played[1] if len(played) == 2 else None,
                               growth=args.growth_bonus)
        if args.text:
            print(text(found), end="")
        else:
            emit(found)
        return
    fights_dir = args.fights or json.loads((Path(args.results[0]) / "metadata.json").read_text())["fights"]
    solver = _fight_solver(args.solver, fights_dir)
    prices = with_growth(FIXED, args.growth_bonus)
    found = report(played[0], solver_rows(solver, load_set(fights_dir)[0]) if solver else None, prices)
    found["solver_dir"] = None if solver is None else str(solver)
    if len(played) == 2:
        found["second"] = report(played[1], None, prices)["summary"]
        found["paired"] = paired_plays(played[0], played[1], prices)
    if args.text:
        print(text(found), end="")
    else:
        emit(found)


def _guard_memory(floor_gb):
    """End this process group when the machine's available memory falls below
    `floor_gb` (a fit plays as many engines as a round): run it with `setsid`."""
    import os
    import signal
    import threading

    def watch():
        while True:
            with open("/proc/meminfo") as stream:
                info = dict(line.split(":", 1) for line in stream)
            available = int(info["MemAvailable"].split()[0]) / 2 ** 20
            if available < floor_gb:
                print(json.dumps(dict(watchdog="killed", available_gb=round(available, 2))), flush=True)
                os.killpg(os.getpgrp(), signal.SIGKILL)
            time.sleep(3)

    threading.Thread(target=watch, daemon=True).start()


def run_fit_prices(args, runtime):
    """Fit the potions' prices (`main/learn/potion_prices.py`) on pairs of continuations
    of the checkpoint's own runs (`main/play/potion_value.py`): source runs, the points
    drawn from them, their continuations played, the prices fitted and moved from the
    previous ones (`Smoothed`). Each step is kept in the output and taken from there
    when asked again."""
    from main.env.seeds import excluded_seeds, round_ascensions, round_seeds
    from main.learn import potion_prices
    from main.play import potion_value

    if args.workers < 1 or (args.processes is not None and args.processes < 1):
        raise ValueError("Sampling needs positive numbers of engines and processes")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.memory_floor:
        _guard_memory(args.memory_floor)
    learner, manifest = Learner.load(args.checkpoint, args.device, inference=True)
    previous = potion_prices.load(args.previous)
    excluded = excluded_seeds(args.exclude_seeds, manifest["progress"].get("training_seeds", []))
    seeds = round_seeds(args.seed_root, 0, args.source_runs_per_character, excluded)
    levels = round_ascensions(args.seed_root, 0, seeds, tuple(args.ascensions))
    precision = args.precision or learner.config.precision
    # The policy's own runs: fights greedy as in the rounds, the rest at its temperature.
    runner = RolloutRunner(learner.model, learner.vocabulary, precision=precision, version=learner.policy_version,
                           max_steps=args.max_steps, ascension=learner.config.ascension,
                           temperatures=learner.config.temperatures, greedy_fights=True, iteration=None)
    started = {(character, str(seed)) for character in seeds for seed in seeds[character]}
    points_dir = output / potion_value.POINTS
    began = time.monotonic()
    with Samplers(lambda: potion_value.PointEngine(CliEngine(root=args.engine_root), points_dir, started),
                  learner.model, learner.vocabulary, engines=args.workers, processes=args.processes) as samplers:
        source = potion_value.play_sources(samplers, runner, seeds, levels, output)
        drawn_path = output / "points.json"
        if drawn_path.is_file():
            drawn = json.loads(drawn_path.read_text())
        else:
            found = [potion_value.read_point(path) for path in sorted(points_dir.glob("*.json.gz"))]
            points, strata = potion_value.draw(found, args.pairs // 3)
            drawn = dict(points=points, strata=strata)
            drawn_path.write_text(json.dumps(drawn, ensure_ascii=False) + "\n")
        branches = potion_value.branch_saves(drawn["points"], output / potion_value.BRANCHES)
        results = potion_value.play_branches(samplers, runner, drawn["points"], branches, output, repeat=args.repeat)
    played = time.monotonic() - began
    fitted, report = potion_prices.fit(drawn["points"], results, previous, info=dict(
        checkpoint=str(args.checkpoint), policy_version=learner.policy_version, seed_root=args.seed_root))
    prices = potion_prices.Smoothed(previous, fitted)
    potion_prices.save(fitted, output / "fitted.json")
    potion_prices.save(prices, output / "prices.json")
    report = dict(report, checkpoint=str(args.checkpoint), policy_version=learner.policy_version,
                  previous=args.previous, previous_digest=potion_prices.digest(previous),
                  prices_digest=potion_prices.digest(prices), source=source, strata=drawn["strata"],
                  branches=len(results), failed=sum(row["status"] != "complete" for row in results),
                  determinism=potion_prices.determinism(results), seconds=round(time.monotonic() - began, 1),
                  playing_seconds=round(played, 1))
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    emit({key: report[key] for key in ("pairs", "points", "kept", "held_out_error", "base_held_out_error",
                                       "determinism", "failed", "seconds", "prices_digest")})


COMMANDS = {
    "play-steam": run_play_steam,
    "parameters": run_parameters,
    "smoke": run_smoke,
    "init": run_init,
    "audit": run_audit,
    "import-independent": run_import_independent,
    "reindex": run_reindex,
    "inspect": run_inspect,
    "bootstrap": run_bootstrap,
    "heldout-nll": run_heldout_nll,
    "compare": run_compare,
    "iterate": run_iterate,
    "infer": run_infer,
    "evaluate": run_evaluate,
    "fight-set": run_fight_set,
    "fight-eval": run_fight_eval,
    "fight-compare": run_fight_compare,
    "fit-prices": run_fit_prices,
}
