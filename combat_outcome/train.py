"""Train the entity combat-outcome model: entry state x encounter -> how the fight ends.

Each fight becomes a synthetic public frame (frames.py): the player, deck, relics,
potions and powers the fight was entered with, and the encounter. The project's
entity encoder and global Full-Attention blocks encode it; a head on the action
token predicts the HP lost as a signed fraction of max HP and whether the fight is
lost (model.py). Architecture, training and loss settings come from a profile
(config.py, default configs/combat-outcome.json); the flags below override single values.

Pretraining uses the solver fights of the summary-anchor battles, the same battles
whose actions the policy's Bootstrap clones; it runs beside that Bootstrap and
needs nothing from it. `--pretrained` continues from a saved model on new labels,
such as fights recorded during rollouts.

Baselines on the same seed-grouped split: encounter only, ridge on deck summary,
ridge on summary + id bag. Evaluation: held-out R^2 and error in HP, by kind of fight.

python -m spire_codex_data.anchor export --manifest data/spire-codex/sources/manifest.json --output data/anchors      # writes combat-outcomes.jsonl.gz
python -m combat_outcome.train data/anchors/combat-outcomes.jsonl.gz --save runs/combat-outcome
"""

import argparse
import json
import math
import random
import statistics as st
import time
from dataclasses import asdict
from itertools import chain
from pathlib import Path

import torch
from torch.utils.data import Dataset

from model.loader import _stream
from model.representation import Vocabulary, observation

from .baselines import fit_baselines
from .config import DEFAULT_CONFIG, CombatOutcomeConfig, override
from .data import SCHEMA, entities, load_data, split_runs
from .frames import encounter_id, frame_for
from .metrics import report as fit_report
from .model import CombatOutcomeNet, check_encounters, load_model, save_model

OVERRIDES = ("epochs", "batch_size", "lr", "holdout", "seed", "summary_entity", "ridge_base", "loader_workers")


class Batches(Dataset):
    """Observations of each batch, built where the loader runs."""

    def __init__(self, batches, summary):
        self.batches, self.summary = batches, summary

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        return [observation(frame_for(entities(r), r["encounter"], self.summary)) for r in self.batches[index]]


def batch_loss(frac, logit, batch, training):
    y = torch.tensor([r["y"] / r["max_hp"] for r in batch], device=frac.device)
    lost = torch.tensor([float(r["lost"]) for r in batch], device=frac.device)
    return torch.nn.functional.mse_loss(frac, y) \
        + training.defeat_weight * torch.nn.functional.binary_cross_entropy_with_logits(logit, lost)


def public_catalog(path=None):
    """Static content, independent of the fights selected for training or evaluation."""
    if path is not None:
        catalog = json.loads(Path(path).read_text())
    else:
        from model.engine import CliEngine

        with CliEngine() as engine:
            catalog = engine.send({"cmd": "public_catalog"})
    if catalog.get("type") != "public_catalog" or not any(
            e.get("entity_type") == "encounter" for e in catalog.get("public", {}).get("entities", [])):
        raise ValueError("Expected a static public_catalog with encounter entities")
    return catalog


def fit(train, test, config, device, pretrained=None, catalog=None):
    """Train on `train`, report on `test`. Returns (net, vocabulary, report)."""
    training = config.training
    started = time.monotonic()
    encounters = {encounter_id(r["encounter"]) for r in chain(train, test)}
    if pretrained:
        if catalog is not None:
            raise ValueError("A pretrained model keeps its frozen vocabulary; --catalog is for initialization")
        net, vocabulary, manifest = load_model(pretrained, device)
        if manifest["model"] != asdict(config.model) or \
                manifest["training"]["summary_entity"] != training.summary_entity:
            raise ValueError("The pretrained model was built with a different profile")
        net.train()
    else:
        catalog = public_catalog(catalog)
        check_encounters(encounters, {"content_id=" + e["content_id"] for e in catalog["public"]["entities"]
                                      if e.get("entity_type") == "encounter"})
        frames = (frame_for(entities(r), r["encounter"], training.summary_entity) for r in train)
        vocabulary = Vocabulary.from_frames(chain([catalog], frames), config.model)
        net = CombatOutcomeNet(config.model).to(device)
    check_encounters(encounters, vocabulary.lookup)
    opt = torch.optim.AdamW(net.parameters(), lr=training.lr, weight_decay=training.weight_decay)
    steps = training.epochs * (math.ceil(len(train) / training.batch_size) + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=training.lr, total_steps=steps,
                                                pct_start=training.warmup_fraction)

    def batches(rows, size):
        parts = [rows[i:i + size] for i in range(0, len(rows), size)]
        return zip(parts, _stream(Batches(parts, training.summary_entity), training.loader_workers))

    def forward(batch, inputs):
        base = torch.tensor([r["base"] for r in batch], device=device) if training.ridge_base else None
        return net(inputs, vocabulary, base)

    def predict(rows):
        net.eval()
        hp, defeat = [], []
        with torch.no_grad():
            for batch, inputs in batches(rows, 256):
                frac, logit = forward(batch, inputs)
                hp += [float(f) * r["max_hp"] for f, r in zip(frac, batch)]
                defeat += torch.sigmoid(logit).tolist()
        net.train()
        return hp, defeat

    def evaluate():
        hp, defeat = predict(test)
        return dict(fit_report(test, hp), defeat=dict(
            rate=round(st.mean(r["lost"] for r in test), 4), predicted=round(st.mean(defeat), 4)))

    history = []
    for epoch in range(training.epochs):
        total = 0.0
        random.Random(f"{training.seed}:{epoch}").shuffle(train)
        for batch, inputs in batches(train, training.batch_size):
            loss = batch_loss(*forward(batch, inputs), batch, training)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), training.max_grad_norm)
            opt.step()
            sched.step()
            total += loss.item() * len(batch)
        if epoch % 5 == 4 or epoch == training.epochs - 1:
            history.append(dict(epoch=epoch + 1, train_loss=round(total / len(train), 5),
                                test_r2=evaluate()["all"]["r2"], seconds=round(time.monotonic() - started)))
            print(json.dumps(history[-1]), flush=True)
    return net, vocabulary, dict(test=evaluate(), history=history,
                                 parameters=sum(p.numel() for p in net.parameters()),
                                 vocabulary=len(vocabulary.symbols), seconds=round(time.monotonic() - started))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, help="A combat-outcome label file, or a directory of them")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="combat-outcome profile (model, training)")
    parser.add_argument("--save", type=Path, help="Directory for the trained model (weights, manifest)")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--pretrained", type=Path, help="Saved model to continue from; its vocabulary is kept")
    initialization.add_argument("--catalog", type=Path,
                                help="Exported public_catalog for initialization (default: query the local engine)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int, dest="batch_size")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--holdout", type=float)
    parser.add_argument("--loader-workers", type=int, help="processes that build the batches")
    parser.add_argument("--limit", type=int, help="use at most this many training rows (learning curve)")
    parser.add_argument("--train-runs", type=int, help="use only this many training seeds (learning curve by deck diversity)")
    parser.add_argument("--folds", type=int, help="seed-grouped cross-validation: number of folds (overrides --holdout)")
    parser.add_argument("--fold", type=int, default=0, help="which fold is the test set")
    parser.add_argument("--baselines-only", action="store_true", help="skip the entity model")
    parser.add_argument("--no-baselines", action="store_true", help="skip the ridge baselines")
    parser.add_argument("--ridge-base", action=argparse.BooleanOptionalAction,
                        help="the network predicts the residual of the deck-summary ridge baseline")
    parser.add_argument("--summary-entity", action=argparse.BooleanOptionalAction,
                        help="add a deck-aggregate entity to the input")
    parser.add_argument("--report", type=Path, help="report path (default: model_report*.json beside the data)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = CombatOutcomeConfig.load(args.config)
    config.training = training = override(config.training, args, OVERRIDES)
    if training.ridge_base and (args.pretrained or args.no_baselines):
        parser.error("--ridge-base needs the baselines and cannot continue from a saved model")
    torch.manual_seed(training.seed)
    rows = load_data(args.data)
    train, test, runs, test_runs = split_runs(rows, training.seed, training.holdout, args.folds, args.fold)
    if args.train_runs:
        kept = sorted({r["run"] for r in train})
        random.Random(f"{training.seed}:train-runs").shuffle(kept)
        kept = set(kept[:args.train_runs])
        train = [r for r in train if r["run"] in kept]
    if args.limit:
        random.Random(training.seed).shuffle(train)
        train = train[:args.limit]
    report = dict(schema=SCHEMA, rows=len(rows), train=len(train), test=len(test), runs=len(runs),
                  test_runs=len(test_runs), train_runs=len({r["run"] for r in train}),
                  fold=args.fold if args.folds else None, folds=args.folds,
                  sources=sorted({str(r["source"]) for r in rows}),
                  # Whose fights these are: the solver's, or a policy's by its weights.
                  policies=sorted({str(r["policy"]) for r in rows}),
                  pretrained=str(args.pretrained) if args.pretrained else None,
                  loss_rate=round(st.mean(r["lost"] for r in rows), 3))
    if not args.no_baselines:
        report["baselines"] = fit_baselines(train, test, training.ridge_base)
    if args.baselines_only:
        return finish(args, report)
    net, vocabulary, report["entity_model"] = fit(train, test, config, args.device, args.pretrained, args.catalog)
    finish(args, report)
    if args.save:
        save_model(args.save, net, vocabulary, config, schema=SCHEMA, data=str(args.data), report=report)


def finish(args, report):
    text = json.dumps(report, indent=2)
    print(text)
    name = f"model_report{'-limit' + str(args.limit) if args.limit else ''}.json"
    path = args.report or (args.data if args.data.is_dir() else args.data.parent) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


if __name__ == "__main__":
    main()
