"""Train the critic: out-of-combat decision frames -> the run is won, the act is passed.

The rows are the out-of-combat decisions of complete runs (data.py), split by seed.
The network (model.py) is the policy's encoder and global layers with a two-logit
head; its architecture is the `model` section of the policy profile, the frozen
vocabulary comes from the policy checkpoint, and so do the initial weights of the
shared body unless `--init random`. Batches are packed as the policy's trainer packs
them (batches.py) and the optimizer is grouped as the policy's: Muon for the
Transformer matrices, AdamW for the lookup tables, the head and the rest.

Reported on the held-out seeds, for both targets: explained variance of the ±1 return,
Brier score, log loss, AUC, and calibration by act, by character and by predicted
probability; beside them the constant base rate and two logistic regressions on the
same split (baselines.py). The report goes to the console and into the manifest.

python -m critic.train runs/eval-a0/dev-v8 --checkpoint runs/bootstrap/epoch-000017 --save runs/critic/v1
python -m critic.train runs/buffer --pretrained runs/critic/v1 --save runs/critic/v2      # the next iteration
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from model.checkpoint import load_model as load_policy
from model.config import ModelConfig
from model.loader import _stream
from model.optim import HybridOptimizer

from .baselines import fit_baselines
from .batches import labels, pack, parse
from .config import DEFAULT_CONFIG, CriticConfig, override
from .data import coverage, frame_of, load as load_rows, split
from .metrics import calibration, reliability, summary
from .model import Critic, autocast, load as load_critic, save

OVERRIDES = ("epochs", "init", "train_body", "train_blocks", "muon_lr", "backbone_lr", "embedding_lr", "head_lr",
             "logical_batch_size", "token_budget", "holdout", "seed", "loader_workers", "precision")
# Settings of a ModelConfig that do not change the weights: a critic may differ from
# the policy it starts from in these.
RUNTIME = {"backend", "encoder_precision", "checkpoint_layers", "compile_blocks"}


def architecture(model):
    """The weight-determining fields of a `model` section, with the defaults an older
    manifest leaves out."""
    return {key: value for key, value in asdict(ModelConfig(**model)).items() if key not in RUNTIME}


class Logicals(Dataset):
    """The packed parts of each logical batch of rows, built where the loader runs:
    (positions in `rows`, Batch, [rows, 2] targets) each."""

    def __init__(self, rows, vocabulary, config):
        self.rows, self.vocabulary, self.config = rows, vocabulary, config

    def __len__(self):
        return -(-len(self.rows) // self.config.logical_batch_size)

    def __getitem__(self, index):
        start = index * self.config.logical_batch_size
        rows = self.rows[start:start + self.config.logical_batch_size]
        parts = pack(parse(frame_of(row) for row in rows), self.vocabulary, self.config)
        return [([start + p for p in positions], batch, labels(rows, positions)) for positions, batch in parts]


def batches(rows, vocabulary, config):
    """The logical batches of `rows` in order, each a list of parts."""
    for parts in _stream(Logicals(rows, vocabulary, config), config.loader_workers):
        for _, batch, _ in parts:
            vocabulary.report(batch.unregistered)
        yield parts


def tables_of(net):
    return {id(module.weight) for module in net.body.modules() if isinstance(module, nn.Embedding)}


def trainable(net, training):
    """(name, parameter) of what the settings let train: the head, and of the body all of
    it, its top `train_blocks` global layers with the final norm, or nothing; the lookup
    tables stay frozen at `embedding_lr` zero."""
    layers, tables = len(net.body.blocks), tables_of(net)

    def wanted(name, parameter):
        if name.startswith("head."):
            return True
        if not training.train_body or (id(parameter) in tables and not training.embedding_lr):
            return False
        if training.train_blocks is None:
            return True
        return name.startswith("body.final_norm.") or (
            name.startswith("body.blocks.") and int(name.split(".")[2]) >= layers - training.train_blocks)

    return [(name, parameter) for name, parameter in net.trainable() if wanted(name, parameter)]


def build_optimizer(net, training):
    """Muon for the body's Transformer matrices; AdamW with the policy's groups for the
    rest: the lookup tables, the head, and the backbone's remaining tensors. Settings
    that train no matrices get AdamW alone."""
    muon_ids = {
        id(module.weight) for name, module in net.body.named_modules()
        if isinstance(module, nn.Linear) and (".attention." in name or ".ffn." in name)
        and name.startswith(("blocks.", "encoder.local_blocks.", "encoder.field_block."))
    }
    tables = tables_of(net)
    matrices, groups = [], {}
    for name, parameter in trainable(net, training):
        if id(parameter) in muon_ids:
            matrices.append(parameter)
            continue
        family = "embedding" if id(parameter) in tables else "head" if name.startswith("head.") else "backbone"
        decay = family != "embedding" and parameter.ndim >= 2 and "norm" not in name
        groups.setdefault((family, decay), []).append(parameter)
    rates = dict(embedding=training.embedding_lr, head=training.head_lr, backbone=training.backbone_lr)
    adamw = torch.optim.AdamW(
        [dict(params=parameters, name=family + ("_decay" if decay else "_no_decay"), lr=rates[family],
              weight_decay=training.weight_decay if decay else 0.0)
         for (family, decay), parameters in groups.items()],
        betas=(0.9, 0.999), eps=1e-8, fused=net.device.type == "cuda")
    if matrices:
        muon = torch.optim.Muon(
            [dict(params=matrices, name="muon", lr=training.muon_lr, weight_decay=training.weight_decay)],
            momentum=training.muon_momentum, ns_steps=training.muon_ns_steps)
        optimizer = HybridOptimizer(adamw, muon)
    else:
        optimizer = adamw
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    return optimizer


def rate_factor(update, total, warmup):
    """Linear warmup over `warmup` updates, then a linear decay to zero at the last."""
    warm = min(1.0, (update + 1) / warmup) if warmup else 1.0
    return warm * max(0.0, 1.0 - update / total)


def losses(logits, targets, act_weight):
    """Per-row loss: the run-won cross-entropy plus `act_weight` times the act-passed one."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return loss[:, 0] + act_weight * loss[:, 1]


def base_rates(rows):
    return [sum(r.win for r in rows) / len(rows), sum(r.act_passed for r in rows) / len(rows)]


def constant_loss(rows, act_weight):
    """The training loss of predicting the rows' own base rates: what a head that has
    learned nothing scores on them, and the mark a fitted loss is measured from."""
    def entropy(rate):
        return 0.0 if rate in (0.0, 1.0) else -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))

    win, act = base_rates(rows)
    return entropy(win) + act_weight * entropy(act)


def set_head_bias(net, rates):
    """The head predicts the base rates before it is trained: the constant baseline."""
    with torch.no_grad():
        net.head.bias.copy_(torch.tensor([math.log(max(r, 1e-6) / max(1 - r, 1e-6)) for r in rates]))


@torch.no_grad()
def predict_rows(net, vocabulary, rows, config, device):
    """[rows, 2] probabilities of `rows`, in their order."""
    net.eval()
    out = torch.zeros(len(rows), 2)
    for parts in batches(rows, vocabulary, config):
        for positions, batch, _ in parts:
            with autocast(device, config.precision):
                logits = net(batch.to(device))
            out[positions] = torch.sigmoid(logits.float()).cpu()
    return out


def evaluate(rows, probabilities):
    """Held-out summaries of both targets and their calibration tables."""
    p_win, p_act = probabilities[:, 0].tolist(), probabilities[:, 1].tolist()
    win, act = [r.win for r in rows], [r.act_passed for r in rows]

    def tables(keys):
        return dict(win=calibration(keys, p_win, win), act=calibration(keys, p_act, act))

    return dict(win=summary(p_win, win), act=summary(p_act, act), calibration=dict(
        by_act=tables([r.act for r in rows]), by_character=tables([r.character for r in rows]),
        by_probability=dict(win=reliability(p_win, win), act=reliability(p_act, act))))


def fit(net, vocabulary, train, test, training, device):
    """Train on `train` for the configured epochs, reporting on `test` after each.
    Returns (history, held-out evaluation)."""
    optimizer = build_optimizer(net, training)
    parameters = [parameter for _, parameter in trainable(net, training)]
    chosen = {id(parameter) for parameter in parameters}
    for parameter in net.parameters():
        parameter.requires_grad_(id(parameter) in chosen)
    total = training.epochs * math.ceil(len(train) / training.logical_batch_size)
    history, update, started = [], 0, time.monotonic()
    for epoch in range(1, training.epochs + 1):
        net.train()
        random.Random(f"{training.seed}:{epoch}").shuffle(train)
        loss_sum, count, grad_norms = 0.0, 0, []
        for parts in batches(train, vocabulary, training):
            optimizer.zero_grad(set_to_none=True)
            size = sum(len(positions) for positions, _, _ in parts)
            for _, batch, targets in parts:
                with autocast(device, training.precision):
                    logits = net(batch.to(device))
                # The parts of a logical batch add up to its mean loss, whatever the split.
                loss = losses(logits.float(), targets.to(device), training.act_weight).sum() / size
                loss.backward()
                loss_sum += float(loss.detach()) * size
            norm = torch.nn.utils.clip_grad_norm_(parameters, training.max_grad_norm)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Non-finite gradients; update cancelled")
            grad_norms.append(float(norm))
            factor = rate_factor(update, total, training.warmup_updates)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor
            optimizer.step()
            update += 1
            count += size
        held_out = evaluate(test, predict_rows(net, vocabulary, test, training, device))
        history.append(dict(
            epoch=epoch, updates=update, train_loss=round(loss_sum / count, 5),
            grad_norm=round(sum(grad_norms) / len(grad_norms), 4),
            win=dict(explained_variance=held_out["win"]["explained_variance"], brier=held_out["win"]["brier"]),
            act=dict(explained_variance=held_out["act"]["explained_variance"], brier=held_out["act"]["brier"]),
            seconds=round(time.monotonic() - started)))
        if device.type == "cuda":
            history[-1]["peak_memory_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        print(json.dumps(history[-1]), flush=True)
    return history, held_out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path, nargs="+",
                        help="sampled rounds (round-N/), a buffer directory of rounds, or run directories "
                             "and files (JSON per run, or JSONL)")
    parser.add_argument("--buffer-iterations", type=int,
                        help="of a buffer directory, train on the rounds of its last so many iterations")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="policy profile: its model section is the network, its critic section the settings")
    parser.add_argument("--checkpoint", type=Path, help="policy checkpoint: the frozen vocabulary and, by default, the body's initial weights")
    parser.add_argument("--pretrained", type=Path, help="saved critic to continue from; its vocabulary and version are kept")
    parser.add_argument("--save", type=Path, help="directory for the trained critic (weights, manifest)")
    parser.add_argument("--report", type=Path, help="also write the report here (default: console and manifest only)")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--init", choices=("checkpoint", "random"))
    parser.add_argument("--freeze-body", action="store_false", dest="train_body", default=None,
                        help="train the head only: a linear probe of the policy's representation")
    parser.add_argument("--train-blocks", type=int, dest="train_blocks",
                        help="train only this many global layers from the top, with the final norm and the head")
    parser.add_argument("--muon-lr", type=float, dest="muon_lr")
    parser.add_argument("--backbone-lr", type=float, dest="backbone_lr")
    parser.add_argument("--embedding-lr", type=float, dest="embedding_lr")
    parser.add_argument("--head-lr", type=float, dest="head_lr")
    parser.add_argument("--batch", type=int, dest="logical_batch_size", help="decisions per update")
    parser.add_argument("--token-budget", type=int, dest="token_budget")
    parser.add_argument("--holdout", type=float, help="share of the seeds held out")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", choices=("no", "bf16"))
    parser.add_argument("--loader-workers", type=int, dest="loader_workers", help="processes that pack the batches")
    parser.add_argument("--train-runs", type=int, help="use only this many training seeds (learning curve)")
    parser.add_argument("--no-baselines", action="store_true", help="skip the constant and logistic baselines")
    parser.add_argument("--no-compile", action="store_true", help="do not compile the global layers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if bool(args.pretrained) == bool(args.checkpoint):
        parser.error("give --checkpoint (a fresh critic) or --pretrained (the next iteration), not both")
    config = CriticConfig.load(args.config)
    config.training = training = override(config.training, args, OVERRIDES)
    if args.no_compile:
        config.model = replace(config.model, compile_blocks=False)
    torch.manual_seed(training.seed)
    device = torch.device(args.device)

    rows, data = load_rows(args.data, args.buffer_iterations)
    train, test = split(rows, training.holdout)
    if args.train_runs:
        kept = sorted({r.seed for r in train})
        random.Random(f"{training.seed}:train-runs").shuffle(kept)
        kept = set(kept[:args.train_runs])
        train = [r for r in train if r.seed in kept]
    report = dict(data=dict(data, coverage=coverage(rows), split=dict(
        holdout=training.holdout, train=len(train), train_seeds=len({r.seed for r in train}),
        test=len(test), test_seeds=len({r.seed for r in test}))),
        train=dict(constant_loss=round(constant_loss(train, training.act_weight), 5)))

    if args.pretrained:
        scorer = load_critic(args.pretrained, device)
        net, vocabulary, previous = scorer.net, scorer.vocabulary, scorer.manifest
        if architecture(previous["model"]) != architecture(asdict(config.model)):
            raise ValueError("The saved critic has a different architecture; its manifest holds the profile to use")
        version, policy = previous["version"] + 1, previous["policy"]
        pretrained = str(args.pretrained)
    else:
        from model.rollout import policy_digest

        policy_net, vocabulary, manifest = load_policy(args.checkpoint, "cpu")
        if training.init == "checkpoint" and architecture(manifest["model"]) != architecture(asdict(config.model)):
            raise ValueError("The profile's architecture differs from the checkpoint's; use --init random or its profile")
        net = Critic(config.model)
        if training.init == "checkpoint":
            net.load_body(policy_net.state_dict())
        set_head_bias(net, base_rates(train))
        policy = dict(checkpoint=str(args.checkpoint), policy_version=manifest["progress"].get("policy_version"),
                      digest=policy_digest(policy_net), vocabulary_hash=vocabulary.digest)
        version, pretrained = 1, None
        del policy_net
    net.to(device)
    report["model"] = dict(version=version, pretrained=pretrained, init=training.init if not pretrained else "pretrained",
                           parameters=sum(p.numel() for _, p in net.trainable()),
                           trained=sum(p.numel() for _, p in trainable(net, training)), policy=policy)
    if not args.no_baselines:
        report["baselines"] = fit_baselines(train, test)
        print(json.dumps(dict(baselines=report["baselines"]), indent=1), flush=True)
    report["history"], report["critic"] = fit(net, vocabulary, train, test, training, device)
    report["unregistered"] = vocabulary.unregistered()
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    if args.save:
        save(args.save, net, vocabulary, config, version=version, pretrained=pretrained, policy=policy,
             data=report["data"], report={k: report[k] for k in ("baselines", "critic", "history", "train",
                                                                 "unregistered") if k in report})
    return report


if __name__ == "__main__":
    main()
