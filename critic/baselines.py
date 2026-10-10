"""Baselines on the same seed-grouped split: the constant base rate, a logistic
regression on act and floor only, and one on the player's state summary."""

import math
import statistics as st

import torch

from .metrics import summary

CHARACTERS = ("Ironclad", "Silent", "Defect", "Regent", "Necrobinder")
BASELINES = ("constant", "logistic_act_floor", "logistic_state")


def act_floor(row):
    """Where the decision stands: the act, the floor and their interaction."""
    acts = [float(row.act == act) for act in (1, 2, 3)]
    return acts + [row.floor / 50.0] + [a * row.floor / 50.0 for a in acts]


def state(row):
    """Act and floor plus the character and the player's state summary."""
    s = row.state
    max_hp = max(1.0, float(s.get("max_hp") or 0))
    return act_floor(row) + [float(row.character == c) for c in CHARACTERS] + [
        float(s.get("hp") or 0) / max_hp, max_hp / 100.0, float(s.get("gold") or 0) / 100.0,
        s.get("deck", 0) / 20.0, s.get("relics", 0) / 5.0, s.get("potions", 0) / 3.0]


def fit_logistic(x, y, l2=1e-2, iterations=100):
    """Newton's method on standardized columns with a small ridge; returns a predictor."""
    x, y = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z = torch.cat([torch.ones(len(x), 1, dtype=torch.float64), (x - mean) / std], dim=1)
    w = torch.zeros(z.shape[1], dtype=torch.float64)
    w[0] = math.log(max(float(y.mean()), 1e-6) / max(1 - float(y.mean()), 1e-6))
    ridge = l2 * torch.eye(z.shape[1], dtype=torch.float64)
    ridge[0, 0] = 0.0
    for _ in range(iterations):
        p = torch.sigmoid(z @ w)
        gradient = z.T @ (p - y) + ridge @ w
        hessian = z.T @ (z * (p * (1 - p)).unsqueeze(1)) + ridge
        step = torch.linalg.solve(hessian, gradient)
        w -= step
        if float(step.abs().max()) < 1e-8:
            break

    def predict(rows):
        if not rows:
            return []
        features = torch.tensor(rows, dtype=torch.float64)
        z = torch.cat([torch.ones(len(features), 1, dtype=torch.float64), (features - mean) / std], dim=1)
        return torch.sigmoid(z @ w).tolist()
    return predict


def fit_baselines(train, test):
    """Every baseline's held-out summary for both targets."""
    targets = dict(win=lambda r: r.win, act=lambda r: r.act_passed)
    out = {}
    for name in BASELINES:
        out[name] = {}
        for target, label in targets.items():
            if name == "constant":
                rate = st.mean(label(r) for r in train)
                predictions = [rate] * len(test)
            else:
                features = act_floor if name == "logistic_act_floor" else state
                model = fit_logistic([features(r) for r in train], [float(label(r)) for r in train])
                predictions = model([features(r) for r in test])
            out[name][target] = summary(predictions, [label(r) for r in test])
    return out
