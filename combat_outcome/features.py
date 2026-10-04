"""Tabular deck features and the ridge regression of the combat-outcome baselines."""

from collections import Counter

import torch


def number(stats, key):
    value = (stats or {}).get(key, 0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def deck_summary(info):
    deck = info["deck"]
    n = max(1, len(deck))
    f = Counter()
    for c in deck:
        stats = c.get("stats") or {}
        hits = max(1.0, number(stats, "Repeat"))
        f["n_" + str(c.get("card_type"))] += 1
        f["rarity_" + str(c.get("rarity"))] += 1
        f["cost_sum"] += max(0, c.get("cost") or 0)
        f["x_cost"] += bool(c.get("x_cost"))
        f["upgraded"] += bool(c.get("upgraded"))
        f["damage"] += number(stats, "Damage") * hits
        f["aoe_damage"] += number(stats, "Damage") * hits * (c.get("target_type") == "AllEnemies")
        f["block"] += number(stats, "Block")
        f["draw"] += number(stats, "Cards")
        f["energy"] += number(stats, "Energy")
        f["strength"] += number(stats, "StrengthPower")
        f["dexterity"] += number(stats, "DexterityPower")
        f["weak_vuln"] += number(stats, "WeakPower") + number(stats, "VulnerablePower")
        f["exhaust"] += "Exhaust" in (c.get("keywords") or [])
        f["basic"] += ("STRIKE_" in c["content_id"]) or ("DEFEND_" in c["content_id"])
    out = {k: v / n for k, v in f.items() if not k.startswith(("n_", "rarity_"))}
    out.update({k: v / n for k, v in f.items() if k.startswith(("n_", "rarity_"))})
    out.update(deck_size=len(deck), upgraded_total=f["upgraded"], damage_total=f["damage"],
               block_total=f["block"], relic_count=len(info["relics"]), max_hp=info["max_hp"])
    return out


def bag(info):
    out = Counter()
    for c in info["deck"]:
        out[("up:" if c.get("upgraded") else "card:") + c["content_id"]] += 1
    for r in info["relics"]:
        out["relic:" + r] = 1
    return out


class Design:
    def __init__(self, rows, groups):
        self.groups = groups
        self.names = sorted({k for r in rows for g in groups for k in r[g]})
        self.index = {k: i for i, k in enumerate(self.names)}

    def matrix(self, rows):
        x = torch.zeros(len(rows), len(self.names), dtype=torch.float64)
        for i, r in enumerate(rows):
            for g in self.groups:
                for k, v in r[g].items():
                    j = self.index.get(k)
                    if j is not None:
                        x[i, j] = float(v)
        return x


def ridge(design, rows, targets, l2, chunk=4096):
    """Ridge on standardized columns. Moments and normal equations are accumulated a
    chunk of rows at a time, so the design matrix of a large data set is never held whole."""
    parts = [(rows[i:i + chunk], torch.tensor(targets[i:i + chunk], dtype=torch.float64))
             for i in range(0, len(rows), chunk)]
    n, k = len(rows), len(design.names)
    total, square = torch.zeros(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
    for part, _ in parts:
        x = design.matrix(part)
        total += x.sum(0)
        square += (x * x).sum(0)
    mean = total / n
    std = ((square - n * mean * mean).clamp_min(0) / max(1, n - 1)).sqrt().clamp_min(1e-6)
    offset = sum(targets) / n
    a, b = l2 * torch.eye(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
    for part, y in parts:
        z = (design.matrix(part) - mean) / std
        a += z.T @ z
        b += z.T @ (y - offset)
    w = torch.linalg.solve(a, b)

    def predict(rows):
        out = []
        for i in range(0, len(rows), chunk):
            out += (((design.matrix(rows[i:i + chunk]) - mean) / std) @ w + offset).tolist()
        return out
    return predict
