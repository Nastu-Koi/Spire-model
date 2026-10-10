"""Held-out quality of probabilities: explained variance of the ±1 return, Brier score,
log loss, ranking, and calibration tables."""

import math
import statistics as st
from collections import defaultdict

EPSILON = 1e-7


def explained_variance(probabilities, outcomes):
    """1 - MSE(2p-1, ±1) / Var(±1): the share of the ±1 return's variance the critic's
    value 2p-1 explains. Zero for the constant base rate of these outcomes, negative
    for anything worse. None when the outcomes do not vary."""
    returns = [2.0 * y - 1.0 for y in outcomes]
    mean = st.mean(returns)
    variance = st.mean((r - mean) ** 2 for r in returns)
    if variance == 0:
        return None
    error = st.mean((2.0 * p - 1.0 - r) ** 2 for p, r in zip(probabilities, returns))
    return 1.0 - error / variance


def brier(probabilities, outcomes):
    return st.mean((p - y) ** 2 for p, y in zip(probabilities, outcomes))


def log_loss(probabilities, outcomes):
    return -st.mean(math.log(min(max(p, EPSILON), 1 - EPSILON)) if y else
                    math.log(min(max(1 - p, EPSILON), 1 - EPSILON))
                    for p, y in zip(probabilities, outcomes))


def auc(probabilities, outcomes):
    """Probability that a positive outranks a negative, ties counting half."""
    positives = [p for p, y in zip(probabilities, outcomes) if y]
    negatives = [p for p, y in zip(probabilities, outcomes) if not y]
    if not positives or not negatives:
        return None
    ranked = sorted(zip(probabilities, outcomes))
    total, index = 0.0, 0
    while index < len(ranked):
        tied = index
        while tied < len(ranked) and ranked[tied][0] == ranked[index][0]:
            tied += 1
        # Average rank of the tie, one-based.
        rank = (index + tied + 1) / 2
        total += rank * sum(1 for _, y in ranked[index:tied] if y)
        index = tied
    n = len(positives)
    return (total - n * (n + 1) / 2) / (n * len(negatives))


def summary(probabilities, outcomes):
    probabilities, outcomes = list(map(float, probabilities)), [bool(y) for y in outcomes]
    fit = explained_variance(probabilities, outcomes)
    rank = auc(probabilities, outcomes)
    return dict(n=len(outcomes), rate=round(st.mean(outcomes), 4), predicted=round(st.mean(probabilities), 4),
                brier=round(brier(probabilities, outcomes), 5),
                explained_variance=None if fit is None else round(fit, 4),
                log_loss=round(log_loss(probabilities, outcomes), 5),
                auc=None if rank is None else round(rank, 4))


def calibration(keys, probabilities, outcomes):
    """Predicted mean against the actual rate for every value of `keys`."""
    groups = defaultdict(lambda: ([], []))
    for key, p, y in zip(keys, probabilities, outcomes):
        groups[key][0].append(float(p))
        groups[key][1].append(bool(y))
    return [dict(key=key, n=len(ys), actual=round(st.mean(ys), 4), predicted=round(st.mean(ps), 4),
                 gap=round(st.mean(ps) - st.mean(ys), 4))
            for key, (ps, ys) in sorted(groups.items(), key=lambda item: str(item[0]))]


def reliability(probabilities, outcomes, bins=10):
    """Calibration by predicted probability: equal-width bins over [0, 1]."""
    keys = [min(int(float(p) * bins), bins - 1) for p in probabilities]
    rows = calibration(keys, probabilities, outcomes)
    for row in rows:
        row["bin"] = [round(row["key"] / bins, 2), round((row["key"] + 1) / bins, 2)]
        del row["key"]
    return rows
