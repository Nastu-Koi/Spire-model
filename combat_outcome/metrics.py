"""Held-out fit in HP units, overall and by kind of fight."""

import statistics as st
from collections import defaultdict


def r2(pred, true):
    mean = st.mean(true)
    spread = sum((t - mean) ** 2 for t in true)
    return 1 - sum((p - t) ** 2 for p, t in zip(pred, true)) / spread if spread else None


def report(rows, predictions):
    """R^2 and mean absolute error over all rows and for each kind of fight."""
    groups = defaultdict(lambda: ([], []))
    for row, p in zip(rows, predictions):
        for name in ("all", row["kind"]):
            groups[name][0].append(p)
            groups[name][1].append(row["y"])
    out = {}
    for name, (pred, true) in sorted(groups.items()):
        fit = r2(pred, true)
        out[name] = dict(n=len(true), r2=None if fit is None else round(fit, 4),
                         mae=round(st.mean(abs(p - t) for p, t in zip(pred, true)), 2))
    return out
