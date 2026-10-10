"""Whether a trained critic may weigh the policy's decisions.

A critic whose values are wrong is worse than none: advantage weighting gives the
actions it overrates the large weights. The gate reads the report of a training run
(`critic.train`) and holds the critic to floors measured on the held-out seeds: enough
of them to measure anything, explained variance above both an absolute floor and the
act-and-floor logistic regression on the same split, calibration overall, by act, by
character and by predicted probability, and a training loss that has not fallen far
more than the held-out Brier score improved, which is what memorized runs look like.
The grounds of the default floors are in issue 18's comments
(`.scratch/a0-training-baseline/issues/18-win-probability-critic.md`); they are to be
set again once a critic has been trained on a full sampled round.

    result = check(report)                 # {"passed": ..., "failed": [...], "checks": {...}}
    result = check(report, Gate(min_explained_variance=0.05))
"""

from dataclasses import asdict, dataclass, fields

HEADS = ("win", "act")
# The baseline a critic has to beat: seven parameters on where the decision stands.
BASELINE = "logistic_act_floor"


@dataclass
class Gate:
    # Held-out seeds (runs) under which explained variance is split noise. An
    # iteration's round of 1,000 runs holds out 200 of them give or take 13: the
    # floor stands clear of that, or the split alone turns away half the first critics.
    min_heldout_seeds: int = 150
    # Explained variance of the ±1 return, of each head: an absolute floor, and a
    # margin over the act-and-floor logistic regression on the same split.
    min_explained_variance: float = 0.10
    min_margin_over_baseline: float = 0.05
    # Predicted mean against the actual rate: over all held-out decisions, in every
    # act, for every character, and in every tenth of predicted probability that
    # holds at least `min_bin_share` of the decisions.
    max_overall_gap: float = 0.02
    max_act_gap: float = 0.05
    max_character_gap: float = 0.10
    max_bin_gap: float = 0.15
    min_bin_share: float = 0.05
    # The fall of the training loss under the constant's, over the improvement of the
    # held-out Brier score over the constant's: memorized runs lower the first alone.
    max_memorization_ratio: float = 3.0

    def __post_init__(self):
        if self.min_heldout_seeds < 0 or not 0 <= self.min_bin_share <= 1 or self.max_memorization_ratio < 0:
            raise ValueError("Gate floors must be nonnegative and the bin share a fraction")

    @classmethod
    def of(cls, overrides=()):
        """The default gate with `name=value` overrides, as a command line gives them."""
        kinds = {field.name: field.type for field in fields(cls)}
        values = {}
        for item in overrides:
            name, separator, value = item.partition("=")
            if not separator or name not in kinds:
                raise ValueError(f"Unknown critic gate setting {item!r}; settings: {', '.join(kinds)}")
            values[name] = int(value) if kinds[name] is int else float(value)
        return cls(**values)


def _largest(rows):
    gaps = [abs(row["gap"]) for row in rows]
    return max(gaps) if gaps else None


def check(report, gate=None):
    """Hold a training report to the gate. Returns `passed`, the names of the checks
    that failed, and every check with its value and its limit. A value the report
    cannot give (no baselines, outcomes that do not vary) fails its check."""
    gate = gate or Gate()
    checks = {}

    def at_least(name, value, limit):
        checks[name] = dict(value=value, limit=limit, passed=value is not None and value >= limit)

    def at_most(name, value, limit):
        checks[name] = dict(value=value, limit=limit, passed=value is not None and value <= limit)

    critic, baselines = report["critic"], report.get("baselines") or {}
    at_least("heldout_seeds", report["data"]["split"]["test_seeds"], gate.min_heldout_seeds)
    for head in HEADS:
        measured = critic[head]
        variance = measured["explained_variance"]
        at_least(f"{head}_explained_variance", variance, gate.min_explained_variance)
        reference = (baselines.get(BASELINE) or {}).get(head, {}).get("explained_variance")
        at_least(f"{head}_margin_over_baseline",
                 None if variance is None or reference is None else round(variance - reference, 4),
                 gate.min_margin_over_baseline)
        at_most(f"{head}_overall_gap", round(abs(measured["predicted"] - measured["rate"]), 4), gate.max_overall_gap)
        calibration = critic["calibration"]
        at_most(f"{head}_act_gap", _largest(calibration["by_act"][head]), gate.max_act_gap)
        at_most(f"{head}_character_gap", _largest(calibration["by_character"][head]), gate.max_character_gap)
        bins = calibration["by_probability"][head]
        total = sum(row["n"] for row in bins)
        at_most(f"{head}_bin_gap", _largest([row for row in bins if row["n"] >= gate.min_bin_share * total]),
                gate.max_bin_gap)
    constant = baselines.get("constant")
    fall = improvement = None
    if constant and report.get("history") and "train" in report:
        fall = report["train"]["constant_loss"] - report["history"][-1]["train_loss"]
        improvement = sum(constant[head]["brier"] - critic[head]["brier"] for head in HEADS)
    # A loss that did not fall memorized nothing; one that fell must show on held-out runs.
    memorized = None if fall is None else fall > 0 and fall > gate.max_memorization_ratio * max(improvement, 0.0)
    checks["memorization"] = dict(
        value=None if fall is None else dict(train_loss_fall=round(fall, 5), heldout_brier_gain=round(improvement, 5)),
        limit=gate.max_memorization_ratio, passed=memorized is False)
    failed = [name for name, entry in checks.items() if not entry["passed"]]
    return dict(passed=not failed, failed=failed, checks=checks, gate=asdict(gate))
