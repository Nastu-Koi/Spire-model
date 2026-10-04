"""Ridge baselines on the same seed-grouped split as the entity model."""

import sys

from .data import entities
from .features import Design, bag, deck_summary, ridge
from .frames import summary_info
from .metrics import report as fit_report

BASELINES = (("encounter_only", ["context"]), ("ridge_summary", ["context", "summary"]),
             ("ridge_summary_bag", ["context", "summary", "bag"]))
RIDGE_BASE = "ridge_summary"


def tabular(row):
    info = summary_info(row, entities(row))
    context = {"enc:" + row["encounter"]: 1, "kind:" + row["kind"]: 1, "char:" + str(row["character"]): 1,
               "act": row["act"], "max_hp": row["max_hp"], "start_hp": row["start_hp"],
               "ascension": row["ascension"] or 0}
    # Names repeat across every row of a large data set: keep one copy of each.
    return {group: {sys.intern(str(k)): v for k, v in values.items()}
            for group, values in (("context", context), ("summary", deck_summary(info)), ("bag", bag(info)))}


def fit_baselines(train, test, ridge_base=False):
    """Report every baseline on the test rows. With `ridge_base`, each row also gets
    `base`: the RIDGE_BASE prediction as a fraction of max HP, the entity model's offset."""
    tab_train, tab_test = [tabular(r) for r in train], [tabular(r) for r in test]
    out = {}
    for name, groups in BASELINES:
        design = Design(tab_train, groups)
        model = ridge(design, tab_train, [r["y"] for r in train], l2=float(len(design.names)))
        if name == RIDGE_BASE and ridge_base:
            for rows, tab in ((train, tab_train), (test, tab_test)):
                for r, b in zip(rows, model(tab)):
                    r["base"] = b / r["max_hp"]
        out[name] = fit_report(test, model(tab_test))
    return out
