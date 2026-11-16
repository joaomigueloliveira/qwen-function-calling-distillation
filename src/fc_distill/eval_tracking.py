"""How eval results are laid out in W&B, so that every run, baseline or checkpoint, is comparable.

One W&B eval run holds every eval of one model (test, test-clean, BFCL), because W&B panels compare
values within a run. Each eval's scores are flat summary keys, with W&B-safe names (no hyphens,
`+` or leading digits):

    <eval>/<metric>, <eval>/<metric>_lo, <eval>/<metric>_hi, <eval>/n
    <eval>/slice/<dim>/<bucket>/{n, exact}        xLAM slices
    bfcl/slice/<bucket>/..., bfcl/category/<cat>/...

and every slice, with all its metrics and intervals, is one row of a `slices` table. The full
schema, and what trainers log: docs/notes/wandb-dashboard.md.
"""

import re
from collections.abc import Iterator, Mapping
from typing import Any

from fc_distill.tracking import Tracker

# The metrics given per slice in the summary (the table has all of them).
SLICE_METRICS = ("exact", "bfcl_exact")
# Keys of an eval block that hold slices rather than scores.
NESTED = ("overall", "slices", "categories")


def key(name: str) -> str:
    """A W&B-safe key segment: "test-clean" -> "test_clean", "1-2" -> "d1_2", "3+" -> "d3plus"."""
    out = name.replace("-", "_").replace("+", "plus")
    return f"d{out}" if re.match(r"\d", out) else out


def log_evals(tracker: Tracker, evals: Mapping[str, Mapping[str, Any]]) -> None:
    """Log one model's eval blocks ({"test": ..., "test-clean": ..., "bfcl": ...}, each shaped like
    an fc-eval summary or a results/baselines.json entry) as summary keys and a `slices` table."""
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, block in evals.items():
        ev = key(name)
        scalars = {k: v for k, v in block.items() if k not in NESTED} | block.get("overall", {})
        summary |= {f"{ev}/{k}": v for k, v in _split_intervals(scalars).items()}
        for path, dim, bucket, scores in _slices(block):
            prefix = f"{ev}/{path}"
            summary[f"{prefix}/n"] = scores["n"]
            summary |= {f"{prefix}/{m}": scores[m] for m in SLICE_METRICS if m in scores}
            rows.append({"eval": ev, "dim": dim, "bucket": bucket} | _split_intervals(scores))
    tracker.summary(summary)
    columns = list(dict.fromkeys(c for row in rows for c in row))
    tracker.table("slices", columns, [[row.get(c) for c in columns] for row in rows])


def _slices(block: Mapping[str, Any]) -> Iterator[tuple[str, str, str, Mapping[str, Any]]]:
    """(key path, dim, bucket, scores) for xLAM's slices[dim][bucket] and BFCL's flat slices and
    categories."""
    for top, dim in (("slices", "slice"), ("categories", "category")):
        for name, value in block.get(top, {}).items():
            if "n" in value:
                yield f"{dim}/{key(name)}", dim, key(name), value
            else:
                for bucket, scores in value.items():
                    yield f"slice/{key(name)}/{key(bucket)}", key(name), key(bucket), scores


def _split_intervals(block: Mapping[str, Any]) -> dict[str, Any]:
    """Scalars only, with each `<metric>_ci` interval split into `<metric>_lo` and `_hi` (None for
    an empty slice, whose interval is None)."""
    out: dict[str, Any] = {}
    for k, v in block.items():
        if k.endswith("_ci"):
            metric = k.removesuffix("_ci")
            out[f"{metric}_lo"], out[f"{metric}_hi"] = v or (None, None)
        elif not isinstance(v, (dict, list)):
            out[k] = v
    return out
