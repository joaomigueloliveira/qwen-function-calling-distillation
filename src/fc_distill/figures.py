"""Figures for the README and the story site, drawn from results/methods.json (#31).

Each figure is shaped once into tidy rows. `fc-figures` writes those rows as JSON for the site
(#22), which draws them interactively, and draws the same rows as SVG for the README, in a light
and a dark variant.

    uv run fc-figures            # docs/figures/<name>{.json,.svg,-dark.svg}
"""

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fc_distill.tracking import ENTITY, PROJECT

if TYPE_CHECKING:
    from matplotlib.axes import Axes

Rows = list[dict[str, Any]]
# The W&B app on CoreWeave's Forge, where the public project lives (docs/wandb-dashboard.md).
WANDB_APP = "https://forge.coreweave.com/wandb"
DEFAULT_RESULTS = Path("results/methods.json")
DEFAULT_OUT = Path("docs/figures")
SOURCE = "results/methods.json"
# The runs whose curves the cost figure draws, and the runs BFCL was scored along (#70, #88).
CURVES = ("A", "B", "C", "D")
BFCL_RUNS = ("B", "Bf", "D", "Cf")


def cost_accuracy(data: Mapping[str, Any]) -> dict[str, Rows]:
    """Every method's checkpoints on validation against GPU-hours, its pick on test-clean, and the
    untrained baselines on both splits."""
    checkpoints: Rows = []
    picks: Rows = []
    for key, m in data["methods"].items():
        for c in m["checkpoints"]:
            checkpoints.append(
                {
                    "method": key,
                    "labels": m["labels"],
                    "step": c["step"],
                    "gpu_hours": c["gpu_hours"],
                    "exact": c["exact"],
                    "pick": c["step"] == m["selected"],
                }
            )
        test_clean = m["official"].get("test-clean")
        if test_clean is not None:
            pick = next(c for c in m["checkpoints"] if c["step"] == m["selected"])
            picks.append(
                {
                    "method": key,
                    "labels": m["labels"],
                    "step": m["selected"],
                    "gpu_hours": pick["gpu_hours"],
                    "exact": test_clean["exact"],
                    "ci": test_clean["exact_ci"],
                }
            )
    baselines = data["baselines"]
    rows = [
        {"model": model, "split": "validation", "exact": block["exact"]}
        for model, block in baselines["validation"].items()
    ] + [
        {"model": model, "split": "test-clean", "exact": splits["test-clean"]["exact"]}
        for model, splits in baselines["official"].items()
    ]
    return {"checkpoints": checkpoints, "picks": picks, "baselines": rows}


def bfcl_along(data: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Rows]:
    """BFCL exact at every checkpoint scored there, for the methods in `keys`, and the untrained
    baselines on BFCL: whether a method's lead holds along its run or only at its pick."""
    runs: Rows = []
    for key in keys:
        m = data["methods"][key]
        for c in m["checkpoints"]:
            if c["bfcl"] is None:
                continue
            runs.append(
                {
                    "method": key,
                    "labels": m["labels"],
                    "step": c["step"],
                    "gpu_hours": c["gpu_hours"],
                    "exact": c["bfcl"]["exact"],
                    "ci": c["bfcl"]["exact_ci"],
                    "pick": c["step"] == m["selected"],
                }
            )
    baselines = [
        {"model": model, "exact": splits["bfcl"]["exact"]}
        for model, splits in data["baselines"]["official"].items()
    ]
    return {"runs": runs, "baselines": baselines}


def readouts(data: Mapping[str, Any]) -> Rows:
    """Per model, in the report's order: the share of default-made-explicit rows answered right,
    and the shares of test-clean answers that are fenced or span several lines."""
    models = [*data["baselines"]["validation"], *data["methods"]]
    rows: Rows = []
    for model in models:
        defaults = data["defaults"].get(model, {})
        formats = data["formats"].get(model, {})
        rows.append(
            {
                "model": model,
                "default_made_explicit": defaults.get("default_made_explicit"),
                "fenced": formats.get("fenced"),
                "multi_line": formats.get("multi_line"),
            }
        )
    return rows


def wandb_links(data: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    """The W&B page of each method's training run, for the methods in `keys` that logged one."""
    return {
        key: f"{WANDB_APP}/{ENTITY}/{PROJECT}/runs/{run}"
        for key in keys
        if (run := data["methods"][key]["wandb_run"]) is not None
    }


# --- drawing -------------------------------------------------------------------------------------

LABELS = {
    "A": "A",
    "B": "B",
    "Bf": "filtered B",
    "Bg": "grounded B",
    "C": "C",
    "Cf": "filtered C",
    "D": "D",
    "B→D": "B then D",
}
# Hue is the method's family; hollow markers and dashed lines mark the methods that used labels.
FAMILY = {"A": "A", "B": "B", "Bf": "B", "Bg": "B", "C": "C", "Cf": "C", "D": "D", "B→D": "D"}
# The dataviz reference palette's first three categorical slots, validated all-pairs in both
# modes, and its neutral tokens. Aqua is under 3:1 on the light surface: every series is labelled.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "A": "#52514e",
        "B": "#2a78d6",
        "D": "#eb6834",
        "C": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "A": "#c3c2b7",
        "B": "#3987e5",
        "D": "#d95926",
        "C": "#199e70",
    },
}

# Where a line's label goes when its end would sit on another line: at the line's start, or nudged
# up or down (points). Set by eye on the current results.
COST_LABELS: dict[str, tuple[str, float]] = {"B": ("start", 0)}
BFCL_LABELS: dict[str, tuple[str, float]] = {"B": ("end", -6), "Bf": ("end", -10)}

CAPTIONS = {
    "cost-accuracy": (
        "Left: exact match on validation at every checkpoint against GPU-hours, the teacher's "
        "passes included, for A, B, C and D; rings mark each run's pick (its best validation "
        "score, ADR 0006). Right: each method's pick on test-clean with its 95% interval; hollow "
        "markers used labels. Dashed lines: the untrained 0.8B and the 2B teacher."
    ),
    "bfcl-along": (
        "Exact match on BFCL, tools never seen in training, at every checkpoint scored there, "
        "against GPU-hours. Rings mark the pick, which validation chose; dashed lines used labels. "
        "Each point's 95% interval is about ±2.5 points (in the JSON, not drawn)."
    ),
    "readouts": (
        "Left: exact match on the 123 test-clean rows where gold and a baseline differed only by "
        "an optional argument at its default. Middle and right: the share of test-clean answers "
        "that are fenced or span several lines; the scorer reads both."
    ),
}


def _style(ax: "Axes", t: Mapping[str, str]) -> None:
    ax.set_facecolor(t["surface"])
    ax.grid(True, color=t["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["text2"], labelsize=9)
    ax.xaxis.label.set_color(t["text2"])
    ax.yaxis.label.set_color(t["text2"])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")


def _title(ax: "Axes", t: Mapping[str, str], text: str, size: float = 11) -> None:
    ax.set_title(text, loc="left", fontsize=size, color=t["text"])


def _baseline(ax: "Axes", t: Mapping[str, str], y: float, label: str, *, vertical: bool) -> None:
    if vertical:
        ax.axvline(y, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
        ax.annotate(
            label,
            (y, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=t["muted"],
        )
    else:
        ax.axhline(y, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
        ax.annotate(
            label,
            (0, y),
            xycoords=("axes fraction", "data"),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
            color=t["muted"],
        )


def _line(
    ax: "Axes",
    t: Mapping[str, str],
    key: str,
    labels: bool,
    rows: Rows,
    place: tuple[str, float] = ("end", 0),
) -> None:
    color = t[FAMILY[key]]
    xs = [r["gpu_hours"] for r in rows]
    ys = [100 * r["exact"] for r in rows]
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=2,
        linestyle=(0, (5, 3)) if labels else "-",
        marker="o",
        markersize=4.5,
        markerfacecolor=t["surface"] if labels else color,
        markeredgecolor=color,
    )
    for r in rows:
        if r["pick"]:
            ax.plot(
                r["gpu_hours"],
                100 * r["exact"],
                "o",
                markersize=11,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.5,
            )
    where, nudge = place
    at = 0 if where == "start" else -1
    ax.annotate(
        LABELS[key],
        (xs[at], ys[at]),
        xytext=(-6 if where == "start" else 6, nudge),
        textcoords="offset points",
        ha="right" if where == "start" else "left",
        va="center",
        fontsize=9,
        color=t["text"],
    )


def _cost_accuracy_figure(fig_data: Mapping[str, Rows], t: Mapping[str, str]) -> Any:
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    for ax in (left, right):
        _style(ax, t)
    by_method: dict[str, Rows] = {}
    for r in fig_data["checkpoints"]:
        by_method.setdefault(r["method"], []).append(r)
    for key in CURVES:
        if key in by_method:
            rows = by_method[key]
            _line(left, t, key, rows[0]["labels"], rows, COST_LABELS.get(key, ("end", 0)))
    for b in fig_data["baselines"]:
        if b["split"] == "validation":
            name = "2B teacher" if b["model"] == "raw 2B" else "untrained 0.8B"
            _baseline(left, t, 100 * b["exact"], name, vertical=False)
    left.set_xlabel("GPU-hours (RTX 3060)")
    left.set_ylabel("exact match, validation (%)")
    _title(left, t, "Along each run, on validation")

    picks = sorted(fig_data["picks"], key=lambda r: r["exact"])
    for i, r in enumerate(picks):
        color = t[FAMILY[r["method"]]]
        lo, hi = (100 * v for v in r["ci"])
        right.plot([lo, hi], [i, i], color=color, linewidth=2, solid_capstyle="round")
        right.plot(
            100 * r["exact"],
            i,
            "o",
            markersize=7,
            color=color,
            markeredgecolor=color,
            markerfacecolor=t["surface"] if r["labels"] else color,
            markeredgewidth=1.5,
        )
        right.annotate(
            f"{r['gpu_hours']:.1f} h",
            (1, i),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=t["muted"],
        )
    right.set_yticks(range(len(picks)), [LABELS[r["method"]] for r in picks])
    for b in fig_data["baselines"]:
        if b["split"] == "test-clean":
            name = "2B" if b["model"] == "raw 2B" else "0.8B"
            _baseline(right, t, 100 * b["exact"], name, vertical=True)
    right.set_xlim(50, 100)
    right.set_ylim(-0.7, len(picks) - 0.3)
    right.grid(False, axis="y")
    right.set_xlabel("exact match, test-clean (%)")
    _title(right, t, "Each pick on test-clean")
    fig.tight_layout()
    return fig


def _bfcl_figure(fig_data: Mapping[str, Rows], t: Mapping[str, str]) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(figsize=(8, 4.4))
    _style(ax, t)
    by_method: dict[str, Rows] = {}
    for r in fig_data["runs"]:
        by_method.setdefault(r["method"], []).append(r)
    for key, rows in by_method.items():
        _line(ax, t, key, rows[0]["labels"], rows, BFCL_LABELS.get(key, ("end", 0)))
    for b in fig_data["baselines"]:
        name = "2B teacher" if b["model"] == "raw 2B" else "untrained 0.8B"
        _baseline(ax, t, 100 * b["exact"], name, vertical=False)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.08)
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel("GPU-hours (RTX 3060)")
    ax.set_ylabel("exact match, BFCL (%)")
    _title(ax, t, "Unseen tools (BFCL) along each run")
    fig.tight_layout()
    return fig


def _readouts_figure(rows: Rows, t: Mapping[str, str]) -> Any:
    import matplotlib.pyplot as plt

    panels = (
        ("default_made_explicit", "exact on default rows (%)"),
        ("fenced", "fenced answers (%)"),
        ("multi_line", "multi-line answers (%)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    names = [LABELS.get(r["model"], r["model"]) for r in rows]
    ys = range(len(rows))
    for ax, (field, title) in zip(axes, panels, strict=True):
        _style(ax, t)
        values = [100 * r[field] if r[field] is not None else 0.0 for r in rows]
        ax.barh(ys, values, height=0.62, color=t["B"])
        for y, r, v in zip(ys, rows, values, strict=True):
            text = "n/a" if r[field] is None else f"{v:.1f}"
            ax.annotate(
                text,
                (v, y),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=t["text2"],
            )
        _title(ax, t, title, 10)
        ax.grid(False, axis="y")
        ax.set_xlim(0, max(values) * 1.25 or 1)
    axes[0].set_yticks(list(ys), names)
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


DRAW: dict[str, Callable[[Any, Mapping[str, str]], Any]] = {
    "cost-accuracy": _cost_accuracy_figure,
    "bfcl-along": _bfcl_figure,
    "readouts": _readouts_figure,
}


def figures(data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each figure's rows, caption, source and W&B links: what the site draws, and the SVGs."""
    every = tuple(data["methods"])
    bfcl = tuple(k for k in BFCL_RUNS if k in data["methods"])
    shaped: dict[str, tuple[Any, tuple[str, ...]]] = {
        "cost-accuracy": (cost_accuracy(data), every),
        "bfcl-along": (bfcl_along(data, bfcl), bfcl),
        "readouts": (readouts(data), every),
    }
    return {
        name: {
            "caption": CAPTIONS[name],
            "source": SOURCE,
            "wandb": wandb_links(data, keys),
            "data": rows,
        }
        for name, (rows, keys) in shaped.items()
    }


def write(data: Mapping[str, Any], out: Path) -> list[Path]:
    """Write every figure's JSON and its light and dark SVGs to `out`; return the paths."""
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt

    # Text stays text and ids are stable, so a re-run with the same numbers changes no file.
    matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "fc-figures"})
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, figure in figures(data).items():
        path = out / f"{name}.json"
        path.write_text(json.dumps(figure, indent=1, ensure_ascii=False) + "\n")
        written.append(path)
        for theme, suffix in (("light", ""), ("dark", "-dark")):
            t = THEMES[theme]
            fig = DRAW[name](figure["data"], t)
            fig.patch.set_facecolor(t["surface"])
            path = out / f"{name}{suffix}.svg"
            fig.savefig(path, facecolor=t["surface"], metadata={"Date": None})
            # matplotlib's SVG has trailing spaces; the repo's whitespace hook would strip them.
            lines = path.read_text().splitlines()
            path.write_text("".join(line.rstrip() + "\n" for line in lines))
            plt.close(fig)
            written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    for path in write(json.loads(args.results.read_text()), args.out):
        print(f"wrote {path}")
