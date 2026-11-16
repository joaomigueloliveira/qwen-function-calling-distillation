"""The numbers the story site quotes (#22), computed from results/ when the site builds.

The site's prose calls these in inline Python, so no number on the page is copied by hand:

    story = Story.load()
    story.score("B")                         # "83.4%", on test-clean
    story.claim("B", "raw 2B", "test-clean")  # "159 vs 52 rows, Holm p < 0.0001"
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fc_distill.figures import wandb_links

# The repository root, so a page in any folder of site/ finds results/ and data/.
ROOT = Path(__file__).resolve().parents[2]

# The method the README's claim names (ADR 0009, amended by #95).
HEADLINE = "B"


class Story:
    def __init__(self, methods: Mapping[str, Any], serving: Mapping[str, Any]) -> None:
        self.methods = methods
        self.serving_data = serving

    @classmethod
    def load(cls, results: Path = ROOT / "results") -> "Story":
        return cls(
            json.loads((results / "methods.json").read_text()),
            json.loads((results / "serving.json").read_text()),
        )

    def _exact(self, who: str, split: str) -> float:
        baselines = self.methods["baselines"]
        if who in baselines["validation"]:
            if split == "validation":
                return float(baselines["validation"][who]["exact"])
            return float(baselines["official"][who][split]["exact"])
        m = self.methods["methods"][who]
        if split == "validation":
            return float(self._selected(who)["exact"])
        return float(m["official"][split]["exact"])

    def _selected(self, key: str) -> Mapping[str, Any]:
        m = self.methods["methods"][key]
        return next(c for c in m["checkpoints"] if c["step"] == m["selected"])

    def gpu_hours(self, key: str) -> str:
        """A method's GPU-hours at its selected checkpoint, as printed ("3.1")."""
        return f"{self._selected(key)['gpu_hours']:.1f}"

    def facts(self, key: str) -> dict[str, str]:
        """A method's row of the results table, as printed: its selected checkpoint's scores."""
        m = self.methods["methods"][key]
        return {
            "name": m["name"],
            "labels": "yes" if m["labels"] else "no",
            "step": str(m["selected"]),
            "gpu_hours": self.gpu_hours(key),
            "validation": self.score(key, "validation"),
            "test": self.score(key, "test"),
            "test_clean": self.score(key),
            "bfcl": self.score(key, "bfcl"),
            "wandb": wandb_links(self.methods, (key,)).get(key, ""),
        }

    def score(self, who: str, split: str = "test-clean") -> str:
        """A model's strict exact on a split, as printed: a baseline, or a method's selected
        checkpoint."""
        return f"{100 * self._exact(who, split):.1f}%"

    def claim(self, a: str, b: str, split: str) -> str:
        """One of the README's paired claims (methods.json's `claims`), as printed: the rows only
        each model gets right, and the Holm-adjusted p-value over every claim."""
        c = next(c for c in self.methods["claims"] if (c["a"], c["b"], c["eval"]) == (a, b, split))
        return f"{c['only_a_right']} vs {c['only_b_right']} rows, Holm p {p_value(c['holm_p'])}"

    def headline(self) -> dict[str, str]:
        """The destination claim's numbers for the headline method, on test-clean."""
        gains = self.methods["headline"][f"{HEADLINE}/test-clean"]
        return {
            "method": HEADLINE,
            "before": self.score("raw 0.8B"),
            "after": self.score(HEADLINE),
            "teacher": self.score("raw 2B"),
            "with_labels": self.score("A"),
            "share_of_a": f"{100 * gains['share_of_a']:.0f}%",
            "gap_recovered": f"{100 * gains['gap_recovered']:.0f}%",
            "gpu_hours": self.gpu_hours(HEADLINE),
        }

    def serving(self) -> dict[str, str]:
        """How much faster the 0.8B serves than the 2B in vLLM: every prompt batched, and one
        request at a time (results/serving.json)."""
        return {
            "batched": f"{self.serving_data['speedup']:.1f}\u00d7",
            "one_at_a_time": f"{self.serving_data['latency_speedup']:.1f}\u00d7",
        }


def p_value(p: float) -> str:
    """'< 0.0001', '= 0.009' (three decimals under 0.01), '= 0.10' (two otherwise)."""
    if p < 0.0001:
        return "< 0.0001"
    return f"= {p:.3f}" if p < 0.01 else f"= {p:.2f}"
