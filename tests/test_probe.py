import math

import numpy as np

from fc_distill.probe import summarize_position


def softmax_from_summary(entry: dict, temperature: float = 1.0) -> tuple[dict[str, float], float]:
    """The site's reconstruction (site/essay.js, soft()): exact top tokens, the rest by bin."""
    top = {tok: math.exp(logit / temperature) for tok, logit in entry["top"]}
    rest = sum(count * math.exp(centre / temperature) for centre, count in entry["rest"])
    z = sum(top.values()) + rest
    return {tok: v / z for tok, v in top.items()}, rest / z


def test_the_summary_rebuilds_both_models_softmax() -> None:
    rng = np.random.default_rng(0)
    vocab = [f"t{i}" for i in range(5000)]
    teacher = rng.normal(0, 3, 5000)
    student = rng.normal(0, 3, 5000)
    teacher[42], student[7] = 25.0, 22.0  # each model has a clear favourite

    summary = summarize_position(teacher, student, vocab, k=8, bins=120)

    for name, logits in (("2B", teacher), ("0.8B", student)):
        full = np.exp(logits - logits.max())
        full /= full.sum()
        top, rest = softmax_from_summary(summary[name])
        # Both models list the same tokens: the union of each one's top 8, teacher's order first.
        assert list(top) == [tok for tok, _ in summary["2B"]["top"]]
        assert {"t42", "t7"} <= set(top)
        assert 8 <= len(top) <= 16
        for tok, p in top.items():
            assert math.isclose(p, full[int(tok[1:])], rel_tol=1e-3)
        assert math.isclose(
            rest, 1 - sum(full[int(t[1:])] for t in top), rel_tol=1e-2, abs_tol=1e-6
        )
        assert sum(count for _, count in summary[name]["rest"]) == 5000 - len(top)
