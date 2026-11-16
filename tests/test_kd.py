import math

import numpy as np
import pytest

from fc_distill.kd import place_targets
from fc_distill.teacher_topk import RowTopk


def _row(positions: list[int], tokens: list[int]) -> RowTopk:
    n = len(positions)
    return RowTopk(
        row_id=7,
        positions=np.array(positions, dtype=np.int32),
        tokens=np.array(tokens, dtype=np.int32),
        ids=np.array([[t, 99] for t in tokens], dtype=np.int32),
        logprobs=np.array([[-0.1, -2.5]] * n, dtype=np.float32),
        token_logprob=np.full(n, -0.1, dtype=np.float32),
        token_rank=np.ones(n, dtype=np.int32),
    )


def test_place_targets_puts_each_stored_position_at_its_label_and_pads_the_rest() -> None:
    labels = np.array([-100, -100, 5, 6, -100])  # a right-padded row: prompt, answer, padding

    ids, logprobs = place_targets(_row([2, 3], [5, 6]), labels)

    np.testing.assert_array_equal(ids, [[-1, -1], [-1, -1], [5, 99], [6, 99], [-1, -1]])
    np.testing.assert_allclose(logprobs[2:4], [[-0.1, -2.5], [-0.1, -2.5]])
    np.testing.assert_array_equal(logprobs[[0, 1, 4]], 0.0)


def test_place_targets_refuses_a_row_whose_labels_are_not_the_stored_tokens() -> None:
    with pytest.raises(ValueError, match="row 7"):
        place_targets(_row([2, 3], [5, 6]), np.array([-100, -100, 5, 8]))
    with pytest.raises(ValueError, match="row 7"):
        place_targets(_row([2, 3], [5, 6]), np.array([-100, 5, 6, -100]))


# The loss itself is torch, which only the train extra installs: these run on the GPU machine's
# venv (`uv run --no-sync pytest`) and are skipped on CPU-only CI.


def test_topk_kl_matches_a_hand_computed_kl_with_the_rest_entry() -> None:
    torch = pytest.importorskip("torch")
    from fc_distill.kd import topk_kl

    # Teacher: 0.5, 0.3 on the two stored tokens, 0.2 elsewhere; student: 0.25, 0.25, 0.5.
    teacher = torch.tensor([[math.log(0.5), math.log(0.3)]])
    student = torch.tensor([[math.log(0.25), math.log(0.25)]])

    kl = topk_kl(teacher, student)

    expected = 0.5 * math.log(2) + 0.3 * math.log(1.2) + 0.2 * math.log(0.4)
    assert kl.item() == pytest.approx(expected, rel=1e-5)


def test_topk_kl_is_the_cross_entropy_when_the_teacher_is_one_hot() -> None:
    torch = pytest.importorskip("torch")
    from fc_distill.kd import topk_kl

    teacher = torch.tensor([[0.0, -60.0]])  # all of its mass on the first token
    student = torch.tensor([[math.log(0.2), math.log(0.7)]])

    assert topk_kl(teacher, student).item() == pytest.approx(-math.log(0.2), rel=1e-5)


def test_topk_kl_is_zero_when_the_student_matches_the_teacher() -> None:
    torch = pytest.importorskip("torch")
    from fc_distill.kd import topk_kl

    both = torch.tensor([[math.log(0.6), math.log(0.1)]])

    assert topk_kl(both, both.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_topk_kl_stays_finite_when_the_stored_top_k_sums_above_one() -> None:
    torch = pytest.importorskip("torch")
    from fc_distill.kd import topk_kl

    # float32 rounding: the stored entries sum to 1.0000004, so 1 - Σ is negative.
    teacher = torch.tensor([[math.log(0.6), math.log(0.4000004)]])
    student = torch.tensor([[math.log(0.5), math.log(0.5)]], requires_grad=True)

    kl = topk_kl(teacher, student)
    kl.sum().backward()

    assert torch.isfinite(kl).all()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
