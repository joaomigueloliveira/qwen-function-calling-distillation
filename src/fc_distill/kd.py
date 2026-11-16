"""Method C's loss (ADR 0010): forward KL from the teacher's stored top-k log-probs to the student,
at every answer position, T = 1, no hard term.

At each position the teacher's distribution is its k stored entries plus one rest entry, the mass
outside them; the student's is its log-probs of the same k tokens plus its own mass outside them.
The student's log-probs come from TRL's fused LM head (`score_ids`), which never builds the full
logits, so the run costs what B's does. fc-sft --targets teacher-topk wires it in.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from fc_distill.teacher_topk import IGNORE, RowTopk

if TYPE_CHECKING:
    import torch

# The student's rest mass is 1 - Σ of its k probabilities; float rounding can take that to 0 or
# below. The floor keeps its log finite. Below it the term gives no gradient, but float32 can't
# resolve 1 - Σ under about 6e-8 anyway, so a smaller rest is noise either way.
STUDENT_REST_FLOOR = 1e-8


def place_targets(
    row: RowTopk, labels: npt.NDArray[np.integer[Any]]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
    """One row's stored top-k, laid out over its token positions as TRL's labels are: ids (-1 where
    there is no target) and log-probs (0 there). Refuses a row whose labelled positions or tokens
    differ from the stored ones, so a target can't land on the wrong token."""
    positions = np.flatnonzero(labels != IGNORE)
    if not (
        np.array_equal(positions, row.positions) and np.array_equal(labels[positions], row.tokens)
    ):
        raise ValueError(f"row {row.row_id}: its labels aren't the tokens the teacher scored")
    k = row.ids.shape[1]
    ids = np.full((len(labels), k), -1, dtype=np.int64)
    logprobs = np.zeros((len(labels), k), dtype=np.float32)
    ids[positions] = row.ids
    logprobs[positions] = row.logprobs
    return ids, logprobs


def topk_kl(teacher: "torch.Tensor", student: "torch.Tensor") -> "torch.Tensor":
    """KL(teacher ‖ student) per position, over the k stored tokens plus one rest entry.

    `teacher` and `student` are [N, k] log-probs of the same tokens, each normalized over the
    whole vocabulary. The teacher's rest is clamped at 0: its stored top k can sum just above 1 in
    float32, and 0 · log 0 counts as 0."""
    import torch

    t, s = teacher.float(), student.float()
    p = t.exp()
    p_rest = (1 - p.sum(-1)).clamp(min=0)
    q_rest = (1 - s.exp().sum(-1)).clamp(min=STUDENT_REST_FLOOR)
    return (p * (t - s)).sum(-1) + torch.xlogy(p_rest, p_rest) - p_rest * q_rest.log()


class KdCollator:
    """TRL's collator, plus the batch's stored targets laid out like its labels (`score_ids`,
    `teacher_logprobs`). The row ids travel with the examples and are dropped from the batch."""

    def __init__(self, base: Any, targets: Mapping[int, RowTopk]) -> None:
        self._base, self._targets = base, targets

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch: dict[str, Any] = self._base(examples)
        per_row = [
            place_targets(self._targets[ex["row_id"]], labels.numpy())
            for ex, labels in zip(examples, batch["labels"], strict=True)
        ]
        batch["score_ids"] = torch.from_numpy(np.stack([ids for ids, _ in per_row]))
        batch["teacher_logprobs"] = torch.from_numpy(np.stack([lp for _, lp in per_row]))
        return batch


def kd_trainer(targets: Mapping[int, RowTopk], **kwargs: Any) -> Any:
    """An SFTTrainer whose loss is `topk_kl` at the answer positions, summed and divided by the
    tokens in the optimizer step as TRL divides B's cross-entropy. It also logs the student's
    cross-entropy on the written token (`nll`), the number B's loss is, for comparison."""
    from trl import SFTTrainer

    class KdTrainer(SFTTrainer):  # type: ignore[misc]
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            import torch

            teacher = inputs.pop("teacher_logprobs")
            outputs = model(**inputs, use_cache=False, fused_lm_head=True)
            mask = outputs.label_mask
            # The fused head drops position 0 and scores position p from the hidden state at
            # p - 1; the teacher's entry for token p is shifted the same way.
            kl = topk_kl(teacher[:, 1:][mask], outputs.score_log_probs[mask])
            tokens = mask.sum() if num_items_in_batch is None else num_items_in_batch
            loss = kl.sum() / torch.as_tensor(tokens, device=kl.device, dtype=kl.dtype).clamp(min=1)
            mode = "train" if self.model.training else "eval"
            n = mask.sum().clamp(min=1)
            metrics = self._metrics[mode]
            metrics["nll"].append((-outputs.log_probs[mask].sum() / n).item())
            metrics["kl"].append((kl.detach().sum() / n).item())
            metrics["entropy"].append((outputs.entropy[mask].sum() / n).item())
            metrics["mean_token_accuracy"].append((outputs.is_top1[mask].float().sum() / n).item())
            # TRL's running token count, as B's curves have it (one GPU: no gathering).
            if mode == "train":
                self._total_train_tokens += int(inputs["attention_mask"].sum().item())
            metrics["num_tokens"] = [self._total_train_tokens]
            return (loss, outputs) if return_outputs else loss

    trainer = KdTrainer(**kwargs)
    trainer.data_collator = KdCollator(trainer.data_collator, targets)
    return trainer
