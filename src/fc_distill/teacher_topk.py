"""Method C's teacher pass (ADR 0010): the 2B's top-k log-probs at every answer position of B's
training rows, scored by vLLM on the very token ids fc-sft trains on.

uv run --no-sync fc-teacher-topk --teacher-run runs/eval-train-Qwen_Qwen3.5-2B
uv run --no-sync fc-teacher-topk --teacher-run runs/eval-train-Qwen_Qwen3.5-2B --n-rows 200  # smoke

The rows are method B's (every teacher output that parses, rewritten in gold's format) and are
tokenized by TRL's own dataset preparation, as fc-sft's SFTTrainer does it, so the stored
positions are the positions the student's loss covers. vLLM's `prompt_logprobs` scores each
position of the given ids; the answer positions (those with a label) keep their top k and the
written token's log-prob and rank. Before any GPU work, every row's labelled tokens are decoded
and checked against the answer text it trains on. Writes topk.npz and summary.json to
runs/<name>/. The file is derived from the gated xLAM prompts: back it up only to the private W&B
project.

Needs both extras (`uv sync --extra inference --extra train`): TRL tokenizes, vLLM scores.
"""

import argparse
import gc
import json
import os
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from fc_distill.data import load_split
from fc_distill.evaluate import GPU_MEM, MAX_MODEL_LEN
from fc_distill.sft import MAX_LENGTH, STUDENT, SftRecord, load_teacher_run, teacher_rows
from fc_distill.tracking import data_provenance, git_state, hub_revision

K = 20  # ADR 0010: vLLM's default max_logprobs
IGNORE = -100  # TRL's label on positions the loss skips
# What the chat template writes after an assistant answer; the loss covers it too.
END_OF_TURN = "<|im_end|>\n"
# The TRL whose private `_prepare_dataset` `training_tokens` calls (see there).
TRL_VERSION = "1.17."
# Tokens vLLM scores per step. Every prompt token needs logits over the whole 248,320-token
# vocabulary, which vLLM's memory profile leaves out: at its default 8,192 the full pass ran out
# of memory on the 3060. Longer rows are still scored whole, over several steps (chunked prefill).
SCORED_TOKENS_PER_STEP = 2048


def answer_matches(record: SftRecord, labelled_text: str) -> bool:
    """Whether the decoded labelled tokens are exactly the record's answer and the end of turn:
    a check on the tokenization that doesn't go through TRL."""
    return labelled_text == record["completion"][0]["content"] + END_OF_TURN


def answer_positions(labels: Sequence[int]) -> list[int]:
    """The positions the loss covers: those with a label (TRL masks the prompt with -100)."""
    return [i for i, label in enumerate(labels) if label != IGNORE]


class Position(NamedTuple):
    """The teacher's top k at one position, most likely first, and the written token's own
    log-prob and rank (1 = the teacher's top token)."""

    ids: list[int]
    logprobs: list[float]
    token_logprob: float
    token_rank: int


def top_k_entries(
    *, token_ids: Sequence[int], logprobs: Sequence[float], ranks: Sequence[int], token: int, k: int
) -> Position:
    """One position of vLLM's prompt log-probs: the k entries ranked 1 to k, and the written
    token's own log-prob and rank (vLLM adds it as an extra entry when it ranks below k)."""
    by_id = {tid: (rank, lp) for tid, lp, rank in zip(token_ids, logprobs, ranks, strict=True)}
    if token not in by_id:
        raise ValueError(f"written token {token} has no log-prob at this position")
    token_rank, token_logprob = by_id[token]
    top = sorted((rank, tid, lp) for tid, (rank, lp) in by_id.items() if rank <= k)[:k]
    return Position(
        ids=[tid for _, tid, _ in top],
        logprobs=[lp for _, _, lp in top],
        token_logprob=token_logprob,
        token_rank=token_rank,
    )


Ints = npt.NDArray[np.int32]
Floats = npt.NDArray[np.float32]


class RowTopk(NamedTuple):
    """One training row's answer positions: position p in the token ids, the written token
    input_ids[p], the teacher's top-k ids and log-probs ([n, k]) and the written token's own
    log-prob and rank.

    Entry p is the teacher's distribution over token p given tokens 0..p-1, as vLLM's prompt
    log-probs are. The student's matching distribution comes from its logits at p - 1 (a causal
    LM's output at i predicts token i + 1), which is TRL's shift too."""

    row_id: int
    positions: Ints
    tokens: Ints
    ids: Ints
    logprobs: Floats
    token_logprob: Floats
    token_rank: Ints


_FIELDS = ("positions", "tokens", "ids", "logprobs", "token_logprob", "token_rank")


def save_topk(path: Path, rows: Sequence[RowTopk]) -> None:
    """All rows in one .npz: each field concatenated over rows, with offsets into it."""
    lengths = [len(r.positions) for r in rows]
    np.savez(
        path,
        row_ids=np.array([r.row_id for r in rows], dtype=np.int64),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        **{f: np.concatenate([getattr(r, f) for r in rows]) for f in _FIELDS},
    )


def load_topk(path: Path) -> dict[int, RowTopk]:
    """`save_topk`'s rows, by row id."""
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    offsets = data["offsets"]
    return {
        int(row_id): RowTopk(int(row_id), *(data[f][offsets[i] : offsets[i + 1]] for f in _FIELDS))
        for i, row_id in enumerate(data["row_ids"])
    }


class TopkRun(NamedTuple):
    """A finished fc-teacher-topk run: its rows by id, and the GPU-hours of its scoring calls."""

    rows: dict[int, RowTopk]
    gpu_hours: float


def load_topk_run(run_dir: Path, teacher_run: Path) -> TopkRun:
    """The run in `run_dir`, which must have scored `teacher_run`'s outputs (the text B and C
    train on)."""
    summary = json.loads((run_dir / "summary.json").read_text())
    if Path(summary["teacher_run"]).resolve() != Path(teacher_run).resolve():
        raise ValueError(f"{run_dir} scored {summary['teacher_run']}, not {teacher_run}")
    return TopkRun(load_topk(run_dir / "topk.npz"), summary["gpu_hours"])


def coverage(rows: Sequence[RowTopk], *, k: int) -> dict[str, float]:
    """How much the stored top k says: the share of answer positions whose written token is the
    teacher's top-1, and the probability mass the top k hold (what the tail entry stands in for)."""
    mass = np.concatenate([np.exp(r.logprobs.astype(np.float64)).sum(axis=1) for r in rows])
    ranks = np.concatenate([r.token_rank for r in rows])
    return {
        "rows": len(rows),
        "positions": int(mass.size),
        "written_is_top1": float((ranks == 1).mean()),
        "written_in_top_k": float((ranks <= k).mean()),
        "mass_median": float(np.median(mass)),
        "mass_p01": float(np.quantile(mass, 0.01)),
        "mass_min": float(mass.min()),
    }


class Tokenized(NamedTuple):
    """One row as fc-sft trains on it: its input ids and the answer positions (those with a
    label). Numpy, because 52,424 rows as lists of Python ints filled the machine's 15 GB."""

    input_ids: Ints
    positions: Ints


def training_tokens(
    records: Sequence[tuple[int, SftRecord]], tokenizer: Any
) -> dict[int, Tokenized]:
    """Each row's input ids and answer positions exactly as fc-sft's SFTTrainer builds them:
    TRL's own dataset preparation (chat template, completion mask, truncation), keyed by row id."""
    import trl
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    if not trl.__version__.startswith(TRL_VERSION):
        raise RuntimeError(f"written against TRL {TRL_VERSION}x, found {trl.__version__}")

    dataset = Dataset.from_list([dict(r) | {"row_id": i} for i, r in records])
    # `_prepare_dataset` reads only these three attributes of the trainer (TRL 1.17.1): the
    # template override (None unless assistant_only_loss), the loss mask and the tokenizer.
    trainer = argparse.Namespace(
        chat_template=None, completion_only_loss=True, _tokenizer=tokenizer
    )
    config = SFTConfig(
        output_dir="unused", max_length=MAX_LENGTH, completion_only_loss=True, report_to="none"
    )
    prepared = SFTTrainer._prepare_dataset(
        trainer, dataset, tokenizer, config, False, None, "train"
    )
    return {
        r["row_id"]: Tokenized(
            np.array(r["input_ids"], dtype=np.int32),
            np.array(answer_positions(r["labels"]), dtype=np.int32),
        )
        for r in prepared.select_columns(["row_id", "input_ids", "labels"])
    }


def score_row(row_id: int, input_ids: Ints, positions: Ints, logprobs: Any, k: int) -> RowTopk:
    """One row's answer positions from vLLM's flat prompt log-probs (`FlatLogprobs`)."""
    picked = []
    for p in positions:
        a, b = logprobs.start_indices[p], logprobs.end_indices[p]
        picked.append(
            top_k_entries(
                token_ids=logprobs.token_ids[a:b],
                logprobs=logprobs.logprobs[a:b],
                ranks=logprobs.ranks[a:b],
                token=int(input_ids[p]),
                k=k,
            )
        )
    if any(len(p.ids) != k for p in picked):
        raise ValueError(f"row {row_id}: a position has fewer than {k} entries")
    return RowTopk(
        row_id=row_id,
        positions=positions,
        tokens=input_ids[positions],
        ids=np.array([p.ids for p in picked], dtype=np.int32).reshape(-1, k),
        logprobs=np.array([p.logprobs for p in picked], dtype=np.float32).reshape(-1, k),
        token_logprob=np.array([p.token_logprob for p in picked], dtype=np.float32),
        token_rank=np.array([p.token_rank for p in picked], dtype=np.int32),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--teacher-run", type=Path, required=True, help="the fc-eval train run B used")
    ap.add_argument("--n-rows", type=int, default=None, help="random rows, seed 0 (default: all)")
    ap.add_argument(
        "--chunk",
        type=int,
        default=512,
        help="rows per vLLM call; their prompt log-probs (k + 1 per position) sit in host memory",
    )
    ap.add_argument("--gpu-mem", type=float, default=GPU_MEM, help="vLLM gpu_memory_utilization")
    ap.add_argument("--out", type=Path, default=None, help="default: runs/<run name>")
    ap.add_argument(
        "--allow-dirty", action="store_true", help="run even with uncommitted changes (recorded)"
    )
    args = ap.parse_args()
    if not args.allow_dirty and git_state()["dirty"]:
        ap.error("uncommitted changes to tracked files; commit them or pass --allow-dirty")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")  # no nvcc here (see fc-eval)
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    teacher = load_teacher_run(args.teacher_run)
    rows = [ex for ex in load_split("train") if ex["id"] in teacher.outputs]
    if args.n_rows is not None:  # sampled before the parse, as fc-sft does
        rows = random.Random(0).sample(rows, args.n_rows)
    kept, counts = teacher_rows(rows, teacher.outputs, rule="none")
    records = dict(kept)
    # fc-sft's tokenizer: the student's, the same file as the teacher's (ADR 0010).
    tokenizer = AutoTokenizer.from_pretrained(STUDENT)
    tokens = training_tokens(kept, tokenizer)
    order = [row_id for row_id, _ in kept if row_id in tokens]
    answer_mismatches = sum(
        not answer_matches(records[i], tokenizer.decode(tokens[i].input_ids[tokens[i].positions]))
        for i in order
    )
    counts |= {
        "dropped_by_tokenization": len(kept) - len(order),
        "answer_mismatches": answer_mismatches,
    }
    print("rows:", json.dumps(counts))
    if answer_mismatches or counts["dropped_by_tokenization"]:
        raise RuntimeError("the tokenized rows aren't B's answers; nothing scored")
    del rows, kept, records  # host memory: vLLM's process needs it next
    gc.collect()

    name = f"topk-{K}-{Path(args.teacher_run).name}" + (
        "" if args.n_rows is None else f"-n{args.n_rows}"
    )
    out: Path = args.out or Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=teacher.model,
        revision=teacher.revision,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=args.gpu_mem,
        max_logprobs=K,
        # Every prompt position is scored; a cached prefix would skip some of them.
        enable_prefix_caching=False,
        max_num_batched_tokens=SCORED_TOKENS_PER_STEP,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    params = SamplingParams(
        max_tokens=1, temperature=0.0, prompt_logprobs=K, detokenize=False, flat_logprobs=True
    )

    results: list[RowTopk] = []
    # GPU time is the scoring calls alone, as fc-eval's `seconds` is the generation alone:
    # model load and tokenization fall outside, for B's teacher pass and for this one.
    vllm_mismatches, seconds = 0, 0.0
    for start in range(0, len(order), args.chunk):
        chunk = order[start : start + args.chunk]
        prompts = [TokensPrompt(prompt_token_ids=tokens[i].input_ids.tolist()) for i in chunk]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        seconds += time.perf_counter() - t0
        for row_id, o in zip(chunk, outputs, strict=True):
            input_ids, positions = tokens[row_id]
            if list(o.prompt_token_ids or []) != input_ids.tolist():
                vllm_mismatches += 1
                continue
            results.append(score_row(row_id, input_ids, positions, o.prompt_logprobs, K))
        print(f"{start + len(chunk)} / {len(order)} rows, {seconds / 3600:.2f} GPU-h")

    save_topk(out / "topk.npz", results)
    summary = {
        "teacher_run": str(args.teacher_run),
        "model": teacher.model,
        "k": K,
        "rows": counts | {"scored": len(results), "vllm_token_mismatches": vllm_mismatches},
        "coverage": coverage(results, k=K),
        "seconds": round(seconds, 1),
        "gpu_hours": seconds / 3600,
        "file_bytes": (out / "topk.npz").stat().st_size,
        "provenance": git_state()
        | {"model_revision": hub_revision(teacher.model, teacher.revision)}
        | data_provenance()
        | {
            "vllm": vllm.__version__,
            "scoring": "prompt_logprobs on fc-sft's token ids, raw log-probs (T = 1)",
            "tokenizer": STUDENT,
            "max_length": MAX_LENGTH,
            "max_num_batched_tokens": SCORED_TOKENS_PER_STEP,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if vllm_mismatches:
        raise RuntimeError(f"{vllm_mismatches} rows' token ids changed inside vLLM")
