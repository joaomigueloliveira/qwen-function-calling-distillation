# Training cost on the RTX 3060

Issue #36. What a LoRA SFT run of the 0.8B student costs on one RTX 3060 12 GB, and the trainer
settings that came out of it. Method A (SFT on gold labels), `fc-sft`, measured 2026-11-07 on
TRL 1.17.1, PEFT 0.21.2, transformers 5.20.2, torch 2.14.0 and flash-linear-attention 0.6.0.

## Headline

With the chosen setting, one epoch costs **0.072 GPU-hours per 1,000 train rows**, so the full train
split (54,561 rows) takes **about 3.9 GPU-hours**. Throughput is **2,188 tokens/s** and peak
memory **3.9 GiB**, or 4.6 GiB for the worst batch the train split can produce.

```bash
uv sync --extra inference --extra train
uv run --no-sync fc-sft --batch-size 4 --grad-accum 4 --group-by-length --save-steps 500 --wandb
```

| Setting | Value |
|---|---|
| LoRA | r=16, α=32, dropout 0, `all-linear` (10,822,656 trainable, 1.42%) |
| Precision | bf16 weights, fused fp32 LM head (TRL) |
| Optimiser | AdamW, lr 2e-4, cosine, 3% warmup, one epoch |
| Batch | 4 rows per micro-batch × 4 accumulation = 16 rows per step |
| Memory | gradient checkpointing on |
| Data order | `--group-by-length` (HF `LengthGroupedSampler`) |
| Length | `--max-length 4096`: the longest train row is 3,550 tokens, so nothing is truncated |
| Loss | answer tokens only (`completion_only_loss`), thinking off (ADR 0003) |

The LoRA rank, α and learning rate are defaults picked for the cost run, not tuned. ADR 0006 made
them every method's shared setup (#37).

## How it is measured

- **GPU-hours**: optimizer-step time from `GpuClock`. Logging, checkpoint saves, model load and
  tokenisation are excluded. HF's `train_runtime` agrees to within a second on a 380 s run.
- **Tokens/s**: non-padding tokens, prompt included, over that step time.
- **Peak**: `torch.cuda.max_memory_reserved`. The CUDA context adds roughly 0.3–0.5 GB on the card,
  which has 11.6 GiB usable.
- Every point in the sweeps trains the same 512 random train rows (seed 0) for one epoch.

## Findings

### 1. Memory goes to the output head, and TRL's fused head handles it

The vocabulary is 248,320 tokens, so fp32 logits cost about 0.95 MiB per token, and the backward pass
needs as much again. With a plain PyTorch loss, a batch of 4 × 600 tokens runs out of memory trying
to allocate the 2.2 GiB logits tensor. TRL 1.17 patches in a fused LM head that scores only the
labelled positions, in tiles. With it, the same batch peaks at about 3 GiB.

### 2. Gated DeltaNet needs its kernels

Qwen3.5 stacks 18 Gated DeltaNet layers with 6 attention layers. Without `flash-linear-attention`
(the `train` extra), transformers runs the DeltaNet layers on a slow reference PyTorch kernel. The
short convolution still falls back to PyTorch because `causal_conv1d` isn't installed. That's unused
speed-up headroom; it would likely need nvcc to build.

### 3. Padding was the main waste

| Micro-batch × accumulation | Checkpointing | Length grouping | Tokens/s | GPU-h per 1k rows | Peak |
|---|---|---|---:|---:|---:|
| 4 × 4 | on | no | 1,395 | 0.114 | 4.3 GiB |
| 8 × 2 | on | no | 1,171 | 0.135 | 6.8 GiB |
| 16 × 1 | on | no | OOM | | |
| **4 × 4** | **on** | **yes** | **2,188** | **0.072** | **3.9 GiB** |
| 8 × 2 | on | yes | 2,158 | 0.073 | 6.3 GiB |
| 16 × 1 | on | yes | 1,870 | 0.085 | 10.3 GiB |
| 1 × 16 | on | n/a | 2,056 | 0.077 | 2.3 GiB |

Rows run from a few hundred tokens to 3,550 (mean 581). In a random batch every row is padded to the
longest one. Without grouping, bigger batches were *slower* per real token. Grouping rows of similar
length made batch 4 1.6× faster, and after that bigger batches stopped helping. Batch 1 has no padding
at all, but one row per kernel launch leaves the GPU underused.

### 4. Gradient checkpointing stays on

| Micro-batch × accumulation | Checkpointing | Tokens/s | Peak on the 512 rows | Four longest train rows |
|---|---|---:|---:|---|
| 1 × 16 | off | 2,933 | 9.35 GiB | **OOM** |
| 2 × 8 | off | OOM | | |
| 4 × 4 | off | OOM | | |
| 1 × 16 | on | 2,056 | 2.3 GiB | 2.5 GiB |
| 4 × 4, grouped | on | 2,188 | 3.9 GiB | 4.6 GiB |

Without checkpointing the activations dominate. The fastest setting, 1 × 16 without checkpointing,
fits the 512-row sample, whose longest row is 2,582 tokens, but not the train split's longest rows
(3,550). An overnight run would die on them. The stress test trained the four longest train rows
(3,550, 3,550, 3,043 and 2,975 tokens) in one step. With length grouping, those four land in one
batch, and the sampler puts that batch first. So a run either fails in its first minute or never.

### 5. Length grouping changes data order

`LengthGroupedSampler` shuffles the rows, then sorts by length within megabatches of 200 rows. Each
optimizer step therefore sees rows of similar length rather than a fully random mix. This is the
usual trade. The full run's loss fell smoothly over its first 50 steps; its whole curve is the
check. A method that needs fully
random order can drop the flag at a cost of 1.6× in GPU-hours.

## Runs

- Official 1,000-row cost run, without grouping: 0.105 GPU-h per 1k rows, 1,502 tokens/s, 4.36 GiB
  ([W&B](https://forge.coreweave.com/wandb/joaomigueloliveira30/qwen-function-calling-distillation/runs/1680g270)).
- First full method A epoch, with the chosen setting
  ([W&B](https://forge.coreweave.com/wandb/joaomigueloliveira30/qwen-function-calling-distillation/runs/hmjki6iw)):
  2,235 tokens/s and a 4.62 GiB peak at step 50. The epoch cost 3.84 GPU-hours, against the 3.9
  forecast above; its results are in [`methods.md`](methods.md).
