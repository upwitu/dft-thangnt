# dft-thangnt

Dynamic Fine-Tuning (DFT) — a standalone, runnable implementation of
[arXiv:2508.05629](https://arxiv.org/abs/2508.05629), *"On the Generalization of
SFT: A Reinforcement Learning Perspective with Reward Rectification"*
(Wu et al., ICLR 2026), built on [Unsloth](https://github.com/unslothai/unsloth)
and [TRL](https://github.com/huggingface/trl).

## What DFT is

Read as a policy gradient, standard SFT implicitly assigns each token a reward
of `1 / p_theta(token)`. Rare tokens therefore dominate the update, which the
paper identifies as a cause of SFT's weak generalization relative to RL.

DFT rectifies this by rescaling each token's cross-entropy by the model's own
**detached** probability of that token, cancelling the inverse-probability
weighting and leaving a uniform reward of 1 across the expert trajectory —
equation (9) of the paper:

```
L_DFT = E -sum_t  sg(p_theta(y*_t | y*_<t, x)) * log p_theta(y*_t | y*_<t, x)
```

In code that is one line on top of the usual per-token cross-entropy
([dft_loss.py](dft_loss.py)):

```python
token_loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
with torch.no_grad():
    coefficients = F.softmax(logits, dim=-1).gather(-1, labels).squeeze(-1)
loss = masked_token_mean(token_loss * coefficients, labels)
```

DFT trains on **positive demonstrations only** — no preference pairs, no reward
model, no teacher. It is a drop-in replacement for SFT, not a preference-alignment
method.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and an
NVIDIA GPU with a driver new enough for CUDA 12.4 wheels.

```bash
git clone https://github.com/upwitu/dft-thangnt.git
cd dft-thangnt
uv sync
```

Verify the install end to end:

```bash
uv run python train_dft.py --model-path Qwen/Qwen2.5-1.5B-Instruct --smoke-test
```

That runs 3 optimizer steps on the bundled sample data and writes a LoRA adapter
to `outputs/`.

## Train on your own data

```bash
uv run python train_dft.py \
    --model-path /path/to/base-model \
    --dataset /path/to/your_demonstrations.jsonl \
    --output-dir outputs/my-run \
    --num-train-epochs 2
```

To train the SFT baseline the paper compares against, with everything else held
fixed:

```bash
uv run python train_dft.py --model-path ... --dataset ... --method sft
```

### Dataset format

A JSONL file of positive demonstrations, one per line, in either of two shapes.
Every row in a file must use the same shape — a mixed file is rejected, because
prompt masking would otherwise apply to only part of the data.

**Prompt + completion** — loss covers the completion only:

```json
{
  "prompt": [{"role": "user", "content": "What does DFT stand for?"}],
  "completion": [{"role": "assistant", "content": "Dynamic Fine-Tuning."}]
}
```

**Messages** — a full conversation; loss covers every assistant turn:

```json
{
  "messages": [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "What does DFT stand for?"},
    {"role": "assistant", "content": "Dynamic Fine-Tuning."}
  ]
}
```

Both are applied through the model's own chat template. See
[`data/sample_sft_dataset.jsonl`](data/sample_sft_dataset.jsonl) for a working
example.

## Verifying a run

**The DFT training loss can never exceed 1/e ≈ 0.3679.** Each token contributes
`-p*log(p)`, which peaks at `p = 1/e`. The script checks this after training and
warns if the bound is broken.

A loss above 1/e does not mean the model is bad — it means the loss reduction or
the gradient-accumulation scaling is wrong. The usual cause is
`model_accepts_loss_kwargs`: Transformers skips its own
`loss / gradient_accumulation_steps` whenever it believes the loss was instead
normalized by the global token count, which it infers from that flag — true for
any model whose `forward` takes `**kwargs`, i.e. every causal LM. Accelerate
cannot compensate, because `Trainer` builds it with `num_steps=1`. Left
unhandled, accumulated micro-batch losses are summed and never averaged, so both
the gradient and the logged loss come out `gradient_accumulation_steps` times too
large. [train_dft.py](train_dft.py) sets the flag `False` on the trainer.

Evaluation reports the **plain unweighted LM loss**, not the DFT objective, so
`eval_loss` stays directly comparable between a `--method dft` run and a
`--method sft` one. Pass `--eval-dataset` to enable it.

Run the loss unit tests, which check `dft_loss` against an independent loop-based
evaluation of equation (9), the 1/e bound, the stop-gradient, and the shift
alignment:

```bash
uv sync --group dev
uv run pytest tests/ -q
```

## Options

```
--method {dft,sft}          dft applies the reweighted objective (default);
                            sft is the baseline, everything else held fixed
--model-path PATH           HF hub id or local directory (required)
--dataset PATH              Demonstration JSONL (default: bundled sample)
--eval-dataset PATH         Optional held-out JSONL; eval reports plain LM loss
--output-dir DIR            Adapter destination (default: outputs)

--lora-r INT                LoRA rank (default: 32)
--lora-alpha INT            LoRA alpha (default: 64)
--lora-dropout FLOAT        LoRA dropout (default: 0.0)

--learning-rate FLOAT       Learning rate (default: 1e-4)
--num-train-epochs FLOAT    Epochs (default: 2)
--max-steps INT             Train for exactly N optimizer steps instead
--batch-size INT            Per-device batch (default: 2)
--grad-accum INT            Gradient accumulation (default: 8)
--warmup-ratio FLOAT        LR warmup ratio (default: 0.03)
--weight-decay FLOAT        Weight decay (default: 0.01)
--max-grad-norm FLOAT       Gradient clipping (default: 1.0)
--max-length INT            Max sequence length (default: 2048)

--no-load-in-4bit           Load base in bf16/fp16 instead of 4-bit QLoRA
--seed INT                  Random seed (default: 42)
--merge                     Also save a merged 16-bit model
--smoke-test                3 steps on up to 32 rows, to check the pipeline
```

## Memory note

During a DFT training step the script withholds `labels` from the forward pass so
the model does not also compute its own LM loss. That would upcast the
`[batch, sequence, vocabulary]` logits to fp32 a second time; on a 248k-token
vocabulary at length 8192 the duplicate is ~7.6 GB, enough on its own to push a 4B
model off a 40 GB card. Evaluation passes `labels` through normally, since it
wants exactly that plain LM loss.

## Provenance

The loss in [dft_loss.py](dft_loss.py) is ported from the reference
implementation in `kd-baselines` (`src/kd_baselines/losses.py`), where DFT runs as
a teacher-free distillation method. This repository extracts it into a standalone
trainer that runs on any base model and any demonstration data.

Paper authors' own code: https://github.com/yongliang-wu/DFT

## License

Not yet chosen. Until a `LICENSE` file is added, the default applies: all rights
reserved.
