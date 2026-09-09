# dft-thangnt

Direct Fine-Tuning (DFT) — preference alignment for causal LLMs using
[Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl).

Three objectives are supported:

| Method  | Trainer                        | Reference model | Notes |
| ------- | ------------------------------ | --------------- | ----- |
| `simpo` | `CPOTrainer(loss_type="simpo")` | not needed      | Length-normalized, target reward margin `gamma`. Default. |
| `orpo`  | `ORPOTrainer`                  | not needed      | Combined SFT + odds-ratio loss, good when starting from a base model. |
| `dpo`   | `DPOTrainer`                   | needed          | Classic DPO; uses the PEFT adapter-disabled path as its reference. |

Training runs QLoRA by default (4-bit base + LoRA adapters), so a 4B model
aligns comfortably in well under 24 GB of VRAM.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and an
NVIDIA GPU with a driver new enough for CUDA 12.4 wheels.

```bash
git clone https://github.com/upwitu/dft-thangnt.git
cd dft-thangnt
uv sync
```

`uv sync` creates `.venv/` from the exact pins in `uv.lock`, pulling
`torch`/`xformers`/`triton` from PyTorch's cu124 index and everything else from
PyPI. No conda environment and no manual CUDA setup needed.

To verify the install end to end:

```bash
uv run python train_dft.py --model-path Qwen/Qwen2.5-1.5B-Instruct --smoke-test
```

That runs 3 optimizer steps on the bundled sample data and writes a LoRA adapter
to `outputs/`.

## Train on your own data

```bash
uv run python train_dft.py \
    --model-path /path/to/your/base-model \
    --dataset /path/to/your/preference_pairs.jsonl \
    --output-dir outputs/my-run \
    --method simpo \
    --num-train-epochs 2
```

Or through the wrapper, which tees a numbered log into `logs/`:

```bash
./train_dft.sh --model-path Qwen/Qwen2.5-3B-Instruct --dataset data/mine.jsonl
```

### Dataset format

A JSONL file, one preference pair per line. Every row needs `prompt`, `chosen`
and `rejected`; any other key (such as `category` below) is ignored and is free
for your own bookkeeping.

```json
{
  "category": "refuse_missing_capability",
  "prompt": [
    {"role": "system", "content": "# Assistant Policy\n- set_air_conditioning is NOT available."},
    {"role": "user", "content": "turn on the air conditioning"}
  ],
  "chosen":   [{"role": "assistant", "content": "I'm sorry, air conditioning control isn't available here."}],
  "rejected": [{"role": "assistant", "content": "Got it! Turning it on right away."}]
}
```

**`prompt`** is either a list of chat messages — rendered with the model's own
chat template, so multi-turn context and prior `tool` results work — or a plain
pre-rendered string, used verbatim.

**`chosen` / `rejected`** are each a single assistant turn, given as a one-element
message list, a bare message object, or a plain string. A message may carry
`content`, `tool_calls`, or both; tool calls are rendered into the
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` form that Qwen-style
models emit:

```json
{
  "prompt": [{"role": "user", "content": "set it to 21 degrees up front"}],
  "chosen":   [{"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "set_temperature", "arguments": {"zone": "front", "celsius": 21}}}]}],
  "rejected": [{"role": "assistant", "content": "Sure, I've set it to 21."}]
}
```

The shortest possible form is plain strings:

```json
{"prompt": [{"role": "user", "content": "Which objective is reference-free?"}],
 "chosen": "Both SimPO and ORPO are.", "rejected": "All three are."}
```

See [`data/sample_preference_dataset.jsonl`](data/sample_preference_dataset.jsonl)
for 12 rows covering all of these shapes.

## Options

```
--method {simpo,orpo,dpo}   Alignment objective (default: simpo)
--model-path PATH           HF hub id or local directory (required)
--dataset PATH              Preference JSONL (default: bundled sample)
--output-dir DIR            Adapter + merged model destination (default: outputs)

--lora-r INT                LoRA rank (default: 32)
--lora-alpha INT            LoRA alpha (default: 64)
--lora-dropout FLOAT        LoRA dropout (default: 0.0)

--learning-rate FLOAT       Default 5e-6 (SimPO/DPO) or 8e-6 (ORPO)
--num-train-epochs FLOAT    Train for N epochs (default: 2)
--max-steps INT             Train for exactly N optimizer steps
--batch-size INT            Per-device batch (default: 2, or 1 for DPO)
--grad-accum INT            Gradient accumulation (default: 4, or 8 for DPO)
--warmup-ratio FLOAT        LR warmup ratio (default: 0.1)
--beta FLOAT                Preference loss beta (default: 0.1)
--simpo-gamma FLOAT         SimPO reward margin (default: 0.5)

--max-seq-length INT        Total sequence cap (default: 4096)
--max-prompt-length INT     Prompt cap (default: 2048)

--no-load-in-4bit           Load base in bf16/fp16 instead of 4-bit QLoRA
--eos-token STR             Override the turn-terminating token
--seed INT                  Random seed (default: 3407)
--no-merge                  Save only the adapter, skip the merged 16-bit model
--smoke-test                3 steps on up to 32 examples, to check the pipeline
```

### Choosing the EOS token

ChatML-derived models (Qwen, Yi, …) end an assistant turn with `<|im_end|>`
rather than the tokenizer's nominal `eos_token`. The script detects this by
looking for `<|im_end|>` in the model's chat template and falls back to
`tokenizer.eos_token` otherwise. Pass `--eos-token` if your model uses something
else — getting this wrong is the usual cause of a model that never stops
generating.

## Outputs

```
outputs/
├── dft_lora_adapter/    LoRA adapter + tokenizer (small, resumable)
└── dft_merged_model/    Base + adapter merged to 16-bit, ready for vLLM
```

Pass `--no-merge` to skip the merged copy when you are short on disk.

## Notes

- The script forces single-GPU training: it clears any inherited `WORLD_SIZE`,
  `RANK` and related DDP variables, then pins `cuda:0`. Select a physical GPU
  with `CUDA_VISIBLE_DEVICES`.
- A few compatibility shims at the top disable the `torchao` integration and
  stub the sub-byte `torch.intN` dtypes, which some torch builds lack. They must
  run before `unsloth` is imported, which is why the imports are ordered the way
  they are.
- `save_strategy="no"` — only the final adapter is written. Set it in
  `train_dft.py` if you want intermediate checkpoints.

## License

Not yet chosen. Until a `LICENSE` file is added, the default applies: all rights
reserved.
