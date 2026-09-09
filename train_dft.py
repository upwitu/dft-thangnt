#!/usr/bin/env python3
"""
DFT (Direct Fine-Tuning) / preference alignment for causal LLMs.

Supports three reference-free or reference-based alignment objectives:
  1. SimPO -- CPOTrainer(loss_type="simpo"), reference-free, length-normalized.
  2. ORPO  -- ORPOTrainer, reference-free, combined SFT + odds-ratio loss.
  3. DPO   -- DPOTrainer, standard direct preference optimization.

Uses Unsloth FastLanguageModel for ~2x faster training and low VRAM usage.

Bring your own data: pass any JSONL file of preference pairs via --dataset.
See README.md for the expected schema.
"""

import os

os.environ.setdefault("HF_TRUST_REMOTE_CODE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import argparse
import logging

import torch
import torch.utils

logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("datasets.fingerprint").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Compatibility shims.
#
# TorchAO ships dtypes (torch.int1..int7 / uint1..uint7) that older torch
# builds lack, and transformers probes for torchao at import time. We stub the
# dtypes and hard-disable the torchao integration so Unsloth's 4-bit path does
# not trip over a partially available install.
# ---------------------------------------------------------------------------
for _i in range(1, 8):
    if not hasattr(torch, f"int{_i}"):
        setattr(torch, f"int{_i}", torch.int8)
    if not hasattr(torch, f"uint{_i}"):
        setattr(torch, f"uint{_i}", torch.uint8)

if not hasattr(torch.utils, "_pytree"):
    import torch.utils._pytree  # noqa: F401
if not hasattr(torch.utils._pytree, "register_constant"):
    torch.utils._pytree.register_constant = lambda cls: cls

# Setting a sys.modules entry to None makes `import torchao` raise ImportError,
# which is exactly what the probes below expect when torchao is unavailable.
sys.modules["torchao"] = None
sys.modules["torchao.quantization"] = None
sys.modules["torchao.prototype"] = None

# Unsloth must be imported before transformers/trl so its patches land first.
from unsloth import FastLanguageModel  # noqa: E402

import transformers  # noqa: E402

transformers.utils.import_utils.is_torchao_available = lambda *a, **k: False
transformers.utils.is_torchao_available = lambda *a, **k: False

try:
    from transformers.models.auto.processing_auto import AutoProcessor

    transformers.AutoProcessor = AutoProcessor
except Exception:
    pass

from datasets import Dataset  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "sample_preference_dataset.jsonl")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DFT / preference alignment (SimPO, ORPO, DPO) with Unsloth + TRL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", default="simpo", choices=["simpo", "orpo", "dpo"],
                   help="Alignment objective.")
    p.add_argument("--model-path", required=True,
                   help="Base model to align: a HF hub id or a local directory.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="JSONL file of preference pairs (see README for schema).")
    p.add_argument("--output-dir", default="outputs",
                   help="Where the LoRA adapter and merged model are written.")

    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha.")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")

    p.add_argument("--learning-rate", type=float, default=None,
                   help="Learning rate. Default is method-specific (SimPO/DPO 5e-6, ORPO 8e-6).")
    p.add_argument("--num-train-epochs", type=float, default=None,
                   help="Train for this many epochs (mutually exclusive with --max-steps).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Train for exactly this many optimizer steps.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Per-device train batch size. Default 2 (SimPO/ORPO) or 1 (DPO).")
    p.add_argument("--grad-accum", type=int, default=None,
                   help="Gradient accumulation steps. Default 4 (SimPO/ORPO) or 8 (DPO).")
    p.add_argument("--warmup-ratio", type=float, default=0.1, help="LR warmup ratio.")
    p.add_argument("--beta", type=float, default=0.1, help="Preference loss beta.")
    p.add_argument("--simpo-gamma", type=float, default=0.5,
                   help="SimPO target reward margin (SimPO only).")

    p.add_argument("--max-seq-length", type=int, default=4096, help="Max total sequence length.")
    p.add_argument("--max-prompt-length", type=int, default=2048, help="Max prompt length.")

    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True,
                   help="Load the base model in 4-bit (QLoRA). Default on.")
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false",
                   help="Load the base model in bf16/fp16 instead of 4-bit.")
    p.add_argument("--eos-token", default=None,
                   help="Override the EOS token used to terminate completions. "
                        "Auto-detected from the chat template when omitted.")
    p.add_argument("--seed", type=int, default=3407, help="Random seed.")
    p.add_argument("--logging-steps", type=int, default=5, help="Log every N steps.")

    p.add_argument("--merge", dest="merge", action="store_true", default=True,
                   help="Also save a merged 16-bit model. Default on.")
    p.add_argument("--no-merge", dest="merge", action="store_false",
                   help="Save only the LoRA adapter, skip the merged model.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 3 optimizer steps on a handful of examples to verify the pipeline.")
    return p


def load_preference_records(path: str) -> list:
    """Read a JSONL preference file and validate its schema."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Pass --dataset /path/to/your.jsonl, or see README.md for the expected schema."
        )

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            missing = [k for k in ("prompt", "chosen", "rejected") if k not in rec]
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: missing required key(s) {missing}. "
                    "Every row needs 'prompt', 'chosen' and 'rejected'."
                )
            records.append(rec)

    if not records:
        raise ValueError(f"{path} contains no usable rows.")
    return records


def resolve_eos_token(tokenizer, override: str | None) -> str:
    """Pick the token that terminates an assistant turn.

    Chat-template models (Qwen, Yi, and other ChatML derivatives) end each turn
    with <|im_end|> rather than the tokenizer's nominal eos_token, so prefer
    that when the template actually uses it.
    """
    if override:
        return override

    template = getattr(tokenizer, "chat_template", None) or ""
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in template and "<|im_end|>" in vocab:
        return "<|im_end|>"

    eos = getattr(tokenizer, "eos_token", None)
    if not eos:
        raise ValueError(
            "Could not determine an EOS token for this model. Pass --eos-token explicitly."
        )
    return eos


def format_assistant_turn(msg, eos_token: str) -> str:
    """Render one assistant message (content and/or tool calls) as target text."""
    if not msg:
        return eos_token

    # Allow a bare string completion as well as a message dict.
    if isinstance(msg, str):
        return msg.strip() + eos_token

    text = msg.get("content") or ""
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        fn_args = fn.get("arguments", "{}")
        args_str = json.dumps(fn_args) if isinstance(fn_args, dict) else str(fn_args)
        name_json = json.dumps(fn.get("name", ""))
        text += f'\n<tool_call>\n{{"name": {name_json}, "arguments": {args_str}}}\n</tool_call>'

    return text.strip() + eos_token


def first_turn(value):
    """Preference completions may be a message list or a single message/string."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


def main() -> None:
    args = build_parser().parse_args()

    # Unsloth occasionally prompts on stdin; auto-accept so unattended runs work.
    if not sys.stdin.isatty():
        import builtins
        builtins.input = lambda *a, **k: "y"

    # Force single-GPU: stale DDP vars from a launcher would otherwise confuse HF.
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = "cuda:0"
    else:
        device = "cpu"

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(
        f"device={device} method={args.method.upper()} "
        f"lora_r={args.lora_r} lora_alpha={args.lora_alpha} "
        f"4bit={args.load_in_4bit} bf16={use_bf16}"
    )

    adapter_path = os.path.join(args.output_dir, "dft_lora_adapter")
    merged_path = os.path.join(args.output_dir, "dft_merged_model")
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the dataset before loading the model: a schema error should surface
    # in seconds, not after a multi-GB checkpoint load.
    records = load_preference_records(args.dataset)
    print(f"Loaded {len(records)} preference pairs from {args.dataset}")

    print(f"Loading base model from {args.model_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    eos_token = resolve_eos_token(tokenizer, args.eos_token)
    print(f"Using EOS token: {eos_token!r}")
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    tokenizer.eos_token = eos_token
    text_tokenizer.eos_token = eos_token

    formatted = []
    for rec in records:
        prompt = rec["prompt"]
        prompt_text = (
            prompt if isinstance(prompt, str)
            else tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        )
        formatted.append({
            "prompt": prompt_text,
            "chosen": format_assistant_turn(first_turn(rec["chosen"]), eos_token),
            "rejected": format_assistant_turn(first_turn(rec["rejected"]), eos_token),
        })

    dataset = Dataset.from_list(formatted)

    # Resolve the training schedule.
    if args.smoke_test:
        max_steps, num_train_epochs = 3, 1
        dataset = dataset.select(range(min(32, len(dataset))))
    elif args.max_steps is not None:
        max_steps, num_train_epochs = args.max_steps, 1
    elif args.num_train_epochs is not None:
        max_steps, num_train_epochs = -1, args.num_train_epochs
    else:
        max_steps, num_train_epochs = -1, 2

    # Per-method defaults: (batch_size, grad_accum, learning_rate).
    # DPO holds two model copies in memory, so it gets a smaller batch and a
    # correspondingly larger accumulation to keep the effective batch at 8.
    method_defaults = {
        "simpo": (2, 4, 5e-6),
        "orpo": (2, 4, 8e-6),
        "dpo": (1, 8, 5e-6),
    }
    default_batch, default_accum, default_lr = method_defaults[args.method]
    batch_size = args.batch_size or default_batch
    grad_accum = args.grad_accum or default_accum
    lr = args.learning_rate or default_lr

    common = dict(
        output_dir=adapter_path,
        beta=args.beta,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_seq_length,
    )

    if args.method == "simpo":
        from trl import CPOTrainer, CPOConfig
        trainer = CPOTrainer(
            model=model,
            args=CPOConfig(loss_type="simpo", simpo_gamma=args.simpo_gamma, **common),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    elif args.method == "orpo":
        from trl import ORPOTrainer, ORPOConfig
        trainer = ORPOTrainer(
            model=model,
            args=ORPOConfig(**common),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        from trl import DPOTrainer, DPOConfig
        trainer = DPOTrainer(
            model=model,
            args=DPOConfig(**common),
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    # When max_steps is set it overrides the epoch count, so report whichever
    # one actually governs the run.
    schedule = f"max_steps={max_steps}" if max_steps > 0 else f"epochs={num_train_epochs}"
    print(
        f"Starting {args.method.upper()} training: {schedule} lr={lr} "
        f"batch={batch_size}x{grad_accum} examples={len(dataset)}"
    )
    trainer.train()

    print(f"Saving LoRA adapter to {adapter_path} ...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    if args.merge:
        print(f"Merging LoRA weights into a 16-bit model at {merged_path} ...")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print("DFT alignment training completed.")


if __name__ == "__main__":
    main()
