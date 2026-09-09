#!/usr/bin/env python3
"""Dynamic Fine-Tuning (DFT) -- arXiv:2508.05629, ICLR 2026.

DFT is a one-line correction to SFT: each token's cross-entropy is rescaled by
the model's own detached probability of that token. It needs only positive
demonstrations -- no preference pairs, no reward model, no teacher.

Pass --method sft to train the plain SFT baseline the paper compares against,
with everything else held fixed.
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
# not trip over a partially available install. These must run before unsloth
# is imported.
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
from trl import SFTConfig, SFTTrainer  # noqa: E402

from dft_loss import IGNORE_INDEX, MAX_DFT_LOSS, dft_loss  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "sample_sft_dataset.jsonl")


class DFTTrainer(SFTTrainer):
    """SFTTrainer whose *training* objective is DFT instead of plain SFT.

    Evaluation deliberately keeps the standard unweighted LM loss, so eval_loss
    stays directly comparable between a --method dft run and a --method sft one.
    """

    def __init__(self, *args, use_dft: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_dft = use_dft

        # compute_loss below returns a loss already averaged over the tokens of
        # *this* micro-batch. Transformers skips its own
        # `loss / gradient_accumulation_steps` whenever it believes the loss was
        # instead normalized by the global token count, which it infers from
        # model_accepts_loss_kwargs -- true for any model whose forward takes
        # **kwargs, i.e. every causal LM used here. Accelerate cannot compensate
        # because Trainer builds it with num_steps=1, making its division a
        # no-op. Left True, the accumulated micro-batch losses are summed and
        # never averaged, so gradient and logged loss both come out
        # gradient_accumulation_steps times too large.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not (self.use_dft and model.training):
            outputs = model(**inputs, use_cache=False)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        labels = inputs["labels"]

        # Withhold labels so the model does not also compute its own LM loss:
        # that upcasts the [batch, sequence, vocabulary] logits to fp32 a second
        # time. On a 248k-token vocabulary at length 8192 the duplicate is
        # ~7.6GB, enough on its own to push a 4B model off a 40GB card.
        forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**forward_inputs, use_cache=False)

        loss = dft_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dynamic Fine-Tuning (arXiv:2508.05629) with Unsloth + TRL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method", default="dft", choices=["dft", "sft"],
                   help="dft applies the reweighted objective; sft is the baseline.")
    p.add_argument("--model-path", required=True,
                   help="Base model to fine-tune: a HF hub id or a local directory.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="JSONL of demonstrations (see README for the schema).")
    p.add_argument("--eval-dataset", default=None,
                   help="Optional held-out JSONL. Eval always reports plain LM loss.")
    p.add_argument("--output-dir", default="outputs",
                   help="Where the LoRA adapter and merged model are written.")

    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank.")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha.")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout.")

    p.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    p.add_argument("--num-train-epochs", type=float, default=2.0, help="Epochs to train.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Train for exactly this many optimizer steps instead.")
    p.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size.")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="LR warmup ratio.")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm.")
    p.add_argument("--max-length", type=int, default=2048, help="Max total sequence length.")

    p.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True,
                   help="Load the base model in 4-bit (QLoRA). Default on.")
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false",
                   help="Load the base model in bf16/fp16 instead of 4-bit.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--logging-steps", type=int, default=10, help="Log every N steps.")

    p.add_argument("--merge", dest="merge", action="store_true", default=False,
                   help="Also save a merged 16-bit model.")
    p.add_argument("--smoke-test", action="store_true",
                   help="3 optimizer steps on a handful of rows, to verify the pipeline.")
    return p


def load_jsonl(path: str) -> list:
    """Read a JSONL file of demonstrations and validate its shape."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Pass --dataset /path/to/your.jsonl, or see README.md for the schema."
        )

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e

            has_pc = "prompt" in row and "completion" in row
            has_messages = "messages" in row
            if not (has_pc or has_messages):
                raise ValueError(
                    f"{path}:{lineno}: each row needs either 'prompt' + 'completion', "
                    "or 'messages'. See README.md."
                )
            if has_pc and has_messages:
                raise ValueError(
                    f"{path}:{lineno}: row has both 'messages' and 'prompt'/'completion'. "
                    "Pick one format and use it for the whole file."
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"{path} contains no usable rows.")

    # TRL keys its loss masking off the columns present, so the file must be
    # consistent: a mixed file would silently train on prompts in some rows.
    formats = {("messages" in r) for r in rows}
    if len(formats) > 1:
        raise ValueError(
            f"{path} mixes the two row formats. Every row must use the same one, "
            "otherwise prompt masking would apply to only part of the data."
        )
    return rows


def tokenize_conversation(messages: list, tokenizer, max_length: int) -> dict | None:
    """Tokenize a messages row, training on assistant turns only.

    TRL's own ``assistant_only_loss`` is unreachable here: Unsloth patches
    SFTTrainer._prepare_dataset and it recognises only ``labels``, ``input_ids``,
    ``prompt``+``completion`` or a text field -- a ``messages`` column falls
    through to a "you must specify a formatting_func" error. So we emit
    ``input_ids``/``labels`` directly, which is the branch Unsloth does accept.

    Each turn is tokenized as the token-level delta between the template
    rendered up to and including it and the template rendered without it, so
    role headers and separators land in the right span without any
    template-specific string parsing. Returns None if the row has no assistant
    content left after truncation.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    previous: list[int] = []

    for index, message in enumerate(messages):
        upto = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True)
        if upto[:len(previous)] != previous:
            raise ValueError(
                "This model's chat template does not grow monotonically as turns "
                "are added, so assistant spans cannot be located reliably. "
                "Convert your data to the prompt+completion format instead."
            )
        segment = upto[len(previous):]
        input_ids.extend(segment)
        # The assistant's own tokens are supervised; role headers preceding a
        # non-assistant turn, and every prompt token, are not.
        labels.extend(segment if message.get("role") == "assistant"
                      else [IGNORE_INDEX] * len(segment))
        previous = upto

    input_ids, labels = input_ids[:max_length], labels[:max_length]
    if all(label == IGNORE_INDEX for label in labels):
        return None
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}


def main() -> None:
    args = build_parser().parse_args()

    if not sys.stdin.isatty():
        import builtins
        builtins.input = lambda *a, **k: "y"

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

    adapter_path = os.path.join(args.output_dir, f"{args.method}_lora_adapter")
    merged_path = os.path.join(args.output_dir, f"{args.method}_merged_model")
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate the data before the multi-GB checkpoint load, so a schema error
    # surfaces in seconds.
    rows = load_jsonl(args.dataset)
    conversational = "messages" in rows[0]
    print(f"Loaded {len(rows)} rows from {args.dataset} "
          f"({'messages' if conversational else 'prompt+completion'} format)")

    eval_rows = load_jsonl(args.eval_dataset) if args.eval_dataset else None
    if eval_rows is not None and ("messages" in eval_rows[0]) != conversational:
        raise ValueError("--eval-dataset uses a different row format than --dataset.")

    print(f"Loading base model from {args.model_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_length,
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

    def to_dataset(source: list, label: str) -> Dataset:
        if not conversational:
            return Dataset.from_list(source)
        tokenized, dropped = [], 0
        for row in source:
            encoded = tokenize_conversation(row["messages"], tokenizer, args.max_length)
            if encoded is None:
                dropped += 1
            else:
                tokenized.append(encoded)
        if not tokenized:
            raise ValueError(
                f"Every {label} row lost all of its assistant tokens. Either no row "
                "has an assistant turn, or --max-length truncates them all away."
            )
        if dropped:
            print(f"Dropped {dropped}/{len(source)} {label} rows with no assistant "
                  f"tokens left after truncation to --max-length {args.max_length}.")
        return Dataset.from_list(tokenized)

    train_dataset = to_dataset(rows, "train")
    eval_dataset = to_dataset(eval_rows, "eval") if eval_rows else None
    if args.smoke_test:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))

    max_steps = 3 if args.smoke_test else (args.max_steps if args.max_steps is not None else -1)

    config = SFTConfig(
        output_dir=adapter_path,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=max_steps,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_length,
        packing=False,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_strategy="no",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[],
        seed=args.seed,
        # Train on the response only. Messages rows arrive already tokenized
        # with their prompt spans masked, so the flag applies to the
        # prompt+completion path alone.
        **({} if conversational else {"completion_only_loss": True}),
    )

    trainer = DFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        use_dft=(args.method == "dft"),
    )

    schedule = f"max_steps={max_steps}" if max_steps > 0 else f"epochs={args.num_train_epochs}"
    print(
        f"Starting {args.method.upper()} training: {schedule} lr={args.learning_rate} "
        f"batch={args.batch_size}x{args.grad_accum} rows={len(train_dataset)}"
    )
    result = trainer.train()

    train_loss = result.training_loss
    if args.method == "dft" and train_loss > MAX_DFT_LOSS:
        print(
            f"\nWARNING: DFT train loss {train_loss:.4f} exceeds the -p*log(p) "
            f"bound of 1/e = {MAX_DFT_LOSS:.4f}, which the objective cannot "
            "exceed. The loss reduction or the gradient-accumulation scaling is "
            "wrong -- see README, 'Verifying a run'.",
            file=sys.stderr,
        )
    elif args.method == "dft":
        print(f"DFT train loss {train_loss:.4f} is within the 1/e = {MAX_DFT_LOSS:.4f} bound.")

    print(f"Saving LoRA adapter to {adapter_path} ...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    if args.merge:
        print(f"Merging LoRA weights into a 16-bit model at {merged_path} ...")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    print(f"{args.method.upper()} training completed.")


if __name__ == "__main__":
    main()
