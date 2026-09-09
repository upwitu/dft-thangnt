"""Check that messages rows are supervised on assistant spans and nothing else."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_loss import IGNORE_INDEX

# train_dft imports unsloth, which needs a GPU-capable install. The function
# under test does not, so skip cleanly where the import is unavailable.
train_dft = pytest.importorskip("train_dft", reason="requires the unsloth stack")
tokenize_conversation = train_dft.tokenize_conversation


class FakeTokenizer:
    """Minimal ChatML-like tokenizer: one token per word, plus role headers.

    With thinking=True it mimics Qwen3, whose template unconditionally inserts
    an empty <think> block in front of every assistant turn.
    """

    unk_token_id = 0

    def __init__(self, monotonic: bool = True, thinking: bool = False):
        self.monotonic = monotonic
        self.thinking = thinking
        self.vocab: dict[str, int] = {}
        if thinking:  # reserve ids so the markers exist in the vocabulary
            self._id("<think>"), self._id("</think>")

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab) + 1)

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        pieces = []
        for message in messages:
            pieces.append(f"<{message['role']}>")
            if self.thinking and message["role"] == "assistant":
                pieces += ["<think>", "\n\n", "</think>", "\n\n"]
            pieces.extend(message["content"].split())
            pieces.append("<end>")
        if not self.monotonic:
            # A template that rewrites earlier turns, e.g. by re-emitting a
            # system block that depends on later content.
            pieces = pieces[::-1]
        return [self._id(p) for p in pieces]

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def decode(self, ids):
        back = {v: k for k, v in self.vocab.items()}
        return "".join(back[i] for i in ids)

    def decode_pieces(self, ids):
        back = {v: k for k, v in self.vocab.items()}
        return [back[i] for i in ids]


CONVERSATION = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "what is dft"},
    {"role": "assistant", "content": "dynamic fine tuning"},
    {"role": "user", "content": "does it need pairs"},
    {"role": "assistant", "content": "no"},
]


def test_supervises_only_assistant_spans():
    tok = FakeTokenizer()
    out = tokenize_conversation(CONVERSATION, tok, max_length=1000)

    supervised = [i for i, label in zip(out["input_ids"], out["labels"])
                  if label != IGNORE_INDEX]
    assert tok.decode_pieces(supervised) == [
        "<assistant>", "dynamic", "fine", "tuning", "<end>",
        "<assistant>", "no", "<end>",
    ]


def test_labels_align_with_input_ids():
    tok = FakeTokenizer()
    out = tokenize_conversation(CONVERSATION, tok, max_length=1000)
    assert len(out["input_ids"]) == len(out["labels"]) == len(out["attention_mask"])
    # Where a label is supervised it must equal the input token at that index:
    # the objective predicts the next token, and the shift happens in the loss.
    for token, label in zip(out["input_ids"], out["labels"]):
        assert label in (IGNORE_INDEX, token)


def test_input_ids_reconstruct_the_full_render():
    tok = FakeTokenizer()
    out = tokenize_conversation(CONVERSATION, tok, max_length=1000)
    assert out["input_ids"] == tok.apply_chat_template(CONVERSATION)


def test_prompt_only_conversation_is_dropped():
    tok = FakeTokenizer()
    assert tokenize_conversation(
        [{"role": "user", "content": "hello"}], tok, max_length=1000
    ) is None


def test_truncation_that_removes_every_assistant_token_drops_the_row():
    tok = FakeTokenizer()
    # Keep only the first few tokens, which are all system/user.
    assert tokenize_conversation(CONVERSATION, tok, max_length=3) is None


def test_truncation_keeps_the_surviving_prefix():
    tok = FakeTokenizer()
    out = tokenize_conversation(CONVERSATION, tok, max_length=12)
    assert len(out["input_ids"]) == 12
    assert any(label != IGNORE_INDEX for label in out["labels"])


def test_non_monotonic_template_is_rejected():
    tok = FakeTokenizer(monotonic=False)
    with pytest.raises(ValueError, match="monotonically"):
        tokenize_conversation(CONVERSATION, tok, max_length=1000)


# --- thinking blocks -------------------------------------------------------

THINKING = [
    {"role": "user", "content": "what is dft"},
    {"role": "assistant", "content": "dynamic fine tuning"},
]


def test_no_think_masks_the_block_but_keeps_it_in_input_ids():
    """The template always supplies the block, so it must stay in context."""
    tok = FakeTokenizer(thinking=True)
    out = tokenize_conversation(THINKING, tok, max_length=1000, no_think=True)

    assert out["input_ids"] == tok.apply_chat_template(THINKING)
    supervised = [i for i, label in zip(out["input_ids"], out["labels"])
                  if label != IGNORE_INDEX]
    assert tok.decode_pieces(supervised) == [
        "<assistant>", "dynamic", "fine", "tuning", "<end>",
    ]


def test_think_flag_supervises_the_block():
    tok = FakeTokenizer(thinking=True)
    out = tokenize_conversation(THINKING, tok, max_length=1000, no_think=False)
    supervised = [i for i, label in zip(out["input_ids"], out["labels"])
                  if label != IGNORE_INDEX]
    assert tok.decode_pieces(supervised) == [
        "<assistant>", "<think>", "\n\n", "</think>", "\n\n",
        "dynamic", "fine", "tuning", "<end>",
    ]


def test_no_think_is_a_no_op_without_think_tokens():
    tok = FakeTokenizer(thinking=False)
    with_flag = tokenize_conversation(CONVERSATION, tok, max_length=1000, no_think=True)
    without = tokenize_conversation(CONVERSATION, tok, max_length=1000, no_think=False)
    assert with_flag == without


def test_no_think_row_with_only_a_think_block_is_dropped():
    """An assistant turn whose entire content is thinking leaves nothing to learn."""
    tok = FakeTokenizer(thinking=True)
    rows = [{"role": "user", "content": "x"}, {"role": "assistant", "content": ""}]
    out = tokenize_conversation(rows, tok, max_length=1000, no_think=True)
    # The header and <end> still carry supervision, so the row survives; the
    # block itself must not.
    supervised = [i for i, label in zip(out["input_ids"], out["labels"])
                  if label != IGNORE_INDEX]
    assert "<think>" not in tok.decode_pieces(supervised)
