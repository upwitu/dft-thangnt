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
    """Minimal ChatML-like tokenizer: one token per word, plus role headers."""

    def __init__(self, monotonic: bool = True):
        self.monotonic = monotonic
        self.vocab: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        return self.vocab.setdefault(piece, len(self.vocab) + 1)

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        pieces = []
        for message in messages:
            pieces.append(f"<{message['role']}>")
            pieces.extend(message["content"].split())
            pieces.append("<end>")
        if not self.monotonic:
            # A template that rewrites earlier turns, e.g. by re-emitting a
            # system block that depends on later content.
            pieces = pieces[::-1]
        return [self._id(p) for p in pieces]

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
