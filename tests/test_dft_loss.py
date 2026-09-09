"""Check dft_loss against the closed form of equation (9) in arXiv:2508.05629."""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dft_loss import IGNORE_INDEX, MAX_DFT_LOSS, dft_loss, masked_token_mean


def reference_dft(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Independent, loop-based evaluation of -sum_t sg(p_t) log p_t / count."""
    logits, labels = logits[:, :-1], labels[:, 1:]
    total, count = 0.0, 0
    for b in range(labels.shape[0]):
        for t in range(labels.shape[1]):
            y = labels[b, t].item()
            if y == IGNORE_INDEX:
                continue
            p = F.softmax(logits[b, t].float(), dim=-1)[y].item()
            total += -p * math.log(p)
            count += 1
    return total / count


def test_matches_closed_form():
    torch.manual_seed(0)
    logits = torch.randn(3, 7, 11)
    labels = torch.randint(0, 11, (3, 7))
    labels[0, :3] = IGNORE_INDEX  # a masked prompt span
    got = dft_loss(logits, labels).item()
    assert got == pytest.approx(reference_dft(logits, labels), rel=1e-5)


def test_never_exceeds_one_over_e():
    """Each token contributes -p*log(p) <= 1/e, so the mean must too."""
    torch.manual_seed(1)
    for scale in (0.01, 1.0, 5.0, 50.0):  # near-uniform through near-one-hot
        logits = torch.randn(4, 9, 23) * scale
        labels = torch.randint(0, 23, (4, 9))
        assert dft_loss(logits, labels).item() <= MAX_DFT_LOSS + 1e-6


def test_peaks_at_probability_one_over_e():
    """A token whose probability is exactly 1/e should contribute exactly 1/e."""
    vocab = 4
    # Build logits whose softmax puts exactly 1/e on the target token.
    p = 1.0 / math.e
    rest = (1.0 - p) / (vocab - 1)
    probs = torch.tensor([p] + [rest] * (vocab - 1))
    logits = probs.log().view(1, 1, vocab).repeat(1, 2, 1)
    labels = torch.tensor([[IGNORE_INDEX, 0]])
    assert dft_loss(logits, labels).item() == pytest.approx(MAX_DFT_LOSS, rel=1e-5)


def test_ignores_masked_positions():
    """Changing logits under IGNORE_INDEX positions must not move the loss."""
    torch.manual_seed(2)
    logits = torch.randn(2, 6, 13)
    labels = torch.randint(0, 13, (2, 6))
    labels[:, :4] = IGNORE_INDEX
    before = dft_loss(logits, labels).item()
    logits[:, :3] = torch.randn(2, 3, 13) * 10  # positions feeding masked labels
    assert dft_loss(logits, labels).item() == pytest.approx(before, rel=1e-6)


def test_probability_coefficient_is_detached():
    """Gradient must match -sg(p) * d/dtheta log p, not the product rule."""
    torch.manual_seed(3)
    logits = torch.randn(1, 3, 5, requires_grad=True)
    labels = torch.tensor([[IGNORE_INDEX, 2, 4]])

    dft_loss(logits, labels).backward()
    got = logits.grad.clone()

    # Recompute with the coefficient supplied as an explicit constant.
    logits2 = logits.detach().clone().requires_grad_(True)
    shifted, shifted_labels = logits2[:, :-1], labels[:, 1:]
    logp = F.log_softmax(shifted.float(), dim=-1)
    mask = shifted_labels.ne(IGNORE_INDEX)
    safe = shifted_labels.clamp(min=0)
    tok_logp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    coeff = tok_logp.exp().detach()
    (-(coeff * tok_logp) * mask).sum().div(mask.sum()).backward()

    assert torch.allclose(got, logits2.grad, atol=1e-6)


def test_shift_alignment():
    """Label at position t must be scored by the logits at position t-1."""
    vocab = 6
    logits = torch.full((1, 3, vocab), -10.0)
    logits[0, 0, 5] = 10.0  # position 0 confidently predicts token 5
    labels = torch.tensor([[IGNORE_INDEX, 5, IGNORE_INDEX]])
    # Only the pair (logits[0], labels[1]) survives, and it is a confident hit,
    # so p is near 1 and -p*log(p) is near 0.
    assert dft_loss(logits, labels).item() < 1e-3


def test_raises_when_every_label_is_masked():
    logits = torch.randn(1, 4, 7)
    labels = torch.full((1, 4), IGNORE_INDEX)
    with pytest.raises(ValueError, match="no response tokens"):
        dft_loss(logits, labels)


@pytest.mark.parametrize("logits,labels,match", [
    (torch.randn(3, 5), torch.randint(0, 5, (3, 5)), "batch, sequence, vocabulary"),
    (torch.randn(2, 5, 7), torch.randint(0, 7, (2,)), "batch, sequence"),
    (torch.randn(2, 5, 7), torch.randint(0, 7, (2, 4)), "share their batch"),
    (torch.randn(2, 1, 7), torch.randint(0, 7, (2, 1)), "At least two tokens"),
])
def test_shape_validation(logits, labels, match):
    with pytest.raises(ValueError, match=match):
        dft_loss(logits, labels)


def test_masked_token_mean_divides_by_unmasked_count():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    labels = torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX, 1]])
    assert masked_token_mean(values, labels).item() == pytest.approx(3.0)


def test_dft_is_smaller_than_sft_on_confident_tokens():
    """The reweighting shrinks the contribution of low-probability tokens."""
    torch.manual_seed(4)
    logits = torch.randn(2, 8, 17)
    labels = torch.randint(0, 17, (2, 8))
    shifted, shifted_labels = logits[:, :-1], labels[:, 1:]
    sft = F.cross_entropy(
        shifted.reshape(-1, 17).float(), shifted_labels.reshape(-1), ignore_index=IGNORE_INDEX
    ).item()
    assert dft_loss(logits, labels).item() < sft
