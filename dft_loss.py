"""The Dynamic Fine-Tuning objective from arXiv:2508.05629.

Standard SFT minimizes the per-token negative log-likelihood.  Read as a
policy gradient, that objective implicitly assigns each token a reward of
``1 / pi_theta(token)``, so rare tokens dominate the update and destabilize
training.  DFT rectifies this by rescaling each token's loss by the model's
own *detached* probability of that token, which cancels the inverse-probability
weighting and leaves a uniform reward of 1 across the expert trajectory:

    L_DFT = E_(x,y*) [ - sum_t sg(pi_theta(y*_t | y*_<t, x)) * log pi_theta(y*_t | y*_<t, x) ]

which is equation (9) of the paper.

Ported from the reference implementation in kd-baselines
(``src/kd_baselines/losses.py``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100

# Each token contributes -p*log(p), which is maximized at p = 1/e. A DFT loss
# above this bound means the reduction or the gradient-accumulation scaling is
# wrong, not that the model is bad -- see README, "Verifying a run".
MAX_DFT_LOSS = 1.0 / torch.e


def masked_token_mean(token_values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Average over response tokens only, ignoring IGNORE_INDEX positions."""
    mask = labels.ne(IGNORE_INDEX)
    count = mask.sum()
    if not torch.any(mask):
        raise ValueError(
            "This batch contains no response tokens. Every label is masked -- "
            "usually the prompt is longer than --max-length, so truncation "
            "removed the whole completion."
        )
    return (token_values * mask).sum() / count


def dft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Dynamic Fine-Tuning loss, equation (9) of arXiv:2508.05629.

    ``logits`` is [batch, sequence, vocabulary] straight from the model and
    ``labels`` is [batch, sequence] with IGNORE_INDEX on positions that should
    not be trained on. Both are shifted here for next-token alignment, so pass
    them exactly as the model saw them.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary].")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence].")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share their batch/sequence dimensions.")
    if labels.shape[1] < 2:
        raise ValueError("At least two tokens are required for next-token alignment.")

    logits, labels = logits[:, :-1].contiguous(), labels[:, 1:].contiguous()

    # fp32 for the softmax: in bf16 the probability of a confident token
    # saturates, and it is a multiplicative factor on every token's loss.
    logits_fp32 = logits.to(torch.float32)
    vocab_size = logits_fp32.shape[-1]

    token_loss = F.cross_entropy(
        logits_fp32.view(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
        ignore_index=IGNORE_INDEX,
    ).view(labels.shape)

    # sg(.) -- the probability is a constant coefficient, never a gradient path.
    with torch.no_grad():
        probs = F.softmax(logits_fp32, dim=-1)
        safe_labels = labels.clamp(min=0)
        prob_coefficients = probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    return masked_token_mean(token_loss * prob_coefficients, labels)
