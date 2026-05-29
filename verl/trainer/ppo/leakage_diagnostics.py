"""Leakage diagnostics utilities for CV-SDPO.

Core diagnostic metrics (cos_full_ans, beta_mean, entropy tracking) are computed
inline in compute_cv_sdpo_clean_target() in core_algos.py and logged to wandb
automatically during training.

This module provides optional standalone analysis tools for deeper investigation,
such as deliberation/shortcut token mass tracking.
"""

import torch
from typing import Optional


# Token groups for deliberation vs shortcut analysis
SEARCH_TOKENS = ["Wait", "Maybe", "Alternatively", "Consider", "Check", "Suppose", "Let"]
SHORTCUT_TOKENS = ["Therefore", "Thus", "Hence", "answer", "option", "boxed", "final"]


def compute_token_group_mass(
    log_probs: torch.Tensor,
    token_ids: list[int],
    response_mask: Optional[torch.Tensor] = None,
) -> float:
    """Compute total probability mass on a group of token IDs.

    Args:
        log_probs: Full vocabulary log-probs, shape (batch, seq_len, vocab_size).
        token_ids: List of vocabulary indices to sum over.
        response_mask: Binary mask, shape (batch, seq_len).

    Returns:
        Average probability mass across valid positions.
    """
    if not token_ids or log_probs is None:
        return 0.0

    indices = torch.tensor(token_ids, device=log_probs.device, dtype=torch.long)
    selected_log_probs = log_probs[..., indices]  # (batch, seq_len, len(token_ids))
    mass_per_position = selected_log_probs.exp().sum(dim=-1)  # (batch, seq_len)

    if response_mask is not None:
        valid_count = response_mask.sum().clamp(min=1.0)
        return (mass_per_position * response_mask).sum().item() / valid_count.item()
    return mass_per_position.mean().item()


def build_token_id_groups(tokenizer) -> dict[str, list[int]]:
    """Build token ID groups for search and shortcut tokens from a tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        Dictionary with "search" and "shortcut" keys mapping to token ID lists.
    """
    groups = {"search": [], "shortcut": []}

    for token_text in SEARCH_TOKENS:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if token_ids:
            groups["search"].append(token_ids[0])

    for token_text in SHORTCUT_TOKENS:
        token_ids = tokenizer.encode(token_text, add_special_tokens=False)
        if token_ids:
            groups["shortcut"].append(token_ids[0])

    return groups
