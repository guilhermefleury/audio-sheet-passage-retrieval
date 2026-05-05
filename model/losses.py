"""
Triplet (contrastive) loss used during training, equation (1) from the paper.

Since CrossModalEncoder already L2-normalises its outputs, dot product equals
cosine similarity, so we can use torch.mm instead of computing cosine distances.
"""

import torch


def triplet_loss(x: torch.Tensor, y: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    In-batch triplet loss with cosine similarity.

    For each matching pair (x_i, y_i) in the batch, every other y_j (j != i)
    is a negative. The loss penalises cases where the similarity gap between
    the positive and any negative is smaller than `margin`.

    Args:
        x      : [B, emb_dim]  sheet embeddings  (L2-normalised)
        y      : [B, emb_dim]  audio embeddings  (L2-normalised)
        margin : required minimum similarity gap (alpha in the paper)

    Returns:
        scalar loss
    """
    # [B, B] all pairwise cosine similarities (dot product works because normalised)
    sims = torch.mm(x, y.t())

    # Positive similarities sit on the diagonal: sim(x_i, y_i)
    pos_sims = sims.diagonal().unsqueeze(1)  # [B, 1]  — broadcast over negatives

    # Gap = positive_sim - negative_sim, for every off-diagonal entry
    eye = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    gaps = (pos_sims - sims)[~eye]           # [B*(B-1)]

    # Hinge: only penalise gaps smaller than the margin
    return torch.clamp(margin - gaps, min=0).mean()
