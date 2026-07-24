"""Joint loss for SCP-Soft and component-isolated ablations."""
import torch
import torch.nn.functional as F


def soft_bce(logits, targets):
    """Soft-target BCE (binary_cross_entropy_with_logits handles soft targets)."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


def corn_loss(ord_logits, y_ord_bin):
    """CORN ordinal loss (Cao 2020) — K-1 monotone binary classifiers.

    ord_logits: [N, num_stmt, K-1=4] — sigmoids interpreted as P(rank > k)
    y_ord_bin : [N, num_stmt] integer in {0..5}; only positive (>0) computed
    Bin thresholds: 1, 2, 3, 4 (ranks above which each binary triggers).
    """
    pos_mask = (y_ord_bin > 0).unsqueeze(-1).float()  # [N, J, 1]
    K1 = ord_logits.shape[-1]
    bin_thresholds = torch.arange(1, K1 + 1, device=y_ord_bin.device).view(1, 1, K1)
    targets = (y_ord_bin.unsqueeze(-1) > bin_thresholds).float()
    losses = F.binary_cross_entropy_with_logits(ord_logits, targets, reduction="none")
    losses = losses * pos_mask
    denom = pos_mask.sum().clamp(min=1)
    return losses.sum() / denom


def hierarchy_loss(super_logits, sub_logits, stmt_logits,
                   edges, super2idx, sub2idx, stmt2idx):
    """L_hier = mean over edges of max(0, p_child - p_parent)^2."""
    p_super = torch.sigmoid(super_logits)
    p_sub = torch.sigmoid(sub_logits)
    p_stmt = torch.sigmoid(stmt_logits)
    losses = []
    for e in edges:
        if e["level"] == "super_to_sub":
            p_p = p_super[:, super2idx[e["parent"]]]
            p_c = p_sub[:, sub2idx[e["child"]]]
        else:  # sub_to_stmt
            p_p = p_sub[:, sub2idx[e["parent"]]]
            p_c = p_stmt[:, stmt2idx[e["child"]]]
        losses.append((torch.clamp(p_c - p_p, min=0) ** 2).mean())
    return torch.stack(losses).mean()


class SCPSoftLoss:
    """Configurable joint loss covering rows 1-5 of the ablation ladder."""

    def __init__(self, edges, super2idx, sub2idx, stmt2idx,
                 alpha=0.3, lambda_ord=0.3, lambda_hier=0.5,
                 use_soft=True, use_parent=True, use_ord=True, use_hier=True,
                 warmup_epochs=5):
        self.edges = edges
        self.super2idx = super2idx
        self.sub2idx = sub2idx
        self.stmt2idx = stmt2idx
        self.alpha = alpha
        self.lambda_ord = lambda_ord
        self.lambda_hier = lambda_hier
        self.use_soft = use_soft
        self.use_parent = use_parent
        self.use_ord = use_ord
        self.use_hier = use_hier
        self.warmup_epochs = warmup_epochs

    def __call__(self, out, batch, epoch=0):
        warmup = min(1.0, epoch / max(1, self.warmup_epochs))
        # Statement supervision (soft or hard depending on flag)
        y_target = batch["y_stmt"] if self.use_soft else batch["y_hard"]
        L = F.binary_cross_entropy_with_logits(out["stmt_logits"], y_target)

        if self.use_parent:
            y_super = batch["y_super"] if self.use_soft else (batch["y_super"] > 0).float()
            y_sub = batch["y_sub"] if self.use_soft else (batch["y_sub"] > 0).float()
            L = L + self.alpha * (
                F.binary_cross_entropy_with_logits(out["super_logits"], y_super) +
                F.binary_cross_entropy_with_logits(out["sub_logits"], y_sub)
            )

        if self.use_ord:
            L = L + warmup * self.lambda_ord * corn_loss(out["ord_logits"], batch["y_ord"])

        if self.use_hier:
            L = L + warmup * self.lambda_hier * hierarchy_loss(
                out["super_logits"], out["sub_logits"], out["stmt_logits"],
                self.edges, self.super2idx, self.sub2idx, self.stmt2idx)

        return L
