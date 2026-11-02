"""Pure PyTorch Direct Preference Optimization (DPO) loss core.

Run: python 01_DPO_Loss_Core.py
All log probabilities are sums over response tokens only; prompt/padding tokens
must be excluded via loss_mask. Inputs are unshifted causal-LM logits and labels.
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sequence_log_probs(logits: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    """Sum log p(label[t+1] | tokens[:t+1]) over masked next-token targets.

    logits: [B,T,V], labels/mask: [B,T]. Mask denotes target positions.
    For a decoder-only LM, the first label cannot be predicted by logits.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or loss_mask.shape != labels.shape:
        raise ValueError('Expected logits [B,T,V] and labels/mask [B,T]')
    if logits.shape[1] < 2:
        raise ValueError('Sequence length must be at least 2')
    targets = labels[:, 1:].long()
    mask = loss_mask[:, 1:].bool()
    if not bool(mask.any(dim=1).all()):
        raise ValueError('Each sequence needs at least one response token')
    if bool(((targets < 0) | (targets >= logits.shape[-1]))[mask].any()):
        raise ValueError('Unmasked target token is outside vocabulary')
    safe_targets = targets.masked_fill(~mask, 0)
    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum(-1)


def dpo_loss(policy_chosen_logps: Tensor, policy_rejected_logps: Tensor,
             reference_chosen_logps: Tensor, reference_rejected_logps: Tensor,
             beta: float = 0.1) -> tuple[Tensor, dict[str, Tensor]]:
    """Stable standard DPO objective, averaged over preference pairs."""
    if beta <= 0:
        raise ValueError('beta must be positive')
    if not (policy_chosen_logps.shape == policy_rejected_logps.shape ==
            reference_chosen_logps.shape == reference_rejected_logps.shape):
        raise ValueError('All log-probability tensors must have identical shapes')
    chosen_logratio = policy_chosen_logps - reference_chosen_logps
    rejected_logratio = policy_rejected_logps - reference_rejected_logps
    preference_logits = beta * (chosen_logratio - rejected_logratio)
    losses = F.softplus(-preference_logits)  # -log(sigmoid(x)), stable
    metrics = {
        'loss_per_pair': losses.detach(),
        'chosen_rewards': (beta * chosen_logratio).detach(),
        'rejected_rewards': (beta * rejected_logratio).detach(),
        'reward_margin': preference_logits.detach(),
        'preference_probability': torch.sigmoid(preference_logits.detach()),
        'accuracy': (preference_logits.detach() > 0).float().mean(),
    }
    return losses.mean(), metrics


class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        if beta <= 0:
            raise ValueError('beta must be positive')
        self.beta = beta

    def forward(self, policy_chosen_logits: Tensor, policy_rejected_logits: Tensor,
                reference_chosen_logits: Tensor, reference_rejected_logits: Tensor,
                chosen_labels: Tensor, rejected_labels: Tensor,
                chosen_mask: Tensor, rejected_mask: Tensor):
        pc = sequence_log_probs(policy_chosen_logits, chosen_labels, chosen_mask)
        pr = sequence_log_probs(policy_rejected_logits, rejected_labels, rejected_mask)
        # The reference model must remain frozen during optimization.
        with torch.no_grad():
            rc = sequence_log_probs(reference_chosen_logits, chosen_labels, chosen_mask)
            rr = sequence_log_probs(reference_rejected_logits, rejected_labels, rejected_mask)
        return dpo_loss(pc, pr, rc, rr, self.beta)


def self_test():
    torch.manual_seed(42)
    batch, length, vocab = 4, 7, 20
    chosen_labels = torch.randint(vocab, (batch, length))
    rejected_labels = torch.randint(vocab, (batch, length))
    chosen_mask = torch.zeros(batch, length, dtype=torch.bool)
    rejected_mask = torch.zeros_like(chosen_mask)
    chosen_mask[:, 3:] = True
    rejected_mask[:, 2:6] = True
    pc = torch.randn(batch, length, vocab, requires_grad=True)
    pr = torch.randn(batch, length, vocab, requires_grad=True)
    rc = torch.randn(batch, length, vocab)
    rr = torch.randn(batch, length, vocab)
    criterion = DPOLoss(beta=0.1)
    loss, metrics = criterion(pc, pr, rc, rr, chosen_labels, rejected_labels,
                              chosen_mask, rejected_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert pc.grad is not None and torch.isfinite(pc.grad).all()
    assert pr.grad is not None and torch.isfinite(pr.grad).all()
    # Identical policy and reference gives DPO loss log(2).
    zero_loss, _ = dpo_loss(torch.tensor([1.]), torch.tensor([-2.]),
                            torch.tensor([1.]), torch.tensor([-2.]))
    assert torch.allclose(zero_loss, torch.log(torch.tensor(2.)))
    # Extreme margins should not overflow or produce NaN.
    extreme_loss, _ = dpo_loss(torch.tensor([-1e6, 1e6]), torch.zeros(2),
                               torch.zeros(2), torch.zeros(2))
    assert torch.isfinite(extreme_loss)
    # Masked target IDs (e.g. -100) must be ignored safely.
    ignored = chosen_labels.clone()
    ignored[~chosen_mask] = -100
    assert torch.allclose(sequence_log_probs(pc.detach(), ignored, chosen_mask),
                          sequence_log_probs(pc.detach(), chosen_labels, chosen_mask))
    print('All DPO core tests passed.')
    print('Loss:', loss.item(), 'Accuracy:', metrics['accuracy'].item())


if __name__ == '__main__':
    self_test()
