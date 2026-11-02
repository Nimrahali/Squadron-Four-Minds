"""Implicit DPO rewards, Bradley-Terry preference accuracy and calibration.

Run: python 02_Implicit_Reward_Evaluator.py
Requires torch, matplotlib. Saves bradley_terry_calibration.png.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def implicit_reward(policy_logps: Tensor, reference_logps: Tensor, beta: float = 0.1) -> Tensor:
    """r(x,y) = beta * (log pi_policy(y|x) - log pi_ref(y|x))."""
    if beta <= 0:
        raise ValueError('beta must be positive')
    if policy_logps.shape != reference_logps.shape:
        raise ValueError('Policy and reference log probabilities must match')
    return beta * (policy_logps - reference_logps)


def bradley_terry_probability(reward_a: Tensor, reward_b: Tensor) -> Tensor:
    """P(a preferred over b) = sigmoid(r_a - r_b)."""
    return torch.sigmoid(reward_a - reward_b)


def evaluate_preferences(policy_a: Tensor, policy_b: Tensor, reference_a: Tensor,
                         reference_b: Tensor, labels: Tensor, beta: float = 0.1,
                         bins: int = 10) -> dict:
    """Labels: 1 if A is preferred, 0 if B is preferred.

    Inputs are sequence log-probabilities (sums across response tokens).
    Returns accuracy, NLL, Brier score, ECE and per-bin reliability data.
    """
    if bins < 1:
        raise ValueError('bins must be >= 1')
    if not (policy_a.shape == policy_b.shape == reference_a.shape == reference_b.shape == labels.shape):
        raise ValueError('All inputs must have the same shape')
    if labels.numel() == 0 or not bool(((labels == 0) | (labels == 1)).all()):
        raise ValueError('Labels must be a nonempty tensor of 0s and 1s')
    with torch.no_grad():
        reward_a = implicit_reward(policy_a, reference_a, beta).float().reshape(-1)
        reward_b = implicit_reward(policy_b, reference_b, beta).float().reshape(-1)
        y = labels.float().reshape(-1)
        logits = reward_a - reward_b
        probs = torch.sigmoid(logits)
        predictions = (probs >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
        nll = F.binary_cross_entropy_with_logits(logits, y).item()
        brier = ((probs - y) ** 2).mean().item()
        # Reliability of the predicted class (confidence versus correctness).
        confidence = torch.maximum(probs, 1 - probs)
        correctness = (predictions == y).float()
        edges = torch.linspace(0.5, 1.0, bins + 1, device=probs.device)
        calibration = []
        ece = 0.0
        for i in range(bins):
            in_bin = (confidence >= edges[i]) & ((confidence < edges[i+1]) if i < bins-1 else (confidence <= edges[i+1]))
            count = int(in_bin.sum().item())
            if count:
                avg_conf = confidence[in_bin].mean().item()
                avg_acc = correctness[in_bin].mean().item()
                ece += (count / y.numel()) * abs(avg_conf - avg_acc)
                calibration.append({'confidence': avg_conf, 'accuracy': avg_acc, 'count': count})
        return {'accuracy': accuracy, 'nll': nll, 'brier_score': brier, 'ece': ece,
                'reward_a': reward_a.cpu(), 'reward_b': reward_b.cpu(),
                'win_probabilities': probs.cpu(), 'calibration': calibration}


def plot_bradley_terry(results: dict, filename: str = 'bradley_terry_calibration.png') -> None:
    """Save sigmoid response curve and empirical reliability diagram."""
    margins = torch.linspace(-8, 8, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(margins.numpy(), torch.sigmoid(margins).numpy(), label='BT sigmoid')
    axes[0].axhline(0.5, linestyle='--', color='gray', linewidth=1)
    axes[0].axvline(0, linestyle='--', color='gray', linewidth=1)
    axes[0].set(xlabel='Reward difference r(A) - r(B)', ylabel='P(A wins)',
                title='Bradley-Terry preference probability')
    axes[0].legend()
    axes[1].plot([0.5, 1], [0.5, 1], linestyle='--', color='gray', label='Perfect calibration')
    calibration = results['calibration']
    if calibration:
        axes[1].scatter([b['confidence'] for b in calibration],
                        [b['accuracy'] for b in calibration],
                        s=[max(25, b['count'] * 2) for b in calibration], label='Empirical bins')
    axes[1].set(xlim=(0.5, 1), ylim=(0, 1), xlabel='Mean predicted-class confidence',
                ylabel='Observed classification accuracy', title='Confidence calibration')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)


def self_test():
    torch.manual_seed(7)
    n = 500
    reference_a = -torch.rand(n) * 12
    reference_b = -torch.rand(n) * 12
    # Generate policy logps with synthetic preference signal.
    labels = torch.randint(0, 2, (n,))
    signal = (2 * labels.float() - 1) * 4.0
    policy_a = reference_a + signal + torch.randn(n) * 2
    policy_b = reference_b + torch.randn(n) * 2
    result = evaluate_preferences(policy_a, policy_b, reference_a, reference_b, labels, beta=0.2)
    assert 0 <= result['accuracy'] <= 1
    assert 0 <= result['ece'] <= 1
    assert torch.isfinite(result['win_probabilities']).all()
    assert torch.allclose(bradley_terry_probability(torch.tensor(1.), torch.tensor(1.)), torch.tensor(0.5))
    assert torch.isfinite(bradley_terry_probability(torch.tensor(1e6), torch.tensor(-1e6)))
    plot_bradley_terry(result)
    print('All implicit reward evaluator tests passed.')
    print({k: round(result[k], 4) for k in ('accuracy', 'nll', 'brier_score', 'ece')})
    print('Saved bradley_terry_calibration.png')


if __name__ == '__main__':
    self_test()
