import torch
import torch.nn as nn


def compute_rope_frequencies(dim: int, seq_len: int, theta_base: float = 10000.0) -> torch.Tensor:
    """
    Computes RoPE frequency matrix theta_i = theta_base ** (-2(i-1)/dim) for i in [1, dim/2].
    Returns shape: (seq_len, dim // 2)
    """
    assert dim % 2 == 0, "Dimension must be even for 2D complex plane rotation"
    i = torch.arange(0, dim // 2, dtype=torch.float32)
    theta = 1.0 / (theta_base ** (2 * i / dim))  # (dim // 2,)
    m = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
    freqs = torch.outer(m, theta)  # (seq_len, dim // 2)
    return freqs


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Applies 2D vector rotation to Query or Key tensors.
    Args:
        x: Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        freqs: Frequency matrix of shape (seq_len, head_dim // 2)
    Returns:
        Rotated tensor of same shape as x.
    """
    batch_size, num_heads, seq_len, head_dim = x.shape
    x_even = x[..., 0::2]  # (batch_size, num_heads, seq_len, head_dim // 2)
    x_odd = x[..., 1::2]   # (batch_size, num_heads, seq_len, head_dim // 2)

    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, head_dim // 2)
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, head_dim // 2)

    # 2D Planar rotation: [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd = x_even * sin + x_odd * cos

    x_out = torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)
    return x_out


def verify_relative_distance_invariance():
    """
    Validates that <RoPE(q, m), RoPE(k, n)> depends purely on relative offset (m - n).
    """
    head_dim = 64
    seq_len = 100
    freqs = compute_rope_frequencies(head_dim, seq_len)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    # Position pairs with identical relative offset delta = 15
    m1, n1 = 40, 25
    m2, n2 = 70, 55

    q_m1 = apply_rope(q, freqs[m1:m1+1])
    k_n1 = apply_rope(k, freqs[n1:n1+1])
    dot1 = torch.sum(q_m1 * k_n1).item()

    q_m2 = apply_rope(q, freqs[m2:m2+1])
    k_n2 = apply_rope(k, freqs[n2:n2+1])
    dot2 = torch.sum(q_m2 * k_n2).item()

    print(f"--- RoPE RELATIVE INVARIANCE TEST ---")
    print(f"Dot product at positions ({m1}, {n1}) [offset = {m1-n1}]: {dot1:.6f}")
    print(f"Dot product at positions ({m2}, {n2}) [offset = {m2-n2}]: {dot2:.6f}")
    diff = abs(dot1 - dot2)
    print(f"Absolute Difference: {diff:.8f}")
    assert diff < 1e-5, "RoPE Relative Invariance Test Failed!"
    print("STATUS: RoPE RELATIVE INVARIANCE PASSED\n")


if __name__ == "__main__":
    verify_relative_distance_invariance()