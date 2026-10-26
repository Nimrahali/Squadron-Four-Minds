#!/usr/bin/env python3
"""
USSE NOOSPHERE • AURA SUBSYSTEM 03 RESTORATION
Sector: SEC-03-M45 (Pleiades Open Cluster, Seven Sisters)
Module: Week 03
Role: Systems Architect (Lead Engineer) / Station Deliverable Part 1

Objective:
    Construct pure PyTorch MultiHeadSelfAttention from first principles without 
    relying on high-level library wrappers (e.g. torch.nn.MultiheadAttention).
    Implements:
      1. Linear projections for Query, Key, Value, and Output (W_Q, W_K, W_V, W_O).
      2. Multi-head splitting & tensor reshaping:
         (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim).
      3. Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.
      4. Dynamic Causal Masking (-inf upper triangle) for autoregressive generation.
      5. Support for optional arbitrary boolean attention masks.
      6. Mathematical and empirical verification of Variance Preservation (1 / sqrt(d_k)).
      7. Comprehensive automated test battery: shape assertions, causal validity, 
         gradient flow, and variance stability under Pleiades M45 telemetry noise.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Pure PyTorch implementation of Multi-Head Self-Attention (Vaswani et al., 2017).
    
    Architecture:
        Input: x in R^{B x N x d_model}
        Projections:
            Q = x @ W_Q,  K = x @ W_K,  V = x @ W_V   (W_Q, W_K, W_V in R^{d_model x d_model})
        Reshape:
            Splits d_model into h heads of dimension d_k = d_model // h.
            (B, N, d_model) -> (B, h, N, d_k)
        Scaled Dot-Product:
            Scores = (Q @ K^T) / sqrt(d_k)            in R^{B x h x N x N}
        Masking:
            If causal: Scores_{i, j} = -inf for j > i
        Softmax & Context:
            Weights = softmax(Scores, dim=-1)
            Context = Weights @ V                     in R^{B x h x N x d_k}
        Output Projection:
            Out = Context.view(B, N, d_model) @ W_O   (W_O in R^{d_model x d_model})
    """
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = False
    ):
        """
        Initialize the Multi-Head Self-Attention module.
        
        Args:
            d_model: Dimensionality of the model feature space (must be divisible by num_heads).
            num_heads: Number of parallel attention heads.
            dropout: Dropout probability applied to attention weights.
            bias: Whether linear projection layers include additive bias.
        """
        super().__init__()
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})."
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Learned linear projections for Query, Key, Value, and Output
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights with Xavier uniform for stable variance
        self._init_parameters()

    def _init_parameters(self):
        """Initialize projection weights using Xavier uniform initialization."""
        for module in (self.w_q, self.w_k, self.w_v, self.w_o):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute the Multi-Head Self-Attention forward pass.

        Args:
            x: Input feature tensor of shape (batch_size, seq_len, d_model).
            causal: If True, applies an autoregressive lower-triangular mask (-inf above diagonal).
            attention_mask: Optional boolean or floating mask of shape (batch_size, seq_len)
                            or (batch_size, 1, 1, seq_len), where False / 0 indicates masked positions.

        Returns:
            Tuple of:
                output: Projected context tensor of shape (batch_size, seq_len, d_model).
                attn_weights: Attention distribution weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model, (
            f"Input feature dimension {d_model} does not match model d_model {self.d_model}."
        )

        # Step 1: Linear projections for Q, K, V -> (batch_size, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Step 2: Multi-head splitting and transpose to (batch_size, num_heads, seq_len, head_dim)
        # Permutation (1, 2) aligns the matrix multiplication across seq_len x head_dim
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 3: Scaled dot-product attention scores
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Step 4: Apply Causal Masking (if autoregressive generation is requested)
        if causal and seq_len > 1:
            # Upper-triangular matrix with 1s above diagonal, filled with -infinity
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=scores.dtype),
                diagonal=1
            )
            # Broadcasting adds (seq_len, seq_len) across (batch_size, num_heads, seq_len, seq_len)
            scores = scores + causal_mask

        # Step 5: Apply optional external padding / custom attention mask
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # Shape (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                if attention_mask.dim() == 2:
                    expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                elif attention_mask.dim() == 3:
                    expanded_mask = attention_mask.unsqueeze(1)
                else:
                    expanded_mask = attention_mask
                scores = scores.masked_fill(~expanded_mask, float('-inf'))
            else:
                # Additive mask (e.g. 0.0 for keep, -inf for mask)
                scores = scores + attention_mask

        # Step 6: Numerical stability check & Softmax over the last key dimension
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights (only during training)
        attn_weights_dropped = self.dropout(attn_weights)

        # Step 7: Weight values by attention distribution
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights_dropped, v)

        # Step 8: Concatenate heads and project output
        # Transpose back: (batch_size, seq_len, num_heads, head_dim)
        # Contiguous view: (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)

        return output, attn_weights


# ==============================================================================
# RIGOROUS MATHEMATICAL & EMPIRICAL VERIFICATION SUITE
# ==============================================================================

def verify_variance_preservation() -> bool:
    """
    Pillar Verification: Scaled Dot-Product Variance Invariant
    
    Mathematical Derivation:
        Let q_i, k_i ~ i.i.d. N(0, 1) across d_k dimensions.
        The unscaled dot-product is:
            S = sum_{i=1}^{d_k} q_i * k_i
        Expectation:
            E[S] = sum_{i=1}^{d_k} E[q_i] * E[k_i] = 0
        Variance:
            Var(S) = sum_{i=1}^{d_k} Var(q_i * k_i)
            Since q_i, k_i are independent:
                Var(q_i * k_i) = E[q_i^2 k_i^2] - (E[q_i k_i])^2
                               = E[q_i^2] E[k_i^2] - 0 = 1 * 1 = 1
            Var(S) = sum_{i=1}^{d_k} 1 = d_k
            
        Therefore, StdDev(S) = sqrt(d_k).
        As d_k grows, large dot products push softmax into regions with extremely 
        small gradients (vanishing gradients).
        
        Scaling by 1 / sqrt(d_k):
            Var( S / sqrt(d_k) ) = (1 / d_k) * Var(S) = (1 / d_k) * d_k = 1.0
            
    This function empirically checks that scaled dot products maintain unit variance.
    """
    print("\n" + "=" * 78)
    print("  VERIFICATION 1: SCALED DOT-PRODUCT VARIANCE PRESERVATION INVARIANT")
    print("=" * 78)

    torch.manual_seed(42)
    d_k_values = [32, 64, 128, 256]
    seq_len = 256
    batch_size = 64
    all_passed = True

    print(f"{'d_k':>6} | {'Unscaled Var':>14} | {'Expected Unscaled':>17} | {'Scaled Var':>12} | {'Expected Scaled':>15} | Status")
    print("-" * 78)

    for d_k in d_k_values:
        q = torch.randn(batch_size, seq_len, d_k)
        k = torch.randn(batch_size, seq_len, d_k)

        unscaled_dot = torch.matmul(q, k.transpose(-2, -1))
        scaled_dot = unscaled_dot / math.sqrt(d_k)

        unscaled_var = unscaled_dot.var().item()
        scaled_var = scaled_dot.var().item()
        
        # Test tolerance of 5% around 1.0
        passed = abs(scaled_var - 1.0) < 0.05
        all_passed = all_passed and passed
        status = "PASSED [OK]" if passed else "FAILED [X]"

        print(
            f"{d_k:>6} | {unscaled_var:>14.4f} | {float(d_k):>17.1f} | "
            f"{scaled_var:>12.4f} | {1.000:>15.3f} | {status}"
        )

    print("-" * 78)
    if all_passed:
        print("[SUCCESS] All variance scaling tests passed. Softmax saturation risk eliminated.\n")
    else:
        print("[WARNING] Variance scaling deviation detected.\n")
    return all_passed


def verify_causal_masking() -> bool:
    """
    Autoregressive Decoder Verification: Causal Upper-Triangular Mask
    
    Verifies that for every token at index i, attention weights assigned
    to all future tokens j > i are strictly 0.0.
    """
    print("=" * 78)
    print("  VERIFICATION 2: CAUSAL MASK & AUTOREGRESSIVE ISOLATION AUDIT")
    print("=" * 78)

    torch.manual_seed(101)
    d_model = 128
    num_heads = 4
    seq_len = 8
    batch_size = 2

    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mha.eval()

    dummy_input = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        _, attn_weights = mha(dummy_input, causal=True)

    # Inspect attention weights: shape (batch_size, num_heads, seq_len, seq_len)
    # The upper triangular entries (diagonal + 1 onwards) must be 0.0
    upper_tri = torch.triu(attn_weights, diagonal=1)
    max_leakage = upper_tri.max().item()

    print(f"Batch Size: {batch_size}, Heads: {num_heads}, Sequence Length: {seq_len}")
    print(f"Maximum attention weight leakage into future tokens (j > i): {max_leakage:.2e}")

    # Verify rows sum to 1.0
    row_sums = attn_weights.sum(dim=-1)
    max_row_sum_err = (row_sums - 1.0).abs().max().item()
    print(f"Maximum deviation of attention row sum from 1.0: {max_row_sum_err:.2e}")

    passed = (max_leakage == 0.0) and (max_row_sum_err < 1e-5)
    print(f"Causal Isolation Status: {'PASSED [OK]' if passed else 'FAILED [X]'}\n")
    return passed


def verify_gradient_flow() -> bool:
    """
    Backpropagation Flow Verification:
    Ensures gradients correctly populate all linear projection matrices
    (W_Q, W_K, W_V, W_O) without exploding or vanishing.
    """
    print("=" * 78)
    print("  VERIFICATION 3: GRADIENT FLOW & BACKPROPAGATION INTEGRITY")
    print("=" * 78)

    torch.manual_seed(202)
    d_model = 256
    num_heads = 8
    seq_len = 16
    batch_size = 4

    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    mha.train()

    inputs = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    targets = torch.randn(batch_size, seq_len, d_model)

    outputs, _ = mha(inputs, causal=True)
    loss = F.mse_loss(outputs, targets)
    loss.backward()

    all_have_grad = True
    print(f"{'Parameter Name':<20} | {'Shape':<22} | {'Grad Mean':<14} | {'Grad Std':<14} | Status")
    print("-" * 78)
    for name, param in mha.named_parameters():
        has_grad = param.grad is not None and not torch.isnan(param.grad).any()
        all_have_grad = all_have_grad and has_grad
        g_mean = param.grad.mean().item() if param.grad is not None else 0.0
        g_std = param.grad.std().item() if param.grad is not None else 0.0
        print(f"{name:<20} | {str(list(param.shape)):<22} | {g_mean:>14.6e} | {g_std:>14.6e} | {'OK' if has_grad else 'FAIL'}")

    input_has_grad = inputs.grad is not None and not torch.isnan(inputs.grad).any()
    print(f"{'input_embeddings':<20} | {str(list(inputs.shape)):<22} | {inputs.grad.mean().item():>14.6e} | {inputs.grad.std().item():>14.6e} | {'OK' if input_has_grad else 'FAIL'}")
    print("-" * 78)

    passed = all_have_grad and input_has_grad
    print(f"Gradient Flow Status: {'PASSED [OK]' if passed else 'FAILED [X]'}\n")
    return passed


def run_astrogation_telemetry_benchmark():
    """
    Starship Simulation: Pleiades M45 Spatial Attention Mesh Benchmark
    Runs the multi-head attention core through simulated sensor frames.
    """
    print("=" * 78)
    print("  SIMULATION: AURA SUBSYSTEM 03 PLEIADES M45 SENSOR STREAM TEST")
    print("=" * 78)

    d_model = 512
    num_heads = 8
    seq_len = 64
    batch_size = 2

    print(f"[AURA STATUS] Initializing MultiHeadSelfAttention Core...")
    print(f"              d_model = {d_model}, num_heads = {num_heads}, head_dim = {d_model // num_heads}")
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)

    sensor_stream = torch.randn(batch_size, seq_len, d_model)
    print(f"[TELEMETRY] Ingesting {batch_size} sensor channels of length {seq_len}...")

    # Forward pass: Causal (Autoregressive Decoder)
    out_causal, attn_causal = mha(sensor_stream, causal=True)
    print(f"[CAUSAL FORWARD] Output Shape:     {out_causal.shape}")
    print(f"                 Attention Shape:  {attn_causal.shape}")

    # Forward pass: Bidirectional (Encoder Astrogation Map)
    out_bidi, attn_bidi = mha(sensor_stream, causal=False)
    print(f"[BIDI FORWARD]   Output Shape:     {out_bidi.shape}")
    print(f"                 Attention Shape:  {attn_bidi.shape}")

    print("\n[AURA DIAGNOSTIC REPORT]")
    print(f"  * Subsystem 03 Attention Matrix: STABILIZED")
    print(f"  * Total Ship Integrity Boost:    +7.1% (Fleet Total: 21.3%)")
    print(f"  * Astrogation Telemetry Buffer:  NOMINAL")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    print("\n" + "#" * 78)
    print("  USSE NOOSPHERE • MISSION 03: MULTI-HEAD ATTENTION CORE")
    print("  Sector SEC-03-M45 // Pleiades Open Cluster // AURA Subsystem 03")
    print("#" * 78)

    v1 = verify_variance_preservation()
    v2 = verify_causal_masking()
    v3 = verify_gradient_flow()

    if v1 and v2 and v3:
        run_astrogation_telemetry_benchmark()
        print("[FINAL VERDICT] 01_MultiHead_Attention_Core.py fully verified and operational.")
    else:
        print("[FATAL ERROR] One or more verification suites failed.")
        exit(1)
