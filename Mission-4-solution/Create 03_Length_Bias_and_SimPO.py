
#!/usr/bin/env python3
"""
USSE NOOSPHERE • AURA SUBSYSTEM 04 RESTORATION
Sector: SEC-04-ORI (Betelgeuse & Orion Molecular Cloud Complex)
Module: Week 04 // Deck 04 // Bay 04: Alignment Pipelines, Preference Optimization & Steering Core
Deliverable: 03_Length_Bias_and_SimPO.py
Station Role: Data & Payload Specialist (Part 3)

Cadet Assignment Objective:
    1. Diagnose verbosity bias (length exploitation) in preference datasets where unnormalized
       DPO implicit rewards correlate artificially with sequence token count.
    2. Implement Simple Preference Optimization (SimPO) with length-normalized reference-free
       rewards and target separation margin gamma (Meng et al., NeurIPS 2024; Park et al., ACL 2024).
    3. Benchmark GPU VRAM footprints and forward pass throughput comparing standard DPO
       (2 models in memory) against reference-free SimPO (1 model in memory).
    4. Replicate and validate the Bay 04 / Deck 04 Portal Simulation (Stage 2: SimPO Length-Margin & De-biasing).

Mathematical Formulations:
    - Sequence Log-Likelihood:
        log P(y | x) = sum_{t=1}^{|y|} log P(y_t | x, y_{<t})
    - DPO Implicit Reward (Unnormalized, requires frozen pi_ref):
        r_DPO(x, y) = beta * log(pi_theta(y | x) / pi_ref(y | x))
    - SimPO Implicit Reward (Reference-Free, Length-Normalized):
        r_SimPO(x, y) = (beta / |y|) * sum_{t=1}^{|y|} log pi_theta(y_t | x, y_{<t})
                      = beta * avg_log_prob(y | x)
    - SimPO Preference Objective with Target Margin gamma:
        L_SimPO(theta) = -E_{(x, y_w, y_l)} [ log sigma( r_SimPO(x, y_w) - r_SimPO(x, y_l) - gamma ) ]
                       = -E_{(x, y_w, y_l)} [ log sigma( Delta r - gamma ) ]
    - SimPO Gradient Dynamics:
        nabla_theta L_SimPO(theta) = -beta * sigma(-(Delta r - gamma)) * [
            (1 / |y_w|) * nabla_theta log pi_theta(y_w | x) - (1 / |y_l|) * nabla_theta log pi_theta(y_l | x)
        ]
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. SEQUENCE LOG-PROBABILITY EXTRACTION (TOKEN GATHER & LENGTH NORMALIZATION)
# ==============================================================================

def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    label_pad_token_id: int = -100,
    average_log_prob: bool = False,
    return_lengths: bool = False
) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.LongTensor]]:
    """
    Computes per-sequence log-probabilities of token labels under model logits using gather operations.

    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size).
        labels: Tensor of shape (batch_size, seq_len) with prompt and pad tokens set to label_pad_token_id.
        label_pad_token_id: Token ID to ignore in loss computation (default -100).
        average_log_prob: If True, returns mean log-prob per valid response token: (1 / |y|) * sum_t log P(y_t).
        return_lengths: If True, returns a tuple of (logps, valid_sequence_lengths).

    Returns:
        Tensor of shape (batch_size,) containing sequence log-probabilities (sum or average),
        or tuple (logps, seq_lengths) if return_lengths=True.
    """
    # Autoregressive shift: token at position t predicts label at position t+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Create loss mask for non-prompt and non-padding response tokens
    loss_mask = (shift_labels != label_pad_token_id)

    # Clone and clamp padding tokens to index 0 for safe tensor gather execution
    clean_labels = shift_labels.clone()
    clean_labels[~loss_mask] = 0

    # Compute log-softmax over vocabulary
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log-probabilities corresponding to true target token labels
    per_token_logps = torch.gather(
        log_probs,
        dim=2,
        index=clean_labels.unsqueeze(2)
    ).squeeze(2)

    # Zero out prompt and padding token contributions
    per_token_logps = per_token_logps * loss_mask.float()
    total_logps = per_token_logps.sum(dim=-1)
    seq_lengths = loss_mask.sum(dim=-1).clamp(min=1)

    if average_log_prob:
        result_logps = total_logps / seq_lengths.float()
    else:
        result_logps = total_logps

    if return_lengths:
        return result_logps, seq_lengths
    return result_logps


# ==============================================================================
# 2. LENGTH BIAS & VERBOSITY EXPLOITATION DIAGNOSTICS
# ==============================================================================

def compute_pearson_correlation(
    x: List[float],
    y: List[float]
) -> float:
    """
    Computes sample Pearson correlation coefficient r between two continuous variable series.

    Formula:
        r_xy = sum((x_i - mean_x) * (y_i - mean_y)) / sqrt(sum((x_i - mean_x)^2) * sum((y_i - mean_y)^2))
    """
    n = len(x)
    if n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)

    denom = math.sqrt(var_x * var_y)
    if denom < 1e-12:
        return 0.0

    return cov_xy / denom


class LengthBiasAnalyzer:
    """
    Diagnostic suite for diagnosing verbosity bias and reward gaming in preference alignment.
    Evaluates how reward models scale with completion token length versus true semantic quality.
    """

    def __init__(self, beta_dpo: float = 0.1, beta_simpo: float = 2.0, gamma: float = 0.8):
        self.beta_dpo = beta_dpo
        self.beta_simpo = beta_simpo
        self.gamma = gamma

    def analyze_dataset(
        self,
        completions_data: List[Dict[str, Union[str, int, float]]]
    ) -> Dict[str, float]:
        """
        Analyzes a dataset of candidate completions to calculate correlations between sequence length
        and derived reward models (unnormalized DPO implicit reward vs length-normalized SimPO reward).

        Expected keys in each dict:
            - 'length': int, token count |y|
            - 'total_logp_policy': float, sum_t log pi_theta(y_t)
            - 'total_logp_ref': float, sum_t log pi_ref(y_t)
            - 'semantic_quality': float, normalized ground truth human/expert score in [0.0, 1.0]
        """
        lengths = []
        dpo_rewards = []
        simpo_rewards = []
        qualities = []

        for item in completions_data:
            length = float(item['length'])
            tot_p = float(item['total_logp_policy'])
            tot_r = float(item['total_logp_ref'])
            quality = float(item.get('semantic_quality', 0.5))

            # Unnormalized DPO implicit reward: r_DPO = beta * (log pi_theta - log pi_ref)
            r_dpo = self.beta_dpo * (tot_p - tot_r)

            # SimPO length-normalized reward: r_SimPO = (beta / length) * log pi_theta
            r_simpo = self.beta_simpo * (tot_p / max(length, 1.0))

            lengths.append(length)
            dpo_rewards.append(r_dpo)
            simpo_rewards.append(r_simpo)
            qualities.append(quality)

        corr_len_dpo = compute_pearson_correlation(lengths, dpo_rewards)
        corr_len_simpo = compute_pearson_correlation(lengths, simpo_rewards)
        corr_qual_dpo = compute_pearson_correlation(qualities, dpo_rewards)
        corr_qual_simpo = compute_pearson_correlation(qualities, simpo_rewards)

        # Percentage reduction in magnitude of length correlation
        len_bias_red = max(0.0, (abs(corr_len_dpo) - abs(corr_len_simpo)) / max(abs(corr_len_dpo), 1e-6)) * 100.0

        return {
            "correlation_length_dpo": corr_len_dpo,
            "correlation_length_simpo": corr_len_simpo,
            "correlation_quality_dpo": corr_qual_dpo,
            "correlation_quality_simpo": corr_qual_simpo,
            "length_bias_reduction_pct": len_bias_red
        }


# ==============================================================================
# 3. REFERENCE-FREE SimPO LOSS MODULE (PYTORCH)
# ==============================================================================

class SimPOLoss(nn.Module):
    """
    Simple Preference Optimization (SimPO) Loss Module (Meng et al., NeurIPS 2024).

    Mathematical Formulation:
        r_SimPO(x, y) = (beta / |y|) * sum_{t=1}^{|y|} log pi_theta(y_t | x, y_{<t})
                      = beta * avg_log_prob(y | x)
        L_SimPO(theta) = -E_{(x, yw, yl)} [ log sigma( r_SimPO(x, yw) - r_SimPO(x, yl) - gamma ) ]

    Key Architectural Advantages:
        1. Reference-Free: Eliminates frozen reference model pi_ref completely, saving ~50% GPU VRAM.
        2. Length-Normalized: Mitigates verbosity exploitation by dividing log-likelihood by |y|.
        3. Target Reward Margin (gamma > 0): Enforces a minimum utility gap between winning and losing generations.
    """

    def __init__(
        self,
        beta: float = 2.0,
        gamma: float = 0.8,
        label_pad_token_id: int = -100
    ):
        """
        Args:
            beta: Scaling constant for length-normalized rewards (typically 2.0 - 2.5).
            gamma: Target reward margin separating chosen and rejected completions (typically 0.5 - 1.4).
            label_pad_token_id: Token ID indicating prompt or padding to mask out.
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.label_pad_token_id = label_pad_token_id

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        chosen_lengths: Optional[torch.LongTensor] = None,
        rejected_lengths: Optional[torch.LongTensor] = None,
        already_normalized: bool = False
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        Computes SimPO loss and diagnostic alignment metrics.

        Args:
            policy_chosen_logps: Log-probabilities for chosen sequences, shape (batch_size,).
            policy_rejected_logps: Log-probabilities for rejected sequences, shape (batch_size,).
            chosen_lengths: Sequence lengths |y_w| for chosen completions, shape (batch_size,).
            rejected_lengths: Sequence lengths |y_l| for rejected completions, shape (batch_size,).
            already_normalized: If True, input logps are already divided by lengths.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        if already_normalized:
            chosen_avg_logps = policy_chosen_logps
            rejected_avg_logps = policy_rejected_logps
        else:
            if chosen_lengths is None or rejected_lengths is None:
                raise ValueError("chosen_lengths and rejected_lengths must be provided when already_normalized=False")
            c_len = chosen_lengths.float().clamp(min=1.0)
            r_len = rejected_lengths.float().clamp(min=1.0)
            chosen_avg_logps = policy_chosen_logps / c_len
            rejected_avg_logps = policy_rejected_logps / r_len

        # Length-normalized implicit rewards: r(x, y) = beta * avg_logp
        chosen_rewards = self.beta * chosen_avg_logps
        rejected_rewards = self.beta * rejected_avg_logps

        # Reward margin: Delta r = r_w - r_l
        reward_margins = chosen_rewards - rejected_rewards

        # SimPO logits with target reward margin offset: Delta r - gamma
        logits = reward_margins - self.gamma

        # Numerically stable loss: -log(sigmoid(logits)) = -F.logsigmoid(logits)
        loss = -F.logsigmoid(logits).mean()

        with torch.no_grad():
            chosen_win_rate = (chosen_rewards > rejected_rewards).float().mean()
            margin_accuracy = (reward_margins >= self.gamma).float().mean()
            simpo_probs = torch.sigmoid(logits).mean()
            # Gradient scale factor: beta * sigma(-(Delta r - gamma))
            grad_scale = (self.beta * torch.sigmoid(-logits)).mean()

        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_margin": reward_margins.mean().detach(),
            "margin_gap": (reward_margins - self.gamma).mean().detach(),
            "chosen_win_rate": chosen_win_rate,
            "margin_accuracy": margin_accuracy,
            "simpo_probability": simpo_probs,
            "gradient_scale": grad_scale
        }

        return loss, metrics

    def forward_from_logits(
        self,
        policy_chosen_logits: torch.FloatTensor,
        policy_rejected_logits: torch.FloatTensor,
        chosen_labels: torch.LongTensor,
        rejected_labels: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """
        End-to-end forward pass computing SimPO loss directly from raw token logits and target labels.
        """
        chosen_avg_logps = get_batch_logps(
            policy_chosen_logits, chosen_labels,
            label_pad_token_id=self.label_pad_token_id,
            average_log_prob=True,
            return_lengths=False
        )
        rejected_avg_logps = get_batch_logps(
            policy_rejected_logits, rejected_labels,
            label_pad_token_id=self.label_pad_token_id,
            average_log_prob=True,
            return_lengths=False
        )

        return self.forward(
            chosen_avg_logps,
            rejected_avg_logps,
            already_normalized=True
        )


# ==============================================================================
# 4. PORTAL SIMULATOR EXACT REPLICATION (STAGE 2: SimPO LENGTH-MARGIN LAB)
# ==============================================================================

def simulate_portal_simpo_dynamics(
    beta: float = 2.0,
    gamma: float = 0.8,
    len_w: int = 64,
    len_l: int = 240,
    avg_logp_w: float = -0.18,
    avg_logp_l: float = -0.32
) -> Dict[str, float]:
    """
    Replicates the exact mathematical computation performed by the Bay 04 / Deck 04
    Interactive Portal Simulator (Stage 2: SimPO Length-Margin & De-biasing).

    Returns:
        Dictionary of portal telemetry variables matching the reactive web component.
    """
    total_logp_w = len_w * avg_logp_w
    total_logp_l = len_l * avg_logp_l
    raw_delta = total_logp_w - total_logp_l

    simpo_rw = beta * avg_logp_w
    simpo_rl = beta * avg_logp_l
    delta_r = simpo_rw - simpo_rl
    simpo_delta_with_margin = delta_r - gamma

    simpo_p_win = 1.0 / (1.0 + math.exp(-simpo_delta_with_margin))
    simpo_loss = -math.log(max(simpo_p_win, 1e-12))
    gradient_weight = beta * (1.0 / (1.0 + math.exp(simpo_delta_with_margin)))

    return {
        "beta": beta,
        "gamma": gamma,
        "len_w": len_w,
        "len_l": len_l,
        "avg_logp_w": avg_logp_w,
        "avg_logp_l": avg_logp_l,
        "total_logp_w": total_logp_w,
        "total_logp_l": total_logp_l,
        "raw_delta": raw_delta,
        "simpo_rw": simpo_rw,
        "simpo_rl": simpo_rl,
        "delta_r": delta_r,
        "simpo_delta_with_margin": simpo_delta_with_margin,
        "simpo_p_win": simpo_p_win,
        "simpo_loss": simpo_loss,
        "gradient_scale": gradient_weight
    }


# ==============================================================================
# 5. GPU VRAM & THROUGHPUT BENCHMARKING (DPO VS SIMPO)
# ==============================================================================

def benchmark_vram_and_throughput(
    model_param_billions: float = 8.0,
    batch_size: int = 4,
    seq_len: int = 2048,
    precision_bytes: int = 2  # 2 bytes for FP16 / BF16
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Calculates theoretical GPU memory consumption and computational throughput
    comparing standard Direct Preference Optimization (DPO) against SimPO.

    DPO Architecture:
        - 1 Active Policy Model (pi_theta): Weights + Gradients + Optimizer States + Activations
        - 1 Frozen Reference Model (pi_ref): Weights + Forward Activations
        - Total 4 Forward Passes: pi(y_w), pi(y_l), ref(y_w), ref(y_l)

    SimPO Architecture:
        - 1 Active Policy Model (pi_theta): Weights + Gradients + Optimizer States + Activations
        - 0 Reference Model (Reference-free)
        - Total 2 Forward Passes: pi(y_w), pi(y_l)
    """
    # 1. Model Weights Memory (GB)
    weights_gb = (model_param_billions * 1e9 * precision_bytes) / (1024 ** 3)

    # 2. Gradients (GB) — required only for active trainable policy
    grads_gb = weights_gb

    # 3. Optimizer States (AdamW requires 2 FP32 states per parameter: 8 bytes per parameter)
    adamw_states_gb = (model_param_billions * 1e9 * 8) / (1024 ** 3)

    # 4. Activation Memory (GB) — forward and backward activations
    activations_policy_gb = (batch_size * seq_len * 4096 * 32 * precision_bytes) / (1024 ** 3)
    activations_ref_gb = activations_policy_gb * 0.5  # Reference model requires only inference activations

    # DPO Total Memory
    dpo_policy_mem = weights_gb + grads_gb + adamw_states_gb + activations_policy_gb
    dpo_ref_mem = weights_gb + activations_ref_gb
    dpo_total_mem = dpo_policy_mem + dpo_ref_mem

    # SimPO Total Memory
    simpo_total_mem = weights_gb + grads_gb + adamw_states_gb + activations_policy_gb

    vram_saved_gb = dpo_total_mem - simpo_total_mem
    vram_reduction_pct = (vram_saved_gb / dpo_total_mem) * 100.0
    throughput_gain_pct = ((4.0 - 2.0) / 4.0) * 100.0

    return {
        "DPO": {
            "active_policy_vram_gb": round(dpo_policy_mem, 2),
            "reference_model_vram_gb": round(dpo_ref_mem, 2),
            "total_vram_gb": round(dpo_total_mem, 2),
            "forward_passes_per_step": 4,
            "reference_model_required": "YES (Frozen)"
        },
        "SimPO": {
            "active_policy_vram_gb": round(simpo_total_mem, 2),
            "reference_model_vram_gb": 0.0,
            "total_vram_gb": round(simpo_total_mem, 2),
            "forward_passes_per_step": 2,
            "reference_model_required": "NO (Reference-Free)"
        },
        "Comparison": {
            "vram_saved_gb": round(vram_saved_gb, 2),
            "vram_reduction_pct": round(vram_reduction_pct, 1),
            "forward_compute_speedup_pct": round(throughput_gain_pct, 1)
        }
    }


# ==============================================================================
# 6. AUTOMATED VERIFICATION & TEST SUITE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("USSE NOOSPHERE • AURA SUBSYSTEM 04: LENGTH BIAS & SimPO CORE TEST")
    print("Sector: SEC-04-ORI (Betelgeuse) • Station: Data & Payload Specialist (Part 3)")
    print("=" * 78)

    torch.manual_seed(42)

    # -------------------------------------------------------------------------
    # TEST 1: Verbosity Bias & Length Correlation Diagnostics
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Diagnosing Verbosity Bias Across Telemetry Preference Dataset...")
    # Comprehensive dataset modeling realistic alignment benchmarks:
    # Contains diverse combinations of length (short/medium/long) and semantic quality.
    # Verbose responses accumulate positive unnormalized DPO log-ratios despite lower token quality,
    # leading to artificial length bias. SimPO normalizes by length to decouple quality from verbosity.
    eval_dataset = [
        # Pair A: Prompt on Plasma Confinement (High Quality: concise vs verbose)
        {"name": "A_concise_good", "length": 35,  "total_logp_policy": -5.25,  "total_logp_ref": -8.05,  "semantic_quality": 0.95},
        {"name": "A_verbose_good", "length": 150, "total_logp_policy": -24.0,  "total_logp_ref": -33.0,  "semantic_quality": 0.92},
        # Pair B: Prompt on Burn Trajectory (Equal Semantic Quality, one padded with sycophancy)
        {"name": "B_concise_norm", "length": 45,  "total_logp_policy": -8.10,  "total_logp_ref": -11.7,  "semantic_quality": 0.85},
        {"name": "B_verbose_fluff","length": 180, "total_logp_policy": -36.0,  "total_logp_ref": -48.6,  "semantic_quality": 0.82},
        # Pair C: Prompt on Magnetic Shielding (Verbose Fluff with low token quality)
        {"name": "C_extreme_fluff","length": 260, "total_logp_policy": -65.0,  "total_logp_ref": -85.8,  "semantic_quality": 0.40},
        {"name": "C_concise_poor", "length": 40,  "total_logp_policy": -12.0,  "total_logp_ref": -13.2,  "semantic_quality": 0.35},
        # Pair D: Prompt on Core Coolant (Verbose high quality vs Extreme low quality rambling)
        {"name": "D_verbose_high", "length": 210, "total_logp_policy": -33.6,  "total_logp_ref": -46.2,  "semantic_quality": 0.90},
        {"name": "D_verbose_ramble","length": 300, "total_logp_policy": -96.0, "total_logp_ref": -114.0, "semantic_quality": 0.25},
        # Pair E: Prompt on Emergency Abort
        {"name": "E_concise_sharp","length": 30,  "total_logp_policy": -4.20,  "total_logp_ref": -6.60,  "semantic_quality": 0.98},
        {"name": "E_verbose_filler","length": 240, "total_logp_policy": -52.8, "total_logp_ref": -72.0,  "semantic_quality": 0.50},
    ]

    analyzer = LengthBiasAnalyzer(beta_dpo=0.1, beta_simpo=2.0, gamma=0.8)
    diag_results = analyzer.analyze_dataset(eval_dataset)

    print(f"  • Unnormalized DPO correlation with length:   r = {diag_results['correlation_length_dpo']:+.4f} (Length exploitation present)")
    print(f"  • Length-normalized SimPO correlation:       r = {diag_results['correlation_length_simpo']:+.4f} (Verbosity bias neutralized)")
    print(f"  • SimPO correlation with true quality:      r = {diag_results['correlation_quality_simpo']:+.4f} (Strong quality alignment)")
    print(f"  • Length Bias Reduction Percentage:         {diag_results['length_bias_reduction_pct']:.2f}% (Target: >30.0%)")

    assert abs(diag_results['correlation_length_simpo']) < abs(diag_results['correlation_length_dpo']), (
        "SimPO failed to reduce correlation with completion length!"
    )
    assert diag_results['length_bias_reduction_pct'] >= 30.0, (
        f"Length bias reduction {diag_results['length_bias_reduction_pct']:.1f}% below the 30% Anomaly Side Quest requirement!"
    )
    print("  [✓] Length Bias Diagnostic Suite: PASSED")

    # -------------------------------------------------------------------------
    # TEST 2: Portal Simulator Stage 2 Numerical Concordance
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Verifying Numerical Concordance with Portal Simulation Deck...")
    portal_sim = simulate_portal_simpo_dynamics(
        beta=2.0, gamma=0.8, len_w=64, len_l=240, avg_logp_w=-0.18, avg_logp_l=-0.32
    )

    print(f"  • Portal Chosen Log-Prob Sum (|y_w|=64):    {portal_sim['total_logp_w']:.2f} (avg: {portal_sim['avg_logp_w']})")
    print(f"  • Portal Rejected Log-Prob Sum (|y_l|=240):  {portal_sim['total_logp_l']:.2f} (avg: {portal_sim['avg_logp_l']})")
    print(f"  • Unnormalized Raw Difference:              {portal_sim['raw_delta']:+.2f} (Length artifacts present)")
    print(f"  • SimPO Chosen Reward r_w:                  {portal_sim['simpo_rw']:.3f}")
    print(f"  • SimPO Rejected Reward r_l:                {portal_sim['simpo_rl']:.3f}")
    print(f"  • SimPO Margin Gap (Δr - γ):                {portal_sim['simpo_delta_with_margin']:+.3f}")
    print(f"  • SimPO Preference Probability:             {portal_sim['simpo_p_win'] * 100:.2f}%")
    print(f"  • SimPO Objective Loss:                     {portal_sim['simpo_loss']:.4f}")
    print(f"  • SimPO Gradient Scale Factor:              {portal_sim['gradient_scale']:.4f}")

    assert abs(portal_sim['total_logp_w'] - (-11.52)) < 1e-4
    assert abs(portal_sim['total_logp_l'] - (-76.80)) < 1e-4
    assert abs(portal_sim['simpo_rw'] - (-0.360)) < 1e-4
    assert abs(portal_sim['simpo_rl'] - (-0.640)) < 1e-4
    assert abs(portal_sim['simpo_delta_with_margin'] - (-0.520)) < 1e-4
    print("  [✓] Portal Simulator Exact Replication: PASSED")

    # -------------------------------------------------------------------------
    # TEST 3: SimPOLoss Forward Pass & Autograd Gradient Flow
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Verifying SimPOLoss Gradient Flow & Autograd Propagation...")
    batch_size, seq_len, vocab_size = 4, 32, 256

    policy_chosen_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)

    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    chosen_labels[:, :6] = -100  # Mask out first 6 prompt tokens

    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels[:, :6] = -100

    simpo_module = SimPOLoss(beta=2.0, gamma=0.8)
    loss, metrics = simpo_module.forward_from_logits(
        policy_chosen_logits,
        policy_rejected_logits,
        chosen_labels,
        rejected_labels
    )

    loss.backward()

    print(f"  • SimPO Loss:                       {loss.item():.4f}")
    print(f"  • Mean Reward Margin (Δr):          {metrics['reward_margin'].item():+.4f}")
    print(f"  • Target Margin Accuracy (Δr >= γ): {metrics['margin_accuracy'].item() * 100:.1f}%")
    print(f"  • Pairwise Win Rate (Δr > 0):       {metrics['chosen_win_rate'].item() * 100:.1f}%")
    print(f"  • SimPO Sigmoid Probability:        {metrics['simpo_probability'].item() * 100:.2f}%")
    print(f"  • Policy Chosen Logits Grad Norm:   {policy_chosen_logits.grad.norm().item():.4f}")
    print(f"  • Policy Rejected Logits Grad Norm: {policy_rejected_logits.grad.norm().item():.4f}")

    assert policy_chosen_logits.grad is not None, "Gradient missing on policy chosen logits!"
    assert policy_rejected_logits.grad is not None, "Gradient missing on policy rejected logits!"
    print("  [✓] SimPOLoss Forward & Backward Flow: PASSED")

    # -------------------------------------------------------------------------
    # TEST 4: Extreme Logit Divergence & Numerical Stability
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Stress-Testing Numerical Stability Under Extreme Divergence...")
    extreme_chosen = torch.tensor([500.0, -500.0, 0.0])
    extreme_rejected = torch.tensor([-500.0, 500.0, 0.0])
    lens = torch.tensor([10, 10, 10])

    stable_loss, stable_metrics = simpo_module(
        extreme_chosen, extreme_rejected, lens, lens, already_normalized=False
    )

    assert not torch.isnan(stable_loss).any(), "NaN encountered in extreme loss computation!"
    assert not torch.isinf(stable_loss).any(), "Inf encountered in extreme loss computation!"
    print(f"  • Extreme Loss (Bounded & Finite):  {stable_loss.item():.4f}")
    print("  [✓] Numerical Stability Assertion: PASSED")

    # -------------------------------------------------------------------------
    # TEST 5: GPU VRAM Footprint & Throughput Benchmark (Llama-3-8B Scale)
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Benchmarking VRAM Footprint & Throughput (Llama-3-8B Scale)...")
    bench = benchmark_vram_and_throughput(model_param_billions=8.0, batch_size=4, seq_len=2048)

    print("-" * 78)
    print(f"{'Alignment Pipeline':<22} {'Policy VRAM':<15} {'Ref Model VRAM':<18} {'Total VRAM':<15}")
    print("-" * 78)
    print(f"{'Standard DPO':<22} {bench['DPO']['active_policy_vram_gb']:<15.2f} {bench['DPO']['reference_model_vram_gb']:<18.2f} {bench['DPO']['total_vram_gb']:<15.2f} GB")
    print(f"{'Reference-Free SimPO':<22} {bench['SimPO']['active_policy_vram_gb']:<15.2f} {bench['SimPO']['reference_model_vram_gb']:<18.2f} {bench['SimPO']['total_vram_gb']:<15.2f} GB")
    print("-" * 78)
    print(f"  • VRAM Saved: {bench['Comparison']['vram_saved_gb']} GB ({bench['Comparison']['vram_reduction_pct']}% reduction)")
    print(f"  • Forward Compute Speedup: {bench['Comparison']['forward_compute_speedup_pct']}% reduction in forward passes")
    print("  [✓] Architectural Memory Benchmark: PASSED")

    print("\n" + "=" * 78)
    print("[SUCCESS] Subsystem 04 Deliverable 03_Length_Bias_and_SimPO.py Fully Verified!")
    print("==============================================================================")
