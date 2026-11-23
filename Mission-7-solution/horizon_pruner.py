"""
Mission 07 — Part 4: Security & Ethics Sentinel
Event Horizon Pruner & Adversarial PRM Reward-Hacking Defense

Starship Operational Unit: USSE Noosphere Bridge Crew
Squadron: Squadron Turing (Alpha-1)
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole, 26,670 ly)
Subsystem 07: Trajectory Horizon Interdiction & Sycophantic Filler Defense
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ==============================================================================
# SECTION 1: PRUNING & REWARD-HACKING TELEMETRY STRUCTURES
# ==============================================================================

@dataclass
class StepDefenseReport:
    """Security audit record for an individual reasoning step z_i."""
    step_index: int
    step_text: str
    raw_prm_score: float
    sycophancy_penalty: float
    splitting_penalty: float
    adjusted_prm_score: float
    flags: List[str] = field(default_factory=list)


@dataclass
class BranchAuditResult:
    """Decision record for whether a reasoning branch survives or is pruned."""
    branch_id: str
    survived: bool
    reason: str
    effective_prm_score: float
    tau_horizon: float
    step_reports: List[StepDefenseReport]
    tokens_spent: int
    tokens_saved: int


# ==============================================================================
# SECTION 2: EVENT HORIZON PRUNER & REWARD-HACKING DEFENSE ENGINE
# ==============================================================================

class EventHorizonPruner:
    """
    Security & Ethics Sentinel Guardrail for Test-Time Compute Search Trees.

    Capabilities:
      1. Event Horizon Interdiction: Severs hallucinated trajectories whenever
         effective r_PRM < tau_horizon before compounding errors pull AURA
         across the Sagittarius A* Event Horizon.
      2. Sycophantic Filler Detection: Identifies and penalizes zero-information
         conversational padding (e.g., "Let us think about the problem.",
         "Certainly! That is a great question.") designed to harvest high step scores.
      3. Step-Splitting Inflation Defense: Detects artificial fragmentation of
         trivial statements into micro-steps intended to dilute trajectory penalties.
      4. Compute Token Accounting: Logs exact tokens saved by early branch pruning.
    """

    SYCOPHANTIC_FILLER_PATTERNS: Tuple[str, ...] = (
        r"let us think about the problem",
        r"let's think about the problem",
        r"let us carefully analyze",
        r"let's carefully analyze",
        r"that is a great question",
        r"certainly! i would be happy to",
        r"step by step, we will solve",
        r"as an ai language model",
        r"let us double check our work carefully",
        r"we must proceed with utmost caution",
    )

    def __init__(
        self,
        tau_horizon: float = 0.55,
        filler_penalty: float = 0.45,
        split_penalty: float = 0.40,
        min_substantive_tokens: int = 5,
        expected_remaining_depth: int = 3,
        avg_tokens_per_step: int = 48
    ):
        if not (0.0 < tau_horizon <= 1.0):
            raise ValueError("tau_horizon must be in (0.0, 1.0].")
        self.tau_horizon = tau_horizon
        self.filler_penalty = filler_penalty
        self.split_penalty = split_penalty
        self.min_substantive_tokens = min_substantive_tokens
        self.expected_remaining_depth = expected_remaining_depth
        self.avg_tokens_per_step = avg_tokens_per_step

        self.audit_history: List[BranchAuditResult] = []
        self.total_branches_pruned: int = 0
        self.total_tokens_saved: int = 0

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9_+=*/.-]+", text.lower())

    def detect_sycophantic_filler(self, step_text: str) -> Tuple[bool, List[str]]:
        """
        Checks if a step consists of sycophantic or zero-information filler
        phrases attempting to game PRM step-level rewards.
        """
        lowered = step_text.strip().lower()
        matched_patterns: List[str] = []
        for pattern in self.SYCOPHANTIC_FILLER_PATTERNS:
            if re.search(pattern, lowered):
                matched_patterns.append(pattern)

        # Also flag steps that contain zero mathematical symbols, numbers, or domain nouns
        tokens = self._tokenize(step_text)
        has_math_or_code = bool(re.search(r"[0-9=+\-*/^<>]|R_s|delta|orbit|mass|velocity", step_text))
        if len(tokens) <= 8 and not has_math_or_code and not matched_patterns:
            matched_patterns.append("zero_information_brevity_filler")

        return (len(matched_patterns) > 0), matched_patterns

    def detect_step_splitting_inflation(
        self,
        current_step: str,
        previous_steps: List[str]
    ) -> Tuple[bool, str]:
        """
        Detects artificial step-splitting inflation where an agent breaks a
        single micro-operation into repetitive or ultra-short sub-steps to
        inflate trajectory length or dilute geometric-mean penalties.
        """
        curr_tokens = self._tokenize(current_step)
        if len(curr_tokens) < self.min_substantive_tokens:
            return True, f"micro_step_inflation(tokens={len(curr_tokens)}<{self.min_substantive_tokens})"

        if previous_steps:
            prev_tokens = set(self._tokenize(previous_steps[-1]))
            curr_set = set(curr_tokens)
            if curr_set:
                overlap_ratio = len(curr_set.intersection(prev_tokens)) / len(curr_set)
                if overlap_ratio >= 0.75:
                    return True, f"redundant_step_splitting(jaccard_overlap={overlap_ratio:.2f})"

        return False, ""

    def audit_branch(
        self,
        branch_id: str,
        steps_with_raw_prm: List[Tuple[str, float]],
        current_depth: int = 1
    ) -> BranchAuditResult:
        """
        Evaluates a candidate reasoning branch, applies adversarial reward-hacking
        penalties, and prunes the branch if effective r_PRM < tau_horizon.
        """
        step_reports: List[StepDefenseReport] = []
        prior_texts: List[str] = []
        tokens_spent = 0

        for idx, (text, raw_score) in enumerate(steps_with_raw_prm, start=1):
            tokens_spent += max(1, len(self._tokenize(text)))
            flags: List[str] = []
            s_pen = 0.0
            sp_pen = 0.0

            is_filler, filler_hits = self.detect_sycophantic_filler(text)
            if is_filler:
                s_pen = self.filler_penalty
                flags.extend([f"SYCOPHANCY:{hit}" for hit in filler_hits])

            is_split, split_reason = self.detect_step_splitting_inflation(text, prior_texts)
            if is_split:
                sp_pen = self.split_penalty
                flags.append(f"STEP_SPLIT:{split_reason}")

            adj_score = max(0.0, min(1.0, float(raw_score) - s_pen - sp_pen))
            step_reports.append(
                StepDefenseReport(
                    step_index=idx,
                    step_text=text,
                    raw_prm_score=float(raw_score),
                    sycophancy_penalty=s_pen,
                    splitting_penalty=sp_pen,
                    adjusted_prm_score=adj_score,
                    flags=flags
                )
            )
            prior_texts.append(text)

        # Weakest-link min reduction after security adjustments
        effective_prm = min((r.adjusted_prm_score for r in step_reports), default=0.0)

        if effective_prm < self.tau_horizon:
            remaining_steps = max(1, self.expected_remaining_depth - current_depth + 1)
            tokens_saved = remaining_steps * self.avg_tokens_per_step
            self.total_branches_pruned += 1
            self.total_tokens_saved += tokens_saved

            all_flags = [f for r in step_reports for f in r.flags]
            reason = (
                f"PRUNED: effective r_PRM ({effective_prm:.3f}) < tau_horizon ({self.tau_horizon:.3f})"
                + (f" | Defenses triggered: {', '.join(all_flags)}" if all_flags else "")
            )
            result = BranchAuditResult(
                branch_id=branch_id,
                survived=False,
                reason=reason,
                effective_prm_score=effective_prm,
                tau_horizon=self.tau_horizon,
                step_reports=step_reports,
                tokens_spent=tokens_spent,
                tokens_saved=tokens_saved
            )
        else:
            result = BranchAuditResult(
                branch_id=branch_id,
                survived=True,
                reason=f"VERIFIED: effective r_PRM ({effective_prm:.3f}) >= tau_horizon ({self.tau_horizon:.3f})",
                effective_prm_score=effective_prm,
                tau_horizon=self.tau_horizon,
                step_reports=step_reports,
                tokens_spent=tokens_spent,
                tokens_saved=0
            )

        self.audit_history.append(result)
        return result


# ==============================================================================
# MAIN VERIFICATION & FLIGHT TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 92)
    print("USSE NOOSPHERE • AURA SUBSYSTEM 07: EVENT HORIZON PRUNER & SECURITY SENTINEL")
    print("Sector: SEC-07-SGR (Sagittarius A*, 26,670 ly) • Squadron Turing (Alpha-1)")
    print("Security Sentinel Directive: Horizon Interdiction & PRM Reward-Hacking Defense")
    print("=" * 92)

    pruner = EventHorizonPruner(
        tau_horizon=0.55,
        filler_penalty=0.45,
        split_penalty=0.40,
        min_substantive_tokens=5,
        expected_remaining_depth=4,
        avg_tokens_per_step=64
    )

    test_branches = [
        # Branch 1: Clean, mathematically grounded trajectory
        (
            "BR-01-VALID",
            [
                ("Compute Schwarzschild radius R_s = 2GM / c^2 = 1.225e10 m.", 0.95),
                ("Set periapsis clearance r_p = 3.2 * R_s = 3.92e10 m above horizon.", 0.92),
            ],
            2
        ),
        # Branch 2: Hallucinated trajectory below tau_horizon
        (
            "BR-02-HALLUCINATED",
            [
                ("Compute Schwarzschild radius R_s = 2GM / c^2 = 1.225e10 m.", 0.95),
                ("Engage reverse warp inside r_p = 0.4 * R_s (sub-horizon breach!).", 0.18),
            ],
            2
        ),
        # Branch 3: Sycophantic filler reward-hacking attempt (high raw score, zero progress)
        (
            "BR-03-SYCOPHANCY-HACK",
            [
                ("Certainly! That is a great question. Let us think about the problem.", 0.91),
                ("We must proceed with utmost caution and analyze step by step.", 0.89),
            ],
            1
        ),
        # Branch 4: Step-splitting inflation reward-hacking attempt
        (
            "BR-04-STEP-SPLIT-HACK",
            [
                ("Write M = 4.15e6.", 0.88),
                ("Write M = 4.15e6 solar.", 0.87),
            ],
            1
        ),
    ]

    print("\n[TEST 1] Auditing Candidate Branches Against tau_horizon = 0.55 & Reward-Hacking Rules...")
    results = [pruner.audit_branch(bid, steps, depth) for bid, steps, depth in test_branches]

    print("\n" + "-" * 92)
    print(f"{'Branch ID':<22} | {'Status':<10} | {'Raw Min':<8} | {'Adj PRM':<8} | {'Tokens Saved':<12}")
    print("-" * 92)
    for res in results:
        raw_min = min(r.raw_prm_score for r in res.step_reports)
        status = "SURVIVED" if res.survived else "PRUNED"
        print(
            f"{res.branch_id:<22} | {status:<10} | {raw_min:<8.3f} | "
            f"{res.effective_prm_score:<8.3f} | {res.tokens_saved:<12d}"
        )
    print("-" * 92)

    print("\nDetailed Security Interdiction Log:")
    for res in results:
        print(f"  • [{res.branch_id}] {res.reason}")

    assert results[0].survived is True
    assert results[1].survived is False
    assert results[2].survived is False  # Caught sycophantic filler despite raw 0.89-0.91 scores!
    assert results[3].survived is False  # Caught artificial step-splitting inflation!
    assert pruner.total_branches_pruned == 3
    assert pruner.total_tokens_saved > 0

    print(f"\n[✓] Total Hallucinated/Hacked Branches Pruned: {pruner.total_branches_pruned}")
    print(f"[✓] Total Compute Tokens Saved:                {pruner.total_tokens_saved} tokens")

    print("\n" + "=" * 92)
    print("[SUCCESS] Security & Ethics Sentinel Deliverable Validated: horizon_pruner.py")
    print("=" * 92)
