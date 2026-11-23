"""
Mission 07 — Part 3: Data & Payload Specialist
Process Reward Model (PRM) Step Evaluator & Math-Shepherd Automated Supervision

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
"""

import math
from typing import Dict, List


class OutcomeRewardEvaluator:
    """
    Outcome Reward Model (ORM): Grades a reasoning chain solely by its final terminal step.
    Vulnerable to false-positive trajectories where intermediate algebraic errors cancel out.
    """

    def __init__(self, expected_final_marker: str):
        self.expected_final_marker = expected_final_marker

    def score_trajectory(self, steps: List[str]) -> float:
        if not steps:
            return 0.0
        return 1.0 if self.expected_final_marker in steps[-1] else 0.0


class ProcessRewardEvaluator:
    """
    Process Reward Model (PRM): Evaluates the logical validity of every intermediate reasoning step z_t,
    aggregates trajectory scores (min, prod, geometric_mean), and computes Math-Shepherd rollout labels.
    """

    def step_score(self, step: str) -> float:
        """
        TODO (Data Specialist): Score an individual reasoning step in (0.0, 1.0].
        Penalize steps containing '[FLAWED]' or invalid sign cancellations, and reward verified deductions.
        """
        if "[FLAWED]" in step or "SIGN_ERROR" in step:
            return 0.12
        if "[MINOR_DRIFT]" in step:
            return 0.55
        return 0.94

    def score_all_steps(self, steps: List[str]) -> List[float]:
        return [self.step_score(s) for s in steps]

    def aggregate_trajectory_min(self, steps: List[str]) -> float:
        """
        TODO (Data Specialist): Weakest-link reduction: min_{t=1..T} r_t.
        """
        if not steps:
            return 0.0
        return min(self.score_all_steps(steps))

    def aggregate_trajectory_prod(self, steps: List[str]) -> float:
        """
        TODO (Data Specialist): Joint validity reduction: prod_{t=1..T} r_t.
        """
        if not steps:
            return 0.0
        prod = 1.0
        for r in self.score_all_steps(steps):
            prod *= r
        return prod

    def aggregate_trajectory_geometric_mean(self, steps: List[str]) -> float:
        """
        TODO (Data Specialist): Length-normalized geometric mean: (prod_{t=1..T} r_t)^(1/T).
        """
        if not steps:
            return 0.0
        prod = self.aggregate_trajectory_prod(steps)
        return prod ** (1.0 / len(steps))

    def math_shepherd_estimation(
        self,
        intermediate_step: str,
        rollout_final_answers: List[str],
        ground_truth_answer: str,
    ) -> Dict[str, float]:
        """
        TODO (Data Specialist): Implement Math-Shepherd (Wang et al., 2024) step supervision estimation.
        Given M Monte Carlo completion answers `rollout_final_answers` sampled from `intermediate_step`:
          - soft_label = (1 / M) * sum(1 for ans in rollout_final_answers if ans == ground_truth_answer)
          - hard_label = 1.0 if any rollout matches ground_truth_answer else 0.0
        """
        if not rollout_final_answers:
            return {"soft_label": 0.0, "hard_label": 0.0}
        matches = sum(1 for ans in rollout_final_answers if ans == ground_truth_answer)
        soft_label = matches / len(rollout_final_answers)
        hard_label = 1.0 if matches > 0 else 0.0
        return {"soft_label": soft_label, "hard_label": hard_label}


# ==============================================================================
# Verification Harness for Part 3 (Data & Payload Specialist)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 3): PRM vs. ORM & Math-Shepherd Verification ===")

    orm = OutcomeRewardEvaluator(expected_final_marker="v_esc = 24.0 km/s")
    prm = ProcessRewardEvaluator()

    # Trajectory 1: Rigorous Kerr Geodesic Proof
    valid_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: Integrate conserved orbital energy E and angular momentum L_z",
        "Step 3: Solve radial potential boundary -> v_esc = 24.0 km/s",
    ]

    # Trajectory 2: False-Positive Trap (Double-sign cancellation in Step 2 reaches lucky right scalar!)
    false_positive_chain = [
        "Step 1: Set metric tensor g_uv for Kerr spin parameter a = 0.9M",
        "Step 2: [FLAWED] Flip sign of frame-dragging term twice (-1 * -1 = +1) via illegal tensor contraction",
        "Step 3: Accidental cancellation yields -> v_esc = 24.0 km/s",
    ]

    for label, chain in [("Rigorous Chain", valid_chain), ("False-Positive Trap", false_positive_chain)]:
        orm_score = orm.score_trajectory(chain)
        prm_min = prm.aggregate_trajectory_min(chain)
        prm_prod = prm.aggregate_trajectory_prod(chain)
        prm_geo = prm.aggregate_trajectory_geometric_mean(chain)
        print(
            f"{label:20s} | ORM={orm_score:.2f} | PRM(min)={prm_min:.3f} | "
            f"PRM(prod)={prm_prod:.3f} | PRM(geo)={prm_geo:.3f}"
        )

    assert orm.score_trajectory(false_positive_chain) == 1.0, "ORM should be fooled by the terminal answer!"
    assert prm.aggregate_trajectory_min(false_positive_chain) < 0.20, "PRM(min) must catch the flawed intermediate step!"

    # Test Math-Shepherd Monte Carlo Step Estimation
    shepherd = prm.math_shepherd_estimation(
        intermediate_step=valid_chain[1],
        rollout_final_answers=["24.0 km/s", "24.0 km/s", "22.5 km/s", "24.0 km/s"],
        ground_truth_answer="24.0 km/s",
    )
    print(f"Math-Shepherd Step Estimation (M=4 rollouts): {shepherd}")
    assert abs(shepherd["soft_label"] - 0.75) < 1e-6 and shepherd["hard_label"] == 1.0
    print("[OK] Part 3 (Data & Payload Specialist) verification passed.")
