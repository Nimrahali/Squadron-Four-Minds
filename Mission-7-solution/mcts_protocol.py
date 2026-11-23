"""
Mission 07 — Part 2: Communications & Protocol Officer
4-Phase Monte Carlo Tree Search (MCTS) & PUCT Selection Engine

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-07-SGR (Sagittarius A* Supermassive Black Hole)
Subsystem 07: Gravitational Search Horizon & Process Reward Evaluator
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class MCTSNode:
    """Represents a node in the Monte Carlo Reasoning Tree."""
    state: str
    prior_prob: float = 1.0
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    @property
    def q_value(self) -> float:
        """Empirical mean action-value Q(s, a) = W(s, a) / N(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSEngine:
    """
    Implements the 4 canonical phases of Monte Carlo Tree Search (MCTS) for LLM reasoning:
    1. Selection (via Predictor + Upper Confidence Bound applied to Trees, PUCT)
    2. Expansion (attaching candidate reasoning steps with LLM policy priors P(a|s))
    3. Simulation / Rollout (estimating value via PRM or trajectory completion)
    4. Backpropagation (updating N(s,a) and W(s,a) along the ancestor path)
    """

    def __init__(self, c_puct: float = 1.5):
        self.c_puct = c_puct

    def puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """
        TODO (Protocol Officer): Compute the exact PUCT score for `child` under `parent`:
            PUCT(s, a) = Q(s, a) + c_puct * P(a | s) * sqrt(max(1, N(s))) / (1 + N(s, a))
        """
        parent_visits = max(1, parent.visit_count)
        u_value = self.c_puct * child.prior_prob * (math.sqrt(parent_visits) / (1 + child.visit_count))
        return child.q_value + u_value

    def select_best_child(self, node: MCTSNode) -> MCTSNode:
        """
        TODO (Protocol Officer): Return the child of `node` that maximizes `self.puct_score(node, child)`.
        """
        if not node.children:
            raise ValueError("Cannot select a child from a leaf node with no children.")
        return max(node.children, key=lambda child: self.puct_score(node, child))

    def select_leaf(self, root: MCTSNode) -> MCTSNode:
        """
        Phase 1 (Selection): Traverse down the tree from `root` using `select_best_child`
        until reaching a leaf node (no children or terminal).
        """
        curr = root
        while curr.children and not curr.is_terminal:
            curr = self.select_best_child(curr)
        return curr

    def expand(self, node: MCTSNode, action_priors: Dict[str, float]) -> List[MCTSNode]:
        """
        TODO (Protocol Officer) — Phase 2 (Expansion):
        Instantiate and attach child `MCTSNode` objects for each `(state_str, prior_prob)` in `action_priors`.
        """
        for state_str, prior_prob in action_priors.items():
            child = MCTSNode(state=state_str, prior_prob=prior_prob, parent=node)
            node.children.append(child)
        return node.children

    def simulate_rollout(self, node: MCTSNode, rollout_evaluator: Callable[[str], float]) -> float:
        """
        TODO (Protocol Officer) — Phase 3 (Simulation / Rollout):
        Evaluate `node.state` using `rollout_evaluator` and clamp the scalar reward to [0.0, 1.0].
        """
        reward = float(rollout_evaluator(node.state))
        return max(0.0, min(1.0, reward))

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        TODO (Protocol Officer) — Phase 4 (Backpropagation):
        Traverse from `node` up to the root (`curr = curr.parent`), incrementing `visit_count += 1`
        and adding `reward` to `value_sum` for every node along the path.
        """
        curr: Optional[MCTSNode] = node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += reward
            curr = curr.parent


# ==============================================================================
# Verification Harness for Part 2 (Communications & Protocol Officer)
# ==============================================================================
if __name__ == "__main__":
    print("=== Subsystem 07 (Part 2): MCTS & PUCT Protocol Verification ===")

    engine = MCTSEngine(c_puct=1.5)
    root = MCTSNode(state="Root: Sagittarius A* Ergosphere Geodesic Entry", visit_count=16)

    branch_a = MCTSNode(state="Branch A: Prograde Frame-Dragging Assist", prior_prob=0.50, parent=root)
    branch_a.visit_count = 5
    branch_a.value_sum = 0.75 * 5  # Q(s, A) = 0.75

    branch_b = MCTSNode(state="Branch B: Retrograde Penrose Extraction", prior_prob=0.40, parent=root)
    branch_b.visit_count = 1
    branch_b.value_sum = 0.55 * 1  # Q(s, B) = 0.55

    root.children = [branch_a, branch_b]

    puct_a = engine.puct_score(root, branch_a)
    puct_b = engine.puct_score(root, branch_b)
    selected = engine.select_best_child(root)

    print(f"Branch A -> Q={branch_a.q_value:.2f}, N={branch_a.visit_count}, P={branch_a.prior_prob:.2f} | PUCT = {puct_a:.4f}")
    print(f"Branch B -> Q={branch_b.q_value:.2f}, N={branch_b.visit_count}, P={branch_b.prior_prob:.2f} | PUCT = {puct_b:.4f}")
    print(f"Selected Next Node for Expansion: {selected.state}")

    assert abs(puct_a - 1.25) < 1e-6, f"Expected PUCT(A)=1.25, got {puct_a}"
    assert abs(puct_b - 1.75) < 1e-6, f"Expected PUCT(B)=1.75, got {puct_b}"
    assert selected is branch_b, "MCTS failed to select under-explored high-PUCT Branch B!"

    # Test Backpropagation
    engine.backpropagate(branch_b, reward=0.95)
    assert branch_b.visit_count == 2 and root.visit_count == 17, "Backpropagation failed to increment visit counts!"
    print("[OK] Part 2 (Communications & Protocol Officer) verification passed.")
