"""Tree-of-Thoughts beam search (BFS) and depth-first backtracking.

Run: python tot_solver.py --beam-width 3 --max-expansions 30
The included numeric puzzle is a reproducible demonstration; replace the
callbacks to solve another problem.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, Iterable, TypeVar

State = TypeVar('State', bound=Hashable)


@dataclass
class ThoughtNode(Generic[State]):
    state: State
    score: float = 0.0
    parent: 'ThoughtNode[State] | None' = field(default=None, repr=False)
    depth: int = 0
    action: str = ''

    def path(self) -> list['ThoughtNode[State]']:
        result = []
        node: ThoughtNode[State] | None = self
        while node is not None:
            result.append(node)
            node = node.parent
        return list(reversed(result))


@dataclass
class SearchResult(Generic[State]):
    strategy: str
    solution: ThoughtNode[State] | None
    expansions: int
    generated: int
    visited: int
    budget_exhausted: bool


class TreeOfThoughtsSolver(Generic[State]):
    """Callback-based search; expansion means one call to successors()."""

    def __init__(self, successors: Callable[[State], Iterable[tuple[str, State]]],
                 is_goal: Callable[[State], bool],
                 evaluate: Callable[[State], float],
                 beam_width: int = 3, max_expansions: int = 100):
        if beam_width < 1 or max_expansions < 0:
            raise ValueError('beam_width must be >= 1 and max_expansions >= 0')
        self.successors = successors
        self.is_goal = is_goal
        self.evaluate = evaluate
        self.beam_width = beam_width
        self.max_expansions = max_expansions

    def solve(self, initial_state: State, strategy: str = 'bfs') -> SearchResult[State]:
        if strategy not in ('bfs', 'dfs'):
            raise ValueError("strategy must be 'bfs' or 'dfs'")
        root = ThoughtNode(initial_state, float(self.evaluate(initial_state)))
        if self.is_goal(initial_state):
            return SearchResult(strategy, root, 0, 0, 1, False)
        seen: set[State] = {initial_state}
        expansions = generated = 0
        frontier = [root]
        while frontier and expansions < self.max_expansions:
            if strategy == 'bfs':
                next_level: list[ThoughtNode[State]] = []
                for node in frontier:
                    if expansions >= self.max_expansions:
                        break
                    expansions += 1
                    children = self._children(node, seen)
                    generated += len(children)
                    for child in children:
                        if self.is_goal(child.state):
                            return SearchResult(strategy, child, expansions, generated, len(seen), False)
                    next_level.extend(children)
                # Beam search retains the highest-ranked B nodes per level.
                next_level.sort(key=lambda n: n.score, reverse=True)
                frontier = next_level[:self.beam_width]
            else:
                node = frontier.pop()
                expansions += 1
                children = self._children(node, seen)
                generated += len(children)
                for child in children:
                    if self.is_goal(child.state):
                        return SearchResult(strategy, child, expansions, generated, len(seen), False)
                # DFS explores highest-scoring successor first, backtracking via stack.
                children.sort(key=lambda n: n.score)
                frontier.extend(children)
        return SearchResult(strategy, None, expansions, generated, len(seen),
                            bool(frontier) and expansions >= self.max_expansions)

    def _children(self, node: ThoughtNode[State], seen: set[State]) -> list[ThoughtNode[State]]:
        children = []
        for action, state in self.successors(node.state):
            if state in seen:
                continue
            seen.add(state)
            children.append(ThoughtNode(state, float(self.evaluate(state)), node,
                                        node.depth + 1, action))
        return children


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare ToT beam BFS and DFS')
    parser.add_argument('--beam-width', type=int, default=3)
    parser.add_argument('--max-expansions', type=int, default=30)
    parser.add_argument('--target', type=int, default=17)
    args = parser.parse_args()

    # Example: starting at 1, reach target using +2, *2, or +3.
    # Restrict the domain to keep the demonstration finite.
    limit = max(args.target * 2, 10)

    def successors(x: int) -> Iterable[tuple[str, int]]:
        for action, value in (('+2', x + 2), ('*2', x * 2), ('+3', x + 3)):
            if 0 <= value <= limit:
                yield action, value

    solver = TreeOfThoughtsSolver(successors, lambda x: x == args.target,
                                  lambda x: -abs(args.target - x),
                                  args.beam_width, args.max_expansions)
    print(f'Target: {args.target} | Beam width: {args.beam_width} | Expansion budget: {args.max_expansions}')
    print('Strategy | Found | Expansions | Generated | Visited | Budget exhausted | Path')
    for strategy in ('bfs', 'dfs'):
        result = solver.solve(1, strategy)
        path = ' -> '.join(str(n.state) for n in result.solution.path()) if result.solution else 'none'
        print(f'{strategy.upper():8} | {str(result.solution is not None):5} | {result.expansions:10} | '
              f'{result.generated:9} | {result.visited:7} | {str(result.budget_exhausted):16} | {path}')


if __name__ == '__main__':
    main()
