"""
TargetingPolicy: budget-constrained targeting and ROI simulation.

Compares targeting strategies (random, propensity-based, CATE-based) under
a fixed budget by selecting the top N units by a strategy-specific ranking
score, then evaluating the TRUE incremental value of that selection using
a causal model's CATE estimates, regardless of which score was used to
rank and select the group.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PolicyResult:
    """Stores ROI simulation output for a single targeting strategy."""
    strategy_name: str
    n_targeted: int
    total_value_gained: float
    total_cost: float
    roi: float


class TargetingPolicy:
    """
    Simulates budget-constrained targeting policies and their ROI.

    Parameters
    ----------
    budget : float
        Total marketing budget available.
    cost_per_unit : float
        Cost to treat (advertise to) a single unit.
    """

    def __init__(self, budget: float, cost_per_unit: float) -> None:
        self.budget = budget
        self.cost_per_unit = cost_per_unit
        self.n_units = int(np.floor(budget / cost_per_unit))
        logger.info(
            "Budget $%.2f / cost $%.2f => targeting %d units",
            budget, cost_per_unit, self.n_units,
        )

    def simulate_roi(
        self,
        strategy_name: str,
        selection_scores: np.ndarray,
        true_tau_hat: np.ndarray,
        value_per_outcome: float,
    ) -> PolicyResult:
        """
        Select the top n_units by selection_scores, then evaluate the TRUE
        incremental value of that selection using true_tau_hat.

        Parameters
        ----------
        strategy_name : str
            Label for this targeting strategy, e.g. "random", "propensity", "cate".
        selection_scores : np.ndarray, shape (n,)
            Score used to rank and select units for this strategy. May differ
            from true_tau_hat, e.g. raw response probability for propensity targeting.
        true_tau_hat : np.ndarray, shape (n,)
            The causal model's CATE estimate for every unit. Used to evaluate
            gain for whichever units end up selected, regardless of strategy.
        value_per_outcome : float
            Dollar value assigned to one incremental outcome.

        Returns
        -------
        PolicyResult
        """
        if len(selection_scores) != len(true_tau_hat):
            raise ValueError("selection_scores and true_tau_hat must be the same length")

        top_idx = np.argsort(-selection_scores)[: self.n_units]
        selected_tau_hat = true_tau_hat[top_idx]

        total_value_gained = selected_tau_hat.sum() * value_per_outcome
        total_cost = self.n_units * self.cost_per_unit
        roi = (total_value_gained - total_cost) / total_cost

        result = PolicyResult(
            strategy_name=strategy_name,
            n_targeted=self.n_units,
            total_value_gained=total_value_gained,
            total_cost=total_cost,
            roi=roi,
        )
        logger.info("%-12s  ROI = %.2f", strategy_name, roi)
        return result

    def compare_strategies(self, results: list[PolicyResult]) -> pd.DataFrame:
        """Return a ranked comparison table across strategies."""
        rows = [
            {
                "strategy": r.strategy_name,
                "n_targeted": r.n_targeted,
                "total_value_gained": r.total_value_gained,
                "total_cost": r.total_cost,
                "roi": r.roi,
            }
            for r in results
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("roi", ascending=False)
            .reset_index(drop=True)
        )