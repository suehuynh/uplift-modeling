"""
Evaluator: Qini curves, AUUC, and comparative model plots.

References
----------
Radcliffe (2007) — "Using Control Groups to Target on Predicted Lift"
Diemert et al. (2018) — "A Large Scale Benchmark for Uplift Modeling"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Stores evaluation output for a single model."""
    model_name: str
    auuc: float
    qini_gains: np.ndarray       # cumulative incremental conversions
    qini_proportions: np.ndarray # fraction of population targeted


class Evaluator:
    """
    Computes uplift evaluation metrics and generates comparison plots.

    Supported metrics
    -----------------
    - Qini curve  : cumulative incremental conversions vs. % population targeted
    - AUUC        : Area Under the Uplift Curve (Qini area above random baseline)

    Usage
    -----
    >>> evaluator = Evaluator()
    >>> result = evaluator.evaluate(
    ...     model_name="T-Learner",
    ...     y=test.outcome,
    ...     treatment=test.treatment,
    ...     tau_hat=model.predict_uplift(test.X),
    ... )
    >>> evaluator.plot_qini(results=[result_a, result_b, result_c])
    """

    def __init__(self, n_bins: int = 100) -> None:
        """
        Parameters
        ----------
        n_bins : int
            Number of equal-sized quantile bins for the Qini curve.
        """
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model_name: str,
        y: pd.Series,
        treatment: pd.Series,
        tau_hat: np.ndarray,
    ) -> EvalResult:
        """
        Compute the Qini curve and AUUC for a single model.

        Parameters
        ----------
        model_name : str
            Label used in plots and logs.
        y : pd.Series, shape (n,)
            Observed binary outcomes.
        treatment : pd.Series, shape (n,)
            Binary treatment indicator {0, 1}.
        tau_hat : np.ndarray, shape (n,)
            Predicted uplift scores (higher = more responsive).

        Returns
        -------
        EvalResult
        """
        gains, proportions = self._qini_curve(
            y.to_numpy(), treatment.to_numpy(), tau_hat
        )
        auuc = self._area_between_curves(gains, proportions)
        result = EvalResult(
            model_name=model_name,
            auuc=auuc,
            qini_gains=gains,
            qini_proportions=proportions,
        )
        logger.info("%-20s  AUUC = %.5f", model_name, auuc)
        return result

    def plot_qini(
        self,
        results: list[EvalResult],
        title: str = "Qini Curves",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot Qini curves for one or more models on a single axis.

        Parameters
        ----------
        results : list[EvalResult]
            One entry per model to compare.
        title : str
            Plot title.
        save_path : str, optional
            If provided, save the figure to this path.

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        for res in results:
            ax.plot(
                res.qini_proportions,
                res.qini_gains,
                label=f"{res.model_name} (AUUC={res.auuc:.4f})",
                linewidth=2,
            )

        # Random targeting baseline (diagonal)
        ax.plot([0, 1], [0, results[0].qini_gains[-1]],
                linestyle="--", color="grey", label="Random targeting")

        ax.set_xlabel("Proportion of population targeted", fontsize=12)
        ax.set_ylabel("Cumulative incremental conversions", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            logger.info("Saved Qini plot to %s", save_path)

        return fig

    def summary_table(self, results: list[EvalResult]) -> pd.DataFrame:
        """
        Return a ranked DataFrame of AUUC scores.

        Parameters
        ----------
        results : list[EvalResult]

        Returns
        -------
        pd.DataFrame
            Columns: model_name, auuc. Sorted descending by AUUC.
        """
        rows = [{"model": r.model_name, "auuc": r.auuc} for r in results]
        return (
            pd.DataFrame(rows)
            .sort_values("auuc", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _qini_curve(
        self,
        y: np.ndarray,
        treatment: np.ndarray,
        tau_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Qini curve.

        Units are sorted descending by tau_hat, then incrementally targeted.
        At each bin, we compute the net lift:
            gains_k = (conversions_treated_k / n_treated_k
                       - conversions_control_k / n_control_k) * n_treated_k

        Parameters
        ----------
        y : np.ndarray, shape (n,)
        treatment : np.ndarray, shape (n,)
        tau_hat : np.ndarray, shape (n,)

        Returns
        -------
        gains : np.ndarray, shape (n_bins + 1,)
        proportions : np.ndarray, shape (n_bins + 1,)
        """
        n = len(y)
        order = np.argsort(-tau_hat)   # descending
        y_sorted = y[order]
        t_sorted = treatment[order]

        gains = [0.0]
        proportions = [0.0]
        bin_size = n // self.n_bins

        for k in range(1, self.n_bins + 1):
            idx = k * bin_size
            y_k = y_sorted[:idx]
            t_k = t_sorted[:idx]

            n_treated = t_k.sum()
            n_control = (1 - t_k).sum()

            conv_treated = y_k[t_k == 1].sum()
            conv_control = y_k[t_k == 0].sum()

            if n_treated == 0 or n_control == 0:
                gains.append(gains[-1])
            else:
                net_lift = (conv_treated / n_treated - conv_control / n_control) * n_treated
                gains.append(net_lift)

            proportions.append(idx / n)

        return np.array(gains), np.array(proportions)

    def _area_between_curves(
        self,
        gains: np.ndarray,
        proportions: np.ndarray,
    ) -> float:
        """
        Compute AUUC = area between Qini curve and random baseline,
        using the trapezoid rule.
        """
        random_gains = np.linspace(0, gains[-1], len(gains))
        return float(np.trapezoid(gains - random_gains, proportions))