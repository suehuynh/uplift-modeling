"""Tests for src/evaluation/evaluator.py — uses synthetic uplift scores."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import Evaluator, EvalResult


@pytest.fixture
def synthetic_data():
    """Perfect uplift: treated units with high tau_hat always convert."""
    rng = np.random.default_rng(1)
    n = 2_000
    treatment = pd.Series(rng.integers(0, 2, size=n))
    # tau_hat positively correlated with outcome in treated group
    tau_hat = rng.uniform(0, 1, size=n)
    y = pd.Series(
        ((treatment == 1) & (tau_hat > 0.5)).astype(int)
    )
    return y, treatment, tau_hat


class TestEvaluator:

    def test_evaluate_returns_eval_result(self, synthetic_data):
        y, treatment, tau_hat = synthetic_data
        ev = Evaluator(n_bins=20)
        result = ev.evaluate("TestModel", y, treatment, tau_hat)
        assert isinstance(result, EvalResult)

    def test_auuc_positive_for_good_model(self, synthetic_data):
        y, treatment, tau_hat = synthetic_data
        ev = Evaluator(n_bins=20)
        result = ev.evaluate("TestModel", y, treatment, tau_hat)
        assert result.auuc > 0

    def test_random_model_auuc_near_zero(self):
        rng = np.random.default_rng(2)
        n = 10_000   # larger n to reduce variance
        y = pd.Series(rng.integers(0, 2, size=n))
        treatment = pd.Series(rng.integers(0, 2, size=n))
        tau_hat = rng.uniform(0, 1, size=n)  # random scores
        ev = Evaluator(n_bins=50)
        result = ev.evaluate("Random", y, treatment, tau_hat)
        assert abs(result.auuc) < 20.0  # random model should not dominate

    def test_qini_curve_shape(self, synthetic_data):
        y, treatment, tau_hat = synthetic_data
        ev = Evaluator(n_bins=20)
        result = ev.evaluate("TestModel", y, treatment, tau_hat)
        assert len(result.qini_gains) == 21      # n_bins + 1
        assert len(result.qini_proportions) == 21

    def test_qini_starts_at_zero(self, synthetic_data):
        y, treatment, tau_hat = synthetic_data
        ev = Evaluator(n_bins=20)
        result = ev.evaluate("TestModel", y, treatment, tau_hat)
        assert result.qini_gains[0] == 0.0
        assert result.qini_proportions[0] == 0.0

    def test_summary_table_sorted_descending(self, synthetic_data):
        y, treatment, tau_hat = synthetic_data
        ev = Evaluator(n_bins=20)
        r1 = ev.evaluate("ModelA", y, treatment, tau_hat)
        r2 = ev.evaluate("ModelB", y, treatment, np.zeros(len(y)))
        table = ev.summary_table([r1, r2])
        assert table.iloc[0]["auuc"] >= table.iloc[1]["auuc"]