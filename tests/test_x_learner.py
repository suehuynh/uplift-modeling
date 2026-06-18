"""Tests for src/models/t_learner.py"""

import numpy as np
import pandas as pd
import pytest

from src.models.x_learner import XLearner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_uplift_data():
    """
    Synthetic dataset with a known, designed treatment effect.

    Units where f0 > 0 respond strongly to treatment (true uplift = 0.5).
    Units where f0 <= 0 do not respond to treatment (true uplift = 0).
    This lets us verify the x-learner recovers the correct *direction*
    of heterogeneous treatment effects.
    """
    rng = np.random.default_rng(42)
    n = 2_000

    X = pd.DataFrame({
        "f0": rng.standard_normal(n),
        "f1": rng.standard_normal(n),
    })
    treatment = pd.Series(rng.integers(0, 2, size=n))

    # Base conversion probability
    base_prob = 0.1
    # Treatment effect only kicks in when f0 > 0
    true_uplift = np.where(X["f0"] > 0, 0.5, 0.0)
    prob = base_prob + treatment.to_numpy() * true_uplift
    prob = np.clip(prob, 0, 1)

    outcome = pd.Series(rng.binomial(1, prob))

    return X, treatment, outcome, true_uplift


# ---------------------------------------------------------------------------
# XLearner tests
# ---------------------------------------------------------------------------

class TestXLearner:

    def test_predict_before_fit_raises(self, synthetic_uplift_data):
        X, _, _, _ = synthetic_uplift_data
        model = XLearner(name="x-learner")
        with pytest.raises(RuntimeError):
            model.predict_uplift(X)

    def test_fit_returns_self(self, synthetic_uplift_data):
        X, treatment, outcome, _ = synthetic_uplift_data
        model = XLearner(name="x-learner")
        result = model.fit(X, treatment, outcome)
        assert result is model

    def test_fit_sets_is_fitted(self, synthetic_uplift_data):
        X, treatment, outcome, _ = synthetic_uplift_data
        model = XLearner(name="x-learner")
        model.fit(X, treatment, outcome)
        assert model._is_fitted is True

    def test_predict_uplift_shape(self, synthetic_uplift_data):
        X, treatment, outcome, _ = synthetic_uplift_data
        model = XLearner(name="x-learner")
        model.fit(X, treatment, outcome)
        tau_hat = model.predict_uplift(X)
        assert tau_hat.shape == (X.shape[0],)

    def test_scale_pos_weight_ignored(self, synthetic_uplift_data):
        X, treatment, outcome, _ = synthetic_uplift_data
        model = XLearner(name="x-learner", scale_pos_weight=5.0)
        assert model.model_t.get_params()["scale_pos_weight"] != 5.0
        assert model.model_c.get_params()["scale_pos_weight"] != 5.0

    def test_model_recovers_uplift_direction(self, synthetic_uplift_data):
        """
        Units with f0 > 0 have true uplift = 0.5; units with f0 <= 0
        have true uplift = 0. The model should predict higher uplift
        for the f0 > 0 group on average.
        """
        X, treatment, outcome, true_uplift = synthetic_uplift_data
        model = XLearner(name="x-learner")
        model.fit(X, treatment, outcome)
        tau_hat = model.predict_uplift(X)

        high_uplift_mask = true_uplift > 0
        mean_tau_high = tau_hat[high_uplift_mask].mean()
        mean_tau_low = tau_hat[~high_uplift_mask].mean()

        assert mean_tau_high > mean_tau_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])