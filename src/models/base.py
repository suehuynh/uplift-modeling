"""
Abstract base class for all uplift models.

Concrete implementations live in separate modules:
    src/models/s_learner.py
    src/models/t_learner.py
    src/models/x_learner.py
    src/models/causal_forest.py

Each child class must implement:
    fit()            — train on (X, treatment, outcome)
    predict_uplift() — return CATE estimates tau_hat
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UpliftModel(ABC):
    """
    Abstract base class for heterogeneous treatment effect estimators.

    All models share the same fit/predict interface so the Evaluator
    and policy layer can treat them interchangeably.

    Parameters
    ----------
    name : str
        Human-readable identifier shown in plots and logs.
    random_state : int
        Seed passed down to base learners.
    """

    def __init__(self, name: str, random_state: int = 42) -> None:
        self.name = name
        self.random_state = random_state
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Interface every subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> "UpliftModel":
        """
        Train the model.

        Parameters
        ----------
        X : pd.DataFrame, shape (n, p)
            Feature matrix.
        treatment : pd.Series, shape (n,)
            Binary treatment indicator {0, 1}.
        outcome : pd.Series, shape (n,)
            Binary outcome {0, 1}.

        Returns
        -------
        self : UpliftModel
            Fluent interface for method chaining.
        """

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict individual treatment effect (CATE / tau_hat).

        Parameters
        ----------
        X : pd.DataFrame, shape (n, p)
            Feature matrix.

        Returns
        -------
        tau_hat : np.ndarray, shape (n,)
            Estimated uplift for each unit. Higher = more responsive to treatment.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all subclasses
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. Call .fit() first."
            )

    def _validate_inputs(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> None:
        if len(X) != len(treatment) or len(X) != len(outcome):
            raise ValueError("X, treatment, and outcome must have the same length.")
        unique_t = set(treatment.unique())
        if not unique_t.issubset({0, 1}):
            raise ValueError(f"treatment must be binary {{0,1}}, got {unique_t}")
        unique_y = set(outcome.unique())
        if not unique_y.issubset({0, 1}):
            raise ValueError(f"outcome must be binary {{0,1}}, got {unique_y}")

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, status={status})"