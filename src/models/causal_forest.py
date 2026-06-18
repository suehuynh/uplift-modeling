from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional

from config.config import Config
from src.models.base import UpliftModel
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


logger = logging.getLogger(__name__)


class CausalForest(UpliftModel):
    def __init__(self, name, scale_pos_weight: Optional[float] = None):
        self.cfg = Config()
        super().__init__(name, self.cfg.random_state)
        self.model = CausalForestDML(
            model_y=RandomForestClassifier(random_state=self.cfg.random_state, n_estimators=50),
            model_t=RandomForestClassifier(random_state=self.cfg.random_state, n_estimators=50),
            random_state=self.cfg.random_state,
            n_estimators=50,
            discrete_outcome=True,
            discrete_treatment=True
        )
    
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
        self.model.fit(Y=outcome, T=treatment, X=X)
        self._is_fitted = True

        return self


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
        self._check_is_fitted()
        return self.model.effect(X)

