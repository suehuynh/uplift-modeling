from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.config import Config
from src.models.base import UpliftModel

logger = logging.getLogger(__name__)


class SLearner(UpliftModel):
    def __init__(self, name, scale_pos_weight: float = 1.0):
        self.cfg = Config()
        super().__init__(name, self.cfg.random_state)
        self.model = XGBClassifier(**self.cfg.xgb_params, scale_pos_weight=scale_pos_weight)

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
        treatment_named = treatment.rename(self.cfg.treatment_col)
        X_and_treatment = pd.concat([X, treatment_named], axis=1)
        self.model.fit(X_and_treatment, outcome)

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

        X_all_treatment = X.copy()
        X_all_treatment[self.cfg.treatment_col] = 1

        X_all_control = X.copy()
        X_all_control[self.cfg.treatment_col] = 0

        return self.model.predict_proba(X_all_treatment)[:, 1] - self.model.predict_proba(X_all_control)[:, 1]
    
