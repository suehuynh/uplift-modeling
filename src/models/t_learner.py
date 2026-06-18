from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.config import Config
from src.models.base import UpliftModel

logger = logging.getLogger(__name__)


class TLearner(UpliftModel):
    def __init__(self, name, scale_pos_weight: float = 1.0):
        self.cfg = Config()
        super().__init__(name, self.cfg.random_state)
        self.model_t, self.model_c = XGBClassifier(**self.cfg.xgb_params, scale_pos_weight=scale_pos_weight), XGBClassifier(**self.cfg.xgb_params, scale_pos_weight=scale_pos_weight)
    

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

        X_control, y_control = X[treatment==0], outcome[treatment==0]
        X_treatment, y_treatment = X[treatment==1], outcome[treatment==1]
        self.model_c.fit(X_control, y_control)
        self.model_t.fit(X_treatment, y_treatment)

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
        # return self.model_t.predict(X) - self.model_c.predict(X)
        return self.model_t.predict_proba(X)[:, 1] - self.model_c.predict_proba(X)[:, 1]
    
