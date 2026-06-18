from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional
from xgboost import XGBClassifier, XGBRegressor

from config.config import Config
from src.models.base import UpliftModel
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class XLearner(UpliftModel):
    def __init__(self, name, scale_pos_weight: Optional[float] = None):
        self.cfg = Config()
        super().__init__(name, self.cfg.random_state)
        self.model_t, self.model_c = XGBClassifier(**self.cfg.xgb_params), XGBClassifier(**self.cfg.xgb_params)
        self.model_t_cate, self.model_c_cate = XGBRegressor(**self.cfg.xgbr_params), XGBRegressor(**self.cfg.xgbr_params)
        self.g = LogisticRegression(solver="lbfgs", penalty=None)

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
        # Stage 1
        X_control, y_control = X[treatment==0], outcome[treatment==0]
        X_treatment, y_treatment = X[treatment==1], outcome[treatment==1]

        self.model_c.fit(X_control, y_control)
        self.model_t.fit(X_treatment, y_treatment)

        # Stage 2
        mu0_hat = self.model_c.predict_proba(X_treatment)[:, 1]
        mu1_hat = self.model_t.predict_proba(X_control)[:, 1]

        D1 = y_treatment - mu0_hat
        D0 = mu1_hat - y_control

        self.model_c_cate.fit(X_control, D0)
        self.model_t_cate.fit(X_treatment, D1)

        self.g.fit(X, treatment)

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
        propensity = self.g.predict_proba(X)[:, 1]
        return propensity * self.model_c_cate.predict(X) + (1 - propensity) * self.model_t_cate.predict(X)
