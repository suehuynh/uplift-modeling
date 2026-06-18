from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.config import Config
from src.data.loader import UpliftDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

from config.config import Config

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    Fits and applies feature transformations for the Criteo Uplift dataset.

    Transformation pipeline (applied to skewed features only):
        1. Shift non-positive values so minimum becomes 1
        2. Apply log1p transformation

    Parameters
    ----------
    cfg : Config
        Central configuration object.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.loader = DataLoader(cfg)
        self.log_transformer = FunctionTransformer(np.log1p, validate=True)
        self.shift_constants: dict[str, float] = {}

    def split(
        self,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
    ) -> tuple[UpliftDataset, UpliftDataset, UpliftDataset]:
        """Load data and return stratified train/val/test splits."""
        val_size = val_size or self.cfg.val_size
        test_size = test_size or self.cfg.test_size
        loader = self.loader.load(self.cfg.subsample_size)
        return loader.split(val_size=val_size, test_size=test_size)

    def fit_transform(
        self,
        train: UpliftDataset,
        val: UpliftDataset,
        test: UpliftDataset,
    ) -> tuple[UpliftDataset, UpliftDataset, UpliftDataset]:
        """
        Fit on train, apply to all splits.

        Parameters
        ----------
        train, val, test : UpliftDataset

        Returns
        -------
        train, val, test : UpliftDataset
            Transformed in place.
        """
        self._fit_shift(train.X)
        for ds in (train, val, test):
            ds.X = self._apply_shift(ds.X)

        train.X[self.cfg.skewed_features] = self.log_transformer.fit_transform(
            train.X[self.cfg.skewed_features]
        )
        for ds in (val, test):
            ds.X[self.cfg.skewed_features] = self.log_transformer.transform(
                ds.X[self.cfg.skewed_features]
            )
        return train, val, test

    def save(
        self,
        train: UpliftDataset,
        val: UpliftDataset,
        test: UpliftDataset,
    ) -> None:
        """Persist splits to parquet."""
        for ds, name in [(train, "train"), (val, "val"), (test, "test")]:
            df = pd.concat([ds.X, ds.treatment, ds.outcome, ds.secondary_outcome], axis=1)
            df.to_parquet(self.cfg.data_dir / f"{name}.parquet")
            logger.info("Saved %s to %s", name, self.cfg.data_dir)
    
    def calculate_weight(self, train: UpliftDataset):
        scale_pos_weight_visit = (train.outcome == 0).sum() / (train.outcome == 1).sum()
        scale_pos_weight_conv = (train.secondary_outcome == 0).sum() / (train.secondary_outcome == 1).sum()
        return scale_pos_weight_visit, scale_pos_weight_conv

    def _fit_shift(self, X: pd.DataFrame) -> None:
        """Compute and store shift constants from train set."""
        for feature in self.cfg.skewed_features:
            n_non_positive = (X[feature] <= 0).sum()
            if n_non_positive:
                self.shift_constants[feature] = abs(X[feature].min()) + 1

    def _apply_shift(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored shift constants to a feature matrix."""
        X = X.copy()
        for feature, constant in self.shift_constants.items():
            X[feature] = X[feature] + constant
        return X