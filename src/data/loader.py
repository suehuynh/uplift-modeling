"""
DataLoader for the Criteo Uplift Modeling dataset.
 
Dataset schema (Criteo):
    - f0..f11 : float features
    - treatment : int {0, 1}
    - visit     : int {0, 1}  (outcome used in this project)
    - conversion : int {0, 1} (secondary outcome)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import Config

logger = logging.getLogger(__name__)


@dataclass
class UpliftDataset:
    """
    Container for a single train/val/test split.
    """
    X: pd.DataFrame
    treatment: pd.Series
    outcome: pd.Series
    secondary_outcome: pd.Series
    split: str # "train" | "val" | "test"

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:
        tr_rate = float(self.treatment.mean()) if len(self.treatment) > 0 else 0.0
        y_rate = float(self.outcome.mean()) if len(self.outcome) > 0 else 0.0
        return (
            f"UpliftDataset(split={self.split!r}, n={len(self):,}, "
            f"treatment_rate={tr_rate:.3f}, outcome_rate={y_rate:.3f})"
        )


class DataLoader:
    """Loads, subsamples, and splits the Criteo Uplift dataset.

    Parameters
    ----------
    cfg:
        An instance of `Config` from `config/config.py`.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.data_path = cfg.data_path
        self.outcome_col = cfg.outcome_col
        self.secondary_outcome_col = cfg.secondary_outcome_col
        self.random_state = cfg.random_state
        self._df: Optional[pd.DataFrame] = None

    def load(self, subsample_size: Optional[int] = None) -> "DataLoader":
        """Load the dataset from HuggingFace and optionally subsample it.

        Parameters
        ----------
        subsample_size:
            If provided and smaller than the dataset, randomly subsample to this
            many rows. Defaults to `cfg.subsample_size` when None.
        """
        if subsample_size is None:
            subsample_size = self.cfg.subsample_size

        logger.info("Loading dataset from %s", self.data_path)
        ds = load_dataset(self.data_path)

        # Handle DatasetDict vs Dataset
        if hasattr(ds, "get") and "train" in ds:
            ds = ds["train"]

        try:
            df = ds.to_pandas()
        except Exception:
            # Fallback: try to construct DataFrame directly
            df = pd.DataFrame(ds)

        if subsample_size is not None and len(df) > subsample_size:
            df = df.sample(n=subsample_size, random_state=self.random_state).reset_index(drop=True)

        self._df = df.reset_index(drop=True)
        logger.info("Loaded %d rows, %d cols", len(self._df), self._df.shape[1])

        self._validate()
        return self

    def split(self, val_size: Optional[float] = None, test_size: Optional[float] = None) -> Tuple[UpliftDataset, UpliftDataset, UpliftDataset]:
        """Stratified train / val / test split (stratified on treatment and outcome).

        Uses `cfg.feature_cols`, `cfg.treatment_col`, and `cfg.outcome_col`.
        """
        if val_size is None:
            val_size = self.cfg.val_size
        if test_size is None:
            test_size = self.cfg.test_size

        if self._df is None:
            raise RuntimeError("Call .load() before .split()")

        X = self._df[self.cfg.feature_cols]
        treatment = self._df[self.cfg.treatment_col]
        outcome = self._df[self.cfg.outcome_col]
        secondary_outcome = self._df[self.cfg.secondary_outcome_col]

        # Build stratification labels to preserve joint distribution
        strata = treatment.astype(str) + "_" + outcome.astype(str)
        X_temp, X_test, t_temp, t_test, y_temp, y_test, y_secondary_temp, y_secondary_test = train_test_split(
            X, treatment, outcome, secondary_outcome,
            test_size=test_size,
            stratify=strata,
            random_state=self.random_state,
        )

        val_frac = val_size / (1.0 - test_size)
        strata_temp = t_temp.astype(str) + "_" + y_temp.astype(str)

        X_train, X_val, t_train, t_val, y_train, y_val, y_secondary_train, y_secondary_val = train_test_split(
            X_temp, t_temp, y_temp, y_secondary_temp,
            test_size=val_frac,
            stratify=strata_temp,
            random_state=self.random_state,
        )

        train = UpliftDataset(X_train.reset_index(drop=True), t_train.reset_index(drop=True), y_train.reset_index(drop=True), y_secondary_train.reset_index(drop=True), "train")
        val = UpliftDataset(X_val.reset_index(drop=True), t_val.reset_index(drop=True), y_val.reset_index(drop=True), y_secondary_val.reset_index(drop=True), "val")
        test = UpliftDataset(X_test.reset_index(drop=True), t_test.reset_index(drop=True), y_test.reset_index(drop=True), y_secondary_test.reset_index(drop=True), "test")

        for ds in (train, val, test):
            logger.info(ds)

        return train, val, test

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of key dataset statistics.

        Rows: overall, treated, control.
        Cols: n, treatment_rate, visit_rate, conversion_rate.
        """
        if self._df is None:
            raise RuntimeError("Call .load() first")

        df = self._df
        rows = []
        for label, mask in [
            ("overall", pd.Series([True] * len(df), index=df.index)),
            ("treated", df[self.cfg.treatment_col] == 1),
            ("control", df[self.cfg.treatment_col] == 0),
        ]:
            sub = df[mask]
            rows.append({
                "split": label,
                "n": len(sub),
                "treatment_rate": float(sub[self.cfg.treatment_col].mean()),
                "visit_rate": float(sub[self.cfg.outcome_col].mean()),
                "conversion_rate": float(sub[self.cfg.secondary_outcome_col].mean()),
            })
        return pd.DataFrame(rows).set_index("split")

    def _validate(self) -> None:
        required = set(self.cfg.feature_cols + [self.cfg.treatment_col, self.cfg.outcome_col, self.cfg.secondary_outcome_col])
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        logger.info("Schema validation passed.")

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call .load() before .data()")
        return self._df
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = Config()
    loader = DataLoader(cfg)
    loader.load()
    print(loader.summary())