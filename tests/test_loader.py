"""Tests for src/data/loader.py — runs on a tiny synthetic DataFrame."""

import pytest
import numpy as np
import pandas as pd

from config.config import Config
from src.data.loader import DataLoader, UpliftDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

cfg = Config()
@pytest.fixture
def tmp_csv(tmp_path):
    """Write a minimal CSV that matches the Criteo schema."""
    rng = np.random.default_rng(0)
    n = 10_000
    df = pd.DataFrame(
        rng.standard_normal((n, 12)), columns=cfg.feature_cols
    )
    df["treatment"]  = rng.integers(0, 2, size=n)
    df["visit"]      = rng.integers(0, 2, size=n)
    df["conversion"] = rng.integers(0, 2, size=n)
    path = tmp_path / "criteo_mock.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------

class TestDataLoader:

    def test_load_full(self, tmp_csv):
        loader = DataLoader(tmp_csv)
        loader.load()
        assert loader._df is not None
        assert len(loader._df) == 10_000

    def test_load_subsample(self, tmp_csv):
        loader = DataLoader(tmp_csv).load(n_rows=1_000)
        assert len(loader._df) == 1_000

    def test_split_sizes(self, tmp_csv):
        loader = DataLoader(tmp_csv).load(n_rows=5_000)
        train, val, test = loader.split(val_size=0.1, test_size=0.1)
        total = len(train) + len(val) + len(test)
        assert total == 5_000
        assert len(test) == pytest.approx(500, abs=50)
        assert len(val)  == pytest.approx(450, abs=60)

    def test_split_returns_uplift_datasets(self, tmp_csv):
        loader = DataLoader(tmp_csv).load(n_rows=2_000)
        for ds in loader.split():
            assert isinstance(ds, UpliftDataset)

    def test_treatment_binary(self, tmp_csv):
        loader = DataLoader(tmp_csv).load()
        train, _, _ = loader.split()
        assert set(train.treatment.unique()).issubset({0, 1})

    def test_summary_returns_dataframe(self, tmp_csv):
        loader = DataLoader(tmp_csv).load()
        summary = loader.summary()
        assert isinstance(summary, pd.DataFrame)
        assert set(summary.index) == {"overall", "treated", "control"}

    def test_load_before_split_raises(self, tmp_csv):
        loader = DataLoader(tmp_csv)
        with pytest.raises(RuntimeError):
            loader.split()