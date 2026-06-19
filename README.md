# Uplift Modeling: Budget-Constrained Targeting on Criteo

> This project designs targeting policies that maximize incremental impact, not just predicted conversion.

## Overview

An end-to-end uplift modeling pipeline built on the Criteo Uplift dataset (~25M rows), estimating heterogeneous treatment effects to inform budget-constrained marketing targeting decisions. Rather than predicting who is most likely to convert, this project identifies who is most likely to convert *because* of treatment, the "persuadables" segment that drives incremental value from a marketing intervention.

Four causal inference approaches are implemented and rigorously compared on held-out test data: T-Learner, S-Learner, X-Learner, and Causal Forest.

## Results

| Model | AUUC | Notes |
|---|---|---|
| **Causal Forest** | **1194.17** | Best performer; direct heterogeneity-optimized splitting |
| X-Learner | 1149.64 | Corrects for treatment/control group imbalance via cross-imputation |
| S-Learner | 880.67 | Single model with treatment as feature |
| T-Learner | 723.44 | Baseline meta-learner; weakest under group imbalance |

All four models substantially outperform random targeting on the held-out test set. See `notebooks/04_evaluation.ipynb` or `results.png` for full Qini curve analysis and discussion.

## Key Findings

- **Randomization quality validated**: covariate balance and propensity score overlap analysis confirmed the underlying experiment was well-randomized (ROC-AUC ≈ 0.51 for treatment prediction from covariates)
- **T-Learner's independent-model design is vulnerable to group imbalance**: with an 85/15 treatment/control split, the control model's reduced sample size introduced measurable bias
- **X-Learner's cross-imputation directly corrects this weakness**, producing the closest average uplift estimate to the empirical ATE among all meta-learners
- **Causal Forest outperforms all meta-learners** when leaf size is tuned relative to outcome rarity (a leaf of 450+ samples is needed for stable estimates at a ~4.7% positive outcome rate)
- **A critical scale_pos_weight calibration bug** was diagnosed in X-Learner: class-imbalance correction in stage-1 models, while necessary for T-Learner and S-Learner, directly corrupts X-Learner's cross-imputed targets since there is no second model to cancel the bias against

## Project Structure
```
uplift-modeling/

├── config/
│   └── config.py              # Single source of truth for paths, hyperparameters, random_state
├── src/
│   ├── data/
│   │   ├── loader.py           # DataLoader, UpliftDataset
│   │   └── preprocessor.py     # Preprocessor: log-transform, train/val/test split
│   ├── models/
│   │   ├── base.py             # UpliftModel abstract base class
│   │   ├── t_learner.py
│   │   ├── s_learner.py
│   │   ├── x_learner.py
│   │   └── causal_forest.py
│   └── evaluation/
│       └── evaluator.py        # Qini curve, AUUC, comparison plots
├── results/
│   ├── results.png
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── tests/                      # pytest unit tests for all core components
├── requirements.txt
└── README.md
```

## Dataset

Criteo Uplift Prediction Dataset v2 (~25M rows). Download from [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and place the CSV at `data/criteo-uplift-v2.1.csv` before running `01_eda.ipynb`.

### Setup

```bash
git clone https://github.com/suehuynh/uplift-modeling
cd uplift-modeling
pip install -e .
pytest tests/ -v
```

### Tech Stack

Python, EconML, XGBoost, scikit-learn, pandas, matplotlib, pytest

### Author
Sue Huynh | [Portfolio](https://suehuynh.framer.website/) | [LinkedIn](https://www.linkedin.com/in/sue-huynh/) | [GitHub](https://github.com/suehuynh)
