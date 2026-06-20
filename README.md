# Uplift Modeling: Budget-Constrained Targeting on Criteo

> This project designs targeting policies that maximize incremental impact, not just predicted conversion.

## Executive Briefing: CATE-Driven Budget Allocation

* **The Challenge:** Deploying a mass marketing campaign across 25M rows under strict budget constraints causes massive financial leakage if managed traditionally. A baseline strategy of random targeting yields a devastating **-96% ROI**. Meanwhile, traditional propensity models fail because they waste spend on "sure things" (users who would have converted anyway without an incentive).
* **The Solution:** We translated individual Conditional Average Treatment Effect (CATE) point estimates from a tuned **Causal Forest** into an optimized, budget-constrained targeting policy object (`TargetingPolicy`). 
* **The Business Impact:** Our policy extracts **2,291 incremental conversions** out of a \$10,000 budget, outperforming random allocation by **5.7x** and dropping the campaign’s structural breakeven floor to **\$4.36 per conversion**. For high-value customer tiers where an outcome is worth \$5.00, this shifts the campaign from a massive loss-leader into a profitable engine yielding a **+14.5% net ROI** at scale, and up to **+183% ROI** on conservative, highly surgical spends.

## Overview

An end-to-end uplift modeling pipeline built on the Criteo Uplift dataset (~25M rows), estimating heterogeneous treatment effects to inform budget-constrained marketing targeting decisions. Rather than predicting who is most likely to convert, this project identifies who is most likely to convert *because* of treatment, the "persuadables" segment that drives incremental value from a marketing intervention.

Four causal inference approaches are implemented and rigorously compared on held-out test data: T-Learner, S-Learner, X-Learner, and Causal Forest.

## Results

### 1. Algorithmic Ranking Performance
| Model | AUUC | Notes |
|---|---|---|
| **Causal Forest** | **1194.17** | Best performer; direct heterogeneity-optimized splitting |
| X-Learner | 1149.64 | Corrects for treatment/control group imbalance via cross-imputation |
| S-Learner | 880.67 | Single model with treatment as feature |
| T-Learner | 723.44 | Baseline meta-learner; weakest under group imbalance |

### 2. Policy Simulation Framework (Fixed Budget = \$10,000, Cost/Contact = \$0.18)
| Strategy | N Targeted | Total Value Gained (Uplift) | Total Cost | ROI (at $V=\$1.00$) |
| :--- | :--- | :--- | :--- | :--- |
| **CATE (Causal Forest)** | 55,555 | **2,291.44** | \$9,999.90 | **-77.09%** |
| Propensity Model | 55,555 | 2,167.48 | \$9,999.90 | -78.33% |
| Random Selection | 55,555 | 399.46 | \$9,999.90 | -96.01% |


All four models substantially outperform random targeting on the held-out test set.
> See `notebooks/04_evaluation.ipynb` or `results.png` for full Qini curve analysis and discussion.
> See `notebooks/05_business_simulation.ipynb` for budget sensitivity charts and joint frontier optimization plots.

## Key Findings

- **Randomization quality validated**: covariate balance and propensity score overlap analysis confirmed the underlying experiment was well-randomized (ROC-AUC ≈ 0.51 for treatment prediction from covariates)
- **T-Learner's independent-model design is vulnerable to group imbalance**: with an 85/15 treatment/control split, the control model's reduced sample size introduced measurable bias
- **X-Learner's cross-imputation directly corrects this weakness**, producing the closest average uplift estimate to the empirical ATE among all meta-learners
- **Causal Forest outperforms all meta-learners** when leaf size is tuned relative to outcome rarity (a leaf of 450+ samples is needed for stable estimates at a ~4.7% positive outcome rate)
- **A critical scale_pos_weight calibration bug** was diagnosed in X-Learner: class-imbalance correction in stage-1 models, while necessary for T-Learner and S-Learner, directly corrupts X-Learner's cross-imputed targets since there is no second model to cancel the bias against
- **Propensity targeting triggers marketing cannibalization:** Prioritizing users based on baseline conversion probability underperforms the CATE policy, proving that standard ML models waste budget by targeting organic converts who do not require an incentive.
- **The Efficiency Frontier exhibits distinct diminishing returns:** Sensitivity analyses show that a 10x expansion in marketing spend (from \$1k to \$10k) yields only a 4x increase in incremental lift. This forces the true breakeven cost per conversion up from **\$1.77** to **\$4.36**, establishing an explicit operational roadmap for campaign profitability based on product margins.

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
│   ├── 04_evaluation.ipynb
│   └── 05_business_simulation.ipynb # Translates CATE predictions into budget allocation policies
├── tests/                      # pytest unit tests for all core components
├── requirements.txt
└── README.md
```

## Dataset

This project leverages the **Criteo Uplift Prediction Dataset v2** (~25M rows). To optimize local memory allocation and stream data seamlessly, the pipeline utilizes the memory-mapped Hugging Face repository at `criteo/criteo-uplift`.

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
