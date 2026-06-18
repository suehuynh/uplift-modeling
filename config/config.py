class Config:
    # General
    random_state: int = 42
    
    # Data
    data_path: str = "criteo/criteo-uplift"
    from pathlib import Path
    data_dir: Path = Path(__file__).parent.parent / "processed"

    subsample_size: int = 2000000
    test_size: float = 0.2
    val_size: float = 0.1

    feature_cols = [f"f{i}" for i in range(12)]
    treatment_col = "treatment"
    outcome_col = "visit"
    secondary_outcome_col = "conversion"

    # XGBClassifier Parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': random_state
    }

    # XGBRegressor Parameters
    xgbr_params = {
        'objective': 'reg:squarederror',
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': random_state
    }

    # Features
    skewed_features = ["f1", "f3", "f4", "f5", "f7", "f8", "f9", "f10", "f11"]

    # Logging
    wandb_project: str = "uplift-modeling"

