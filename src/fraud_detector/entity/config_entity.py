# src/fraud_detector/entity/config_entity.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str  # URL dataset (e.g. .csv)
    local_data_file: Path  # Path where the dataset will be saved locally


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    processed_data_file: Path
    preprocessor_name: str
    numerical_features: list
    categorical_features: list
    drop_columns: list
    test_size: float  # We add these here so that they are in the transformation entity
    random_state: int


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    trained_model_name: str
    # Parameters for Logistic Regression
    solver: str
    C: float
    class_weight: str  # O dict manual weights
    random_state: int
    max_iter: int
    # Paramters for XGBoost
    xgb_objective: str
    xgb_n_estimators: int
    xgb_learning_rate: float
    xgb_max_depth: int
    xgb_subsample: float
    xgb_colsample_bytree: float
    # Add more parameters for XGBoost here


@dataclass(frozen=True)
class ModelEvaluationConfig:
    # We will add this entity later
    pass
