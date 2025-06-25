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
    # We will add this entity later
    pass


@dataclass(frozen=True)
class ModelEvaluationConfig:
    # We will add this entity later
    pass
