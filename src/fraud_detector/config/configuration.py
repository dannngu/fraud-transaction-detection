# src/fraud_detector/config/configuration.py
from fraud_detector.constants import *
from fraud_detector.entity.config_entity import (
    ModelEvaluationConfig,  # Import your new entities
)
from fraud_detector.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
)
from fraud_detector.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)  # Load params.yaml too

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Access the 'data_ingestion' section of config.yaml
        config = self.config.data_ingestion

        # Create the root directory for ingestion artifacts
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file)
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        # Ahora usamos params.data_transformation para test_size, random_state
        params = self.params.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            processed_data_file=Path(config.processed_data_file),
            preprocessor_name=config.preprocessor_name,
            numerical_features=list(params.numerical_features), # To access for data_transformation
            categorical_features=list(params.categorical_features), # To access for data_transformation
            drop_columns=list(params.drop_columns),
            # Access directly from self.params.test_size
            test_size=float(self.params.test_size),
            # Access directly from self.params.random_state
            random_state=int(self.params.random_state)
        )

        return data_transformation_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training  # Section config.yaml
        # Logistic Regression section - params.yaml
        params = self.params.LogisticRegression
        xgb_params = self.params.XGBoost  # XGBoost section - params.yaml

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir=Path(config.root_dir),
            trained_model_name=config.trained_model_name,
            # Parameters for Logistic Regression
            solver=params.solver,
            C=float(params.C),
            class_weight=params.class_weight,
            random_state=int(params.random_state),
            max_iter=int(params.max_iter),
            # Parameters for XGBoost
            xgb_objective=xgb_params.objective,
            xgb_n_estimators=int(xgb_params.n_estimators),
            xgb_learning_rate=float(xgb_params.learning_rate),
            xgb_max_depth=int(xgb_params.max_depth),
            xgb_subsample=float(xgb_params.subsample),
            xgb_colsample_bytree=float(xgb_params.colsample_bytree)
        )

        return model_training_config

    # ... (we will add methods for training and evaluation later)
