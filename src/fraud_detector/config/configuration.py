# src/fraud_detector/config/configuration.py
from fraud_detector.constants import *
from fraud_detector.utils.common import read_yaml, create_directories
from fraud_detector.entity.config_entity import (DataIngestionConfig,
                                                 DataTransformationConfig,
                                                 ModelTrainingConfig,
                                                 ModelEvaluationConfig)  # Import your new entities


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
            numerical_features=list(params.numerical_features),
            categorical_features=list(params.categorical_features),
            drop_columns=list(params.drop_columns),
            # Access directly from self.params.test_size
            test_size=float(self.params.test_size),
            # Access directly from self.params.random_state
            random_state=int(self.params.random_state)
        )

        return data_transformation_config

    # ... (we will add methods for training and evaluation later)
