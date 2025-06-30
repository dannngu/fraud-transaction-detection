
import os
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_detector import logger
from fraud_detector.components.model_trainer import ModelTrainer
from fraud_detector.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()

        # Load the preprocessed data saved by the previous step
        # To get the path where it was saved
        data_transformation_config = config.get_data_transformation_config()
        X_train = np.load(os.path.join(
            data_transformation_config.root_dir, "X_train.npy"))
        X_test = np.load(os.path.join(
            data_transformation_config.root_dir, "X_test.npy"))
        y_train = np.load(os.path.join(
            data_transformation_config.root_dir, "y_train.npy"))
        y_test = np.load(os.path.join(
            data_transformation_config.root_dir, "y_test.npy"))

        # The path to the preprocessor is also required
        preprocessor_path = os.path.join(
            data_transformation_config.root_dir, data_transformation_config.preprocessor_name)

        model_trainer = ModelTrainer(config=model_training_config)
        model_trainer.train(X_train=X_train, y_train=pd.Series(
            y_train), preprocessor_path=Path(preprocessor_path))


"""
Python will start reading from here only
"""
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")

    except Exception as e:
        logger.exception(e)
        raise e
