import os

import numpy as np

from fraud_detector import logger
from fraud_detector.components.data_transformation import DataTransformation
from fraud_detector.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(
            config=data_transformation_config)

        # The path to the input data file must come from the ingestion stage
        # For simplicity, we take it directly from config.yaml/data_ingestion.local_data_file
        # In a real DVC pipeline, this would be handled by DVC
        # To get the path of the ingested file
        data_ingestion_config = config.get_data_ingestion_config()
        X_train, X_test, y_train, y_test, preprocessor_path = \
            data_transformation.initiate_data_transformation(
                data_ingestion_config.local_data_file)

        # Now that the variables are defined, we save them.
        # The preprocessor is saved in its already defined location in the component.
        # And the transformed data in .npy files to pass them on to the next stage.
        np.save(os.path.join(
            data_transformation_config.root_dir, "X_train.npy"), X_train)
        np.save(os.path.join(
            data_transformation_config.root_dir, "X_test.npy"), X_test)
        np.save(os.path.join(
            data_transformation_config.root_dir, "y_train.npy"), y_train)
        np.save(os.path.join(
            data_transformation_config.root_dir, "y_test.npy"), y_test)


"""
Python will start reading from here only
"""
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(
            f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx========x")

    except Exception as e:
        logger.exception(e)
        raise e


