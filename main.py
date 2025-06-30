# main.py
import sys

from fraud_detector import logger

# Import your custom exception
from fraud_detector.exception import CustomException
from fraud_detector.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from fraud_detector.pipeline.stage_02_data_transformation import (
    DataTransformationTrainingPipeline,
)
from fraud_detector.pipeline.stage_03_model_training import ModelTrainingPipeline

# Name of the current stage for the log
# --- Stage 1: Data Ingestion ---
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(
        f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)  # Log the full exception
    raise CustomException(e, sys)  # Re-throws the custom exception


# --- Stage 2: Data Transformation ---
STAGE_NAME_TRANSFORMATION = "Data Transformation Stage"
try:
    logger.info(f"********************")
    logger.info(f">>>>>> Stage {STAGE_NAME_TRANSFORMATION} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>> Stage {STAGE_NAME_TRANSFORMATION} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)

# --- Stage 3: Model training  ---

STAGE_NAME_TRAINING = "Model Training Stage"
try:
    logger.info(f"********************")
    logger.info(f">>>>>> Stage {STAGE_NAME_TRAINING} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(
        f">>>>>> Stage {STAGE_NAME_TRAINING} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise CustomException(e, sys)
