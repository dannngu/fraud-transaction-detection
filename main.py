# main.py
from fraud_detector import logger
from fraud_detector.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from fraud_detector.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline

# Import your custom exception
from fraud_detector.exception import CustomException
import sys

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
