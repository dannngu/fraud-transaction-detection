# src/fraud_detector/components/data_ingestion.py
import os
import urllib.request as request  # # To download files from URLs
from pathlib import Path
from zipfile import ZipFile  # If your dataset is a .zip (common in Kaggle)

from fraud_detector import logger
from fraud_detector.entity.config_entity import DataIngestionConfig
from fraud_detector.utils.common import get_size  # A utility we created


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        logger.info(f"Data Ingestion component initialized.")

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            logger.info(
                f"Downloading data from {self.config.source_URL} to {self.config.local_data_file}...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(
                f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(
                f"File already exists at {self.config.local_data_file} of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):  # # This method is for datasets that come in .zip
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.root_dir
        os.makedirs(unzip_path, exist_ok=True)
        if os.path.exists(self.config.local_data_file) and self.config.local_data_file.suffix == '.zip':
            with ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info(f"Zip file extracted to {unzip_path}")
        elif self.config.local_data_file.suffix != '.zip':
            logger.info(f"File is not a zip file. Skipping extraction.")
        else:
            logger.info(
                f"Local data file not found at {self.config.local_data_file}. Skipping extraction.")
