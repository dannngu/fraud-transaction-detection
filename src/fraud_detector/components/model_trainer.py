# src/fraud_detector/components/model_trainer.py
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb  # Import XGBoost model

# Import ColumnTransformer if the preprocessor is included in the same training pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# To create a complete pipeline (preprocessor + model)
from sklearn.pipeline import Pipeline

from fraud_detector import logger
from fraud_detector.entity.config_entity import ModelTrainingConfig
from fraud_detector.exception import CustomException

# To save the trained model
from fraud_detector.utils.common import load_bin, save_bin


class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        logger.info(f"Model Training component initialized.")

    def train(self, X_train: np.ndarray, y_train: pd.Series, preprocessor_path: Path):
        try:
            # Validate the inputs
            if X_train is None or y_train is None:
                raise ValueError("Training data cannot be None")

            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(
                f"Target distribution: {y_train.value_counts().to_dict()}")

            # Load the saved preprocessor
            preprocessor_obj = load_bin(preprocessor_path)
            logger.info(
                f"Preprocessor loaded successfully from {preprocessor_path}")

            # --- Model 1: Logistic Regression ---
            logger.info("Training Logistic Regression model...")
            # The preprocessor was already applied in data_transformation, so X_train is already transformed.
            # Here, the pipeline would be just the classifier if X_train is already preprocessed.
            # For an end-to-end pipeline that includes the preprocessor, then
            # X_train and X_test would be the original data frames, and the preprocessor would be applied here.

            # NOTE: X_train already transformed (as returned by DataTransformation)
            # In this case, the training pipeline would only be the classifier:
            lr_classifier = LogisticRegression(
                solver=self.config.solver,
                C=self.config.C,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter
            )
            # Train with already transformed data
            lr_classifier.fit(X_train, y_train)
            logger.info("Logistic Regression model trained.")

            # Save only the Logistic Regression classifier
            lr_model_path = os.path.join(
                self.config.root_dir, "logistic_regression_model.joblib")
            save_bin(lr_classifier, lr_model_path)
            logger.info(f"Logistic Regression model saved at: {lr_model_path}")

            # --- Model 2: XGBoost ---
            # Important: For XGBoost, if you use scale_pos_weight, calculate the class proportion
            # Calculating scale_pos_weight for unbalanced datasets
            num_positive = y_train.sum()
            num_negative = len(y_train) - num_positive
            scale_pos_weight_value = num_negative / num_positive if num_positive > 0 else 1
            logger.info(
                f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight_value:.4f}")

            xgb_classifier = xgb.XGBClassifier(
                objective=self.config.xgb_objective,
                n_estimators=self.config.xgb_n_estimators,
                learning_rate=self.config.xgb_learning_rate,
                max_depth=self.config.xgb_max_depth,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                random_state=self.config.random_state,
                scale_pos_weight=scale_pos_weight_value # comment or uncomment and pass if calculated
            )
            logger.info("Training XGBoost model...")
            # Train with already transformed data
            xgb_classifier.fit(X_train, y_train)
            logger.info("XGBoost model trained.")

            # Save the XGBoost classifier
            xgb_model_path = os.path.join(
                self.config.root_dir, "xgboost_model.joblib")
            save_bin(xgb_classifier, xgb_model_path)
            logger.info(f"XGBoost model saved at: {xgb_model_path}")

            # NOTE: For prediction, you'll need to know which model you'll use.
            # You could decide here, or use a specific model in the prediction pipeline.
            # For now, we'll save both, and the evaluation stage will choose or evaluate both.

            # Also save the preprocessor in the models directory for inference
            preprocessor_copy_path = os.path.join(
                self.config.root_dir, "preprocessor.joblib")
            save_bin(preprocessor_obj, preprocessor_copy_path)
            logger.info(f"Preprocessor copied to: {preprocessor_copy_path}")

            return {
                'logistic_regression_path': lr_model_path,
                'xgboost_path': xgb_model_path,
                'preprocessor_path': preprocessor_copy_path
            }

        except Exception as e:
            raise CustomException(e, sys)
