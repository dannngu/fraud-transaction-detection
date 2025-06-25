# src/fraud_detector/components/data_transformation.py
import os
import pandas as pd
import numpy as np
import pathlib as Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline  # We will use this for the preprocessor
from fraud_detector import logger
from fraud_detector.entity.config_entity import DataTransformationConfig
from fraud_detector.utils.common import save_bin  # To save preprocessor
from fraud_detector.exception import CustomException
import sys
# from imblearn.over_sampling import SMOTE # Uncomment if you're using SMOTE here
# from imblearn.combine import SMOTETomek # Or some combination


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        logger.info(f"Data Transformation component initialized.")

    @property
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        Creates a preprocessing pipeline using ColumnTransformer.
"       """
        numerical_cols = self.config.numerical_features
        categorical_cols = self.config.categorical_features

        numerical_pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # ColumnTransformer to apply different transformations to different columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num_pipeline', numerical_pipeline, numerical_cols),
                ('cat_pipeline', categorical_pipeline, categorical_cols)
            ],
            remainder='passthrough'  # Keeps unspecified columns
        )
        logger.info("Preprocessing pipeline (ColumnTransformer) created.")
        return preprocessor

    def initiate_data_transformation(self, data_path: Path):
        """
        Initiates the data transformation process, including data splitting,
        preprocessor tuning, and saving the preprocessed data and the preprocessor.
        """
        try:
            # Load the dataset
            df = pd.read_csv(data_path)
            logger.info(f"Loaded data for transformation from {data_path}")

            # Identify the target column and characteristics
            target_column_name = "isFraud"  # Target column for fraud

            # We ensure that the columns to be dropped exist before dropping them
            columns_to_drop_effective = [
                col for col in self.config.drop_columns if col in df.columns]

            # Create new features (Feature Engineering - you can add more depending on your EDA)
            # Make sure these columns are not in drop_columns if you're going to use them
            if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
                df['balance_diff_orig'] = df['oldbalanceOrg'] - \
                    df['newbalanceOrig']
            if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
                df['balance_diff_dest'] = df['oldbalanceDest'] - \
                    df['newbalanceDest']
            if 'amount' in df.columns:
                df['is_amount_zero'] = (df['amount'] == 0).astype(
                    int)  # Zero amount indicator

            # Update feature lists if new ones were added
            # This is key: if you create new features, they must be in numerical_features
            # (or categorical_features if they are of that type).
            # For simplicity, we assume that 'balance_diff_orig', 'balance_diff_dest', and 'is_amount_zero'
            # are already implicitly covered if the original ones were there or if we added them.
            # In a real project, self.config.numerical_features, etc., would be updated
            # after feature engineering if they weren't already passed as parameters.

            # For this example, if 'balance_diff_orig', 'balance_diff_dest', and 'is_amount_zero' are created,
            # make sure they are also added to numerical_features in params.yaml
            # or handle it dynamically here:
            if 'balance_diff_orig' in df.columns and 'balance_diff_orig' not in self.config.numerical_features:
                self.config.numerical_features.append('balance_diff_orig')
            if 'balance_diff_dest' in df.columns and 'balance_diff_dest' not in self.config.numerical_features:
                self.config.numerical_features.append('balance_diff_dest')
            if 'is_amount_zero' in df.columns and 'is_amount_zero' not in self.config.numerical_features:
                self.config.numerical_features.append('is_amount_zero')

            input_feature_df = df.drop(
                columns=[target_column_name] + columns_to_drop_effective, axis=1)
            target_feature_df = df[target_column_name]
            logger.info(
                f"Input features shape: {input_feature_df.shape}, Target feature shape: {target_feature_df.shape}")

            # Split the data into training and testing
            X_train, X_test, y_train, y_test = train_test_split(
                input_feature_df, target_feature_df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=target_feature_df  # IMPORTANTE para mantener el desbalance de clases
            )
            logger.info(
                f"Data split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets.")

            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object

            # Fit and transform the training data
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            # Only transform (do not fit) the test data
            X_test_transformed = preprocessor_obj.transform(X_test)
            logger.info(
                "Data transformation (scaling, one-hot encoding) applied to train and test sets.")

            # Optional: Apply SMOTE or other resampling techniques here if they are not used in the model pipeline
            # If you decide to use SMOTE here, you'll need to import the imbalanced-learn library
            # and add configurations in params.yaml/DataTransformationConfig
            # Example:
            # from imblearn.over_sampling import SMOTE
            # if self.config.apply_smote: # Assuming an attribute in DataTransformationConfig
            # smote = SMOTE(random_state=self.config.random_state, sampling_strategy='minority')
            # X_train_transformed, y_train = smote.fit_resample(X_train_transformed, y_train)
            # logger.info(f"SMOTE applied. New X_train shape: {X_train_transformed.shape}, y_train shape: {y_train.shape}")
            # # Also consider SMOTETomek or undersampling techniques if appropriate

            # Save the trained preprocessor
            preprocessor_path = os.path.join(
                self.config.root_dir, self.config.preprocessor_name)
            save_bin(preprocessor_obj, preprocessor_path)
            logger.info(f"Preprocessor saved at: {preprocessor_path}")

            # Save the preprocessed data if needed (for DVC or for later use)
            # Although numpy arrays are often passed directly to the next stage,
            # saving them allows you to version the intermediate state with DVC.
            # You can save them as .npy files (much better for large arrays) or CSVs.
            # For simplicity and efficiency, we'll pass the arrays directly.

            return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys)
