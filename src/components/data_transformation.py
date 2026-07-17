import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data.
        As per the notebook, we apply StandardScaler to TotalCharges.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            numeric_transformer = StandardScaler()
            
            # The notebook applies StandardScaler to TotalCharges.
            # Other features engineered are kept as is.
            numeric_features = ['TotalCharges']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, numeric_features)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method")
            raise MyException(e, sys) from e

    def _clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies data cleaning and feature engineering based on the notebook.
        """
        logging.info("Starting custom feature engineering from notebook")
        df = df.copy()

        # 1. TotalCharges cleaning
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')

        # 2. Map Categorical Variables
        mapping_dict = {'Yes': 1, 'No': 0, 'No internet service': 2, 'No phone service': 2}
        
        if 'gender' in df.columns:
            df['gender'] = df['gender'].replace({'Female': 0, 'Male': 1}).astype(float)
            
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].replace(mapping_dict).astype(float)
                
        ternary_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        for col in ternary_cols:
            if col in df.columns:
                df[col] = df[col].replace(mapping_dict).astype(float)

        # 3. Create Dummy Columns
        dummy_cols = ['Contract', 'InternetService', 'PaymentMethod']
        for col in dummy_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                df = df.join(dummies)
                df.drop(columns=[col], inplace=True)
                
        # 4. Feature Engineering
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr'])
            tenuredf = pd.get_dummies(df['tenure_group'], prefix='tenuregroup', dtype=int)
            df = df.join(tenuredf)
            df.drop(columns=['tenure_group'], inplace=True)
            
            df['is_new_customer'] = (df['tenure'] <= 6).astype(int)

        if 'TotalCharges' in df.columns and 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df['avg_monthly_charge'] = df['TotalCharges'] / df['tenure'].replace(0, 1)
            df['charge_diff'] = df['MonthlyCharges'] - df['avg_monthly_charge']

        service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        available_services = [c for c in service_cols if c in df.columns]
        if available_services:
            df['total_services'] = df[available_services].sum(axis=1)

        addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        available_addons = [c for c in addon_cols if c in df.columns]
        if available_addons:
            df['no_addon_services'] = (df[available_addons].sum(axis=1) == 0).astype(int)

        # 5. Drop Highly Correlated / Low Impact Columns (from notebook analysis)
        columns_to_drop = [
            'customerID', 'tenuregroup_4-6yr', 'total_services', 'gender', 
            'charge_diff', 'tenuregroup_1-2yr', 'PhoneService', 'MultipleLines'
        ]
        
        # Also drop drop_columns configured in schema
        schema_drop = self._schema_config.get('drop_columns')
        if schema_drop and isinstance(schema_drop, str):
            columns_to_drop.append(schema_drop)
        elif schema_drop and isinstance(schema_drop, list):
            columns_to_drop.extend(schema_drop)
            
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
            
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            # Map target variable (Churn)
            if TARGET_COLUMN in train_df.columns:
                train_df[TARGET_COLUMN] = train_df[TARGET_COLUMN].replace({'Yes': 1, 'No': 0}).astype(int)
            if TARGET_COLUMN in test_df.columns:
                test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].replace({'Yes': 1, 'No': 0}).astype(int)

            # Apply custom feature engineering
            train_df = self._clean_and_engineer_features(train_df)
            test_df = self._clean_and_engineer_features(test_df)
            logging.info("Custom transformations applied to train and test data")

            # Align test set columns to match train set (fixes get_dummies differences)
            test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
            logging.info("Aligned test dataset columns with train dataset")

            # Drop missing values (TotalCharges coerce introduces NaNs for empty strings)
            train_df.dropna(inplace=True)
            test_df.dropna(inplace=True)
            logging.info("Dropped NaN values from datasets")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority", random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            # Test data MUST NOT be resampled to avoid data leakage / distorted evaluation
            input_feature_test_final = input_feature_test_arr
            target_feature_test_final = target_feature_test_df
            logging.info("SMOTEENN applied to training df only.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(
                self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e

