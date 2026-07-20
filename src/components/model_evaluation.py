from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.logger import logging
from src.utils.main_utils import load_object, read_yaml_file
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import CustomerChurnEstimator
from dataclasses import dataclass


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[CustomerChurnEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            estimator = CustomerChurnEstimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if estimator.is_model_present(model_path=model_path):
                return estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies data cleaning and feature engineering based on the notebook.
        """
        logging.info("Starting custom feature engineering in evaluation")
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

        # 5. Drop Highly Correlated / Low Impact Columns
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

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(
                "Test data loaded and now transforming it for prediction...")

            # Map target variable (Churn) if needed
            if TARGET_COLUMN in test_df.columns:
                test_df[TARGET_COLUMN] = test_df[TARGET_COLUMN].replace({'Yes': 1, 'No': 0}).astype(int)

            test_df = self._clean_and_engineer_features(test_df)
            test_df.dropna(inplace=True)

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")

            # Align test data with trained model expected features
            if hasattr(trained_model, "preprocessing_object"):
                expected_features = getattr(trained_model.preprocessing_object, "feature_names_in_", None)
                if expected_features is not None:
                    x = x.reindex(columns=expected_features, fill_value=0)
                    logging.info("Aligned test dataset columns with trained model features")

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(
                    f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(
                f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
