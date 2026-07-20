import sys
from src.entity.config_entity import CustomerChurnPredictorConfig
from src.entity.s3_estimator import CustomerChurnEstimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame
import pandas as pd
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH

class CustomerChurnData:
    def __init__(self,
                 gender,
                 SeniorCitizen,
                 Partner,
                 Dependents,
                 tenure,
                 PhoneService,
                 MultipleLines,
                 InternetService,
                 OnlineSecurity,
                 OnlineBackup,
                 DeviceProtection,
                 TechSupport,
                 StreamingTV,
                 StreamingMovies,
                 Contract,
                 PaperlessBilling,
                 PaymentMethod,
                 MonthlyCharges,
                 TotalCharges,
                 customerID=None
                 ):
        """
        Customer Churn Data constructor
        Input: all features required for the model prediction
        """
        try:
            self.customerID = customerID
            self.gender = gender
            self.SeniorCitizen = SeniorCitizen
            self.Partner = Partner
            self.Dependents = Dependents
            self.tenure = tenure
            self.PhoneService = PhoneService
            self.MultipleLines = MultipleLines
            self.InternetService = InternetService
            self.OnlineSecurity = OnlineSecurity
            self.OnlineBackup = OnlineBackup
            self.DeviceProtection = DeviceProtection
            self.TechSupport = TechSupport
            self.StreamingTV = StreamingTV
            self.StreamingMovies = StreamingMovies
            self.Contract = Contract
            self.PaperlessBilling = PaperlessBilling
            self.PaymentMethod = PaymentMethod
            self.MonthlyCharges = MonthlyCharges
            self.TotalCharges = TotalCharges

        except Exception as e:
            raise MyException(e, sys) from e

    def get_customer_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from CustomerChurnData class input
        """
        try:
            customer_input_dict = self.get_customer_data_as_dict()
            return DataFrame(customer_input_dict)

        except Exception as e:
            raise MyException(e, sys) from e

    def get_customer_data_as_dict(self):
        """
        This function returns a dictionary from CustomerChurnData class input
        """
        logging.info(
            "Entered get_customer_data_as_dict method of CustomerChurnData class")

        try:
            input_data = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges]
            }
            
            if self.customerID is not None:
                input_data["customerID"] = [self.customerID]

            logging.info("Created customer data dict")
            logging.info(
                "Exited get_customer_data_as_dict method of CustomerChurnData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class CustomerChurnClassifier:
    def __init__(self, prediction_pipeline_config: CustomerChurnPredictorConfig = CustomerChurnPredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for predicting the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def _clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies data cleaning and feature engineering based on the notebook.
        """
        logging.info("Starting custom feature engineering in prediction pipeline")
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

    def predict(self, dataframe) -> str:
        """
        This is the method of CustomerChurnClassifier
        Returns: Prediction
        """
        try:
            logging.info(
                "Entered predict method of CustomerChurnClassifier class")
            model = CustomerChurnEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            
            # 1. Apply custom feature engineering
            dataframe = self._clean_and_engineer_features(dataframe)
            
            # 2. Align dataframe with model expected features
            # When get_dummies is applied on a single row, it misses categories not present in the row.
            # We must reindex to match the columns the preprocessor expects.
            loaded_model_obj = model.load_model()
            if hasattr(loaded_model_obj, "preprocessing_object"):
                expected_features = getattr(loaded_model_obj.preprocessing_object, "feature_names_in_", None)
                if expected_features is not None:
                    dataframe = dataframe.reindex(columns=expected_features, fill_value=0)
                    logging.info("Aligned prediction dataframe columns with trained model features")

            # 3. Predict
            result = model.predict(dataframe)

            return result

        except Exception as e:
            raise MyException(e, sys)
