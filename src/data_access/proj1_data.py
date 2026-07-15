from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
import sys
from typing import Optional
from src.exception import MyException
import pandas as pd
import numpy as np


class ProjData:
    def __init__(self) -> None:
        try:
            self.mongo_client = MongoDBClient(database=DATABASE_NAME)

        except Exception as e:
            raise MyException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]

            else:
                collection = self.mongo_client.client[database_name][collection_name]

            print("Fetching data from mongoDB")
            df = pd.DataFrame(list(collection.find()))
            print(f"Data fetched with len: {len(df)}")
            if 'customerID' in df.columns.to_list():
                df.drop(columns=['customerID'], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise MyException(e, sys)
