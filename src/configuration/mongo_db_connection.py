from src.constants import *
import os
import sys
from dotenv import load_dotenv
from src.exception import MyException
import certifi
from src.logger import logging
import pymongo
from src.constants import DATABASE_NAME
ca = certifi.where()


class MongoDBClient:
    client = None

    def __init__(self, database: str = DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                load_dotenv()
                db_url = os.getenv("DB_URL")
                if db_url is None:
                    raise Exception("DB_URL is not set")
                MongoDBClient.client = pymongo.MongoClient(
                    db_url, tlsCAFile=ca)

            self.client = MongoDBClient.client
            self.database = self.client[database]
            self.db_name = database
            logging.info("MongoDB connected sucessfully")

        except Exception as e:
            raise MyException(e, sys)
