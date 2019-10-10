"""
Image Search - Multimedia Web Database Systems Fall 2019 Project
This is the CLI for the Task3 of Phase 2 of the Project
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import pandas
from classes.mongo import MongoWrapper
from classes.global_constants import GlobalConstants
import json


class ExcelReader:
    def __init__(self):
        self.constants = GlobalConstants()
        self.mongo_wrapper = MongoWrapper(self.constants.Mongo().DB_NAME)
        pass

    def save_hand_csv_mongo(self, filename):
        """Reads the HandsInfo CSV and saves it to Mongo collection Metadata"""
        data = pandas.read_csv(filename)
        data_json = json.loads(data.to_json(orient='records'))
        self.mongo_wrapper.drop_collection("metadata")  # Drop Metadata Collection
        self.mongo_wrapper.bulk_insert("metadata", data_json)  # Insert new Metadata
