"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is the file for interacting with CSV Files
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import pandas
from classes.mongo import MongoWrapper
from classes.global_constants import GlobalConstants
import json


class CSVReader:
    def __init__(self):
        self.constants = GlobalConstants()
        self.mongo_wrapper = MongoWrapper(self.constants.Mongo().DB_NAME)
        pass

    def save_hand_csv_mongo(self, filename):
        """Reads the HandsInfo CSV and saves it to Mongo collection Metadata"""
        data = pandas.read_csv(filename)
        data_json = json.loads(data.to_json(orient='records'))
        self.mongo_wrapper.drop_collection(self.constants.METADATA)  # Drop Metadata Collection
        self.mongo_wrapper.bulk_insert(self.constants.METADATA, data_json)  # Insert new Metadata
