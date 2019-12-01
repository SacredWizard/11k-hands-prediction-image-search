"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is the file for interacting with CSV Files
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import json

import pandas
import csv
import numpy as np
from os import path, makedirs
from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper
import utils.termweight as tw


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

    def save_csv_multiple(self, input_data):
        """
        Reads the csv files and saves it to the collection
        :param input_data: a json of the form {"collectionName": ["filename1.csv", "filename2.csv"]}
        :return:
        """
        for inp in input_data:
            self.mongo_wrapper.drop_collection(inp)  # Drop Collection
            for filename in input_data[inp]:
                data = pandas.read_csv(filename)
                data_json = json.loads(data.to_json(orient='records'))
                self.mongo_wrapper.bulk_insert(inp, data_json)  # Insert new Metadata

    # method to format rows to output to csv
    def prepare_rows(self, latent_semantics):
        round_digits = 6
        result = []
        for i,ls in enumerate(latent_semantics):
            term_weight = {}
            for x in ls:
                term_weight[x[0]] = round(x[1], round_digits)
            result.append(("LS"+str(i+1)+", "+str(term_weight)[1:-1]).split(','))
        return result

    # method to save latent semantics to csv
    def save_to_csv(self, data_latent_semantics, feature_latent_semantics, filename, subject_subject=False,
                    image_metadata=False):
        current_path = path.dirname(path.dirname(path.realpath(__file__)))
        _finalPath = path.join(current_path,"output")
        if not path.exists(_finalPath):
            makedirs(_finalPath)
        images = data_latent_semantics['imageId'].tolist()
        data_latent_semantics = np.array(data_latent_semantics['reducedDimensions'].tolist())
        data_tw = tw.get_data_latent_semantics(data_latent_semantics, data_latent_semantics.shape[1], images)

        with open(path.join(_finalPath, filename+".csv"), mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            if subject_subject:
                csv_writer.writerow(["Top-k Latent Semantics"])
            elif image_metadata:
                csv_writer.writerow(["LS in Image Space"])
            else:
                csv_writer.writerow(["Data Latent Semantics"])
            csv_writer.writerows(self.prepare_rows(data_tw))
            if not subject_subject:
                feature_tw = tw.get_feature_latent_semantics(feature_latent_semantics, feature_latent_semantics.
                                                             shape[0], image_metadata=image_metadata)
                if image_metadata:
                    csv_writer.writerow(["LS in Metadata Space"])
                else:
                    csv_writer.writerow(["Feature Latent Semantics"])
                csv_writer.writerows(self.prepare_rows(feature_tw))
