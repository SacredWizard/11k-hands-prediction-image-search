import pymongo
import pymongo.errors

from classes import global_constants


class GlobalConnections:

    def __init__(self, dbname=None):
        self.constants = global_constants.GlobalConstants().Mongo()
        try:
            if dbname:
                self.mongo_client = pymongo.MongoClient(
                    self.constants.MONGO_URL, serverSelectionTimeoutMS=self.constants.MONGO_SERVER_TIMEOUT)\
                    .get_database(dbname)
            else:
                self.mongo_client = pymongo.MongoClient(
                    self.constants.MONGO_URL, serverSelectionTimeoutMS=self.constants.MONGO_SERVER_TIMEOUT)\
                    .get_database(self.constants.DB_NAME)
        except pymongo.errors.ConnectionFailure as e:
            print("Connection Failure:\n{}".format(e))
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print("Timeout:\n{}".format(e))
        except Exception as e:
            print("Exception has occurred:\n{}".format(e))

    def saverecord(self, collection_name, rec):
        try:
            self.mongo_client.features[collection_name].update(
                {"imageId": "{}".format(rec.imageId)}, {"$set": rec}, upsert=True)
        except Exception as e:
            print("Exception while saving record:\n{}".format(e))

    def bulk_insert(self, collection, records, parallel=False, threads=4):
        try:
            self.mongo_client[collection].insert_many(records)
        except pymongo.errors.BulkWriteError as e:
            print("Bulk Write Error:\n{}".format(e))
        except Exception as e:
            print("Error: {}".format(e))
