import sys

import pymongo
import pymongo.errors

from classes import globalconstants


class MongoWrapper:

    def __init__(self, dbname=None):
        """
        Init Method, Initialize the connection and connect to the given database, else read the database name from
        the constants file and use it
        :param dbname: Optional Database name
        """
        self.constants = globalconstants.GlobalConstants().Mongo()
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
            sys.exit(1)
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print("Timeout:\n{}".format(e))
            sys.exit(1)
        except Exception as e:
            print("Exception has occurred:\n{}".format(e))
            sys.exit(1)

    def get_db_connection(self):
        """
        Get the database Connection
        :return: Mongo Client
        """
        return self.mongo_client

    def distinct(self, collection_name, field, query):
        """
        Run a distinct query on a collection
        :param collection_name: the collection on which you want to run the distinct query
        :param field: Field for which you want to get the distinct values
        :param query: Query for filtering the documents
        :return: distinct values for the field
        """
        try:
            return self.mongo_client[collection_name].distinct(field, query)
        except pymongo.errors.ExecutionTimeout as e:
            print("Timeout : {}".format(e))
            sys.exit(1)
        except Exception as e:
            print("Exception occured while running mongo distinct query: {}".format(e))
            sys.exit(1)

    def find(self, collection_name, query, fields_filter=None, count=None):
        """
        Find records in Mongo
        :param collection_name: Collection Name
        :param query: Query in JSON
        :param fields_filter: Filter for the output
        :return: Mongo Cursor
        """
        try:
            if query == '':
                if count and count > 0:
                    return self.mongo_client[collection_name].find().limit(count)
                else:
                    return self.mongo_client[collection_name].find()
            else:
                if count and count > 0:
                    return self.mongo_client[collection_name].find(query, fields_filter).limit(count)
                else:
                    return self.mongo_client[collection_name].find(query, fields_filter)
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print("Timeout:\n{}".format(e))
            sys.exit(1)
        except Exception as e:
            print("Exception occurred:\n{}".format(e))
            sys.exit(1)

    def save_record(self, collection_name, rec):
        """
        Save record to mongo collection
        :param collection_name: Collection Name
        :param rec: Record to save
        :return: Object Id
        """
        try:
            return self.mongo_client.features[collection_name].update(
                {"imageId": "{}".format(rec.imageId)}, {"$set": rec}, upsert=True)
        except Exception as e:
            print("Exception while saving record:\n{}".format(e))
            sys.exit(1)

    def bulk_insert(self, collection, records, parallel=False, threads=4):
        """
        Bulk Insert Records in to Mongo
        :param collection: Collection Name
        :param records: Array of Dicts {}, {}
        :param parallel: Boolean for parallel Execution (Future Development)
        :param threads: Number of threads for parallel Execution (Future Development)
        :return:
        """
        try:
            self.mongo_client[collection].insert_many(records)
            print("Successfully Inserted {} Documents in {}".format(len(records), collection))
        except pymongo.errors.BulkWriteError as e:
            print("Bulk Write Error:\n{}".format(e))
            sys.exit(1)
        except TypeError as e:
            print("TypeError:\n{}".format(e))
            sys.exit(1)
        except Exception as e:
            print("Exception:\n{}".format(e))
            sys.exit(1)

    def drop_collection(self, collection):
        """
        Drops Collection in Mongo
        :param collection: Collection Name
        :return:
        """
        try:
            self.mongo_client[collection].drop()
        except Exception as e:
            print("Exception while dropping the collection:\n{}".format(e))
            sys.exit(1)


