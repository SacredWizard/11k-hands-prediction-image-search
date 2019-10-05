import pymongo
import pymongo.errors

from classes import global_constants


class GlobalConnections:

    def __init__(self):
        constants = global_constants.GlobalConstants()
        try:
            self.mongo_client = pymongo.MongoClient(constants.MONGO_URL,
                                                    serverSelectionTimeoutMS=constants.MONGO_SERVER_TIMEOUT)
        except pymongo.errors.ConnectionFailure as e:
            print("Connection Failure:\n{}".format(e))
        except pymongo.errors.ServerSelectionTimeoutError as e:
            print("Timeout:\n{}".format(e))
        except Exception as e:
            print("Exception has occurred:\n{}".format(e))
