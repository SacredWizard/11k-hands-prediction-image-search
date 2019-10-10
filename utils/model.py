"""
Image Search - Multimedia Web Database Systems Fall 2019 Project Group 17
This is a module for saving and loading machine learning models
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""
import pickle


class Model:
    def __init__(self):
        """
        Constructor for the Model Class
        """
        pass

    @staticmethod
    def save_model(model, filename):
        """
        Saves the model into a file
        :param model: Learning Model
        :param filename: Filename for the model
        :return:
        """
        with open(filename, 'wb') as file:
            pickle.dump(model, filename)

    @staticmethod
    def load_model(filename):
        """
        Loads the learning model from filename
        :param filename: Filename to load the model from
        :return: The learning Model
        """
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
