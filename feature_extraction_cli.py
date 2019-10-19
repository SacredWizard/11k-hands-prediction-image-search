"""
Measuring Similarity of Hand Images Using Color Moments and Histogram of Oriented Gradients
This is the CLI for the First Phase of the Project
Author : Sumukh Ashwin Kamath
(ASU ID - 1217728013 email - skamath6@asu.edu
"""

import argparse
import os
import sys
import warnings
from utils.inputhelper import get_input_feature_extractor_model, get_input_image, get_input_folder, get_task_number

import numpy

from classes.featureextraction import ExtractFeatures

warnings.filterwarnings("ignore")


def numpy_set_params():
    """Set limits for numpy"""
    numpy.set_printoptions(suppress=True)
    numpy.set_printoptions(threshold=sys.maxsize)


def get_input_dist():
    """Gets the distance measure to be used for computing similarity"""
    try:
        dist = int(input("Enter the distance measure to be used for computing similarity: Choices\n"
                         "1. Weighted Manhattan\n2. Manhattan\n3. Euclidean\n"))
        if dist not in [1, 2, 3]:
            print("Please enter a valid choice")
            return get_input_dist()
        elif dist == 1:
            dist = "WM"
        elif dist == 2:
            dist = "MH"
        elif dist == 3:
            dist = "EUC"
        return dist
    except ValueError as exp:
        print("Please enter a valid choice")
        return get_input_dist()


def get_input_data():
    """Get Inputs from User"""
    number_of_tasks = 2
    choice = get_task_number(number_of_tasks)
    if choice == 1:
        folder = get_input_folder()
        image = get_input_image(folder)
        model = get_input_feature_extractor_model()
        feature_extractor = ExtractFeatures(folder, model)
        result = feature_extractor.execute(image)
        # if model == "LBP":
        #     result = [float(x) for x in result.strip('[]').split(",")]
        print(numpy.array(result))

    elif choice == 2:
        folder = get_input_folder()
        model = get_input_feature_extractor_model()

        feature_extractor = ExtractFeatures(folder, model)
        feature_extractor.execute()

    elif choice == 3:
        pass
    # f older = get_input_folder()
    # image = get_input_image(folder)
    # model = get_input_model()
    # dist = get_input_dist()
    # k_count = get_input_k()
    # img = io.imread(os.path.join(folder, image))
    # print("Searching {} images closest to {}".format(k_count, os.path.join(folder, image)))
    # result, max_dist = search_k_nearest(img, model, dist, k_count)
    #
    # f = plt.figure()
    # f.add_subplot(3, 5, 1)
    # plt.imshow(img)
    # plt.title("Query Image")
    #
    # count = 2
    # for r in result:
    #     f.add_subplot(3, 5, count)
    #     plt.imshow(io.imread(os.path.join(folder, r[0])))
    #     plt.title("{}\nDistance: {}\nSimilarity: {}%".format(r[0], round(r[1], 3),
    #     round((1 - (r[1]/max_dist)) * 100),2))
    #     count = count + 1
    # plt.show()


def main():
    """Main function for the program"""
    parser = argparse.ArgumentParser(description='CLI for executing Phase 1 of the project')
    parser.add_argument('--full', type=bool, help='Displays full array')
    full = parser.parse_args().full
    if full:
        numpy_set_params()
    get_input_data()


if __name__ == "__main__":
    main()
