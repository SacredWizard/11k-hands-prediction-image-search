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

import numpy

from classes.featureextraction import ExtractFeatures

warnings.filterwarnings("ignore")


def numpy_set_params():
    """Set limits for numpy"""
    numpy.set_printoptions(suppress=True)
    numpy.set_printoptions(threshold=sys.maxsize)


def get_input_folder():
    """Get input from User for folder"""
    folder = str(input("Enter the folder path: "))
    if not folder or not os.path.isdir(folder):
        print("Please enter a valid folder path")
        return get_input_folder()
    return folder


def get_input_image(folder):
    """Get input from User for image"""
    image = str(input("Enter the Image filename: "))
    formats = [".jpeg", ".png", ".jpg"]
    if not image or not os.path.isfile(os.path.join(folder, image)):
        print("The image does not exist in the folder")
        return get_input_image(folder)
    elif not image.endswith(tuple(formats)):
        print("Please enter a valid Image name")
        return get_input_image(folder)
    return image


def get_input_model():
    """Gets the Input Model"""
    try:
        model = int(input("Enter the model name: Choices:\n1. CM\n2. HOG\n3. SIFT\n4. LBP\n"))
        if model not in [1, 2, 3, 4]:
            print("Please enter a valid choice")
            return get_input_model()
        elif model == 1:
            model = "CM"
        elif model == 2:
            model = "HOG"
        elif model == 3:
            model = "SIFT"
        elif model == 4:
            model = "LBP"
        return model
    except ValueError as exp:
        print("Enter a valid choice")
        return get_input_model()


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


def get_input_k():
    """Getting the value of k from user"""
    try:
        count = int(input("Enter the value for k: "))
        return count
    except ValueError as exp:
        print("Enter a valid Integer")
        return get_input_k()


def get_input_data():
    """Get Inputs from User"""
    try:
        choice = int(input("Enter the task you want to Perform. Choices: \n1. Task-1 \n2. Task-2\n3. Task-3\n"))
        if choice == 1:
            folder = get_input_folder()
            image = get_input_image(folder)
            model = get_input_model()

            feature_extractor = ExtractFeatures(folder, model,
                                                image)
            result = feature_extractor.execute()
            if model == "LBP":
                result = [float(x) for x in result.strip('[]').split(",")]
            print(numpy.array(result))

        elif choice == 2:
            folder = get_input_folder()
            model = get_input_model()

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
            #     plt.title("{}\nDistance: {}\nSimilarity: {}%".format(r[0], round(r[1], 3), round((1 - (r[1]/max_dist)) * 100),2))
            #     count = count + 1
            # plt.show()
        else:
            print("Enter a valid choice")
            get_input_data()
    except ValueError as exp:
        print("Enter a valid choice: {}".format(exp))
        get_input_data()


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
