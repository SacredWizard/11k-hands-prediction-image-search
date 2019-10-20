import os
from classes.globalconstants import GlobalConstants
from classes.mongo import MongoWrapper

global_constants = GlobalConstants()
mongo_wrapper = MongoWrapper()


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
    if not image or not os.path.isfile(os.path.join(folder, image)):
        print("The image does not exist in the folder")
        return get_input_image(folder)
    elif not image.endswith(global_constants.JPG_EXTENSION):
        print("Please enter a valid Image filename")
        return get_input_image(folder)
    return image


def get_input_image_label():
    """Gets the image label input from the user"""
    try:
        label_map = {}
        input_string = "\n".join(["{}. {}".format(i, label) for i, label in enumerate(global_constants.image_labels,
                                                                                      start=1)])
        choice = int(input("Enter the Label. Choices: \n{}\n".format(input_string)))
        if choice not in range(1, len(global_constants.image_labels)+1):
            print("Enter a valid choice.")
            return get_input_image_label()
        else:
            return global_constants.image_labels[choice - 1]
    except ValueError as exp:
        print("Enter a valid choice: {}".format(exp))
        return get_input_image_label()


def get_task_number(number_of_tasks):
    """Getting the task number from user"""
    try:
        input_string = "\n".join(["{}. Task-{}".format(i, str(i)) for i in range(1, number_of_tasks+1)])
        choice = int(input("Enter the task you want to Perform. Choices: \n{}\n".format(input_string)))
        if choice not in range(1, number_of_tasks+1):
            print("Enter a valid choice.")
            return get_task_number(number_of_tasks)
        else:
            return choice
    except ValueError as exp:
        print("Enter a valid choice: {}".format(exp))
        return get_task_number(number_of_tasks)


def get_input_subject_id():
    """Gets the subject id as input from the user"""
    try:
        subject_ids_db = mongo_wrapper.distinct(global_constants.METADATA, "id", {})
        subject_id = int(input("Enter the subject ID: "))
        if subject_id not in subject_ids_db:
            print("Subject ID not found. Please enter the correct subject ID")
            return get_input_subject_id()
        return subject_id
    except ValueError as exp:
        print("Enter a valid Subject Id")
        return get_input_subject_id()


def get_input_feature_extractor_model():
    """Gets the Input Model"""
    try:
        model = int(input("Enter the feature extractor model: Choices:\n1. Color Moments\n"
                          "2. Histogram of Oriented Gradients\n3. SIFT\n4. Local Binary Patterns\n"))
        model_map = {1: "CM", 2: "HOG", 3: "SIFT", 4: "LBP"}
        if model not in [1, 2, 3, 4]:
            print("Please enter a valid choice")
            return get_input_feature_extractor_model()
        return model_map[model]
    except ValueError as exp:
        print("Enter a valid choice")
        return get_input_feature_extractor_model()


def get_input_dimensionality_reduction_model():
    """Gets the Dimensionality Reduction Model"""
    try:
        model = int(input("Enter the Dimensionality Reduction Model: Choices:\n1. PCA\n2. SVD\n3. NMF\n4. LDA\n"))
        model_map = {1: "PCA", 2: "SVD", 3: "NMF", 4: "LDA"}
        if model not in [1, 2, 3, 4]:
            print("Please enter a valid choice")
            return get_input_feature_extractor_model()
        return model_map[model]
    except ValueError as exp:
        print("Enter a valid choice")
        return get_input_feature_extractor_model()


# def get_input_dist():
#     """Gets the distance measure to be used for computing similarity"""
#     try:
#         dist = int(input("Enter the distance measure to be used for computing similarity: Choices\n"
#                          "1. Weighted Manhattan\n2. Manhattan\n3. Euclidean\n"))
#         if dist not in [1, 2, 3]:
#             print("Please enter a valid choice")
#             return get_input_dist()
#         elif dist == 1:
#             dist = "WM"
#         elif dist == 2:
#             dist = "MH"
#         elif dist == 3:
#             dist = "EUC"
#         return dist
#     except ValueError as exp:
#         print("Please enter a valid choice")
#         return get_input_dist()


def get_input_k():
    """Getting the value of k from user"""
    try:
        count = int(input("Enter the value for k: "))
        return count
    except ValueError as exp:
        print("Enter a valid Integer: {}".format(exp))
        return get_input_k()


def get_input_m():
    """Getting the value of m from user"""
    try:
        count = int(input("Enter the value for m: "))
        return count
    except ValueError as exp:
        print("Enter a valid Integer: {}".format(exp))
        return get_input_m()
