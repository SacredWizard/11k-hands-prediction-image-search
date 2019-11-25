import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
from classes.dimensionreduction import DimensionReduction
from utils.model import Model
from utils.excelcsv import CSVReader
from classes.mongo import MongoWrapper
from classes.globalconstants import GlobalConstants
from utils.inputhelper import get_input_subject_id
import pandas as pd
import numpy as np
import scipy.stats as stats
import operator
import utils.imageviewer as imgvwr
from utils.excelcsv import CSVReader
import phase2.task1 as p1task1
import phase2.task5 as p2task5
import phase2.task6 as p2task6
import time
import warnings
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
model_interact = Model()
global_constants = GlobalConstants()
mongo_wrapper = MongoWrapper(global_constants.Mongo().DB_NAME)
csv_reader = CSVReader()

def compute_latent_semantic_for_label(fea_ext_mod, dim_red_mod, label, k_value, folder):

    # p2task5.run_task3(fea_ext_mod, dim_red_mod, label, k_value)

    dim_reduction = DimensionReduction(fea_ext_mod, dim_red_mod, k_value, label, folder_metadata=folder)
    obj_lat, feat_lat, model = dim_reduction.execute()
    # Saves the returned model
    filename = "{0}_{1}_{2}_{3}_{4}".format(fea_ext_mod, dim_red_mod, label,
                                        str(k_value), os.path.basename(folder))
    model_interact.save_model(model=model, filename=filename)
    return obj_lat,feat_lat, model


def reduced_dimensions_for_unlabelled_folder(fea_ext_mod, dim_red_mod, k_value, label, train_set, test_set):
    dim_reduction = DimensionReduction(fea_ext_mod, dim_red_mod, k_value, label)
    filename = "{0}_{1}_{2}_{3}_{4}".format(fea_ext_mod, dim_red_mod, label,
                                        str(k_value), os.path.basename(train_set))
    model = model_interact.load_model(filename=filename)
    red_dim = []
    unlabelled_image_list = os.listdir(test_set)
    for image in unlabelled_image_list:
        red_dim.append(dim_reduction.compute_query_image(model, test_set, image))
    df = pd.DataFrame({"imageId": unlabelled_image_list, "reducedDimensions": red_dim})
    return df

def main():
    fea_ext_mod = "HOG"
    dim_red_mod = "NMF"
    dist_func = "euclidean"
    k_value = 12
    training_set = 'C:\mwdb\commoncode\CSE515\Dataset3\Labelled\Set2'
    test_set = 'C:\mwdb\commoncode\CSE515\Dataset3\\Unlabelled\Set 1'
    label = "dorsal"
    obj_lat,feat_lat, model = compute_latent_semantic_for_label(fea_ext_mod, 
                                        dim_red_mod, label , k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, label, str(k_value))
    csv_reader.save_to_csv(obj_lat, feat_lat, filename)

    
    x_train = obj_lat['reducedDimensions'].tolist()
    x_test = reduced_dimensions_for_unlabelled_folder(fea_ext_mod, dim_red_mod, k_value, label, training_set, test_set)

    dim_red = DimensionReduction(fea_ext_mod,dim_red_mod,k_value)
    labelled_aspect = dim_red.get_metadata("imageName", obj_lat['imageId'].tolist())['aspectOfHand'].tolist()
    y_train = [i.split(' ')[0] for i in labelled_aspect]
    unlabelled_aspect = dim_red.get_metadata("imageName", x_test['imageId'].tolist())['aspectOfHand'].tolist()
    y_test = [i.split(' ')[0] for i in unlabelled_aspect]
    lr = LogisticRegression()
    lr.fit(x_train.array, np.asarray(y_train))
    x_pred = lr.predict(x_test)
    y_pred = lr.predict(y_test)

    result_dorsal = []
    for root, dirs, images in os.walk(test_set):
        for image_name in images:
            result_dorsal.append((p2task5.run_task4(fea_ext_mod, dim_red_mod, 
                            test_set, image_name, dist_func, "dorsal", k_value, m_value=1))[0]['score'])
    result_palmar = []
    for root, dirs, images in os.walk(test_set):
        for image_name in images:
            result_palmar.append((p2task5.run_task4(fea_ext_mod, dim_red_mod, 
                            test_set, image_name, dist_func, "palmar", k_value, m_value=1))[0]['score'])

    for root, dirs, images in os.walk(test_set):
        for i, image_name in enumerate(images):
            if result_dorsal[i] > result_palmar[i]:
                print(image_name, ' : dorsal')
            else:
                print(image_name, ' : palmar')


if __name__ == "__main__":
    main()

    