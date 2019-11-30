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
import random as random
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

    dim_reduction = DimensionReduction(fea_ext_mod, dim_red_mod, k_value, label, folder_metadata=folder, metadata_collection="labelled")
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
    red_dims = []
    unlabelled_image_list = os.listdir(test_set)
    for image in unlabelled_image_list:
        red_dim = dim_reduction.compute_query_image(model, test_set, image)
        red_dims.append(red_dim[0])
    df = pd.DataFrame({"imageId": unlabelled_image_list, "reducedDimensions": red_dims})
    return df

def main():
    fea_ext_mod = "HOG"
    dim_red_mod = "SVD"
    dist_func = "euclidean"
    k_value = 30
    training_set = os.path.abspath('Dataset3\Labelled\Set2')
    test_set = os.path.abspath('Dataset3\\Unlabelled\Set 1')
    label = "dorsal"
    obj_lat,feat_lat, model = compute_latent_semantic_for_label(fea_ext_mod, 
                                        dim_red_mod, label , k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, label, str(k_value))
    csv_reader.save_to_csv(obj_lat, feat_lat, filename)
    x_train = obj_lat['reducedDimensions'].tolist()
    
    red_dim_unlabelled_images = reduced_dimensions_for_unlabelled_folder(fea_ext_mod, dim_red_mod,
                                    k_value, label, training_set, test_set)
    x_test = red_dim_unlabelled_images['reducedDimensions'].tolist()

    dim_red = DimensionReduction(fea_ext_mod,dim_red_mod,k_value)
    labelled_aspect = dim_red.get_metadata("imageName", obj_lat['imageId'].tolist())['aspectOfHand'].tolist()
    y_train = [i.split(' ')[0] for i in labelled_aspect]

    label_p = 'palmar'
    obj_lat_p,feat_lat_p, model_p = compute_latent_semantic_for_label(fea_ext_mod, 
                                        dim_red_mod, label_p , k_value, training_set)
    filename = "p3task1_{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod, label_p, str(k_value))
    csv_reader.save_to_csv(obj_lat_p, feat_lat_p, filename)
    x_train += (obj_lat_p['reducedDimensions'].tolist())
    labelled_aspect = dim_red.get_metadata("imageName", obj_lat_p['imageId'].tolist())['aspectOfHand'].tolist()
    y_train += ([i.split(' ')[0] for i in labelled_aspect])
    
    zip_train = list(zip(x_train, y_train))
    random.shuffle(zip_train)
    x_train, y_train = zip(*zip_train)

    unlabelled_aspect = dim_red.get_metadata("imageName", red_dim_unlabelled_images['imageId'].tolist())['aspectOfHand'].tolist()
    y_test = [i.split(' ')[0] for i in unlabelled_aspect]
    lr = LogisticRegression(penalty='l2', random_state=np.random.RandomState(42), solver='lbfgs', max_iter =300,
                                         multi_class='ovr',class_weight='balanced', n_jobs=-1, l1_ratio=0)
    lr.fit(x_train, y_train)
    # y_pred = lr.predict(x_test)
    print("Accuracy:",lr.score(x_test,y_test))

if __name__ == "__main__":
    main()

    