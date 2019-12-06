from sklearn.metrics import accuracy_score
from phase3.task1 import compute_latent_semantic_for_label, reduced_dimensions_for_unlabelled_folder
from classes.dimensionreduction import DimensionReduction
from utils.excelcsv import CSVReader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
import random
from utils.inputhelper import get_input_folder
import os
import pandas as pd
from utils.model import Model


from random import seed

csv_reader = CSVReader()

class DecisionTree(object):

    def __init__(self):
        self.max_depth = 20
        self.min_size = 5
        self.tree = {}

    # Fit training data
    def fit(self, x_train, y_train):
        train_set = x_train
        for i in range(len(train_set)):
            train_set[i].append(y_train[i])
        self.tree = self.decision_tree(train_set)

    # Split a dataset based on an attribute and an attribute value
    def dataset_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def calc_gini_index(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def get_best_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.dataset_split(index, row[index], dataset)
                gini = self.calc_gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # Create a leaf node value
    def leaf_node(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.leaf_node(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.leaf_node(left), self.leaf_node(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.get_best_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.leaf_node(right)
        else:
            node['right'] = self.get_best_split(right)
            self.split(node['right'], depth+1)

    # Make a prediction with a decision tree
    def predict1(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict1(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict1(node['right'], row)
            else:
                return node['right']

    # Classification and Regression Tree Algorithm
    def decision_tree(self, train):
        root = self.get_best_split(train)
        self.split(root, 1)
        return root


    def predict(self, test):
        predictions = list()
        for row in test:
            prediction = self.predict1(self.tree, row)
            predictions.append(prediction)
        return predictions

def get_x_train(fea_ext_mod, dim_red_mod, k_value, train_set):
    model_interact = Model()
    dim_reduction = DimensionReduction(fea_ext_mod, dim_red_mod, k_value, folder_metadata=train_set,
                                       metadata_collection="labelled")
    obj_lat, feat_lat, model = dim_reduction.execute()
    filename = "{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod,
                                            str(k_value), os.path.basename(train_set))
    model_interact.save_model(model=model, filename=filename)
    return obj_lat

def get_x_test(fea_ext_mod, dim_red_mod, k_value, train_set, test_set):
        model_interact = Model()
        dim_reduction = DimensionReduction(fea_ext_mod, dim_red_mod, k_value)
        filename = "{0}_{1}_{2}_{3}".format(fea_ext_mod, dim_red_mod,
                                                str(k_value), os.path.basename(train_set))
        model = model_interact.load_model(filename=filename)
        red_dims = []
        unlabelled_image_list = os.listdir(test_set)
        for image in unlabelled_image_list:
            red_dim = dim_reduction.compute_query_image(model, test_set, image)
            red_dims.append(red_dim[0])
        df = pd.DataFrame({"imageId": unlabelled_image_list, "reducedDimensions": red_dims})
        return df

def get_y(fea_ext_mod, dim_red_mod, k_value, collection_name, red_dim=None, obj_lat=None):
    dim_red = DimensionReduction(fea_ext_mod, dim_red_mod, k_value)
    if collection_name == 'unlabelled':
        aspect = dim_red.get_metadata("imageName", red_dim['imageId'].tolist())['aspectOfHand'].tolist()
    else:
        aspect = dim_red.get_metadata_collection("imageName", obj_lat['imageId'].tolist(), collection_name)['aspectOfHand'].tolist()
    return [i.split(' ')[0] for i in aspect]


def main():
    fea_ext_mod = "HOG"
    dim_red_mod = "PCA"
    dist_func = "euclidean"
    k_value = 30
    training_set = 'Dataset3/Labelled/Set2'
    test_set = 'Dataset3/Unlabelled/Set 2'
    # # training_set = get_input_folder("Labelled")
    # # test_set = get_input_folder("Classify")
    # label = "dorsal"
    # obj_lat, feat_lat, model = compute_latent_semantic_for_label(fea_ext_mod,
    #                                                              dim_red_mod, label, k_value, training_set)
    #
    # label_p = 'palmar'
    # obj_lat_p, feat_lat_p, model_p = compute_latent_semantic_for_label(fea_ext_mod,
    #                                                                    dim_red_mod, label_p, k_value, training_set)
    #
    # x_train = obj_lat['reducedDimensions'].tolist()
    # x_train += (obj_lat_p['reducedDimensions'].tolist())
    # red_dim_unlabelled_images = reduced_dimensions_for_unlabelled_folder(fea_ext_mod, dim_red_mod, k_value, label,
    #                                                                      training_set, test_set)
    # x_test = red_dim_unlabelled_images['reducedDimensions'].tolist()
    #
    # dim_red = DimensionReduction(fea_ext_mod, dim_red_mod, k_value)
    # labelled_aspect = dim_red.get_metadata("imageName", obj_lat['imageId'].tolist())['aspectOfHand'].tolist()
    # y_train = [i.split(' ')[0] for i in labelled_aspect]
    #
    # labelled_aspect = dim_red.get_metadata("imageName", obj_lat_p['imageId'].tolist())['aspectOfHand'].tolist()
    # y_train += ([i.split(' ')[0] for i in labelled_aspect])
    #
    # unlabelled_aspect = dim_red.get_metadata("imageName", red_dim_unlabelled_images['imageId'].tolist())[
    #     'aspectOfHand'].tolist()
    # y_test = [i.split(' ')[0] for i in unlabelled_aspect]

    obj_lat_train = get_x_train(fea_ext_mod, dim_red_mod, k_value, training_set)
    x_train = obj_lat_train['reducedDimensions'].tolist()
    red_dim = get_x_test(fea_ext_mod, dim_red_mod, k_value, training_set, test_set)
    x_test = red_dim['reducedDimensions'].tolist()
    y_train = get_y(fea_ext_mod, dim_red_mod, k_value, 'labelled', obj_lat=obj_lat_train)
    y_test = get_y(fea_ext_mod, dim_red_mod, k_value, 'unlabelled', red_dim=red_dim)

    # # scale
    x_train = StandardScaler().fit_transform(x_train)
    x_train = x_train.tolist()

    # shuffle the training data
    c = list(zip(x_train, y_train))

    random.shuffle(c)

    x_train, y_train = zip(*c)
    #
    # # Test CART on dataset
    seed(1)
    clf = DecisionTree()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    print("---------------------------")
    accuracy = accuracy_score(y_test, predictions) * 100
    print("Accuracy: " + str(accuracy) + "%")

    # precision = precision_score(y_test, predictions, pos_label="dorsal")
    # print("Precision: " + str(precision) + "%")
    #
    # recall = recall_score(y_test, predictions, pos_label="dorsal")
    # print("Recall: " + str(recall) + "%")

    unlabelled_images = red_dim['imageId']
    print("---------------------------")
    print("Results:")
    print("Image ID, Prediction, Actual")
    for image_id, p, a in zip(unlabelled_images, predictions, y_test):
        print("(" + image_id + ", " + p + ", " + a + ")")

if __name__ == "__main__":
    main()