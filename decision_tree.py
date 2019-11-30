from sklearn.metrics import accuracy_score
from phase3.task1 import compute_latent_semantic_for_label, reduced_dimensions_for_unlabelled_folder
from classes.dimensionreduction import DimensionReduction
from utils.excelcsv import CSVReader
from sklearn.preprocessing import StandardScaler
import numpy as np
import random


from random import seed
from csv import reader

csv_reader = CSVReader()

class DecisionTree(object):

    def __init__(self):
        self.max_depth = 20
        self.min_size = 5
        self.tree = {}

    # Evaluate an algorithm using a cross validation split
    def fit(self,x_train, y_train):
        train_set = x_train
        for i in range(len(train_set)):
            train_set[i] = np.append(train_set[i], y_train[i])
            # train_set[i].append(y_train[i])
        self.tree =  self.decision_tree(train_set)

    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
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
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
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
        root = self.get_split(train)
        self.split(root, 1)
        return root


    def predict(self, test):
        predictions = list()
        for row in test:
            prediction = self.predict1(self.tree, row)
            predictions.append(prediction)
        return predictions

def main():
    fea_ext_mod = "LBP"
    dim_red_mod = "LDA"
    dist_func = "euclidean"
    k_value = 30
    training_set = 'Dataset3/Labelled/Set1'
    test_set = 'Dataset3/Unlabelled/Set 1'
    label = "dorsal"
    obj_lat, feat_lat, model = compute_latent_semantic_for_label(fea_ext_mod,
                                                                 dim_red_mod, label, k_value, training_set)

    label_p = 'palmar'
    obj_lat_p, feat_lat_p, model_p = compute_latent_semantic_for_label(fea_ext_mod,
                                                                       dim_red_mod, label_p, k_value, training_set)

    x_train = obj_lat['reducedDimensions'].tolist()
    x_train += (obj_lat_p['reducedDimensions'].tolist())
    red_dim_unlabelled_images = reduced_dimensions_for_unlabelled_folder(fea_ext_mod, dim_red_mod, k_value, label,
                                                                         training_set, test_set)
    x_test = red_dim_unlabelled_images['reducedDimensions'].tolist()

    dim_red = DimensionReduction(fea_ext_mod, dim_red_mod, k_value)
    labelled_aspect = dim_red.get_metadata("imageName", obj_lat['imageId'].tolist())['aspectOfHand'].tolist()
    y_train = [i.split(' ')[0] for i in labelled_aspect]

    labelled_aspect = dim_red.get_metadata("imageName", obj_lat_p['imageId'].tolist())['aspectOfHand'].tolist()
    y_train += ([i.split(' ')[0] for i in labelled_aspect])

    unlabelled_aspect = dim_red.get_metadata("imageName", red_dim_unlabelled_images['imageId'].tolist())[
        'aspectOfHand'].tolist()
    y_test = [i.split(' ')[0] for i in unlabelled_aspect]

    print(x_train)
    # scale
    x_train = StandardScaler().fit_transform(x_train)
    x_train.tolist()
    print(x_train)

    # shuffle the training data
    c = list(zip(x_train, y_train))

    random.shuffle(c)

    x_train, y_train = zip(*c)

    # Test CART on dataset
    seed(1)
    clf = DecisionTree()
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    print('Accuracy:', accuracy)

if __name__ == "__main__":
    main()