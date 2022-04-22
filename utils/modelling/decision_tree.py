from sklearn import tree
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from utils.checks import *
from utils.modelling.performance import *

def trainDecisionTree(X_train, y_train, max_leaf_nodes=None):
    if max_leaf_nodes is None:
        max_leaf_nodes = len(y_train.unique())
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(X_train, y_train)
    print('Trained decision tree \n')

    return clf


def plotTreeStructureSave(clf, given_name):

    plt.figure(figsize=(30,30))

    tree.plot_tree(clf, fontsize=10, feature_names=clf.feature_names_in_, class_names=clf.classes_)
    plt.savefig('{given_name}/tree_plot.png'.format(given_name=given_name))

    print('Tree structure plot saved')


def featureImportanceSave(clf, given_name):
    importance_df = pd.DataFrame({'importance':clf.feature_importances_, 'feature':clf.feature_names_in_}).sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df.to_csv('{given_name}/feature_importance.csv'.format(given_name=given_name), index=False)

    print('Feature importance csv saved')

def make_model(given_name, X_train, y_train, X_train_ups, y_train_ups, X_test, y_test, model_type = 'classification'):
    # check if X is a list (CV should be applied in that case)
    if isinstance(X_train, list):
        y_train_pred_list = list()
        y_test_pred_list = list()

        for fold_id in range(len(X_train_ups)):
            all_classes_represented = False
            max_leaf_nodes = len(y_train[fold_id].unique())
            while not all_classes_represented:
                print('Nr of max leaf nodes: {max_leaf_nodes}'.format(max_leaf_nodes=max_leaf_nodes))
                clf = trainDecisionTree(X_train_ups[fold_id], y_train_ups[fold_id], max_leaf_nodes=max_leaf_nodes)
                all_classes_represented = checkAllClassesHaveLeafNode(clf)
                max_leaf_nodes += 1

            y_train_pred_list.append(clf.predict(X_train[fold_id]))
            y_test_pred_list.append(clf.predict(X_test[fold_id]))

        # Merge list of lists into one list
        y_train_pred = [item for sublist in y_train_pred_list for item in sublist]
        y_test_pred = [item for sublist in y_test_pred_list for item in sublist]

        y_test = [item for sublist in y_test for item in sublist]
        y_train = [item for sublist in y_train for item in sublist]
        y_ups = [item for sublist in y_train_ups for item in sublist]

        # Merge list of dataframes into one dataframe
        X_test = pd.concat(X_test).reset_index(drop=True)
        X_train = pd.concat(X_train).reset_index(drop=True)
        X_ups = pd.concat(X_train_ups).reset_index(drop=True)

        # train model one last time on all samples
        all_classes_represented = False
        max_leaf_nodes = len(set(y_ups))
        while not all_classes_represented:
            print('Nr of max leaf nodes: {max_leaf_nodes}'.format(max_leaf_nodes=max_leaf_nodes))
            clf = trainDecisionTree(X_ups, y_ups, max_leaf_nodes=max_leaf_nodes)
            all_classes_represented = checkAllClassesHaveLeafNode(clf)
            max_leaf_nodes += 1

    # If just regular train/test split has been applied
    else:
        all_classes_represented = False
        max_leaf_nodes = len(y_train.unique())
        while not all_classes_represented:
            print('Nr of max leaf nodes: {max_leaf_nodes}'.format(max_leaf_nodes=max_leaf_nodes))
            clf = trainDecisionTree(X_train_ups, y_train_ups, max_leaf_nodes=max_leaf_nodes)
            all_classes_represented = checkAllClassesHaveLeafNode(clf)
            max_leaf_nodes += 1

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

    if model_type == 'classification':
        plotConfusionMatrixSave(given_name, y_train, y_train_pred, data_type='train')
        plotConfusionMatrixSave(given_name, y_test, y_test_pred, data_type='test')
        classificationReportSave(given_name, y_train, y_train_pred, data_type='train')
        classificationReportSave(given_name, y_test, y_test_pred, data_type='test')

    # plot the final tree
    plotTreeStructureSave(clf, given_name)
    featureImportanceSave(clf, given_name + '/feature_importance')

    return clf
