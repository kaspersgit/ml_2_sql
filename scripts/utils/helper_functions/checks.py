import numpy as np


def checkTargetHard(target, logging):
    if target.nunique() == 1:
        raise Exception("Target column needs more than 1 unique value")


def checkFeatures(features, logging):
    featNullCount = features.isnull().sum()
    nullf = featNullCount[featNullCount > 0]
    if len(nullf) > 0:
        logging.info(f"NULL values found in the data, for the following: \n{nullf}")


def checkInputDataHard(data, config, logging):
    """
    Checks at start ensuring target and feature columns are good to go
    """
    checkTargetHard(data[config["target"]], logging)
    checkFeatures(data[config["features"]], logging)


def checkAllClassesHaveLeafNode(clf):
    """
    Check if all classes are represented by a leaf node in a given decision tree classifier.

    Parameters:
    -----------
    clf : sklearn.tree.DecisionTreeClassifier object
        The decision tree classifier to be checked.

    Returns:
    --------
    bool
        True if all classes are represented by a leaf node, False otherwise.
    """
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    # feature = clf.tree_.feature
    # threshold = clf.tree_.threshold
    class_values = clf.tree_.value
    classes = clf.classes_

    end_leaf_classes = []

    for node_id in range(n_nodes):
        if children_left[node_id] == children_right[node_id]:
            # feature_index = feature[node_id]
            class_name = clf.classes_[np.argmax(class_values[node_id])]
            end_leaf_classes.append(class_name)

    if len(set(end_leaf_classes)) == len(classes):
        print("All classes are represented by a leaf node\n")

        return True
    else:
        print(
            "{}/{} classes are represented by a leaf node ({})".format(
                len(set(end_leaf_classes)), len(clf.classes_), len(end_leaf_classes)
            )
        )
        print(
            "Missing class(es): {}\n".format(set(clf.classes_) - set(end_leaf_classes))
        )

        return False
