import pandas as pd
import numpy as np

def checkAllClassesHaveLeafNode(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    class_values = clf.tree_.value
    classes = clf.classes_

    end_leaf_classes = []

    for node_id in range(n_nodes):
        if children_left[node_id] == children_right[node_id]:
            feature_index = feature[node_id]
            class_name = clf.classes_[np.argmax(class_values[node_id])]
            end_leaf_classes.append(class_name)

    if len(set(end_leaf_classes)) == len(classes):
        print('All classes are represented by a leaf node\n')

        return True
    else:
        print('{}/{} classes are represented by a leaf node ({})'.format(len(set(end_leaf_classes)),len(clf.classes_),len(end_leaf_classes)))
        print('Missing class(es): {}\n'.format(set(clf.classes_) - set(end_leaf_classes)))

        return False
