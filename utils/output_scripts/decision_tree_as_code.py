# thanks to
# https://web.archive.org/web/20171005203850/http://www.kdnuggets.com/2017/05/simplifying-decision-tree-interpretation-decision-rules-python.html

# outputing the tree as code
from sklearn.tree import _tree
import pandas as pd
import numpy as np
from contextlib import redirect_stdout


def tree_to_sql(tree):
    '''
	Outputs a decision tree model as an SQL Case When statement

	Parameters:
	-----------
	tree: sklearn decision tree model
		The decision tree to represent as an SQL function
	'''

    tree_ = tree.tree_
    feature_name = [
        tree.feature_names_in_[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        # add indentation for better readability
        indent = "  " * depth

        # Check if it is a leaf node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if threshold % 1 == 0.5:
                cast = '::INT'
            else:
                cast = ''

            print("{}CASE WHEN {}{} <= {} THEN".format(indent, name, cast, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}ELSE  -- if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            print("{}END".format(indent))
        else:
            class_values = tree_.value[node]
            samples = tree_.n_node_samples[node]
            max_value = int(np.max(class_values))
            predicted_class = tree.classes_[np.argmax(class_values)]
            print("{} {} -- train data precision: {:.2f} ({}/{})".format(indent, max_value / samples, max_value / samples, max_value,
                                                                samples))

    recurse(0, 1)

    print('AS pred_classification')


def getDecisionPath(clf, sample):
    # WIP to path to see why decision was made
    # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature_index = clf.tree_.feature
    feature_names = clf.feature_names_in_
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(sample)
    leaf_id = clf.apply(sample)

    sample_id = 1

    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[
                 node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]
                 ]

    print("Rules used to predict sample {id}:\n".format(id=sample_id))
    for node_id in range(len(node_index)):
        id = node_index[node_id]
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if sample.iloc[sample_id, feature_index[id]] <= threshold[id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print(
            "decision node {node} : ({feature_name} = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                sample=sample_id,
                feature_index=feature_index[id],
                feature_name=feature_names[feature_index[id]],
                value=sample.iloc[sample_id, feature[id]],
                inequality=threshold_sign,
                threshold=threshold[id],
            )
        )

        if leaf_id[sample_id] == node_index[node_id + 1]:
            print('Classification: {classification}'.format(classification=clf.classes_[np.argmax(values[id])]))

def save_model_and_extras(clf, model_name, sql_split, logging):
    # Write printed output to file
    with open('{model_name}/model/tree_in_sql.sql'.format(model_name=model_name), 'w') as f:
        with redirect_stdout(f):
            tree_to_sql(clf)
    print('SQL version of decision tree saved')
