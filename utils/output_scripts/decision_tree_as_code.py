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


def getLeafNodesDecisionDf(clf):
    # Get full tree with nodes information
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature_index = clf.tree_.feature
    feature_names = clf.feature_names_in_
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    col_names = ['node', 'classification', 'precision'] + list(clf.classes_)
    end_leaf_class = pd.DataFrame(columns=col_names)

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)

    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            class_name = clf.classes_[np.argmax(values[i])]
            precision = np.max(values[i]) / np.sum(values[i])

            to_append = [i, class_name, precision] + list(values[i][0])
            a_series = pd.Series(to_append, index=end_leaf_class.columns)
            end_leaf_class = end_leaf_class.append(a_series, ignore_index=True)

    if len(set(end_leaf_classes)) == len(clf.classes_):
        print('All classes are represented by a leaf node')
    else:
        print('Not all classes are represented by a leaf node \nMissing class(es): {}'.format(
            set(clf.classes_) - set(end_leaf_classes)))

    print('{}/{} classes are represented by a leaf node ({})\n'.format(len(set(end_leaf_classes)), len(clf.classes_),
                                                                       len(end_leaf_classes)))
    end_leaf_class.sort_values('classification').reset_index(drop=True)

    return end_leaf_class


# https://stackoverflow.com/questions/66297576/how-to-retrieve-the-full-branch-path-leading-to-each-leaf-node-of-a-sklearn-deci
def retrieve_branches(number_nodes, children_left_list, children_right_list):
    """Retrieve decision tree branches"""

    # Calculate if a node is a leaf
    is_leaves_list = [(False if cl != cr else True) for cl, cr in zip(children_left_list, children_right_list)]

    # Store the branches paths
    paths = []

    for i in range(number_nodes):
        if is_leaves_list[i]:
            # Search leaf node in previous paths
            end_node = [path[-1] for path in paths]

            # If it is a leave node yield the path
            if i in end_node:
                output = paths.pop(np.argwhere(i == np.array(end_node))[0][0])
                yield output

        else:

            # Origin and end nodes
            origin, end_l, end_r = i, children_left_list[i], children_right_list[i]

            # Iterate over previous paths to add nodes
            for index, path in enumerate(paths):
                if origin == path[-1]:
                    paths[index] = path + [end_l]
                    paths.append(path + [end_r])

            # Initialize path in first iteration
            if i == 0:
                paths.append([i, children_left_list[i]])
                paths.append([i, children_right_list[i]])


def getDecisionsPerLeafNode(clf, model_name):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature_index = clf.tree_.feature
    feature_names = clf.feature_names_in_
    threshold = clf.tree_.threshold
    class_values = clf.tree_.value

    paths_to_leafs = retrieve_branches(n_nodes, children_left, children_right)
    classification_list = []
    node_visited_id_list = []
    decision_list = []

    for path in paths_to_leafs:
        predicted_class = clf.classes_[np.argmax(class_values[path[-1]])]
        classification_list = classification_list + [predicted_class] * len(path)
        node_visited_id_list = node_visited_id_list + path
        for n in range(len(path)):
            # loop through the path to get the decision made here
            node_id = path[n]
            if n + 1 == len(path):
                decision = 'leaf node'
            elif children_left[node_id] == path[n + 1]:
                # went to the left hence value is less than threshold
                # print('left')
                # print('{feature_name} <= {threshold}'.format(feature_name=feature_names[feature_index[node_id]], threshold=threshold[node_id]))
                decision = '{feature_name} <= {threshold}'.format(feature_name=feature_names[feature_index[node_id]],
                                                                  threshold=threshold[node_id])
            elif children_right[node_id] == path[n + 1]:
                # went to the right hence value is more than threshold
                # print('right')
                # print('{feature_name} > {threshold}'.format(feature_name=feature_names[feature_index[node_id]], threshold=threshold[node_id]))
                decision = '{feature_name} > {threshold}'.format(feature_name=feature_names[feature_index[node_id]],
                                                                 threshold=threshold[node_id])
            else:
                print('something went wrong')

            decision_list.append(decision)

    decision_dict = {'classification': classification_list, 'node_visited_id': node_visited_id_list,
                     'decision': decision_list}
    decisions_df = pd.DataFrame(decision_dict).drop_duplicates().sort_values(
        ['classification', 'node_visited_id']).reset_index(drop=True)
    decisions_df.to_csv('{model_name}/decisions_to_classification.csv'.format(model_name=model_name), index=False)
    print('Decisions per leaf node saved')


def save_model_and_extras(clf, model_name, sql_split, logging):
    # Write printed output to file
    with open('{model_name}/model/tree_in_sql.sql'.format(model_name=model_name), 'w') as f:
        with redirect_stdout(f):
            tree_to_sql(clf)
    print('SQL version of decision tree saved')
