# thanks to
# https://web.archive.org/web/20171005203850/http://www.kdnuggets.com/2017/05/simplifying-decision-tree-interpretation-decision-rules-python.html

# outputing the tree as code
from sklearn.tree import _tree
import numpy as np
from contextlib import redirect_stdout
import logging

logger = logging.getLogger(__name__)


def tree_to_sql(tree):
    """
    Outputs a decision tree model as an SQL Case When statement

    Parameters:
    -----------
    tree: sklearn decision tree model
            The decision tree to represent as an SQL function
    """

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
                cast = "::INT"
            else:
                cast = ""

            print("{}CASE WHEN {}{} <= {} THEN".format(indent, name, cast, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}ELSE  -- if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
            print("{}END".format(indent))
        else:
            if hasattr(tree, "classes_"):
                class_values = tree_.value[node]
                samples = tree_.n_node_samples[node]
                max_value = int(np.max(class_values))
                predicted_class = tree.classes_[np.argmax(class_values)]

                # Check if predicted class is an integer
                if np.issubdtype(type(predicted_class), np.integer):
                    print(
                        "{} {} -- train data precision: {:.2f} ({}/{})".format(
                            indent,
                            predicted_class,
                            max_value / samples,
                            max_value,
                            samples,
                        )
                    )
                else:
                    print(
                        "{} '{}' -- train data precision: {:.2f} ({}/{})".format(
                            indent,
                            predicted_class,
                            max_value / samples,
                            max_value,
                            samples,
                        )
                    )
            else:
                prediction = tree_.value[node][0, 0]
                samples = tree_.n_node_samples[node]
                print("{} {} -- samples ({})".format(indent, prediction, samples))

    print("SELECT")

    recurse(0, 1)

    print("AS prediction")
    print("FROM <source_table> -- change to your table name")


def save_model_and_extras(clf, model_name, post_params):
    # Write printed output to file
    with open(
        "{model_name}/model/tree_in_sql.sql".format(model_name=model_name), "w"
    ) as f:
        with redirect_stdout(f):
            tree_to_sql(clf)
    logger.info("SQL version of decision tree saved")
