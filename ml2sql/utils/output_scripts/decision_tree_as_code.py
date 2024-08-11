import numpy as np
from pathlib import Path
import logging
from sklearn.tree import _tree
import sys

logger = logging.getLogger(__name__)


def tree_to_sql(tree, file=sys.stdout):
    """
    Outputs a decision tree model as an SQL Case When statement to a file or stdout

    Parameters:
    -----------
    tree: sklearn decision tree model
        The decision tree to represent as an SQL function
    file: file object, optional (default=sys.stdout)
        The file to write the output to. If not specified, prints to console.
    """

    tree_ = tree.tree_
    feature_name = [
        tree.feature_names_in_[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth):
        indent = "  " * depth

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if threshold % 1 == 0.5:
                cast = "::INT"
            else:
                cast = ""

            print(f"{indent}CASE WHEN {name}{cast} <= {threshold} THEN", file=file)
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}ELSE  -- if {name} > {threshold}", file=file)
            recurse(tree_.children_right[node], depth + 1)
            print(f"{indent}END", file=file)
        else:
            if hasattr(tree, "classes_"):
                class_values = tree_.value[node]
                samples = tree_.n_node_samples[node]
                max_value = int(np.max(class_values))
                predicted_class = tree.classes_[np.argmax(class_values)]

                if np.issubdtype(type(predicted_class), np.integer):
                    print(
                        f"{indent}{predicted_class} -- train data precision: {max_value / samples:.2f} ({max_value}/{samples})",
                        file=file,
                    )
                else:
                    print(
                        f"{indent}'{predicted_class}' -- train data precision: {max_value / samples:.2f} ({max_value}/{samples})",
                        file=file,
                    )
            else:
                prediction = tree_.value[node][0, 0]
                samples = tree_.n_node_samples[node]
                print(f"{indent}{prediction} -- samples ({samples})", file=file)

    print("SELECT", file=file)
    recurse(0, 1)
    print("AS prediction", file=file)
    print("FROM <source_table> -- change to your table name", file=file)


def save_model_and_extras(clf, model_name, post_params):
    output_path = Path(model_name) / "model" / "decisiontree_in_sql.sql"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        tree_to_sql(clf, file=f)

    logger.info("SQL version of decision tree saved")

    # If you want to also print to console, you can call the function again without the file parameter
    # tree_to_sql(clf)
