# rule set classifieres
from imodels import RuleFitClassifier, OneRClassifier

# rule list classifiers
from imodels import OptimalRuleListClassifier

# import performance measurements
from sklearn import metrics

import numpy as np
import pandas as pd

# Train rule fit model to every class (1 vs All)
# return the top 5 rules for each model in a dataframe
def RuleFitListRules(X_train, X_test, y_train, y_test):
    rules_dicts = []


    for idx, class_class in enumerate(y_train.unique()):
        print(idx, class_class)
        y_test_adjusted = np.where(y_test == class_class,1,0)
        model = RuleFitClassifier(random_state=42)  # initialize a model
        model.fit(X_train, y_train == class_class)

        top5rules = model.get_rules().sort_values('importance', ascending=False).reset_index(drop=True).head(5)

        rules_dict = {}
        rules_dict['classification'] = class_class
        rules_dict['precision'] = metrics.precision_score(y_test_adjusted, model.predict(X_test), zero_division=0)
        rules_dict['recall'] = metrics.recall_score(y_test_adjusted, model.predict(X_test), zero_division=0)
        rules_dict['rule'] = top5rules['rule']
        rules_dict['coef'] = top5rules['coef']
        rules_dict['support'] = top5rules['support']
        rules_dict['importance'] = top5rules['importance']

        rules_dicts.append(rules_dict)

    return pd.DataFrame(rules_dicts).set_index(['classification','precision','recall']).apply(pd.Series.explode).reset_index()

def saveRuleFitRules(given_name, X_train, X_test, y_train, y_test):
    rules_df = RuleFitListRules(X_train, X_test, y_train, y_test)
    rules_df.to_csv('{given_name}/top_5_rules_per_classification.csv'.format(given_name=given_name))
