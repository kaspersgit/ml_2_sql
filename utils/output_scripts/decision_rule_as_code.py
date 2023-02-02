import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from contextlib import redirect_stdout

def save_model_and_extras(clf, model_name, split, logging):
    # Decision rule not yet translated to SQL
    print('SQL version of Decision Rule not yet available')
    logging.info('SQL version of Decision Rule not yet available')
