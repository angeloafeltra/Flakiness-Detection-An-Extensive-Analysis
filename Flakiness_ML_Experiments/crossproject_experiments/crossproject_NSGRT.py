import sys
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
from utils.eval_utils import eval_and_log_metrics
import numpy as np
from utils.crossproject_utils import NSGRT_PreProcessing
from utils.crossproject_utils import RareTransfer
from sklearn.model_selection import train_test_split

def run(dataset, pipeline, experiment_ID):


    list_project=dataset['nameProject'].unique()
    for target_project in list_project[0:]:
        print(target_project)

        target_set=dataset.loc[dataset['nameProject']==target_project]
        X_target=target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1).to_numpy()
        y_target=target_set[col.TARGET].to_numpy()

        if np.count_nonzero(y_target==1)<5:
            continue

        X_target_L, X_target_NL, y_target_L, y_target_NL= train_test_split(X_target, y_target,stratify=y_target, test_size=0.60)

        source_set=dataset.loc[dataset['nameProject']!=target_project]
        X_source=source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1).to_numpy()
        y_source=source_set[col.TARGET].to_numpy()

        nsgrt=NSGRT_PreProcessing(10)
        X_source, y_source, w= nsgrt.select_source(X_source,y_source,X_target_L,y_target_L)

        clf=RandomForestClassifier(random_state=42,n_estimators=50, criterion='gini', max_depth=3)
        rareTransfer=RareTransfer(X_target_L,y_target_L,clf)
        rareTransfer.fit(X_source,y_source,10,w)

        y_predict=rareTransfer.predict(X_target_L)
        eval_and_log_metrics('Train',y_target_L,y_predict)

        y_predict=rareTransfer.predict(X_target_NL)
        validation_utils.val_and_log_metrics(y_target_NL,y_predict)




if __name__ == "__main__":
    run(sys.argv[1])




