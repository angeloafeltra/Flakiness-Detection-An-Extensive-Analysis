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
from utils.crossproject_utils import TCA
from utils.burak_utils import classic_burakFilter


def run(dataset, pipeline, experiment_ID):


    list_project=dataset['nameProject'].unique()
    for target_project in list_project[0:]:
        print(target_project)

        target_set=dataset.loc[dataset['nameProject']==target_project]
        X_target=target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1).to_numpy()
        y_target=target_set[col.TARGET].to_numpy()

        if np.count_nonzero(y_target==1)<5:
            continue


        source_set=dataset.loc[dataset['nameProject']!=target_project]
        X_source=source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1).to_numpy()
        y_source=source_set[col.TARGET].to_numpy()


        X_source, y_source=  classic_burakFilter(X_source,
                                                 y_source,
                                                 X_target,
                                                 10)

        tca=TCA(X_target,n_components=X_source.shape[1])
        X_source=tca.fit_transform(X_source,X_target)

        clf=RandomForestClassifier(random_state=42,n_estimators=100, criterion='gini',class_weight='balanced')
        clf.fit(X_source,y_source)
        X_target=tca.transform(X_target)
        y_predict=clf.predict(X_target)
        validation_utils.val_and_log_metrics(y_target,y_predict)




if __name__ == "__main__":
    run(sys.argv[1])




