import sys

import pandas as pd
import os

import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings
import pickle
import utils.validation_utils as validation_utils
from utils.burak_utils import classic_burakFilter

def run(dataset, pipeline, experiment_ID):


    list_project=dataset['nameProject'].unique()
    with mlflow.start_run(run_name='CrossProject_BurakFilter',experiment_id= experiment_ID) as father_run:
        for project in list_project:
            print(project)
            with mlflow.start_run(run_name=project,experiment_id=experiment_ID,nested=True) as child_run:
                train_set=dataset.loc[dataset['nameProject']!=project]
                test_set=dataset.loc[dataset['nameProject']==project]

                X_train_set = train_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_train_set = train_set[col.TARGET]
                cat_train_set = train_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)


                X_test_set = test_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_test_set = test_set[col.TARGET]
                cat_test_set = test_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)

                X_train_burak, y_train_burak =  classic_burakFilter(X_train_set.to_numpy(),
                                                                    y_train_set.to_numpy(),
                                                                    X_test_set.to_numpy(),
                                                                    10)

                mlflow.log_metric("Test Flaky", len(test_set[test_set['isFlaky']==1]))
                mlflow.log_metric("Test Non Flaky", len(test_set[test_set['isFlaky']==0]))
                print("TF:{} - TNF:{}".format(len(test_set[test_set['isFlaky']==1]), len(test_set[test_set['isFlaky']==0])))

                pipeline.fit(X_train_burak,y_train_burak)
                y_predict=pipeline.predict(X_test_set)
                validation_utils.val_and_log_metrics(y_test_set,y_predict)








if __name__ == "__main__":
    run(sys.argv[1])




