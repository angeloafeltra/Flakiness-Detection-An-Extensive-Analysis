import sys

from train_val_test_split import dataset_split
import pandas as pd
import os
import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings
import pickle
import utils.validation_utils as validation_utils

def run(dataset, pipeline, experiment_ID):

    list_project=dataset['nameProject'].unique()

    with mlflow.start_run(run_name='CrossProject_Validation_OneSource',experiment_id= experiment_ID) as father_run:
        for target in list_project:
            print(target)
            with mlflow.start_run(run_name=target,experiment_id=experiment_ID,nested=True) as child_run:
                train_set=dataset.loc[dataset['nameProject']!=target]
                test_set=dataset.loc[dataset['nameProject']==target]

                X_test_set = test_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_test_set = test_set[col.TARGET]
                cat_test_set = test_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)

                list_source=train_set['nameProject'].unique()
                for source in list_source:

                    with mlflow.start_run(run_name=source,experiment_id=experiment_ID,nested=True) as child_run2:
                        source_set=dataset.loc[dataset['nameProject']==source]

                        X_source_set = source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                        y_source_set = source_set[col.TARGET]
                        cat_source_set = source_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)


                        mlflow.log_metric("Test Flaky", len(test_set[test_set['isFlaky']==1]))
                        mlflow.log_metric("Test Non Flaky", len(test_set[test_set['isFlaky']==0]))
                        print("TF:{} - TNF:{}".format(len(test_set[test_set['isFlaky']==1]), len(test_set[test_set['isFlaky']==0])))

                        pipeline.fit(X_source_set,y_source_set)
                        y_predict=pipeline.predict(X_test_set)
                        validation_utils.val_and_log_metrics(y_test_set,y_predict)








if __name__ == "__main__":
    run(sys.argv[1])




