import sys

import pandas as pd
import os
import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings
import pickle
import utils.validation_utils as validation_utils
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import copy
import numpy as np


def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

def run(dataset, pipeline, experiment_ID):

    list_project=dataset['nameProject'].unique()

    with mlflow.start_run(run_name='CrossProject_LocalModel_Repository',experiment_id= experiment_ID) as father_run:
        for project in list_project:
            print(project)
            with mlflow.start_run(run_name=project,experiment_id=experiment_ID,nested=True) as child_run:
                train_set=dataset.loc[dataset['nameProject']!=project].reset_index(drop=True)
                test_set=dataset.loc[dataset['nameProject']==project].reset_index(drop=True)


                X_test_set = test_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_test_set = test_set[col.TARGET]

                local_model={
                    'Repository': [],
                    'Centroid': [],
                    'Local pipeline': []
                }

                #1. Calcolo il centroide di ogni repository e addestro un modello per ogni repository
                centroid=[]
                for repo in train_set['nameProject'].unique():
                    repo_set=train_set.loc[train_set['nameProject']==repo]
                    X_repo_set=repo_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis=1)
                    y_repo_set=repo_set[col.TARGET]

                    for (columnName, columnData) in X_repo_set.iteritems():
                        centroid.append(X_repo_set[columnName].mean())

                    local_pip=copy.copy(pipeline)
                    local_pip.fit(X=X_repo_set,y=y_repo_set)

                    local_model['Repository'].append(repo)
                    local_model['Centroid'].append(copy.copy(centroid))
                    local_model['Local pipeline'].append(copy.copy(local_pip))
                    centroid.clear()



                #2. Per ogni campione identifico la repo e eseguo la prediction
                y_predict=[]
                for i in range(len(X_test_set)):
                    row=X_test_set.iloc[i, ].to_numpy()
                    row=row.reshape(1,26)
                    distances = euclidean(row, local_model['Centroid'])
                    index_repo=np.argmin(distances,axis=0)
                    y_predict.append(local_model['Local pipeline'][index_repo].predict(row))



                mlflow.log_metric("Test Flaky", len(test_set[test_set['isFlaky']==1]))
                mlflow.log_metric("Test Non Flaky", len(test_set[test_set['isFlaky']==0]))
                print("TF:{} - TNF:{}".format(len(test_set[test_set['isFlaky']==1]), len(test_set[test_set['isFlaky']==0])))
                validation_utils.val_and_log_metrics(y_test_set,y_predict)




if __name__ == "__main__":
    run(sys.argv[1])




