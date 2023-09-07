import sys
import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
import copy
import numpy as np
import pandas as pd
import os


def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

def run(dataset, pipeline, experiment_ID):

    list_project=dataset['nameProject'].unique()

    with mlflow.start_run(run_name='CrossProject_LocalModel_Repository',experiment_id= experiment_ID) as father_run:

        for target in list_project:
            print(target)

            with mlflow.start_run(run_name=target,experiment_id=experiment_ID,nested=True) as child_run:
                source_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]!=target].reset_index(drop=True)
                target_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target].reset_index(drop=True)


                X_target_set = target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_target_set = target_set[col.TARGET]

                local_model={
                    'Repository': [],
                    'Centroid': [],
                    'Local pipeline': []
                }

                #1. Calcolo il centroide di ogni repository e addestro un modello per ogni repository
                centroid=[]
                for repo in source_set[col.CATEGORICAL_FEATURES[0]].unique():
                    subsource_set=source_set.loc[source_set[col.CATEGORICAL_FEATURES[0]]==repo]
                    X_subsource_set=subsource_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis=1)
                    y_subsource_set=subsource_set[col.TARGET]

                    for columnName in X_subsource_set.columns:
                        centroid.append(X_subsource_set[columnName].mean())

                    local_pip=copy.copy(pipeline)
                    local_pip.fit(X=X_subsource_set,y=y_subsource_set)

                    local_model['Repository'].append(repo)
                    local_model['Centroid'].append(copy.copy(centroid))
                    local_model['Local pipeline'].append(copy.copy(local_pip))
                    centroid.clear()



                #2. Per ogni campione identifico la repo e eseguo la prediction
                y_predict=[]
                for i in range(len(X_target_set)):
                    target_instance=X_target_set.iloc[i, ].to_numpy()
                    target_instance=target_instance.reshape(1,-1)
                    distances = euclidean(target_instance, local_model['Centroid'])
                    index_repo=np.argmin(distances,axis=0)
                    y_predict.append(local_model['Local pipeline'][index_repo].predict(target_instance))


                validation_utils.val_and_log_metrics(y_target_set,y_predict,'Target')
                df=pd.DataFrame()
                df['True Lable']=y_target_set
                df['Predict Lable']=y_predict
                df['Repository']=local_model['Repository'][index_repo]
                df.to_csv('Target Predict Log.csv')
                mlflow.log_artifact('Target Predict Log.csv','Target Predict Log')
                os.remove('Target Predict Log.csv')




if __name__ == "__main__":
    run(sys.argv[1])




