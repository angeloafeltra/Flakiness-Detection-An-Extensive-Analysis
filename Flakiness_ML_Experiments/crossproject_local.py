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

def run(datasetName):
    warnings.filterwarnings("ignore")

    # Creo un nuovo esperimento su mlflow
    experiment =mlflow.get_experiment_by_name("FlakinessMLExperiment_{}".format(datasetName))
    if not experiment:
        experiment_ID=mlflow.create_experiment("FlakinessMLExperiment_{}".format(datasetName))
    else:
        experiment_ID=experiment.experiment_id


    all_run=mlflow.search_runs(experiment_ids=[experiment_ID]) #Ottengo tutte le run dell'esperimento
    row=all_run[all_run['metrics.Val_F1'] == all_run['metrics.Val_F1'].max()].head(1)
    pipeline=pickle.load(open(os.path.join('.',
                                            'mlruns',
                                            row['experiment_id'].to_string(index=False, header=False),
                                            row['run_id'].to_string(index=False, header=False),
                                            'artifacts',
                                            row['tags.mlflow.runName'].to_string(index=False, header=False),
                                            'model.pkl'),'rb'))

    dataset = pd.read_csv(os.path.join('..','Dataset','{}.csv'.format(datasetName)))
    list_project=dataset['nameProject'].unique()



    with mlflow.start_run(run_name='CrossProject_LocalModel',experiment_id= experiment_ID) as father_run:
        for project in list_project:
            print(project)
            with mlflow.start_run(run_name=project,experiment_id=experiment_ID,nested=True) as child_run:
                train_set=dataset.loc[dataset['nameProject']!=project].reset_index(drop=True)
                test_set=dataset.loc[dataset['nameProject']==project].reset_index(drop=True)

                X_train_set = train_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_train_set = train_set[col.TARGET]
                cat_train_set = train_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)


                X_test_set = test_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_test_set = test_set[col.TARGET]
                cat_test_set = test_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)



                #1. Eseguo il clustering
                km=KMeans(random_state=42, n_init='auto')
                elbow=KElbowVisualizer(km, k=(2,20))
                elbow.fit(X=X_train_set)
                ideal_cluster=elbow.elbow_value_
                mlflow.log_metric("Numero Cluster", ideal_cluster)
                print("Numero Cluster: {}".format(ideal_cluster))
                km=KMeans(random_state=42, n_clusters=ideal_cluster, n_init='auto')
                km.fit(X=X_train_set)
                X_train_set['cluster']=km.predict(X=X_train_set)

                local_model={
                    'Index_Cluster': [],
                    'Cluster': [],
                    'Local pipeline': []
                }

                #2. Per ogni cluster addestro un modello
                for i in range(0,ideal_cluster):
                    cluster=X_train_set.loc[X_train_set['cluster']==i]
                    local_pip=copy.copy(pipeline)
                    local_pip.fit(X_train_set[col.NUMERICAL_FEATURES],y_train_set)

                    local_model['Index_Cluster'].append(i)
                    local_model['Cluster'].append(cluster)
                    local_model['Local pipeline'].append(local_pip)


                #3. Per ogni campione identifico il cluster d'appartenenza e eseguo la prediction
                y_predict=[]
                for i in range(len(X_test_set)):
                    row=X_test_set.iloc[i, ].to_numpy()
                    row=row.reshape(1,26)
                    index_cluster=km.predict(X=row)
                    local_pip=local_model['Local pipeline'][int(index_cluster)]
                    y_predict.append(local_pip.predict(row))


                mlflow.log_metric("Test Flaky", len(test_set[test_set['isFlaky']==1]))
                mlflow.log_metric("Test Non Flaky", len(test_set[test_set['isFlaky']==0]))
                print("TF:{} - TNF:{}".format(len(test_set[test_set['isFlaky']==1]), len(test_set[test_set['isFlaky']==0])))
                validation_utils.val_and_log_metrics(y_test_set,y_predict)




if __name__ == "__main__":
    run(sys.argv[1])




