import sys

import pandas as pd

import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from utils.crossproject_utils import calculate_distribution, features_importance
import os
from utils.eval_utils import eval_and_log_metrics

def run(dataset, pipeline, experiment_ID):

    list_project=dataset['nameProject'].unique()

    with mlflow.start_run(run_name='CrossProject_LocalModel_Clustering',experiment_id= experiment_ID) as father_run:

        for target in list_project:
            print(target)

            #Inserire qui il criterio di filtraggio per skippare il target
            tmp=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]
            testFlaky=len(tmp[tmp[col.TARGET]==1])
            testNonFlaky=len(tmp[tmp[col.TARGET]==0])

            if testFlaky<20 or testFlaky>testNonFlaky:
                print("La repository non rispetta i criteri")
                continue

            with mlflow.start_run(run_name=target,experiment_id=experiment_ID,nested=True) as child_run:

                source_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]!=target].reset_index(drop=True)
                target_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target].reset_index(drop=True)

                source_TF=len(source_set[source_set[col.TARGET]==1])
                source_TNF=len(source_set[source_set[col.TARGET]==0])
                target_TF=len(target_set[target_set[col.TARGET]==1])
                target_TNF=len(target_set[target_set[col.TARGET]==0])

                mlflow.log_metric("Source Test Flaky", source_TF)
                mlflow.log_metric("Source Test Non Flaky", source_TNF)
                mlflow.log_metric("Target Test Flaky", target_TF)
                mlflow.log_metric("Target Test Non Flaky", target_TNF)

                print("Source TF:{} - TNF:{}\nTarget TF:{} - TNF:{} ".format(source_TF,
                                                                              source_TNF,
                                                                              target_TF,
                                                                              target_TNF))

                X_source_set = source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_source_set = source_set[col.TARGET]


                X_target_set = target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_target_set = target_set[col.TARGET]



                #1. Eseguo il clustering
                #1.1 Sfrutto l'elbow point per calcolare il numero di cluster ideali
                km=KMeans(random_state=42, n_init='auto')
                elbow=KElbowVisualizer(km, k=(2,20))
                elbow.fit(X=X_source_set)
                ideal_cluster=elbow.elbow_value_


                mlflow.log_metric("Numero Cluster", ideal_cluster)
                print("Numero Cluster: {}".format(ideal_cluster))

                #1.2 Creo i cluster
                km=KMeans(random_state=42, n_clusters=ideal_cluster, n_init='auto')
                km.fit(X=X_source_set)
                X_source_set['cluster']=km.predict(X=X_source_set)


                #2. Per ogni cluster addestro un modello locale
                local_model={
                    'Index_Cluster': [],
                    'X_cluster': [],
                    'y_cluster': [],
                    'Local pipeline': []
                }

                for i in range(0,ideal_cluster):
                    X_cluster_set=X_source_set.loc[X_source_set['cluster']==i]
                    X_cluster_set=X_cluster_set.drop(['cluster'],axis=1)
                    indices=X_source_set.index[X_source_set['cluster'] == i].tolist()
                    y_cluster_set=y_source_set.iloc[indices]

                    clusterTF=np.count_nonzero(y_cluster_set.to_numpy())
                    clusterTNF=y_cluster_set.to_numpy().size-clusterTF

                    mlflow.log_metric("Cluster {} test flaky".format(i), clusterTF)
                    mlflow.log_metric("Cluster {} test non flaky".format(i), clusterTNF)
                    print("Cluster {} TF:{} - TNF:{}".format(i,
                                                             clusterTF,
                                                             clusterTNF))


                    X_cluster_train, X_cluster_test, y_cluster_train, y_cluster_test = train_test_split(X_cluster_set, y_cluster_set,
                                                                                                        stratify = y_cluster_set,
                                                                                                        test_size = 0.2,
                                                                                                        random_state = 42)


                    local_pip=copy.copy(pipeline)
                    local_pip.fit(X_cluster_train,y_cluster_train)
                    y_predict=local_pip.predict(X_cluster_train)
                    eval_and_log_metrics("Cluster {} ".format(i),y_cluster_train,y_predict)

                    y_predict=local_pip.predict(X_cluster_test)
                    validation_utils.val_and_log_metrics(y_cluster_test,y_predict,'Cluster {}'.format(i))

                    local_model['Index_Cluster'].append(i)
                    local_model['X_cluster'].append(X_cluster_set)
                    local_model['y_cluster'].append(y_cluster_set)
                    local_model['Local pipeline'].append(copy.copy(local_pip))

                    #Explenability Cluster
                    cluster_set=pd.concat([X_cluster_set, y_cluster_set], axis=1)
                    df=calculate_distribution(cluster_set,target_set)
                    df.to_csv('Distribution Cluster {}-Target.csv'.format(i),index=False)
                    mlflow.log_artifact('Distribution Cluster {}-Target.csv'.format(i),'Distribution')
                    os.remove('Distribution Cluster {}-Target.csv'.format(i))

                    df=calculate_distribution(cluster_set.loc[cluster_set[col.TARGET]==0],
                                              target_set.loc[target_set[col.TARGET]==0])
                    df.to_csv('Distribution Non Flaky Test Cluster {}-Target.csv'.format(i),index=False)
                    mlflow.log_artifact('Distribution Non Flaky Test Cluster {}-Target.csv'.format(i),'Distribution')
                    os.remove('Distribution Non Flaky Test Cluster {}-Target.csv'.format(i))

                    df=calculate_distribution(cluster_set.loc[cluster_set[col.TARGET]==1],
                                              target_set.loc[target_set[col.TARGET]==1])
                    df.to_csv('Distribution Flaky Test Cluster {}-Target.csv'.format(i),index=False)
                    mlflow.log_artifact('Distribution Flaky Test Cluster {}-Target.csv'.format(i),'Distribution')
                    os.remove('Distribution Flaky Test Cluster {}-Target.csv'.format(i))


                    fi=features_importance(local_pip.get_params('steps')['model'])
                    fi.to_csv('Feature Importances Classifier Cluster {}.csv'.format(i),index=False)
                    mlflow.log_artifact('Feature Importances Classifier Cluster {}.csv'.format(i),'Feature Importances Classifier')
                    os.remove('Feature Importances Classifier Cluster {}.csv'.format(i))


                #3. Per ogni campione identifico il cluster d'appartenenza e eseguo la prediction
                y_predict=[]
                indexs_cluster=[]
                for i in range(len(X_target_set)):
                    X_target_instance=X_target_set.iloc[i, ].to_numpy()
                    X_target_instance=X_target_instance.reshape(1,-1)
                    index_cluster=km.predict(X=X_target_instance)[0]
                    indexs_cluster.append(index_cluster)
                    local_pip=local_model['Local pipeline'][int(index_cluster)]
                    y_predict.append(local_pip.predict(X_target_instance)[0])


                validation_utils.val_and_log_metrics(y_target_set,y_predict,'Target')

                df=pd.DataFrame()
                df['True Lable']=y_target_set
                df['Predict Lable']=y_predict
                df['Cluster']=indexs_cluster
                df.to_csv('Target Predict Log.csv',index=False)
                mlflow.log_artifact('Target Predict Log.csv','Target Predict Log')
                os.remove('Target Predict Log.csv')




if __name__ == "__main__":
    run(sys.argv[1])




