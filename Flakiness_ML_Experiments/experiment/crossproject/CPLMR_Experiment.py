import copy
import os

import mlflow
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from LocalModel import LocalModel
from experiment.LocalModelExperiment import LocalModelExperiment


class CPLCR_Experiment(LocalModelExperiment):

    def __init__(self,experimentName,typeExperiment):
        super().__init__(experimentName, typeExperiment)



def runExperiment(self,X_train_set,y_train_set,repository_train_set,X_test_set,y_test_set,mlflowExperimentName,log_mlflow=True):

    if log_mlflow:
        experiment_ID=None
        experiment =mlflow.get_experiment_by_name(mlflowExperimentName)
        if not experiment:
            experiment_ID=mlflow.create_experiment(mlflowExperimentName)
        else:
            experiment_ID=experiment.experiment_id

    if log_mlflow:
        with mlflow.start_run(run_name=self.experimentName, experiment_id=experiment_ID) as run:
            self.run(X_train_set,y_train_set,repository_train_set,X_test_set,y_test_set)
            self.logMlFlow()
            mlflow.end_run()
    else:
        self.run(X_train_set,y_train_set,repository_train_set,X_test_set,y_test_set)




def run(self,X_train_set,y_train_set,repository_set,X_test_set,y_test_set):

        list_repository=repository_set.unique()
        for repository in list_repository:
            X_repository=X_train_set.loc[repository_set==repository]
            y_repository=y_train_set.loc[repository_set==repository]
            pipeline=Pipeline(steps =[('scaler',StandardScaler()),
                                      ('clf',RandomForestClassifier(class_weight='balanced', random_state=42))])
            pipeline.fit(X_repository,y_repository)

            localModel=LocalModel(repository,X_repository,y_repository,copy.copy(pipeline))
            localModel.setPerformanceLocalModel(y_repository,pipeline.predict(X_repository))
            super().addLocalModel(localModel)


        list_centroid=[]
        for localModel in self.list_localModel:
            centroid=[]
            X_repository=localModel.getX_set()
            for column in X_repository.columns:
                centroid.append(X_repository[column].mean())
            list_centroid.append(centroid)

        y_predict=[]
        indexs_repo=[]
        for i in range(len(X_test_set)):
            X_target_instance=X_test_set.iloc[i, ].to_numpy()
            X_target_instance=X_target_instance.reshape(1,-1)
            distances = np.sqrt(np.sum((X_target_instance - list_centroid)**2, axis=1))
            index_repo=np.argmin(distances,axis=0)
            pipeline=self.list_localModel[index_repo]
            y_predict.append(pipeline.predict(X_target_instance)[0])

        super().__setPerformance(y_test_set,y_predict)


def logMlFlow(self):
    mlflow.set_tag("TipoEsperimento",self.getTypeExperiment())
    mlflow.log_metric("TP",self.getTP())
    mlflow.log_metric("FP",self.getFP())
    mlflow.log_metric("TN",self.getTN())
    mlflow.log_metric("FN",self.getFN())
    mlflow.log_metric("ACC",self.accuracy)
    mlflow.log_metric("PRE",self.precision)
    mlflow.log_metric("REC",self.recall)
    mlflow.log_metric("F1",self.f1)
    mlflow.log_metric("AUC",self.getAUC())
    mlflow.log_metric("MCC",self.getMCC())
    mlflow.log_metric("Numero LocalModel",len(self.list_localModel))

    self.getSummaryLocalModel().to_csv('LocalModelSummary.csv',index=False)
    mlflow.log_artifact('LocalModelSummary.csv')
    os.remove('LocalModelSummary.csv')


def toString(self):
    return super().toString()