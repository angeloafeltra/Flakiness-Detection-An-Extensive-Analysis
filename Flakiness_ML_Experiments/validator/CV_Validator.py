import copy
import os

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import mlflow
from validator.Validator import Validator

class CV_Validator(Validator):

    data = {'Repository':[],
            'TNF': [],
            'TF': [],
            'TP': [],
            'FN': [],
            'FP': [],
            'TN': [],
            'Pre': [],
            'Rec': [],
            'F1': []
            }

    def __init__(self): pass

    def validation(self,folds,pipeline,log_mlflow=True):

        auc_fold=[]

        tmp_repository=[]
        tmp_true=[]
        tmp_predict=[]
        for fold in folds:
            pipeline.fit(X=fold[1],y=fold[2])
            y_predict=pipeline.predict(fold[4])
            auc = roc_auc_score(fold[5], y_predict)
            auc_fold.append(auc)

            label_set=copy.copy(fold[4])
            label_set['Predict']=y_predict
            tmp_repository.append(fold[3])
            tmp_true.append(fold[5])
            tmp_predict.append(label_set['Predict'])


        repository_set=pd.concat(tmp_repository,axis=0)
        true_set=pd.concat(tmp_true,axis=0)
        prediction_set=pd.concat(tmp_predict,axis=0)

        for repository in repository_set.unique():

            repo_name=repository.split("_")[0]

            y_true = true_set.loc[repository_set==repository]
            y_predict = prediction_set.loc[repository_set==repository]
            result=self.calcStat(y_true,y_predict)

            self.data['Repository'].append(repo_name)
            self.data['TNF'].append(result[0])
            self.data['TF'].append(result[1])
            self.data['TP'].append(result[2])
            self.data['FN'].append(result[3])
            self.data['FP'].append(result[4])
            self.data['TN'].append(result[5])
            self.data['Pre'].append(result[6])
            self.data['Rec'].append(result[7])
            self.data['F1'].append(result[8])

            print("{} - TNF:{} - TF:{} - TP:{} - FN:{} - FP:{} - TN:{} - Pr:{} - Rec:{} - F1:{}".format(
                repo_name, result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]))

        result=self.calcStat(true_set,prediction_set)


        self.data['Repository'].append('Total')
        self.data['TNF'].append(result[0])
        self.data['TF'].append(result[1])
        self.data['TP'].append(result[2])
        self.data['FN'].append(result[3])
        self.data['FP'].append(result[4])
        self.data['TN'].append(result[5])
        self.data['Pre'].append(result[6])
        self.data['Rec'].append(result[7])
        self.data['F1'].append(result[8])

        print("Total - TNF:{} - TF:{} - TP:{} - FN:{} - FP:{} - TN:{} - Pr:{} - Rec:{} - F1:{}".format(
            result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]))
        print('AUC Mean: {}'.format(np.mean(auc_fold)))

        if log_mlflow: self.logMlFlowData(np.mean(auc_fold))


    def calcStat(self,y_true,y_predict):

        y_true=y_true.to_numpy()

        tnf=y_true.shape[0]-np.count_nonzero(y_true)
        tf=np.count_nonzero(y_true)

        CM = confusion_matrix(y_true=y_true, y_pred=y_predict)
        TN, FN, TP, FP = CM[0][0], CM[1][0], CM[1][1], CM[0][1]

        pre=precision_score(y_true=y_true, y_pred=y_predict)*100
        rec=recall_score(y_true=y_true, y_pred=y_predict)*100
        f1=f1_score(y_true=y_true, y_pred=y_predict)*100
        auc=roc_auc_score(y_true, y_predict)*100

        return (tnf,tf,TP,FN,FP,TN,pre,rec,f1,auc)


    def logMlFlowData(self,auc_mean):

        last_run=mlflow.last_active_run()
        id_run=last_run.info.run_id

        with mlflow.start_run(run_id=id_run) as run:
            mlflow.log_metric('AUC Mean',auc_mean)
            df=pd.DataFrame(self.data)
            df.to_csv('Cross-Validation Dataset.csv',index=False)
            mlflow.log_artifact('Cross-Validation Dataset.csv')
            os.remove('Cross-Validation Dataset.csv')

            mlflow.end_run()