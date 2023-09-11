import copy
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import utils.columns as col




def val_and_log_metrics(y_true, y_predict, prefix="TEST"):

    acc = accuracy_score(y_true, y_predict)
    pr = precision_score(y_true, y_predict)
    rec = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)


    print("    |--- {}_ACC".format(prefix), acc)
    print("    |--- {}_PR".format(prefix), pr)
    print("    |--- {}_REC".format(prefix), rec)
    print("    |--- {}_F1".format(prefix), f1)


    mlflow.log_metric("{}_ACC".format(prefix), acc)
    mlflow.log_metric("{}_PR".format(prefix), pr)
    mlflow.log_metric("{}_REC".format(prefix), rec)
    mlflow.log_metric("{}_F1".format(prefix), f1)


    if (type(y_true) is np.ndarray and len(np.unique(y_true))>1) or (type(y_true) is pd.Series and y_true.nunique()>1):

        auc = roc_auc_score(y_true, y_predict)
        tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
        fpr, tpr, _ = roc_curve(y_true,  y_predict)

        print("    |--- {}_AUC".format(prefix), auc)

        mlflow.log_metric("{}_AUC".format(prefix), auc)
        mlflow.log_metric("{}_TN".format(prefix), tn)
        mlflow.log_metric("{}_FP".format(prefix), fp)
        mlflow.log_metric("{}_FN".format(prefix), fn)
        mlflow.log_metric("{}_TP".format(prefix), tp)


        plt.clf()
        plt.plot(fpr,tpr,label="auc="+str(auc))
        plt.legend(loc=4)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        mlflow.log_figure(plt.gcf(),"ROC-AUC Curve {}.png".format(prefix))

        plt.clf()
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_predict),
                                      display_labels=[0,1])
        disp.plot()
        mlflow.log_figure(plt.gcf(),"Confusion Matrix {}.png".format(prefix))

        return acc, pr, rec, f1, tn, fp, fn, tp

    return acc, pr, rec, f1


def soa_validation(dataset, pipeline):

    skf= StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    test_predict=[]
    auc_fold=[]
    for train_index, test_index in skf.split(dataset, dataset[col.TARGET]):

        train, test= dataset.iloc[train_index], dataset.iloc[test_index]
        X_train= train.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
        y_train= train[col.TARGET]
        X_test= test.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
        y_test= test[col.TARGET]

        pipeline.fit(X=X_train,y=y_train)

        y_predict=pipeline.predict(X_test)
        test['Predict']=y_predict
        test_predict.append(copy.copy(test))
        auc = roc_auc_score(test[col.TARGET], y_predict)
        auc_fold.append(auc)



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


    prediction_set=pd.concat(test_predict,axis=0)
    for repository in prediction_set[col.CATEGORICAL_FEATURES[0]].unique():

        repo_name=repository.split("_")[0]
        repo_set=prediction_set.loc[prediction_set[col.CATEGORICAL_FEATURES[0]]==repository]
        tnf=repo_set.loc[repo_set[col.TARGET]==0].shape[0]
        tf=repo_set.loc[repo_set[col.TARGET]==1].shape[0]

        CM = confusion_matrix(y_true=repo_set[col.TARGET], y_pred=repo_set['Predict'])
        TN, FN, TP, FP = CM[0][0], CM[1][0], CM[1][1], CM[0][1]

        pre=precision_score(y_true=repo_set[col.TARGET], y_pred=repo_set['Predict'])*100
        rec=recall_score(y_true=repo_set[col.TARGET], y_pred=repo_set['Predict'])*100
        f1=f1_score(y_true=repo_set[col.TARGET], y_pred=repo_set['Predict'])*100

        data['Repository'].append(repo_name)
        data['TNF'].append(tnf)
        data['TF'].append(tf)
        data['TP'].append(TP)
        data['FN'].append(FN)
        data['FP'].append(FP)
        data['TN'].append(TN)
        data['Pre'].append(pre)
        data['Rec'].append(rec)
        data['F1'].append(f1)

        print("Repository:{}    TNF:{}   TF:{}   TP:{}   FN:{}   FP:{}   TN:{}   Pr:{}   Rec:{}   F1:{}".format(
            repo_name, tnf, tf, TP, FN, FP, TN, pre, rec, f1 ))


    CM = confusion_matrix(y_true=prediction_set[col.TARGET], y_pred=prediction_set['Predict'])
    TN, FN, TP, FP = CM[0][0], CM[1][0], CM[1][1], CM[0][1]

    pre=precision_score(y_true=prediction_set[col.TARGET], y_pred=prediction_set['Predict'])*100
    rec=recall_score(y_true=prediction_set[col.TARGET], y_pred=prediction_set['Predict'])*100
    f1=f1_score(y_true=prediction_set[col.TARGET], y_pred=prediction_set['Predict'])*100

    data['Repository'].append('Total')
    data['TNF'].append(tnf)
    data['TF'].append(tf)
    data['TP'].append(TP)
    data['FN'].append(FN)
    data['FP'].append(FP)
    data['TN'].append(TN)
    data['Pre'].append(pre)
    data['Rec'].append(rec)
    data['F1'].append(f1)

    print("Repository:{}    TNF:{}   TF:{}   TP:{}   FN:{}   FP:{}   TN:{}   Pr:{}   Rec:{}   F1:{}".format(
        'Total', tnf, tf, TP, FN, FP, TN, pre, rec, f1 ))

    df=pd.DataFrame(data)
    df.to_csv('Cross-Validation Dataset.csv',index=False)
    mlflow.log_artifact('Cross-Validation Dataset.csv')
    os.remove('Cross-Validation Dataset.csv')

    print('AUC Mean: {}'.format(np.mean(auc_fold)))
    mlflow.log_metric("AUC mean",np.mean(auc_fold))
