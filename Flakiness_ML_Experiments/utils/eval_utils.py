from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import mlflow

def eval_and_log_metrics(prefix,  y_true, y_predict):
    acc=accuracy_score(y_true,y_predict)
    pr=precision_score(y_true,y_predict)
    rec=recall_score(y_true,y_predict)
    f1=f1_score(y_true,y_predict)

    print("    |--- {}_ACC".format(prefix),acc)
    print("    |--- {}_PR".format(prefix),pr)
    print("    |--- {}_REC".format(prefix),rec)
    print("    |--- {}_F1".format(prefix),f1)

    mlflow.log_metric("{}_ACC".format(prefix),acc)
    mlflow.log_metric("{}_PR".format(prefix),pr)
    mlflow.log_metric("{}_REC".format(prefix),rec)
    mlflow.log_metric("{}_F1".format(prefix),f1)
    #Aggiungere Validation True label & Predict label

    return acc,pr,rec,f1