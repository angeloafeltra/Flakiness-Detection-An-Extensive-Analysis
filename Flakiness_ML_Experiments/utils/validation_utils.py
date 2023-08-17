from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt



import mlflow

def val_and_log_metrics(y_true, y_predict):
    acc = accuracy_score(y_true, y_predict)
    pr = precision_score(y_true, y_predict)
    rec = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    auc = roc_auc_score(y_true, y_predict)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    fpr, tpr, _ = roc_curve(y_true,  y_predict)

    print("    |--- TEST_ACC",acc)
    print("    |--- TEST_PR",pr)
    print("    |--- TEST_REC",rec)
    print("    |--- TEST_F1",f1)
    print("    |--- TEST_AUC",auc)

    mlflow.log_metric("Test_ACC", acc)
    mlflow.log_metric("Test_PR", pr)
    mlflow.log_metric("Test_REC", rec)
    mlflow.log_metric("Test_F1", f1)
    mlflow.log_metric("Test_AUC", auc)
    mlflow.log_metric("TN", tn)
    mlflow.log_metric("FP", fp)
    mlflow.log_metric("FN", fn)
    mlflow.log_metric("TP", tp)

    plt.plot(fpr,tpr,label="auc="+str(auc))
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return acc, pr, rec, f1, tn, fp, fn, tp