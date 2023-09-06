from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt
import mlflow

def val_and_log_metrics(y_true, y_predict, prefix="TEST"):
    acc = accuracy_score(y_true, y_predict)
    pr = precision_score(y_true, y_predict)
    rec = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    auc = roc_auc_score(y_true, y_predict)
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()
    fpr, tpr, _ = roc_curve(y_true,  y_predict)

    print("    |--- {}_ACC".format(prefix), acc)
    print("    |--- {}_PR".format(prefix), pr)
    print("    |--- {}_REC".format(prefix), rec)
    print("    |--- {}_F1".format(prefix), f1)
    print("    |--- {}_AUC".format(prefix), auc)

    mlflow.log_metric("{}_ACC".format(prefix), acc)
    mlflow.log_metric("{}_PR".format(prefix), pr)
    mlflow.log_metric("{}_REC".format(prefix), rec)
    mlflow.log_metric("{}_F1".format(prefix), f1)
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