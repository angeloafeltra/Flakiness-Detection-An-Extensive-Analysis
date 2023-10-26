import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from experiment.Experiment import Experiment
import mlflow

class WPFP_Experiment(Experiment):

    repositoryTarget=None
    tf=None
    tnf=None

    def __init__(self,experimentName,typeExperiment,repositoryTarge,pipeline):
        super().__init__(experimentName,typeExperiment,pipeline)
        self.repositoryTarget=repositoryTarge

    def setTestFlaky(self,tf): self.tf=tf
    def setTestNonFlaky(self,tnf): self.tnf=tnf

    def getTestFlaky(self): return self.tf
    def getTestNonFlaky(self): return self.tnf



    def run(self,X_train_set,y_train_set,X_test_set,y_test_set):

        y_true=y_test_set.to_numpy()
        self.setTestNonFlaky(y_true.shape[0]-np.count_nonzero(y_true))
        self.setTestFlaky(np.count_nonzero(y_true))

        super().run(X_train_set,y_train_set,X_test_set,y_test_set)



    def logMlFlow(self):

        super().logMlFlow()
        mlflow.log_param("TestFlaky",self.getTestFlaky())
        mlflow.log_param("TestNonFlaky",self.getTestNonFlaky())

    def toString(self):

        txt="{} - TNF:{} - TF:{} - TP:{} - FN:{} - FP:{} - TN:{} - Pr:{} - Rec:{} - F1:{} - AUC:{} - MCC:{}".format(
            self.experimentName,
            self.getTestNonFlaky(),
            self.getTestFlaky(),
            self.getTP(),
            self.getFN(),
            self.getFP(),
            self.getTN(),
            self.getPrecision(),
            self.getRecall(),
            self.getF1(),
            self.getAUC(),
            self.getMCC())

        return txt




