import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from validator.Validator import Validator
from experiment.WPFP_Experiment import WPFP_Experiment

class WPFP_Validator(Validator):


    def __init__(self): pass

    def validation(self,listRepository,pipeline,mlflowExperimentName,log_mlflow=True):

        tmp25Pre=[]
        tmp25Rec=[]
        tmp25F1=[]

        tmp50Pre=[]
        tmp50Rec=[]
        tmp50F1=[]

        tmp75Pre=[]
        tmp75Rec=[]
        tmp75F1=[]


        for repository in listRepository:
            repository_name=repository[0].split('_')[0]
            X_targetSet=repository[1]
            y_targetSet=repository[2]

            if not self.filtro(y_targetSet): continue


            for i in [0.25,0.50,0.75]:

                tmpPRE=[]
                tmpREC=[]
                tmpF1=[]
                tmpAUC=[]


                for iter in range(10):
                    X_repo_set,X_test_set,y_repo_set,y_test_set=train_test_split(X_targetSet,
                                                                                 y_targetSet,
                                                                                 stratify=y_targetSet,
                                                                                 test_size=0.25,
                                                                                 random_state=iter)

                    trainSize=(i/0.75)*1
                    if trainSize!=1:
                        X_train_set,_,y_train_set,_=train_test_split(X_repo_set,
                                                                     y_repo_set,
                                                                     stratify=y_repo_set,
                                                                     test_size=1-trainSize,
                                                                     random_state=iter)
                    else:
                        X_train_set=X_repo_set
                        y_train_set=y_repo_set

                    wpfp_experiment=WPFP_Experiment('WPFP_{}{}_{}'.format(repository_name, int(i*100),iter),'WPFP_Validation',repository_name,pipeline)
                    wpfp_experiment.runExperiment(X_train_set,y_train_set,X_test_set,y_test_set,mlflowExperimentName,log_mlflow)

                    tmpPRE.append(wpfp_experiment.getPrecision())
                    tmpREC.append(wpfp_experiment.getRecall())
                    tmpF1.append(wpfp_experiment.getF1())


                print("{}_{} - Pr:{} - Rec:{} - F1:{} ".format(
                    repository_name, i*100, np.mean(tmpPRE), np.mean(tmpREC), np.mean(tmpF1)))

                if i==0.25:
                    tmp25Pre.append(np.mean(tmpPRE))
                    tmp25Rec.append(np.mean(tmpREC))
                    tmp25F1.append(np.mean(tmpF1))
                if i==0.50:
                    tmp50Pre.append(np.mean(tmpPRE))
                    tmp50Rec.append(np.mean(tmpREC))
                    tmp50F1.append(np.mean(tmpF1))
                if i==0.75:
                    tmp75Pre.append(np.mean(tmpPRE))
                    tmp75Rec.append(np.mean(tmpREC))
                    tmp75F1.append(np.mean(tmpF1))

        print("Prestazioni Medie 25 - Pr:{} - Rec:{} - F1:{}".format(
            np.mean(tmp25Pre),np.mean(tmp25Rec),np.mean(tmp25F1)
        ))
        print("Prestazioni Medie 50 - Pr:{} - Rec:{} - F1:{}".format(
            np.mean(tmp50Pre),np.mean(tmp50Rec),np.mean(tmp50F1)
        ))
        print("Prestazioni Medie 75 - Pr:{} - Rec:{} - F1:{}".format(
            np.mean(tmp75Pre),np.mean(tmp75Rec),np.mean(tmp75F1)
        ))