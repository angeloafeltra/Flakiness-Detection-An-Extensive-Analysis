import copy
import numpy as np
import pandas as pd
from experiment.CPFP_Experiment import CPFP_Experiment
from experiment.crossproject.CPBF_Experiment import CPBF_Experiment
from experiment.crossproject.CPTCA import CPTCA_Experiment
from experiment.crossproject.CPCTCA import CPCTCA_Experiment
from validator.Validator import Validator
import numpy as np

class CPFP_Validator(Validator):


    def __init__(self): pass


    def validation(self,listRepository,pipeline,mlflowExperimentName,log_mlflow=True):
        tmpPRE=[]
        tmpREC=[]
        tmpF1=[]

        for repository in listRepository:
            repository_name=repository[0].split('_')[0]


            X_targetSet=repository[1]
            y_targetSet=repository[2]

            if not self.filtro(y_targetSet): continue

            tmpX=[]
            tmpY=[]
            for source_repository in listRepository:
                if source_repository[0]!=repository[0]:
                    tmpX.append(source_repository[1])
                    tmpY.append(source_repository[2])

            X_sourceSet=pd.concat(tmpX,axis=0)
            y_sourceSet=pd.concat(tmpY,axis=0)


            cpfp_experiment=CPFP_Experiment('CPFP_{}'.format(repository_name),'CrossProject_Validation',repository_name,pipeline)
            cpfp_experiment.runExperiment(X_sourceSet,y_sourceSet,X_targetSet,y_targetSet,mlflowExperimentName,log_mlflow)
            print(cpfp_experiment.toString())

            tmpPRE.append(cpfp_experiment.getPrecision())
            tmpREC.append(cpfp_experiment.getRecall())
            tmpF1.append(cpfp_experiment.getF1())


        print("Prestazioni Medie - Pr:{} - Rec:{} - F1:{}".format(
            np.mean(tmpPRE), np.mean(tmpREC), np.mean(tmpF1)))



