from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import copy
from experiment.Experiment import Experiment
from imblearn.pipeline import Pipeline

class Custom_Tuning:

    TUNING = [

        {   'classificatore' : KNeighborsClassifier().__class__.__name__,
            'tuning_param' : {
                'n_neighbors': [9,15,21],
                'metric': ['euclidean','manhattan']
            }
            },
        {
            'classificatore' : SVC().__class__.__name__,
            'tuning_param' : {
                'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
        },
        {
            'classificatore' : LogisticRegression().__class__.__name__,
            'tuning_param' : {
                'C': [0.001, 0.01, 0.1, 1, 10, 100]
            }
        },
        {
            'classificatore' : DecisionTreeClassifier().__class__.__name__,
            'tuning_param' : {
                'criterion': ['gini','entropy'],
                'max_depth': [None, 4, 6, 8, 10]
            }
        },
        {
            'classificatore' : RandomForestClassifier().__class__.__name__,
            'tuning_param' : {
                'criterion': ['gini','entropy'],
                'max_depth': [None, 4, 6, 8, 10],
                'n_estimators': [100, 150, 200, 300]
            }
        },
        {
            'classificatore' : XGBClassifier().__class__.__name__,
            'tuning_param' : {

            }
        },
        {
            'classificatore' : Perceptron().__class__.__name__,
            'tuning_param' : {
                'penalty': ['l2', 'l1', None]
            }
        }
    ]

    def __init__(self): pass

    def runTuning(self,oldPipeline,namePipeline,X_train_set,y_train_set,X_test_set,y_test_set,mlflowExperimentName,print_experiment=True,log_mlflow=True):

        listExperiment=list()


        for step in oldPipeline.steps:
            if step[0]=='clf':
                estimator_name=step[1].__class__.__name__

        for tuningParam in self.TUNING:
            if tuningParam['classificatore']==estimator_name:
                gridParam=tuningParam['tuning_param']



        list_params=ParameterGrid(gridParam)
        for param,i in zip(list(list_params),range(0,len(list_params))):

            #Creo una nuova pipeline
            stepsPipeline=list()
            for step in oldPipeline.steps:
                if step[0]=='clf':
                    stepsPipeline.append(('clf',copy.copy(step[1].set_params(**param))))
                else:
                    stepsPipeline.append(copy.copy(step))
            pipeline=Pipeline(steps=stepsPipeline)


            exp=Experiment('{}_Tuning{}'.format(namePipeline,i),'Tuning',copy.copy(pipeline))
            exp.runExperiment(X_train_set,
                              y_train_set,
                              X_test_set,
                              y_test_set,
                              mlflowExperimentName,
                              log_mlflow)
            if print_experiment: print(exp.toString())
            listExperiment.append(exp)

        return listExperiment


