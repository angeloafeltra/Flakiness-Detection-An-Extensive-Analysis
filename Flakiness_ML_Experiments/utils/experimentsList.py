import experiments.Pipeline1 as Pipeline1
import experiments.Pipeline1 as Pipeline2
import experiments.Pipeline3 as Pipeline3
import experiments.Pipeline4 as Pipeline4
import experiments.Pipeline5 as Pipeline5
import experiments.Pipeline6 as Pipeline6
import experiments.Pipeline7 as Pipeline7
import experiments.Pipeline8 as Pipeline8
import experiments.Pipeline9 as Pipeline9
import experiments.Pipeline10 as Pipeline10
import experiments.Pipeline11 as Pipeline11
import experiments.Pipeline12 as Pipeline12
import experiments.Pipeline13 as Pipeline13
import experiments.Pipeline14 as Pipeline14
import experiments.Pipeline15 as Pipeline15
import experiments.Pipeline16 as Pipeline16
import experiments.Pipeline17 as Pipeline17
import experiments.Pipeline18 as Pipeline18

import crossproject_experiments.crossproject_classic as crossproject_classic
import crossproject_experiments.crossproject_one_source as crossproject_one_source
import crossproject_experiments.crossproject_burakfilter as burakfilter
import crossproject_experiments.crossproject_local_clustering as local_clustering
import crossproject_experiments.crossproject_local_repository as local_repository
import crossproject_experiments.crossproject_NSGRT as crossproject_NSGRT
import crossproject_experiments.crossproject_TCA as TCA
import crossproject_experiments.crossprojet_MIC_SM_FS as MIC_SM_FS


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron


EXPERIMENTS = [
    (Pipeline1,'Pipeline1'),
    (Pipeline2,'Pipeline2'),
    (Pipeline3,'Pipeline3'),
    (Pipeline4,'Pipeline4'),
    (Pipeline5,'Pipeline5'),
    (Pipeline6,'Pipeline6'),
    (Pipeline7,'Pipeline7'),
    (Pipeline8,'Pipeline8'),
    (Pipeline9,'Pipeline9'),
    (Pipeline10,'Pipeline10'),
    (Pipeline11,'Pipeline11'),
    (Pipeline12,'Pipeline12'),
    (Pipeline13,'Pipeline13'),
    (Pipeline14,'Pipeline14'),
    (Pipeline15,'Pipeline15'),
    (Pipeline16,'Pipeline16'),
    (Pipeline17,'Pipeline17'),
    (Pipeline18,'Pipeline18')
]


CLASSIFICATORI = [
    KNeighborsClassifier(),
    SVC(kernel='rbf', class_weight='balanced', random_state=42),
    LogisticRegression(class_weight='balanced', random_state=42),
    DecisionTreeClassifier(class_weight='balanced', random_state=42),
    RandomForestClassifier(class_weight='balanced', random_state=42),
    XGBClassifier(),
    Perceptron(class_weight='balanced', random_state=42)
]


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

CROSS_PROJECT_EXPERIMENT=[
    (crossproject_classic, 'crossproject_classic'),
    (burakfilter, 'burak_filter'),
    (local_clustering, 'localmodel_clustering'),
    (local_repository, 'localmodel_repository'),
    (TCA, 'TCA'),
    (MIC_SM_FS, 'MIC_SM_FS')
]
