import experiments.Pipeline1 as Pipeline1
import experiments.Pipeline1 as Pipeline2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Perceptron

EXPERIMENTS = [
    (Pipeline1,'Pipeline1')
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
