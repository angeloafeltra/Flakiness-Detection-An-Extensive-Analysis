import numpy as np
from sklearn.ensemble import RandomForestClassifier


def informationGain_epplsilon_feature(X,y):
    eppsilon_feature=[]

    randomForest=RandomForestClassifier(n_estimators=200,random_state=0,n_jobs=1)
    columns_name=X.columns  #Nome Features
    randomForest.fit(X=X,y=y)
    importanceFeatures=randomForest.feature_importances_
    indices = np.argsort(importanceFeatures)[::-1] #Ordino gli indici in maniera decrescente


    for f in range(X.shape[1]):
        if importanceFeatures[indices[f]] < 0.02:
            eppsilon_feature.append(X.columns[indices[f]])

    return eppsilon_feature