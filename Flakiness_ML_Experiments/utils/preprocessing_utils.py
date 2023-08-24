import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


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


def multicollinearity_eppsilon_feature(X):
    eppsilon_features=[]

    eliminato = True
    while eliminato:
        max = 0

        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]  # Calcolo il vif
        vif["features"] = X.columns

        for vif_value, feature in zip(vif["VIF Factor"], vif["features"]):
            if vif_value >= 5:
                if vif_value > max:
                    max = vif_value
                    feature_da_rimuovere = feature

        if max > 0:
            eliminato = True
            X = X.drop([feature_da_rimuovere], axis=1)
            eppsilon_features.append(feature_da_rimuovere)
        else:
            eliminato = False

    return eppsilon_features