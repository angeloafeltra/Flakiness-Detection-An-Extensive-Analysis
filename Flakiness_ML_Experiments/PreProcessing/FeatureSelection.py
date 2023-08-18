from sklearn.base import BaseEstimator, TransformerMixin
from utils.multicollinearity import multicollinearity_eppsilon_feature
from utils.informationGain import informationGain_epplsilon_feature
import pandas as pd
import numpy as np
import copy
import mlflow

class FeatureSelection(BaseEstimator, TransformerMixin):


    def fit(self, X, y=None):
        self.eppsilon_features=[]
        X_copy=copy.copy(X)
        if isinstance(X_copy, np.ndarray):
            X_copy = pd.DataFrame(X_copy)

        #Multicollinearity
        self.eppsilon_features=self.eppsilon_features+multicollinearity_eppsilon_feature(X_copy)
        X_copy.drop(self.eppsilon_features,axis=1)
        #Information Gain
        self.eppsilon_features=self.eppsilon_features+informationGain_epplsilon_feature(X_copy,y)
        mlflow.log_param("Eppsilon_features",len(self.eppsilon_features))
        return self



    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X=X.drop(self.eppsilon_features,axis=1)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X,y)
        X=self.transform(X)
        return X

