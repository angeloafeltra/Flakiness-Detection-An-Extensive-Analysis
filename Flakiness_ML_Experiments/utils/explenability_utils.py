import utils.columns as col
import pandas as pd
import mlflow
import numpy as np
import os


def calculate_distribution(set):

    data={
        'column':[],

        'Min':[],
        'Max': [],
        'Range': [],
        'Inter-Quartile Range': [],
        'Mean': [],
        'Median': [],
        'Variance': [],
        'STD': [],
        'Skewness': [],
        'Kurtosis': []
    }

    for columnName in set.columns:

        if columnName==col.TARGET or columnName in col.CATEGORICAL_FEATURES: continue
        data['column'].append(columnName)
        data['Min'].append(set[columnName].min())
        data['Max'].append(set[columnName].max())
        data['Range'].append(set[columnName].max() - set[columnName].min())
        quantileS=set[columnName].quantile([0.25, 0.75])
        data['Inter-Quartile Range'].append(quantileS[0.75] - quantileS[0.25])
        data['Mean'].append(set[columnName].mean())
        data['Median'].append(set[columnName].median())
        data['Variance'].append(set[columnName].var())
        data['STD'].append(set[columnName].std())
        data['Skewness'].append(set[columnName].skew())
        data['Kurtosis'].append(set[columnName].kurtosis())



    df=pd.DataFrame(data)
    return df


def features_importance(clf):

    data={
        'features': [],
        'importance': []
    }

    importanceFeatures=clf.feature_importances_
    indices = np.argsort(importanceFeatures)[::-1]
    for i in indices:
        data['features'].append(col.NUMERICAL_FEATURES[i])
        data['importance'].append(importanceFeatures[i])

    df=pd.DataFrame(data)
    return df


def log_distribution(set,path,folder):
    df=calculate_distribution(set)
    df.to_csv(path,index=False)
    mlflow.log_artifact(path,folder)
    os.remove(path)


def log_featureImportance(clf,path,folder):
    fi=features_importance(clf)
    fi.to_csv(path,index=False)
    mlflow.log_artifact(path,folder)
    os.remove(path)