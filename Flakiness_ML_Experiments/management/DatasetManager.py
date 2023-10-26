import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


class DatasetManager:

    NUMERICAL_FEATURES = [
        "tloc",
        "tmcCabe",
        "assertionDensity",
        "assertionRoulette",
        "mysteryGuest",
        "eagerTest",
        "sensitiveEquality",
        "resourceOptimism",
        "conditionalTestLogic",
        "fireAndForget",
        "testRunWar",
        "loc",
        "lcom2",
        "lcom5",
        "cbo",
        "wmc",
        "rfc",
        "mpc",
        "halsteadVocabulary",
        "halsteadLength",
        "halsteadVolume",
        "classDataShouldBePrivate",
        "complexClass",
        "spaghettiCode",
        "functionalDecomposition",
        "godClass"
    ]
    CATEGORICAL_FEATURES = [
        "nameProject",
        "testCase"
    ]
    TARGET = 'isFlaky'
    trainTuple=None
    validationTuple=None
    testTuple=None
    datasetName=None

    def __init__(self,datasetName):
        self.dataset=self.__readDataset(datasetName)
        self.datasetName=datasetName


    def __readDataset(self,datasetName):
        dataset = pd.read_csv(os.path.join('..','Dataset','{}.csv'.format(datasetName)))
        return dataset

    def getDataset(self): return self.dataset
    def getDatasetName(self): return self.datasetName

    def getNumericalFeatures(self): return self.NUMERICAL_FEATURES
    def getCategoricalFeatures(self): return self.CATEGORICAL_FEATURES
    def getTargetFeature(self): return self.TARGET

    def getTrainSet(self): return self.trainTuple

    def getValidationSet(self): return self.validationTuple

    def getTestSet(self): return self.testTuple

    def dataset_split(self, val_size, test_size):
        X_dataset = self.dataset.drop([self.TARGET], axis = 1)
        y_dataset = self.dataset[self.TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset,
                                                            stratify = y_dataset,
                                                            test_size = test_size,
                                                            random_state = 42)

        X_train = X_train.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)

        X_test = X_test.reset_index(drop = True)
        y_test = y_test.reset_index(drop = True)
        test_set = pd.concat([X_test, y_test], axis = 1)

        cat_test= X_test.drop(self.NUMERICAL_FEATURES, axis = 1)
        X_test= X_test.drop(self.CATEGORICAL_FEATURES,axis=1)
        self.testTuple=(test_set,X_test,y_test,cat_test)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          stratify = y_train,
                                                          test_size = val_size,
                                                          random_state = 42)

        X_train = X_train.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        train_set = pd.concat([X_train, y_train], axis = 1)

        cat_train= X_train.drop(self.NUMERICAL_FEATURES, axis = 1)
        X_train= X_train.drop(self.CATEGORICAL_FEATURES,axis=1)
        self.trainTuple=(train_set,X_train,y_train,cat_train)

        X_val = X_val.reset_index(drop = True)
        y_val = y_val.reset_index(drop = True)
        val_set = pd.concat([X_val, y_val], axis = 1)

        cat_val= X_val.drop(self.NUMERICAL_FEATURES, axis = 1)
        X_val= X_val.drop(self.CATEGORICAL_FEATURES,axis=1)
        self.validationTuple=(val_set,X_val,y_val,cat_val)

    def getFold(self,num_fold):
        self.skf=StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)

        list_fold=list()
        for train_index, test_index in self.skf.split(self.dataset, self.dataset[self.TARGET]):
            train, test= self.dataset.iloc[train_index], self.dataset.iloc[test_index]
            X_train= train.drop([self.TARGET] + self.CATEGORICAL_FEATURES, axis = 1)
            y_train= train[self.TARGET]
            repo_train=train[self.CATEGORICAL_FEATURES[0]]
            X_test= test.drop([self.TARGET] + self.CATEGORICAL_FEATURES, axis = 1)
            y_test= test[self.TARGET]
            repo_test=test[self.CATEGORICAL_FEATURES[0]]

            list_fold.append((repo_train,X_train,y_train,repo_test,X_test,y_test))
        return list_fold

    def getTargetSet(self,repositoryTarget):
        repositySet=self.dataset.loc[self.dataset[self.getCategoricalFeatures()[0]]==repositoryTarget]
        repositySet=repositySet.reset_index(drop = True)
        X_repositorySet=repositySet.drop([self.TARGET] + self.CATEGORICAL_FEATURES,axis=1)
        y_repositorySet=repositySet[self.TARGET]
        cat_repositorySet=repositySet.drop([self.TARGET] + self.NUMERICAL_FEATURES,axis=1)
        return (repositySet,X_repositorySet,y_repositorySet,cat_repositorySet)

    def getSourceSet(self,repositoryTarget):
        sourceSet=self.dataset.loc[self.dataset[self.getCategoricalFeatures()[0]]!=repositoryTarget]
        sourceSet=sourceSet.reset_index(drop = True)
        X_sourceSet=sourceSet.drop([self.TARGET] + self.CATEGORICAL_FEATURES,axis=1)
        y_sourceSet=sourceSet[self.TARGET]
        cat_sourceSet=sourceSet.drop([self.TARGET] + self.NUMERICAL_FEATURES,axis=1)
        return (sourceSet,X_sourceSet,y_sourceSet,cat_sourceSet)

    def getListRepositorySet(self):
        list_repoSet=[]

        for repositoryTarget in self.dataset[self.CATEGORICAL_FEATURES[0]].unique():
            repositySet=self.dataset.loc[self.dataset[self.getCategoricalFeatures()[0]]==repositoryTarget]
            repositySet=repositySet.reset_index(drop = True)
            X_repositorySet=repositySet.drop([self.TARGET] + self.CATEGORICAL_FEATURES,axis=1)
            y_repositorySet=repositySet[self.TARGET]
            cat_repositorySet=repositySet.drop([self.TARGET] + self.NUMERICAL_FEATURES,axis=1)
            list_repoSet.append((repositoryTarget,X_repositorySet,y_repositorySet))
        return list_repoSet