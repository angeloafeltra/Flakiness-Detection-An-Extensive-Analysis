import utils.columns as columns
import numpy as np
from utils import validation_utils
from scipy.spatial import KDTree
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import utils.columns as col
import utils.validation_utils as validation_utils

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier


import numpy as np
import utils.columns as col
from sklearn.utils import check_array
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import KERNEL_PARAMS
from scipy import linalg



def calculate_distribution(source_set,target_set):

    data={
        'column':[],

        'Mean_Source':[],
        'STD_Source': [],
        'Mean_Target':[],
        'STD_Target': [],

        'Min_Source':[],
        'Max_Source': [],
        'Min_Target':[],
        'Max_Target': [],
    }

    for columnName in col.NUMERICAL_FEATURES:
        data['column'].append(columnName)

        data['Mean_Source'].append(source_set[columnName].mean())
        data['STD_Source'].append(source_set[columnName].std())
        data['Mean_Target'].append(target_set[columnName].mean())
        data['STD_Target'].append(target_set[columnName].std())

        data['Min_Source'].append(source_set[columnName].min())
        data['Max_Source'].append(source_set[columnName].max())
        data['Min_Target'].append(target_set[columnName].min())
        data['Max_Target'].append(target_set[columnName].max())

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


def feature_similarity(source, target):
    fs=[]
    tmp1_source=[]
    tmp2_target=[]
    for column in col.NUMERICAL_FEATURES:
        tmpS=[]
        tmpT=[]

        #Min
        tmpS.append(source[column].min())
        tmpT.append(target[column].min())
        #Max
        tmpS.append(source[column].max())
        tmpT.append(target[column].max())
        #Range
        tmpS.append(source[column].max() - source[column].min())
        tmpT.append(target[column].max() - target[column].min())
        #Inter-quantile range
        quantileS=source[column].quantile([0.25, 0.75])
        tmpS.append(quantileS[0.75] - quantileS[0.25])
        quantileT=source[column].quantile([0.25, 0.75])
        tmpT.append(quantileT[0.75] - quantileT[0.25])
        #mean
        tmpS.append(source[column].mean())
        tmpT.append(target[column].mean())
        #median
        tmpS.append(source[column].median())
        tmpT.append(target[column].median())
        #variance
        tmpS.append(source[column].var())
        tmpT.append(target[column].var())
        #standard deivation
        tmpS.append(source[column].std())
        tmpT.append(target[column].std())
        #skewness
        tmpS.append(source[column].skew())
        tmpT.append(target[column].skew())
        #kurtosis
        tmpS.append(source[column].kurtosis())
        tmpT.append(target[column].kurtosis())

        tmp1_source.append(tmpS)
        tmp2_target.append(tmpT)


    for i in range(0,len(col.NUMERICAL_FEATURES)):
        somma=0
        for j in range(0, len(tmp1_source[0])):
            if tmp1_source[i][j]==0 and tmp2_target[i][j]==0:
                m=1
            else:
                if tmp1_source[i][j]<=tmp2_target[i][j]:
                    m=tmp1_source[i][j]/tmp2_target[i][j]
                else:
                    m=tmp2_target[i][j]/tmp1_source[i][j]
            somma=somma+m

        fs.append(somma/len(tmp1_source[0]))

    return fs

class NSGRT_PreProcessing:

    def __init__(self, n_neighbors):
        self.n_neighbors=n_neighbors


    def select_source(self,Xsource, ysource, Xtarget,ytarget=None):
        #Xsource,ysource= self._NNfilter(Xsource,ysource,Xtarget)
        Xsource,ysource= self._NNfilter_supervisioned(Xsource,ysource,Xtarget,ytarget)
        Xsource,ysource= self._SMOTE(Xsource,ysource)
        w=self._weightInizialization(Xsource,Xtarget)
        return Xsource,ysource,w



    def _euclidean(self,point, data):
        # Euclidean distance between points a & data
        return np.sqrt(np.sum((point - data)**2, axis=1))

    ##1. Filtro di burak
    def _NNfilter(self,Xsource,ysource,Xtarget):
        knn=KNeighborsClassifier()
        knn.fit(Xsource,ysource)

        Xdata=[]
        ydata=[]

        for instance in Xtarget:
            neighbors_index=knn.kneighbors(instance.reshape(1,-1), self.n_neighbors, return_distance=False)
            for neighbor_index in neighbors_index[0]:
                if not list(Xsource[neighbor_index]) in Xdata:
                    Xdata.append(list(Xsource[neighbor_index]))
                    ydata.append(ysource[neighbor_index])

        return np.asanyarray(Xdata), np.asarray(ydata)

    def _NNfilter_supervisioned(self,Xsource,ysource,Xtarget,ytarget):


        Xdata=[]
        ydata=[]

        for instance,label in zip(Xtarget,ytarget):
            indici_selezionati = ysource[:] == label
            sottomatrice = Xsource[indici_selezionati]
            distances = self._euclidean(instance, sottomatrice)
            indici_ordinati = np.argsort(distances)
            sottomatrice_sorted = sottomatrice[indici_ordinati]
            for neighbor in sottomatrice_sorted[0:10]:
                if not list(neighbor) in Xdata:
                    Xdata.append(list(neighbor))
                    ydata.append(label)
        return np.asanyarray(Xdata), np.asarray(ydata)

    ##2. SMOTE
    def _SMOTE(self,Xsource,ysource):
        smote=SMOTE(k_neighbors=4,random_state=42)
        Xsource, ysource=smote.fit_resample(Xsource,ysource)
        return Xsource, ysource

    def _max_min(self,X):
        #Calcolo il massimo ed il minimo di ogni colonna

        shape=np.asarray(X).shape
        max=np.zeros(shape[1])
        min=np.zeros(shape[1])
        for i in range(0,shape[1]):
            a=X[:,i]
            max[i]=np.max(a)
            min[i]=np.min(a)

        return max,min

    ##3. Gravitation law's weight inizialization
    def _weightInizialization(self,Xsource, Xtarget):
        max, min= self._max_min(Xtarget)

        shape=Xsource.shape

        s=np.zeros(shape[0])
        w=np.zeros(shape[0])

        for i in range(0,shape[0]):
            tmp=0
            for j in range(0,shape[1]):
                if Xsource[i][j]<=max[j] and Xsource[i][j]>=min[j]: tmp=tmp+1

            s[i]=tmp
            w[i]=s[i]/(1.0*np.power(shape[1] - s[i] + 1, 2))
        return w

class RareTransfer:

    def __init__(self, X_target,y_target,estimator):
        self.clf = estimator
        self.X_target=X_target
        self.y_target=y_target
        self.weight_target= np.ones([self.X_target.shape[0]])*self.X_target.shape[1]


    def fit(self, X_source, y_source, N, initWeight_source):

        # Preparo il set di training
        train_data = np.concatenate((X_source, self.X_target), axis=0)
        train_label = np.concatenate((y_source, self.y_target), axis=0)
        weight_source = initWeight_source

        # Initialize weights
        weights = np.concatenate((weight_source, self.weight_target), axis=0)

        beta_src = 1 / (1 + np.sqrt(np.log(2 * X_source.shape[0])/ N))

        for i in range(N):

            #Normalize weights
            weights= weights[:] / np.sum(weights)

            self.clf.fit(train_data,train_label,sample_weight=weights)
            y_predict=self.clf.predict(train_data)

            #Calcolo l'errore sul source
            error_rate_source = self._calculate_error_rate(y_source, y_predict[0:X_source.shape[0]], weights[0:X_source.shape[0]])
            #Claloclo l'errore sul target
            error_rate_target = self._calculate_error_rate(self.y_target, y_predict[X_source.shape[0]:X_source.shape[0] + self.X_target.shape[0]], weights[X_source.shape[0]:X_source.shape[0] + self.X_target.shape[0]])
            #print('error rate target: {}'.format(error_rate_target))

            Cl = 1 - error_rate_source #label-dependent cost factor
            beta_target =  error_rate_target/(1 - error_rate_target)

            # Aggiusto pesi source
            for j in range(X_source.shape[0]):
                weights[j] = Cl*weights[j] * np.power(beta_src, np.abs(y_predict[j] - y_source[j]))
            # Aggiusto pesi target
            for j in range(self.X_target.shape[0]):
                weights[X_source.shape[0] + j] = weights[X_source.shape[0] + j] * np.power(beta_target, np.abs(y_predict[X_source.shape[0] + j] - self.y_target[j]))


    def _calculate_error_rate(self, y_true, y_predict, weight):
            total = np.sum(weight)
            return np.sum(weight[:] * np.abs(y_true - y_predict)/ total)

    def predict(self,X):
        y_predict=self.clf.predict(X)
        return y_predict


