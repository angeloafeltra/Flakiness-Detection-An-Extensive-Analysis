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


def dataset_generator(dataset):

    SPE_DATA={
        'source': [],
        'target': [],
        "source_tloc": [],
        "source_tmcCabe": [],
        "source_assertionDensity": [],
        "source_assertionRoulette": [],
        "source_mysteryGuest": [],
        "source_eagerTest": [],
        "source_sensitiveEquality": [],
        "source_resourceOptimism": [],
        "source_conditionalTestLogic": [],
        "source_fireAndForget": [],
        "source_testRunWar": [],
        "source_loc": [],
        "source_lcom2": [],
        "source_lcom5": [],
        "source_cbo": [],
        "source_wmc": [],
        "source_rfc": [],
        "source_mpc": [],
        "source_halsteadVocabulary": [],
        "source_halsteadLength": [],
        "source_halsteadVolume": [],
        "source_classDataShouldBePrivate": [],
        "source_complexClass": [],
        "source_spaghettiCode": [],
        "source_functionalDecomposition": [],
        "source_godClass": [],
        "target_tloc": [],
        "target_tmcCabe": [],
        "target_assertionDensity": [],
        "target_assertionRoulette": [],
        "target_mysteryGuest": [],
        "target_eagerTest": [],
        "target_sensitiveEquality": [],
        "target_resourceOptimism": [],
        "target_conditionalTestLogic": [],
        "target_fireAndForget": [],
        "target_testRunWar": [],
        "target_loc": [],
        "target_lcom2": [],
        "target_lcom5": [],
        "target_cbo": [],
        "target_wmc": [],
        "target_rfc": [],
        "target_mpc": [],
        "target_halsteadVocabulary": [],
        "target_halsteadLength": [],
        "target_halsteadVolume": [],
        "target_classDataShouldBePrivate": [],
        "target_complexClass": [],
        "target_spaghettiCode": [],
        "target_functionalDecomposition": [],
        "target_godClass": [],
        'f1-score': [],
    }

    list_project=dataset['nameProject'].unique()
    for source_project in list_project:
        h_source_project=dataset.loc[dataset['nameProject']==source_project]
        target_projects=dataset.loc[dataset['nameProject']!=source_project]

        for target_project in target_projects['nameProject'].unique():
            h_target_project=dataset.loc[dataset['nameProject']==target_project]

            X_source_set = h_source_project.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
            y_source_set = h_source_project[col.TARGET]

            X_target_set = h_target_project.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
            y_target_set = h_target_project[col.TARGET]

            print("Source: {} - Target: {}".format(source_project, target_project))

            SPE_DATA['source'].append(source_project)
            SPE_DATA['target'].append(target_project)

            for (columnName, columnData) in X_source_set.iteritems():
                SPE_DATA['source_{}'.format(columnName)].append(X_source_set[columnName].mean())
                SPE_DATA['target_{}'.format(columnName)].append(X_target_set[columnName].mean())

            std=StandardScaler()
            X_source_set_std=std.fit_transform(X_source_set)
            X_target_set_std=std.transform(X_target_set)

            #model=TCA(LogisticRegression(class_weight='balanced', random_state=42), Xt=X_target_set_std)
            model.fit(X_source_set_std, y_source_set)
            y_predict=model.predict(X_target_set_std)
            acc, pr, rec, f1, tn, fp, fn, tp=validation_utils.val_and_log_metrics(y_target_set,y_predict)

            SPE_DATA['f1-score'].append(f1)

    df = pd.DataFrame(data=SPE_DATA)
    df.to_csv('FlakeFlagger_SPE_dataset.csv', index=True)
    return df



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


class TCA():


    def __init__(self,
                 Xt=None,
                 n_components=20,
                 mu=0.1,
                 kernel="rbf",
                 copy=True,
                 verbose=1,
                 random_state=42):

        self.Xt=Xt
        self.n_components=n_components
        self.mu=mu
        self.kernel=kernel
        self.copy=copy
        self.verbose=verbose
        self.random_state=random_state



    def fit_transform(self, Xs, Xt):

        Xs = check_array(Xs)
        Xt = check_array(Xt)

        self.Xs_ = Xs
        self.Xt_ = Xt

        n = len(Xs)
        m = len(Xt)

        # Compute Kernel Matrix K
        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}

        Kss = pairwise.pairwise_kernels(Xs, Xs, metric=self.kernel, **kernel_params)
        Ktt = pairwise.pairwise_kernels(Xt, Xt, metric=self.kernel, **kernel_params)
        Kst = pairwise.pairwise_kernels(Xs, Xt, metric=self.kernel, **kernel_params)

        K = np.concatenate((Kss, Kst), axis=1)
        K = np.concatenate((K, np.concatenate((Kst.transpose(), Ktt), axis=1)), axis=0)

        # Compute L
        Lss = np.ones((n,n)) * (1./(n**2))
        Ltt = np.ones((m,m)) * (1./(m**2))
        Lst = np.ones((n,m)) * (-1./(n*m))

        L = np.concatenate((Lss, Lst), axis=1)
        L = np.concatenate((L, np.concatenate((Lst.transpose(), Ltt), axis=1)), axis=0)

        # Compute H
        H = np.eye(n+m) - 1/(n+m) * np.ones((n+m, n+m))

        # Compute solution
        a = np.eye(n+m) + self.mu * K.dot(L.dot(K))
        b = K.dot(H.dot(K))
        sol = linalg.lstsq(a, b)[0]

        values, vectors = linalg.eigh(sol)

        args = np.argsort(np.abs(values))[::-1][:self.n_components]

        self.vectors_ = np.real(vectors[:, args])

        Xs_enc = K.dot(self.vectors_)[:n]

        return Xs_enc


    def transform(self, X, domain="tgt"):

        X = check_array(X)

        kernel_params = {k: v for k, v in self.__dict__.items()
                         if k in KERNEL_PARAMS[self.kernel]}

        Kss = pairwise.pairwise_kernels(X, self.Xs_, metric=self.kernel, **kernel_params)
        Kst = pairwise.pairwise_kernels(X, self.Xt_, metric=self.kernel, **kernel_params)

        K = np.concatenate((Kss, Kst), axis=1)

        return K.dot(self.vectors_)


