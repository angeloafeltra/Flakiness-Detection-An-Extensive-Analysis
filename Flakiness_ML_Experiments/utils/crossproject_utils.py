from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import utils.columns as col
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from utils.burak_utils import classic_burakFilter
from utils.burak_utils import supervisioned_burakFilter


class FeatureBased_PreProcessing:

    def __init__(self, n_neighbors=10):
        self.std=StandardScaler()
        self.smote=SMOTE(sampling_strategy='auto',k_neighbors=4)
        self.neighbors= n_neighbors

    def fit(self,X_set,y_set=None,X_target=None,y_target=None):
        if (not y_set is None) and (not X_target is None) and (not y_target is None):
            #Devo applicare prima il filtro
            return
        else:
            self.std.fit(X_set)
        return self

    def transform(self,X_set,y_set=None,X_target=None,y_target=None):

        if (not y_set is None) and (not X_target is None) and (not y_target is None):
            #Applico prima il filtro
            X_burak, y_burak, _ , _ , _ , _ = classic_burakFilter(X_set.to_numpy(),
                                                                  y_set.to_numpy(),
                                                                  X_target.to_numpy(),
                                                                  self.neighbors,
                                                                  y_target.to_numpy())

            burak_TF=np.count_nonzero(y_burak)
            if burak_TF<6: return None, None

            X_burak_std=self.std.fit_transform(X_burak)
            X_burak_smote, y_burak_smote= self.smote.fit_resample(X_burak_std, y_burak)

            return X_burak_smote, y_burak_smote
        else:
            X_set_std=self.std.transform(X_set)
            return X_set_std


        '''
        def transform(self,X_set,y_set=None,X_target=None,y_target=None):
        X_set_std= self.std.transform(X_set)

        if (not y_set is None) and (not X_target is None) and (not y_target is None):
            X_target=self.std.transform(X_target)
            X_burak, y_burak, _ , _ , _ , _ = classic_burakFilter(X_set_std.to_numpy(),
                                                                  y_set.to_numpy(),
                                                                  X_target,
                                                                  self.neighbors,
                                                                  y_target.to_numpy())
            burak_TF=np.count_nonzero(y_burak)

            if burak_TF<6: return None, None

            X_burak_smote, y_burak_smote= self.smote.fit_resample(X_burak, y_burak)

            return X_burak_smote, y_burak_smote
        else:
            return X_set_std
        '''

    def fit_transform(self,X_set,y_set=None,X_target=None,y_target=None):
        self.fit(X_set,y_set,X_target,y_target)
        return self.transform(X_set,y_set,X_target,y_target)



class IG_SM_FS_PreProcessing:


    def __init__(self,perc_featureToSelect=0.5):

        self.perc_featureToSelect=int(perc_featureToSelect*100)

    def fit(self,X_source,y_source,X_target):
        pass
        #Calcolo la feature importance tramite IG
        randomForest=RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=1)
        randomForest.fit(X=X_source,y=y_source)
        self.importanceFeatures=randomForest.feature_importances_

        #Calcolo la feature similarity
        self.featureSimilarity=self.__feature_similarity(X_source,X_target)

        #Calcolo la feature weight
        alpha=0.5
        self.feature_weight=[]
        for i in range(0,len(col.NUMERICAL_FEATURES)):
            weight=alpha * self.importanceFeatures[i] + (1-alpha) * self.featureSimilarity[i]
            self.feature_weight.append(weight)

        #Seleziono le features
        num_feature_select=int(len(col.NUMERICAL_FEATURES) * (self.perc_featureToSelect/100))

        liste_combinate = list(zip(self.feature_weight, col.NUMERICAL_FEATURES))
        liste_combinate_ordinate = sorted(liste_combinate, key=lambda x: x[0], reverse=True)
        # Dividi le liste ordinate
        tmp_weight, tmp_col = zip(*liste_combinate_ordinate)

        self.feature=list(tmp_col[0:num_feature_select])

    def transform(self,X_source,y_source=None,X_target=None):

        if not X_target is None:
            return X_source[self.feature], X_target[self.feature]
        else:
            return X_source[self.feature]


    def fit_transform(self,X_source,y_source,X_target):
        self.fit(X_source,y_source,X_target)
        return self.transform(X_source,y_source,X_target)

    def __feature_similarity(self,source,target):
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






