import pandas as pd
from preProcessing.CustomPreProcessing import CustomPreProcessing
from imblearn.over_sampling import SMOTE

class TrAdaPreProcessing(CustomPreProcessing):

    def __init__(self):
        self.smote=SMOTE(sampling_strategy='auto')


    def fit(self, X_source, X_target, y_source, y_target,k):

        columns_name=X_source.columns
        Xdata=[]
        ydata=[]

        for instance,label in zip(Xtarget,ytarget):
            indici_selezionati = ysource[:] == label
            sottomatrice = Xsource[indici_selezionati]
            distances = euclidean(instance, sottomatrice)
            indici_ordinati = np.argsort(distances)
            sottomatrice_sorted = sottomatrice[indici_ordinati]
            for neighbor in sottomatrice_sorted[0:k]:
                if not list(neighbor) in Xdata:
                    Xdata.append(list(neighbor))
                    ydata.append(label)


        Xdata=np.asanyarray(Xdata)
        ydata=np.asarray(ydata)

        X_data_smote, y_data_smote=smote.fit_resample(Xdata, ydata)
        return pd.DataFrame(X_data_smote,columns=columns_name), pd.Series(y_data_smote)


    def transform(self, X, y=None): pass

    def fit_transform(self, X_source, X_target, y_source, y_target,k, **fit_params):
        return self.fit(X_source, X_target, y_source, y_target,k)