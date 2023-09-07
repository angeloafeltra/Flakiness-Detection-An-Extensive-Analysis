import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

def classic_burakFilter(X_source, y_source, X_target, k):

    knn=KNeighborsClassifier()
    knn.fit(X_source,y_source)

    Xdata=[]
    ydata=[]

    for instance in X_target:
        neighbors_index=knn.kneighbors(instance.reshape(1,-1), k, return_distance=False)
        for neighbor_index in neighbors_index[0]:
            if not list(X_source[neighbor_index]) in Xdata:
                Xdata.append(list(X_source[neighbor_index]))
                ydata.append(y_source[neighbor_index])

    return np.asanyarray(Xdata), np.asarray(ydata)

def supervisioned_burakFilter(sources_set,target_set,k):
    source_set=np.empty((0, sources_set.shape[1]))
    count=0
    for x in target_set:
        indici_selezionati = sources_set[:, 24] == x[24]
        sottomatrice = sources_set[indici_selezionati]

        distances = euclidean(x, sottomatrice)
        indici_ordinati = np.argsort(distances)
        sottomatrice_sorted = sottomatrice[indici_ordinati]
        source_set=np.concatenate((source_set, sottomatrice_sorted[:k, :]), axis=0)

    return source_set