import numpy as np

def euclidean(point, data):
    # Euclidean distance between points a & data
    return np.sqrt(np.sum((point - data)**2, axis=1))

def burak(X_train,y_train,X_test,k):
    X_neighbors = np.empty((0, X_train.shape[1]))
    y_neighbors = np.empty((0, ))
    for x in X_test[0:2]:
        distances = euclidean(x, X_train)
        indici_ordinati = np.argsort(distances)
        X_sorted = X_train[indici_ordinati]
        y_sorted = [y for _, y in sorted(zip(distances, y_train))]

        X_neighbors=np.concatenate((X_neighbors, X_sorted[:k, :]), axis=0)
        y_neighbors=np.concatenate((y_neighbors, y_sorted[:k]), axis=0)

    return X_neighbors, y_neighbors