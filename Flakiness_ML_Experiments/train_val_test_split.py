import utils.columns as col
from sklearn.model_selection import train_test_split
import pandas as pd


def dataset_split(dataset, val_size, test_size):

    X_dataset = dataset.drop([col.TARGET], axis = 1)
    y_dataset = dataset[col.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset,
                                                        stratify = y_dataset,
                                                        test_size = test_size,
                                                        random_state = 42)

    X_train = X_train.reset_index(drop = True)
    y_train = y_train.reset_index(drop = True)

    X_test = X_test.reset_index(drop = True)
    y_test = y_test.reset_index(drop = True)
    test_set = pd.concat([X_test, y_test], axis = 1)

    if val_size!=0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                            stratify = y_train,
                                                            test_size = val_size,
                                                            random_state = 42)

        X_train = X_train.reset_index(drop = True)
        y_train = y_train.reset_index(drop = True)
        train_set = pd.concat([X_train, y_train], axis = 1)

        X_val = X_val.reset_index(drop = True)
        y_val = y_val.reset_index(drop = True)
        val_set = pd.concat([X_val, y_val], axis = 1)
        return train_set, val_set, test_set

    else:
        train_set = pd.concat([X_train, y_train], axis = 1)
        return train_set, test_set

