import sys

from train_val_test_split import dataset_split
import pandas as pd
import os
import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings

def run(datasetName, experiment_name):
    warnings.filterwarnings("ignore")


    dataset = pd.read_csv(os.path.join('..','Dataset',datasetName))
    train_set, val_set, test_set = dataset_split(dataset,0.20,0.10)


    print("TRAIN SET - TF:{}, TNF:{}".format(len(train_set[train_set['isFlaky'] == 1]),
                                             len(train_set[train_set['isFlaky'] == 0])))
    print("VAL SET - TF:{}, TNF:{}".format(len(val_set[val_set['isFlaky'] == 1]),
                                           len(val_set[val_set['isFlaky'] == 0])))
    print("TEST SET - TF:{}, TNF:{}".format(len(test_set[test_set['isFlaky'] == 1]),
                                            len(test_set[test_set['isFlaky'] == 0])))

    X_train_set = train_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
    y_train_set = train_set[col.TARGET]
    cat_train_set = train_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)

    X_val_set = val_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
    y_val_set = val_set[col.TARGET]
    cat_val_set = val_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)

    X_train_set = train_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
    y_train_set = train_set[col.TARGET]
    cat_train_set = train_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)



    # Creo un nuovo esperimento su mlflow
    experiment =mlflow.get_experiment_by_name("FlakinessMLExperiment")
    if not experiment:
        experiment_ID=mlflow.create_experiment("FlakinessMLExperiment")
    else:
        experiment_ID=experiment.experiment_id



    for (experiment,pipName) in experimentList.EXPERIMENTS:
        if pipName == experiment_name:
            experiment.run(X_train_set, y_train_set, X_val_set, y_val_set,pipName, experiment_ID)


if __name__ == "__main__":
    print("ENTRA")
    run(sys.argv[1],sys.argv[2])




