import sys

from tuning_model import tuning_pipeline
from train_val_test_split import dataset_split
import pandas as pd
import os
import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings
import pickle
import utils.validation_utils as validation_utils

def run(datasetName):
    warnings.filterwarnings("ignore")


    dataset = pd.read_csv(os.path.join('..','Dataset','{}.csv'.format(datasetName)))
    train_set, val_set, test_set = dataset_split(dataset,0.20,0.20)


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

    X_test_set = test_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
    y_test_set = test_set[col.TARGET]
    cat_test_set = test_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)


    # Creo un nuovo esperimento su mlflow
    experiment =mlflow.get_experiment_by_name("FlakinessMLExperiment_{}".format(datasetName))
    if not experiment:
        experiment_ID=mlflow.create_experiment("FlakinessMLExperiment_{}".format(datasetName))
    else:
        experiment_ID=experiment.experiment_id


    ########################################
    # Evaluated
    ########################################

    print("Evaluated")
    for (experiment,pipName) in experimentList.EXPERIMENTS:
        print(pipName)
        experiment.run(X_train_set, y_train_set, X_val_set, y_val_set,pipName, experiment_ID)


    ########################################
    # Tuning
    ########################################

    all_run=mlflow.search_runs(experiment_ids=[experiment_ID]) #Ottengo tutte le run dell'esperimento

    row=all_run[all_run['metrics.Val_F1'] == all_run['metrics.Val_F1'].max()].head(1)
    pipeline=pickle.load(open(os.path.join('.',
                                        'mlruns',
                                        row['experiment_id'].to_string(index=False, header=False),
                                        row['run_id'].to_string(index=False, header=False),
                                        'artifacts',
                                        row['tags.mlflow.runName'].to_string(index=False, header=False),
                                        'model.pkl'),'rb'))

    print("\nTuning {}".format(row['tags.mlflow.runName'].to_string(index=False, header=False)))
    tuning_pipeline(X_train_set,
                    y_train_set,
                    X_val_set,
                    y_val_set,
                    pipeline,
                    row['tags.mlflow.runName'].to_string(index=False, header=False),
                    row['run_id'].to_string(index=False, header=False),
                    row['experiment_id'].to_string(index=False, header=False))


    ########################################
    # Validation
    ########################################

    all_run=mlflow.search_runs(experiment_ids=[experiment_ID]) #Ottengo tutte le run dell'esperimento
    row=all_run[all_run['metrics.Val_F1'] == all_run['metrics.Val_F1'].max()].head(1)
    pipeline=pickle.load(open(os.path.join('.',
                                        'mlruns',
                                        row['experiment_id'].to_string(index=False, header=False),
                                        row['run_id'].to_string(index=False, header=False),
                                        'artifacts',
                                        row['tags.mlflow.runName'].to_string(index=False, header=False),
                                        'model.pkl'),'rb'))


    print("\nValidation\nMiglior pipeline: {} con un F1_Score di {} - Validation".format(row['tags.mlflow.runName'].to_string(index=False, header=False),
                                                                        row['metrics.Val_F1'].to_string(index=False, header=False)))



    with mlflow.start_run(run_id=row['run_id'].to_string(index=False, header=False),
                            experiment_id=row['experiment_id'].to_string(index=False, header=False),
                            nested=True) as active_run:
        y_predict=pipeline.predict(X_test_set)
        validation_utils.val_and_log_metrics(y_test_set,y_predict)
        mlflow.set_tag("Best Pipeline","Best Pipeline")

        validation_utils.soa_validation(dataset,pipeline)







if __name__ == "__main__":
    run(sys.argv[1])




