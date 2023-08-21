import sys

import pandas as pd
import os
import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings
import pickle
import utils.validation_utils as validation_utils
import utils.burak_filter as burak_filter

def run(datasetName):
    warnings.filterwarnings("ignore")

    # Creo un nuovo esperimento su mlflow
    experiment =mlflow.get_experiment_by_name("FlakinessMLExperiment_{}".format(datasetName))
    if not experiment:
        experiment_ID=mlflow.create_experiment("FlakinessMLExperiment_{}".format(datasetName))
    else:
        experiment_ID=experiment.experiment_id


    all_run=mlflow.search_runs(experiment_ids=[experiment_ID]) #Ottengo tutte le run dell'esperimento
    row=all_run[all_run['metrics.Val_F1'] == all_run['metrics.Val_F1'].max()].head(1)
    pipeline=pickle.load(open(os.path.join('.',
                                            'mlruns',
                                            row['experiment_id'].to_string(index=False, header=False),
                                            row['run_id'].to_string(index=False, header=False),
                                            'artifacts',
                                            row['tags.mlflow.runName'].to_string(index=False, header=False),
                                            'model.pkl'),'rb'))

    dataset = pd.read_csv(os.path.join('..','Dataset','{}.csv'.format(datasetName)))
    list_project=dataset['nameProject'].unique()



    with mlflow.start_run(run_name='CrossProject_BurakFilter',experiment_id= experiment_ID) as father_run:
        for project in list_project:
            print(project)
            with mlflow.start_run(run_name=project,experiment_id=experiment_ID,nested=True) as child_run:
                train_set=dataset.loc[dataset['nameProject']!=project]
                test_set=dataset.loc[dataset['nameProject']==project]

                X_train_set = train_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_train_set = train_set[col.TARGET]
                cat_train_set = train_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)


                X_test_set = test_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_test_set = test_set[col.TARGET]
                cat_test_set = test_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)

                X_train_burak, y_train_burak = burak_filter.burak(X_train_set.to_numpy(),
                                                                    y_train_set.to_numpy(),
                                                                    X_test_set.to_numpy(),
                                                                    10)

                mlflow.log_metric("Test Flaky", len(test_set[test_set['isFlaky']==1]))
                mlflow.log_metric("Test Non Flaky", len(test_set[test_set['isFlaky']==0]))
                print("TF:{} - TNF:{}".format(len(test_set[test_set['isFlaky']==1]), len(test_set[test_set['isFlaky']==0])))

                pipeline.fit(X_train_burak,y_train_burak)
                y_predict=pipeline.predict(X_test_set)
                validation_utils.val_and_log_metrics(y_test_set,y_predict)








if __name__ == "__main__":
    run(sys.argv[1])




