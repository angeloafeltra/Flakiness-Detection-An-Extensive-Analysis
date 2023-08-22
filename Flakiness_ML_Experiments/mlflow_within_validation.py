import sys

from train_val_test_split import dataset_split
import pandas as pd
import os
import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings
import pickle
import utils.validation_utils as validation_utils
from train_val_test_split import dataset_split

def run(datasetName):

    warnings.filterwarnings("ignore")

    dataset = pd.read_csv(os.path.join('..','Dataset','{}.csv'.format(datasetName)))

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


    list_project=dataset['nameProject'].unique()

    with mlflow.start_run(run_name='WithIn_Validation',experiment_id= experiment_ID) as father_run:
        for project in list_project:
            print(project)
            with mlflow.start_run(run_name=project,experiment_id=experiment_ID,nested=True) as child_run:
                repo_set=dataset.loc[dataset['nameProject']==project]

                if len(repo_set[repo_set['isFlaky']==1])<2:
                    continue

                mlflow.log_metric("Test Flaky", len(repo_set[repo_set['isFlaky']==1]))
                mlflow.log_metric("Test Non Flaky", len(repo_set[repo_set['isFlaky']==0]))
                print("TF:{} - TNF:{}".format(len(repo_set[repo_set['isFlaky']==1]),
                                                len(repo_set[repo_set['isFlaky']==0])))


                train_set, test_set= dataset_split(repo_set, 0, 0.20)

                X_train_set = train_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_train_set = train_set[col.TARGET]
                cat_train_set = train_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)

                X_test_set = test_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_test_set = test_set[col.TARGET]
                cat_train_set = test_set.drop([col.TARGET] + col.NUMERICAL_FEATURES, axis = 1)


                pipeline.fit(X_train_set,y_train_set)
                y_predict=pipeline.predict(X_test_set)
                validation_utils.val_and_log_metrics(y_test_set,y_predict)








if __name__ == "__main__":
    run(sys.argv[1])




