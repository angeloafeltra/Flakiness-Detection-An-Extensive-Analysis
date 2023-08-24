import sys

import pandas as pd
import os
import utils.columns as col
import utils.experimentsList as experimentList
import mlflow
import warnings
import pickle

def run(datasetName, experimentName):
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


    for (experiment,expName) in experimentList.CROSS_PROJECT_EXPERIMENT:
        if expName==experimentName:
            experiment.run(dataset, pipeline, experiment_ID)








if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])




