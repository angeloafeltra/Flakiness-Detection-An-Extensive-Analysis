import sys

import pandas as pd
import os
import utils.columns as col
import mlflow
import warnings
import pickle
import utils.validation_utils as validation_utils
from utils import eval_utils
from sklearn.model_selection import train_test_split
import numpy as np

def run(datasetName,perc):

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

    with mlflow.start_run(run_name='WithIn_Validation_{}'.format(perc),experiment_id= experiment_ID) as father_run:
        for target in list_project:
            print("Target: ",target)

            tmp=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]
            testFlaky=len(tmp[tmp[col.TARGET]==1])
            testNonFlaky=len(tmp[tmp[col.TARGET]==0])

            if testFlaky<20 or testFlaky>testNonFlaky:
                print("La repository non rispetta i criteri")
                continue

            repository_name=target.split('_')[0]
            with mlflow.start_run(run_name="WP{}_{}".format(int(perc*100),repository_name),experiment_id=experiment_ID,nested=True) as child_run:
                repo_set=dataset.loc[dataset['nameProject']==target]

                mlflow.log_metric("Test Flaky", len(repo_set[repo_set['isFlaky']==1]))
                mlflow.log_metric("Test Non Flaky", len(repo_set[repo_set['isFlaky']==0]))
                print("TF:{} - TNF:{}".format(len(repo_set[repo_set['isFlaky']==1]),
                                                len(repo_set[repo_set['isFlaky']==0])))


                scores={
                    'Train ACC':[],
                    'Train PRE':[],
                    'Train REC':[],
                    'Train F1':[],

                    'Test ACC':[],
                    'Test PRE':[],
                    'Test REC':[],
                    'Test F1': [],
                    'Test TN': [],
                    'Test FP': [],
                    'Test FN': [],
                    'Test TP': [],
                    'Test AUC': []
                }


                for iter in range(10):

                    X_repo_set= repo_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                    y_repo_set = repo_set[col.TARGET]

                    X_repo_set,X_test_set,y_repo_set,y_test_set=train_test_split(X_repo_set,
                                                                                 y_repo_set,
                                                                                 stratify=y_repo_set,
                                                                                 test_size=0.25,
                                                                                 random_state=iter)

                    train_size= (i/0.75)*1
                    if train_size!=1:
                        X_train_set,_,y_train_set,_=train_test_split(X_repo_set,
                                                                                 y_repo_set,
                                                                                 stratify=y_repo_set,
                                                                                 test_size=1-train_size,
                                                                                 random_state=iter)
                    else:
                        X_train_set=X_repo_set
                        y_train_set=y_repo_set


                    pipeline.fit(X_train_set,y_train_set)
                    y_predict=pipeline.predict(X_train_set)
                    acc,pr,rec,f1=eval_utils.eval_metrics(y_train_set,y_predict)
                    scores['Train ACC'].append(acc)
                    scores['Train PRE'].append(pr)
                    scores['Train F1'].append(f1)
                    scores['Train REC'].append(rec)

                    y_predict=pipeline.predict(X_test_set)
                    acc, pr, rec, f1, tn, fp, fn, tp, auc= validation_utils.val_metrics(y_test_set,y_predict)
                    scores['Test ACC'].append(acc)
                    scores['Test PRE'].append(pr)
                    scores['Test REC'].append(rec)
                    scores['Test F1'].append(f1)
                    scores['Test TN'].append(tn)
                    scores['Test FN'].append(fp)
                    scores['Test TP'].append(tp)
                    scores['Test FP'].append(fp)
                    scores['Test AUC'].append(auc)

                train_acc=np.mean(scores['Train ACC'])
                train_pre=np.mean(scores['Train PRE'])
                train_f1=np.mean(scores['Train F1'])
                train_rec=np.mean(scores['Train REC'])

                eval_utils.log_eval__metrics('WP_Train',train_acc,train_pre,train_rec,train_f1)

                test_acc=np.mean(scores['Test ACC'])
                test_pre=np.mean(scores['Test PRE'])
                test_rec=np.mean(scores['Test REC'])
                test_f1=np.mean(scores['Test F1'])
                test_TN=np.mean(scores['Test TN'])
                test_FN=np.mean(scores['Test FN'])
                test_TP=np.mean(scores['Test TP'])
                test_FP=np.mean(scores['Test FP'])
                test_AUC=np.mean(scores['Test AUC'])

                validation_utils.log_val_metrics(test_acc,
                                                 test_pre,
                                                 test_rec,
                                                 test_f1,
                                                 test_TN,
                                                 test_FP,
                                                 test_FN,
                                                 test_TP,
                                                 test_AUC,
                                                 'WP_Test')










if __name__ == "__main__":
    perc=[0.25,0.50,0.75]
    for i in perc:
        print("Percentuale Informazioni repository: {}".format(i))
        run(sys.argv[1],i)




