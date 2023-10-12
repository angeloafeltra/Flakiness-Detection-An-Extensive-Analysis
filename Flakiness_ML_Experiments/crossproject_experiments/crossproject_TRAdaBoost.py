import sys
import pandas as pd
import os



import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
from utils.eval_utils import eval_and_log_metrics
import numpy as np
from sklearn.model_selection import train_test_split
from adapt.instance_based import TrAdaBoost
from utils.burak_utils import supervisioned_burakFilter
from imblearn.over_sampling import SMOTE

def run(dataset, pipeline, experiment_ID):


    list_project=dataset['nameProject'].unique()

    for i in [0.25,0.50,0.75]:
        with mlflow.start_run(run_name= 'CrossProject_TRAdaBoost_{} (CPTRAda)'.format(i),experiment_id= experiment_ID) as father_run:

            for target in list_project[0:]:
                print(target)

                #Inserire qui il criterio di filtraggio per skippare il target
                tmp=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]
                testFlaky=len(tmp[tmp[col.TARGET]==1])
                testNonFlaky=len(tmp[tmp[col.TARGET]==0])

                if testFlaky<20 or testFlaky>testNonFlaky:
                    print("La repository non rispetta i criteri")
                    continue

                repository_name=target.split('_')[0]
                with mlflow.start_run(run_name="CPTRAda{}_{}".format(int(i*100),repository_name),experiment_id=experiment_ID,nested=True) as child_run:


                    target_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]
                    source_set=dataset.loc[dataset['nameProject']!=target]

                    source_TF=len(source_set[source_set[col.TARGET]==1])
                    source_TNF=len(source_set[source_set[col.TARGET]==0])
                    target_TF=len(target_set[target_set[col.TARGET]==1])
                    target_TNF=len(target_set[target_set[col.TARGET]==0])

                    mlflow.log_metric("Source Test Flaky", source_TF)
                    mlflow.log_metric("Source Test Non Flaky", source_TNF)
                    mlflow.log_metric("Target Test Flaky", target_TF)
                    mlflow.log_metric("Target Test Non Flaky", target_TNF)

                    print("Source TF:{} - TNF:{}\nTarget TF:{} - TNF:{} ".format(source_TF,
                                                                                 source_TNF,
                                                                                 target_TF,
                                                                                 target_TNF))


                    X_source_set=source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                    y_source_set=source_set[col.TARGET]
                    X_target_set=target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                    y_target_set=target_set[col.TARGET]

                    X_target_set, X_target_NL, y_target_set, y_target_NL=train_test_split(X_target_set,
                                                                                 y_target_set,
                                                                                 stratify=y_target_set,
                                                                                 test_size=0.25,
                                                                                 random_state=42)

                    train_size= (i/0.75)*1
                    if train_size!=1:
                        X_target_L,_,y_target_L,_=train_test_split(X_target_set,
                                                                   y_target_set,
                                                                   stratify=y_target_set,
                                                                   test_size=1-train_size,
                                                                   random_state=42)
                    else:
                        X_target_L=X_target_set
                        y_target_L=y_target_set



                    #1. Filtro di burak
                    X_burak, y_burak= supervisioned_burakFilter(X_source_set.to_numpy(),
                                                                y_source_set.to_numpy(),
                                                                X_target_L.to_numpy(),
                                                                y_target_L.to_numpy(),
                                                                10)

                    #2. SMOTE
                    smote=SMOTE(sampling_strategy='auto')
                    X_burak_smote, y_burak_smote=smote.fit_resample(X_burak, y_burak)


                    #3. TrAdaBoost
                    model = TrAdaBoost(pipeline.get_params('steps')['model'], n_estimators=10, Xt=X_target_L, yt=y_target_L, random_state=42)

                    model.fit(X_burak_smote, y_burak_smote)
                    y_predict=model.predict(X_burak_smote,)
                    eval_and_log_metrics("CP_Source_Train", y_burak_smote, y_predict)

                    y_predict=model.predict(X_target_NL)
                    validation_utils.val_and_log_metrics(y_target_NL,y_predict,'CP_Target')




if __name__ == "__main__":
    run(sys.argv[1])




