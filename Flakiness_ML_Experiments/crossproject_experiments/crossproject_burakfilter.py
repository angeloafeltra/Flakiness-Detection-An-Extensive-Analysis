import sys

import pandas as pd

import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
from utils.burak_utils import classic_burakFilter
import numpy as np
from sklearn.model_selection import train_test_split
import os
from utils.eval_utils import eval_and_log_metrics
from utils.explenability_utils import log_distribution, log_featureImportance

def run(dataset, pipeline, experiment_ID):


    list_project=dataset[col.CATEGORICAL_FEATURES[0]].unique()

    with mlflow.start_run(run_name='CrossProject_BurakFilter (CPBF)',experiment_id= experiment_ID) as father_run:

        for target in list_project:

            print(target)
            tmp=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]
            testFlaky=len(tmp[tmp[col.TARGET]==1])
            testNonFlaky=len(tmp[tmp[col.TARGET]==0])

            if testFlaky<20 or testFlaky>testNonFlaky:
                print("La repository non rispetta i criteri")
                continue

            repository_name=target.split('_')[0]
            with mlflow.start_run(run_name="CPBF_{}".format(repository_name),experiment_id=experiment_ID,nested=True) as child_run:

                source_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]!=target]
                target_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]

                source_TF=len(source_set[source_set[col.TARGET]==1])
                source_TNF=len(source_set[source_set[col.TARGET]==0])
                target_TF=len(target_set[target_set[col.TARGET]==1])
                target_TNF=len(target_set[target_set[col.TARGET]==0])

                mlflow.log_metric("Source Test Flaky", source_TF)
                mlflow.log_metric("Source Test Non Flaky", source_TNF)
                mlflow.log_metric("Target Test Flaky", target_TF)
                mlflow.log_metric("Target Test Non Flaky", target_TNF)



                X_source_set = source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_source_set = source_set[col.TARGET]


                X_target_set = target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_target_set = target_set[col.TARGET]


                X_burak, y_burak, TF_countTF, TF_countTNF, TNF_countTF, TNF_countTNF =  classic_burakFilter(X_source_set.to_numpy(),
                                                                                                            y_source_set.to_numpy(),
                                                                                                            X_target_set.to_numpy(),
                                                                                                            10,
                                                                                                            y_target_set.to_numpy())
                #Log per explenability
                mlflow.log_metric("TF selezionati da istanze Flaky",TF_countTF)
                mlflow.log_metric("TNF selezionati da istanze Flaky", TF_countTNF)
                mlflow.log_metric("TNF selezionati da istanze non Flaky", TNF_countTNF)
                mlflow.log_metric("TF selezionati da istanze non Flaky", TNF_countTF)

                burak_set=np.insert(X_burak, -1, y_burak, axis=1)
                columns=[]
                columns=columns+col.NUMERICAL_FEATURES
                columns.append(col.TARGET)
                burak_set=pd.DataFrame(burak_set,columns=columns)

                burak_TF=np.count_nonzero(y_burak)
                burak_TNF=y_burak.size - burak_TF

                mlflow.log_metric("Burak Test Flaky", burak_TF)
                mlflow.log_metric("Burak Test Non Flaky", burak_TNF)

                print("Source TF:{} - TNF:{}\nTarget TF:{} - TNF:{}\nBurak_Set TF:{} - TNF:{}".format(source_TF,
                                                                                                        source_TNF,
                                                                                                        target_TF,
                                                                                                        target_TNF,
                                                                                                        burak_TF,
                                                                                                        burak_TNF))


                if burak_TF<6: continue

                if burak_TF>=2:
                    X_burak_train, X_burak_test, y_burak_train, y_burak_test = train_test_split(X_burak, y_burak,
                                                                                                stratify = y_burak,
                                                                                                test_size = 0.2,
                                                                                                random_state = 42)
                    pipeline.fit(X_burak_train,y_burak_train)
                    y_predict=pipeline.predict(X_burak_train)
                    eval_and_log_metrics("CP_Source_Train",y_burak_train,y_predict)

                    y_predict=pipeline.predict(X_burak_test)
                    validation_utils.val_and_log_metrics(y_burak_test,y_predict,'CP_Source_Test')
                else:
                    pipeline.fit(X_burak,y_burak)
                    y_predict=pipeline.predict(X_burak)
                    eval_and_log_metrics("CP_Source_Train",y_burak,y_predict)

                y_predict=pipeline.predict(X_target_set)
                validation_utils.val_and_log_metrics(y_target_set,y_predict,'CP_Target')

                #Explenability
                log_distribution(burak_set, 'Distribution Source(Burak).csv', 'Distribution')
                log_distribution(target_set, 'Distribution Target.csv', 'Distribution')
                log_distribution(burak_set.loc[burak_set[col.TARGET]==0],
                                 'Distribution Non Flaky Test Source(Burak).csv',
                                 'Distribution')
                log_distribution(target_set.loc[target_set[col.TARGET]==0],
                                 'Distribution Non Flaky Test Target.csv',
                                 'Distribution')
                log_distribution(burak_set.loc[burak_set[col.TARGET]==1],
                                 'Distribution Flaky Test Source(Burak).csv',
                                 'Distribution')
                log_distribution(target_set.loc[target_set[col.TARGET]==1],
                                 'Distribution Flaky Test Target.csv',
                                 'Distribution')
                log_featureImportance(pipeline.get_params('steps')['model'],
                                      'Feature Importances Classifier.csv',
                                      'Feature Importances Classifier')




if __name__ == "__main__":
    run(sys.argv[1])




