import sys

import pandas as pd

import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
from utils.burak_utils import classic_burakFilter
import numpy as np
from sklearn.model_selection import train_test_split
from utils.crossproject_utils import calculate_distribution, features_importance
import os
from utils.eval_utils import eval_and_log_metrics

def run(dataset, pipeline, experiment_ID):


    list_project=dataset[col.CATEGORICAL_FEATURES[0]].unique()

    with mlflow.start_run(run_name='CrossProject_BurakFilter',experiment_id= experiment_ID) as father_run:

        for target in list_project:

            print(target)

            with mlflow.start_run(run_name=target,experiment_id=experiment_ID,nested=True) as child_run:

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

                if burak_TF>=2:
                    X_burak_train, X_burak_test, y_burak_train, y_burak_test = train_test_split(X_burak, y_burak,
                                                                                                stratify = y_burak,
                                                                                                test_size = 0.2,
                                                                                                random_state = 42)
                    pipeline.fit(X_burak_train,y_burak_train)
                    y_predict=pipeline.predict(X_burak_train)
                    eval_and_log_metrics("Burak Train",y_burak_train,y_predict)

                    y_predict=pipeline.predict(X_burak_test)
                    validation_utils.val_and_log_metrics(y_burak_test,y_predict,'Burak')
                else:
                    pipeline.fit(X_burak,y_burak)
                    y_predict=pipeline.predict(X_burak)
                    eval_and_log_metrics("Burak Train",y_burak,y_predict)

                y_predict=pipeline.predict(X_target_set)
                validation_utils.val_and_log_metrics(y_target_set,y_predict,'Target')

                #Explenability
                df=calculate_distribution(burak_set,target_set)
                df.to_csv('Distribution Source(Burak)-Target.csv',index=False)
                mlflow.log_artifact('Distribution Source(Burak)-Target.csv','Distribution')
                os.remove('Distribution Source(Burak)-Target.csv')

                df=calculate_distribution(burak_set.loc[burak_set[col.TARGET]==0],
                                          target_set.loc[target_set[col.TARGET]==0])
                df.to_csv('Distribution Non Flaky Test Source(Burak)-Target.csv',index=False)
                mlflow.log_artifact('Distribution Non Flaky Test Source(Burak)-Target.csv','Distribution')
                os.remove('Distribution Non Flaky Test Source(Burak)-Target.csv')

                df=calculate_distribution(burak_set.loc[burak_set[col.TARGET]==1],
                                          target_set.loc[target_set[col.TARGET]==1])
                df.to_csv('Distribution Flaky Test Source(Burak)-Target.csv',index=False)
                mlflow.log_artifact('Distribution Flaky Test Source(Burak)-Target.csv','Distribution')
                os.remove('Distribution Flaky Test Source(Burak)-Target.csv')


                fi=features_importance(pipeline.get_params('steps')['model'])
                fi.to_csv('Feature Importances Classifier.csv',index=False)
                mlflow.log_artifact('Feature Importances Classifier.csv','Feature Importances Classifier')
                os.remove('Feature Importances Classifier.csv')








if __name__ == "__main__":
    run(sys.argv[1])




