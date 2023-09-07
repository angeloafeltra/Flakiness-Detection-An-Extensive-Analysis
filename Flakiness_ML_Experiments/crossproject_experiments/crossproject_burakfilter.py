import sys
import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
from utils.burak_utils import classic_burakFilter
import numpy as np
from sklearn.model_selection import train_test_split

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
                mlflow.log_metric("Target Test Flaky", source_TF)
                mlflow.log_metric("Target Test Non Flaky", source_TNF)

                print("Source TF:{} - TNF:{}\n Target TF:{} - TNF:{}\n Burak_Set TF:{} - TNF:{}".format(source_TF,
                                                                              source_TNF,
                                                                              target_TF,
                                                                              target_TNF))


                X_source_set = source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_source_set = source_set[col.TARGET]



                X_target_set = target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_target_set = target_set[col.TARGET]


                X_burak, y_burak =  classic_burakFilter(X_source_set.to_numpy(),
                                                                    y_source_set.to_numpy(),
                                                                    X_target_set.to_numpy(),
                                                                    10)

                burak_TF=np.count_nonzero(y_burak)
                burak_TNF=y_burak.size - burak_TF

                mlflow.log_metric("Target Test Flaky", source_TF)
                mlflow.log_metric("Target Test Non Flaky", source_TNF)

                print("Source TF:{} - TNF:{}\n Target TF:{} - TNF:{}\n Burak_Set TF:{} - TNF:{}".format(source_TF,
                                                                                                        source_TNF,
                                                                                                        target_TF,
                                                                                                        target_TNF,
                                                                                                        burak_TF,
                                                                                                        burak_TNF))


                X_burak_train, X_burak_test, y_burak_train, y_burak_test = train_test_split(X_burak, y_burak,
                                                                                            stratify = y_burak,
                                                                                            test_size = 0.2,
                                                                                            random_state = 42)


                pipeline.fit(X_burak_train,y_burak_train)
                y_predict=pipeline.predict(X_burak_test)
                validation_utils.val_and_log_metrics(y_burak_test,y_predict,'Burak')
                y_predict=pipeline.predict(X_target_set)
                validation_utils.val_and_log_metrics(y_target_set,y_predict,'Target')








if __name__ == "__main__":
    run(sys.argv[1])




