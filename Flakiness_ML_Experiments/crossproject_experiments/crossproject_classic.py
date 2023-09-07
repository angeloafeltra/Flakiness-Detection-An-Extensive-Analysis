import sys
import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
from sklearn.model_selection import train_test_split
from utils.crossproject_utils import calculate_distribution, features_importance
import os

def run(dataset, pipeline, experiment_ID):

    list_project = dataset[col.CATEGORICAL_FEATURES[0]].unique()

    with mlflow.start_run(run_name= 'CrossProject_Classic',experiment_id= experiment_ID) as father_run:

        for target in list_project:
            print("Target: ",target)

            #Inserire qui il criterio di filtraggio per skippare il target

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

                print("Source TF:{} - TNF:{}\n Target TF:{} - TNF:{} ".format(source_TF,
                                                                              source_TNF,
                                                                              target_TF,
                                                                              target_TNF))

                X_source_set = source_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_source_set = source_set[col.TARGET]
                X_target_set = target_set.drop([col.TARGET] + col.CATEGORICAL_FEATURES, axis = 1)
                y_target_set = target_set[col.TARGET]




                X_source_set_train, X_source_set_test, y_source_set_train, y_source_set_test = train_test_split(X_source_set, y_source_set,
                                                                                                                stratify = y_source_set,
                                                                                                                test_size = 0.2,
                                                                                                                random_state = 42)


                pipeline.fit(X_source_set_train, y_source_set_train)
                y_predict=pipeline.predict(X_source_set_test)
                validation_utils.val_and_log_metrics(y_source_set_test, y_predict,'Source')

                y_predict=pipeline.predict(X_target_set)
                validation_utils.val_and_log_metrics(y_target_set,y_predict,'Target')

                #Explenability
                df=calculate_distribution(source_set,target_set)
                df.to_csv('Distribution Source-Target.csv')
                mlflow.log_artifact('Distribution Source-Target.csv','Distribution Source-Target')
                os.remove('Distribution Source-Target.csv')

                df=calculate_distribution(source_set.loc[source_set[col.TARGET]==0],
                                          target_set.loc[target_set[col.TARGET]==0])
                df.to_csv('Distribution Non Flaky Test Source-Target.csv')
                mlflow.log_artifact('Distribution Non Flaky Test Source-Target.csv','Distribution Non Flaky Test Source-Target')
                os.remove('Distribution Non Flaky Test Source-Target.csv')

                df=calculate_distribution(source_set.loc[source_set[col.TARGET]==1],
                                          target_set.loc[target_set[col.TARGET]==1])
                df.to_csv('Distribution Flaky Test Source-Target.csv')
                mlflow.log_artifact('Distribution Flaky Test Source-Target.csv','Distribution Flaky Test Source-Target')
                os.remove('Distribution Flaky Test Source-Target.csv')


                fi=features_importance(pipeline.get_params('steps')['model'])
                fi.to_csv('Feature Importances Classifier.csv')
                mlflow.log_artifact('Feature Importances Classifier.csv','Feature Importances Classifier')
                os.remove('Feature Importances Classifier.csv')






if __name__ == "__main__":
    run(sys.argv[1])




