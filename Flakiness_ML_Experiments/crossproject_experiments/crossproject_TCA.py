import sys
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import utils.columns as col
import mlflow
import utils.validation_utils as validation_utils
import numpy as np
from adapt.feature_based import TCA
from utils.burak_utils import classic_burakFilter
from sklearn.preprocessing import StandardScaler
from utils.eval_utils import eval_and_log_metrics
from utils.crossproject_utils import calculate_distribution, features_importance


def run(dataset, pipeline, experiment_ID):


    list_project=dataset['nameProject'].unique()

    with mlflow.start_run(run_name= 'CrossProject_TCA', experiment_id= experiment_ID) as father_run:

        for target in list_project[0:]:
            print("Target: ",target)

            #Inserire qui il criterio di filtraggio per skippare il target
            tmp=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]
            testFlaky=len(tmp[tmp[col.TARGET]==1])
            testNonFlaky=len(tmp[tmp[col.TARGET]==0])

            if testFlaky<20 or testFlaky>testNonFlaky:
                print("La repository non rispetta i criteri")
                continue


            with mlflow.start_run(run_name=target, experiment_id=experiment_ID, nested=True) as child_run:

                target_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]==target]
                source_set=dataset.loc[dataset[col.CATEGORICAL_FEATURES[0]]!=target]

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

                # Utilizzo il fitro di burak per ridurre il source set
                X_burak, y_burak, _ , _ , _ , _ =  classic_burakFilter(X_source_set.to_numpy(),
                                                                        y_source_set.to_numpy(),
                                                                        X_target_set.to_numpy(),
                                                                        10,
                                                                        y_target_set.to_numpy())


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


                # Applico la TCA
                std=StandardScaler()
                tca=TCA(X_target_set,n_components=len(col.NUMERICAL_FEATURES))
                clf=pipeline.get_params('steps')['model']

                if burak_TF>=2:
                    X_burak_train, X_burak_test, y_burak_train, y_burak_test = train_test_split(X_burak, y_burak,
                                                                                                stratify = y_burak,
                                                                                                test_size = 0.2,
                                                                                                random_state = 42)
                    X_burak_train=std.fit_transform(X_burak_train)
                    X_burak_test=std.transform(X_burak_test)

                    #Eseguo la TCA sul train
                    print("Eseguo la tca...")
                    X_tca_train=tca.fit_transform(X_burak_train,X_target_set)

                    tca_set=np.insert(X_tca_train, -1, y_burak_train.to_numpy(), axis=1)
                    columns=[]
                    columns=columns+col.NUMERICAL_FEATURES
                    columns.append(col.TARGET)
                    tca_set=pd.DataFrame(tca_set,columns=columns)



                    #Addesto il modello
                    clf.fit(X_tca_train,y_burak_train)
                    #Faccio la predict sul train
                    y_predict=clf.predict(X_tca_train)
                    eval_and_log_metrics("Train",y_burak_train,y_predict)
                    #Eseguo la TCA su test
                    print("Eseguo la tca su test...")
                    X_tca_test=tca.transform(X_burak_test)
                    #Faccio la predict su test
                    y_predict=clf.predict(X_tca_test)
                    validation_utils.val_and_log_metrics(y_burak_test,y_predict,'Source (TCA)')

                else:
                    X_burak=std.fit_transform(X_burak)
                    print("Eseguo la tca...")
                    X_tca=tca.fit_transform(X_burak,X_target_set)

                    tca_set=np.insert(X_tca, -1, y_burak.to_numpy(), axis=1)
                    columns=[]
                    columns=columns+col.NUMERICAL_FEATURES
                    columns.append(col.TARGET)
                    tca_set=pd.DataFrame(tca_set,columns=columns)


                    #Addesto il modello
                    clf.fit(X_tca,y_burak)
                    #Faccio la predict sul train
                    y_predict=clf.predict(X_tca)
                    eval_and_log_metrics("Train",y_burak,y_predict)



                #Eseguo la TCA sul target
                X_target_set=std.transform(X_target_set)
                X_target_tca=tca.transform(X_target_set)

                target_tca_set=np.insert(X_target_tca, -1, y_target_set.to_numpy(), axis=1)
                columns=[]
                columns=columns+col.NUMERICAL_FEATURES
                columns.append(col.TARGET)
                target_tca_set=pd.DataFrame(target_tca_set,columns=columns)



                y_predict=clf.predict(X_target_tca,y_target_set)
                validation_utils.val_and_log_metrics(y_target_set,y_predict,'Target (TCA)')


                #Explenability

                df=calculate_distribution(target_set,target_tca_set)
                df.to_csv('Distribution Target-Target(TCA).csv',index=False)
                mlflow.log_artifact('Distribution Target-Target(TCA).csv','Distribution')
                os.remove('Distribution Target-Target(TCA).csv')

                df=calculate_distribution(tca_set,target_tca_set)
                df.to_csv('Distribution Source(TCA)-Target(TCA).csv',index=False)
                mlflow.log_artifact('Distribution Source(TCA)-Target(TCA).csv','Distribution')
                os.remove('Distribution Source(TCA)-Target(TCA).csv')

                df=calculate_distribution(tca_set.loc[tca_set[col.TARGET]==0],
                                          target_tca_set.loc[target_tca_set[col.TARGET]==0])
                df.to_csv('Distribution Non Flaky Test Source(TCA)-Target(TCA).csv',index=False)
                mlflow.log_artifact('Distribution Non Flaky Test Source(TCA)-Target(TCA).csv','Distribution')
                os.remove('Distribution Non Flaky Test Source(TCA)-Target(TCA).csv')

                df=calculate_distribution(tca_set.loc[tca_set[col.TARGET]==1],
                                          target_tca_set.loc[target_tca_set[col.TARGET]==1])
                df.to_csv('Distribution Flaky Test Source(TCA)-Target(TCA).csv',index=False)
                mlflow.log_artifact('Distribution Flaky Test Source(TCA)-Target(TCA).csv','Distribution')
                os.remove('Distribution Flaky Test Source(TCA)-Target(TCA).csv')


                fi=features_importance(clf)
                fi.to_csv('Feature Importances Classifier.csv',index=False)
                mlflow.log_artifact('Feature Importances Classifier.csv','Feature Importances Classifier')
                os.remove('Feature Importances Classifier.csv')





if __name__ == "__main__":
    run(sys.argv[1])




