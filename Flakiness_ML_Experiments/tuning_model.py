from sklearn.model_selection import ParameterGrid
import utils.experimentsList as experimentList
import mlflow
import utils.eval_utils

def tuning_pipeline(X_train_set, y_train_set, X_val_set, y_val_set, pipeline, pipelineName, parentID, experimentID):

    for step in pipeline.steps:
        if step[0]=='model':
            estimator=step[1].__class__.__name__

    for tuningModel in experimentList.TUNING:
        if tuningModel['classificatore'] == estimator :
            list_params=ParameterGrid(tuningModel['tuning_param'])


    with mlflow.start_run(run_name='Tuning_{}'.format(pipelineName), experiment_id=experimentID,):

        for param,i in zip(list(list_params),range(0,len(list_params))):
            print('|---{}_Tuning{}'.format(pipelineName,i))
            with mlflow.start_run(experiment_id=experimentID,
                                    nested=True,
                                    run_name='{}_Tuning{}'.format(pipelineName,i)):

                for step in pipeline.steps:
                    if step[0]=='model':
                        step[1].set_params(**param)
                        mlflow.log_params(step[1].get_params())

                pipeline.fit(X=X_train_set, y=y_train_set)
                y_pred=pipeline.predict(X=X_train_set)
                utils.eval_utils.eval_and_log_metrics('Train',y_train_set,y_pred)
                y_pred=pipeline.predict(X_val_set)
                utils.eval_utils.eval_and_log_metrics('Val',y_val_set,y_pred)
                mlflow.sklearn.log_model(
                    pipeline, '{}_Tuning{}'.format(pipelineName,i), pyfunc_predict_fn="predict"
                )

    mlflow.end_run()


if __name__ == "__main__":
    tuning_pipeline()