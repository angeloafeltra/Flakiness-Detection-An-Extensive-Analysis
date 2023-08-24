import mlflow
import utils.experimentsList
import utils.eval_utils

from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline

def run(X_train_set, y_train_set, X_val_set, y_val_set, pipelineName, experimentID):

    with mlflow.start_run(run_name=pipelineName, experiment_id=experimentID) as father_run:
        mlflow.set_tag("Pipeline","Pipeline")

        for clf in utils.experimentsList.CLASSIFICATORI:

            with mlflow.start_run(run_name='{}_{}'.format(pipelineName,clf.__class__.__name__), experiment_id=experimentID, nested=True) as child_run:
                print("|--- {}".format(clf.__class__.__name__))
                pipeline = Pipeline(steps = [('scaler',MinMaxScaler()),("model", clf)]).set_output(transform = "pandas")
                pipeline.fit(X=X_train_set, y=y_train_set)
                y_pred=pipeline.predict(X=X_train_set)
                utils.eval_utils.eval_and_log_metrics('Train',y_train_set,y_pred)
                y_pred=pipeline.predict(X_val_set)
                utils.eval_utils.eval_and_log_metrics('Val',y_val_set,y_pred)
                mlflow.sklearn.log_model(
                    pipeline, "{}_{}".format(pipelineName,clf.__class__.__name__), pyfunc_predict_fn="predict"
                )

    mlflow.end_run()

if __name__ == "__main__":
    run()

