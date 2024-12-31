import dagshub
import mlflow

# Correcting the method name from set_tracking_url to set_tracking_uri
mlflow.set_tracking_uri("https://dagshub.com/ishanpeshkar/mlops_project.mlflow")

dagshub.init(repo_owner='ishanpeshkar', repo_name='mlops_project', mlflow=True)

with mlflow.start_run():
    mlflow.log_param('parameter name', 'value')
    mlflow.log_metric('metric name', 1)
