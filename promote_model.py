from mlflow.tracking import MlflowClient

# Initialize the MLflow client
client = MlflowClient()

# Specify the model name and version you want to promote
model_name = "Best Model"  # Make sure this matches the name of your registered model
model_version = 2  # Replace with the correct version based on your context

# Transition the model to the 'Production' stage
client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production"
)

print(f"Model version {model_version} in '{model_name}' promoted to Production stage.")
