import mlflow
import pandas as pd
import os

# mlflow models serve -m models:/Best_model/

experiment_name = "Bank_churn_classifier"
highest_accuracy_run = mlflow.search_runs(
    experiment_names=[experiment_name],
    max_results=1,
    order_by=["metrics.'Accuracy' DESC"],
)

best_run_id = highest_accuracy_run.iloc[0]['run_id']

# print (highest_accuracy_run.columns)

result = mlflow.register_model(
    f"runs:/{best_run_id}/sklearn-model", "Best_model"
)


bashCommand = f"mlflow models serve -m runs:/{best_run_id}-model --port 5000"

os.system(bashCommand)