import mlflow
import pandas as pd
import os

# mlflow models serve -m runs:/415f0d7ce962454390268d6b7f308f78/sklearn-model --port 5000

experiment_name = "3_hyperparameter_opmization"
highest_accuracy_run = mlflow.search_runs(
    experiment_names=[experiment_name],
    max_results=1,
    order_by=["metrics.'Accuracy' DESC"],
)

best_run_id = highest_accuracy_run.iloc[0]['run_id']
print(best_run_id)

bashCommand = "mlflow models serve -m runs:/415f0d7ce962454390268d6b7f308f78/sklearn-model --port 5000"

os.system(bashCommand)