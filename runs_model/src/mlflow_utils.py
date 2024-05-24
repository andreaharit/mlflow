
import mlflow



def create_or_find_experiment(experiment_name: str):
    """
    Create a new experiment and set it.
    But if it already exists it will set it for the run. 
    """

    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id



def best_run (experiment_name: str, metric: str) -> None:
    """
    Find the model with the highest score between all runs inside an experiment according to a metric.
    """
    highest_accuracy_run = mlflow.search_runs(
    experiment_names=[experiment_name],
    max_results=1,
    order_by=[f"metrics.'{metric}' DESC"],
    )

    best_run_id = highest_accuracy_run.iloc[0]['run_id']

    result = mlflow.register_model(
    f"runs:/{best_run_id}/sklearn-model", "Best_model"
    )