# Make pipeline
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# MlFlow
import mlflow
from mlflow_utils import create_or_find_experiment, best_run
from model_utils import load_data, clean_df, split_data, set_metrics
from mlflow.models.signature import infer_signature

# Hyperopt
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import hp
from hyperopt import space_eval
from functools import partial
from parameters import parameters

# Typing
from typing import Dict
from typing import List
from typing import Optional

# General importing
import warnings
import pandas as pd
from configs import experiment_naming, max_evaluations, comparision_metric


def get_sklearn_pipeline(
    chosen_model, numerical_features: List[str], categorical_features: Optional[List[str]] = []
) -> Pipeline:
    """
    Builds the sklearn pipeline.

    :chosen_model: The classification model to fit.
    :param numerical_features: The numerical features.
    :param categorical_features: The categorical features.
    :return: The sklearn pipeline.
    """

    # Numerical preprocessing
    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()  
    )

    # ColumnTransformer for numerical and categorical
    preprocessing = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),            
            (
                "categorical",
                OneHotEncoder(handle_unknown='ignore'),
                categorical_features,
            ),
        ]
    )
    # Model pipeline
    pipeline = ImbPipeline([
        ('preprocessor', preprocessing),
        ('smote', SMOTE(random_state=42)),
        ('model', chosen_model)
    ])
 
    return pipeline


def objective_function(
    params: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    chosen_model,
    numerical_features: List[str],
    categorical_features: List[str],
) -> float:
    """
    Objective function to minimize for hyperparam optmization.

    :params: The hyperparameter values to evaluate.
    :param X_train: The training data.
    :param X_test: The test data.
    :param y_train: The training target.
    :param y_test: The test target.
    :param numerical_features: The numerical features.
    :param categorical_features: The categorical features.
    :chosen_model: The classification model to be tested.
    :return: The score of the model.
    """
    # Make pipeline object
    pipeline = get_sklearn_pipeline(chosen_model = chosen_model, numerical_features = numerical_features, categorical_features= categorical_features)

    # Set parameters       
    pipeline.set_params(**params)

    # Starts run children for parameters
    with mlflow.start_run(nested=True) as run:

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Colects metrics for minimization
        metrics, report = set_metrics(
            X_train = X_train, 
            X_test = X_test, 
            y_train = y_train, 
            y_test = y_test, 
            y_pred = y_pred, 
            pipeline = pipeline)

        # Logs information of the child run into MLflow
        mlflow.log_params(pipeline["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, f"{run.info.run_id}-model")

    return -metrics[comparision_metric]


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category= UserWarning)
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Loading data
    df = load_data('../data/BankChurners.csv')
    # Cleaning data
    df = clean_df(df)
    # Splitting Data
    target_column = "Attrition_Flag"
    X_train, X_test, y_train, y_test = split_data(df = df, target_column = target_column)
 
    # Signature of dataset for mlflow
    signature = infer_signature(X_train, y_train)

    # Separate numerical and cateforical features
    numerical_features =  X_train.select_dtypes(exclude=['object', 'category']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns


    # Reads parameters for hyperopmization
    hyper_parameters = parameters    

    # Starts experiment
    experiment_id = create_or_find_experiment(experiment_name = experiment_naming)
    
    # Starts runs for each model
    for key,value in hyper_parameters.items():
        # Sets parameters for hyperopt
        run_name = key
        chosen_model = value["algo"]
        space = value["space"]

        # Starts parent run
        with mlflow.start_run(run_name= run_name) as run:

            # Starts hyperparm optimization with minimizing function
            best_params = fmin(
                fn = partial(
                    objective_function,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    chosen_model = chosen_model,
                    numerical_features=numerical_features,
                    categorical_features=categorical_features,
                ),
                space=space,
                algo=tpe.suggest,
                max_evals=max_evaluations,
                trials=Trials(),
            )

            # Starts pipeline for best parameters
            pipeline = get_sklearn_pipeline(chosen_model = chosen_model, numerical_features=numerical_features, categorical_features = categorical_features)

            # Sets best parameters 
            best_parameters = space_eval(space, best_params)
            print (best_parameters)
            pipeline.set_params(**best_parameters)

            # Fitting and predicting
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Gets metrics
            metrics, report = set_metrics(
                X_train = X_train, 
                X_test = X_test, 
                y_train = y_train, 
                y_test = y_test, 
                y_pred = y_pred, 
                pipeline = pipeline)

            # Logging information of best model per classification algo into MLFlow
            mlflow.log_params(pipeline["model"].get_params())
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(sk_model=pipeline, signature=signature, artifact_path="sklearn-model", registered_model_name= f"best-model-{run_name}")
        
    # Finding and registering the best model of all classifiers   
    best_run(experiment_name = experiment_naming, metric = comparision_metric)