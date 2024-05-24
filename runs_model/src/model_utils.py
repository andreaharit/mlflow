# General imports
import pandas as pd

# For splitting
from sklearn.model_selection import train_test_split

# For model encoding and inputing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# For pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# For metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

# For plots
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import PrecisionRecallDisplay

# For typing
from typing import Dict
from typing import List
from typing import Optional

def load_data(filepath):
    """
    Load data from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(filepath)
    return df

def clean_df(df):
    """
    Clean the input DataFrame by removing specific columns.

    Parameters:
        df (pandas.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pandas.DataFrame: The cleaned DataFrame with specified columns dropped.
    """
    df = df[df.columns[:-2]]
    df = df.drop(['CLIENTNUM'], axis=1)
    return df

def split_data (df, target_column):
    # Splitting data
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def preprocessing (X_train):    
    # Identifying categoricals and numericals
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_train.select_dtypes(exclude=['object', 'category']).columns

    # Numerical preprocessing
    numerical_pipeline = make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler()  
    )

    # Categorical preprocessing
    categorical_pipeline = make_pipeline(
        OneHotEncoder(handle_unknown='ignore')
    )

    # ColumnTransformer 
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_pipeline, categorical_cols),
            ('num', numerical_pipeline, numerical_cols) 
        ],
        remainder='passthrough'
    )
    return preprocessor


def get_pipeline (preprocessor, model):
    pipe = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('Model', model),
        
    ])
    return pipe



def set_metrics(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.DataFrame, 
    y_test: pd.DataFrame, 
    y_pred: pd.DataFrame, 
    pipeline):

    cv_results = cross_validate(pipeline, X_train, y_train, cv=5, scoring='accuracy', return_train_score=True)
    precision = precision_score(y_true = y_test, y_pred= y_pred, pos_label= 'Existing Customer', average=None)
    recall = recall_score(y_true = y_test, y_pred= y_pred, pos_label= 'Existing Customer', average=None)
    f1 = f1_score(y_true = y_test, y_pred= y_pred, pos_label= 'Existing Customer', average=None)    


    metrics = {
    "R2_train_score" : round(pipeline.score(X_train, y_train), 3),
    "R2_test_score": round(pipeline.score(X_test, y_test), 3),
    "ROC_AUC": round(roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]), 3),
    "Accuracy": round(accuracy_score(y_true = y_test, y_pred = y_pred), 3),
    "Precision_Attrited": round(precision[0], 3),
    "Precision_Existing": round(precision[1],3),
    "Recall_Attrited": round(recall[0],3),
    "Recall_Existing": round(recall[1],3),
    "F1-Score_Attrited": round(f1[0],3),
    "F1-Score_Existing":  round(f1[1],3),

    # Cross validation
    "Cross_Mean_test_accuracy": round(cv_results['test_score'].mean(), 3),
    "Cross_Mean_train_accuracy": round(cv_results['train_score'].mean(), 3),
    "Cross_mean_fit_time": round(cv_results['fit_time'].mean(), 3),
    "Cross_mean_score_time": round(cv_results['score_time'].mean(),3)
    }
    # Classification report
    report = classification_report(y_test, y_pred)

    return metrics, report


def plots(y_test: pd.DataFrame, y_pred: pd.DataFrame, pos_label):
    """
    Get performance plots.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param prefix: Prefix for the plot names.
    :return: Performance plots.
    """
    roc_figure = plt.figure()
    RocCurveDisplay.from_predictions(y_test, y_pred, ax=plt.gca(), pos_label = pos_label)

    cm_figure = plt.figure()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=plt.gca())

    pr_figure = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_test, y_pred, ax=plt.gca(), pos_label = pos_label)

    return {
        "Roc_curve": roc_figure,
        "Confusion_matrix": cm_figure,
        "Precision_recall_curve": pr_figure,
    }
