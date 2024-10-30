import numpy as np
import yaml
#from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer, make_column_selector
import sys
import data_explorer
import joblib
import pandas as pd
import mlflow
from sklearn.ensemble import RandomForestRegressor

def load_params():
    with open("params_2.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def pipeline_making(data, X_train, y_train, model_type):
    data = pd.read_csv(data)
    X_train = pd.read_csv(X_train)
    y_train = pd.read_csv(y_train)
    nc = data_explorer.pca_get(data)
    standar = StandardScaler()
    #minmax = MinMaxScaler()
    pca = PCA(n_components=nc)
    preprocessing = ColumnTransformer([
    #('minmax', minmax, make_column_selector(dtype_include=np.number)),
    ('stan', standar, make_column_selector(dtype_include=np.number)),
    ('pca', pca, make_column_selector(dtype_include=np.number))
    ], remainder = 'passthrough')
    if model_type == 'logistic_regression':
        model = make_pipeline(preprocessing, LinearRegression(**model_params))
    elif model_type == 'random_forest':
        model = make_pipeline(preprocessing, RandomForestRegressor(**model_params))
    #model.fit(X_train, y_train)
    client = mlflow.tracking.MlflowClient()
    try:
        mlflow.create_experiment(params['mlflow']['experiment_name'])
    except mlflow.exceptions.MlflowException as e:
        print(f"Experiment may already exist: {e}")
    try:
        experiment = client.get_experiment_by_name(params['mlflow']['experiment_name'])
        print(f"Using existing experiment: {experiment.name}")
    except mlflow.exceptions.MlflowException:
        # If the experiment does not exist, create it
        experiment = client.create_experiment(params['mlflow']['experiment_name'])
        print(f"Created new experiment: {experiment.name}")
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    with mlflow.start_run():
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, f"{model_type}_model")
    return model

if __name__ == '__main__':
    data = sys.argv[1]
    X_train = sys.argv[2]
    y_train = sys.argv[3]
    model_type = sys.argv[4]
    params = load_params()
    model_params = params['models'][model_type]
    model = pipeline_making(data, X_train, y_train, model_type)
    model_dir = params['data']['models']
    model_path = f"{model_dir}/{model_type}_model.joblib"
    joblib.dump(model, model_path)