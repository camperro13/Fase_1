import numpy as np
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

def pipeline_making(data, X_train, y_train):
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
    model = make_pipeline(preprocessing, LinearRegression())
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    data = sys.argv[1]
    X_train = sys.argv[2]
    y_train = sys.argv[3]
    model = pipeline_making(data, X_train, y_train)
    joblib.dump(model, 'models/model_4.joblib')
    