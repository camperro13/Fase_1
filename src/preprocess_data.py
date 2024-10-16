import pandas as pd
import sys
import numpy as np
import data_explorer
import data_versioning
from sklearn.model_selection import train_test_split

def preprocess_data(data):
    column_names = []
    data = pd.read_csv(data)
    #Cambiar nombres a variables para evitar tener los mismos nombres.
    for i in range (0, len(data.columns)):
        column_names.append(data.columns[i])
    for i in range (0, len(data.columns)):
        if data.columns[i] in column_names:
            column_names[i] = column_names[i] + '_' + str(i)
    data.columns = column_names
    #Convertir a variables numéricas
    for col in  data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    #data_explorer.plot_correlation_matrix(data)
    #data_explorer.plot_histograms(data)
    nc = data_explorer.pca_get(data)
    #Remover outliners
    for variable in data.iloc[:,:-2].columns:
        percentile_10 = data[variable].quantile(0.10)
        percentile_90 = data[variable].quantile(0.90)
        iqr = percentile_90 - percentile_10
        upper_limit = percentile_90 + 1.5 * iqr
        lower_limit = percentile_10 - 1.5 * iqr
        outliers = data[(data[variable] < lower_limit) | (data[variable] > upper_limit)]
        data = data[~data[variable].isin(outliers[variable])]
    #Remover variables altamente correlacionadas
    corr_matrix = data.corr().abs()# get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))# find features with correlation greater than 0.98
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]# drop highly correlated features
    data.drop(to_drop, axis=1, inplace=True)
    #data_versioning.save(data)
    #Dividir el dataset
    X = data.iloc[:,:-2]
    Y = data.iloc[:,[-2,-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    data = sys.argv[1]
    X_train, X_test, y_train, y_test = preprocess_data(data)
    X_train.to_csv('data/processed/X_train.csv', index=False) 
    X_test.to_csv('data/processed/X_test.csv', index=False) 
    y_train.to_csv('data/processed/y_train.csv', index=False) 
    y_test.to_csv('data/processed/y_test.csv', index=False) 