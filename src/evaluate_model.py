from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import sys
import pandas as pd
import joblib

def evaluate_model(model, X_train, y_train, X_test, y_test):
    print("Model Evaluation (train):")
    Y_train_pred_lin = model.predict(X_train)
    train_r2_lin = r2_score(y_train, Y_train_pred_lin)
    train_mse_lin = mean_squared_error(y_train, Y_train_pred_lin)
    print('r2_score: ' + str(train_r2_lin))
    print('mean_squared_error: ' + str(train_mse_lin))
    print("Model Evaluation (test):")
    y_pred = model.predict(X_test)
    r2_score_ = r2_score(y_test, y_pred)
    mean_squared_error_ = mean_squared_error(y_test, y_pred)
    print('r2_score: ' + str(r2_score_))
    print('mean_squared_error: ' + str(mean_squared_error_))

if __name__ == '__main__':
    model = sys.argv[1]
    X_test = sys.argv[2]
    y_test = sys.argv[3]
    X_train = sys.argv[4]
    y_train = sys.argv[5]
    X_train = pd.read_csv(X_train, encoding='unicode_escape')
    y_train = pd.read_csv(y_train, encoding='unicode_escape')
    X_test = pd.read_csv(X_test, encoding='unicode_escape')
    y_test = pd.read_csv(y_test, encoding='unicode_escape')
    model = joblib.load('models/model_4.joblib')
    evaluate_model(model, X_train, y_train, X_test, y_test)