from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import sys
import pandas as pd
import joblib
import os

def evaluate_model(model_path, X_train, y_train, X_test, y_test, report_path):
    model = joblib.load(model_path)
    Y_train_pred_lin = model.predict(X_train)
    train_r2_lin = r2_score(y_train, Y_train_pred_lin)
    train_mse_lin = mean_squared_error(y_train, Y_train_pred_lin)
    r2_score_train = str(train_r2_lin)
    mean_squared_error_train =  str(train_mse_lin)
    y_pred = model.predict(X_test)
    r2_score_ = r2_score(y_test, y_pred)
    mean_squared_error_ = mean_squared_error(y_test, y_pred)
    r2_score_test = str(r2_score_)
    mean_squared_error_test = str(mean_squared_error_)
    write_evaluation_report(report_path, r2_score_train, mean_squared_error_train, r2_score_test, mean_squared_error_test)

def write_evaluation_report(report_path, r2_score_train, mean_squared_error_train, r2_score_test, mean_squared_error_test):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write("Train Report:\n")
        f.write("R2 score:\n")
        f.write(r2_score_train)
        f.write("\nMean squared error:\n")
        f.write(mean_squared_error_train)
        f.write("\n")
        f.write("\nTest Report:\n")
        f.write("R2 score:\n")
        f.write(r2_score_test)
        f.write("\nMean squared error:\n")
        f.write(mean_squared_error_test)

if __name__ == '__main__':
    model_path = sys.argv[1]
    X_test = sys.argv[2]
    y_test = sys.argv[3]
    X_train = sys.argv[4]
    y_train = sys.argv[5]
    report_path = sys.argv[6]
    X_train = pd.read_csv(X_train, encoding='unicode_escape')
    y_train = pd.read_csv(y_train, encoding='unicode_escape')
    X_test = pd.read_csv(X_test, encoding='unicode_escape')
    y_test = pd.read_csv(y_test, encoding='unicode_escape')
    evaluate_model(model_path, X_train, y_train, X_test, y_test, report_path)