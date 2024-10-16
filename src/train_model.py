import sys
import joblib
import pandas as pd

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    model = sys.argv[1]
    X_train = sys.argv[2]
    y_train = sys.argv[3]
    X_train = pd.read_csv(X_train, encoding='unicode_escape')
    y_train = pd.read_csv(y_train, encoding='unicode_escape')
    model = joblib.load('models/not_trained/model_4.joblib')
    model = train_model(model, X_train, y_train)
    joblib.dump(model, 'models/model_4.joblib')