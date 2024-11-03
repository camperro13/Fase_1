import pandas as pd
import pytest
import sys
import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

sys.path.insert(1, '/Users/juancarloscamperovilla/Documents/GitHub/MLOps/Residencial_build/Fase_1/src')
from load_data_v2 import load_data
from evaluate_model_v2 import evaluate_model
from preprocess_data_v2 import preprocess_data

@pytest.fixture
def sample_excel_file(tmp_path):
    # Create a sample Excel file for testing
    df = pd.DataFrame({
        'A': ['Column1', 'Data1', 'Data2'],
        'B': ['Column2', 'Data3', 'Data4']
    })
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False, header=False)
    return file_path

@pytest.fixture
def sample_data_x(tmp_path):
    # Generate sample regression data
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    # Save the data to CSV files
    X_train_file = tmp_path / "X_train.csv"
    y_train_file = tmp_path / "y_train.csv"
    X_test_file = tmp_path / "X_test.csv"
    y_test_file = tmp_path / "y_test.csv"
    pd.DataFrame(X_train).to_csv(X_train_file, index=False)
    pd.DataFrame(y_train).to_csv(y_train_file, index=False)
    pd.DataFrame(X_test).to_csv(X_test_file, index=False)
    pd.DataFrame(y_test).to_csv(y_test_file, index=False)
    return X_train_file, y_train_file, X_test_file, y_test_file, y

@pytest.fixture
def sample_data_y(tmp_path):
    # Create sample data for testing
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target_1'] = y + np.random.normal(0, 1, size=y.shape)  # Add noise to the target
    df['target_2'] = y + np.random.normal(0, 1, size=y.shape)  # Add another target
    csv_file = tmp_path / "sample_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file

@pytest.fixture
def sample_model(tmp_path):
    # Train a sample model and save it
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    model = LinearRegression()
    model.fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    return model_path

def test_load_data(sample_excel_file):
    # Load the data using the provided function
    data = load_data(sample_excel_file)
    # Check the structure and contents of the loaded data
    print(data)
    assert isinstance(data, pd.DataFrame), "Loaded data should be a DataFrame"
    assert list(data.columns) == ['Data1', 'Data3'], "Column names should match"
    assert data.shape == (1, 2), "Data should have 2 rows and 2 columns"
    assert data.iloc[0]['Data1'] == 'Data2', "First row"

def test_preprocess_data(sample_data_y):
    # Test the preprocess_data function
    X_train, X_test, y_train, y_test = preprocess_data(sample_data_y)
    # Check shapes of the returned datasets
    assert X_train.shape[0] + X_test.shape[0] == 100, "Total number of samples should be 100"
    assert y_train.shape[0] + y_test.shape[0] == 100, "Total number of target samples should be 100"
    # Check that the returned X datasets are DataFrames
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
    assert isinstance(y_train, pd.DataFrame), "y_train should be a DataFrame"
    assert isinstance(y_test, pd.DataFrame), "y_test should be a DataFrame"
    # Check for numeric data types in X_train
    assert all(np.issubdtype(X_train[col].dtype, np.number) for col in X_train.columns), "All columns in X_train should be numeric"
    # Check for the correct number of features in the training set
    assert X_train.shape[1] == 4, "X_train should have 4 features after processing (considering dropping highly correlated ones)"
    assert y_train.shape[1] == 2, "y_train should have 2 target variables"


def test_evaluate_model(sample_data_x, sample_model, tmp_path):
    X_train_file, y_train_file, X_test_file, y_test_file, _ = sample_data_x
    model_path = sample_model
    report_path = tmp_path / "evaluation_report.txt"
    # Load data into DataFrames
    X_train = pd.read_csv(X_train_file, encoding='unicode_escape')
    y_train = pd.read_csv(y_train_file, encoding='unicode_escape')
    X_test = pd.read_csv(X_test_file, encoding='unicode_escape')
    y_test = pd.read_csv(y_test_file, encoding='unicode_escape')
    # Run the evaluation
    evaluate_model(model_path, X_train, y_train, X_test, y_test, report_path)
    # Check if the report is created
    assert os.path.exists(report_path), "Report file should be created"
    # Read the report and validate contents
    with open(report_path) as f:
        report_contents = f.read()
        assert "Train Report" in report_contents, "Train report should be present"
        assert "Test Report" in report_contents, "Test report should be present"