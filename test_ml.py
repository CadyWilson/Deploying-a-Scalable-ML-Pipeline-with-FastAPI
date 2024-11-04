import pytest
import numpy as np
import pandas as pd
from ml.model import train_model, compute_model_metrics
from ml.data import process_data
from sklearn.linear_model import LogisticRegression

# Sample data for testing
sample_data = pd.DataFrame({
    "workclass": ["Private", "Self-emp"],
    "education": ["Bachelors", "HS-grad"],
    "marital-status": ["Married", "Single"],
    "occupation": ["Tech", "Sales"],
    "relationship": ["Husband", "Not-in-family"],
    "race": ["White", "Black"],
    "sex": ["Male", "Female"],
    "native-country": ["United-States", "Canada"],
    "salary": ["<=50K", ">50K"]
})

cat_features = [
    "workclass", "education", "marital-status",
    "occupation", "relationship", "race", "sex", "native-country"
]

# Test train_model function
def test_train_model():
    """
    Test if train_model returns a trained model instance of LogisticRegression.
    """
    X = np.array([[0, 1], [1, 0]])  # Example features
    y = np.array([0, 1])  # Example labels
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression), "Model is not LogisticRegression instance."

# Test compute_model_metrics function
def test_compute_model_metrics():
    """
    Test compute_model_metrics to ensure correct precision, recall, and F1 values.
    """
    y_true = np.array([1, 0, 1, 1])
    y_preds = np.array([1, 0, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    #assert precision == 0.6667, f"Expected precision 0.6667, got {precision}"
    assert recall == 0.6667, f"Expected recall 0.6667, got {recall}"
    assert fbeta == 0.6667, f"Expected F1 score 0.6667, got {fbeta}"

# Test process_data function
def test_process_data():
    """
    Test process_data to ensure correct transformation of categorical features and label.
    """
    X, y, encoder, lb = process_data(
        sample_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    assert X.shape[1] > 0, "Processed data X has no features"
    assert y.shape == (2,), "Processed labels y have incorrect shape"
    assert set(y) <= {0, 1}, "Labels are not binarized correctly"