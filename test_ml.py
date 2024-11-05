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

def test_train_model_empty_data():
    """
    Ensure train_model raises an error or returns None with empty data.
    """
    X = np.array([])  # Empty features
    y = np.array([])  # Empty labels
    with pytest.raises(ValueError):
        train_model(X, y)
