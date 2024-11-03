import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Define the paths
# use os.getcwd() if the absolute path doesn't work
project_path = "/home/cady/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")

# TODO: Load the census.csv data
data = pd.read_csv(data_path)

# TODO: Split the provided data into training and testing datasets
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.2, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# TODO: Use the process_data function provided to process the training data
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)

# Process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb
)

# TODO: Use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# Save the model and encoder
save_model(model, model_path)
save_model(encoder, encoder_path)

# Load the model (optional step to verify the saving works correctly)
model = load_model(model_path)

# TODO: Use the inference function to run the model inferences on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: Compute performance on model slices using the performance_on_categorical_slice function
# Iterate through the categorical features
for col in cat_features:
    # Iterate through the unique values in one categorical feature
    for slice_value in sorted(test[col].unique()):
        count = test[test[col] == slice_value].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test,
            column_name=col,
            slice_value=slice_value,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model
        )
        # Append the slice metrics to the output file
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slice_value}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
