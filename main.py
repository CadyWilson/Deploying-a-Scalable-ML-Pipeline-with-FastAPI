import os
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# Define the paths
project_path = os.getcwd()  # Get the current working directory
encoder_path = os.path.join(project_path, "model", "encoder.pkl")  # Path for the encoder
model_path = os.path.join(project_path, "model", "model.pkl")      # Path for the model

# Load the encoder and model
encoder = load_model(encoder_path)
model = load_model(model_path)


print(f"Loading encoder from: {encoder_path}")
print(f"Loading model from: {model_path}")


# Define the FastAPI app
app = FastAPI()

# Define the Pydantic model for input validation
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# Create a GET request on the root
@app.get("/")
async def get_root():
    """Say hello!"""
    return {"message": "Howdy World!"}

# Create a POST request for model inference
@app.post("/data/")
async def post_inference(data: Data):
    data_dict = data.dict()
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

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
    data_processed, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder
    )
    _inference = inference(model, data_processed)  # Use the inference function to get predictions
    return {"result": apply_label(_inference)}

# Commenting here for Github Action purposes, version 8 billion and 11
# Now we're literally over 50 tries in. No exaggeration
# again, again!