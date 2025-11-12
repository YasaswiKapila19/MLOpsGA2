# app.py
import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title = "Iris Application")

model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"home":"Welcome to your IRIS prediction application"}

@app.post("/predict")
def predict(input: IrisInput):
    df = pd.DataFrame([{
        "sepal_length": input.sepal_length,
        "sepal_width": input.sepal_width,
        "petal_length": input.petal_length,
        "petal_width": input.petal_width
    }])
    y_pred = model.predict(df)
    return {"prediction": y_pred.tolist()}

