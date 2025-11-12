import time
import pandas as pd
import joblib
from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Iris Application")

# Load model at startup
model = joblib.load("model.joblib")

# Initialize Prometheus instrumentator1we
Instrumentator().instrument(app).expose(app)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"Request processed in {process_time:.4f} seconds")
    return response

@app.get("/")
def home():
    return {"home": "Welcome to your IRIS prediction application"}

@app.post("/predict")
def predict(input: IrisInput):
    start_time = time.time()

    df = pd.DataFrame([{
        "sepal_length": input.sepal_length,
        "sepal_width": input.sepal_width,
        "petal_length": input.petal_length,
        "petal_width": input.petal_width
    }])
    y_pred = model.predict(df)
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.4f}s")
    return {"prediction": y_pred.tolist(), "inference_time": inference_time}
