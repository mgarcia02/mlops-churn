import os
import json

import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

DEFAULT_MODEL_PATH = "models/churn_pipeline.joblib"
DEFAULT_METADATA_PATH = "models/metadata.json"

app = FastAPI(title="Churn Inference API")

# Define the input and output schemas
class CustomerInput(BaseModel):
    gender: str = Field(..., example="Female") 
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str
    Dependents: str  
    tenure: int = Field(..., ge=0, example=5)
    PhoneService: str 
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0, example=70.5) 
    TotalCharges: float | None = Field(None, ge=0, example=350.0) 

class PredictionOutput(BaseModel): 
    churn_prediction: int 
    churn_probability: float

# Inicializa las variables globales
model = None
metadata = None

@app.on_event("startup")
def load_artifacts(): 
    global model, metadata 
    
    if not os.path.exists(DEFAULT_MODEL_PATH): 
        raise RuntimeError(f"Model file not found at {DEFAULT_MODEL_PATH}") 
    if not os.path.exists(DEFAULT_METADATA_PATH): 
        raise RuntimeError(f"Metadata file not found at {DEFAULT_METADATA_PATH}") 
    
    model = joblib.load(DEFAULT_MODEL_PATH) 
    with open(DEFAULT_METADATA_PATH, "r") as f: 
        metadata = json.load(f)

# Endpoints
@app.get("/health") 
def health(): 
    return {"status": "ok"} 
    
@app.get("/model-info") 
def model_info(): 
    return { 
        "model_path": metadata.get("model_path"), 
        "saved_at": metadata.get("saved_at"), 
        "sklearn_version": metadata.get("sklearn_version"), 
        "metrics": metadata.get("metrics"), 
    }

@app.post("/predict", response_model=PredictionOutput) 
def predict(input_data: CustomerInput): 
    # Convierte la entrada en un DataFrame
    df = pd.DataFrame([input_data.dict()]) 
    # Predicción
    pred = model.predict(df)[0] 
    
    if hasattr(model, "predict_proba"): 
        prob = float(model.predict_proba(df)[:, 1][0]) 
    else: 
        prob = float(pred) 
    
    return PredictionOutput( 
        churn_prediction=int(pred), 
        churn_probability=prob
    )