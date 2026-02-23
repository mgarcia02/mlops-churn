import os
import json
import uuid

import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.logging_config.setup import setup_logging, get_logger
from src.logging_config.utils import log_info, log_error

# ============================================================ 
# Configuración de rutas por defecto 
# ============================================================
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = "models/churn_pipeline.joblib"
DEFAULT_METADATA_PATH = "models/metadata.json"

# ============================================================ 
# Inicialización de FastAPI y variables globales 
# ============================================================
app = FastAPI(title="Churn Inference API")

logger = None
run_id = str(uuid.uuid4())
model = None
metadata = None

# ============================================================ 
# Esquemas de entrada y salida (Pydantic) 
# ============================================================
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

# ============================================================ 
# Eventos de inicio: logging + carga de artefactos 
# ============================================================
@app.on_event("startup")
def startup_logger():
    """Inicializa el sistema de logging al arrancar la API."""
    global logger
    
    setup_logging()
    logger = get_logger("inference")
    log_info(logger, "Logger inicializado", run_id=run_id)

@app.on_event("startup")
def load_artifacts():
    """Carga el modelo y la metadata al iniciar la API."""
    global model, metadata 
    
    if not os.path.exists(DEFAULT_MODEL_PATH): 
        raise RuntimeError(f"Model file not found at {DEFAULT_MODEL_PATH}") 
    if not os.path.exists(DEFAULT_METADATA_PATH): 
        raise RuntimeError(f"Metadata file not found at {DEFAULT_METADATA_PATH}") 
    
    model = joblib.load(DEFAULT_MODEL_PATH) 
    with open(DEFAULT_METADATA_PATH, "r") as f: 
        metadata = json.load(f)

# ============================================================ 
# Endpoints 
# ============================================================
@app.get("/health") 
def health(): 
    """Endpoint simple para comprobar que la API está funcionando."""
    return {"status": "ok"} 
    
@app.get("/model-info") 
def model_info():
    """Devuelve información del modelo cargado."""
    return { 
        "model_path": metadata.get("model_path"), 
        "saved_at": metadata.get("saved_at"), 
        "sklearn_version": metadata.get("sklearn_version"), 
        "metrics": metadata.get("metrics"), 
    }

@app.post("/predict", response_model=PredictionOutput) 
def predict(input_data: CustomerInput):
    """Realiza una predicción de churn a partir de los datos del cliente."""
    input_dict = input_data.dict()
    
    try: 
        log_info(logger, "Request recibido", run_id=run_id)
    
        # Convertir a DataFrame
        df = pd.DataFrame([input_dict])

        # Predicción
        pred = model.predict(df)[0]
        log_info(logger, "Predicción generada", run_id=run_id, prediction=int(pred))
        
        # Probabilidad (si existe predict_proba)
        if hasattr(model, "predict_proba"): 
            prob = float(model.predict_proba(df)[:, 1][0]) 
        else: 
            prob = float(pred) 
        
        return PredictionOutput( 
            churn_prediction=int(pred), 
            churn_probability=prob
        )
    except Exception as e:
        # Log completo del error 
        log_error(logger, "Error durante la predicción", run_id=run_id, input_dict=input_dict)
        
        # Respuesta limpia para el cliente 
        return JSONResponse(status_code=500, content={"detail": "Error interno durante la predicción"})
        
