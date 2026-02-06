import os
import argparse
from datetime import datetime
import json

import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DEFAULT_RAW = os.path.join(ROOT, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
DEFAULT_MODEL_DIR = os.path.join(ROOT, 'models')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, 'churn_pipeline.joblib')
METADATA_PATH = os.path.join(DEFAULT_MODEL_DIR, 'metadata.json')

def load_data(path: str) -> pd.DataFrame: 
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path)

def clean_and_split(df: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.2, seed: int = 42):
    df = df.dropna(subset=[target_col]).copy()

    for col in ['id', 'customerID', 'customer_id']:
        if col in df.columns:
            df = df.drop(columns=[col])

    if 'TotalCharges' in df.columns: 
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})

    x = df.drop(columns=[target_col]) 
    y = df[target_col] 

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y) 
    
    return x_train, x_test, y_train, y_test

def build_pipeline(x_sample: pd.DataFrame):
    cat_cols = x_sample.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = x_sample.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, cat_cols),
        ('num', num_pipeline, num_cols)
    ])

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline, num_cols, cat_cols

def train_pipeline(pipeline, x_train, y_train): 
    pipeline.fit(x_train, y_train)

    return pipeline

def evaluate_model(pipeline, x_test, y_test):
    preds = pipeline.predict(x_test)
    probs = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    return {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds)),
        'recall': float(recall_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs)) if probs is not None else None
    }

def build_metadata(model_path, metrics, num_cols, cat_cols, seed):
    return {
        "saved_at": datetime.utcnow().isoformat(),
        "model_path": model_path,
        "metrics": metrics,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "random_state": seed,
        "sklearn_version": sklearn.__version__
    }

def save_artifacts(pipeline, model_path, metadata, metadata_path):
    save_pipeline(pipeline, model_path)
    save_metadata(metadata_path, metadata)

def save_pipeline(pipeline, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

def save_metadata(path: str, metadata: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)

def main(data_path: str, model_path: str, test_size: float, seed: int):
    print(f"Cargando datos desde {data_path}")
    df = load_data(data_path)

    x_train, x_test, y_train, y_test = clean_and_split(df, test_size=test_size, seed=seed)

    pipeline, num_cols, cat_cols = build_pipeline(x_train)

    print(f"Iniciando entrenamiento con {len(x_train)} ejemplos de entrenamiento y {len(x_test)} ejemplos de test")
    pipeline = train_pipeline(pipeline, x_train, y_train)

    print("Evaluando el modelo")
    metrics = evaluate_model(pipeline, x_test, y_test)

    metadata = build_metadata(model_path, metrics, num_cols, cat_cols, seed)
    save_artifacts(pipeline, model_path, metadata, METADATA_PATH)

    print(f"Modelo guardado en {model_path}")
    print("Métricas:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena y guarda un pipeline para churn prediction")
    parser.add_argument("--data", type=str, default=DEFAULT_RAW, help="ruta al CSV de datos")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="ruta donde guardar el pipeline")
    parser.add_argument("--test-size", type=float, default=0.2, help="proporción de test")
    parser.add_argument("--seed", type=int, default=42, help="semilla aleatoria")
    args = parser.parse_args()
    main(args.data, args.model, args.test_size, args.seed)