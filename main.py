"""
PUNTO 4 - API con FastAPI para consumir el modelo de churn.

Como se usa:
    1. Instalar:  pip install fastapi uvicorn scikit-learn pandas joblib
    2. Levantar:  uvicorn main:app --reload
    3. Abrir docs: http://127.0.0.1:8000/docs

La API carga 'modelo.pkl' (el pipeline entrenado en el Punto 3) y expone
el endpoint POST /predict, que recibe los datos de UN cliente y devuelve
si se va a dar de baja (churn) y con que probabilidad.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- Cargamos el modelo una sola vez, al arrancar ---
BASE = Path(__file__).parent
modelo = joblib.load(BASE / "models" / "modelo.pkl")

app = FastAPI(title="API Prediccion de Churn", version="1.0")

# CORS: permite que la interfaz (index.html) llame a la API desde el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Estructura de los datos que entran (validacion automatica) ---
# Los nombres deben coincidir EXACTO con las columnas del entrenamiento.
class Cliente(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)            # 0 o 1
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)                  # meses de antiguedad
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=845.5)


@app.get("/")
def inicio():
    return {"mensaje": "API de prediccion de churn activa. Ver /docs"}


@app.get("/app")
def frontend():
    """Sirve la interfaz web desde el mismo servidor."""
    return FileResponse(BASE / "index.html")


@app.get("/metrics")
def metricas():
    """Devuelve las metricas de evaluacion del modelo (calculadas sobre el
    conjunto de test al entrenar). La interfaz las usa para mostrar
    como rinde el modelo."""
    with open(BASE / "models" / "metricas.json", encoding="utf-8") as f:
        return json.load(f)


@app.post("/predict")
def predecir(cliente: Cliente):
    # Convertimos el cliente en un DataFrame de 1 fila (lo que espera el pipeline)
    datos = pd.DataFrame([cliente.model_dump()])

    # Prediccion (0 = se queda, 1 = se da de baja)
    prediccion = int(modelo.predict(datos)[0])
    # Probabilidad de que se de de baja
    probabilidad = float(modelo.predict_proba(datos)[0][1])

    return {
        "churn": bool(prediccion),
        "probabilidad_churn": round(probabilidad, 4),
        "mensaje": "El cliente probablemente se da de baja"
                   if prediccion == 1
                   else "El cliente probablemente se queda",
    }
