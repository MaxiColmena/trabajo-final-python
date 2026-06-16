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
from pydantic import BaseModel, Field