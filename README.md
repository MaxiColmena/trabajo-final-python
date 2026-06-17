# Predicción de Churn — Telco Customer Churn

Trabajo Práctico Final — **Taller de Lenguajes de Programación III: Python para Ciencia de Datos**

**Alumnos:** 
|- Colman, Maximo Javier Alexis.
|- Martínez, Javier Nicolás.
|- Pereyra, Ramiro Nicolás.

Solución integral de Machine Learning que predice si un cliente de una empresa de
telecomunicaciones se va a dar de baja (churn), cubriendo el ciclo completo:
análisis exploratorio → preprocesamiento → modelado → API → interfaz web.

## El problema

Retener un cliente cuesta mucho menos que conseguir uno nuevo. El modelo permite
detectar clientes con alta probabilidad de baja para que el área comercial actúe
antes (descuentos, mejoras de plan, contacto proactivo).

## Dataset

- **Telco Customer Churn (IBM)** — Kaggle
- 7.043 registros × 21 columnas.
- Target: `Churn` (Yes/No) — clases desbalanceadas (~26,5% churn)
- Archivo: `dataset-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Análisis y modelado | Python, Pandas, Seaborn, Matplotlib, scikit-learn |
| Modelo | Random Forest dentro de un Pipeline (imputación + estandarización + one-hot) |
| Exportación | joblib (`models/modelo.pkl`) |
| Backend / API | FastAPI + Uvicorn |
| Frontend | HTML + JavaScript (fetch a la API) |

## Estructura del proyecto

```
desarrollo/
├── notebook.ipynb        # EDA, preprocesamiento, modelado, evaluación y exportación
├── train_modelo.py       # Script de entrenamiento (mismo pipeline que el notebook)
├── main.py               # API FastAPI: POST /predict y GET /metrics
├── index.html            # Interfaz web que consume la API
├── models/
│   ├── modelo.pkl        # Pipeline completo exportado con joblib
│   └── metricas.json     # Métricas de evaluación (las muestra la interfaz)
├── dataset-churn/        # Dataset original
└── requirements.txt      # Dependencias
```

## Cómo ejecutarlo

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. (Opcional) Reentrenar el modelo — ya viene entrenado en models/
python train_modelo.py
#    o ejecutar notebook.ipynb de punta a punta

# 3. Levantar la API
uvicorn main:app --reload
#    Documentación interactiva: http://127.0.0.1:8000/docs

# 4. Abrir la interfaz
#    http://127.0.0.1:8000/app  (servida desde la API)
#    o doble clic en index.html (también funciona, usa CORS)
```

## Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/` | Estado de la API |
| GET | `/metrics` | Métricas del modelo (accuracy, recall, F1, ROC AUC, matriz de confusión) |
| POST | `/predict` | Recibe los datos de un cliente (JSON) y devuelve la predicción y probabilidad |

Ejemplo de `POST /predict`:

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 95.0, "TotalCharges": 95.0
}
```

Respuesta:

```json
{ "churn": true, "probabilidad_churn": 0.78, "mensaje": "El cliente probablemente se da de baja" }
```

## Modelo y métricas

Se compararon Regresión Logística (línea base) y Random Forest; se eligió
**Random Forest** (200 árboles, `class_weight='balanced'`) por capturar relaciones
no lineales y ser robusto a outliers. Las métricas sobre el 20% de test
(estratificado) están en `models/metricas.json` y se muestran en la interfaz.
