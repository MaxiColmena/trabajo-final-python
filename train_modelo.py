"""
PUNTO 3 - Entrenamiento del modelo de prediccion de churn (baja de clientes)
Dataset: Telco Customer Churn (IBM)

Que hace este script:
1. Carga y limpia los datos.
2. Arma un Pipeline (preprocesamiento + modelo) en un solo objeto.
3. Entrena un RandomForest.
4. Evalua con metricas adecuadas a clases desbalanceadas.
5. Guarda TODO el pipeline en 'modelo.pkl' con joblib.

La clave: guardamos el Pipeline COMPLETO, no solo el modelo. Asi la API
recibe los datos crudos del cliente y el pipeline los transforma igual
que en el entrenamiento. Sin esto, la API se rompe.
"""

import json
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, recall_score, f1_score,
)

# --- Rutas ---
BASE = Path(__file__).parent
CSV = BASE / "dataset-churn" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CARPETA_MODELOS = BASE / "models"
CARPETA_MODELOS.mkdir(exist_ok=True)
SALIDA_MODELO = CARPETA_MODELOS / "modelo.pkl"
SALIDA_METRICAS = CARPETA_MODELOS / "metricas.json"

# ============================================================
# 1. CARGA Y LIMPIEZA
# ============================================================
df = pd.read_csv(CSV)
print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# customerID es solo un identificador, no aporta nada -> lo borramos
df = df.drop(columns=["customerID"])

# TotalCharges viene como TEXTO y tiene 11 celdas en blanco.
# Lo convertimos a numero; los blancos quedan como NaN (los imputamos despues).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# El target Churn viene como "Yes"/"No" -> lo pasamos a 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Separamos variables (X) del objetivo (y)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Que columnas son numericas y cuales categoricas
columnas_numericas = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
columnas_categoricas = [c for c in X.columns if c not in columnas_numericas]

print(f"Numericas ({len(columnas_numericas)}): {columnas_numericas}")
print(f"Categoricas ({len(columnas_categoricas)}): {columnas_categoricas}")

# ============================================================
# 2. PIPELINE DE PREPROCESAMIENTO + MODELO
# ============================================================
# Numericas: rellenamos faltantes con la mediana y estandarizamos (media 0, desvio 1)
preproceso_num = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
])

# Categoricas: las convertimos a columnas 0/1 (one-hot).
# handle_unknown='ignore' -> si en produccion llega una categoria nueva, no explota.
preproceso_cat = OneHotEncoder(handle_unknown="ignore")

preprocesador = ColumnTransformer(transformers=[
    ("num", preproceso_num, columnas_numericas),
    ("cat", preproceso_cat, columnas_categoricas),
])

# class_weight='balanced' -> compensa que hay muchos mas "No churn" que "churn"
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,   # usa todos los nucleos disponibles para entrenar mas rapido
)

pipeline = Pipeline(steps=[
    ("preprocesador", preprocesador),
    ("modelo", modelo),
])

# ============================================================
# 3. ENTRENAMIENTO
# ============================================================
# stratify=y -> mantiene la misma proporcion de churn en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
print("\nModelo entrenado.")

# ============================================================
# 4. EVALUACION
# ============================================================
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n--- Reporte de clasificacion ---")
print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))

print("--- Matriz de confusion ---")
cm = confusion_matrix(y_test, y_pred)
print(f"                 Pred No   Pred Si")
print(f"Real No churn     {cm[0,0]:>6}   {cm[0,1]:>6}")
print(f"Real Churn        {cm[1,0]:>6}   {cm[1,1]:>6}")

print(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

# ============================================================
# 5. GUARDAR EL MODELO Y LAS METRICAS
# ============================================================
# compress=3 reduce el archivo de ~41 MB a ~3 MB (mas liviano para el repo)
joblib.dump(pipeline, SALIDA_MODELO, compress=3)

# Guardamos las metricas en JSON para que la API las exponga en /metrics
# y la interfaz web pueda mostrar como rinde el modelo.
metricas = {
    "modelo": "Random Forest (200 arboles, class_weight=balanced)",
    "registros_entrenamiento": len(X_train),
    "registros_test": len(X_test),
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "recall_churn": round(recall_score(y_test, y_pred), 4),
    "f1_churn": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "matriz_confusion": cm.tolist(),
}
with open(SALIDA_METRICAS, "w", encoding="utf-8") as f:
    json.dump(metricas, f, indent=2, ensure_ascii=False)

print(f"\nModelo guardado en: {SALIDA_MODELO}")
print(f"Metricas guardadas en: {SALIDA_METRICAS}")
