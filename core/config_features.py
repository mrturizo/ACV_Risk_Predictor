"""Configuración centralizada de columnas de entrada para el modelo ACV.

Este módulo define el contrato de datos de entrada que deben cumplir
las aplicaciones web y de escritorio antes de llamar al predictor.
"""

from typing import List

# Columna objetivo del modelo
TARGET_COLUMN: str = "stroke"

# Columnas de entrada que usará el modelo definitivo (ordenadas)
# IMPORTANTE: Este orden DEBE coincidir exactamente con el orden que espera
# el modelo lr_pca25_cw.pkl según su feature_names_in_ (sin 'stroke').
# Este es el orden exacto que el pipeline de PyCaret espera recibir.
MODEL_INPUT_COLUMNS: List[str] = [
    "sleep time",
    "Minutes sedentary activity",
    "Waist Circumference",
    "Systolic blood pressure",
    "Diastolic blood pressure",
    "High-density lipoprotein",
    "Triglyceride",
    "Low-density lipoprotein",
    "Fasting Glucose",
    "Glycohemoglobin",
    "energy",
    "protein",
    "Dietary fiber",
    "Potassium",
    "Sodium",
    "gender",
    "age",
    "Race",
    "Marital status",
    "alcohol",
    "smoke",
    "sleep disorder",
    "Health Insurance",
    "General health condition",
    "depression",
    "diabetes",
    "hypertension",
    "high cholesterol",
    "Coronary Heart Disease",
    "Body Mass Index",
]


