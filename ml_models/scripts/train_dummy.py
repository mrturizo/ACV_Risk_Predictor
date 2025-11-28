"""
Script para generar un modelo dummy de PyCaret para pruebas del sistema.

Este modelo se usa para validar la arquitectura antes de tener datos reales
o modelos entrenados con datos reales.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from pycaret.classification import setup, create_model, finalize_model, save_model
import sys

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_dummy_data(n_samples: int = 1000) -> pd.DataFrame:
    """Genera datos dummy para entrenar un modelo de prueba.
    
    Args:
        n_samples: Número de muestras a generar.
        
    Returns:
        DataFrame con datos dummy simulados.
    """
    np.random.seed(42)
    
    # Variables demográficas
    age = np.random.randint(18, 90, n_samples)
    gender = np.random.choice([0, 1], n_samples)  # 0: Femenino, 1: Masculino
    
    # Variables clínicas
    hypertension = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    heart_disease = np.random.choice([0, 1], n_samples, p=[0.90, 0.10])
    ever_married = np.random.choice([0, 1], n_samples, p=[0.20, 0.80])
    work_type = np.random.choice([0, 1, 2, 3, 4], n_samples)
    Residence_type = np.random.choice([0, 1], n_samples)
    avg_glucose_level = np.random.normal(95, 20, n_samples)
    bmi = np.random.normal(28, 5, n_samples)
    smoking_status = np.random.choice([0, 1, 2, 3], n_samples)
    
    # Crear target con alguna lógica simple (no realista, solo para dummy)
    # Mayor probabilidad de stroke si: edad alta, hipertensión, glucosa alta
    stroke_prob = (
        0.1 * (age > 65) +
        0.2 * hypertension +
        0.15 * heart_disease +
        0.1 * (avg_glucose_level > 120) +
        0.05 * (bmi > 30) +
        np.random.normal(0, 0.1, n_samples)
    )
    stroke = (stroke_prob > 0.3).astype(int)
    
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status,
        'stroke': stroke
    })
    
    return data


def train_dummy_model(
    output_path: Optional[Path] = None,
    model_name: str = "dummy_stroke_model"
) -> Path:
    """Entrena un modelo dummy y lo guarda.
    
    Args:
        output_path: Ruta donde guardar el modelo. Si es None, usa models/
        model_name: Nombre del modelo.
        
    Returns:
        Ruta al modelo guardado.
    """
    print("Generando datos dummy...")
    data = generate_dummy_data(n_samples=1000)
    
    print(f"Datos generados: {data.shape}")
    print(f"Distribución de stroke: {data['stroke'].value_counts().to_dict()}")
    
    # Configurar PyCaret
    print("\nConfigurando PyCaret...")
    clf = setup(
        data=data,
        target='stroke',
        train_size=0.8,
        session_id=42,
        silent=True,
        verbose=False
    )
    
    # Crear y entrenar modelo (usando Random Forest por ser rápido y estable)
    print("Entrenando modelo dummy (Random Forest)...")
    model = create_model('rf', verbose=False)
    
    # Finalizar modelo (entrena con todos los datos)
    print("Finalizando modelo...")
    final_model = finalize_model(model)
    
    # Guardar modelo
    if output_path is None:
        output_path = project_root / "models"
    
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / f"{model_name}.pkl"
    
    print(f"\nGuardando modelo en: {model_file}")
    save_model(final_model, str(model_file)[:-4])  # PyCaret agrega .pkl automáticamente
    
    print(f"✓ Modelo dummy guardado exitosamente!")
    print(f"  Ubicación: {model_file}")
    print(f"\nColumnas esperadas por el modelo:")
    print(f"  {', '.join([col for col in data.columns if col != 'stroke'])}")
    
    return model_file


if __name__ == "__main__":
    print("=" * 60)
    print("Generador de Modelo Dummy para ACV Risk Predictor")
    print("=" * 60)
    print()
    
    model_path = train_dummy_model()
    
    print("\n" + "=" * 60)
    print("¡Modelo dummy generado con éxito!")
    print("=" * 60)
    print("\nEste modelo puede usarse para probar la aplicación.")
    print("NO debe usarse para predicciones reales.")

