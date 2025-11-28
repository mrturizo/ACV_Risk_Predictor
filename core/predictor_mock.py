"""Versión mock del predictor para pruebas sin PyCaret."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class StrokePredictorMock:
    """Versión mock del predictor para pruebas sin PyCaret.
    
    Esta clase simula el comportamiento del StrokePredictor real
    para permitir probar la interfaz web sin necesidad de PyCaret.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Inicializa el predictor mock.
        
        Args:
            model_path: Ignorado en la versión mock, pero se acepta
                       para compatibilidad con la interfaz.
        """
        self.model = "mock_model"
        self.model_path = model_path or Path("models/mock_model.pkl")
        self.required_columns: List[str] = [
            'age', 'gender', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        logger.info("Predictor Mock inicializado (sin PyCaret)")
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Realiza una predicción mock basada en reglas simples.
        
        Args:
            data: DataFrame con los datos del paciente.
                  
        Returns:
            Diccionario con resultados de predicción simulados.
        """
        if data.empty:
            raise ValueError("El DataFrame de entrada está vacío.")
        
        # Obtener la primera fila
        row = data.iloc[0]
        
        # Lógica simple de predicción mock (NO es realista, solo para pruebas)
        risk_score = 0.0
        
        # Factores de riesgo simples
        if 'age' in row and row['age'] > 65:
            risk_score += 0.2
        if 'hypertension' in row and row['hypertension'] == 1:
            risk_score += 0.3
        if 'heart_disease' in row and row['heart_disease'] == 1:
            risk_score += 0.25
        if 'avg_glucose_level' in row and row['avg_glucose_level'] > 140:
            risk_score += 0.15
        if 'bmi' in row and row['bmi'] > 30:
            risk_score += 0.1
        
        # Agregar algo de aleatoriedad para variar resultados
        risk_score += np.random.uniform(-0.1, 0.1)
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Decidir predicción
        prediction_str = "STROKE RISK" if risk_score > 0.5 else "NOT STROKE RISK"
        
        result = {
            'prediction': prediction_str,
            'probability': risk_score,
            'details': {
                'model_path': str(self.model_path),
                'raw_prediction': 1 if risk_score > 0.5 else 0,
                'raw_score': risk_score,
                'note': 'Esta es una predicción MOCK para pruebas. No use para decisiones médicas reales.'
            }
        }
        
        logger.info(f"Predicción MOCK realizada: {prediction_str} (probabilidad: {risk_score:.2%})")
        
        return result
    
    def get_required_columns(self) -> List[str]:
        """Retorna las columnas requeridas."""
        return self.required_columns.copy()

