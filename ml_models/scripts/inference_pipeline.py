"""
Pipeline de inferencia para cargar modelo de producción y hacer predicciones.

Este módulo proporciona funciones para cargar el modelo de producción,
aplicar preprocesamiento y realizar predicciones en datos nuevos.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, Any, Optional, Union

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pycaret.classification import load_model, predict_model
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False

from ml_models.scripts.preprocessing import load_preprocessor, apply_preprocessing

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rutas
PRODUCTION_MODELS_DIR = project_root / "models"
PREPROCESSOR_PATH = project_root / "ml_models" / "trained_models" / "preprocessor.pkl"


def load_production_model(model_path: Optional[Path] = None) -> Any:
    """Carga el modelo de producción.
    
    Args:
        model_path: Ruta al modelo. Si None, busca en models/.
        
    Returns:
        Modelo cargado.
    """
    if model_path is None:
        # Buscar modelo en producción
        model_files = list(PRODUCTION_MODELS_DIR.glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No se encontró modelo en {PRODUCTION_MODELS_DIR}")
        model_path = model_files[0]
    
    logger.info(f"Cargando modelo desde: {model_path}")
    
    # Intentar cargar como modelo PyCaret primero
    if PYCARET_AVAILABLE:
        try:
            model = load_model(str(model_path)[:-4])  # PyCaret agrega .pkl
            logger.info("Modelo cargado como PyCaret")
            return model, 'pycaret'
        except Exception:
            pass
    
    # Cargar como pickle estándar
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Modelo cargado como pickle estándar")
        return model, 'sklearn'
    except Exception as e:
        raise ValueError(f"Error cargando modelo: {e}")


def preprocess_new_data(
    data: pd.DataFrame,
    preprocessor_path: Optional[Path] = None
) -> pd.DataFrame:
    """Aplica el preprocesamiento a datos nuevos.
    
    Args:
        data: DataFrame con datos nuevos.
        preprocessor_path: Ruta al preprocesador. Si None, usa la ruta por defecto.
        
    Returns:
        DataFrame preprocesado.
    """
    if preprocessor_path is None:
        preprocessor_path = PREPROCESSOR_PATH
    
    if not preprocessor_path.exists():
        logger.warning(f"Preprocesador no encontrado en {preprocessor_path}")
        logger.warning("Usando datos sin preprocesamiento adicional")
        return data
    
    logger.info(f"Cargando preprocesador desde: {preprocessor_path}")
    preprocessor = load_preprocessor(preprocessor_path)
    
    # Aplicar preprocesamiento
    # Asumimos que los datos ya vienen en el formato correcto
    # Si el preprocesador espera un formato específico, ajustar aquí
    try:
        if hasattr(preprocessor, 'transform'):
            data_processed = preprocessor.transform(data)
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
                data_processed = pd.DataFrame(data_processed, columns=feature_names, index=data.index)
            else:
                data_processed = pd.DataFrame(data_processed, index=data.index)
        else:
            data_processed = data
    except Exception as e:
        logger.warning(f"Error aplicando preprocesador: {e}")
        logger.warning("Usando datos sin preprocesamiento")
        data_processed = data
    
    return data_processed


def predict(
    data: pd.DataFrame,
    model: Any,
    model_type: str = 'sklearn',
    return_proba: bool = True
) -> Dict[str, Any]:
    """Realiza predicción con el modelo.
    
    Args:
        data: DataFrame con datos a predecir.
        model: Modelo entrenado.
        model_type: Tipo de modelo ('sklearn' o 'pycaret').
        return_proba: Si retornar probabilidades.
        
    Returns:
        Diccionario con predicciones y probabilidades.
    """
    if model_type == 'pycaret' and PYCARET_AVAILABLE:
        # PyCaret maneja el preprocesamiento internamente
        predictions = predict_model(model, data=data, verbose=False)
        
        # PyCaret retorna un DataFrame con predicciones
        if 'Label' in predictions.columns:
            y_pred = predictions['Label'].values
        elif 'prediction_label' in predictions.columns:
            y_pred = predictions['prediction_label'].values
        else:
            y_pred = predictions.iloc[:, -1].values
        
        if 'Score' in predictions.columns and return_proba:
            y_proba = predictions['Score'].values
        elif 'prediction_score' in predictions.columns and return_proba:
            y_proba = predictions['prediction_score'].values
        else:
            y_proba = None
    else:
        # sklearn estándar
        y_pred = model.predict(data)
        y_proba = model.predict_proba(data)[:, 1] if return_proba and hasattr(model, 'predict_proba') else None
    
    result = {
        'prediction': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        'probability': y_proba.tolist() if y_proba is not None and isinstance(y_proba, np.ndarray) else y_proba
    }
    
    return result


def predict_batch(
    data: pd.DataFrame,
    model_path: Optional[Path] = None,
    preprocessor_path: Optional[Path] = None
) -> pd.DataFrame:
    """Realiza predicción en lote.
    
    Args:
        data: DataFrame con múltiples muestras.
        model_path: Ruta al modelo.
        preprocessor_path: Ruta al preprocesador.
        
    Returns:
        DataFrame con predicciones agregadas.
    """
    # Cargar modelo
    model, model_type = load_production_model(model_path)
    
    # Preprocesar datos
    data_processed = preprocess_new_data(data, preprocessor_path)
    
    # Predecir
    predictions = predict(data_processed, model, model_type, return_proba=True)
    
    # Agregar predicciones al DataFrame original
    result_df = data.copy()
    result_df['stroke_prediction'] = predictions['prediction']
    result_df['stroke_probability'] = predictions['probability']
    
    return result_df


if __name__ == "__main__":
    # Ejemplo de uso
    print("Pipeline de Inferencia - ACV Risk Predictor")
    print("=" * 60)
    print("\nEste módulo proporciona funciones para:")
    print("  - load_production_model(): Cargar modelo de producción")
    print("  - preprocess_new_data(): Aplicar preprocesamiento")
    print("  - predict(): Realizar predicción")
    print("  - predict_batch(): Predicción en lote")
    print("\nUso:")
    print("  from ml_models.scripts.inference_pipeline import predict_batch")
    print("  predictions = predict_batch(new_data)")

