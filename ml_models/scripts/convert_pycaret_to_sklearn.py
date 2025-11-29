"""Script para convertir un modelo PyCaret a un Pipeline de sklearn puro.

Este script carga un modelo entrenado con PyCaret y extrae el Pipeline de sklearn
subyacente, guardándolo como un modelo puro de sklearn que no requiere PyCaret.

Uso:
    python ml_models/scripts/convert_pycaret_to_sklearn.py
"""

from pathlib import Path
import sys
import joblib
import pickle
import logging

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_sklearn_pipeline(pycaret_model_path: Path) -> object:
    """Extrae el Pipeline de sklearn de un modelo PyCaret.
    
    Args:
        pycaret_model_path: Ruta al modelo .pkl de PyCaret.
        
    Returns:
        Pipeline de sklearn extraído.
    """
    logger.info(f"Cargando modelo PyCaret desde: {pycaret_model_path}")
    
    try:
        # Intentar cargar con joblib primero
        model = joblib.load(pycaret_model_path)
        logger.info(f"Modelo cargado con joblib. Tipo: {type(model)}")
    except Exception as e:
        logger.warning(f"Fallo joblib.load: {e}. Intentando con pickle...")
        with open(pycaret_model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Modelo cargado con pickle. Tipo: {type(model)}")
    
    # El modelo de PyCaret es un Pipeline de sklearn
    # Verificar que tenga el atributo 'steps' o 'named_steps'
    if hasattr(model, 'steps') or hasattr(model, 'named_steps'):
        logger.info("✅ Modelo es un Pipeline de sklearn. Se puede usar directamente.")
        return model
    else:
        raise ValueError(f"El modelo no es un Pipeline de sklearn. Tipo: {type(model)}")


def save_sklearn_model(pipeline: object, output_path: Path) -> None:
    """Guarda un Pipeline de sklearn como modelo puro.
    
    Args:
        pipeline: Pipeline de sklearn a guardar.
        output_path: Ruta donde guardar el modelo.
    """
    logger.info(f"Guardando Pipeline de sklearn en: {output_path}")
    
    # Guardar con joblib (estándar para modelos sklearn)
    joblib.dump(pipeline, output_path)
    logger.info(f"✅ Modelo guardado exitosamente: {output_path}")


def main():
    """Función principal para convertir el modelo."""
    # Rutas
    models_dir = project_root / "models"
    input_model = models_dir / "lr_pca25_cw.pkl"
    output_model = models_dir / "lr_pca25_cw_sklearn.pkl"
    
    if not input_model.exists():
        logger.error(f"❌ Modelo no encontrado: {input_model}")
        return
    
    try:
        # Extraer Pipeline de sklearn
        sklearn_pipeline = extract_sklearn_pipeline(input_model)
        
        # Guardar como modelo sklearn puro
        save_sklearn_model(sklearn_pipeline, output_model)
        
        logger.info("✅ Conversión completada exitosamente!")
        logger.info(f"   Modelo original: {input_model}")
        logger.info(f"   Modelo convertido: {output_model}")
        logger.info(f"   Tamaño original: {input_model.stat().st_size / 1024:.2f} KB")
        logger.info(f"   Tamaño convertido: {output_model.stat().st_size / 1024:.2f} KB")
        
    except Exception as e:
        logger.error(f"❌ Error durante la conversión: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

