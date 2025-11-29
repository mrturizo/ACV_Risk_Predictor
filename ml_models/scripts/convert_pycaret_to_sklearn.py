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
    
    loaded_obj = None
    
    # Intentar cargar con joblib primero
    try:
        loaded_obj = joblib.load(pycaret_model_path)
        logger.info(f"Modelo cargado con joblib. Tipo: {type(loaded_obj)}")
    except Exception as e:
        logger.warning(f"Fallo joblib.load: {e}. Intentando con pickle...")
        # CRÍTICO: El archivo puede contener múltiples objetos serializados
        # Intentar cargar todos los objetos del archivo
        objects_loaded = []
        with open(pycaret_model_path, 'rb') as f:
            try:
                # Intentar cargar todos los objetos del archivo
                while True:
                    try:
                        obj = pickle.load(f)
                        objects_loaded.append(obj)
                        logger.info(f"Objeto {len(objects_loaded)} cargado: {type(obj)}")
                    except EOFError:
                        logger.info(f"Fin del archivo. Total de objetos cargados: {len(objects_loaded)}")
                        break
            except Exception as e2:
                logger.warning(f"Error al cargar objetos: {e2}")
        
        # Buscar el primer objeto que tenga métodos predict/predict_proba o sea un Pipeline
        for i, obj in enumerate(objects_loaded):
            # Verificar si es un Pipeline o tiene métodos predict
            if hasattr(obj, 'steps') or hasattr(obj, 'named_steps'):
                loaded_obj = obj
                logger.info(f"✅ Usando objeto {i+1}/{len(objects_loaded)} (tiene steps/named_steps): {type(loaded_obj)}")
                break
            elif hasattr(obj, 'predict') or hasattr(obj, 'predict_proba'):
                loaded_obj = obj
                logger.info(f"✅ Usando objeto {i+1}/{len(objects_loaded)} (tiene métodos predict): {type(loaded_obj)}")
                break
            elif isinstance(obj, dict):
                # Si es un diccionario, buscar dentro
                for key in ['model', 'pipeline', 'estimator', 'final_model', 'best_model']:
                    if key in obj:
                        candidate = obj[key]
                        if hasattr(candidate, 'steps') or hasattr(candidate, 'named_steps') or hasattr(candidate, 'predict'):
                            loaded_obj = candidate
                            logger.info(f"✅ Usando objeto {i+1}/{len(objects_loaded)} en clave '{key}': {type(loaded_obj)}")
                            break
                    if loaded_obj is not None:
                        break
        
        # Si no se encontró ningún objeto válido, usar el último (puede ser el Pipeline)
        if loaded_obj is None and objects_loaded:
            logger.warning("No se encontró objeto con métodos predict, usando el último objeto cargado")
            loaded_obj = objects_loaded[-1]
    
    if loaded_obj is None:
        raise ValueError("No se pudo cargar ningún objeto válido del archivo")
    
    # El modelo de PyCaret es un Pipeline de sklearn
    # Verificar que tenga el atributo 'steps' o 'named_steps' o métodos predict
    if hasattr(loaded_obj, 'steps') or hasattr(loaded_obj, 'named_steps'):
        logger.info("✅ Modelo es un Pipeline de sklearn. Se puede usar directamente.")
        return loaded_obj
    elif hasattr(loaded_obj, 'predict') or hasattr(loaded_obj, 'predict_proba'):
        logger.info("✅ Modelo tiene métodos predict/predict_proba. Se puede usar directamente.")
        return loaded_obj
    else:
        raise ValueError(f"El modelo no es un Pipeline de sklearn ni tiene métodos predict. Tipo: {type(loaded_obj)}")


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

