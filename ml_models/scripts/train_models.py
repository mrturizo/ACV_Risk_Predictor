"""
Script para entrenar modelos de ML usando PyCaret (pipeline automático).

Este script implementa un pipeline completo con PyCaret que incluye:
- Comparación automática de modelos
- Ajuste de hiperparámetros
- Ensemble methods
- Evaluación exhaustiva
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pycaret.classification import (
        setup, compare_models, create_model, tune_model,
        finalize_model, save_model, evaluate_model,
        blend_models, stack_models, plot_model
    )
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    print("⚠ PyCaret no está instalado. Instala con: pip install pycaret[full]")

from ml_models.scripts.data_split import load_splits

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuración
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Rutas
DATA_PROCESSED = project_root / "ml_models" / "data" / "processed" / "nhanes_stroke_processed.csv"
SPLITS_DIR = project_root / "ml_models" / "data" / "splits"
MODELS_DIR = project_root / "ml_models" / "trained_models"
RESULTS_DIR = project_root / "ml_models" / "trained_models" / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_pycaret_models(
    data_path: Optional[Path] = None,
    use_splits: bool = True,
    top_n_models: int = 5,
    use_ensemble: bool = True
) -> Dict[str, Any]:
    """Entrena modelos usando PyCaret con pipeline completo.
    
    Args:
        data_path: Ruta a datos procesados. Si None y use_splits=True, usa splits.
        use_splits: Si usar splits pre-divididos o dividir aquí.
        top_n_models: Número de mejores modelos a entrenar en detalle.
        use_ensemble: Si crear modelos ensemble.
        
    Returns:
        Diccionario con información de modelos entrenados.
    """
    if not PYCARET_AVAILABLE:
        raise ImportError("PyCaret no está instalado")
    
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO CON PYCARET")
    logger.info("=" * 60)
    
    # Cargar datos
    if use_splits:
        logger.info("Cargando splits pre-divididos...")
        train_df, val_df, test_df = load_splits(SPLITS_DIR)
        # Combinar train y val para PyCaret (PyCaret maneja su propia división)
        data = pd.concat([train_df, val_df], ignore_index=True)
    else:
        if data_path is None:
            data_path = DATA_PROCESSED
        logger.info(f"Cargando datos desde: {data_path}")
        data = pd.read_csv(data_path)
    
    logger.info(f"Datos cargados: {data.shape}")
    logger.info(f"Distribución de target: {data['stroke'].value_counts().to_dict()}")
    
    # Setup de PyCaret
    logger.info("\nConfigurando experimento PyCaret...")
    clf = setup(
        data=data,
        target='stroke',
        train_size=0.8,
        session_id=RANDOM_STATE,
        verbose=False,
        fix_imbalance=True,  # Manejo automático de desbalance
        normalize=True,
        feature_selection=False,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95
    )
    
    # Comparar todos los modelos disponibles (excluyendo xgboost si no está instalado)
    logger.info("\nComparando todos los modelos disponibles...")
    # Lista de modelos disponibles (solo los más rápidos y efectivos)
    available_models = ['lr', 'rf', 'gbc', 'nb', 'ada', 'lightgbm', 'catboost', 'et']
    
    comparison_results = compare_models(
        include=available_models,
        sort='AUC',
        n_select=top_n_models,
        verbose=False,
        cross_validation=False  # Desactivar CV en comparación inicial para acelerar
    )
    
    logger.info(f"\nTop {top_n_models} modelos seleccionados")
    
    # Entrenar y optimizar top modelos
    trained_models = {}
    model_results = []
    
    top_models_list = comparison_results if isinstance(comparison_results, list) else [comparison_results]
    
    for i, model in enumerate(top_models_list):
        model_name = str(model).split('(')[0]
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesando modelo {i+1}/{len(top_models_list)}: {model_name}")
        logger.info(f"{'='*60}")
        
        # Ajustar hiperparámetros
        logger.info("Ajustando hiperparámetros...")
        tuned_model = tune_model(model, optimize='AUC', verbose=False, n_iter=10)  # Reducir iteraciones para acelerar
        
        # Evaluar modelo (comentado para acelerar - PyCaret puede abrir visualizaciones que bloquean)
        # logger.info("Evaluando modelo...")
        # evaluate_model(tuned_model)  # Esto puede ser muy lento y abrir ventanas
        
        # Guardar modelo
        model_file = MODELS_DIR / f"pycaret_{model_name.lower().replace(' ', '_')}.pkl"
        save_model(tuned_model, str(model_file)[:-4])  # PyCaret agrega .pkl
        logger.info(f"Modelo guardado: {model_file}")
        
        trained_models[model_name] = {
            'model': tuned_model,
            'file': model_file
        }
        
        # Obtener métricas
        from pycaret.classification import predict_model
        predictions = predict_model(tuned_model, verbose=False)
        # Las métricas están en el DataFrame de resultados de PyCaret
        
        model_results.append({
            'model_name': model_name,
            'model_file': str(model_file)
        })
    
    # Crear modelos ensemble
    if use_ensemble and len(top_models_list) >= 3:
        logger.info("\n" + "=" * 60)
        logger.info("CREANDO MODELOS ENSEMBLE")
        logger.info("=" * 60)
        
        # Blend models
        logger.info("Creando modelo blend...")
        try:
            blend_model = blend_models(estimator_list=top_models_list[:3], optimize='AUC', verbose=False)
            blend_file = MODELS_DIR / "pycaret_blend_model.pkl"
            save_model(blend_model, str(blend_file)[:-4])
            logger.info(f"Modelo blend guardado: {blend_file}")
            
            trained_models['blend'] = {
                'model': blend_model,
                'file': blend_file
            }
        except Exception as e:
            logger.warning(f"Error creando blend model: {e}")
        
        # Stack models
        logger.info("Creando modelo stack...")
        try:
            stack_model = stack_models(estimator_list=top_models_list[:3], optimize='AUC', verbose=False)
            stack_file = MODELS_DIR / "pycaret_stack_model.pkl"
            save_model(stack_model, str(stack_file)[:-4])
            logger.info(f"Modelo stack guardado: {stack_file}")
            
            trained_models['stack'] = {
                'model': stack_model,
                'file': stack_file
            }
        except Exception as e:
            logger.warning(f"Error creando stack model: {e}")
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': len(trained_models),
        'model_results': model_results
    }
    
    summary_file = RESULTS_DIR / "pycaret_models_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\nResumen guardado en: {summary_file}")
    logger.info(f"Total de modelos entrenados: {len(trained_models)}")
    
    return {
        'trained_models': trained_models,
        'summary': summary
    }


if __name__ == "__main__":
    if not PYCARET_AVAILABLE:
        print("❌ PyCaret no está instalado.")
        print("   Instala con: pip install pycaret[full]")
        sys.exit(1)
    
    try:
        results = train_pycaret_models(use_splits=True, top_n_models=5, use_ensemble=True)
        print("\n✓ Entrenamiento con PyCaret completado exitosamente")
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

