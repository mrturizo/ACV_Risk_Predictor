"""
Script para seleccionar el mejor modelo basado en métricas clínicas.

Este script compara todos los modelos entrenados (sklearn y PyCaret)
y selecciona el mejor basado en métricas relevantes para datos clínicos.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import shutil
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Importar para evaluación
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rutas
MODELS_DIR = project_root / "ml_models" / "trained_models"
RESULTS_DIR = project_root / "ml_models" / "trained_models" / "results"
PRODUCTION_MODELS_DIR = project_root / "models"
SPLITS_DIR = project_root / "ml_models" / "data" / "splits"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_sklearn_results() -> List[Dict[str, Any]]:
    """Carga resultados de modelos sklearn.
    
    Returns:
        Lista de diccionarios con resultados de modelos sklearn.
    """
    results = []
    
    # Buscar archivos de resultados
    result_files = list(MODELS_DIR.glob("*_results.json"))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                result['source'] = 'sklearn'
                result['result_file'] = str(result_file)
                results.append(result)
        except Exception as e:
            logger.warning(f"Error cargando {result_file}: {e}")
    
    return results


def load_pycaret_results() -> List[Dict[str, Any]]:
    """Carga resultados de modelos PyCaret.
    
    Nota: PyCaret no guarda métricas detalladas en el summary, así que
    usamos los modelos sklearn que sí tienen métricas completas.
    Si necesitamos evaluar PyCaret, se puede hacer después.
    
    Returns:
        Lista de diccionarios con resultados de modelos PyCaret.
    """
    results = []
    
    summary_file = RESULTS_DIR / "pycaret_models_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                for model_result in summary.get('model_results', []):
                    model_result['source'] = 'pycaret'
                    # PyCaret no guarda métricas en el summary, así que las dejamos vacías
                    # Se pueden evaluar después si es necesario
                    model_result['metrics'] = {}
                    results.append(model_result)
        except Exception as e:
            logger.warning(f"Error cargando {summary_file}: {e}")
    
    return results


def calculate_clinical_score(metrics: Dict[str, float]) -> float:
    """Calcula un score ponderado basado en métricas clínicas.
    
    Para datos clínicos, priorizamos:
    - Recall/Sensitivity (no perder casos de riesgo real)
    - Specificity (evitar falsos positivos)
    - F1-Score (balance)
    - ROC-AUC (capacidad discriminativa)
    - PR-AUC (importante con desbalance)
    
    Args:
        metrics: Diccionario con métricas del modelo.
        
    Returns:
        Score ponderado (mayor es mejor).
    """
    weights = {
        'recall': 0.30,      # Muy importante: no perder casos reales
        'specificity': 0.20,  # Importante: evitar falsos positivos
        'f1': 0.15,          # Balance entre precision y recall
        'roc_auc': 0.20,     # Capacidad discriminativa general
        'pr_auc': 0.15       # Especialmente importante con desbalance
    }
    
    score = 0.0
    for metric, weight in weights.items():
        value = metrics.get(metric, 0.0)
        score += value * weight
    
    return score


def select_best_model(
    sklearn_results: List[Dict[str, Any]],
    pycaret_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Selecciona el mejor modelo basado en métricas clínicas.
    
    Args:
        sklearn_results: Lista de resultados de modelos sklearn.
        pycaret_results: Lista de resultados de modelos PyCaret.
        
    Returns:
        Diccionario con información del mejor modelo seleccionado.
    """
    all_models = []
    
    # Procesar modelos sklearn
    for result in sklearn_results:
        test_metrics = result.get('test_metrics', {})
        if test_metrics:
            clinical_score = calculate_clinical_score(test_metrics)
            all_models.append({
                'name': f"{result.get('model_name', 'unknown')}_{result.get('balancing_method', 'none')}",
                'source': 'sklearn',
                'model_file': result.get('result_file', '').replace('_results.json', '.pkl'),
                'metrics': test_metrics,
                'clinical_score': clinical_score,
                'full_result': result
            })
    
    # Procesar modelos PyCaret
    # Nota: PyCaret no tiene métricas guardadas, así que por ahora solo incluimos sklearn
    # Si queremos incluir PyCaret, necesitaríamos evaluarlos primero
    for result in pycaret_results:
        model_file = result.get('model_file', '')
        if model_file:
            # Por ahora, PyCaret no tiene métricas, así que no los incluimos en la comparación
            # Se pueden evaluar después si es necesario
            logger.info(f"Modelo PyCaret encontrado pero sin métricas: {result.get('model_name', 'unknown')}")
            # all_models.append({
            #     'name': result.get('model_name', 'unknown'),
            #     'source': 'pycaret',
            #     'model_file': model_file,
            #     'metrics': {},
            #     'clinical_score': 0.0,
            #     'full_result': result
            # })
    
    if not all_models:
        raise ValueError("No se encontraron modelos para comparar")
    
    # Ordenar por clinical_score
    all_models.sort(key=lambda x: x['clinical_score'], reverse=True)
    
    best_model = all_models[0]
    
    logger.info(f"\n{'='*60}")
    logger.info("MEJOR MODELO SELECCIONADO")
    logger.info(f"{'='*60}")
    logger.info(f"Modelo: {best_model['name']}")
    logger.info(f"Fuente: {best_model['source']}")
    logger.info(f"Clinical Score: {best_model['clinical_score']:.4f}")
    
    if best_model['metrics']:
        logger.info(f"\nMétricas:")
        for metric, value in best_model['metrics'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
    
    return best_model


def deploy_to_production(best_model: Dict[str, Any]) -> Path:
    """Copia el mejor modelo a la carpeta de producción.
    
    Args:
        best_model: Diccionario con información del mejor modelo.
        
    Returns:
        Ruta al modelo en producción.
    """
    source_file = Path(best_model['model_file'])
    
    if not source_file.exists():
        raise FileNotFoundError(f"Archivo del modelo no existe: {source_file}")
    
    # Nombre del archivo en producción
    production_file = PRODUCTION_MODELS_DIR / f"best_stroke_model.pkl"
    
    # Copiar modelo
    shutil.copy2(source_file, production_file)
    logger.info(f"Modelo copiado a producción: {production_file}")
    
    # Guardar metadata del modelo
    metadata = {
        'model_name': best_model['name'],
        'source': best_model['source'],
        'clinical_score': best_model['clinical_score'],
        'metrics': best_model.get('metrics', {}),
        'deployed_at': datetime.now().isoformat(),
        'model_file': str(production_file)
    }
    
    metadata_file = PRODUCTION_MODELS_DIR / "model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Metadata guardada en: {metadata_file}")
    
    return production_file


def main():
    """Función principal del script."""
    print("=" * 60)
    print("SELECCIÓN DEL MEJOR MODELO")
    print("=" * 60)
    print()
    
    # Cargar resultados
    print("Cargando resultados de modelos sklearn...")
    logger.info("Cargando resultados de modelos sklearn...")
    sklearn_results = load_sklearn_results()
    print(f"  {len(sklearn_results)} modelos sklearn encontrados")
    logger.info(f"  {len(sklearn_results)} modelos sklearn encontrados")
    
    print("Cargando resultados de modelos PyCaret...")
    logger.info("Cargando resultados de modelos PyCaret...")
    pycaret_results = load_pycaret_results()
    print(f"  {len(pycaret_results)} modelos PyCaret encontrados")
    logger.info(f"  {len(pycaret_results)} modelos PyCaret encontrados")
    
    if not sklearn_results and not pycaret_results:
        logger.error("No se encontraron modelos entrenados.")
        logger.error("Ejecuta primero train_sklearn_models.py y train_pycaret_models.py")
        return False
    
    # Seleccionar mejor modelo
    try:
        best_model = select_best_model(sklearn_results, pycaret_results)
    except Exception as e:
        logger.error(f"Error seleccionando mejor modelo: {e}")
        return False
    
    # Desplegar a producción
    try:
        production_file = deploy_to_production(best_model)
        logger.info(f"\n✓ Mejor modelo desplegado exitosamente")
        logger.info(f"  Ubicación: {production_file}")
    except Exception as e:
        logger.error(f"Error desplegando modelo: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

