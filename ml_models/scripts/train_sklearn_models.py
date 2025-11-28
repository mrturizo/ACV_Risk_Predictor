"""
Script para entrenar múltiples modelos de sklearn con validación cruzada,
grid search, y manejo de desbalance.

Este script entrena varios modelos de clasificación, los evalúa con CV,
ajusta hiperparámetros y maneja el desbalance de clases.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Agregar raíz del proyecto al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, RandomizedSearchCV,
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# Importar módulos locales
from ml_models.scripts.preprocessing import load_preprocessor, apply_preprocessing
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
PREPROCESSOR_PATH = project_root / "ml_models" / "trained_models" / "preprocessor.pkl"
MODELS_DIR = project_root / "ml_models" / "trained_models"
RESULTS_DIR = project_root / "ml_models" / "trained_models" / "results"

# Crear directorios
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Definir modelos y espacios de búsqueda (solo los mejores 4 modelos, sin Neural Network)
MODEL_CONFIGS = {
    'logistic_regression': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'param_grid': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': [None, 'balanced']
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'param_grid': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
    # Eliminados: svm, knn, neural_network (para acelerar entrenamiento)
}

# Técnicas de balanceo
BALANCING_METHODS = {
    'none': None,
    'smote': SMOTE(random_state=RANDOM_STATE),
    'adasyn': ADASYN(random_state=RANDOM_STATE),
    'undersample': RandomUnderSampler(random_state=RANDOM_STATE),
    'smoteenn': SMOTEENN(random_state=RANDOM_STATE)
}


def train_model_with_cv(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: List[str] = None
) -> Dict[str, Any]:
    """Entrena un modelo con validación cruzada estratificada.
    
    Args:
        model: Modelo de sklearn a entrenar.
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        cv_folds: Número de folds para CV.
        scoring: Lista de métricas a calcular.
        
    Returns:
        Diccionario con resultados de CV.
    """
    if scoring is None:
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    results = {
        'cv_scores': {},
        'mean_scores': {},
        'std_scores': {}
    }
    
    for metric in scoring:
        test_key = f'test_{metric}'
        train_key = f'train_{metric}'
        
        if test_key in cv_results:
            results['cv_scores'][f'{metric}_test'] = cv_results[test_key].tolist()
            results['mean_scores'][f'{metric}_test'] = cv_results[test_key].mean()
            results['std_scores'][f'{metric}_test'] = cv_results[test_key].std()
        
        if train_key in cv_results:
            results['cv_scores'][f'{metric}_train'] = cv_results[train_key].tolist()
            results['mean_scores'][f'{metric}_train'] = cv_results[train_key].mean()
            results['std_scores'][f'{metric}_train'] = cv_results[train_key].std()
    
    return results


def apply_balancing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'none'
) -> Tuple[pd.DataFrame, pd.Series]:
    """Aplica técnicas de balanceo a los datos.
    
    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        method: Método de balanceo ('none', 'smote', 'adasyn', 'undersample', 'smoteenn').
        
    Returns:
        Tupla con (X_balanced, y_balanced).
    """
    if method == 'none' or method not in BALANCING_METHODS:
        return X_train, y_train
    
    balancer = BALANCING_METHODS[method]
    X_balanced, y_balanced = balancer.fit_resample(X_train, y_train)
    
    logger.info(f"Balanceo aplicado ({method}): {X_train.shape[0]} -> {X_balanced.shape[0]} muestras")
    logger.info(f"Distribución después de balanceo: {pd.Series(y_balanced).value_counts().to_dict()}")
    
    return pd.DataFrame(X_balanced, columns=X_train.columns), pd.Series(y_balanced)


def train_and_evaluate_model(
    model_name: str,
    model_config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    balancing_method: str = 'none',
    use_grid_search: bool = True
) -> Dict[str, Any]:
    """Entrena y evalúa un modelo completo.
    
    Args:
        model_name: Nombre del modelo.
        model_config: Configuración del modelo (modelo y param_grid).
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        X_val: Features de validación.
        y_val: Target de validación.
        X_test: Features de test.
        y_test: Target de test.
        balancing_method: Método de balanceo a aplicar.
        use_grid_search: Si usar GridSearchCV para ajustar hiperparámetros.
        
    Returns:
        Diccionario con resultados completos del modelo.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Entrenando modelo: {model_name.upper()}")
    logger.info(f"{'='*60}")
    
    # Aplicar balanceo
    X_train_balanced, y_train_balanced = apply_balancing(X_train, y_train, balancing_method)
    
    # Entrenar con CV
    logger.info("Realizando validación cruzada...")
    cv_results = train_model_with_cv(
        model_config['model'],
        X_train_balanced,
        y_train_balanced,
        cv_folds=5
    )
    
    # Ajuste de hiperparámetros
    best_model = model_config['model']
    best_params = None
    
    if use_grid_search:
        logger.info("Ajustando hiperparámetros con GridSearchCV...")
        # Usar RandomizedSearchCV para modelos con muchos parámetros (más rápido)
        if len(model_config['param_grid']) > 5:  # Reducir umbral para usar RandomizedSearch
            search = RandomizedSearchCV(
                model_config['model'],
                model_config['param_grid'],
                n_iter=15,  # Reducir de 20 a 15 iteraciones
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                scoring='roc_auc',
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=0  # Reducir verbosidad
            )
        else:
            search = GridSearchCV(
                model_config['model'],
                model_config['param_grid'],
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0  # Reducir verbosidad
            )
        
        search.fit(X_train_balanced, y_train_balanced)
        best_model = search.best_estimator_
        best_params = search.best_params_
        logger.info(f"Mejores parámetros: {best_params}")
        logger.info(f"Mejor score CV: {search.best_score_:.4f}")
    
    # Entrenar modelo final con todos los datos de entrenamiento
    logger.info("Entrenando modelo final...")
    best_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluar en conjunto de validación
    logger.info("Evaluando en conjunto de validación...")
    y_val_pred = best_model.predict(X_val)
    y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, zero_division=0),
        'recall': recall_score(y_val, y_val_pred, zero_division=0),
        'f1': f1_score(y_val, y_val_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_val_pred_proba) if len(np.unique(y_val)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_val, y_val_pred_proba) if len(np.unique(y_val)) > 1 else 0.0
    }
    
    # Evaluar en conjunto de test
    logger.info("Evaluando en conjunto de test...")
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_test_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'pr_auc': average_precision_score(y_test, y_test_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
    }
    
    # Compilar resultados
    results = {
        'model_name': model_name,
        'balancing_method': balancing_method,
        'best_params': best_params,
        'cv_results': cv_results,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'model': best_model
    }
    
    logger.info(f"\nMétricas en Test:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {test_metrics['pr_auc']:.4f}")
    
    return results


def save_model_results(model_results: Dict[str, Any], output_dir: Path) -> None:
    """Guarda el modelo y sus resultados.
    
    Args:
        model_results: Diccionario con resultados del modelo.
        output_dir: Directorio donde guardar.
    """
    model_name = model_results['model_name']
    balancing = model_results['balancing_method']
    
    # Guardar modelo
    model_file = output_dir / f"{model_name}_{balancing}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model_results['model'], f)
    logger.info(f"Modelo guardado: {model_file}")
    
    # Guardar resultados (sin el modelo)
    results_copy = model_results.copy()
    results_copy.pop('model', None)
    
    results_file = output_dir / f"{model_name}_{balancing}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_copy, f, indent=2, default=str)
    logger.info(f"Resultados guardados: {results_file}")


def main():
    """Función principal del script."""
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELOS CON SKLEARN")
    print("=" * 60)
    print()
    
    # Cargar datos
    logger.info("Cargando datos...")
    try:
        train_df, val_df, test_df = load_splits(SPLITS_DIR)
    except FileNotFoundError:
        logger.error(f"No se encontraron splits en {SPLITS_DIR}")
        logger.error("Ejecuta primero el script de división de datos")
        return False
    
    # Separar features y target
    X_train = train_df.drop(columns=['stroke'])
    y_train = train_df['stroke']
    X_val = val_df.drop(columns=['stroke'])
    y_val = val_df['stroke']
    X_test = test_df.drop(columns=['stroke'])
    y_test = test_df['stroke']
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Distribución de target en train: {y_train.value_counts().to_dict()}")
    
    # Entrenar todos los modelos
    all_results = []
    
    # Probar con y sin balanceo para algunos modelos clave
    # Reducir a solo 'smote' para acelerar (el balanceo generalmente mejora resultados)
    balancing_methods_to_test = ['smote']  # Solo SMOTE para acelerar
    
    for model_name, model_config in MODEL_CONFIGS.items():
        for balancing_method in balancing_methods_to_test:
            try:
                results = train_and_evaluate_model(
                    model_name=model_name,
                    model_config=model_config,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    balancing_method=balancing_method,
                    use_grid_search=True
                )
                
                all_results.append(results)
                save_model_results(results, MODELS_DIR)
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name} con {balancing_method}: {e}")
                continue
    
    # Guardar resumen de todos los modelos
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_trained': len(all_results),
        'results': [
            {
                'model_name': r['model_name'],
                'balancing_method': r['balancing_method'],
                'test_roc_auc': r['test_metrics']['roc_auc'],
                'test_pr_auc': r['test_metrics']['pr_auc'],
                'test_recall': r['test_metrics']['recall'],
                'test_f1': r['test_metrics']['f1']
            }
            for r in all_results
        ]
    }
    
    summary_file = RESULTS_DIR / "sklearn_models_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nResumen guardado en: {summary_file}")
    logger.info(f"\nTotal de modelos entrenados: {len(all_results)}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

