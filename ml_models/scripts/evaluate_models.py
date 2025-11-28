"""
Módulo de evaluación de modelos con métricas, visualizaciones y reportes comparativos.

Este módulo proporciona funciones para evaluar modelos de clasificación,
generar visualizaciones y crear reportes comparativos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Calcula todas las métricas relevantes para clasificación binaria.
    
    Args:
        y_true: Valores reales.
        y_pred: Predicciones (clases).
        y_pred_proba: Probabilidades predichas (opcional).
        
    Returns:
        Diccionario con todas las métricas calculadas.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': 0.0  # Se calculará después
    }
    
    # Calcular specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if tn + fp > 0:
        metrics['specificity'] = tn / (tn + fp)
    
    # Métricas que requieren probabilidades
    if y_pred_proba is not None and len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
    else:
        metrics['roc_auc'] = 0.0
        metrics['pr_auc'] = 0.0
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[Path] = None
) -> None:
    """Visualiza la matriz de confusión.
    
    Args:
        y_true: Valores reales.
        y_pred: Predicciones.
        model_name: Nombre del modelo.
        save_path: Ruta donde guardar la figura.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Stroke', 'Stroke'],
                yticklabels=['No Stroke', 'Stroke'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Matriz de confusión guardada en: {save_path}")
    
    plt.close()


def plot_roc_curves(
    models_results: List[Dict[str, Any]],
    save_path: Optional[Path] = None
) -> None:
    """Compara curvas ROC de múltiples modelos.
    
    Args:
        models_results: Lista de diccionarios con resultados de modelos.
                       Cada diccionario debe tener 'name', 'y_true', 'y_pred_proba'.
        save_path: Ruta donde guardar la figura.
    """
    plt.figure(figsize=(10, 8))
    
    for result in models_results:
        name = result['name']
        y_true = result['y_true']
        y_pred_proba = result['y_pred_proba']
        
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            auc = roc_auc_score(y_true, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad/Recall)')
    plt.title('Curvas ROC - Comparación de Modelos')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Curvas ROC guardadas en: {save_path}")
    
    plt.close()


def plot_precision_recall_curves(
    models_results: List[Dict[str, Any]],
    save_path: Optional[Path] = None
) -> None:
    """Compara curvas Precision-Recall de múltiples modelos.
    
    Args:
        models_results: Lista de diccionarios con resultados de modelos.
        save_path: Ruta donde guardar la figura.
    """
    plt.figure(figsize=(10, 8))
    
    for result in models_results:
        name = result['name']
        y_true = result['y_true']
        y_pred_proba = result['y_pred_proba']
        
        if len(np.unique(y_true)) > 1:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            pr_auc = average_precision_score(y_true, y_pred_proba)
            plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.3f})', linewidth=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall - Comparación de Modelos')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Curvas PR guardadas en: {save_path}")
    
    plt.close()


def compare_models(
    models_metrics: List[Dict[str, Any]],
    metric_priority: str = 'roc_auc'
) -> pd.DataFrame:
    """Crea una tabla comparativa de todos los modelos.
    
    Args:
        models_metrics: Lista de diccionarios con métricas de modelos.
        metric_priority: Métrica principal para ordenar.
        
    Returns:
        DataFrame con comparación de modelos ordenado por métrica prioritaria.
    """
    df = pd.DataFrame(models_metrics)
    
    # Ordenar por métrica prioritaria
    if metric_priority in df.columns:
        df = df.sort_values(metric_priority, ascending=False)
    
    return df


def generate_evaluation_report(
    models_results: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """Genera un reporte completo de evaluación en HTML.
    
    Args:
        models_results: Lista de diccionarios con resultados completos de modelos.
        output_dir: Directorio donde guardar el reporte.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear DataFrame comparativo
    comparison_data = []
    for result in models_results:
        comparison_data.append({
            'Modelo': result.get('name', 'Unknown'),
            'Balancing': result.get('balancing_method', 'none'),
            'Accuracy': result.get('test_metrics', {}).get('accuracy', 0),
            'Precision': result.get('test_metrics', {}).get('precision', 0),
            'Recall': result.get('test_metrics', {}).get('recall', 0),
            'F1-Score': result.get('test_metrics', {}).get('f1', 0),
            'ROC-AUC': result.get('test_metrics', {}).get('roc_auc', 0),
            'PR-AUC': result.get('test_metrics', {}).get('pr_auc', 0)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('ROC-AUC', ascending=False)
    
    # Guardar tabla comparativa
    comparison_file = output_dir / "models_comparison.csv"
    df_comparison.to_csv(comparison_file, index=False)
    logger.info(f"Tabla comparativa guardada en: {comparison_file}")
    
    # Generar visualizaciones
    roc_results = []
    pr_results = []
    
    for result in models_results:
        if 'y_true' in result and 'y_pred_proba' in result:
            roc_results.append({
                'name': result.get('name', 'Unknown'),
                'y_true': result['y_true'],
                'y_pred_proba': result['y_pred_proba']
            })
            pr_results.append({
                'name': result.get('name', 'Unknown'),
                'y_true': result['y_true'],
                'y_pred_proba': result['y_pred_proba']
            })
    
    if roc_results:
        plot_roc_curves(roc_results, output_dir / "roc_curves.png")
        plot_precision_recall_curves(pr_results, output_dir / "pr_curves.png")
    
    # Generar reporte HTML simple
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Evaluación de Modelos</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Evaluación de Modelos - ACV Risk Predictor</h1>
        <h2>Comparación de Modelos</h2>
        {df_comparison.to_html(index=False, classes='table')}
        <h2>Visualizaciones</h2>
        <p><img src="roc_curves.png" alt="ROC Curves" style="max-width: 100%;"></p>
        <p><img src="pr_curves.png" alt="PR Curves" style="max-width: 100%;"></p>
    </body>
    </html>
    """
    
    html_file = output_dir / "evaluation_report.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Reporte HTML guardado en: {html_file}")

