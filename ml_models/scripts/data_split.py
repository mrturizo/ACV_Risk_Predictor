"""
Módulo para división de datos en train/validation/test y creación de splits para CV.

Este módulo proporciona funciones para dividir los datos de manera estratificada
y crear splits para validación cruzada.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import pickle
import logging

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedShuffleSplit
)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target_col: str = 'stroke',
    train_size: float = 0.65,
    val_size: float = 0.20,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide los datos en conjuntos de entrenamiento, validación y test de forma estratificada.
    
    Args:
        df: DataFrame con los datos.
        target_col: Nombre de la columna target.
        train_size: Proporción para entrenamiento (default: 0.65).
        val_size: Proporción para validación (default: 0.20).
        test_size: Proporción para test (default: 0.15).
        random_state: Semilla aleatoria para reproducibilidad.
        
    Returns:
        Tupla con (train_df, val_df, test_df).
        
    Raises:
        ValueError: Si las proporciones no suman 1.0.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Las proporciones train_size + val_size + test_size deben sumar 1.0")
    
    logger.info(f"Dividiendo datos: Train={train_size:.1%}, Val={val_size:.1%}, Test={test_size:.1%}")
    
    # Separar features y target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Primera división: train+val vs test
    test_proportion = test_size / (train_size + val_size + test_size)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Segunda división: train vs val
    val_proportion = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_proportion,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Reconstruir DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Verificar distribuciones
    logger.info(f"Distribución de target en Train: {y_train.value_counts(normalize=True).to_dict()}")
    logger.info(f"Distribución de target en Val: {y_val.value_counts(normalize=True).to_dict()}")
    logger.info(f"Distribución de target en Test: {y_test.value_counts(normalize=True).to_dict()}")
    
    logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Guarda los splits en archivos CSV.
    
    Args:
        train_df: DataFrame de entrenamiento.
        val_df: DataFrame de validación.
        test_df: DataFrame de test.
        output_dir: Directorio donde guardar los archivos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Splits guardados en: {output_dir}")
    logger.info(f"  - Train: {train_path}")
    logger.info(f"  - Val: {val_path}")
    logger.info(f"  - Test: {test_path}")


def load_splits(splits_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los splits desde archivos CSV.
    
    Args:
        splits_dir: Directorio donde están los archivos.
        
    Returns:
        Tupla con (train_df, val_df, test_df).
    """
    train_path = splits_dir / "train.csv"
    val_path = splits_dir / "val.csv"
    test_path = splits_dir / "test.csv"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Splits cargados desde: {splits_dir}")
    logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    
    return train_df, val_df, test_df


def create_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Crea splits para validación cruzada estratificada.
    
    Args:
        X: Features.
        y: Target.
        n_splits: Número de folds (default: 5).
        random_state: Semilla aleatoria.
        shuffle: Si se debe mezclar los datos antes de dividir.
        
    Returns:
        Lista de tuplas (train_indices, test_indices) para cada fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    splits = []
    for train_idx, test_idx in skf.split(X, y):
        splits.append((train_idx, test_idx))
    
    logger.info(f"Creados {n_splits} folds para validación cruzada")
    
    return splits


def save_cv_splits(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    output_path: Path
) -> None:
    """Guarda los splits de CV en un archivo pickle.
    
    Args:
        splits: Lista de splits de CV.
        output_path: Ruta donde guardar el archivo.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(splits, f)
    
    logger.info(f"Splits de CV guardados en: {output_path}")


def load_cv_splits(input_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Carga los splits de CV desde un archivo pickle.
    
    Args:
        input_path: Ruta al archivo.
        
    Returns:
        Lista de splits de CV.
    """
    with open(input_path, 'rb') as f:
        splits = pickle.load(f)
    
    logger.info(f"Splits de CV cargados desde: {input_path}")
    return splits


def create_stratified_shuffle_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Crea múltiples splits estratificados con shuffle.
    
    Útil para validación cruzada con shuffle.
    
    Args:
        X: Features.
        y: Target.
        n_splits: Número de splits a crear.
        test_size: Proporción para test en cada split.
        random_state: Semilla aleatoria.
        
    Returns:
        Lista de tuplas (train_indices, test_indices).
    """
    sss = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state
    )
    
    splits = []
    for train_idx, test_idx in sss.split(X, y):
        splits.append((train_idx, test_idx))
    
    logger.info(f"Creados {n_splits} splits estratificados con shuffle")
    
    return splits

