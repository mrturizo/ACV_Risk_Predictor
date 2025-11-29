"""Preprocesamiento para el modelo de riesgo de ACV.

Este módulo implementa el pipeline de preprocesamiento acordado con el
equipo de Data Science, equivalente a:

- Imputación numérica por mediana
- Normalización tipo z-score (StandardScaler)
- Reducción de dimensionalidad con PCA lineal

El mismo preprocesamiento se aplicará tanto en entrenamiento como en
producción para garantizar coherencia.
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config_features import MODEL_INPUT_COLUMNS


# Parámetros del preprocesamiento acordados con DS
N_PCA_COMPONENTS: int = 25
RANDOM_STATE: int = 123


def build_preprocessor() -> Pipeline:
    """Crea el pipeline de preprocesamiento (imputación + zscore + PCA).

    Returns:
        Pipeline de sklearn sin ajustar.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=N_PCA_COMPONENTS,
                    random_state=RANDOM_STATE,
                    svd_solver="auto",
                    whiten=False,
                ),
            ),
        ]
    )


def _prepare_input_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas y tipos antes de ajustar/transformar.

    - Reordena columnas según MODEL_INPUT_COLUMNS.
    - Convierte todo a numérico (coercion de errores a NaN).
    """
    X = X.copy()
    # Asegurar todas las columnas esperadas
    for col in MODEL_INPUT_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan

    # Reordenar y convertir a numérico
    X = X[MODEL_INPUT_COLUMNS]
    X = X.apply(pd.to_numeric, errors="coerce")
    return X


def fit_preprocessor(X: pd.DataFrame, save_path: Optional[Path] = None) -> Pipeline:
    """Ajusta el preprocesador sobre datos crudos.

    Debe ejecutarse en fase de entrenamiento usando el mismo dataset que
    utiliza DS (por ejemplo, ``nhanes_stroke_clean.csv`` sin la columna
    objetivo ``stroke``).

    Args:
        X: DataFrame con las columnas de entrada sin el target.
        save_path: Ruta opcional para guardar el preprocesador entrenado
            como archivo ``.pkl``.

    Returns:
        Pipeline de sklearn ajustado.
    """
    X_prepared = _prepare_input_frame(X)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_prepared)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, save_path)

    return preprocessor


def load_preprocessor(path: Path) -> Pipeline:
    """Carga un preprocesador previamente entrenado.

    Args:
        path: Ruta al archivo ``.pkl`` del preprocesador.

    Returns:
        Pipeline de sklearn cargado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el preprocesador en {path}")
    preprocessor: Pipeline = joblib.load(path)
    return preprocessor


def transform_inputs(X: pd.DataFrame, preprocessor: Pipeline) -> np.ndarray:
    """Aplica el preprocesador a un DataFrame crudo y devuelve features PCA.

    Args:
        X: DataFrame con las columnas de entrada sin el target.
        preprocessor: Pipeline entrenado devuelto por :func:`fit_preprocessor`.

    Returns:
        Matriz ``numpy.ndarray`` con las características transformadas.
    """
    X_prepared = _prepare_input_frame(X)
    transformed = preprocessor.transform(X_prepared)
    if isinstance(transformed, pd.DataFrame):
        return transformed.to_numpy()
    return transformed


