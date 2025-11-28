"""
Módulo de preprocesamiento de datos para el pipeline de ML.

Este módulo contiene todas las funciones necesarias para preprocesar
los datos crudos antes del entrenamiento de modelos.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import pickle
import logging

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """Carga y valida la estructura básica de los datos.
    
    Args:
        data_path: Ruta al archivo CSV con datos crudos.
        
    Returns:
        DataFrame con los datos cargados.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo está vacío o tiene estructura incorrecta.
    """
    logger.info(f"Cargando datos desde: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"El archivo no existe: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise ValueError(f"Error al leer el archivo CSV: {str(e)}")
    
    if df.empty:
        raise ValueError("El DataFrame está vacío")
    
    # Verificar que existe la columna target
    if 'stroke' not in df.columns:
        raise ValueError("La columna 'stroke' (target) no existe en el dataset")
    
    logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    logger.info(f"Valores faltantes: {df.isnull().sum().sum()}")
    
    return df


def identify_column_types(df: pd.DataFrame, target_col: str = 'stroke') -> Dict[str, List[str]]:
    """Identifica los tipos de columnas en el dataset.
    
    Args:
        df: DataFrame con los datos.
        target_col: Nombre de la columna target.
        
    Returns:
        Diccionario con listas de columnas por tipo.
    """
    # Variables numéricas continuas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Variables categóricas (objeto)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Variables numéricas que son categóricas (pocos valores únicos)
    potential_categorical = []
    for col in numeric_cols:
        if df[col].nunique() < 20 and df[col].dtype in ['int64', 'int32', 'int16', 'int8']:
            potential_categorical.append(col)
    
    # Separar numéricas continuas de categóricas codificadas
    continuous_numeric = [col for col in numeric_cols if col not in potential_categorical]
    
    return {
        'continuous_numeric': continuous_numeric,
        'categorical_numeric': potential_categorical,
        'categorical_object': categorical_cols,
        'all_features': continuous_numeric + potential_categorical + categorical_cols
    }


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = 'mean',
    categorical_strategy: str = 'most_frequent',
    column_types: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Maneja valores faltantes en el dataset.
    
    Args:
        df: DataFrame con los datos.
        numeric_strategy: Estrategia para numéricas ('mean', 'median', 'constant').
        categorical_strategy: Estrategia para categóricas ('most_frequent', 'constant').
        column_types: Diccionario con tipos de columnas. Si es None, se infieren.
        
    Returns:
        Tupla con (DataFrame procesado, diccionario con valores imputados).
    """
    if column_types is None:
        column_types = identify_column_types(df)
    
    df_processed = df.copy()
    imputation_values = {}
    
    # Imputar variables numéricas continuas
    for col in column_types['continuous_numeric']:
        if df_processed[col].isnull().sum() > 0:
            if numeric_strategy == 'mean':
                value = df_processed[col].mean()
            elif numeric_strategy == 'median':
                value = df_processed[col].median()
            else:
                value = 0
            
            df_processed[col].fillna(value, inplace=True)
            imputation_values[col] = {'strategy': numeric_strategy, 'value': value}
            logger.info(f"Imputado {col}: {numeric_strategy} = {value:.4f}")
    
    # Imputar variables categóricas (numéricas codificadas)
    for col in column_types['categorical_numeric']:
        if df_processed[col].isnull().sum() > 0:
            value = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 0
            df_processed[col].fillna(value, inplace=True)
            imputation_values[col] = {'strategy': 'mode', 'value': value}
            logger.info(f"Imputado {col}: mode = {value}")
    
    # Imputar variables categóricas (objeto)
    for col in column_types['categorical_object']:
        if df_processed[col].isnull().sum() > 0:
            value = df_processed[col].mode()[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
            df_processed[col].fillna(value, inplace=True)
            imputation_values[col] = {'strategy': 'mode', 'value': value}
            logger.info(f"Imputado {col}: mode = {value}")
    
    return df_processed, imputation_values


def handle_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    columns: Optional[List[str]] = None,
    column_types: Optional[Dict[str, List[str]]] = None,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detecta y maneja outliers en variables numéricas.
    
    Args:
        df: DataFrame con los datos.
        method: Método de detección ('iqr', 'zscore', 'percentile').
        columns: Lista de columnas a procesar. Si None, usa todas las numéricas continuas.
        column_types: Diccionario con tipos de columnas.
        lower_percentile: Percentil inferior para método 'percentile'.
        upper_percentile: Percentil superior para método 'percentile'.
        
    Returns:
        Tupla con (DataFrame procesado, diccionario con información de outliers).
    """
    if column_types is None:
        column_types = identify_column_types(df)
    
    if columns is None:
        columns = column_types['continuous_numeric']
    
    df_processed = df.copy()
    outliers_info = {}
    
    for col in columns:
        if col not in df_processed.columns:
            continue
        
        original_count = len(df_processed)
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if method == 'iqr':
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df_processed[col].dropna()))
            lower_bound = df_processed[col].mean() - 3 * df_processed[col].std()
            upper_bound = df_processed[col].mean() + 3 * df_processed[col].std()
        elif method == 'percentile':
            lower_bound = df_processed[col].quantile(lower_percentile)
            upper_bound = df_processed[col].quantile(upper_percentile)
        else:
            continue
        
        # Capping (limitar valores extremos)
        outliers_count = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
        
        if outliers_count > 0:
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            
            outliers_info[col] = {
                'method': method,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_count': outliers_count,
                'outliers_pct': (outliers_count / original_count) * 100
            }
            logger.info(f"Outliers en {col}: {outliers_count} ({outliers_info[col]['outliers_pct']:.2f}%)")
    
    return df_processed, outliers_info


def remove_highly_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    target_col: str = 'stroke',
    column_types: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Elimina variables altamente correlacionadas.
    
    Args:
        df: DataFrame con los datos.
        threshold: Umbral de correlación para eliminar (default: 0.95).
        target_col: Nombre de la columna target.
        column_types: Diccionario con tipos de columnas.
        
    Returns:
        Tupla con (DataFrame sin variables correlacionadas, lista de columnas eliminadas).
    """
    if column_types is None:
        column_types = identify_column_types(df)
    
    # Solo considerar variables numéricas continuas
    numeric_cols = column_types['continuous_numeric']
    
    if len(numeric_cols) < 2:
        return df, []
    
    # Calcular matriz de correlación
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Encontrar pares altamente correlacionados
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Identificar columnas a eliminar
    to_remove = []
    for col in upper_triangle.columns:
        if col in to_remove:
            continue
        
        # Encontrar variables correlacionadas con esta
        correlated = upper_triangle.index[upper_triangle[col] > threshold].tolist()
        
        # Si hay correlaciones altas, eliminar todas excepto la primera
        # (mantener la que tiene más información o mejor nombre)
        if correlated:
            # Mantener la columna actual, eliminar las correlacionadas
            to_remove.extend(correlated)
    
    # Eliminar duplicados y asegurar que no eliminamos todas las columnas
    to_remove = list(set(to_remove))
    
    if to_remove:
        logger.info(f"Eliminando {len(to_remove)} variables altamente correlacionadas: {to_remove}")
        df_processed = df.drop(columns=to_remove)
    else:
        df_processed = df.copy()
    
    return df_processed, to_remove


def encode_categorical_variables(
    df: pd.DataFrame,
    encoding_method: str = 'onehot',
    column_types: Optional[Dict[str, List[str]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Codifica variables categóricas.
    
    Args:
        df: DataFrame con los datos.
        encoding_method: Método de encoding ('onehot', 'label').
        column_types: Diccionario con tipos de columnas.
        
    Returns:
        Tupla con (DataFrame codificado, diccionario con encoders).
    """
    if column_types is None:
        column_types = identify_column_types(df)
    
    df_processed = df.copy()
    encoders = {}
    
    # Variables categóricas numéricas (ya están codificadas, solo validar)
    for col in column_types['categorical_numeric']:
        # Estas ya están codificadas, no necesitan transformación
        pass
    
    # Variables categóricas objeto
    for col in column_types['categorical_object']:
        if encoding_method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
            encoders[col] = {'method': 'onehot', 'columns': dummies.columns.tolist()}
            logger.info(f"One-hot encoding para {col}: {len(dummies.columns)} columnas creadas")
        elif encoding_method == 'label':
            # Label encoding
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            encoders[col] = {'method': 'label', 'encoder': le}
            logger.info(f"Label encoding para {col}")
    
    return df_processed, encoders


def normalize_numerical_features(
    df: pd.DataFrame,
    method: str = 'standard',
    column_types: Optional[Dict[str, List[str]]] = None,
    scaler: Optional[Any] = None
) -> Tuple[pd.DataFrame, Any]:
    """Normaliza o estandariza variables numéricas.
    
    Args:
        df: DataFrame con los datos.
        method: Método de normalización ('standard', 'minmax').
        column_types: Diccionario con tipos de columnas.
        scaler: Scaler pre-entrenado. Si None, se crea uno nuevo.
        
    Returns:
        Tupla con (DataFrame normalizado, scaler entrenado).
    """
    if column_types is None:
        column_types = identify_column_types(df)
    
    numeric_cols = column_types['continuous_numeric']
    
    if len(numeric_cols) == 0:
        return df, None
    
    df_processed = df.copy()
    
    # Crear o usar scaler existente
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método de normalización no válido: {method}")
        
        # Entrenar scaler
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        logger.info(f"Scaler {method} entrenado en {len(numeric_cols)} columnas")
    else:
        # Usar scaler pre-entrenado
        df_processed[numeric_cols] = scaler.transform(df_processed[numeric_cols])
        logger.info(f"Scaler {method} aplicado (pre-entrenado)")
    
    return df_processed, scaler


def create_preprocessing_pipeline(
    column_types: Dict[str, List[str]],
    numeric_imputation_strategy: str = 'mean',
    categorical_imputation_strategy: str = 'most_frequent',
    normalization_method: str = 'standard',
    encoding_method: str = 'onehot'
) -> Pipeline:
    """Crea un pipeline completo de preprocesamiento usando sklearn.
    
    Args:
        column_types: Diccionario con tipos de columnas.
        numeric_imputation_strategy: Estrategia de imputación para numéricas.
        categorical_imputation_strategy: Estrategia de imputación para categóricas.
        normalization_method: Método de normalización.
        encoding_method: Método de encoding.
        
    Returns:
        Pipeline de sklearn configurado.
    """
    # Preparar transformadores
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_imputation_strategy)),
        ('scaler', StandardScaler() if normalization_method == 'standard' else MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_imputation_strategy, fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combinar transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, column_types['continuous_numeric']),
            ('cat_num', SimpleImputer(strategy='most_frequent'), column_types['categorical_numeric']),
            ('cat_obj', categorical_transformer, column_types['categorical_object'])
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def apply_preprocessing(
    df: pd.DataFrame,
    preprocessor: Any,
    target_col: str = 'stroke'
) -> Tuple[pd.DataFrame, pd.Series]:
    """Aplica el preprocesador a nuevos datos.
    
    Args:
        df: DataFrame con datos nuevos.
        preprocessor: Preprocesador entrenado.
        target_col: Nombre de la columna target.
        
    Returns:
        Tupla con (X preprocesado, y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_processed = preprocessor.transform(X)
    
    # Convertir a DataFrame si es posible
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
    else:
        X_processed = pd.DataFrame(X_processed, index=X.index)
    
    return X_processed, y


def save_preprocessor(preprocessor: Any, filepath: Path) -> None:
    """Guarda el preprocesador en un archivo.
    
    Args:
        preprocessor: Preprocesador a guardar.
        filepath: Ruta donde guardar el archivo.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    logger.info(f"Preprocesador guardado en: {filepath}")


def load_preprocessor(filepath: Path) -> Any:
    """Carga un preprocesador desde un archivo.
    
    Args:
        filepath: Ruta al archivo del preprocesador.
        
    Returns:
        Preprocesador cargado.
    """
    with open(filepath, 'rb') as f:
        preprocessor = pickle.load(f)
    
    logger.info(f"Preprocesador cargado desde: {filepath}")
    return preprocessor

