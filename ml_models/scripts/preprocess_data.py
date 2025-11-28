"""
Script ejecutable para preprocesar los datos crudos.

Este script aplica todo el pipeline de preprocesamiento y guarda
los datos procesados y el preprocesador entrenado.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ml_models.scripts.preprocessing import (
    load_and_validate_data,
    identify_column_types,
    handle_missing_values,
    handle_outliers,
    remove_highly_correlated_features,
    encode_categorical_variables,
    normalize_numerical_features,
    create_preprocessing_pipeline,
    save_preprocessor
)

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
DATA_RAW = project_root / "ml_models" / "data" / "raw" / "nhanes_stroke_raw.csv"
DATA_PROCESSED_DIR = project_root / "ml_models" / "data" / "processed"
DATA_PROCESSED = DATA_PROCESSED_DIR / "nhanes_stroke_processed.csv"
PREPROCESSOR_DIR = project_root / "ml_models" / "trained_models"
PREPROCESSOR_PATH = PREPROCESSOR_DIR / "preprocessor.pkl"

# Crear directorios si no existen
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSOR_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Función principal del script de preprocesamiento."""
    print("=" * 60)
    print("PREPROCESAMIENTO DE DATOS - ACV Risk Predictor")
    print("=" * 60)
    print()
    
    # 1. Cargar datos
    logger.info("Paso 1: Cargando datos crudos...")
    try:
        df = load_and_validate_data(DATA_RAW)
        print(f"✓ Datos cargados: {df.shape}")
    except Exception as e:
        logger.error(f"Error al cargar datos: {e}")
        return False
    
    # 2. Identificar tipos de columnas
    logger.info("Paso 2: Identificando tipos de columnas...")
    column_types = identify_column_types(df)
    print(f"✓ Variables numéricas continuas: {len(column_types['continuous_numeric'])}")
    print(f"✓ Variables categóricas numéricas: {len(column_types['categorical_numeric'])}")
    print(f"✓ Variables categóricas objeto: {len(column_types['categorical_object'])}")
    
    # 3. Manejar valores faltantes
    logger.info("Paso 3: Manejando valores faltantes...")
    df, imputation_info = handle_missing_values(
        df,
        numeric_strategy='mean',
        categorical_strategy='most_frequent',
        column_types=column_types
    )
    print(f"✓ Valores faltantes imputados")
    
    # 4. Manejar outliers
    logger.info("Paso 4: Manejando outliers...")
    df, outliers_info = handle_outliers(
        df,
        method='iqr',
        column_types=column_types
    )
    if outliers_info:
        print(f"✓ Outliers manejados en {len(outliers_info)} variables")
    else:
        print("✓ No se detectaron outliers significativos")
    
    # 5. Eliminar variables altamente correlacionadas
    logger.info("Paso 5: Eliminando variables altamente correlacionadas...")
    df, removed_cols = remove_highly_correlated_features(
        df,
        threshold=0.95,
        column_types=column_types
    )
    if removed_cols:
        print(f"✓ Eliminadas {len(removed_cols)} variables correlacionadas")
    else:
        print("✓ No se encontraron variables altamente correlacionadas")
    
    # Actualizar tipos de columnas después de eliminar columnas
    column_types = identify_column_types(df)
    
    # 6. Codificar variables categóricas
    logger.info("Paso 6: Codificando variables categóricas...")
    df, encoders_info = encode_categorical_variables(
        df,
        encoding_method='onehot',
        column_types=column_types
    )
    print(f"✓ Variables categóricas codificadas")
    
    # Actualizar tipos de columnas después de encoding
    column_types = identify_column_types(df)
    
    # 7. Normalizar variables numéricas
    logger.info("Paso 7: Normalizando variables numéricas...")
    df, scaler = normalize_numerical_features(
        df,
        method='standard',
        column_types=column_types
    )
    print(f"✓ Variables numéricas normalizadas")
    
    # 8. Crear y entrenar pipeline de preprocesamiento
    logger.info("Paso 8: Creando pipeline de preprocesamiento...")
    # Separar features y target
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    # Crear pipeline
    preprocessor = create_preprocessing_pipeline(
        column_types=identify_column_types(df.drop(columns=['stroke'])),
        numeric_imputation_strategy='mean',
        categorical_imputation_strategy='most_frequent',
        normalization_method='standard',
        encoding_method='onehot'
    )
    
    # Entrenar pipeline
    preprocessor.fit(X, y)
    print(f"✓ Pipeline de preprocesamiento entrenado")
    
    # 9. Guardar datos procesados
    logger.info("Paso 9: Guardando datos procesados...")
    df_processed = pd.concat([X, y], axis=1)
    df_processed.to_csv(DATA_PROCESSED, index=False)
    print(f"✓ Datos procesados guardados en: {DATA_PROCESSED}")
    print(f"  Shape final: {df_processed.shape}")
    
    # 10. Guardar preprocesador
    logger.info("Paso 10: Guardando preprocesador...")
    save_preprocessor(preprocessor, PREPROCESSOR_PATH)
    print(f"✓ Preprocesador guardado en: {PREPROCESSOR_PATH}")
    
    # Resumen final
    print()
    print("=" * 60)
    print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\nResumen:")
    print(f"  - Datos originales: {load_and_validate_data(DATA_RAW).shape}")
    print(f"  - Datos procesados: {df_processed.shape}")
    print(f"  - Variables eliminadas (correlación): {len(removed_cols)}")
    print(f"  - Preprocesador guardado: {PREPROCESSOR_PATH}")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

