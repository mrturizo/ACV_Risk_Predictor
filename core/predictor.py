"""Módulo para carga de modelos y predicción de riesgo de ACV."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging

# Configurar logging primero
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from pycaret.classification import load_model, predict_model
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logger.warning("PyCaret no está instalado. Las funciones de predicción no estarán disponibles.")

from core import MODELS_DIR
from core.config_features import MODEL_INPUT_COLUMNS


class StrokePredictor:
    """Clase para manejar la carga del modelo y realizar predicciones.
    
    Esta clase encapsula la lógica de carga de modelos PyCaret y predicción
    de riesgo de ACV basada en datos clínicos, demográficos y biomédicos.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Inicializa el predictor con un modelo.
        
        Args:
            model_path: Ruta al archivo .pkl del modelo. Si es None, busca
                       en la carpeta models/ el primer archivo .pkl encontrado.
                       
        Raises:
            FileNotFoundError: Si no se encuentra ningún modelo.
            ValueError: Si el modelo no es válido.
        """
        self.model = None
        self.model_path = model_path
        self.required_columns: List[str] = []
        self.is_pycaret_model = False  # Nuevo atributo para identificar el tipo de modelo
        self.preprocessor = None  # Nuevo atributo para el preprocesador
        self.preprocessor_path = None  # Ruta del preprocesador cargado
        self.normalization_scaler = None  # StandardScaler extraído del preprocesador
        self.normalization_params = None  # Parámetros de normalización para modelos sklearn
        self._load_model()
    
    def _load_model(self) -> None:
        """Carga el modelo PyCaret o sklearn desde el archivo especificado.
        
        IMPORTANTE: Si model_path es None, SOLO busca 'lr_pca25_cw.pkl'.
        No busca otros modelos como fallback.
        """
        if self.model_path is None:
            # OBLIGATORIO: Solo usar lr_pca25_cw.pkl - NO buscar otros modelos
            required_model = MODELS_DIR / "lr_pca25_cw.pkl"
            if required_model.exists():
                self.model_path = required_model
                logger.info(f"✅ Modelo requerido seleccionado automáticamente: {self.model_path}")
            else:
                error_msg = (
                    f"ERROR CRÍTICO: El modelo requerido 'lr_pca25_cw.pkl' no se encuentra en {MODELS_DIR}. "
                    f"Este modelo es OBLIGATORIO y no se pueden usar otros modelos como alternativa."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"El archivo del modelo no existe: {self.model_path}")
        
        try:
            logger.info(f"Cargando modelo desde: {self.model_path}")
            
            # Cargar modelo desde disco.
            # NOTA: Muchos modelos finales (como lr_pca25_cw.pkl) fueron serializados
            # con joblib. Si se cargan con pickle directamente, se obtiene un
            # numpy.ndarray en lugar del Pipeline, lo que provoca errores como
            # `'numpy.ndarray' object has no attribute 'predict'`.
            try:
                import joblib
                self.model = joblib.load(self.model_path)
                logger.info("Modelo cargado con joblib.load()")
            except Exception as joblib_err:
                logger.warning(f"Fallo joblib.load({self.model_path}): {joblib_err}. Probando con pickle.load()...")
                import pickle
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Modelo cargado con pickle.load()")

            logger.info(f"Modelo cargado: tipo={type(self.model)}")

            # Detectar si es un Pipeline completo de PyCaret (incluye imputación, balanceo,
            # normalización y PCA). En ese caso, NO debemos aplicar un preprocesador externo.
            model_module = type(self.model).__module__
            if model_module.startswith("pycaret.internal.pipeline"):
                self.is_pycaret_model = True
                self.preprocessor = None
                self.preprocessor_path = None
                logger.info(
                    "Modelo detectado como Pipeline completo de PyCaret; "
                    "se usará su preprocesamiento interno (imputación + zscore + PCA)."
                )
                # No cargar preprocesador externo ni reparar scaler.
                self._extract_required_columns()
                logger.info("Modelo cargado exitosamente")
                return

            # Si no es un Pipeline de PyCaret, lo tratamos como modelo sklearn puro.
            self.is_pycaret_model = False
            logger.info("Modelo tratado como estimador sklearn puro")

            # Cargar preprocesador completo heredado (si existe)
            # IMPORTANTE: Los modelos sklearn fueron entrenados con datos que pasaron por el preprocesador completo,
            # pero los splits tienen nombres de columnas ORIGINALES (sin prefijos). El preprocesador genera
            # columnas con prefijos (num__, cat_num__), por lo que necesitamos mapear de vuelta a nombres originales.
            logger.info("Buscando preprocesador completo...")
            # Priorizar preprocesador reparado
            preprocessor_paths = [
                self.model_path.parent / "preprocessor_fixed.pkl",  # Primero intentar el reparado
                self.model_path.parent / "preprocessor.pkl",
                Path(__file__).parent.parent / "ml_models" / "trained_models" / "preprocessor.pkl"
            ]
            
            logger.info(f"Rutas a verificar: {[str(p) for p in preprocessor_paths]}")
            
            for preprocessor_path in preprocessor_paths:
                logger.info(f"Verificando: {preprocessor_path} (existe: {preprocessor_path.exists()})")
                if preprocessor_path.exists():
                    try:
                        import pickle
                        with open(preprocessor_path, 'rb') as f:
                            self.preprocessor = pickle.load(f)
                        self.preprocessor_path = preprocessor_path  # Guardar la ruta
                        logger.info(f"✅ Preprocesador cargado desde: {preprocessor_path}")
                        logger.info(f"   Tipo: {type(self.preprocessor).__name__}")
                        
                        # Verificar que el StandardScaler esté entrenado
                        scaler_ok = False
                        for name, transformer, columns in self.preprocessor.transformers:
                            if name == 'num' and hasattr(transformer, 'steps'):
                                for step_name, step_transformer in transformer.steps:
                                    if step_name == 'scaler':
                                        from sklearn.preprocessing import StandardScaler
                                        if isinstance(step_transformer, StandardScaler):
                                            has_mean = hasattr(step_transformer, 'mean_') and step_transformer.mean_ is not None
                                            scaler_ok = has_mean
                                            if scaler_ok:
                                                logger.info(f"   ✅ StandardScaler está entrenado (mean_ shape: {step_transformer.mean_.shape})")
                                            else:
                                                logger.warning(f"   ⚠️ StandardScaler NO está entrenado en {preprocessor_path}")
                                            break
                        
                        if scaler_ok or 'fixed' in str(preprocessor_path):
                            # Si el scaler está OK o es el preprocesador reparado, usarlo
                            break
                        else:
                            # Si el scaler no está OK, continuar buscando
                            logger.warning(f"   ⚠️ Preprocesador en {preprocessor_path} no tiene scaler entrenado, continuando búsqueda...")
                            self.preprocessor = None
                            continue
                    except Exception as e:
                        logger.error(f"❌ Error al cargar preprocesador desde {preprocessor_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
            
            if self.preprocessor is not None:
                logger.info("Preprocesador cargado exitosamente")
                
                # Verificar y reparar el StandardScaler si no está entrenado (solo si no es el fixed)
                # Usar la ruta guardada en lugar de str() para evitar errores de compatibilidad de sklearn
                if self.preprocessor_path and 'fixed' not in str(self.preprocessor_path):
                    self._fix_preprocessor_scaler()
            else:
                logger.warning("⚠️ No se encontró el preprocesador. Las predicciones pueden fallar.")
            
            logger.info("Modelo cargado exitosamente")
            
            # Intentar obtener las columnas requeridas del modelo
            self._extract_required_columns()
            
        except Exception as e:
            raise ValueError(f"Error al cargar el modelo: {str(e)}")
    
    def _extract_required_columns(self) -> None:
        """Extrae las columnas requeridas del modelo cargado."""
        try:
            # PyCaret guarda el pipeline de preprocesamiento
            # Intentamos obtener las columnas desde el pipeline
            if hasattr(self.model, 'steps'):
                # Si es un pipeline, intentamos obtener las columnas
                # Esto puede variar según la versión de PyCaret
                pass
            
            # Por ahora, si no podemos extraerlas automáticamente,
            # se pueden definir manualmente o inferir de los datos de prueba
            # Esto se completará cuando tengamos el modelo real
            logger.warning(
                "No se pudieron extraer automáticamente las columnas requeridas. "
                "Se inferirán de los datos de entrada."
            )
            
        except Exception as e:
            logger.warning(f"No se pudieron extraer las columnas del modelo: {e}")
    
    def _fix_preprocessor_scaler(self) -> None:
        """Repara el StandardScaler del preprocesador si no está entrenado.
        
        Si el StandardScaler no tiene mean_ (no está entrenado), lo reentrena
        desde los datos de entrenamiento.
        """
        if self.preprocessor is None:
            return
        
        try:
            if hasattr(self.preprocessor, 'transformers'):
                for name, transformer, columns in self.preprocessor.transformers:
                    if name == 'num' and hasattr(transformer, 'steps'):
                        for step_name, step_transformer in transformer.steps:
                            if step_name == 'scaler':
                                from sklearn.preprocessing import StandardScaler
                                if isinstance(step_transformer, StandardScaler):
                                    has_mean = hasattr(step_transformer, 'mean_') and step_transformer.mean_ is not None
                                    
                                    if not has_mean:
                                        logger.warning("⚠️ StandardScaler no está entrenado. Reentrenando desde datos de entrenamiento...")
                                        
                                        # Cargar datos de entrenamiento
                                        train_path = Path(__file__).parent.parent / "ml_models" / "data" / "splits" / "train.csv"
                                        if train_path.exists():
                                            train_df = pd.read_csv(train_path)
                                            X_train = train_df.drop(columns=['stroke'])
                                            
                                            # Filtrar solo las columnas numéricas que el scaler debería procesar
                                            if isinstance(columns, list):
                                                available_cols = [c for c in columns if c in X_train.columns]
                                                if available_cols:
                                                    X_train_numeric = X_train[available_cols]
                                                    
                                                    # Reentrenar el scaler
                                                    new_scaler = StandardScaler()
                                                    new_scaler.fit(X_train_numeric)
                                                    
                                                    # Recrear el pipeline completo con el nuevo scaler
                                                    from sklearn.pipeline import Pipeline
                                                    from sklearn.impute import SimpleImputer
                                                    
                                                    # Obtener el imputer del pipeline original y reentrenarlo
                                                    imputer = None
                                                    for s_name, s_transformer in transformer.steps:
                                                        if s_name == 'imputer':
                                                            imputer = s_transformer
                                                            break
                                                    
                                                    # Si no hay imputer, crear uno nuevo
                                                    if imputer is None:
                                                        imputer = SimpleImputer(strategy='mean')
                                                    # Reentrenar el imputer también
                                                    imputer.fit(X_train_numeric)
                                                    
                                                    # Crear nuevo pipeline con imputer y scaler reentrenados
                                                    new_pipeline = Pipeline([
                                                        ('imputer', imputer),
                                                        ('scaler', new_scaler)
                                                    ])
                                                    
                                                    # Reentrenar el pipeline completo
                                                    new_pipeline.fit(X_train_numeric)
                                                    
                                                    # Reemplazar el transformer completo en el ColumnTransformer
                                                    # Necesitamos recrear la lista de transformers
                                                    new_transformers = []
                                                    for t_name, t_transformer, t_columns in self.preprocessor.transformers:
                                                        if t_name == name:
                                                            new_transformers.append((t_name, new_pipeline, t_columns))
                                                        else:
                                                            new_transformers.append((t_name, t_transformer, t_columns))
                                                    
                                                    # Recrear el ColumnTransformer
                                                    from sklearn.compose import ColumnTransformer
                                                    self.preprocessor = ColumnTransformer(
                                                        transformers=new_transformers,
                                                        remainder=self.preprocessor.remainder
                                                    )
                                                    
                                                    # Reentrenar el ColumnTransformer completo para que use los nuevos transformers
                                                    # Necesitamos todas las columnas de entrada
                                                    all_input_cols = []
                                                    for t_name, t_transformer, t_columns in new_transformers:
                                                        if isinstance(t_columns, list):
                                                            all_input_cols.extend(t_columns)
                                                    all_input_cols = list(set(all_input_cols))
                                                    
                                                    # Asegurar que tenemos todas las columnas en X_train
                                                    missing_cols = set(all_input_cols) - set(X_train.columns)
                                                    for col in missing_cols:
                                                        X_train[col] = 0.0
                                                    
                                                    X_train_all = X_train[all_input_cols]
                                                    y_train = train_df['stroke']
                                                    
                                                    # Reentrenar el ColumnTransformer completo
                                                    self.preprocessor.fit(X_train_all, y_train)
                                                    
                                                    logger.info(f"✅ StandardScaler reentrenado y ColumnTransformer reconstruido con {len(available_cols)} columnas numéricas")
                                                    break
                                        else:
                                            logger.error(f"❌ No se encontraron datos de entrenamiento en: {train_path}")
                                    else:
                                        logger.info("✅ StandardScaler está entrenado correctamente")
                                        break
        except Exception as e:
            logger.error(f"❌ Error al reparar el StandardScaler: {e}")
            import traceback
            traceback.print_exc()
    
    def _map_preprocessed_columns_to_original(self, data_processed: pd.DataFrame, data_original: pd.DataFrame = None) -> pd.DataFrame:
        """Mapea las columnas procesadas (con prefijos) de vuelta a nombres originales.
        
        El preprocesador genera columnas con prefijos (num__, cat_num__), pero los modelos
        sklearn fueron entrenados con nombres de columnas originales. Este método reconstruye
        un DataFrame con nombres originales basándose en el mapeo del ColumnTransformer.
        
        Args:
            data_processed: DataFrame con columnas procesadas (con prefijos).
            
        Returns:
            DataFrame con nombres de columnas originales.
        """
        if self.preprocessor is None:
            logger.warning("⚠️ No hay preprocesador disponible para mapear columnas")
            return data_processed
        
        try:
            # Obtener los nombres de columnas transformadas del preprocesador
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                transformed_cols = list(self.preprocessor.get_feature_names_out())
            else:
                logger.warning("⚠️ El preprocesador no tiene get_feature_names_out()")
                return data_processed
            
            # Crear mapeo de columnas transformadas a originales
            # El formato es: 'num__column_name' -> 'column_name'
            # o 'cat_num__column_name' -> 'column_name'
            col_mapping = {}
            
            # Iterar sobre los transformers del preprocesador
            if hasattr(self.preprocessor, 'transformers'):
                for name, transformer, columns in self.preprocessor.transformers:
                    if isinstance(columns, list):
                        # Para cada columna original, encontrar su columna transformada
                        for orig_col in columns:
                            # Buscar la columna transformada con el prefijo correspondiente
                            prefix = f"{name}__"
                            transformed_col = f"{prefix}{orig_col}"
                            
                            if transformed_col in transformed_cols:
                                col_mapping[transformed_col] = orig_col
                            else:
                                # Algunas columnas pueden tener nombres ligeramente diferentes
                                # Buscar por coincidencia parcial
                                for tc in transformed_cols:
                                    if tc.startswith(prefix) and orig_col in tc:
                                        col_mapping[tc] = orig_col
                                        break
            
            # Crear nuevo DataFrame con nombres originales
            data_original_names = pd.DataFrame(index=data_processed.index)
            
            # Identificar qué columnas son num__ (normalizadas) y cuáles son cat_num__ (no normalizadas)
            num_cols_processed = [c for c in transformed_cols if c.startswith('num__')]
            cat_num_cols_processed = [c for c in transformed_cols if c.startswith('cat_num__')]
            
            for transformed_col in transformed_cols:
                if transformed_col in data_processed.columns:
                    if transformed_col in col_mapping:
                        # Mapear a nombre original
                        orig_col = col_mapping[transformed_col]
                        
                        # IMPORTANTE CRÍTICO: 
                        # El modelo fue entrenado con TODAS las columnas normalizadas (incluyendo categóricas numéricas)
                        # Por lo tanto, TODAS las columnas deben usar valores transformados (normalizados)
                        # - Columnas num__ (numéricas continuas): usar valores transformados (normalizados)
                        # - Columnas cat_num__ (categóricas numéricas): TAMBIÉN usar valores transformados (normalizados)
                        #   NOTA: Aunque cat_num__ solo pasa por SimpleImputer, el modelo espera valores normalizados
                        #   porque los datos de entrenamiento (train.csv) tienen TODAS las columnas normalizadas
                        if transformed_col.startswith('num__'):
                            # Usar valor transformado (normalizado)
                            data_original_names[orig_col] = data_processed[transformed_col]
                        elif transformed_col.startswith('cat_num__'):
                            # CRÍTICO: Usar valor transformado (normalizado) porque el modelo fue entrenado así
                            # Aunque cat_num__ solo pasa por SimpleImputer, el modelo espera valores normalizados
                            data_original_names[orig_col] = data_processed[transformed_col]
                        else:
                            # Otros tipos, usar valor transformado
                            data_original_names[orig_col] = data_processed[transformed_col]
                    else:
                        # Si no hay mapeo, intentar extraer el nombre original del prefijo
                        # Formato: 'num__column_name' -> 'column_name'
                        if '__' in transformed_col:
                            orig_col = transformed_col.split('__', 1)[1]
                            # CRÍTICO: Todas las columnas deben usar valores transformados (normalizados)
                            # porque el modelo fue entrenado con todas las columnas normalizadas
                            data_original_names[orig_col] = data_processed[transformed_col]
                        else:
                            # Sin prefijo, usar el nombre tal cual
                            data_original_names[transformed_col] = data_processed[transformed_col]
            
            logger.info(f"✅ Columnas mapeadas: {len(data_processed.columns)} -> {len(data_original_names.columns)}")
            logger.info(f"   Primeras columnas mapeadas: {list(data_original_names.columns)[:5]}")
            
            return data_original_names
            
        except Exception as e:
            logger.error(f"❌ Error al mapear columnas: {e}")
            import traceback
            traceback.print_exc()
            return data_processed
    
    def _validate_processing(self, processing_status: Dict[str, Any]) -> tuple[bool, Dict[str, Any]]:
        """Valida que el procesamiento se haya completado correctamente.
        
        Args:
            processing_status: Diccionario con el estado del procesamiento.
            
        Returns:
            Tupla con (es_válido, información_del_estado)
        """
        status = processing_status.copy()

        # Si el modelo es un Pipeline completo de PyCaret (como lr_pca25_cw.pkl),
        # asumimos que el propio pipeline se encarga de TODA la imputación,
        # normalización (z-score) y PCA internamente. En ese caso no exigimos
        # que exista un preprocesador externo.
        if self.is_pycaret_model:
            status.setdefault("errors", [])
            status.setdefault("warnings", [])
            status["is_valid"] = True
            return True, status

        errors = status.get("errors", [])
        warnings = status.get("warnings", [])

        # Verificar si el preprocesador externo se aplicó correctamente
        if status.get("preprocessed", False) and status.get("normalized", False):
            is_valid = True
        elif self.preprocessor is not None:
            # Hay preprocesador pero no se aplicó correctamente
            is_valid = False
            errors.append("El preprocesador está disponible pero no se aplicó correctamente.")
        else:
            # No hay preprocesador disponible para un modelo sklearn puro
            is_valid = False
            errors.append(
                "CRÍTICO: No hay preprocesador disponible. "
                "El modelo sklearn requiere datos procesados y normalizados."
            )

        status["errors"] = errors
        status["warnings"] = warnings
        status["is_valid"] = is_valid

        return is_valid, status
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Realiza la predicción de riesgo de ACV.
        
        Args:
            data: DataFrame con los datos del paciente. Debe contener las
                  columnas esperadas por el modelo.
                  
        Returns:
            Diccionario con:
            - 'prediction': 'STROKE RISK' o 'NOT STROKE RISK'
            - 'probability': Probabilidad de riesgo (0-1)
            - 'details': Información adicional sobre la predicción
            - 'data_before_processing': DataFrame con datos originales antes de procesamiento
            - 'data_after_processing': DataFrame con datos después de procesamiento
            - 'processing_status': Diccionario con estado del procesamiento
            - 'processing_successful': Boolean indicando si el procesamiento fue exitoso
            
        Raises:
            ValueError: Si los datos no tienen el formato correcto o el procesamiento falló.
        """
        if self.model is None:
            raise ValueError("El modelo no está cargado. Llama a _load_model() primero.")
        
        if data.empty:
            raise ValueError("El DataFrame de entrada está vacío.")
        
        try:
            # Guardar datos originales ANTES de cualquier procesamiento
            data_before_processing = data.copy()
            
            # Inicializar estado de procesamiento
            processing_status = {
                'normalized': False,
                'preprocessed': False,
                'columns_processed': 0,
                'scaler_used': False,
                'normalization_method': None,
                'errors': [],
                'warnings': []
            }

            # Si el modelo es un Pipeline completo (PyCaret), asumimos que ya incluye
            # todo el preprocesamiento interno (imputación + balanceo + z-score + PCA).
            if self.is_pycaret_model and self.preprocessor is None:
                processing_status["preprocessed"] = True
                processing_status["normalized"] = True
                processing_status["normalization_method"] = "internal_pipeline"
            
            # Inicializar variable para datos transformados con prefijos
            data_transformed_with_prefixes = pd.DataFrame()
            
            # PASO 1: Preparar columnas y datos ANTES del preprocesamiento
            # Para modelos sklearn, necesitamos las columnas en el orden correcto
            if not self.is_pycaret_model:
                # Obtener las columnas esperadas del modelo si es posible
                if hasattr(self.model, 'feature_names_in_'):
                    expected_cols = list(self.model.feature_names_in_)
                    
                    # Crear un DataFrame temporal con las columnas normalizadas para comparación
                    data_normalized = data.copy()
                    data_normalized.columns = data.columns.str.strip()
                    
                    # Identificar columnas faltantes y mapear columnas con espacios
                    missing_cols = []
                    col_mapping = {}  # Mapeo de columnas esperadas a columnas en data
                    
                    for expected_col in expected_cols:
                        norm_expected = expected_col.strip()
                        
                        # Buscar si existe la columna (con o sin espacio)
                        if expected_col in data.columns:
                            # Existe exactamente como se espera
                            col_mapping[expected_col] = expected_col
                        elif norm_expected in data_normalized.columns:
                            # Existe sin el espacio, mapear
                            # Encontrar el nombre original en data
                            for orig_col in data.columns:
                                if orig_col.strip() == norm_expected:
                                    col_mapping[expected_col] = orig_col
                                    break
                        else:
                            # No existe, agregar a faltantes
                            missing_cols.append(expected_col)
                    
                    # Agregar columnas faltantes con valores por defecto
                    if missing_cols:
                        logger.warning(f"Columnas faltantes detectadas: {missing_cols}")
                        logger.info("Agregando columnas faltantes con valores por defecto...")
                        
                        for col in missing_cols:
                            norm_col = col.strip()
                            # Valores por defecto según el tipo de columna
                            if 'fatty' in col.lower() or 'carbohydrate' in col.lower():
                                data[col] = 0.0  # Valores nutricionales por defecto
                            elif 'alcohol' in col.lower():
                                # Si ya tenemos 'alcohol' (sin espacio), copiar el valor
                                if 'alcohol' in data.columns:
                                    data[col] = data['alcohol']
                                elif any('alcohol' in c for c in data.columns):
                                    # Buscar cualquier columna con 'alcohol'
                                    alcohol_col = [c for c in data.columns if 'alcohol' in c.lower()][0]
                                    data[col] = data[alcohol_col]
                                else:
                                    data[col] = 0
                            else:
                                data[col] = 0.0
                    
                    # Crear nuevo DataFrame con las columnas en el orden correcto
                    data_reordered = pd.DataFrame()
                    for expected_col in expected_cols:
                        if expected_col in col_mapping:
                            data_reordered[expected_col] = data[col_mapping[expected_col]]
                        elif expected_col in data.columns:
                            data_reordered[expected_col] = data[expected_col]
                        else:
                            # Ya fue agregada arriba
                            data_reordered[expected_col] = data[expected_col]
                    
                    data = data_reordered
                    logger.info(f"Datos preparados: {len(data.columns)} columnas en el orden correcto")
            
            # PASO 2: APLICAR PREPROCESADOR COMPLETO
            # Aplicar el preprocesador completo (como se hizo durante el entrenamiento)
            # y luego mapear las columnas de vuelta a nombres originales
            logger.info(f"Verificando tipo de modelo: is_pycaret_model={self.is_pycaret_model}, preprocessor={self.preprocessor is not None}")
            
            if self.preprocessor is not None:
                # Aplicar preprocesador completo
                logger.info("🔧 Aplicando preprocesador completo a los datos de entrada...")
                try:
                    # Asegurar que tenemos todas las columnas que el preprocesador espera
                    if hasattr(self.preprocessor, 'feature_names_in_'):
                        preprocessor_cols = list(self.preprocessor.feature_names_in_)
                        missing_cols = set(preprocessor_cols) - set(data.columns)
                        if missing_cols:
                            logger.warning(f"   Columnas faltantes para preprocesador: {list(missing_cols)[:5]}...")
                            for col in missing_cols:
                                data[col] = 0.0
                        # Reordenar según lo que espera el preprocesador
                        data = data[preprocessor_cols]
                    
                    # Aplicar transformación
                    data_processed = self.preprocessor.transform(data)
                    
                    # Convertir resultado a DataFrame con nombres transformados (con prefijos)
                    if isinstance(data_processed, np.ndarray):
                        if hasattr(self.preprocessor, 'get_feature_names_out'):
                            feature_names = self.preprocessor.get_feature_names_out()
                        else:
                            feature_names = [f'feature_{i}' for i in range(data_processed.shape[1])]
                        data_processed = pd.DataFrame(data_processed, columns=feature_names, index=data.index)
                    
                    logger.info(f"✅ Preprocesador aplicado. Shape: {data_processed.shape}")
                    logger.info(f"   Primeras columnas transformadas: {list(data_processed.columns)[:5]}")
                    
                    # Guardar datos transformados con prefijos (antes del mapeo)
                    data_transformed_with_prefixes = data_processed.copy()
                    
                    # IMPORTANTE: Las columnas cat_num__ NO deben normalizarse, solo las num__
                    # Necesitamos mapear solo las num__ y mantener las cat_num__ con valores originales
                    # Guardar datos originales antes del preprocesamiento para las categóricas
                    data_before_preprocessing = data.copy()
                    
                    # Mapear columnas de vuelta a nombres originales
                    # Las columnas num__ se mapean con valores normalizados
                    # Las columnas cat_num__ se mapean con valores originales (no normalizados)
                    data = self._map_preprocessed_columns_to_original(data_processed, data_before_preprocessing)
                    
                    # IMPORTANTE: Guardar data_after_processing INMEDIATAMENTE después del mapeo
                    # antes de cualquier modificación adicional para la predicción
                    data_after_processing = data.copy()
                    
                    processing_status['preprocessed'] = True
                    processing_status['normalized'] = True  # El preprocesador incluye normalización
                    processing_status['normalization_method'] = 'preprocessor_complete'
                    processing_status['columns_processed'] = len(data.columns)
                    
                except Exception as e:
                    logger.error(f"❌ Error al aplicar preprocesador: {e}")
                    processing_status['errors'].append(f"Error al aplicar preprocesador: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"Error al aplicar preprocesador: {str(e)}")
            else:
                if self.is_pycaret_model:
                    # El Pipeline del modelo (por ejemplo lr_pca25_cw.pkl) ya incluye
                    # todo el preprocesamiento necesario.
                    logger.info(
                        "Usando preprocesamiento interno del Pipeline del modelo "
                        "(no se aplica preprocesador externo)."
                    )
                else:
                    logger.warning("⚠️ No hay preprocesador disponible. Los datos NO se procesarán.")
                    processing_status["warnings"].append("No hay preprocesador disponible")
                    logger.warning("   ⚠️ CRÍTICO: El modelo sklearn fue entrenado con datos procesados.")
                    logger.warning("   Sin preprocesamiento, las predicciones serán incorrectas.")
            
            # PASO 3: Asegurar que las columnas finales coincidan con las que espera el modelo
            # El modelo fue entrenado con datos preprocesados, así que espera los nombres transformados
            if not self.is_pycaret_model and hasattr(self.model, 'feature_names_in_'):
                model_expected_cols = list(self.model.feature_names_in_)
                logger.info(f"Modelo espera {len(model_expected_cols)} columnas después del preprocesamiento")
                logger.debug(f"Primeras 5 columnas esperadas por el modelo: {model_expected_cols[:5]}")
                logger.debug(f"Primeras 5 columnas después del preprocesamiento: {list(data.columns)[:5]}")
                
                # Verificar si las columnas coinciden
                data_cols = list(data.columns)
                if data_cols != model_expected_cols:
                    # Intentar reordenar si todas las columnas están presentes
                    missing_cols = set(model_expected_cols) - set(data_cols)
                    extra_cols = set(data_cols) - set(model_expected_cols)
                    
                    if missing_cols:
                        logger.warning(f"Columnas faltantes después del preprocesamiento: {list(missing_cols)[:5]}...")
                        # Agregar columnas faltantes con 0
                        for col in missing_cols:
                            data[col] = 0.0
                    
                    if extra_cols:
                        logger.warning(f"Columnas extra después del preprocesamiento: {list(extra_cols)[:5]}...")
                        # Eliminar columnas extra (no debería pasar, pero por si acaso)
                        data = data.drop(columns=list(extra_cols))
                    
                    # Reordenar columnas para que coincidan exactamente con el modelo
                    data = data[model_expected_cols]
                    logger.info(f"Columnas reordenadas para coincidir con el modelo: {len(data.columns)} columnas")
                else:
                    logger.info("✓ Las columnas después del preprocesamiento coinciden con las del modelo")
            
            # Realizar predicción según el tipo de modelo
            if self.is_pycaret_model:
                # Tratamos el modelo como un Pipeline sklearn completo:
                # recibe las 30 columnas crudas y aplica dentro imputación + SMOTE +
                # normalización z-score + PCA + clasificador final.
                logger.info(
                    "Usando Pipeline sklearn completo (por ejemplo lr_pca25_cw.pkl) "
                    "para predecir directamente."
                )

                # CRÍTICO: Reordenar columnas al orden exacto que espera el modelo
                # El modelo tiene feature_names_in_ que incluye 'stroke', así que extraemos solo las de entrada
                if hasattr(self.model, 'feature_names_in_'):
                    model_expected_cols = [col for col in self.model.feature_names_in_ if col != 'stroke']
                    logger.info(f"Modelo espera {len(model_expected_cols)} columnas en orden específico")
                    
                    # Verificar que tenemos todas las columnas
                    missing = set(model_expected_cols) - set(data.columns)
                    if missing:
                        logger.warning(f"Columnas faltantes: {list(missing)[:5]}...")
                        for col in missing:
                            data[col] = 0.0
                    
                    # Reordenar al orden exacto del modelo
                    data = data[model_expected_cols]
                    logger.info(f"DataFrame reordenado: {list(data.columns)[:5]}... (total: {len(data.columns)})")
                else:
                    # Si no tiene feature_names_in_, usar MODEL_INPUT_COLUMNS como fallback
                    logger.warning("Modelo no tiene feature_names_in_, usando MODEL_INPUT_COLUMNS")
                    if set(MODEL_INPUT_COLUMNS) <= set(data.columns):
                        data = data[MODEL_INPUT_COLUMNS]
                    else:
                        missing = set(MODEL_INPUT_COLUMNS) - set(data.columns)
                        logger.warning(f"Faltan columnas: {list(missing)[:5]}...")
                        for col in missing:
                            data[col] = 0.0
                        data = data[MODEL_INPUT_COLUMNS]

                # Predicción de clase
                predicted_label = self.model.predict(data)[0]

                # Probabilidades si están disponibles
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(data)[0]
                    prob_class_0 = probabilities[0] if len(probabilities) > 0 else 0.5
                    prob_class_1 = probabilities[1] if len(probabilities) > 1 else 0.5

                    if int(predicted_label) == 1:
                        probability = prob_class_1  # Probabilidad de STROKE
                    else:
                        probability = prob_class_0  # Probabilidad de NO STROKE
                else:
                    probability = 0.5
                    logger.warning(
                        "El Pipeline no soporta predict_proba, usando probabilidad 0.5 por defecto."
                    )
            else:
                # Usar modelo sklearn directamente
                predicted_label = self.model.predict(data)[0]
                
                # Obtener probabilidades si el modelo las soporta
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(data)[0]
                    # probabilities[0] = probabilidad clase 0 (NO stroke)
                    # probabilities[1] = probabilidad clase 1 (STROKE)
                    prob_class_0 = probabilities[0] if len(probabilities) > 0 else 0.5
                    prob_class_1 = probabilities[1] if len(probabilities) > 1 else 0.5
                    
                    # Usar la probabilidad de la clase predicha
                    if int(predicted_label) == 1:
                        probability = prob_class_1  # Probabilidad de STROKE
                    else:
                        probability = prob_class_0  # Probabilidad de NO STROKE
                else:
                    # Si no tiene predict_proba, usar 0.5 como default
                    probability = 0.5
                    logger.warning("El modelo no soporta predict_proba, usando 0.5")
            
            # Convertir a formato estándar
            prediction_str = "STROKE RISK" if int(predicted_label) == 1 else "NOT STROKE RISK"
            
            # Log de valores crudos para diagnóstico
            logger.info(f"Predicción cruda - Label: {predicted_label}, Probability (clase predicha): {probability:.6f}")
            if hasattr(self.model, 'predict_proba'):
                try:
                    probabilities_full = self.model.predict_proba(data)[0]
                    logger.info(f"Probabilidades completas - Clase 0 (NO stroke): {probabilities_full[0]:.6f}, Clase 1 (STROKE): {probabilities_full[1]:.6f}")
                except Exception as e:
                    logger.warning(f"No se pudo obtener probabilidades completas: {e}")
            
            # data_after_processing ya fue guardado después del mapeo (línea ~598)
            # Si no se guardó (porque no había preprocesador), guardarlo ahora
            if 'data_after_processing' not in locals():
                data_after_processing = data.copy()
            
            # Validar el procesamiento antes de hacer la predicción
            processing_valid, validation_status = self._validate_processing(processing_status)
            processing_status.update(validation_status)
            processing_successful = processing_valid and len(processing_status.get('errors', [])) == 0
            
            # Si el procesamiento no fue exitoso, NO hacer la predicción
            if not processing_successful:
                error_msg = "El procesamiento de datos no fue exitoso. No se puede realizar la predicción."
                if processing_status['errors']:
                    error_msg += f" Errores: {', '.join(processing_status['errors'])}"
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # La probabilidad ya es la de la clase predicha, usarla directamente
            stroke_probability = float(probability)
            stroke_probability = max(0.0, min(1.0, stroke_probability))
            
            # Log del resultado final
            logger.info(f"Resultado final - Predicción: {prediction_str}, Probabilidad: {stroke_probability:.4f} ({stroke_probability*100:.2f}%)")
            
            # Advertencia si la probabilidad es extrema (posible problema con el modelo o datos)
            if stroke_probability > 0.99:
                logger.warning(f"⚠️ Probabilidad muy alta ({stroke_probability*100:.2f}%). Verificar si el modelo está funcionando correctamente.")
                processing_status['warnings'].append(f"Probabilidad muy alta ({stroke_probability*100:.2f}%)")
            elif stroke_probability < 0.01:
                logger.warning(f"⚠️ Probabilidad muy baja ({stroke_probability*100:.2f}%). Verificar si el modelo está funcionando correctamente.")
                processing_status['warnings'].append(f"Probabilidad muy baja ({stroke_probability*100:.2f}%)")
            
            # Convertir DataFrames a diccionarios para serialización
            data_before_dict = data_before_processing.to_dict(orient='records')[0] if not data_before_processing.empty else {}
            data_after_dict = data_after_processing.to_dict(orient='records')[0] if not data_after_processing.empty else {}
            data_transformed_dict = data_transformed_with_prefixes.to_dict(orient='records')[0] if not data_transformed_with_prefixes.empty else {}
            
            result = {
                'prediction': prediction_str,
                'probability': stroke_probability,
                'details': {
                    'model_path': str(self.model_path),
                    'raw_prediction': int(predicted_label),
                    'raw_score': float(probability)
                },
                'data_before_processing': data_before_dict,
                'data_transformed_with_prefixes': data_transformed_dict,
                'data_after_processing': data_after_dict,
                'processing_status': processing_status,
                'processing_successful': processing_successful
            }
            
            logger.info(f"Predicción realizada: {prediction_str} (probabilidad: {stroke_probability:.2%})")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error al realizar la predicción: {str(e)}")
    
    def get_required_columns(self) -> List[str]:
        """Retorna la lista de columnas requeridas por el modelo NHANES.

        Usa el contrato centralizado definido en ``core.config_features``.

        Returns:
            Lista de nombres de columnas esperadas por el modelo.
        """
        if self.required_columns:
            return self.required_columns.copy()

        # Usar contrato centralizado de columnas de entrada
        self.required_columns = MODEL_INPUT_COLUMNS.copy()
        return self.required_columns.copy()
