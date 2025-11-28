"""Utilidades y funciones auxiliares para el proyecto."""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


def load_data_file(file_path: Path, encoding: str = 'utf-8') -> pd.DataFrame:
    """Carga datos desde archivo CSV, Excel o JSON.
    
    Args:
        file_path: Ruta al archivo de datos.
        encoding: Codificación del archivo (por defecto 'utf-8').
        
    Returns:
        DataFrame con los datos cargados.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato del archivo no es soportado.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo no existe: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            # Intentar diferentes encodings si utf-8 falla
            try:
                df = pd.read_csv(file_path, encoding=encoding)
            except UnicodeDecodeError:
                logger.warning(f"Error de encoding con {encoding}, intentando 'latin-1'")
                df = pd.read_csv(file_path, encoding='latin-1')
            return df
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, engine='openpyxl')
        elif suffix == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(
                f"Formato no soportado: {suffix}. "
                "Formatos soportados: .csv, .xlsx, .xls, .json"
            )
    except pd.errors.EmptyDataError:
        raise ValueError("El archivo está vacío o no contiene datos válidos.")
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {str(e)}")


def validate_data_structure(
    data: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Valida que los datos tengan la estructura esperada.
    
    Args:
        data: DataFrame a validar.
        required_columns: Lista de columnas requeridas.
        optional_columns: Lista opcional de columnas adicionales.
        
    Returns:
        Diccionario con:
        - 'valid': bool indicando si es válido
        - 'missing_columns': lista de columnas faltantes
        - 'extra_columns': lista de columnas adicionales no esperadas
        - 'message': mensaje descriptivo
        
    Raises:
        ValueError: Si data no es un DataFrame válido.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data debe ser un pandas DataFrame")
    
    data_columns = set(data.columns)
    required_set = set(required_columns)
    optional_set = set(optional_columns or [])
    expected_set = required_set | optional_set
    
    missing = list(required_set - data_columns)
    extra = list(data_columns - expected_set)
    
    valid = len(missing) == 0
    
    message = ""
    if missing:
        message += f"Columnas faltantes: {', '.join(missing)}. "
    if extra:
        message += f"Columnas adicionales: {', '.join(extra)}. "
    if valid and not extra:
        message = "Estructura de datos válida."
    
    return {
        'valid': valid,
        'missing_columns': missing,
        'extra_columns': extra,
        'message': message.strip()
    }


def load_field_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Carga la configuración de campos desde un archivo JSON.
    
    Args:
        config_path: Ruta al archivo JSON de configuración.
                    Si es None, busca 'field_config.json' en la raíz.
        
    Returns:
        Diccionario con la configuración de campos.
        
    Raises:
        FileNotFoundError: Si el archivo de configuración no existe.
        json.JSONDecodeError: Si el JSON es inválido.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "field_config.json"
    
    if not config_path.exists():
        # Retornar configuración por defecto vacía (se adaptará al modelo)
        return {
            'required_fields': [],
            'optional_fields': [],
            'field_types': {},
            'field_ranges': {}
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_recommendations(
    prediction: str,
    probability: float,
    input_data: pd.DataFrame
) -> List[str]:
    """Genera recomendaciones basadas en la predicción y datos del paciente.
    
    Args:
        prediction: 'STROKE RISK' o 'NOT STROKE RISK'
        probability: Probabilidad de riesgo (0-1)
        input_data: DataFrame con los datos del paciente.
        
    Returns:
        Lista de recomendaciones personalizadas.
    """
    recommendations = []
    
    if prediction == "STROKE RISK":
        recommendations.append(
            "⚠️ Se recomienda consultar con un médico especialista lo antes posible."
        )
        
        if probability > 0.7:
            recommendations.append(
                "🔴 El riesgo es ALTO. Se sugiere evaluación médica inmediata."
            )
        elif probability > 0.5:
            recommendations.append(
                "🟡 El riesgo es MODERADO. Programe una consulta médica en los próximos días."
            )
        else:
            recommendations.append(
                "🟢 El riesgo es BAJO pero presente. Consulte con su médico para seguimiento."
            )
        
        # Recomendaciones basadas en factores de riesgo comunes
        if 'age' in input_data.columns:
            age = input_data['age'].iloc[0] if len(input_data) > 0 else None
            if age and age > 65:
                recommendations.append(
                    "Debido a su edad, se recomienda realizar controles médicos más frecuentes."
                )
        
        if 'hypertension' in input_data.columns:
            hypertension = input_data['hypertension'].iloc[0] if len(input_data) > 0 else None
            if hypertension == 1:
                recommendations.append(
                    "Es importante mantener la presión arterial bajo control con seguimiento médico regular."
                )
        
        if 'heart_disease' in input_data.columns:
            heart_disease = input_data['heart_disease'].iloc[0] if len(input_data) > 0 else None
            if heart_disease == 1:
                recommendations.append(
                    "Si tiene enfermedad cardíaca, siga el tratamiento indicado por su cardiólogo."
                )
        
        if 'avg_glucose_level' in input_data.columns:
            glucose = input_data['avg_glucose_level'].iloc[0] if len(input_data) > 0 else None
            if glucose and glucose > 140:
                recommendations.append(
                    "Los niveles de glucosa elevados requieren atención. Consulte con un endocrinólogo."
                )
        
        if 'bmi' in input_data.columns:
            bmi = input_data['bmi'].iloc[0] if len(input_data) > 0 else None
            if bmi and bmi > 30:
                recommendations.append(
                    "El sobrepeso es un factor de riesgo. Considere un plan de alimentación saludable y ejercicio."
                )
        
        recommendations.append(
            "Evite el consumo de tabaco y alcohol en exceso."
        )
        recommendations.append(
            "Mantenga una dieta balanceada rica en frutas, verduras y baja en sodio."
        )
        
    else:
        recommendations.append(
            "✅ No se detectó riesgo significativo de ACV en este momento."
        )
        recommendations.append(
            "Mantenga hábitos saludables y controles médicos regulares."
        )
        
        # Recomendaciones preventivas
        if 'age' in input_data.columns:
            age = input_data['age'].iloc[0] if len(input_data) > 0 else None
            if age and age > 50:
                recommendations.append(
                    "Realice controles médicos anuales para monitorear su salud."
                )
        
        recommendations.append(
            "Mantenga un estilo de vida activo con ejercicio regular."
        )
        recommendations.append(
            "Siga una dieta equilibrada y mantenga un peso saludable."
        )
    
    return recommendations


def validate_data_types(
    data: pd.DataFrame,
    field_types: Dict[str, type]
) -> Dict[str, Any]:
    """Valida que los tipos de datos sean correctos.
    
    Args:
        data: DataFrame a validar.
        field_types: Diccionario con nombre de campo y tipo esperado.
        
    Returns:
        Diccionario con resultados de validación.
    """
    errors = []
    warnings = []
    
    for field, expected_type in field_types.items():
        if field not in data.columns:
            continue
        
        actual_type = data[field].dtype
        
        # Mapeo de tipos de pandas a tipos de Python
        type_mapping = {
            'int64': int,
            'float64': float,
            'object': str,
            'bool': bool
        }
        
        actual_python_type = type_mapping.get(str(actual_type), type(None))
        
        if expected_type == int and actual_python_type != int:
            try:
                data[field] = pd.to_numeric(data[field], errors='coerce').astype('Int64')
                warnings.append(f"Campo '{field}' convertido a entero")
            except:
                errors.append(f"Campo '{field}' no puede convertirse a entero")
        elif expected_type == float and actual_python_type != float:
            try:
                data[field] = pd.to_numeric(data[field], errors='coerce')
                warnings.append(f"Campo '{field}' convertido a flotante")
            except:
                errors.append(f"Campo '{field}' no puede convertirse a flotante")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def validate_data_ranges(
    data: pd.DataFrame,
    field_ranges: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """Valida que los valores estén en rangos esperados.
    
    Args:
        data: DataFrame a validar.
        field_ranges: Diccionario con rangos por campo.
                     Ejemplo: {'age': {'min': 0, 'max': 120}}
        
    Returns:
        Diccionario con resultados de validación.
    """
    errors = []
    warnings = []
    
    for field, ranges in field_ranges.items():
        if field not in data.columns:
            continue
        
        min_val = ranges.get('min')
        max_val = ranges.get('max')
        
        if min_val is not None:
            below_min = data[data[field] < min_val]
            if len(below_min) > 0:
                warnings.append(
                    f"Campo '{field}' tiene valores por debajo del mínimo ({min_val})"
                )
        
        if max_val is not None:
            above_max = data[data[field] > max_val]
            if len(above_max) > 0:
                warnings.append(
                    f"Campo '{field}' tiene valores por encima del máximo ({max_val})"
                )
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
