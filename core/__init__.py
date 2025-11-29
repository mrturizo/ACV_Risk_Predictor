"""Módulo core: Lógica compartida para predicción de riesgo de ACV."""

# Importar configuraciones centralizadas
try:
    import config
    PROJECT_ROOT = config.PROJECT_ROOT
    MODELS_DIR = config.MODELS_DIR
    DATA_DIR = config.DATA_DIR
    DATA_UPLOADS = config.DATA_UPLOADS
    DATA_OUTPUTS = config.DATA_OUTPUTS
except ImportError:
    # Fallback si config.py no está disponible
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).parent.parent
    MODELS_DIR = PROJECT_ROOT / "models"
    DATA_DIR = PROJECT_ROOT / "data"
    DATA_UPLOADS = DATA_DIR / "uploads"
    DATA_OUTPUTS = DATA_DIR / "outputs"

from core.config_features import MODEL_INPUT_COLUMNS, TARGET_COLUMN
from core.predictor import StrokePredictor
from core.reports import ReportGenerator
from core.utils import (
    load_data_file,
    validate_data_structure,
    validate_data_types,
    validate_data_ranges,
    get_recommendations,
    load_field_config,
    transform_age_to_category,
)

__version__ = "0.1.0"
__all__ = [
    "StrokePredictor",
    "ReportGenerator",
    "load_data_file",
    "validate_data_structure",
    "validate_data_types",
    "validate_data_ranges",
    "get_recommendations",
    "load_field_config",
    "transform_age_to_category",
    "MODEL_INPUT_COLUMNS",
    "TARGET_COLUMN",
    "PROJECT_ROOT",
    "MODELS_DIR",
    "DATA_DIR",
    "DATA_UPLOADS",
    "DATA_OUTPUTS",
]
