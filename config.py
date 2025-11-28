"""
Archivo de configuración centralizado para el proyecto.

Define constantes, rutas y configuraciones que se usan
en toda la aplicación.
"""

from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DATA_UPLOADS = DATA_DIR / "uploads"
DATA_OUTPUTS = DATA_DIR / "outputs"
ML_MODELS_DIR = PROJECT_ROOT / "ml_models"
ML_TRAINED_MODELS = ML_MODELS_DIR / "trained_models"

# Configuración de modelos
DEFAULT_MODEL_NAME = "dummy_stroke_model.pkl"
MODEL_EXTENSIONS = [".pkl"]

# Configuración de archivos
SUPPORTED_FILE_FORMATS = {
    "csv": [".csv"],
    "excel": [".xlsx", ".xls"],
    "json": [".json"]
}
MAX_FILE_SIZE_MB = 50

# Configuración de predicción
PREDICTION_LABELS = {
    0: "NOT STROKE RISK",
    1: "STROKE RISK"
}

# Configuración de reportes
REPORT_OUTPUT_DIR = DATA_OUTPUTS
REPORT_TEMPLATE_PATH = None  # Se definirá cuando creemos templates

# Configuración de logging
LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# Crear directorio de logs si no existe
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

