"""
Script para generar la estructura completa del proyecto ACV Risk Predictor.

Ejecutar: python setup_project.py
"""

import os
from pathlib import Path


def create_structure():
    """Crea toda la estructura de carpetas y archivos base del proyecto."""
    
    # Estructura de carpetas
    folders = [
        "data/uploads",
        "data/outputs",
        "models",
        "core",
        "app_web",
        "app_desktop",
        "tests",
    ]
    
    # Archivos iniciales y su contenido básico
    files = {
        "requirements.txt": """# Dependencias del Proyecto ACV Risk Predictor

# Machine Learning
pycaret[full]>=3.0.0

# Interfaz Web
streamlit>=1.28.0

# Procesamiento de Datos
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0  # Para leer Excel

# Generación de Reportes PDF
reportlab>=4.0.0
fpdf>=2.5.0  # Fallback si ReportLab no funciona

# Visualización
plotly>=5.17.0
matplotlib>=3.7.0

# Utilidades
python-dotenv>=1.0.0
pathlib2>=2.3.7; python_version < '3.4'
""",
        
        "README.md": """# ACV Risk Predictor

Aplicación híbrida (Web y Escritorio) para predicción de riesgo de Accidente Cerebrovascular (ACV) usando Machine Learning.

## Arquitectura

El proyecto utiliza una arquitectura de "Núcleo Compartido":

- **`core/`**: Lógica de negocio compartida (predicción, reportes, validación)
- **`app_web/`**: Interfaz web con Streamlit (ejecutable en Docker)
- **`app_desktop/`**: Interfaz de escritorio con Tkinter (compilable a .exe)

## Instalación

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Colocar el modelo entrenado (.pkl) en la carpeta `models/`

3. Para ejecutar la versión web:
```bash
cd app_web
streamlit run main_streamlit.py
```

4. Para ejecutar la versión de escritorio:
```bash
cd app_desktop
python main_tkinter.py
```

## Estructura del Proyecto

```
ACV_Risk_Predictor/
├── core/              # Lógica compartida
│   ├── predictor.py   # Carga de modelo y predicción
│   ├── reports.py     # Generación de reportes PDF
│   └── utils.py       # Utilidades y validación
├── app_web/           # Interfaz Streamlit
├── app_desktop/       # Interfaz Tkinter
├── models/            # Modelos entrenados (.pkl)
├── data/              # Datos temporales
└── tests/             # Pruebas unitarias
```

## Desarrollo

Ver `.cursorrules` para reglas de desarrollo y arquitectura.
""",
        
        "core/__init__.py": """\"\"\"Módulo core: Lógica compartida para predicción de riesgo de ACV.\"\"\"

from pathlib import Path

# Rutas base del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

__version__ = "0.1.0"
""",
        
        "core/predictor.py": """\"\"\"Módulo para carga de modelos y predicción de riesgo de ACV.\"\"\"

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from pycaret.classification import load_model, predict_model


class StrokePredictor:
    \"\"\"Clase para manejar la carga del modelo y realizar predicciones.
    
    Esta clase encapsula la lógica de carga de modelos PyCaret y predicción
    de riesgo de ACV basada en datos clínicos, demográficos y biomédicos.
    \"\"\"
    
    def __init__(self, model_path: Optional[Path] = None):
        \"\"\"Inicializa el predictor con un modelo.
        
        Args:
            model_path: Ruta al archivo .pkl del modelo. Si es None, busca
                       en la carpeta models/ el primer archivo .pkl encontrado.
                       
        Raises:
            FileNotFoundError: Si no se encuentra ningún modelo.
            ValueError: Si el modelo no es válido.
        \"\"\"
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self) -> None:
        \"\"\"Carga el modelo PyCaret desde el archivo especificado.\"\"\"
        # TODO: Implementar carga del modelo
        pass
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Realiza la predicción de riesgo de ACV.
        
        Args:
            data: DataFrame con los datos del paciente. Debe contener las
                  columnas esperadas por el modelo.
                  
        Returns:
            Diccionario con:
            - 'prediction': 'STROKE RISK' o 'NOT STROKE RISK'
            - 'probability': Probabilidad de riesgo (0-1)
            - 'details': Información adicional sobre la predicción
            
        Raises:
            ValueError: Si los datos no tienen el formato correcto.
        \"\"\"
        # TODO: Implementar predicción
        pass
    
    def get_required_columns(self) -> list:
        \"\"\"Retorna la lista de columnas requeridas por el modelo.
        
        Returns:
            Lista de nombres de columnas esperadas por el modelo.
        \"\"\"
        # TODO: Implementar obtención de columnas requeridas
        pass
""",
        
        "core/reports.py": """\"\"\"Módulo para generación de reportes PDF.\"\"\"

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    try:
        from fpdf import FPDF
        FPDF_AVAILABLE = True
    except ImportError:
        FPDF_AVAILABLE = False


class ReportGenerator:
    \"\"\"Generador de reportes PDF con predicción de riesgo de ACV.
    
    Intenta usar ReportLab primero, si no está disponible usa FPDF como fallback.
    \"\"\"
    
    def __init__(self):
        \"\"\"Inicializa el generador de reportes.\"\"\"
        self.use_reportlab = REPORTLAB_AVAILABLE
        self.use_fpdf = not REPORTLAB_AVAILABLE and FPDF_AVAILABLE
    
    def generate_report(
        self,
        prediction_result: Dict[str, Any],
        input_data: pd.DataFrame,
        output_path: Path,
        recommendations: Optional[list] = None
    ) -> Path:
        \"\"\"Genera un reporte PDF con los resultados de la predicción.
        
        Args:
            prediction_result: Diccionario con resultados de la predicción.
            input_data: DataFrame con los datos ingresados por el usuario.
            output_path: Ruta donde guardar el reporte PDF.
            recommendations: Lista opcional de recomendaciones personalizadas.
            
        Returns:
            Ruta al archivo PDF generado.
            
        Raises:
            RuntimeError: Si no hay librería de PDF disponible.
        \"\"\"
        if self.use_reportlab:
            return self._generate_with_reportlab(
                prediction_result, input_data, output_path, recommendations
            )
        elif self.use_fpdf:
            return self._generate_with_fpdf(
                prediction_result, input_data, output_path, recommendations
            )
        else:
            raise RuntimeError(
                "No hay librería de PDF disponible. Instala reportlab o fpdf."
            )
    
    def _generate_with_reportlab(
        self,
        prediction_result: Dict[str, Any],
        input_data: pd.DataFrame,
        output_path: Path,
        recommendations: Optional[list]
    ) -> Path:
        \"\"\"Genera reporte usando ReportLab.\"\"\"
        # TODO: Implementar generación con ReportLab
        pass
    
    def _generate_with_fpdf(
        self,
        prediction_result: Dict[str, Any],
        input_data: pd.DataFrame,
        output_path: Path,
        recommendations: Optional[list]
    ) -> Path:
        \"\"\"Genera reporte usando FPDF (fallback).\"\"\"
        # TODO: Implementar generación con FPDF
        pass
""",
        
        "core/utils.py": """\"\"\"Utilidades y funciones auxiliares para el proyecto.\"\"\"

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json


def load_data_file(file_path: Path) -> pd.DataFrame:
    \"\"\"Carga datos desde archivo CSV, Excel o JSON.
    
    Args:
        file_path: Ruta al archivo de datos.
        
    Returns:
        DataFrame con los datos cargados.
        
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el formato del archivo no es soportado.
    \"\"\"
    if not file_path.exists():
        raise FileNotFoundError(f"El archivo no existe: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path, engine='openpyxl')
        elif suffix == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(
                f"Formato no soportado: {suffix}. "
                "Formatos soportados: .csv, .xlsx, .xls, .json"
            )
    except Exception as e:
        raise ValueError(f"Error al cargar el archivo: {str(e)}")


def validate_data_structure(
    data: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> Dict[str, Any]:
    \"\"\"Valida que los datos tengan la estructura esperada.
    
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
    \"\"\"
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
    \"\"\"Carga la configuración de campos desde un archivo JSON.
    
    Args:
        config_path: Ruta al archivo JSON de configuración.
                    Si es None, busca 'field_config.json' en la raíz.
        
    Returns:
        Diccionario con la configuración de campos.
        
    Raises:
        FileNotFoundError: Si el archivo de configuración no existe.
        json.JSONDecodeError: Si el JSON es inválido.
    \"\"\"
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
    \"\"\"Genera recomendaciones basadas en la predicción y datos del paciente.
    
    Args:
        prediction: 'STROKE RISK' o 'NOT STROKE RISK'
        probability: Probabilidad de riesgo (0-1)
        input_data: DataFrame con los datos del paciente.
        
    Returns:
        Lista de recomendaciones personalizadas.
    \"\"\"
    recommendations = []
    
    if prediction == "STROKE RISK":
        recommendations.append(
            "Se recomienda consultar con un médico especialista lo antes posible."
        )
        if probability > 0.7:
            recommendations.append(
                "El riesgo es alto. Se sugiere evaluación médica inmediata."
            )
    else:
        recommendations.append(
            "Mantener hábitos saludables y controles médicos regulares."
        )
    
    # TODO: Agregar recomendaciones más específicas basadas en los datos
    
    return recommendations
""",
        
        "app_web/main_streamlit.py": """\"\"\"Aplicación web Streamlit para predicción de riesgo de ACV.\"\"\"

import streamlit as st
from pathlib import Path
import sys

# Agregar el directorio raíz al path para importar core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import StrokePredictor
from core.reports import ReportGenerator
from core.utils import load_data_file, validate_data_structure, get_recommendations


def main():
    \"\"\"Función principal de la aplicación web.\"\"\"
    st.set_page_config(
        page_title="ACV Risk Predictor",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Predictor de Riesgo de ACV")
    st.markdown("---")
    
    # TODO: Implementar interfaz de carga de datos y formulario
    st.info("Interfaz en desarrollo. Implementar carga de archivos y formulario.")
    
    # Placeholder para la lógica
    # predictor = StrokePredictor()
    # report_generator = ReportGenerator()


if __name__ == "__main__":
    main()
""",
        
        "app_web/Dockerfile": """# Dockerfile para la aplicación web Streamlit
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app_web/main_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
""",
        
        "app_desktop/main_tkinter.py": """\"\"\"Aplicación de escritorio Tkinter para predicción de riesgo de ACV.\"\"\"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import sys

# Agregar el directorio raíz al path para importar core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.predictor import StrokePredictor
from core.reports import ReportGenerator
from core.utils import load_data_file, validate_data_structure, get_recommendations


class StrokePredictorApp:
    \"\"\"Aplicación principal de escritorio para predicción de ACV.\"\"\"
    
    def __init__(self, root: tk.Tk):
        \"\"\"Inicializa la aplicación.
        
        Args:
            root: Ventana principal de Tkinter.
        \"\"\"
        self.root = root
        self.root.title("ACV Risk Predictor")
        self.root.geometry("800x600")
        
        # TODO: Implementar interfaz de usuario
        label = tk.Label(root, text="Interfaz en desarrollo", font=("Arial", 16))
        label.pack(pady=50)
    
    def run(self):
        \"\"\"Ejecuta la aplicación.\"\"\"
        self.root.mainloop()


def main():
    \"\"\"Función principal.\"\"\"
    root = tk.Tk()
    app = StrokePredictorApp(root)
    app.run()


if __name__ == "__main__":
    main()
""",
        
        "app_desktop/installer_script.iss": """; Script de InnoSetup para crear instalador .exe
; Este archivo se usará para compilar la aplicación de escritorio

[Setup]
AppName=ACV Risk Predictor
AppVersion=0.1.0
DefaultDirName={pf}\ACV_Risk_Predictor
DefaultGroupName=ACV Risk Predictor
OutputDir=dist
OutputBaseFilename=ACV_Risk_Predictor_Setup
Compression=lzma
SolidCompression=yes

[Files]
; TODO: Agregar archivos necesarios después de compilar con PyInstaller
; Source: "dist\main_tkinter.exe"; DestDir: "{app}"

[Icons]
Name: "{group}\ACV Risk Predictor"; Filename: "{app}\main_tkinter.exe"
Name: "{commondesktop}\ACV Risk Predictor"; Filename: "{app}\main_tkinter.exe"

[Run]
Filename: "{app}\main_tkinter.exe"; Description: "Ejecutar ACV Risk Predictor"; Flags: nowait postinstall skipifsilent
""",
        
        "tests/__init__.py": """\"\"\"Tests para el proyecto ACV Risk Predictor.\"\"\"
""",
        
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Proyecto específico
data/uploads/*
data/outputs/*
!data/uploads/.gitkeep
!data/outputs/.gitkeep
models/*.pkl
!models/.gitkeep

# Docker
.dockerignore

# OS
.DS_Store
Thumbs.db
"""
    }
    
    # Crear carpetas
    print("Creando estructura de carpetas...")
    for folder in folders:
        folder_path = Path(folder)
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Carpeta creada: {folder}")
    
    # Crear archivos .gitkeep en carpetas de datos
    (Path("data/uploads") / ".gitkeep").touch()
    (Path("data/outputs") / ".gitkeep").touch()
    (Path("models") / ".gitkeep").touch()
    
    # Crear archivos
    print("\nCreando archivos base...")
    for filepath, content in files.items():
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  [OK] Archivo creado: {filepath}")
    
    print("\n" + "="*60)
    print("¡Estructura del proyecto generada con éxito!")
    print("="*60)
    print("\nPróximos pasos:")
    print("1. Coloca tu modelo entrenado (.pkl) en la carpeta 'models/'")
    print("2. Instala dependencias: pip install -r requirements.txt")
    print("3. Ejecuta la aplicación web: streamlit run app_web/main_streamlit.py")
    print("4. O ejecuta la aplicación de escritorio: python app_desktop/main_tkinter.py")
    print("\nVer '.cursorrules' para reglas de desarrollo y arquitectura.")


if __name__ == "__main__":
    create_structure()

