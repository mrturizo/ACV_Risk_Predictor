"""
Utilidades para la aplicación de escritorio.

Incluye funciones para manejo de rutas compatibles con PyInstaller.
"""

import sys
import os
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """Obtiene la ruta absoluta a un recurso, compatible con PyInstaller.
    
    Cuando PyInstaller crea un ejecutable, empaqueta los archivos en una carpeta
    temporal y almacena la ruta en sys._MEIPASS. Esta función detecta si estamos
    corriendo desde un .exe o desde el código fuente.
    
    Args:
        relative_path: Ruta relativa al recurso (ej: "models/model.pkl")
        
    Returns:
        Path absoluto al recurso.
        
    Ejemplos:
        >>> resource_path("models/dummy_stroke_model.pkl")
        Path("C:/ruta/absoluta/models/dummy_stroke_model.pkl")
    """
    try:
        # PyInstaller crea una carpeta temporal y almacena la ruta en _MEIPASS
        # Si estamos corriendo desde un .exe, esta variable existe
        base_path = Path(sys._MEIPASS)
    except AttributeError:
        # Si no estamos en un .exe, estamos en modo desarrollo
        # Usar la ruta del directorio actual (raíz del proyecto)
        base_path = Path(os.path.abspath("."))
    
    # Combinar la ruta base con la ruta relativa
    full_path = base_path / relative_path
    
    return full_path


def get_project_root() -> Path:
    """Obtiene la raíz del proyecto, compatible con PyInstaller.
    
    Returns:
        Path a la raíz del proyecto.
    """
    try:
        # Si estamos en un .exe, la raíz es donde está el ejecutable
        if hasattr(sys, '_MEIPASS'):
            # En PyInstaller, el ejecutable está en el directorio padre de _MEIPASS
            # Pero para recursos, usamos _MEIPASS directamente
            # Para la raíz del proyecto, necesitamos otra estrategia
            # Por ahora, usamos el directorio del ejecutable
            if hasattr(sys, 'executable'):
                return Path(sys.executable).parent
        return Path(os.path.abspath("."))
    except Exception:
        return Path(os.path.abspath("."))


def get_models_dir() -> Path:
    """Obtiene el directorio de modelos, compatible con PyInstaller.
    
    Returns:
        Path al directorio de modelos.
    """
    project_root = get_project_root()
    models_dir = project_root / "models"
    
    # Si no existe en la raíz, intentar en _MEIPASS (si estamos en .exe)
    if not models_dir.exists() and hasattr(sys, '_MEIPASS'):
        models_dir = Path(sys._MEIPASS) / "models"
    
    return models_dir


def get_data_dir() -> Path:
    """Obtiene el directorio de datos, compatible con PyInstaller.
    
    Returns:
        Path al directorio de datos.
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    # Si no existe en la raíz, intentar en _MEIPASS (si estamos en .exe)
    if not data_dir.exists() and hasattr(sys, '_MEIPASS'):
        data_dir = Path(sys._MEIPASS) / "data"
    
    return data_dir

