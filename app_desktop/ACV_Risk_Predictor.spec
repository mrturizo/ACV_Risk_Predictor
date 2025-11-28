# -*- mode: python ; coding: utf-8 -*-
"""
Archivo de especificación de PyInstaller para ACV Risk Predictor.

Este archivo define cómo PyInstaller debe empaquetar la aplicación.

Uso:
    pyinstaller ACV_Risk_Predictor.spec
"""

import sys
import os
from pathlib import Path

block_cipher = None

# Obtener ruta del directorio donde está este archivo .spec
# PyInstaller pasa la ruta del spec en SPECPATH
if 'SPECPATH' in globals():
    spec_dir = Path(SPECPATH).parent
else:
    # Fallback: usar el directorio actual de trabajo
    spec_dir = Path(os.getcwd())
    if spec_dir.name != 'app_desktop':
        spec_dir = spec_dir / 'app_desktop'

project_root = spec_dir.parent

# Análisis de la aplicación
# Usar ruta relativa desde el directorio del spec
main_script = 'main_tkinter.py'
if not Path(main_script).exists():
    main_script = str(spec_dir / 'main_tkinter.py')

a = Analysis(
    [main_script],
    pathex=[
        str(project_root),
        str(project_root / 'core'),
        str(spec_dir),
    ],
    binaries=[],
    datas=[
        # Incluir modelos si existen
        *([(str(project_root / 'models'), 'models')] if (project_root / 'models').exists() else []),
        # Incluir config.py si existe
        *([(str(project_root / 'config.py'), '.')] if (project_root / 'config.py').exists() else []),
    ],
    hiddenimports=[
        'pandas',
        'numpy',
        'openpyxl',
        'reportlab',
        'fpdf',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'core',
        'core.predictor',
        'core.reports',
        'core.utils',
        'core.predictor_mock',
        'app_desktop',
        'app_desktop.utils_desktop',
        'pathlib',
        'datetime',
        'traceback',
        'json',
        'logging',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'streamlit',  # No necesitamos Streamlit en la app de escritorio
        'matplotlib',  # Opcional
        'plotly',  # Opcional
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Crear PYZ (archivo comprimido con bytecode)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Crear ejecutable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ACV_Risk_Predictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Sin consola (aplicación GUI)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if Path('icon.ico').exists() else None,  # Icono de la aplicación
    version='1.0.0',  # Versión de la aplicación
)
