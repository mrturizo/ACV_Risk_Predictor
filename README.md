# 🧠 ACV Risk Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)

Aplicación híbrida (Web y Escritorio) para predicción de riesgo de Accidente Cerebrovascular (ACV) usando Machine Learning. Desarrollada con Python, Streamlit (web) y Tkinter (escritorio).

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Versiones Disponibles](#-versiones-disponibles)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tecnologías](#-tecnologías)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)
- [Contacto](#-contacto)

## ✨ Características

- 🔬 **Predicción de Riesgo de ACV**: Utiliza modelos de Machine Learning entrenados con datos clínicos del NHANES
- 📊 **Múltiples Formas de Entrada**: Carga de archivos (CSV, Excel, JSON) o ingreso manual mediante formulario
- 🎯 **Perfiles Rápidos**: 3 perfiles predefinidos para pruebas rápidas (Paciente Sano, Factores de Riesgo, Múltiples Comorbilidades)
- 📄 **Reportes Detallados**: Generación de reportes PDF con análisis de influencia de variables y recomendaciones
- 🌐 **Acceso Web**: Disponible online desde cualquier dispositivo
- 💻 **Aplicación Desktop**: Versión instalable para Windows con interfaz nativa

## 🌐 Versiones Disponibles

### Versión Web (Streamlit)
**Acceso Online**: [🔗 URL de Streamlit Cloud](#) 

> **Nota**: Una vez que el repositorio esté en GitHub, puedes hacer deploy en Streamlit Cloud siguiendo las instrucciones en [GITHUB_SETUP.md](GITHUB_SETUP.md)

- Accesible desde cualquier dispositivo con navegador
- No requiere instalación
- Interfaz responsive y moderna
- Actualizaciones automáticas

### Versión Desktop (Tkinter)
- Instalador para Windows (.exe)
- Funciona sin conexión a internet
- Interfaz nativa de Windows
- Instalación simple con InnoSetup

## 📦 Requisitos

### Para Desarrollo
- Python 3.9 o superior
- pip (gestor de paquetes de Python)

### Para Uso de la App Desktop
- Windows 10 o superior
- No requiere Python instalado (incluido en el instalador)

## 🚀 Instalación

### Instalación Local (Desarrollo)

1. **Clonar el repositorio**:
```bash
git clone https://github.com/mrturizo/ACV_Risk_Predictor.git
cd ACV_Risk_Predictor
```

2. **Crear entorno virtual** (recomendado):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Configurar modelos**:
   - Colocar el modelo entrenado (`.pkl`) en la carpeta `models/`
   - Colocar el preprocesador (`.pkl`) en la carpeta `models/`

### Instalación Desktop (Usuarios Finales)

1. Descargar el instalador `ACV_Risk_Predictor_Setup.exe`
2. Ejecutar el instalador y seguir las instrucciones
3. La aplicación estará disponible en el menú de inicio

## 💻 Uso

### Versión Web

1. Acceder a la URL de Streamlit Cloud
2. Seleccionar método de entrada:
   - **Carga de archivo**: Subir archivo CSV, Excel o JSON
   - **Formulario manual**: Llenar los campos del formulario
   - **Perfil rápido**: Seleccionar uno de los 3 perfiles predefinidos
3. Hacer clic en "Obtener Predicción de Riesgo de ACV"
4. Revisar resultados y descargar reporte PDF (opcional)

### Versión Desktop

1. Abrir la aplicación desde el menú de inicio o escritorio
2. Seguir los mismos pasos que la versión web
3. Los reportes se guardan en la carpeta `data/outputs/`

## 📁 Estructura del Proyecto

```
ACV_Risk_Predictor/
├── app_web/              # Aplicación web (Streamlit)
│   ├── main_streamlit.py
│   └── Dockerfile
├── app_desktop/          # Aplicación desktop (Tkinter)
│   ├── main_tkinter.py
│   ├── utils_desktop.py
│   └── installer_script.iss
├── core/                 # Lógica compartida (núcleo)
│   ├── predictor.py      # Carga de modelos y predicción
│   ├── reports.py        # Generación de reportes PDF
│   ├── utils.py          # Utilidades y validaciones
│   └── profiles.py       # Perfiles de pacientes
├── ml_models/            # Desarrollo de modelos ML
│   ├── scripts/          # Scripts de entrenamiento
│   ├── data/             # Datos de entrenamiento
│   └── trained_models/  # Modelos entrenados
├── models/               # Modelos para producción
├── data/                 # Datos temporales
│   ├── uploads/         # Archivos subidos por usuarios
│   └── outputs/         # Reportes generados
├── tests/                # Pruebas unitarias
├── requirements.txt      # Dependencias Python
└── README.md            # Este archivo
```

## 🛠️ Tecnologías

### Backend
- **Python 3.9+**: Lenguaje principal
- **PyCaret**: Framework de ML automatizado
- **scikit-learn**: Modelos de ML tradicionales
- **pandas**: Manipulación de datos
- **numpy**: Cálculos numéricos

### Frontend Web
- **Streamlit**: Framework para aplicaciones web interactivas
- **Plotly**: Visualizaciones interactivas

### Frontend Desktop
- **Tkinter**: Interfaz gráfica nativa de Python
- **ttk**: Widgets modernos de Tkinter

### Herramientas
- **PyInstaller**: Compilación a ejecutable
- **InnoSetup**: Creación de instalador Windows
- **ReportLab/FPDF**: Generación de PDFs
- **Docker**: Containerización (opcional)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

Para más detalles, consulta [CONTRIBUTING.md](CONTRIBUTING.md) (si existe).

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 📧 Contacto

- **Proyecto**: [ACV Risk Predictor](https://github.com/mrturizo/ACV_Risk_Predictor)
- **Issues**: [GitHub Issues](https://github.com/mrturizo/ACV_Risk_Predictor/issues)

## 🙏 Agradecimientos

- Dataset NHANES para los datos de entrenamiento
- Comunidad de código abierto por las herramientas utilizadas
- Equipo de Data Science por el desarrollo de los modelos

---

**Nota**: Esta aplicación es una herramienta de apoyo y no reemplaza el diagnóstico médico profesional. Siempre consulte con un profesional de la salud para decisiones médicas importantes.
