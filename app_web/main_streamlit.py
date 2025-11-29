"""Aplicación web Streamlit para predicción de riesgo de ACV."""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime
import traceback
import logging

# Configurar logger
logger = logging.getLogger(__name__)

# Agregar el directorio raíz al path para importar core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    StrokePredictor,
    ReportGenerator,
    load_data_file,
    validate_data_structure,
    get_recommendations,
    transform_age_to_category,
    MODELS_DIR,
    DATA_UPLOADS,
    DATA_OUTPUTS,
    MODEL_INPUT_COLUMNS,
)
from core.profiles import get_profile, get_available_profiles

# Intentar importar predictor mock si PyCaret no está disponible
try:
    from core.predictor_mock import StrokePredictorMock
    MOCK_AVAILABLE = True
except ImportError:
    MOCK_AVAILABLE = False

# Configuración de página
    st.set_page_config(
        page_title="ACV Risk Predictor",
        page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None


@st.cache_resource
def load_predictor_cached():
    """Carga el predictor con cache para evitar recargar en cada ejecución.
    
    IMPORTANTE: Esta función SOLO carga el modelo 'lr_pca25_cw.pkl'.
    No busca otros modelos como fallback.
    """
    try:
        # Verificar que MODELS_DIR existe
        if not MODELS_DIR.exists():
            logger.error(f"MODELS_DIR no existe: {MODELS_DIR}")
            logger.error(f"PROJECT_ROOT: {MODELS_DIR.parent}")
            raise FileNotFoundError(f"El directorio de modelos no existe: {MODELS_DIR}")
        
        # OBLIGATORIO: Solo usar lr_pca25_cw.pkl - NO buscar otros modelos
        # ESTRATEGIA: En Streamlit Cloud (sin PyCaret), usar modelo convertido.
        # En desarrollo local (con PyCaret), usar modelo original.
        original_model = MODELS_DIR / "lr_pca25_cw.pkl"
        converted_model = MODELS_DIR / "lr_pca25_cw_sklearn.pkl"
        
        # Verificar si PyCaret está disponible (importado en predictor.py)
        from core.predictor import PYCARET_AVAILABLE
        
        if not PYCARET_AVAILABLE and converted_model.exists():
            # Streamlit Cloud sin PyCaret - usar modelo convertido
            required_model = converted_model
            logger.info(f"✅ [Cloud] Modelo convertido encontrado (sklearn puro): {required_model}")
        elif original_model.exists():
            # Desarrollo local con PyCaret o fallback - usar modelo original
            required_model = original_model
            if PYCARET_AVAILABLE:
                logger.info(f"✅ [Local] Usando modelo original con PyCaret: {required_model}")
            else:
                logger.warning(f"⚠️ [Fallback] Usando modelo original sin PyCaret: {required_model}")
                logger.info("   Se intentará cargar con mocks de PyCaret")
        else:
            error_msg = (
                f"❌ ERROR CRÍTICO: No se encontró ningún modelo en {MODELS_DIR}. "
                f"Se buscó: 'lr_pca25_cw.pkl' y 'lr_pca25_cw_sklearn.pkl'. "
                f"Este modelo es OBLIGATORIO."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"✅ Modelo requerido encontrado: {required_model}")
        try:
            predictor = StrokePredictor(model_path=required_model)
            logger.info(f"✅ Modelo cargado exitosamente: {type(predictor.model)}")
            logger.info(f"✅ Ruta del modelo cargado: {predictor.model_path}")
            return predictor
        except Exception as load_err:
            error_msg = (
                f"❌ ERROR CRÍTICO: No se pudo cargar el modelo requerido 'lr_pca25_cw.pkl'. "
                f"Error: {load_err}. Este modelo es OBLIGATORIO."
            )
            logger.error(error_msg)
            logger.exception(load_err)
            raise ValueError(error_msg)
        
    except (FileNotFoundError, ValueError) as e:
        # Re-lanzar errores específicos sin buscar alternativas
        logger.error(f"❌ Error al cargar el modelo requerido: {e}")
        raise
    except Exception as e:
        error_msg = f"❌ Error crítico inesperado al cargar el modelo: {e}"
        logger.error(error_msg)
        logger.exception(e)
        raise RuntimeError(error_msg)

def load_predictor():
    """Carga el predictor si no está cargado (usa cache).
    
    IMPORTANTE: Solo carga 'lr_pca25_cw.pkl'. No usa MOCK como fallback.
    """
    if st.session_state.predictor is None:
        try:
            predictor = load_predictor_cached()
            if predictor is None:
                st.error("❌ ERROR CRÍTICO: No se pudo cargar el modelo requerido 'lr_pca25_cw.pkl'")
                st.error("Este modelo es OBLIGATORIO. Por favor, verifica que el archivo existe en la carpeta 'models/'.")
                return False
            st.session_state.predictor = predictor
            return True
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            st.error(f"❌ ERROR CRÍTICO: No se pudo cargar el modelo requerido 'lr_pca25_cw.pkl'")
            st.error(f"Detalles: {str(e)}")
            st.error("Este modelo es OBLIGATORIO. Por favor, verifica que el archivo existe en la carpeta 'models/'.")
            logger.exception(e)
            return False
        except Exception as e:
            st.error(f"❌ Error inesperado al cargar el modelo: {str(e)}")
            logger.exception(e)
            return False
    return True


def render_sidebar():
    """Renderiza la barra lateral con información."""
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        Esta aplicación predice el riesgo de Accidente Cerebrovascular (ACV) 
        basándose en datos clínicos, demográficos y biomédicos.
        
        ### Formatos soportados:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - JSON (.json)
        
        ### Método de entrada:
        1. **Cargar archivo**: Sube un archivo con los datos
        2. **Formulario manual**: Completa el formulario paso a paso
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Estado del Modelo")
        
        if st.session_state.predictor is not None:
            is_mock = isinstance(st.session_state.predictor, StrokePredictorMock) if MOCK_AVAILABLE else False
            if is_mock:
                st.error("❌ ERROR: Modo MOCK no permitido")
                st.caption("El modelo requerido 'lr_pca25_cw.pkl' no está disponible")
                # Mostrar información de diagnóstico
                with st.expander("🔍 Diagnóstico - ¿Por qué MOCK?"):
                    st.write(f"**MODELS_DIR:** `{MODELS_DIR}`")
                    st.write(f"**¿Existe MODELS_DIR?** {MODELS_DIR.exists()}")
                    if MODELS_DIR.exists():
                        pkl_files = list(MODELS_DIR.glob("*.pkl"))
                        st.write(f"**Archivos .pkl encontrados:** {[f.name for f in pkl_files]}")
                        required_model = MODELS_DIR / "lr_pca25_cw.pkl"
                        st.write(f"**lr_pca25_cw.pkl existe?** {required_model.exists()}")
                        if not required_model.exists():
                            st.error("❌ El modelo requerido 'lr_pca25_cw.pkl' NO existe")
                    st.write("**Revisa los logs del servidor para más detalles.**")
            else:
                st.success("✅ Modelo cargado")
                if hasattr(st.session_state.predictor, 'model_path'):
                    model_name = st.session_state.predictor.model_path.name
                    st.caption(f"Modelo: {model_name}")
                    # Verificar que es el modelo correcto
                    if model_name != "lr_pca25_cw.pkl":
                        st.warning(f"⚠️ ADVERTENCIA: Se está usando '{model_name}' en lugar de 'lr_pca25_cw.pkl'")
                    else:
                        st.caption("✅ Modelo correcto: lr_pca25_cw.pkl")
                    if hasattr(st.session_state.predictor, 'is_pycaret_model'):
                        model_type = "PyCaret Pipeline" if st.session_state.predictor.is_pycaret_model else "sklearn"
                        st.caption(f"Tipo: {model_type}")
        else:
            st.warning("⚠️ Modelo no cargado")
            st.caption("Modelo requerido: lr_pca25_cw.pkl")
            if st.button("🔄 Cargar Modelo", use_container_width=True):
                load_predictor()


def render_file_upload():
    """Renderiza la sección de carga de archivos."""
    st.header("📁 Cargar Datos desde Archivo")
    
    uploaded_file = st.file_uploader(
        "Selecciona un archivo",
        type=['csv', 'xlsx', 'xls', 'json'],
        help="Formatos soportados: CSV, Excel (.xlsx, .xls), JSON"
    )
    
    if uploaded_file is not None:
        try:
            # Guardar archivo temporalmente
            uploads_dir = DATA_UPLOADS
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = uploads_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Cargar datos
            with st.spinner("Cargando archivo..."):
                data = load_data_file(file_path)
            
            st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
            st.info(f"📊 Filas: {len(data)}, Columnas: {len(data.columns)}")
            
            # Mostrar preview
            with st.expander("👁️ Vista previa de los datos"):
                st.dataframe(data.head(10), use_container_width=True)
            
            # Seleccionar fila si hay múltiples
            if len(data) > 1:
                st.subheader("Seleccionar registro")
                row_index = st.number_input(
                    "Índice de la fila a analizar",
                    min_value=0,
                    max_value=len(data) - 1,
                    value=0,
                    help="Selecciona qué fila del archivo quieres analizar"
                )
                selected_data = data.iloc[[row_index]]
            else:
                selected_data = data
            
            # Asegurar columnas y orden según contrato del modelo
            try:
                selected_data = selected_data[MODEL_INPUT_COLUMNS]
            except KeyError:
                # Si faltan columnas, se manejará más adelante en el predictor
                pass

            # Botón para realizar predicción
            if st.button("🔮 Realizar Predicción", type="primary", use_container_width=True):
                if not load_predictor():
                    return
                
                try:
                    with st.spinner("Realizando predicción..."):
                        result = st.session_state.predictor.predict(selected_data)
                        recommendations = get_recommendations(
                            result['prediction'],
                            result['probability'],
                            selected_data
                        )
                        
                        st.session_state.prediction_result = result
                        st.session_state.input_data = selected_data
                        st.session_state.recommendations = recommendations
                        
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error al realizar la predicción: {str(e)}")
                    st.exception(e)
        
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")
            st.exception(e)


def render_manual_form():
    """Renderiza el formulario manual de entrada de datos con todas las variables NHANES."""
    st.header("✍️ Ingreso Manual de Datos")
    
    # Botones de perfiles de llenado rápido
    st.subheader("⚡ Perfiles de Llenado Rápido")
    col_prof1, col_prof2, col_prof3 = st.columns(3)
    
    profile_data = None
    with col_prof1:
        if st.button("🟢 Paciente Sano", use_container_width=True, help="Llena el formulario con valores de un paciente sano"):
            profile_data = get_profile('sano')
            st.session_state.profile_data = profile_data
            st.rerun()
    
    with col_prof2:
        if st.button("🟡 Factores de Riesgo", use_container_width=True, help="Llena el formulario con valores de un paciente con factores de riesgo"):
            profile_data = get_profile('factores_riesgo')
            st.session_state.profile_data = profile_data
            st.rerun()
    
    with col_prof3:
        if st.button("🔴 Múltiples Comorbilidades", use_container_width=True, help="Llena el formulario con valores de un paciente con múltiples comorbilidades"):
            profile_data = get_profile('comorbilidades')
            st.session_state.profile_data = profile_data
            st.rerun()
    
    # Cargar datos del perfil si existe - SOLO para valores iniciales
    # Los valores del formulario se capturan cuando se envía, no cuando se renderiza
    profile_data = {}
    if 'profile_data' in st.session_state and st.session_state.profile_data:
        profile_data = st.session_state.profile_data.copy()
        # DEBUG: Mostrar qué perfil se está cargando
        st.info(f"🔍 DEBUG: Perfil cargado con {len(profile_data)} valores. Ejemplos: age={profile_data.get('age')}, diabetes={profile_data.get('diabetes')}, hypertension={profile_data.get('hypertension')}, Health Insurance={profile_data.get('Health Insurance')}, General health condition={profile_data.get('General health condition')}")
        # NO limpiar aquí - se limpiará después de que se use para inicializar los campos
    else:
        # Si no hay perfil, mostrar que se están usando valores por defecto
        if 'last_profile_used' not in st.session_state:
            st.info("ℹ️ No hay perfil cargado. Se usarán valores por defecto.")
    
    st.markdown("---")
    
    with st.form("manual_input_form", clear_on_submit=False):
        # Sección 1: Datos Demográficos
        with st.expander("📋 Datos Demográficos", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                # Usar el valor del perfil si existe, sino usar el valor por defecto
                age_default = int(profile_data.get('age', 50)) if profile_data else 50
                age = st.number_input("Edad", min_value=0, max_value=120, value=age_default, step=1)
                gender = st.selectbox(
                    "Género", 
                    options=[1, 2], 
                    index=0 if profile_data.get('gender', 1) == 1 else 1,
                    format_func=lambda x: "Masculino" if x == 1 else "Femenino"
                )
            with col2:
                Race = st.selectbox(
                    "Raza",
                    options=[1, 2, 3, 4, 5],
                    index=max(0, min(4, [1, 2, 3, 4, 5].index(profile_data.get('Race', 3)) if profile_data.get('Race', 3) in [1, 2, 3, 4, 5] else 2)),
                    format_func=lambda x: {1: "Mexicano Americano", 2: "Otro Hispano", 3: "Blanco", 4: "Negro", 5: "Otro"}.get(x, "Otro")
                )
                Marital_status = st.selectbox(
                    "Estado Civil",
                    options=[1, 2, 3, 4, 5],
                    index=max(0, min(4, [1, 2, 3, 4, 5].index(profile_data.get('Marital status', 1)) if profile_data.get('Marital status', 1) in [1, 2, 3, 4, 5] else 0)),
                    format_func=lambda x: {1: "Casado", 2: "Divorciado", 3: "Separado", 4: "Nunca casado", 5: "Viudo"}.get(x, "Otro")
                )
        
        # Sección 2: Signos Vitales y Biométricos
        with st.expander("🩺 Signos Vitales y Biométricos", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                systolic_bp = st.number_input("Presión Sistólica (mmHg)", min_value=0.0, max_value=300.0, value=float(profile_data.get('Systolic blood pressure', 120.0)), step=1.0)
            with col2:
                diastolic_bp = st.number_input("Presión Diastólica (mmHg)", min_value=0.0, max_value=200.0, value=float(profile_data.get('Diastolic blood pressure', 80.0)), step=1.0)
            with col3:
                waist_circ = st.number_input("Circunferencia de Cintura (cm)", min_value=0.0, max_value=200.0, value=float(profile_data.get('Waist Circumference', 85.0)), step=0.1)
            with col4:
                bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=0.0, max_value=100.0, value=float(profile_data.get('Body Mass Index', 23.0)), step=0.1)
        
        # Sección 3: Laboratorios
        with st.expander("🧪 Laboratorios", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fasting_glucose = st.number_input("Glucosa en Ayunas (mmol/L)", min_value=0.0, max_value=30.0, value=float(profile_data.get('Fasting Glucose', 5.5)), step=0.1)
                glycohemoglobin = st.number_input("Hemoglobina Glicosilada (%)", min_value=0.0, max_value=20.0, value=float(profile_data.get('Glycohemoglobin', 5.0)), step=0.1)
            with col2:
                hdl = st.number_input("HDL - Colesterol Bueno (mmol/L)", min_value=0.0, max_value=10.0, value=float(profile_data.get('High-density lipoprotein', 1.5)), step=0.01)
                ldl = st.number_input("LDL - Colesterol Malo (mmol/L)", min_value=0.0, max_value=10.0, value=float(profile_data.get('Low-density lipoprotein', 2.5)), step=0.01)
            triglyceride = st.number_input("Triglicéridos (mmol/L)", min_value=0.0, max_value=20.0, value=float(profile_data.get('Triglyceride', 1.2)), step=0.01)
        
        # Sección 4: Dieta
        with st.expander("🍽️ Dieta", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                energy = st.number_input("Energía (kcal)", min_value=0.0, max_value=10000.0, value=float(profile_data.get('energy', 2000.0)), step=10.0)
                protein = st.number_input("Proteína (g)", min_value=0.0, max_value=500.0, value=float(profile_data.get('protein', 70.0)), step=1.0)
                carbohydrate = st.number_input("Carbohidratos (g)", min_value=0.0, max_value=1000.0, value=float(profile_data.get('Carbohydrate', 250.0)), step=1.0)
            with col2:
                dietary_fiber = st.number_input("Fibra Dietética (g)", min_value=0.0, max_value=200.0, value=float(profile_data.get('Dietary fiber', 25.0)), step=0.1)
                potassium = st.number_input("Potasio (mg)", min_value=0.0, max_value=10000.0, value=float(profile_data.get('Potassium', 3000.0)), step=10.0)
                sodium = st.number_input("Sodio (mg)", min_value=0.0, max_value=20000.0, value=float(profile_data.get('Sodium', 2500.0)), step=10.0)
            
            # Ácidos grasos
            col3, col4 = st.columns(2)
            with col3:
                total_saturated_fatty = st.number_input("Ácidos Grasos Saturados (g)", min_value=0.0, max_value=200.0, value=float(profile_data.get('Total saturated fatty acids', 20.0)), step=0.1)
                total_monounsaturated_fatty = st.number_input("Ácidos Grasos Monoinsaturados (g)", min_value=0.0, max_value=200.0, value=float(profile_data.get('Total monounsaturated fatty acids', 25.0)), step=0.1)
            with col4:
                total_polyunsaturated_fatty = st.number_input("Ácidos Grasos Poliinsaturados (g)", min_value=0.0, max_value=200.0, value=float(profile_data.get('Total polyunsaturated fatty acids', 15.0)), step=0.1)
        
        # Sección 5: Estilo de Vida
        with st.expander("🏃 Estilo de Vida", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                sleep_time = st.number_input("Tiempo de Sueño (horas)", min_value=0.0, max_value=24.0, value=float(profile_data.get('sleep time', 7.0)), step=0.5)
                sedentary_minutes = st.number_input("Minutos de Actividad Sedentaria", min_value=0.0, max_value=1440.0, value=float(profile_data.get('Minutes sedentary activity', 300.0)), step=10.0)
            with col2:
                alcohol = st.selectbox(
                    "Consumo de Alcohol",
                    options=[0, 1],
                    index=0 if profile_data.get('alcohol', 0) == 0 else 1,
                    format_func=lambda x: "No" if x == 0 else "Sí"
                )
                smoke = st.selectbox(
                    "Fumador",
                    options=[0, 1],
                    index=0 if profile_data.get('smoke', 0) == 0 else 1,
                    format_func=lambda x: "No" if x == 0 else "Sí"
                )
            sleep_disorder = st.selectbox(
                "Trastorno del Sueño",
                options=[1, 2],
                index=0 if profile_data.get('sleep disorder', 1) == 1 else 1,
                format_func=lambda x: "No" if x == 1 else "Sí"
            )
        
        # Sección 6: Condiciones de Salud
        with st.expander("🏥 Condiciones de Salud", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                health_insurance = st.selectbox(
                    "Seguro de Salud",
                    options=[1, 2],
                    index=0 if profile_data.get('Health Insurance', 1) == 1 else 1,
                    format_func=lambda x: "Sí" if x == 1 else "No"
                )
                general_health = st.selectbox(
                    "Condición de Salud General",
                    options=[1, 2, 3, 4],
                    index=max(0, min(3, [1, 2, 3, 4].index(profile_data.get('General health condition', 2)) if profile_data.get('General health condition', 2) in [1, 2, 3, 4] else 1)),
                    format_func=lambda x: {1: "Excelente", 2: "Buena", 3: "Regular", 4: "Mala"}.get(x, "Desconocida")
                )
                depression = st.selectbox(
                    "Depresión",
                    options=[1, 2],
                    index=0 if profile_data.get('depression', 1) == 1 else 1,
                    format_func=lambda x: "No" if x == 1 else "Sí"
                )
            with col2:
                diabetes = st.selectbox(
                    "Diabetes",
                    options=[0, 1],
                    index=0 if profile_data.get('diabetes', 0) == 0 else 1,
                    format_func=lambda x: "No" if x == 0 else "Sí"
                )
                hypertension = st.selectbox(
                    "Hipertensión",
                    options=[0, 1],
                    index=0 if profile_data.get('hypertension', 0) == 0 else 1,
                    format_func=lambda x: "No" if x == 0 else "Sí"
                )
                high_cholesterol = st.selectbox(
                    "Colesterol Alto",
                    options=[0, 1],
                    index=0 if profile_data.get('high cholesterol', 0) == 0 else 1,
                    format_func=lambda x: "No" if x == 0 else "Sí"
                )
            coronary_heart_disease = st.selectbox(
                "Enfermedad Coronaria",
                options=[0, 1],
                index=0 if profile_data.get('Coronary Heart Disease', 0) == 0 else 1,
                format_func=lambda x: "No" if x == 0 else "Sí"
            )
        
        submitted = st.form_submit_button("🔮 Realizar Predicción", type="primary", use_container_width=True)
        
        if submitted:
            if not load_predictor():
                return
            
            try:
                # DEBUG: Mostrar los valores capturados del formulario - SIEMPRE VISIBLE
                st.markdown("---")
                st.markdown("### 🔍 DEBUG - Información de Diagnóstico")
                with st.expander("🔍 Ver valores capturados del formulario", expanded=True):
                    st.write("**Valores capturados del formulario:**")
                    st.write(f"- Edad: {age} (tipo: {type(age).__name__})")
                    st.write(f"- Género: {gender} (tipo: {type(gender).__name__})")
                    st.write(f"- Presión Sistólica: {systolic_bp} (tipo: {type(systolic_bp).__name__})")
                    st.write(f"- Presión Diastólica: {diastolic_bp}")
                    st.write(f"- BMI: {bmi}")
                    st.write(f"- Glucosa: {fasting_glucose}")
                    st.write(f"- Diabetes: {diabetes}")
                    st.write(f"- Hipertensión: {hypertension}")
                    st.write(f"- Colesterol Alto: {high_cholesterol}")
                    st.write(f"- Enfermedad Coronaria: {coronary_heart_disease}")
                
                # Transformar edad continua a categoría (1, 2, o 3)
                age_category = transform_age_to_category(age)
                
                # Crear diccionario con los datos ingresados
                data_dict = {
                    "sleep time": [sleep_time],
                    "Minutes sedentary activity": [sedentary_minutes],
                    "Waist Circumference": [waist_circ],
                    "Systolic blood pressure": [systolic_bp],
                    "Diastolic blood pressure": [diastolic_bp],
                    "High-density lipoprotein": [hdl],
                    "Triglyceride": [triglyceride],
                    "Low-density lipoprotein": [ldl],
                    "Fasting Glucose": [fasting_glucose],
                    "Glycohemoglobin": [glycohemoglobin],
                    "energy": [energy],
                    "protein": [protein],
                    "Dietary fiber": [dietary_fiber],
                    "Potassium": [potassium],
                    "Sodium": [sodium],
                    "gender": [gender],
                    "age": [age_category],  # Usar categoría transformada
                    "Race": [Race],
                    "Marital status": [Marital_status],
                    "alcohol": [alcohol],
                    "smoke": [smoke],
                    "sleep disorder": [sleep_disorder],
                    "Health Insurance": [health_insurance],
                    "General health condition": [general_health],
                    "depression": [depression],
                    "diabetes": [diabetes],
                    "hypertension": [hypertension],
                    "high cholesterol": [high_cholesterol],
                    "Coronary Heart Disease": [coronary_heart_disease],
                    "Body Mass Index": [bmi],
                }
                
                # Crear DataFrame en el orden exacto que espera el modelo
                input_data = pd.DataFrame(data_dict)[MODEL_INPUT_COLUMNS]
                
                # Mostrar datos ANTES del procesamiento
                st.subheader("📊 Datos Antes del Procesamiento")
                with st.expander("Ver datos originales", expanded=True):
                    st.write(f"**Shape:** {input_data.shape[0]} fila(s), {input_data.shape[1]} columna(s)")
                    st.dataframe(input_data, use_container_width=True)
                    
                    # Mostrar estadísticas básicas de algunas columnas clave
                    key_columns = ['age', 'Systolic blood pressure', 'Body Mass Index', 'Fasting Glucose', 
                                  'diabetes', 'hypertension', 'high cholesterol']
                    available_key_cols = [col for col in key_columns if col in input_data.columns]
                    if available_key_cols:
                        st.write("**Estadísticas básicas (columnas clave):**")
                        stats_df = input_data[available_key_cols].describe().T
                        st.dataframe(stats_df, use_container_width=True)
                
                # Realizar procesamiento y predicción
                with st.spinner("Procesando datos y realizando predicción..."):
                    try:
                        result = st.session_state.predictor.predict(input_data)
                        
                        # Verificar si el procesamiento fue exitoso
                        if not result.get('processing_successful', False):
                            st.error("❌ El procesamiento de datos falló. No se puede realizar la predicción.")
                            if 'processing_status' in result and 'errors' in result['processing_status']:
                                for error in result['processing_status']['errors']:
                                    st.error(f"  - {error}")
                            st.stop()
                        
                        # Convertir diccionarios a DataFrames si es necesario
                        data_after_dict = result.get('data_after_processing', {})
                        data_transformed_dict = result.get('data_transformed_with_prefixes', {})
                        
                        if isinstance(data_after_dict, dict):
                            data_after = pd.DataFrame([data_after_dict])
                        else:
                            data_after = data_after_dict if isinstance(data_after_dict, pd.DataFrame) else input_data
                        
                        if isinstance(data_transformed_dict, dict):
                            data_transformed = pd.DataFrame([data_transformed_dict])
                        else:
                            data_transformed = data_transformed_dict if isinstance(data_transformed_dict, pd.DataFrame) else pd.DataFrame()
                        
                        # Mostrar datos TRANSFORMADOS (con prefijos) - INTERMEDIOS
                        if not data_transformed.empty:
                            st.subheader("🔄 Datos Transformados (con prefijos)")
                            with st.expander("Ver datos transformados con prefijos", expanded=True):
                                st.write(f"**Shape:** {data_transformed.shape[0]} fila(s), {data_transformed.shape[1]} columna(s)")
                                st.write("**Nota:** Estos son los datos después del preprocesador, con prefijos `num__` y `cat_num__`")
                                st.dataframe(data_transformed, use_container_width=True)
                                
                                # Mostrar estadísticas de valores normalizados
                                numeric_cols_transformed = [col for col in data_transformed.columns if col.startswith('num__')]
                                if numeric_cols_transformed:
                                    st.write("**Estadísticas de valores normalizados (columnas num__):**")
                                    stats_transformed = data_transformed[numeric_cols_transformed].describe().T
                                    st.dataframe(stats_transformed, use_container_width=True)
                                    
                                    # Validación visual de normalización
                                    max_abs_values = data_transformed[numeric_cols_transformed].abs().max()
                                    values_in_range = (max_abs_values < 10).sum()
                                    total_numeric = len(numeric_cols_transformed)
                                    
                                    if values_in_range >= total_numeric * 0.8:
                                        st.success(f"✅ Normalización correcta: {values_in_range}/{total_numeric} columnas en rango esperado (-10 a 10)")
                                    else:
                                        st.warning(f"⚠️ Normalización sospechosa: Solo {values_in_range}/{total_numeric} columnas en rango esperado")
                        
                        # Mostrar datos DESPUÉS del mapeo (nombres originales)
                        st.subheader("✅ Datos Después del Mapeo (nombres originales)")
                        with st.expander("Ver datos procesados finales", expanded=True):
                            st.write(f"**Shape:** {data_after.shape[0]} fila(s), {data_after.shape[1]} columna(s)")
                            st.write("**Nota:** Estos son los datos después de mapear las columnas de vuelta a nombres originales")
                            st.dataframe(data_after, use_container_width=True)
                            
                            # Mostrar estadísticas básicas de las mismas columnas clave
                            available_key_cols_after = [col for col in key_columns if col in data_after.columns]
                            if available_key_cols_after:
                                st.write("**Estadísticas básicas después del procesamiento (columnas clave):**")
                                stats_df_after = data_after[available_key_cols_after].describe().T
                                st.dataframe(stats_df_after, use_container_width=True)
                        
                        # Tabla comparativa lado a lado
                        st.subheader("📊 Comparación Antes/Después de Transformación")
                        with st.expander("Ver comparación detallada", expanded=True):
                            # Columnas clave para comparar
                            comparison_cols = ['age', 'Systolic blood pressure', 'Diastolic blood pressure', 
                                             'Body Mass Index', 'Fasting Glucose', 'Waist Circumference',
                                             'sleep time', 'Minutes sedentary activity']
                            
                            comparison_data = []
                            for col in comparison_cols:
                                if col in input_data.columns:
                                    before_val = input_data[col].iloc[0] if len(input_data) > 0 else None
                                    
                                    # Buscar valor transformado (con prefijo num__)
                                    transformed_col = f'num__{col}'
                                    transformed_val = None
                                    if not data_transformed.empty and transformed_col in data_transformed.columns:
                                        transformed_val = data_transformed[transformed_col].iloc[0]
                                    
                                    # Buscar valor después del mapeo
                                    after_val = None
                                    if col in data_after.columns:
                                        after_val = data_after[col].iloc[0]
                                    
                                    comparison_data.append({
                                        'Columna': col,
                                        'Antes (Original)': before_val,
                                        'Después (Transformado)': transformed_val if transformed_val is not None else 'N/A',
                                        'Después (Mapeado)': after_val if after_val is not None else 'N/A'
                                    })
                            
                            if comparison_data:
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Mostrar diferencias significativas
                                st.write("**Análisis de transformación:**")
                                for row in comparison_data:
                                    if row['Antes (Original)'] is not None and row['Después (Transformado)'] != 'N/A':
                                        before = float(row['Antes (Original)'])
                                        after = float(row['Después (Transformado)'])
                                        diff = abs(after - before)
                                        if diff > 0.1:  # Si hay diferencia significativa
                                            st.write(f"- **{row['Columna']}**: {before:.2f} → {after:.4f} (diferencia: {diff:.4f})")
                            else:
                                st.info("No hay columnas clave disponibles para comparar")
                        
                        # Mostrar estado del procesamiento
                        st.subheader("🔍 Estado del Procesamiento")
                        processing_status = result.get('processing_status', {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            normalized_icon = "✅" if processing_status.get('normalized', False) else "❌"
                            st.metric("Normalización", normalized_icon)
                        with col2:
                            preprocessed_icon = "✅" if processing_status.get('preprocessed', False) else "❌"
                            st.metric("Preprocesamiento", preprocessed_icon)
                        with col3:
                            cols_processed = processing_status.get('columns_processed', 0)
                            st.metric("Columnas Procesadas", cols_processed)
                        
                        # Mostrar información detallada del procesamiento
                        with st.expander("Ver detalles del procesamiento"):
                            st.write(f"**Método de normalización:** {processing_status.get('normalization_method', 'N/A')}")
                            st.write(f"**Scaler usado:** {'Sí' if processing_status.get('scaler_used', False) else 'No'}")
                            
                            if processing_status.get('warnings'):
                                st.warning("⚠️ Advertencias:")
                                for warning in processing_status['warnings']:
                                    st.write(f"  - {warning}")
                            
                            if processing_status.get('errors'):
                                st.error("❌ Errores:")
                                for error in processing_status['errors']:
                                    st.write(f"  - {error}")
                        
                        # Solo mostrar resultados de predicción si el procesamiento fue exitoso
                        recommendations = get_recommendations(
                            result['prediction'],
                            result['probability'],
                            input_data
                        )
                        
                        st.session_state.prediction_result = result
                        st.session_state.input_data = input_data
                        st.session_state.recommendations = recommendations
                        
                        st.rerun()
                        
                    except ValueError as e:
                        st.error(f"❌ Error en el procesamiento: {str(e)}")
                        st.info("Por favor, verifica los datos ingresados y vuelve a intentar.")
                    except Exception as e:
                        st.error(f"❌ Error inesperado: {str(e)}")
                        st.exception(e)
                    
            except Exception as e:
                st.error(f"❌ Error al realizar la predicción: {str(e)}")
                st.exception(e)


def render_results():
    """Renderiza los resultados de la predicción."""
    if st.session_state.prediction_result is None:
        return
    
    result = st.session_state.prediction_result
    input_data = st.session_state.input_data
    recommendations = st.session_state.get('recommendations', [])
    
    # Mostrar debug si está disponible
    if 'debug_info' in st.session_state and st.session_state.debug_info:
        st.markdown("---")
        st.markdown("### 🔍 DEBUG - Información de Diagnóstico")
        debug = st.session_state.debug_info
        
        with st.expander("🔍 Ver valores capturados del formulario", expanded=False):
            st.write("**Valores capturados del formulario:**")
            for key, value in debug['form_values'].items():
                st.write(f"- {key}: {value} (tipo: {type(value).__name__})")
        
        with st.expander("🔍 Ver DataFrame creado", expanded=False):
            st.write(f"**Total columnas: {debug['dataframe_shape'][1]}**")
            st.write(f"**Shape: {debug['dataframe_shape']}**")
            st.write("**Primeras 15 columnas y valores:**")
            for col, val in debug['dataframe_values'].items():
                st.write(f"  - {col}: {val} (tipo: {type(val).__name__})")
        
        with st.expander("🔍 Ver resultado de la predicción", expanded=False):
            st.write(f"- Predicción: {debug['prediction']}")
            st.write(f"- Probabilidad: {debug['probability']}")
        
        st.markdown("---")
    
    st.header("📊 Resultados de la Predicción")
    st.markdown("---")
    
    # Mostrar resultado principal
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        prediction = result['prediction']
        probability = result['probability']
        
        if prediction == "STROKE RISK":
            st.error(f"## ⚠️ {prediction}")
            risk_color = "🔴"
            risk_level = "ALTO RIESGO"
        else:
            st.success(f"## ✅ {prediction}")
            risk_color = "🟢"
            risk_level = "BAJO RIESGO"
    
    with col2:
        st.metric("Probabilidad", f"{probability:.1%}")
    
    with col3:
        st.metric("Nivel de Riesgo", risk_level)
    
    st.markdown("---")
    
    # Mostrar datos ingresados
    with st.expander("📋 Ver Datos Ingresados", expanded=False):
        st.dataframe(input_data, use_container_width=True)
    
    # Mostrar datos transformados si están disponibles
    data_transformed_dict = result.get('data_transformed_with_prefixes', {})
    data_after_dict = result.get('data_after_processing', {})
    
    if data_transformed_dict or data_after_dict:
        st.markdown("---")
        st.subheader("🔄 Datos Transformados")
        
        if data_transformed_dict:
            with st.expander("📊 Datos Transformados (con prefijos)", expanded=False):
                if isinstance(data_transformed_dict, dict):
                    data_transformed = pd.DataFrame([data_transformed_dict])
                else:
                    data_transformed = data_transformed_dict if isinstance(data_transformed_dict, pd.DataFrame) else pd.DataFrame()
                
                if not data_transformed.empty:
                    st.write("**Datos después del preprocesador (con prefijos `num__` y `cat_num__`):**")
                    st.dataframe(data_transformed, use_container_width=True)
                    
                    # Mostrar valores normalizados clave
                    numeric_cols = [col for col in data_transformed.columns if col.startswith('num__')]
                    if numeric_cols:
                        st.write("**Valores normalizados (columnas num__):**")
                        key_transformed_cols = [col for col in numeric_cols if any(kc in col for kc in ['age', 'Systolic', 'Diastolic', 'BMI', 'Glucose', 'Waist'])]
                        if key_transformed_cols:
                            for col in key_transformed_cols[:10]:  # Primeras 10
                                val = data_transformed[col].iloc[0]
                                st.write(f"- {col}: {val:.4f}")
        
        if data_after_dict:
            with st.expander("✅ Datos Después del Mapeo (nombres originales)", expanded=False):
                if isinstance(data_after_dict, dict):
                    data_after = pd.DataFrame([data_after_dict])
                else:
                    data_after = data_after_dict if isinstance(data_after_dict, pd.DataFrame) else pd.DataFrame()
                
                if not data_after.empty:
                    st.write("**Datos después de mapear columnas a nombres originales:**")
                    st.dataframe(data_after, use_container_width=True)
                    
                    # Comparación de valores clave con todas las columnas numéricas continuas
                    st.write("**Comparación de valores (Antes vs Después de Transformación):**")
                    
                    # Identificar columnas numéricas continuas que deben estar normalizadas
                    continuous_numeric_cols = [
                        'sleep time', 'Minutes sedentary activity', 'Waist Circumference',
                        'Systolic blood pressure', 'Diastolic blood pressure',
                        'High-density lipoprotein', 'Triglyceride', 'Low-density lipoprotein',
                        'Fasting Glucose', 'Glycohemoglobin',
                        'energy', 'protein', 'Carbohydrate', 'Dietary fiber',
                        'Total saturated fatty acids', 'Total monounsaturated fatty acids',
                        'Total polyunsaturated fatty acids', 'Potassium', 'Sodium'
                    ]
                    
                    comparison_rows = []
                    for col in continuous_numeric_cols:
                        if col in input_data.columns and col in data_after.columns:
                            before = input_data[col].iloc[0] if len(input_data) > 0 else None
                            after = data_after[col].iloc[0] if len(data_after) > 0 else None
                            
                            if before is not None and after is not None:
                                diff = abs(float(before) - float(after))
                                is_normalized = (-3 <= float(after) <= 3)
                                
                                comparison_rows.append({
                                    'Columna': col,
                                    'Antes (Original)': f"{float(before):.2f}",
                                    'Después (Normalizado)': f"{float(after):.4f}",
                                    'Diferencia': f"{diff:.4f}",
                                    '¿Normalizado?': "✅" if is_normalized else "❌"
                                })
                    
                    if comparison_rows:
                        comparison_df = pd.DataFrame(comparison_rows)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Resumen de normalización
                        normalized_count = sum(1 for row in comparison_rows if row['¿Normalizado?'] == "✅")
                        total_count = len(comparison_rows)
                        st.write(f"**Resumen:** {normalized_count}/{total_count} columnas están normalizadas correctamente (rango -3 a 3)")
                    else:
                        st.info("No se encontraron columnas numéricas continuas para comparar.")
    
    # Mostrar recomendaciones
    st.subheader("💡 Recomendaciones")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    st.markdown("---")
    
    # Generar y descargar PDF
    st.subheader("📄 Generar Reporte PDF")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📥 Generar y Descargar PDF", type="primary", use_container_width=True):
            try:
                with st.spinner("Generando reporte PDF..."):
                    # Crear nombre de archivo único
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = DATA_OUTPUTS / f"reporte_acv_{timestamp}.pdf"
                    
                    # Asegurar que el directorio existe
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Generar reporte
                    report_generator = ReportGenerator()
                    pdf_path = report_generator.generate_report(
                        result,
                        input_data,
                        output_path,
                        recommendations
                    )
                    
                    # Leer archivo y ofrecer descarga
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="⬇️ Descargar PDF",
                            data=pdf_file.read(),
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                    
                    st.success(f"✅ Reporte generado: {pdf_path.name}")
                    
            except Exception as e:
                st.error(f"❌ Error al generar el reporte: {str(e)}")
                st.exception(e)
    
    with col2:
        if st.button("🔄 Nueva Predicción", use_container_width=True):
            st.session_state.prediction_result = None
            st.session_state.input_data = None
            st.session_state.recommendations = None
            st.rerun()


def main():
    """Función principal de la aplicación web."""
    # Título principal
    st.title("🏥 Predictor de Riesgo de ACV")
    st.markdown("Sistema de predicción de riesgo de Accidente Cerebrovascular basado en Machine Learning")
    st.markdown("---")
    
    # Renderizar sidebar
    render_sidebar()
    
    # Cargar predictor al inicio
    if st.session_state.predictor is None:
        load_predictor()
    
    # Si hay resultados, mostrarlos
    if st.session_state.prediction_result is not None:
        render_results()
    else:
        # Tabs para diferentes métodos de entrada
        tab1, tab2 = st.tabs(["📁 Cargar Archivo", "✍️ Formulario Manual"])
        
        with tab1:
            render_file_upload()
        
        with tab2:
            render_manual_form()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        <small>
        ⚠️ Este sistema es una herramienta de apoyo. No reemplaza la consulta médica profesional.<br>
        Para uso médico real, consulte siempre con un profesional de la salud.
        </small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
