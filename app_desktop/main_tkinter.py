"""Aplicación de escritorio Tkinter para predicción de riesgo de ACV."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from typing import Tuple
import sys
import pandas as pd
from datetime import datetime
import traceback

# Agregar el directorio raíz al path para importar core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    StrokePredictor,
    ReportGenerator,
    load_data_file,
    get_recommendations,
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

# Importar utilidades de escritorio
from app_desktop.utils_desktop import resource_path, get_models_dir


class StrokeApp(tk.Tk):
    """Aplicación principal de escritorio para predicción de ACV."""
    
    def __init__(self):
        """Inicializa la aplicación."""
        super().__init__()
        
        # Configuración de la ventana
        self.title("ACV Risk Predictor - Predicción de Riesgo de Accidente Cerebrovascular")
        self.geometry("1000x800")
        self.minsize(900, 700)
        
        # Configurar icono de la ventana (usar siempre icon.ico en la raíz del bundle)
        try:
            icon_path = resource_path("icon.ico")
            if Path(icon_path).exists():
                self.iconbitmap(str(icon_path))
        except Exception:
            # Si falla, seguir sin icono personalizado
            pass
        
        # Configurar tema y colores profesionales
        self.configure(bg='#f5f5f5')
        
        # Colores del tema médico
        self.colors = {
            'primary': '#1f77b4',      # Azul médico
            'secondary': '#2ca02c',    # Verde médico
            'accent': '#ff7f0e',       # Naranja de alerta
            'background': '#f5f5f5',   # Gris claro
            'surface': '#ffffff',      # Blanco
            'text': '#262730',         # Gris oscuro
            'text_light': '#6c757d',  # Gris medio
            'success': '#28a745',      # Verde éxito
            'warning': '#ffc107',      # Amarillo advertencia
            'danger': '#dc3545'        # Rojo peligro
        }
        
        # Variables de estado
        self.predictor = None
        self.prediction_result = None
        self.input_data = None
        self.recommendations = []
        
        # Variables de formulario - Todas las 35 variables NHANES (30 del dataset procesado + 5 nutricionales adicionales)
        # Demográficas
        self.age_var = tk.StringVar(value="50")
        self.gender_var = tk.StringVar(value="1")
        self.race_var = tk.StringVar(value="3")
        self.marital_status_var = tk.StringVar(value="1")
        
        # Biomédicas
        self.sleep_time_var = tk.StringVar(value="7.0")
        self.sedentary_minutes_var = tk.StringVar(value="300.0")
        self.waist_circ_var = tk.StringVar(value="85.0")
        self.systolic_bp_var = tk.StringVar(value="120.0")
        self.diastolic_bp_var = tk.StringVar(value="80.0")
        self.hdl_var = tk.StringVar(value="1.5")
        self.triglyceride_var = tk.StringVar(value="1.2")
        self.ldl_var = tk.StringVar(value="2.5")
        self.fasting_glucose_var = tk.StringVar(value="5.5")
        self.glycohemoglobin_var = tk.StringVar(value="5.0")
        self.bmi_var = tk.StringVar(value="23.0")
        
        # Dietéticas
        self.energy_var = tk.StringVar(value="2000.0")
        self.protein_var = tk.StringVar(value="70.0")
        self.carbohydrate_var = tk.StringVar(value="250.0")
        self.dietary_fiber_var = tk.StringVar(value="25.0")
        self.total_saturated_fatty_var = tk.StringVar(value="20.0")
        self.total_monounsaturated_fatty_var = tk.StringVar(value="25.0")
        self.total_polyunsaturated_fatty_var = tk.StringVar(value="15.0")
        self.potassium_var = tk.StringVar(value="3000.0")
        self.sodium_var = tk.StringVar(value="2500.0")
        
        # Estilo de vida
        self.alcohol_var = tk.StringVar(value="0")
        self.smoke_var = tk.StringVar(value="0")
        self.sleep_disorder_var = tk.StringVar(value="1")
        
        # Condiciones de salud
        self.health_insurance_var = tk.StringVar(value="1")
        self.general_health_var = tk.StringVar(value="2")
        self.depression_var = tk.StringVar(value="1")
        self.diabetes_var = tk.StringVar(value="0")
        self.hypertension_var = tk.StringVar(value="0")
        self.high_cholesterol_var = tk.StringVar(value="0")
        self.coronary_heart_disease_var = tk.StringVar(value="0")
        
        # Cargar predictor
        self.load_predictor()
        
        # Crear interfaz
        self.create_widgets()
        
        # Centrar ventana
        self.center_window()
    
    def center_window(self):
        """Centra la ventana en la pantalla."""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def load_predictor(self):
        """Carga el predictor de modelos."""
        try:
            # Intentar encontrar modelo usando resource_path para compatibilidad con PyInstaller
            models_dir = get_models_dir()
            # Priorizar el modelo final lr_pca25_cw.pkl
            preferred = models_dir / "lr_pca25_cw.pkl"
            if preferred.exists():
                model_file = preferred
            else:
                model_files = list(models_dir.glob("*.pkl"))
                if not model_files:
                    model_file = None
                else:
                    model_file = model_files[0]
            
            if model_file is not None:
                # Usar resource_path para asegurar compatibilidad con .exe
                relative_path = f"models/{model_file.name}"
                model_path = resource_path(relative_path)
                
                # Si el modelo no existe en la ruta de resource_path, usar la ruta encontrada
                if not model_path.exists():
                    model_path = model_file
                
                self.predictor = StrokePredictor(model_path=model_path)
            else:
                # Si no hay modelo, usar MOCK si está disponible
                if MOCK_AVAILABLE:
                    self.predictor = StrokePredictorMock()
                else:
                    raise FileNotFoundError("No se encontró ningún modelo")
                    
        except (FileNotFoundError, ImportError) as e:
            # Usar MOCK como fallback
            if MOCK_AVAILABLE:
                self.predictor = StrokePredictorMock()
            else:
                messagebox.showerror(
                    "Error",
                    f"No se pudo cargar el modelo:\n{str(e)}\n\n"
                    "La aplicación usará modo MOCK para pruebas."
                )
                # Crear predictor mock manualmente si no está disponible
                self.predictor = None
    
    def create_widgets(self):
        """Crea todos los widgets de la interfaz."""
        # Frame principal
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(
            main_frame,
            text="🏥 Predictor de Riesgo de ACV",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(
            main_frame,
            text="Sistema de predicción basado en Machine Learning",
            font=("Arial", 10)
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Pestaña 1: Formulario Manual
        self.form_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.form_frame, text="✍️ Formulario Manual")
        self.create_form_tab()
        
        # Pestaña 2: Cargar Archivo
        self.file_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.file_frame, text="📁 Cargar Archivo")
        self.create_file_tab()
        
        # Pestaña 3: Resultados
        self.results_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.results_frame, text="📊 Resultados")
        self.create_results_tab()
        
        # Barra de estado
        self.status_bar = ttk.Label(
            main_frame,
            text="Listo",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def create_form_tab(self):
        """Crea la pestaña del formulario manual con todas las variables NHANES."""
        # Frame para botones de perfiles
        profile_frame = ttk.Frame(self.form_frame)
        profile_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(profile_frame, text="⚡ Perfiles de Llenado Rápido:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            profile_frame,
            text="🟢 Paciente Sano",
            command=lambda: self.load_profile('sano')
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            profile_frame,
            text="🟡 Factores de Riesgo",
            command=lambda: self.load_profile('factores_riesgo')
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            profile_frame,
            text="🔴 Múltiples Comorbilidades",
            command=lambda: self.load_profile('comorbilidades')
        ).pack(side=tk.LEFT, padx=5)
        
        # Frame con scrollbar para el formulario
        canvas_frame = ttk.Frame(self.form_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Contenedor principal del formulario
        form_container = ttk.Frame(scrollable_frame)
        form_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        row = 0
        
        # Sección 1: Datos Demográficos
        demo_frame = ttk.LabelFrame(form_container, text="📋 Datos Demográficos", padding="10")
        demo_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        ttk.Label(demo_frame, text="Edad:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(demo_frame, textvariable=self.age_var, width=20).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(demo_frame, text="Género:").grid(row=1, column=0, sticky=tk.W, pady=5)
        gender_combo = ttk.Combobox(demo_frame, textvariable=self.gender_var, values=["1", "2"], state="readonly", width=17)
        gender_combo.grid(row=1, column=1, pady=5, padx=5)
        gender_combo.set("1")
        
        ttk.Label(demo_frame, text="Raza:").grid(row=2, column=0, sticky=tk.W, pady=5)
        race_combo = ttk.Combobox(demo_frame, textvariable=self.race_var, values=["1", "2", "3", "4", "5"], state="readonly", width=17)
        race_combo.grid(row=2, column=1, pady=5, padx=5)
        race_combo.set("3")
        
        ttk.Label(demo_frame, text="Estado Civil:").grid(row=3, column=0, sticky=tk.W, pady=5)
        marital_combo = ttk.Combobox(demo_frame, textvariable=self.marital_status_var, values=["1", "2", "3", "4", "5"], state="readonly", width=17)
        marital_combo.grid(row=3, column=1, pady=5, padx=5)
        marital_combo.set("1")
        
        # Sección 2: Signos Vitales y Biométricos
        vital_frame = ttk.LabelFrame(form_container, text="🩺 Signos Vitales y Biométricos", padding="10")
        vital_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        ttk.Label(vital_frame, text="Presión Sistólica (mmHg):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(vital_frame, textvariable=self.systolic_bp_var, width=20).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(vital_frame, text="Presión Diastólica (mmHg):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(vital_frame, textvariable=self.diastolic_bp_var, width=20).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(vital_frame, text="Circunferencia de Cintura (cm):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(vital_frame, textvariable=self.waist_circ_var, width=20).grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(vital_frame, text="BMI:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(vital_frame, textvariable=self.bmi_var, width=20).grid(row=3, column=1, pady=5, padx=5)
        
        # Sección 3: Laboratorios
        lab_frame = ttk.LabelFrame(form_container, text="🧪 Laboratorios", padding="10")
        lab_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        ttk.Label(lab_frame, text="Glucosa en Ayunas (mmol/L):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(lab_frame, textvariable=self.fasting_glucose_var, width=20).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(lab_frame, text="Hemoglobina Glicosilada (%):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(lab_frame, textvariable=self.glycohemoglobin_var, width=20).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(lab_frame, text="HDL (mmol/L):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(lab_frame, textvariable=self.hdl_var, width=20).grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(lab_frame, text="LDL (mmol/L):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(lab_frame, textvariable=self.ldl_var, width=20).grid(row=3, column=1, pady=5, padx=5)
        
        ttk.Label(lab_frame, text="Triglicéridos (mmol/L):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(lab_frame, textvariable=self.triglyceride_var, width=20).grid(row=4, column=1, pady=5, padx=5)
        
        # Sección 4: Dieta
        diet_frame = ttk.LabelFrame(form_container, text="🍽️ Dieta", padding="10")
        diet_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        ttk.Label(diet_frame, text="Energía (kcal):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.energy_var, width=20).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Proteína (g):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.protein_var, width=20).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Carbohidratos (g):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.carbohydrate_var, width=20).grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Fibra Dietética (g):").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.dietary_fiber_var, width=20).grid(row=3, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Ác. Grasos Saturados (g):").grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.total_saturated_fatty_var, width=20).grid(row=4, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Ác. Grasos Monoinsat. (g):").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.total_monounsaturated_fatty_var, width=20).grid(row=5, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Ác. Grasos Poliinsat. (g):").grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.total_polyunsaturated_fatty_var, width=20).grid(row=6, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Potasio (mg):").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.potassium_var, width=20).grid(row=7, column=1, pady=5, padx=5)
        
        ttk.Label(diet_frame, text="Sodio (mg):").grid(row=8, column=0, sticky=tk.W, pady=5)
        ttk.Entry(diet_frame, textvariable=self.sodium_var, width=20).grid(row=8, column=1, pady=5, padx=5)
        
        # Sección 5: Estilo de Vida
        lifestyle_frame = ttk.LabelFrame(form_container, text="🏃 Estilo de Vida", padding="10")
        lifestyle_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        ttk.Label(lifestyle_frame, text="Tiempo de Sueño (horas):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(lifestyle_frame, textvariable=self.sleep_time_var, width=20).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(lifestyle_frame, text="Minutos Actividad Sedentaria:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(lifestyle_frame, textvariable=self.sedentary_minutes_var, width=20).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(lifestyle_frame, text="Consumo de Alcohol:").grid(row=2, column=0, sticky=tk.W, pady=5)
        alcohol_combo = ttk.Combobox(lifestyle_frame, textvariable=self.alcohol_var, values=["0", "1"], state="readonly", width=17)
        alcohol_combo.grid(row=2, column=1, pady=5, padx=5)
        alcohol_combo.set("0")
        
        ttk.Label(lifestyle_frame, text="Fumador:").grid(row=3, column=0, sticky=tk.W, pady=5)
        smoke_combo = ttk.Combobox(lifestyle_frame, textvariable=self.smoke_var, values=["0", "1"], state="readonly", width=17)
        smoke_combo.grid(row=3, column=1, pady=5, padx=5)
        smoke_combo.set("0")
        
        ttk.Label(lifestyle_frame, text="Trastorno del Sueño:").grid(row=4, column=0, sticky=tk.W, pady=5)
        sleep_disorder_combo = ttk.Combobox(lifestyle_frame, textvariable=self.sleep_disorder_var, values=["1", "2"], state="readonly", width=17)
        sleep_disorder_combo.grid(row=4, column=1, pady=5, padx=5)
        sleep_disorder_combo.set("1")
        
        # Sección 6: Condiciones de Salud
        health_frame = ttk.LabelFrame(form_container, text="🏥 Condiciones de Salud", padding="10")
        health_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5, padx=5)
        row += 1
        
        ttk.Label(health_frame, text="Seguro de Salud:").grid(row=0, column=0, sticky=tk.W, pady=5)
        insurance_combo = ttk.Combobox(health_frame, textvariable=self.health_insurance_var, values=["1", "2"], state="readonly", width=17)
        insurance_combo.grid(row=0, column=1, pady=5, padx=5)
        insurance_combo.set("1")
        
        ttk.Label(health_frame, text="Condición de Salud General:").grid(row=1, column=0, sticky=tk.W, pady=5)
        general_health_combo = ttk.Combobox(health_frame, textvariable=self.general_health_var, values=["1", "2", "3", "4"], state="readonly", width=17)
        general_health_combo.grid(row=1, column=1, pady=5, padx=5)
        general_health_combo.set("2")
        
        ttk.Label(health_frame, text="Depresión:").grid(row=2, column=0, sticky=tk.W, pady=5)
        depression_combo = ttk.Combobox(health_frame, textvariable=self.depression_var, values=["1", "2"], state="readonly", width=17)
        depression_combo.grid(row=2, column=1, pady=5, padx=5)
        depression_combo.set("1")
        
        ttk.Label(health_frame, text="Diabetes:").grid(row=3, column=0, sticky=tk.W, pady=5)
        diabetes_combo = ttk.Combobox(health_frame, textvariable=self.diabetes_var, values=["0", "1"], state="readonly", width=17)
        diabetes_combo.grid(row=3, column=1, pady=5, padx=5)
        diabetes_combo.set("0")
        
        ttk.Label(health_frame, text="Hipertensión:").grid(row=4, column=0, sticky=tk.W, pady=5)
        hypertension_combo = ttk.Combobox(health_frame, textvariable=self.hypertension_var, values=["0", "1"], state="readonly", width=17)
        hypertension_combo.grid(row=4, column=1, pady=5, padx=5)
        hypertension_combo.set("0")
        
        ttk.Label(health_frame, text="Colesterol Alto:").grid(row=5, column=0, sticky=tk.W, pady=5)
        cholesterol_combo = ttk.Combobox(health_frame, textvariable=self.high_cholesterol_var, values=["0", "1"], state="readonly", width=17)
        cholesterol_combo.grid(row=5, column=1, pady=5, padx=5)
        cholesterol_combo.set("0")
        
        ttk.Label(health_frame, text="Enfermedad Coronaria:").grid(row=6, column=0, sticky=tk.W, pady=5)
        coronary_combo = ttk.Combobox(health_frame, textvariable=self.coronary_heart_disease_var, values=["0", "1"], state="readonly", width=17)
        coronary_combo.grid(row=6, column=1, pady=5, padx=5)
        coronary_combo.set("0")
        
        # Botón de predicción
        predict_btn = ttk.Button(
            form_container,
            text="🔮 Realizar Predicción",
            command=self.predict_from_form,
            style="Accent.TButton"
        )
        predict_btn.grid(row=row, column=0, columnspan=2, pady=20)
        
        # Configurar grid weights
        form_container.columnconfigure(0, weight=1)
        form_container.columnconfigure(1, weight=1)
    
    def load_profile(self, profile_name: str):
        """Carga un perfil y llena todos los campos del formulario.
        
        Args:
            profile_name: Nombre del perfil a cargar.
        """
        try:
            profile_data = get_profile(profile_name)
            
            # Llenar todas las variables
            self.age_var.set(str(profile_data.get('age', 50)))
            self.gender_var.set(str(profile_data.get('gender', 1)))
            self.race_var.set(str(profile_data.get('Race', 3)))
            self.marital_status_var.set(str(profile_data.get('Marital status', 1)))
            
            self.sleep_time_var.set(str(profile_data.get('sleep time', 7.0)))
            self.sedentary_minutes_var.set(str(profile_data.get('Minutes sedentary activity', 300.0)))
            self.waist_circ_var.set(str(profile_data.get('Waist Circumference', 85.0)))
            self.systolic_bp_var.set(str(profile_data.get('Systolic blood pressure', 120.0)))
            self.diastolic_bp_var.set(str(profile_data.get('Diastolic blood pressure', 80.0)))
            self.hdl_var.set(str(profile_data.get('High-density lipoprotein', 1.5)))
            self.triglyceride_var.set(str(profile_data.get('Triglyceride', 1.2)))
            self.ldl_var.set(str(profile_data.get('Low-density lipoprotein', 2.5)))
            self.fasting_glucose_var.set(str(profile_data.get('Fasting Glucose', 5.5)))
            self.glycohemoglobin_var.set(str(profile_data.get('Glycohemoglobin', 5.0)))
            self.bmi_var.set(str(profile_data.get('Body Mass Index', 23.0)))
            
            self.energy_var.set(str(profile_data.get('energy', 2000.0)))
            self.protein_var.set(str(profile_data.get('protein', 70.0)))
            self.carbohydrate_var.set(str(profile_data.get('Carbohydrate', 250.0)))
            self.dietary_fiber_var.set(str(profile_data.get('Dietary fiber', 25.0)))
            self.total_saturated_fatty_var.set(str(profile_data.get('Total saturated fatty acids', 20.0)))
            self.total_monounsaturated_fatty_var.set(str(profile_data.get('Total monounsaturated fatty acids', 25.0)))
            self.total_polyunsaturated_fatty_var.set(str(profile_data.get('Total polyunsaturated fatty acids', 15.0)))
            self.potassium_var.set(str(profile_data.get('Potassium', 3000.0)))
            self.sodium_var.set(str(profile_data.get('Sodium', 2500.0)))
            
            self.alcohol_var.set(str(profile_data.get('alcohol', 0)))
            self.smoke_var.set(str(profile_data.get('smoke', 0)))
            self.sleep_disorder_var.set(str(profile_data.get('sleep disorder', 1)))
            
            self.health_insurance_var.set(str(profile_data.get('Health Insurance', 1)))
            self.general_health_var.set(str(profile_data.get('General health condition', 2)))
            self.depression_var.set(str(profile_data.get('depression', 1)))
            self.diabetes_var.set(str(profile_data.get('diabetes', 0)))
            self.hypertension_var.set(str(profile_data.get('hypertension', 0)))
            self.high_cholesterol_var.set(str(profile_data.get('high cholesterol', 0)))
            self.coronary_heart_disease_var.set(str(profile_data.get('Coronary Heart Disease', 0)))
            
            self.status_bar.config(text=f"Perfil '{profile_name}' cargado exitosamente")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el perfil: {str(e)}")
    
    def update_gender_label(self, combo):
        """Actualiza la etiqueta del género."""
        value = combo.get()
        self.gender_label.config(text="Femenino" if value == "0" else "Masculino")
    
    def create_file_tab(self):
        """Crea la pestaña de carga de archivos."""
        # Frame para carga
        load_frame = ttk.Frame(self.file_frame)
        load_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            load_frame,
            text="Selecciona un archivo (CSV, Excel, JSON)",
            font=("Arial", 12)
        ).pack(pady=10)
        
        ttk.Button(
            load_frame,
            text="📁 Seleccionar Archivo",
            command=self.load_file
        ).pack(pady=10)
        
        # Área de preview
        preview_label = ttk.Label(load_frame, text="Vista previa:", font=("Arial", 10, "bold"))
        preview_label.pack(pady=(20, 5))
        
        # Treeview para mostrar datos
        self.file_tree = ttk.Treeview(load_frame, height=10)
        self.file_tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Scrollbar para treeview
        scrollbar = ttk.Scrollbar(load_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        # Botón para predecir desde archivo
        self.predict_file_btn = ttk.Button(
            load_frame,
            text="🔮 Realizar Predicción",
            command=self.predict_from_file,
            state=tk.DISABLED
        )
        self.predict_file_btn.pack(pady=10)
        
        # Variable para almacenar datos cargados
        self.loaded_file_data = None
        self.selected_row_index = 0
    
    def create_results_tab(self):
        """Crea la pestaña de resultados."""
        # Frame principal de resultados
        results_main = ttk.Frame(self.results_frame)
        results_main.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(
            results_main,
            text="📊 Resultados de la Predicción",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # Frame para resultado principal
        result_frame = ttk.LabelFrame(results_main, text="Predicción", padding="10")
        result_frame.pack(fill=tk.X, pady=10, padx=10)
        
        self.prediction_label = ttk.Label(
            result_frame,
            text="No hay predicción realizada",
            font=("Arial", 12)
        )
        self.prediction_label.pack()
        
        self.probability_label = ttk.Label(
            result_frame,
            text="",
            font=("Arial", 10)
        )
        self.probability_label.pack()
        
        # Frame para recomendaciones
        rec_frame = ttk.LabelFrame(results_main, text="💡 Recomendaciones", padding="10")
        rec_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.recommendations_text = scrolledtext.ScrolledText(
            rec_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        self.recommendations_text.pack(fill=tk.BOTH, expand=True)
        
        # Botones de acción
        button_frame = ttk.Frame(results_main)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame,
            text="📥 Generar PDF",
            command=self.generate_pdf
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="🔄 Nueva Predicción",
            command=self.clear_results
        ).pack(side=tk.LEFT, padx=5)
    
    def validate_form_data(self) -> Tuple[bool, dict]:
        """Valida los datos del formulario con todas las 35 variables NHANES.
        
        Returns:
            Tupla (es_válido, datos_validados)
        """
        try:
            # Validar y convertir todos los campos
            age = int(self.age_var.get())
            if not (0 <= age <= 120):
                raise ValueError("La edad debe estar entre 0 y 120 años")
            
            # Crear diccionario de datos con nombres exactos del modelo NHANES (30 columnas)
            data = {
                # Biomédicas
                "sleep time": float(self.sleep_time_var.get()),
                "Minutes sedentary activity": float(self.sedentary_minutes_var.get()),
                "Waist Circumference": float(self.waist_circ_var.get()),
                "Systolic blood pressure": float(self.systolic_bp_var.get()),
                "Diastolic blood pressure": float(self.diastolic_bp_var.get()),
                "High-density lipoprotein": float(self.hdl_var.get()),
                "Triglyceride": float(self.triglyceride_var.get()),
                "Low-density lipoprotein": float(self.ldl_var.get()),
                "Fasting Glucose": float(self.fasting_glucose_var.get()),
                "Glycohemoglobin": float(self.glycohemoglobin_var.get()),
                "Body Mass Index": float(self.bmi_var.get()),

                # Dietéticas
                "energy": float(self.energy_var.get()),
                "protein": float(self.protein_var.get()),
                "Dietary fiber": float(self.dietary_fiber_var.get()),
                "Potassium": float(self.potassium_var.get()),
                "Sodium": float(self.sodium_var.get()),

                # Demográficas
                "gender": int(self.gender_var.get()),
                "age": age,
                "Race": int(self.race_var.get()),
                "Marital status": int(self.marital_status_var.get()),

                # Estilo de vida
                "alcohol": int(self.alcohol_var.get()),
                "smoke": int(self.smoke_var.get()),
                "sleep disorder": int(self.sleep_disorder_var.get()),

                # Salud y condiciones
                "Health Insurance": int(self.health_insurance_var.get()),
                "General health condition": int(self.general_health_var.get()),
                "depression": int(self.depression_var.get()),
                "diabetes": int(self.diabetes_var.get()),
                "hypertension": int(self.hypertension_var.get()),
                "high cholesterol": int(self.high_cholesterol_var.get()),
                "Coronary Heart Disease": int(self.coronary_heart_disease_var.get()),
            }
            
            return True, data
            
        except ValueError as e:
            messagebox.showerror("Error de Validación", f"Datos inválidos:\n{str(e)}")
            return False, {}
        except Exception as e:
            messagebox.showerror("Error", f"Error al validar datos:\n{str(e)}")
            return False, {}
    
    def predict_from_form(self):
        """Realiza predicción desde el formulario."""
        if self.predictor is None:
            messagebox.showerror("Error", "El predictor no está cargado.")
            return
        
        # Validar datos
        is_valid, data_dict = self.validate_form_data()
        if not is_valid:
            return
        
        try:
            self.status_bar.config(text="Realizando predicción...")
            self.update()
            
            # Crear DataFrame y reordenar al orden exacto que espera el modelo
            input_data = pd.DataFrame([data_dict])
            # Asegurar que las columnas estén en el orden correcto
            if set(MODEL_INPUT_COLUMNS) <= set(input_data.columns):
                input_data = input_data[MODEL_INPUT_COLUMNS]
            else:
                # Si faltan columnas, agregarlas con 0
                missing = set(MODEL_INPUT_COLUMNS) - set(input_data.columns)
                for col in missing:
                    input_data[col] = 0.0
                input_data = input_data[MODEL_INPUT_COLUMNS]
            
            # Realizar predicción
            result = self.predictor.predict(input_data)
            recommendations = get_recommendations(
                result['prediction'],
                result['probability'],
                input_data
            )
            
            # Guardar resultados
            self.prediction_result = result
            self.input_data = input_data
            self.recommendations = recommendations
            
            # Mostrar resultados
            self.display_results()
            
            # Cambiar a pestaña de resultados
            self.notebook.select(2)
            
            self.status_bar.config(text="Predicción completada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar la predicción:\n{str(e)}")
            self.status_bar.config(text="Error en la predicción")
            traceback.print_exc()
    
    def load_file(self):
        """Carga un archivo de datos."""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo",
            filetypes=[
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("JSON", "*.json"),
                ("Todos", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            self.status_bar.config(text="Cargando archivo...")
            self.update()
            
            # Cargar archivo
            data = load_data_file(Path(file_path))
            
            if data.empty:
                messagebox.showerror("Error", "El archivo está vacío.")
                return
            
            # Guardar datos
            self.loaded_file_data = data
            self.selected_row_index = 0
            
            # Mostrar preview
            self.display_file_preview(data)
            
            # Habilitar botón de predicción
            self.predict_file_btn.config(state=tk.NORMAL)
            
            self.status_bar.config(text=f"Archivo cargado: {len(data)} filas")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el archivo:\n{str(e)}")
            self.status_bar.config(text="Error al cargar archivo")
            traceback.print_exc()
    
    def display_file_preview(self, data: pd.DataFrame):
        """Muestra preview del archivo cargado."""
        # Limpiar treeview
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Configurar columnas
        columns = list(data.columns)
        self.file_tree["columns"] = columns
        self.file_tree["show"] = "headings"
        
        # Configurar encabezados
        for col in columns:
            self.file_tree.heading(col, text=col)
            self.file_tree.column(col, width=100)
        
        # Insertar datos (máximo 50 filas para preview)
        for idx, row in data.head(50).iterrows():
            values = [str(val) for val in row.values]
            self.file_tree.insert("", tk.END, values=values)
        
        if len(data) > 50:
            self.status_bar.config(
                text=f"Mostrando 50 de {len(data)} filas. Selecciona una fila para analizar."
            )
    
    def predict_from_file(self):
        """Realiza predicción desde archivo cargado."""
        if self.loaded_file_data is None:
            messagebox.showerror("Error", "No hay archivo cargado.")
            return
        
        if self.predictor is None:
            messagebox.showerror("Error", "El predictor no está cargado.")
            return
        
        # Obtener fila seleccionada
        selected = self.file_tree.selection()
        if selected:
            # Obtener índice de la fila seleccionada
            item = self.file_tree.item(selected[0])
            # Encontrar el índice real en el DataFrame
            # Por ahora, usar la primera fila
            row_data = self.loaded_file_data.iloc[[0]]
        
        # Asegurar que sólo se usen las columnas definidas en el contrato del modelo
        try:
            row_data = row_data[MODEL_INPUT_COLUMNS]
        except KeyError:
            # Si faltan columnas, se gestionará en el predictor
            pass
        else:
            # Usar primera fila por defecto
            row_data = self.loaded_file_data.iloc[[0]]
        
        try:
            self.status_bar.config(text="Realizando predicción...")
            self.update()
            
            # Realizar predicción
            result = self.predictor.predict(row_data)
            recommendations = get_recommendations(
                result['prediction'],
                result['probability'],
                row_data
            )
            
            # Guardar resultados
            self.prediction_result = result
            self.input_data = row_data
            self.recommendations = recommendations
            
            # Mostrar resultados
            self.display_results()
            
            # Cambiar a pestaña de resultados
            self.notebook.select(2)
            
            self.status_bar.config(text="Predicción completada")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar la predicción:\n{str(e)}")
            self.status_bar.config(text="Error en la predicción")
            traceback.print_exc()
    
    def display_results(self):
        """Muestra los resultados de la predicción."""
        if self.prediction_result is None:
            return
        
        result = self.prediction_result
        prediction = result['prediction']
        probability = result['probability']
        
        # Actualizar etiqueta de predicción
        if prediction == "STROKE RISK":
            self.prediction_label.config(
                text=f"⚠️ {prediction}",
                foreground="red"
            )
            risk_level = "ALTO RIESGO"
        else:
            self.prediction_label.config(
                text=f"✅ {prediction}",
                foreground="green"
            )
            risk_level = "BAJO RIESGO"
        
        # Actualizar probabilidad
        self.probability_label.config(
            text=f"Probabilidad: {probability:.1%} | Nivel de Riesgo: {risk_level}"
        )
        
        # Actualizar recomendaciones
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        
        for i, rec in enumerate(self.recommendations, 1):
            self.recommendations_text.insert(tk.END, f"{i}. {rec}\n\n")
        
        self.recommendations_text.config(state=tk.DISABLED)
    
    def generate_pdf(self):
        """Genera y guarda el reporte PDF."""
        if self.prediction_result is None or self.input_data is None:
            messagebox.showerror("Error", "No hay resultados para generar el reporte.")
            return
        
        # Solicitar ubicación para guardar
        file_path = filedialog.asksaveasfilename(
            title="Guardar Reporte PDF",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf"), ("Todos", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.status_bar.config(text="Generando reporte PDF...")
            self.update()
            
            # Generar reporte
            report_generator = ReportGenerator()
            pdf_path = report_generator.generate_report(
                self.prediction_result,
                self.input_data,
                Path(file_path),
                self.recommendations
            )
            
            messagebox.showinfo(
                "Éxito",
                f"Reporte PDF generado exitosamente:\n{pdf_path}"
            )
            
            self.status_bar.config(text="Reporte PDF generado")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar el PDF:\n{str(e)}")
            self.status_bar.config(text="Error al generar PDF")
            traceback.print_exc()
    
    def clear_results(self):
        """Limpia los resultados y vuelve al formulario."""
        self.prediction_result = None
        self.input_data = None
        self.recommendations = []
        
        self.prediction_label.config(text="No hay predicción realizada", foreground="black")
        self.probability_label.config(text="")
        
        self.recommendations_text.config(state=tk.NORMAL)
        self.recommendations_text.delete(1.0, tk.END)
        self.recommendations_text.config(state=tk.DISABLED)
        
        # Volver a pestaña de formulario
        self.notebook.select(0)
        
        self.status_bar.config(text="Listo")


def main():
    """Función principal."""
    app = StrokeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
