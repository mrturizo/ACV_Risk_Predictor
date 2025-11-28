# Guía de Pruebas - Interfaz Web

## Pasos para Probar la Aplicación Web

### 1. Verificar que todo esté instalado

```bash
# Verificar que Streamlit esté instalado
python -c "import streamlit; print('OK')"

# Verificar que las dependencias estén instaladas
python -c "from core import StrokePredictor, ReportGenerator; print('OK')"
```

### 2. Opciones para el Modelo

#### Opción A: Usar Modo MOCK (Sin PyCaret) - RECOMENDADO PARA PRUEBAS RÁPIDAS

La aplicación ahora incluye un predictor MOCK que funciona sin PyCaret. 
**Puedes probar la interfaz web inmediatamente sin instalar nada más.**

La aplicación detectará automáticamente si PyCaret no está disponible y usará el modo MOCK.

#### Opción B: Instalar PyCaret y Generar Modelo Dummy

Si quieres usar el modelo real (requiere PyCaret):

```bash
# Instalar PyCaret (puede tardar varios minutos)
pip install pycaret[full]
```

**Nota**: Si tienes problemas con rutas largas en Windows, puedes:
1. Habilitar Long Path support en Windows
2. O usar el modo MOCK que no requiere PyCaret

Luego genera el modelo dummy:

```bash
python ml_models/scripts/train_dummy.py
```

Esto creará un modelo en `models/dummy_stroke_model.pkl`

### 3. Ejecutar la Aplicación Web

```bash
streamlit run app_web/main_streamlit.py
```

Esto abrirá automáticamente tu navegador en `http://localhost:8501`

### 4. Probar la Aplicación

#### Opción A: Cargar Archivo

1. En la pestaña "📁 Cargar Archivo"
2. Haz clic en "Browse files" o arrastra un archivo
3. Usa el archivo de ejemplo: `data/ejemplo_paciente.csv`
4. Selecciona la fila que quieres analizar (si hay múltiples)
5. Haz clic en "🔮 Realizar Predicción"
6. Verás los resultados y podrás descargar el PDF

#### Opción B: Formulario Manual

1. En la pestaña "✍️ Formulario Manual"
2. Completa todos los campos:
   - **Datos Demográficos**: Edad, Género, Estado civil, etc.
   - **Datos Clínicos**: Hipertensión, Enfermedad cardíaca, Glucosa, BMI, etc.
3. Haz clic en "🔮 Realizar Predicción"
4. Verás los resultados y podrás descargar el PDF

### 5. Verificar Funcionalidades

✅ **Carga de archivos**: CSV, Excel, JSON
✅ **Formulario manual**: Todos los campos funcionan
✅ **Predicción**: Muestra resultado (STROKE RISK / NOT STROKE RISK)
✅ **Probabilidades**: Muestra porcentaje de riesgo
✅ **Recomendaciones**: Lista personalizada según el resultado
✅ **Exportación PDF**: Genera y descarga reporte completo

### 6. Solución de Problemas

#### Error: "No se encontró ningún modelo .pkl"
**Solución**: La aplicación usará automáticamente el modo MOCK. Si quieres el modelo real, ejecuta `python ml_models/scripts/train_dummy.py`

#### Error: "PyCaret no está instalado"
**Solución**: La aplicación usará automáticamente el modo MOCK. Si quieres PyCaret, instala con `pip install pycaret[full]`

#### Error de rutas largas en Windows al instalar PyCaret
**Solución**: 
- Opción 1: Usa el modo MOCK (no requiere PyCaret)
- Opción 2: Habilita Long Path support en Windows (ver: https://pip.pypa.io/warnings/enable-long-paths)

#### Modo MOCK activado
**Nota**: Si ves "⚠️ Modo MOCK" en el sidebar, estás usando el predictor simulado. 
Esto es perfecto para probar la interfaz, pero las predicciones NO son reales.

#### Error al cargar archivo
**Solución**: Verifica que el archivo tenga las columnas correctas:
- age, gender, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status

#### La aplicación no se abre
**Solución**: Verifica que el puerto 8501 no esté en uso, o especifica otro:
```bash
streamlit run app_web/main_streamlit.py --server.port 8502
```

### 7. Estructura de Datos Esperada

El modelo dummy espera estas columnas:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| age | int | Edad (0-120) |
| gender | int | 0=Femenino, 1=Masculino |
| hypertension | int | 0=No, 1=Sí |
| heart_disease | int | 0=No, 1=Sí |
| ever_married | int | 0=No, 1=Sí |
| work_type | int | 0-4 (Niño, Gubernamental, Nunca trabajó, Privado, Autónomo) |
| Residence_type | int | 0=Rural, 1=Urbana |
| avg_glucose_level | float | Nivel promedio de glucosa |
| bmi | float | Índice de Masa Corporal |
| smoking_status | int | 0-3 (Desconocido, Fumador, Nunca fumó, Ex-fumador) |

### 8. Archivos de Ejemplo

- `data/ejemplo_paciente.csv`: Archivo CSV con datos de ejemplo
- Puedes crear tus propios archivos en formato CSV, Excel o JSON

## Notas Importantes

⚠️ **El modelo dummy es solo para pruebas**. No debe usarse para predicciones médicas reales.

⚠️ **Este sistema es una herramienta de apoyo**. Siempre consulte con un profesional de la salud.

✅ **La aplicación guarda archivos temporales** en:
- `data/uploads/`: Archivos cargados
- `data/outputs/`: Reportes PDF generados

