# ML Models - Desarrollo de Modelos

Esta carpeta contiene todo lo relacionado con el desarrollo, entrenamiento y almacenamiento de modelos de Machine Learning para la predicción de riesgo de ACV usando el dataset NHANES.

## Estructura

```
ml_models/
├── scripts/                      # Scripts de entrenamiento y evaluación
│   ├── preprocessing.py         # Módulo de preprocesamiento
│   ├── preprocess_data.py       # Script ejecutable de preprocesamiento
│   ├── data_split.py            # División de datos estratificada
│   ├── train_sklearn_models.py # Entrenamiento con sklearn (manual)
│   ├── train_models.py          # Entrenamiento con PyCaret (automático)
│   ├── evaluate_models.py      # Evaluación y visualizaciones
│   ├── select_best_model.py     # Selección del mejor modelo
│   ├── inference_pipeline.py    # Pipeline de inferencia
│   └── train_dummy.py          # Genera modelo dummy para pruebas
├── trained_models/              # Modelos entrenados
│   ├── preprocessor.pkl        # Preprocesador entrenado
│   ├── results/                # Resultados y métricas
│   └── *.pkl                   # Modelos entrenados
├── data/                        # Datos para entrenamiento
│   ├── raw/                    # Datos crudos (NHANES)
│   │   └── nhanes_stroke_raw.csv
│   ├── processed/              # Datos procesados
│   │   └── nhanes_stroke_processed.csv
│   └── splits/                 # Train/Val/Test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── notebooks/                   # Jupyter notebooks
    ├── EDA_ACV_initial.ipynb   # EDA inicial (referencia)
    └── EDA_completo.ipynb      # EDA completo y robusto
```

## Pipeline Completo de ML

### 1. Análisis Exploratorio de Datos (EDA)

**Notebook:** `notebooks/EDA_completo.ipynb`

El EDA completo incluye:
- Análisis de calidad de datos (valores faltantes, duplicados, outliers)
- Análisis univariado (distribuciones, estadísticas descriptivas)
- Análisis bivariado (correlaciones, relaciones con target)
- Análisis multivariado (PCA exploratorio, multicolinealidad)
- Análisis específico para datos clínicos (factores de riesgo)
- Conclusiones y recomendaciones para preprocesamiento

**Ejecutar:**
```bash
jupyter notebook notebooks/EDA_completo.ipynb
```

### 2. Preprocesamiento de Datos

**Script:** `scripts/preprocess_data.py`

Aplica todo el pipeline de preprocesamiento:
- Manejo de valores faltantes (imputación)
- Detección y manejo de outliers
- Eliminación de variables altamente correlacionadas
- Encoding de variables categóricas
- Normalización/Estandarización
- Guarda datos procesados y preprocesador

**Ejecutar:**
```bash
python ml_models/scripts/preprocess_data.py
```

**Módulo:** `scripts/preprocessing.py` contiene todas las funciones de preprocesamiento reutilizables.

### 3. División de Datos

**Módulo:** `scripts/data_split.py`

Divide los datos de forma estratificada:
- 65% entrenamiento
- 20% validación
- 15% test

Mantiene la proporción de la variable target en cada split.

**Uso:**
```python
from ml_models.scripts.data_split import stratified_train_val_test_split, save_splits

train_df, val_df, test_df = stratified_train_val_test_split(df, target_col='stroke')
save_splits(train_df, val_df, test_df, splits_dir)
```

### 4. Entrenamiento de Modelos

#### 4.1 Pipeline con Scikit-Learn (Manual)

**Script:** `scripts/train_sklearn_models.py`

Entrena múltiples modelos con:
- Validación cruzada estratificada (5-fold)
- Grid Search / Randomized Search para ajuste de hiperparámetros
- Manejo de desbalance (SMOTE, ADASYN, Undersampling, SMOTEENN)
- Evaluación exhaustiva en train, val y test

**Modelos entrenados:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- SVM
- Naive Bayes
- K-Nearest Neighbors
- Neural Network (MLP)

**Ejecutar:**
```bash
python ml_models/scripts/train_sklearn_models.py
```

#### 4.2 Pipeline con PyCaret (Automático)

**Script:** `scripts/train_models.py`

Pipeline automático con PyCaret:
- Comparación automática de todos los modelos disponibles
- Selección de top N modelos
- Ajuste automático de hiperparámetros
- Ensemble methods (blend, stack)
- Manejo automático de desbalance

**Ejecutar:**
```bash
python ml_models/scripts/train_models.py
```

### 5. Evaluación de Modelos

**Módulo:** `scripts/evaluate_models.py`

Proporciona funciones para:
- Calcular métricas completas (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
- Visualizar matrices de confusión
- Comparar curvas ROC
- Comparar curvas Precision-Recall
- Generar reportes comparativos en HTML

**Uso:**
```python
from ml_models.scripts.evaluate_models import (
    calculate_metrics, plot_roc_curves, 
    plot_precision_recall_curves, generate_evaluation_report
)
```

### 6. Selección del Mejor Modelo

**Script:** `scripts/select_best_model.py`

Selecciona el mejor modelo basado en métricas clínicas:
- Prioriza Recall (Sensitivity) - no perder casos de riesgo real
- Considera Specificity - evitar falsos positivos
- Evalúa F1-Score, ROC-AUC y PR-AUC
- Calcula score ponderado para ranking
- Copia el mejor modelo a `models/` para producción

**Ejecutar:**
```bash
python ml_models/scripts/select_best_model.py
```

### 7. Pipeline de Inferencia

**Módulo:** `scripts/inference_pipeline.py`

Funciones para hacer predicciones en producción:
- `load_production_model()`: Carga modelo de producción
- `preprocess_new_data()`: Aplica preprocesamiento a datos nuevos
- `predict()`: Realiza predicción individual
- `predict_batch()`: Predicción en lote

**Uso:**
```python
from ml_models.scripts.inference_pipeline import predict_batch

predictions = predict_batch(new_data)
```

## Flujo de Trabajo Completo

### Orden de Ejecución Recomendado:

1. **EDA**: Ejecutar `notebooks/EDA_completo.ipynb` para entender los datos
2. **Preprocesamiento**: `python ml_models/scripts/preprocess_data.py`
3. **División**: Crear splits (puede estar integrado en preprocesamiento)
4. **Entrenamiento sklearn**: `python ml_models/scripts/train_sklearn_models.py`
5. **Entrenamiento PyCaret**: `python ml_models/scripts/train_models.py`
6. **Selección**: `python ml_models/scripts/select_best_model.py`
7. **Inferencia**: Usar `inference_pipeline.py` en producción

## Métricas Clínicas Prioritarias

Para datos clínicos de predicción de riesgo de ACV, las métricas más importantes son:

1. **Recall/Sensitivity** (30% peso): No perder casos de riesgo real
2. **Specificity** (20% peso): Evitar falsos positivos
3. **ROC-AUC** (20% peso): Capacidad discriminativa general
4. **F1-Score** (15% peso): Balance entre precision y recall
5. **PR-AUC** (15% peso): Especialmente importante con desbalance

## Manejo de Desbalance

El pipeline incluye múltiples técnicas de balanceo:
- **SMOTE**: Synthetic Minority Oversampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **RandomUnderSampler**: Submuestreo aleatorio
- **SMOTEENN**: Combinación SMOTE + Edited Nearest Neighbours
- **class_weight='balanced'**: En modelos que lo soporten

## Validación

- **Stratified K-Fold CV** (k=5): Para evaluación robusta
- **División estratificada**: Mantiene proporción de target
- **Conjuntos separados**: Train (65%), Val (20%), Test (15%)
- **Nunca usar test durante entrenamiento**

## Reproducibilidad

- Semilla aleatoria fija: `RANDOM_STATE = 42`
- Splits guardados para reproducibilidad
- Preprocesador guardado para aplicar en inferencia
- Hiperparámetros documentados en resultados

## Modelo Dummy

El modelo dummy (`train_dummy.py`) genera un modelo simple para pruebas del sistema sin necesidad de datos reales. Se usa para validar la arquitectura antes de entrenar modelos reales.

**Ejecutar:**
```bash
python ml_models/scripts/train_dummy.py
```

## Dependencias

Ver `requirements.txt` en la raíz del proyecto. Principales:
- pandas, numpy
- scikit-learn
- pycaret[full]
- imbalanced-learn (para técnicas de balanceo)
- matplotlib, seaborn (para visualizaciones)

## Notas Importantes

- **Preprocesamiento**: El mismo preprocesamiento aplicado en entrenamiento debe aplicarse en inferencia
- **Modelos**: Los modelos se guardan en `trained_models/` y el mejor se copia a `models/` para producción
- **Datos**: Los datos crudos deben estar en `data/raw/` antes de ejecutar el pipeline
- **Resultados**: Todos los resultados y métricas se guardan en `trained_models/results/`

