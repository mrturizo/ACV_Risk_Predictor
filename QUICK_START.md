# Guía de Inicio Rápido

## Pasos Iniciales

### 1. Verificar Estructura
```bash
python verify_structure.py
```

### 2. Instalar Dependencias (sin PyCaret por ahora)
```bash
pip install -r requirements.txt
```

**Nota**: PyCaret se instalará más adelante cuando sea necesario.

### 3. Generar Modelo Dummy (cuando PyCaret esté instalado)
```bash
python ml_models/scripts/train_dummy.py
```

Esto creará un modelo de prueba en `models/dummy_stroke_model.pkl`

## Próximos Pasos de Desarrollo

### Fase 1: Completar Core (Lógica de Negocio)
- [ ] Implementar `core/predictor.py` completamente
- [ ] Implementar `core/reports.py` (generación de PDFs)
- [ ] Completar `core/utils.py` (validaciones)

### Fase 2: Desarrollo de Modelos
- [ ] Realizar EDA en `ml_models/notebooks/`
- [ ] Implementar `ml_models/scripts/train_models.py`
- [ ] Implementar `ml_models/scripts/grid_search.py`
- [ ] Entrenar modelos reales con datos

### Fase 3: Interfaz Web
- [ ] Implementar `app_web/main_streamlit.py`
- [ ] Agregar carga de archivos (CSV, Excel, JSON)
- [ ] Agregar formulario manual de entrada
- [ ] Integrar visualización de resultados

### Fase 4: Interfaz Escritorio
- [ ] Implementar `app_desktop/main_tkinter.py`
- [ ] Agregar funcionalidades de UI
- [ ] Preparar para compilación con PyInstaller
- [ ] Crear instalador con InnoSetup

## Comandos Útiles

### Ejecutar Aplicación Web
```bash
streamlit run app_web/main_streamlit.py
```

### Ejecutar Aplicación Escritorio
```bash
python app_desktop/main_tkinter.py
```

### Verificar Estructura
```bash
python verify_structure.py
```

## Archivos Importantes

- **`.cursorrules`**: Reglas de desarrollo y arquitectura
- **`config.py`**: Configuración centralizada
- **`requirements.txt`**: Dependencias del proyecto
- **`ml_models/README.md`**: Guía de desarrollo de modelos

