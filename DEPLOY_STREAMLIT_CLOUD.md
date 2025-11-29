# Guía de Deploy en Streamlit Cloud

Esta guía describe cómo desplegar la aplicación ACV Risk Predictor en Streamlit Cloud.

## Requisitos Previos

1. Cuenta en [Streamlit Cloud](https://streamlit.io/cloud)
2. Repositorio en GitHub con el código de la aplicación
3. El modelo `lr_pca25_cw.pkl` debe estar en el directorio `models/` del repositorio

## Configuración del Repositorio

### Estructura de Archivos

Asegúrate de que tu repositorio tenga la siguiente estructura:

```
ACV_Risk_Predictor/
├── app_web/
│   ├── main_streamlit.py
│   ├── requirements.txt
│   └── Dockerfile
├── core/
│   ├── __init__.py
│   ├── predictor.py
│   ├── utils.py
│   └── ...
├── models/
│   └── lr_pca25_cw.pkl
├── .streamlit/
│   └── config.toml
├── .dockerignore
└── README.md
```

### Archivos Importantes

1. **`app_web/requirements.txt`**: Contiene todas las dependencias de Python necesarias
2. **`app_web/Dockerfile`**: Configuración de Docker para el contenedor (opcional, Streamlit Cloud puede usar solo requirements.txt)
3. **`.streamlit/config.toml`**: Configuración de Streamlit (puerto, tema, etc.)
4. **`.dockerignore`**: Excluye archivos innecesarios del build de Docker

## Pasos para Deploy

### 1. Preparar el Repositorio

1. Asegúrate de que todos los cambios estén commiteados y pusheados a GitHub:
   ```bash
   git add .
   git commit -m "Preparar para deploy en Streamlit Cloud"
   git push origin main
   ```

2. Verifica que el modelo `models/lr_pca25_cw.pkl` esté en el repositorio (no en .gitignore)

### 2. Conectar con Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Haz clic en "New app"

### 3. Configurar la Aplicación

En el formulario de creación de app:

- **Repository**: Selecciona tu repositorio `ACV_Risk_Predictor`
- **Branch**: `main` (o la rama que uses)
- **Main file path**: `app_web/main_streamlit.py`
- **App URL**: Elige un nombre único para tu app (ej: `acv-risk-predictor`)

### 4. Configuración Avanzada (Opcional)

Si necesitas usar Docker (recomendado para PyCaret):

1. En "Advanced settings", habilita "Use Docker"
2. Streamlit Cloud detectará automáticamente el `Dockerfile` en `app_web/`

### 5. Deploy

1. Haz clic en "Deploy!"
2. Streamlit Cloud comenzará a construir y desplegar tu aplicación
3. El proceso puede tardar varios minutos, especialmente la primera vez (instalación de PyCaret)

## Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'pycaret'"

**Solución**: Verifica que `pycaret>=3.0.0` esté en `app_web/requirements.txt` y que el archivo esté en la ubicación correcta.

### Error: "FileNotFoundError: models/lr_pca25_cw.pkl"

**Solución**: 
1. Verifica que el modelo esté en el repositorio
2. Asegúrate de que `models/` no esté en `.gitignore`
3. Si el modelo es muy grande (>100MB), considera usar Git LFS

### Error: "pandas version conflict"

**Solución**: PyCaret requiere `pandas<2.2.0`. Asegúrate de que `app_web/requirements.txt` tenga:
```
pandas>=1.5.0,<2.2.0
```

### Error: "Build failed - compilation error"

**Solución**: 
1. Usa el Dockerfile proporcionado que incluye herramientas de compilación (gcc, g++)
2. O asegúrate de que las versiones en requirements.txt tengan wheels precompilados

### La app se carga pero muestra "MOCK MODE"

**Solución**: 
1. Verifica que el modelo esté en `models/lr_pca25_cw.pkl`
2. Revisa los logs de Streamlit Cloud para ver errores de carga del modelo
3. Asegúrate de que `joblib` esté en requirements.txt

## Verificación Post-Deploy

1. **Cargar el modelo**: Verifica que la app cargue el modelo correctamente (no debe estar en MOCK mode)
2. **Formulario manual**: Prueba ingresar datos y hacer una predicción
3. **Carga de archivo**: Prueba cargar un archivo CSV/Excel y hacer predicción
4. **Generación de PDF**: Verifica que se pueda generar el reporte PDF

## Actualización de la Aplicación

Para actualizar la aplicación después de hacer cambios:

1. Haz commit y push de los cambios a GitHub
2. Streamlit Cloud detectará automáticamente los cambios y redeployará
3. O puedes hacer "Reboot app" manualmente desde el dashboard

## URL de la Aplicación

Una vez desplegada, tu aplicación estará disponible en:
```
https://[tu-nombre-de-app].streamlit.app
```

## Notas Importantes

- **Primera vez**: El deploy inicial puede tardar 5-10 minutos debido a la instalación de PyCaret y sus dependencias
- **Modelo**: El modelo `lr_pca25_cw.pkl` debe estar en el repositorio (no en .gitignore)
- **Docker**: El Dockerfile está optimizado para instalar PyCaret correctamente con todas sus dependencias de sistema
- **Límites**: Streamlit Cloud tiene límites de recursos. Si tu app es muy pesada, considera optimizaciones

## Recursos Adicionales

- [Documentación de Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [Troubleshooting Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud/troubleshooting)

