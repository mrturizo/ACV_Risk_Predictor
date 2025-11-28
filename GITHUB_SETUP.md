# Guía para Subir el Repositorio a GitHub

## Pasos para Subir el Código a GitHub

### 1. Inicializar Repositorio Git (si no está inicializado)

```bash
git init
```

### 2. Agregar Todos los Archivos

```bash
git add .
```

### 3. Hacer el Primer Commit

```bash
git commit -m "Initial commit: ACV Risk Predictor - Web and Desktop app"
```

### 4. Crear Repositorio en GitHub

1. Ve a [GitHub](https://github.com)
2. Haz clic en el botón "+" en la esquina superior derecha
3. Selecciona "New repository"
4. Configura:
   - **Repository name**: `ACV_Risk_Predictor` (o el nombre que prefieras)
   - **Description**: "Aplicación híbrida para predicción de riesgo de ACV usando Machine Learning"
   - **Visibility**: Público o Privado (según tu preferencia)
   - **NO** marques "Initialize this repository with a README" (ya tenemos uno)
5. Haz clic en "Create repository"

### 5. Conectar y Subir el Código

GitHub te mostrará comandos similares a estos. Reemplaza `<TU-USUARIO>` con tu nombre de usuario de GitHub:

```bash
git remote add origin https://github.com/<TU-USUARIO>/ACV_Risk_Predictor.git
git branch -M main
git push -u origin main
```

### 6. Verificar

Ve a tu repositorio en GitHub y verifica que todos los archivos estén presentes.

## Notas Importantes

- **No subir modelos grandes**: Los archivos `.pkl` grandes están excluidos por `.gitignore`
- **No subir datos sensibles**: Los datos de usuarios están excluidos
- **Verificar antes de hacer push**: Revisa `git status` para ver qué se va a subir

## Siguiente Paso: Deploy en Streamlit Cloud

Una vez que el código esté en GitHub, puedes hacer deploy en Streamlit Cloud siguiendo las instrucciones en el README.

