# Instrucciones para Compilar la Aplicación de Escritorio

## Requisitos Previos

1. **PyInstaller**: Debe estar instalado
   ```bash
   pip install pyinstaller
   ```

2. **Modelo**: Opcional, pero recomendado tener un modelo en `models/`
   - Si no hay modelo, la app usará modo MOCK

## Compilación con PyInstaller

### Opción 1: Usar el Script Automático (Recomendado)

```bash
cd app_desktop
python build_exe.py
```

Este script:
- Verifica que PyInstaller esté instalado
- Lo instala automáticamente si no está
- Ejecuta la compilación
- Muestra el resultado

### Opción 2: Compilación Manual

```bash
cd app_desktop
pyinstaller ACV_Risk_Predictor.spec
```

## Resultado

Después de la compilación, encontrarás:

- **Ejecutable**: `app_desktop/dist/ACV_Risk_Predictor.exe`
- **Archivos temporales**: `app_desktop/build/` (puedes eliminarlos)

## Probar el Ejecutable

1. Navega a `app_desktop/dist/`
2. Ejecuta `ACV_Risk_Predictor.exe`
3. Verifica que:
   - La aplicación se abre correctamente
   - El formulario funciona
   - La carga de archivos funciona
   - Las predicciones funcionan
   - La generación de PDF funciona

## Crear Instalador con InnoSetup

Una vez que el ejecutable funciona:

1. Instala InnoSetup desde: https://jrsoftware.org/isdl.php
2. Abre `app_desktop/installer_script.iss` en InnoSetup
3. Compila el instalador (Build > Compile)
4. El instalador estará en `app_desktop/dist/ACV_Risk_Predictor_Setup.exe`

## Solución de Problemas

### Error: "No se encuentra el modelo"
- Asegúrate de que hay un archivo `.pkl` en `models/`
- O la app usará modo MOCK automáticamente

### Error: "Módulo no encontrado"
- Verifica que todas las dependencias estén en `hiddenimports` del `.spec`
- Recompila con `--clean`: `pyinstaller --clean ACV_Risk_Predictor.spec`

### El ejecutable es muy grande
- Esto es normal, PyInstaller incluye Python y todas las librerías
- Puedes usar UPX para comprimir (ya está habilitado en el .spec)

### La aplicación no se abre
- Verifica que no haya errores en la consola (cambia `console=False` a `console=True` temporalmente)
- Revisa los logs en `app_desktop/dist/`

## Notas Importantes

- El ejecutable es independiente, no requiere Python instalado
- Los modelos deben estar en `models/` para que se incluyan
- El tamaño del ejecutable puede ser grande (100-200 MB) debido a las dependencias
- Para distribución, usa el instalador de InnoSetup

