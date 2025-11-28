# Guía de Compilación - ACV Risk Predictor

## ✅ Estado Actual

**Todas las configuraciones están listas para compilar la aplicación a .exe**

## Archivos Creados

1. **`ACV_Risk_Predictor.spec`**: Configuración de PyInstaller
   - Incluye todos los módulos necesarios
   - Configurado para aplicación GUI (sin consola)
   - Incluye modelos y recursos

2. **`build_exe.py`**: Script automatizado de compilación
   - Verifica requisitos
   - Instala PyInstaller si falta
   - Ejecuta la compilación

3. **`check_build_requirements.py`**: Verificador de requisitos
   - Comprueba que PyInstaller esté instalado
   - Verifica dependencias críticas

4. **`installer_script.iss`**: Script de InnoSetup actualizado
   - Listo para crear instalador después de compilar

## Cómo Compilar

### Paso 1: Verificar Requisitos

```bash
python app_desktop/check_build_requirements.py
```

### Paso 2: Compilar

**Opción A - Script Automático (Recomendado):**
```bash
cd app_desktop
python build_exe.py
```

**Opción B - Manual:**
```bash
cd app_desktop
python -m PyInstaller ACV_Risk_Predictor.spec
```

### Paso 3: Probar el Ejecutable

El ejecutable estará en: `app_desktop/dist/ACV_Risk_Predictor.exe`

Ejecútalo y verifica que todo funciona.

### Paso 4: Crear Instalador (Opcional)

1. Instala InnoSetup: https://jrsoftware.org/isdl.php
2. Abre `app_desktop/installer_script.iss`
3. Compila el instalador
4. El instalador estará en `app_desktop/dist/ACV_Risk_Predictor_Setup.exe`

## Notas Importantes

- ⏱️ **Tiempo de compilación**: Puede tardar 5-15 minutos
- 📦 **Tamaño del ejecutable**: ~100-200 MB (incluye Python y dependencias)
- 🔧 **Modo debug**: Si hay problemas, cambia `console=False` a `console=True` en el .spec
- 📁 **Modelos**: Asegúrate de tener modelos en `models/` o la app usará modo MOCK

## Solución de Problemas

### Error: "No module named 'X'"
- Agrega el módulo a `hiddenimports` en el `.spec`
- Recompila con `--clean`

### El ejecutable no se abre
- Cambia `console=True` temporalmente para ver errores
- Verifica que todos los recursos estén incluidos

### El ejecutable es muy grande
- Esto es normal, incluye Python completo
- UPX está habilitado para comprimir

## Próximos Pasos

Después de compilar exitosamente:
1. ✅ Probar todas las funcionalidades
2. ✅ Crear instalador con InnoSetup
3. ✅ Distribuir la aplicación

