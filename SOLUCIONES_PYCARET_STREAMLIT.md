# Soluciones para el Problema de PyCaret en Streamlit Cloud

## Problema
El modelo `lr_pca25_cw.pkl` fue entrenado con PyCaret y al deserializarlo, pickle intenta importar módulos de PyCaret que no están instalados, causando el error:
```
ModuleNotFoundError: No module named 'pycaret.internal.preprocess'; 'pycaret.internal' is not a package
```

## Soluciones (Probar una por una)

### ✅ Solución 1: Mocks Mejorados + Import Hook (YA APLICADA)

**Estado**: Ya implementada en `core/predictor.py`

**Qué hace**:
- Crea módulos mock completos de PyCaret (incluyendo `pycaret.internal.preprocess`)
- Hace que `pycaret.internal` sea reconocido como paquete (agrega `__path__`)
- Instala un import hook para interceptar importaciones durante pickle.load

**Cómo probar**:
1. Hacer commit y push:
```powershell
git add core/predictor.py
git commit -m "Fix: Mejorar mocks de PyCaret con preprocess e import hook"
git push origin main
```

2. Verificar en Streamlit Cloud que el modelo se carga correctamente.

**Si esta solución NO funciona**, probar Solución 2.

---

### 🔄 Solución 2: Convertir Modelo a Sklearn Puro

**Estado**: Script creado en `ml_models/scripts/convert_pycaret_to_sklearn.py`

**Qué hace**:
- Carga el modelo PyCaret localmente (donde PyCaret está instalado)
- Extrae el Pipeline de sklearn subyacente
- Guarda un nuevo modelo `.pkl` que es sklearn puro (sin dependencias de PyCaret)

**Cómo usar**:

1. **Ejecutar el script localmente** (donde tienes PyCaret instalado):
```powershell
python ml_models/scripts/convert_pycaret_to_sklearn.py
```

2. Esto creará `models/lr_pca25_cw_sklearn.pkl`

3. **Actualizar el código para usar el modelo convertido**:
   - Modificar `core/predictor.py` o `app_web/main_streamlit.py` para buscar `lr_pca25_cw_sklearn.pkl` primero
   - O reemplazar `lr_pca25_cw.pkl` con el convertido

4. Hacer commit y push del nuevo modelo y cambios

**Ventajas**:
- ✅ Elimina completamente la dependencia de PyCaret
- ✅ Modelo más pequeño y rápido de cargar
- ✅ Sin problemas de importación

**Desventajas**:
- ⚠️ Requiere ejecutar el script localmente primero
- ⚠️ Necesitas tener PyCaret instalado localmente para ejecutar el script

---

### 🔄 Solución 3: Instalar PyCaret Completo (Último Recurso)

**Estado**: Archivo `app_web/requirements_minimal_pycaret.txt` creado

**Qué hace**:
- Instala PyCaret completo en Streamlit Cloud
- Resuelve todos los problemas de importación

**Cómo usar**:

1. **Renombrar el archivo de requirements**:
```powershell
# Backup del requirements actual
mv app_web/requirements.txt app_web/requirements_no_pycaret.txt

# Usar el requirements con PyCaret
mv app_web/requirements_minimal_pycaret.txt app_web/requirements.txt
```

2. Hacer commit y push:
```powershell
git add app_web/requirements.txt
git commit -m "Fix: Agregar PyCaret completo para resolver problemas de importación"
git push origin main
```

**Ventajas**:
- ✅ Garantiza que todos los módulos de PyCaret estén disponibles
- ✅ Solución más simple

**Desventajas**:
- ⚠️ Instalación MUY lenta (puede tomar 10-15 minutos)
- ⚠️ Imagen Docker más grande
- ⚠️ Puede causar conflictos de versiones (numpy, scikit-learn, sktime)

---

## Recomendación de Orden de Prueba

1. **PRIMERO**: Probar Solución 1 (ya aplicada) - hacer push y verificar
2. **SEGUNDO**: Si Solución 1 falla, usar Solución 2 (convertir modelo)
3. **ÚLTIMO RECURSO**: Si las anteriores fallan, usar Solución 3 (instalar PyCaret completo)

---

## Verificación

Después de aplicar cualquier solución, verificar en Streamlit Cloud:

1. ✅ El modelo se carga sin errores
2. ✅ No aparece "MOCK MODE"
3. ✅ Las predicciones funcionan correctamente
4. ✅ Los logs no muestran errores de importación de PyCaret

---

## Notas Técnicas

- El modelo `lr_pca25_cw.pkl` es un Pipeline de sklearn que fue serializado con PyCaret
- Una vez cargado, el Pipeline funciona igual que cualquier Pipeline de sklearn
- PyCaret solo se necesita para **entrenar** el modelo, no para **usarlo**
- Los mocks permiten que pickle deserialice el modelo sin importar PyCaret realmente

