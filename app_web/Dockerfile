# Dockerfile para la aplicación web Streamlit
# Optimizado para Streamlit Cloud (sin PyCaret - el modelo se carga con joblib)

FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para scikit-learn y pandas
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias de Python
# Instalar primero numpy y scipy (requeridos por scikit-learn)
COPY app_web/requirements.txt .
RUN pip install --no-cache-dir \
    numpy>=1.21.0,<1.26.0 \
    scipy>=1.10.0 \
    && pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear directorio para modelos si no existe
RUN mkdir -p models

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app_web/main_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
