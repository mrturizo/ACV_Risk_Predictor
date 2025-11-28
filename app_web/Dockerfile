# Dockerfile para la aplicación web Streamlit
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para ejecutar Streamlit
CMD ["streamlit", "run", "app_web/main_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
