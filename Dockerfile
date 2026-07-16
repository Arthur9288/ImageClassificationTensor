FROM python:3.9-slim

# Configurar diretório de trabalho
WORKDIR /app

# Instalar build-essential para compilar dependências se necessário
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requisitos e instalar dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar arquivos da API, modelo e interface
COPY main.py .
COPY modelo_amazonia.keras .
COPY modelo_amazonia.h5 .
COPY static/ ./static/

# Porta exposta
EXPOSE 8000

# Executar a API em modo de produção
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
