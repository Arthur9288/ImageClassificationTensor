from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import base64
import requests as http_requests
from dotenv import load_dotenv

# Carrega variáveis do .env
load_dotenv()

GCP_PROJECT_DEFAULT = os.getenv('GCP_PROJECT_ID', 'seu-projeto-gcp')
API_HOST            = os.getenv('API_HOST', '0.0.0.0')
API_PORT            = int(os.getenv('API_PORT', '8000'))

# Google Earth Engine (opcional)
try:
    import ee
    EE_AVAILABLE = True
except ImportError:
    EE_AVAILABLE = False

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    # Tenta o formato nativo .keras primeiro, depois o legado .h5
    for model_path in ['modelo_amazonia.keras', 'modelo_amazonia.h5']:
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Modelo '{model_path}' carregado com sucesso no startup.")
            break
        except Exception:
            continue
    if model is None:
        print("ERRO: Nenhum modelo encontrado. Execute train.py primeiro.")
    yield

app = FastAPI(
    title="AmazonImageClassification",
    description="API inteligente para detecção de desmatamento em imagens de satélite usando Transfer Learning (MobileNetV2) e Google Earth Engine.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"message": "Modelo não carregado no servidor."}
        )
    
    try:
        # 1. Leitura do arquivo de imagem
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 2. Pré-processamento
        # Resize para 224x224
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        
        # Normalização para o padrão do MobileNetV2: escala [-1, 1]
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 3. Predição
        predictions = model.predict(img_array)
        score = float(predictions[0][0])
        
        # O sigmoid retorna P(classe=1). Com ordenação alfabética:
        # classe 0 = "Desmatamento", classe 1 = "Floresta Intacta"
        # Portanto: score > 0.5 → Floresta Intacta, score <= 0.5 → Desmatamento
        if score > 0.5:
            label = "Floresta Intacta"
            confidence = score
        else:
            label = "Desmatamento"
            confidence = 1.0 - score
            
        return {
            "prediction": label,
            "confidence": confidence
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"Erro ao processar imagem: {str(e)}"}
        )


def _preprocessar_imagem(img_bytes: bytes) -> np.ndarray:
    """Utilitário: converte bytes → array pré-processado para o modelo."""
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def _classificar(score: float) -> dict:
    """Utilitário: converte score sigmoid → label + confidence."""
    if score > 0.5:
        return {"prediction": "Floresta Intacta", "confidence": round(score, 4)}
    else:
        return {"prediction": "Desmatamento", "confidence": round(1.0 - score, 4)}


@app.post("/predict-by-coordinates")
async def predict_by_coordinates(
    lat_min: float = Query(..., description="Latitude mínima (Sul)"),
    lat_max: float = Query(..., description="Latitude máxima (Norte)"),
    lon_min: float = Query(..., description="Longitude mínima (Oeste)"),
    lon_max: float = Query(..., description="Longitude máxima (Leste)"),
    gcp_project: str = Query(GCP_PROJECT_DEFAULT, description="ID do seu projeto Google Cloud"),
    date_start: str = Query("2023-01-01", description="Data de início (YYYY-MM-DD)"),
    date_end:   str = Query("2024-12-31", description="Data de fim   (YYYY-MM-DD)"),
):
    """
    Baixa automaticamente uma imagem Sentinel-2 via Google Earth Engine
    para as coordenadas informadas e retorna a classificação do modelo.

    **Pré-requisito:** autenticar com `earthengine authenticate` uma vez.
    """
    if not EE_AVAILABLE:
        return JSONResponse(
            status_code=501,
            content={"message": "earthengine-api não instalado. "
                                 "Rode: pip install earthengine-api"}
        )
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"message": "Modelo não carregado. Execute train.py primeiro."}
        )

    try:
        # 1. Inicializar Earth Engine
        ee.Initialize(project=gcp_project)

        # 2. Definir área e buscar imagem Sentinel-2
        regiao = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

        imagem = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(regiao)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .select(['B4', 'B3', 'B2'])
            .median()
        )


        # 3. Obter URL de download
        url = imagem.getThumbURL({
            'region': regiao,
            'dimensions': 224,
            'format': 'jpg',
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        })

        # 4. Baixar a imagem
        resposta = http_requests.get(url, timeout=30)
        resposta.raise_for_status()

        if len(resposta.content) < 5000:
            return JSONResponse(
                status_code=422,
                content={"message": "Imagem inválida retornada pelo Earth Engine. "
                                     "Tente ampliar o intervalo de datas ou reduzir o filtro de nuvens."}
            )

        # 5. Pré-processar e classificar
        img_array = _preprocessar_imagem(resposta.content)
        score = float(model.predict(img_array)[0][0])
        resultado = _classificar(score)
        
        # 6. Converter imagem para base64 para mostrar na UI
        img_base64 = base64.b64encode(resposta.content).decode("utf-8")

        # Calcular área (aproximação simples em graus para km)
        delta_lat_km = abs(lat_max - lat_min) * 111
        delta_lon_km = abs(lon_max - lon_min) * 111
        area_km2 = round(delta_lat_km * delta_lon_km, 2)

        return {
            "lat_center": round((lat_min + lat_max) / 2, 6),
            "lon_center": round((lon_min + lon_max) / 2, 6),
            "area_km2": area_km2,
            "image_base64": img_base64,
            **resultado
        }


    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Erro ao consultar Earth Engine: {str(e)}"}
        )


# Montar pasta static por último para não sobrescrever rotas da API
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
