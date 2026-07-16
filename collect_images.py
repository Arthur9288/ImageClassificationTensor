import ee
import requests
import os
import time
import math
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import io

# Tenta importar o TensorFlow para classificar na hora
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Carrega variáveis do arquivo .env (nunca sobe ao GitHub)
load_dotenv()

# ============================================================
# CONFIGURAÇÕES — definidas no arquivo .env
# ============================================================

GCP_PROJECT    = os.getenv('GCP_PROJECT_ID', 'seu-projeto-gcp')
DATE_START     = os.getenv('DATE_START', '2023-01-01')
DATE_END       = os.getenv('DATE_END',   '2024-12-31')
MAX_CLOUD_PCT  = int(os.getenv('MAX_CLOUD_PCT', '10'))

LAT_MIN        = float(os.getenv('LAT_MIN', '-7.50'))
LAT_MAX        = float(os.getenv('LAT_MAX', '-7.00'))
LON_MIN        = float(os.getenv('LON_MIN', '-60.50'))
LON_MAX        = float(os.getenv('LON_MAX', '-60.00'))

MAX_IMAGES     = int(os.getenv('MAX_IMAGES', '100')) # Limite de segurança para teste

# Pasta base de saída
OUTPUT_BASE_DIR = 'data_coletado'
TILE_SIZE_DEG   = 0.02
IMG_SIZE_PX     = 224

# Tenta carregar o modelo treinado para pré-organizar as imagens
MODELO = None
if TF_AVAILABLE:
    for model_path in ['modelo_amazonia.keras', 'modelo_amazonia.h5']:
        if os.path.exists(model_path):
            try:
                MODELO = tf.keras.models.load_model(model_path)
                print(f"🤖 Modelo '{model_path}' carregado para organizar as imagens automaticamente.")
                break
            except Exception:
                continue


# ============================================================
# AUTENTICAÇÃO E INICIALIZAÇÃO
# ============================================================

def inicializar_ee():
    """Autentica e inicializa o Google Earth Engine."""
    try:
        ee.Initialize(project=GCP_PROJECT)
        print("✅ Google Earth Engine inicializado com sucesso.")
    except Exception:
        print("🔑 Autenticação necessária. Abrindo navegador...")
        ee.Authenticate()
        ee.Initialize(project=GCP_PROJECT)
        print("✅ Google Earth Engine inicializado com sucesso.")

# ============================================================
# FUNÇÕES PRINCIPAIS
# ============================================================

def gerar_grade(lat_min, lat_max, lon_min, lon_max, tile_size):
    """
    Gera uma grade de coordenadas (bounding boxes) cobrindo a área.
    Retorna lista de (lat_min, lat_max, lon_min, lon_max) por tile.
    """
    tiles = []
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            tiles.append((
                round(lat, 6),
                round(min(lat + tile_size, lat_max), 6),
                round(lon, 6),
                round(min(lon + tile_size, lon_max), 6)
            ))
            lon += tile_size
        lat += tile_size

    total_cols = math.ceil((lon_max - lon_min) / tile_size)
    total_rows = math.ceil((lat_max - lat_min) / tile_size)
    print(f"📐 Grade: {total_rows} linhas × {total_cols} colunas = {len(tiles)} tiles")
    return tiles


def get_imagem_sentinel2(lat_min, lat_max, lon_min, lon_max):
    """
    Retorna uma imagem Sentinel-2 composta (mediana, sem nuvens)
    para a área e período definidos.
    """
    regiao = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    colecao = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(regiao)
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', MAX_CLOUD_PCT))
        .select(['B4', 'B3', 'B2'])  # Bandas RGB (Red, Green, Blue)
        .median()
    )

    return colecao, regiao


def classificar_imagem_bytes(img_bytes: bytes) -> str:
    """Usa o modelo carregado para classificar a imagem."""
    if MODELO is None:
        return "Nao_Classificado"

    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        score = float(MODELO.predict(img_array, verbose=0)[0][0])
        return "Floresta_Intacta" if score > 0.5 else "Desmatamento"
    except Exception:
        return "Erro_Classificacao"


def baixar_tile(lat_min, lat_max, lon_min, lon_max, output_dir, nome_base):
    """
    Baixa um tile, tenta classificar e salva na pasta correspondente.
    Retorna True se sucesso, False se falhar.
    """
    try:
        imagem, regiao = get_imagem_sentinel2(lat_min, lat_max, lon_min, lon_max)

        url = imagem.getThumbURL({
            'region': regiao,
            'dimensions': IMG_SIZE_PX,
            'format': 'jpg',
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        })

        resposta = requests.get(url, timeout=30)
        resposta.raise_for_status()

        # Verificar se retornou imagem válida (>5KB)
        if len(resposta.content) < 5000:
            return False

        # Classificar a imagem para definir a pasta destino
        classe = classificar_imagem_bytes(resposta.content)
        
        # Cria a subpasta da classe se não existir
        pasta_destino = os.path.join(output_dir, classe)
        Path(pasta_destino).mkdir(parents=True, exist_ok=True)
        
        caminho_final = os.path.join(pasta_destino, f"{nome_base}.jpg")

        with open(caminho_final, 'wb') as f:
            f.write(resposta.content)

        return True

    except Exception as e:
        print(f"    ⚠️  Erro no tile: {e}")
        return False


def coletar_imagens(
    lat_min=LAT_MIN,
    lat_max=LAT_MAX,
    lon_min=LON_MIN,
    lon_max=LON_MAX,
    output_base_dir=OUTPUT_BASE_DIR,
    tile_size=TILE_SIZE_DEG
):
    """
    Pipeline completo: gera grade → baixa imagens → classifica → salva na pasta correta.
    """
    inicializar_ee()

    # Criar diretório da sessão atual para não misturar com coletas antigas
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    pasta_sessao = os.path.join(output_base_dir, f"coleta_{timestamp}")
    Path(pasta_sessao).mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Iniciando nova sessão de coleta em: {pasta_sessao}/\n")

    tiles = gerar_grade(lat_min, lat_max, lon_min, lon_max, tile_size)

    sucesso = 0
    falhas  = 0

    for i, (la_min, la_max, lo_min, lo_max) in enumerate(tiles):
        if sucesso >= MAX_IMAGES:
            print(f"\n✋ Limite de {MAX_IMAGES} imagens atingido. Parando a coleta.")
            break

        nome_base = f"tile_{i:04d}_{la_min}_{lo_min}"

        print(f"  ⬇️  [{i+1}/{len(tiles)}] Baixando e analisando tile ({la_min:.4f}, {lo_min:.4f})...", end=' ', flush=True)

        ok = baixar_tile(la_min, la_max, lo_min, lo_max, pasta_sessao, nome_base)

        if ok:
            print("✅")
            sucesso += 1
        else:
            print("❌ (sem imagem válida ou nuvens)")
            falhas += 1

        # Pequena pausa para não sobrecarregar a API
        time.sleep(0.3)

    print(f"\n{'='*50}")
    print(f"✅ Tiles baixados e organizados: {sucesso}")
    print(f"❌ Falhas:         {falhas}")
    print(f"📁 Imagens organizadas em: {pasta_sessao}/")
    if MODELO is not None:
        print("💡 Revise as pastas geradas (Desmatamento / Floresta_Intacta) e mova as corretas para 'data/' para retreinar o modelo.")
    else:
        print("💡 Como o modelo não foi encontrado, as imagens estão na pasta 'Nao_Classificado'.")



# ============================================================
# FUNÇÃO AVULSA: baixar uma única imagem por coordenadas
# ============================================================

def baixar_imagem_unica(lat, lon, delta=0.02, output_path='imagem_unica.jpg'):
    """
    Baixa uma única imagem centrada nas coordenadas fornecidas.

    Args:
        lat    : latitude central
        lon    : longitude central
        delta  : metade do tamanho da área em graus (~2km com 0.02)
        output_path: caminho de saída da imagem
    """
    inicializar_ee()

    print(f"\n📍 Baixando imagem para: lat={lat}, lon={lon} (área: {delta*2:.3f}°)")

    ok = baixar_tile(
        lat - delta, lat + delta,
        lon - delta, lon + delta,
        output_path
    )

    if ok:
        print(f"✅ Imagem salva em: {output_path}")
    else:
        print("❌ Não foi possível baixar a imagem (tente outro período ou área).")

    return ok


# ============================================================
# PONTO DE ENTRADA
# ============================================================

if __name__ == '__main__':
    # Para coletar uma grade inteira:
    coletar_imagens()

    # Para baixar uma única imagem (descomente):
    # baixar_imagem_unica(lat=-7.25, lon=-60.25)
