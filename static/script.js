// 1. Iniciar o Mapa (Focado na Amazônia)
const map = L.map('map').setView([-3.4653, -62.2159], 5); // Coordenadas centrais da Amazônia

// Camada de satélite (Esri World Imagery)
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
}).addTo(map);

let point1 = null;
let currentRect = null;

// Elementos da UI
const latMinInput = document.getElementById('lat-min');
const lonMinInput = document.getElementById('lon-min');
const latMaxInput = document.getElementById('lat-max');
const lonMaxInput = document.getElementById('lon-max');

const analyzeBtn = document.getElementById('analyze-btn');
const btnText = analyzeBtn.querySelector('.btn-text');
const loader = analyzeBtn.querySelector('.loader');

const resultPanel = document.getElementById('result-panel');
const closeResultBtn = document.getElementById('close-result-btn');
const satelliteImg = document.getElementById('satellite-img');
const statusBadge = document.getElementById('status-badge');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const areaText = document.getElementById('area-text');
const imageOverlay = document.getElementById('image-overlay-status');

// 2. Interação com o Mapa (2 Cliques para Retângulo)
map.on('click', function(e) {
    if (!point1) {
        // Primeiro clique
        point1 = e.latlng;
        if (currentRect) map.removeLayer(currentRect);
        
        // Coloca inputs como aguardando
        latMinInput.value = point1.lat.toFixed(6);
        lonMinInput.value = point1.lng.toFixed(6);
        latMaxInput.value = "...";
        lonMaxInput.value = "...";
        analyzeBtn.disabled = true;
        resultPanel.classList.add('hidden');
    } else {
        // Segundo clique
        const point2 = e.latlng;
        
        const latMin = Math.min(point1.lat, point2.lat).toFixed(6);
        const latMax = Math.max(point1.lat, point2.lat).toFixed(6);
        const lonMin = Math.min(point1.lng, point2.lng).toFixed(6);
        const lonMax = Math.max(point1.lng, point2.lng).toFixed(6);
        
        latMinInput.value = latMin;
        latMaxInput.value = latMax;
        lonMinInput.value = lonMin;
        lonMaxInput.value = lonMax;
        
        const bounds = [[latMin, lonMin], [latMax, lonMax]];
        currentRect = L.rectangle(bounds, {
            color: "#34d399",
            weight: 2,
            fillColor: "#34d399",
            fillOpacity: 0.2
        }).addTo(map);
        
        point1 = null; // Reseta para próxima seleção
        analyzeBtn.disabled = false;
    }
});

// 3. Fechar painel de resultados
closeResultBtn.addEventListener('click', () => {
    resultPanel.classList.add('hidden');
});

// 4. Botão de Análise (Chamada para a API FastAPI)
analyzeBtn.addEventListener('click', async () => {
    const latMin = latMinInput.value;
    const latMax = latMaxInput.value;
    const lonMin = lonMinInput.value;
    const lonMax = lonMaxInput.value;
    
    if (!latMin || latMax === "...") return;
    
    // Estado de Carregamento UI
    analyzeBtn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'block';
    resultPanel.classList.add('hidden');
    
    try {
        const url = `/predict-by-coordinates?lat_min=${latMin}&lat_max=${latMax}&lon_min=${lonMin}&lon_max=${lonMax}`;
        
        const response = await fetch(url, { method: 'POST' });
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || 'Erro ao processar imagem.');
        }
        
        // Sucesso: Preencher a UI com os resultados
        preencherResultados(data);
        
    } catch (error) {
        alert("Erro na análise: " + error.message);
    } finally {
        // Restaura botão UI
        analyzeBtn.disabled = false;
        btnText.style.display = 'block';
        loader.style.display = 'none';
    }
});

// 5. Preencher e Mostrar Resultados
function preencherResultados(data) {
    // 1. Mostrar a imagem (decodifica o base64 vindo do Python)
    satelliteImg.src = `data:image/jpeg;base64,${data.image_base64}`;
    
    // 2. Classificação
    const isDesmatamento = data.prediction === "Desmatamento";
    
    if (isDesmatamento) {
        statusBadge.textContent = "ALERTA: Desmatamento";
        statusBadge.className = "status-badge status-desmatamento";
        imageOverlay.style.backgroundColor = "rgba(239, 68, 68, 0.4)";
        confidenceBar.style.backgroundColor = "var(--alert-color)";
        
        if (currentRect) {
            currentRect.setStyle({color: '#ef4444', fillColor: '#ef4444'});
        }
    } else {
        statusBadge.textContent = "Floresta Intacta";
        statusBadge.className = "status-badge status-floresta";
        imageOverlay.style.backgroundColor = "transparent";
        confidenceBar.style.backgroundColor = "var(--safe-color)";
        
        if (currentRect) {
            currentRect.setStyle({color: '#10b981', fillColor: '#10b981'});
        }

    }
    
    // 3. Confiança
    const confPercent = Math.round(data.confidence * 100);
    confidenceBar.style.width = `${confPercent}%`;
    confidenceText.textContent = `${confPercent}%`;
    
    // 4. Área
    areaText.textContent = `${data.area_km2} km²`;
    
    // Mostrar Painel
    resultPanel.classList.remove('hidden');
}
