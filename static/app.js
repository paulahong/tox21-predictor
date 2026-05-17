// app.js – Handles UI interactions, API communication and Local History

// URL directa de tu Space de Hugging Face
const HUGGING_FACE_API_URL = "https://paulahong-tox21-predictor-api.hf.space/api";

// Utility to show/hide elements
function toggleVisibility(id, show) {
    const el = document.getElementById(id);
    if (!el) return;
    if (show) {
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

// Show loading spinner (Animación molecular activa)
function showLoading() {
    toggleVisibility('loading', true);
    toggleVisibility('results', false);
    toggleVisibility('error', false);
}

// Hide loading spinner
function hideLoading() {
    toggleVisibility('loading', false);
}

// Display error message
function showError(message) {
    let cleanMessage = message;
    try {
        const parsed = JSON.parse(message);
        if (parsed.error) cleanMessage = parsed.error;
    } catch (e) {
        // No era un JSON estructurado, se mantiene el texto original
    }

    const errEl = document.getElementById('error');
    if (errEl) {
        errEl.textContent = cleanMessage;
    }
    toggleVisibility('error', true);
}

// Muestra los resultados de la predicción, renderiza la molécula 2D e inyecta impactos SHAP
function displayResult(toxicity, probability, topFeatures, smiles) {
    const verdictEl = document.getElementById('toxicity-verdict');
    const cardEl = document.getElementById('prediction-card');
    const barEl = document.getElementById('probability-bar');
    const percentEl = document.getElementById('probability-text');
    const summaryEl = document.getElementById('quick-summary');
    const shapList = document.getElementById('shap-list');

    // --- 1. RENDERIZADO DE LA ESTRUCTURA QUÍMICA 2D REESCALADA ---
    try {
        const smilesDrawer = new SmilesDrawer.Drawer({
            width: 140, 
            height: 140, 
            bondThickness: 1.3,
            bondLength: 13,
            fontSizeLarge: 9,
            fontSizeSmall: 6,
            themes: {
                light: {
                    C: '#ffffff', // Blanco/claro para resaltar sobre el fondo oscuro pizarra de la tarjeta
                    O: '#f87171',
                    N: '#60a5fa',
                    F: '#34d399',
                    Cl: '#34d399',
                    Br: '#fb923c',
                    I: '#c084fc',
                    S: '#fbbf24',
                    P: '#f97316'
                }
            }
        });

        SmilesDrawer.parse(smiles, function(tree) {
            smilesDrawer.draw(tree, 'molecule-canvas', 'light', false);
        }, function(err) {
            console.error('Error al parsear el SMILES con SmilesDrawer:', err);
        });
    } catch (e) {
        console.error('Error al inicializar o dibujar con SmilesDrawer:', e);
    }

    // --- 2. CONFIGURACIÓN COMPACTA Y LÓGICA TRIPLE DEL SEMÁFORO (0-35, 35-65, >65) ---
    const percent = (probability * 100).toFixed(1);
    
    if (verdictEl) verdictEl.textContent = toxicity;
    if (percentEl) percentEl.textContent = `${percent}%`;
    if (barEl) barEl.style.width = `${percent}%`;

    // Reseteamos las clases de estado previas de la tarjeta y veredicto
    cardEl.classList.remove('is-toxic', 'is-neutral', 'is-nontoxic');
    if (verdictEl) verdictEl.className = '';

    // Evaluación matemática por rangos
    if (probability <= 0.35) {
        // VERDE: Estable / No Tóxico
        cardEl.classList.add('is-nontoxic'); 
        if (verdictEl) verdictEl.classList.add('text-nontoxic');
        if (barEl) barEl.style.backgroundColor = '#10b981';
        if (percentEl) percentEl.style.color = '#10b981';
        if (summaryEl) summaryEl.textContent = "Estructura molecular estable. No se detectan factores críticos de inducción de estrés oxidativo celulares.";
    } 
    else if (probability > 0.35 && probability <= 0.65) {
        // NARANJA: Advertencia / Zona Gris Neutra
        cardEl.classList.add('is-neutral');
        if (verdictEl) verdictEl.className = ''; // Color neutro por defecto o heredado
        if (barEl) barEl.style.backgroundColor = '#f59e0b'; // Color ámbar/naranja
        if (percentEl) percentEl.style.color = '#f59e0b';
        if (summaryEl) summaryEl.textContent = "Toxicidad indeterminada. Se recomienda precaución; presenta rasgos estructurales moderadamente reactivos.";
    } 
    else {
        // ROJO: Crítico / Tóxico
        cardEl.classList.add('is-toxic');
        if (verdictEl) verdictEl.classList.add('text-toxic');
        if (barEl) barEl.style.backgroundColor = '#ef4444';
        if (percentEl) percentEl.style.color = '#ef4444';
        if (summaryEl) summaryEl.textContent = "Alta probabilidad de que este compuesto altere el potencial eléctrico de las membranas mitocondriales.";
    }

    // --- 3. ACTUALIZACIÓN DINÁMICA DE SHAP (INTEGRA, SIN MODIFICACIONES) ---
    if (shapList) {
        shapList.innerHTML = '';

        if (topFeatures && topFeatures.length > 0) {
            topFeatures.forEach(feat => {
                const li = document.createElement('li');
                
                const isPositive = feat.shap_value > 0;
                const impactClass = isPositive ? 'shap-increase' : 'shap-decrease';
                const sign = isPositive ? '+' : '';

                const descripcionTraducida = obtenerDescripcionDescriptor(feat.descriptor);

                li.innerHTML = `
                    <div class="shap-item">
                        <div class="shap-header">
                            <span class="shap-name"><strong>${feat.descriptor}</strong></span>
                            <span class="shap-badge ${impactClass}">
                                ${feat.impact} (${sign}${feat.shap_value.toFixed(4)})
                            </span>
                        </div>
                        <p class="shap-desc">${descripcionTraducida}</p>
                    </div>
                `;
                shapList.appendChild(li);
            });
        } else {
            shapList.innerHTML = '<li>No SHAP descriptors available for this prediction.</li>';
        }
    }
    
    toggleVisibility('results', true);
}

// Main prediction function - CONECTADA CON HUGGING FACE Y HISTORIAL
async function predictToxicity() {
    const smilesInput = document.getElementById('smiles-input');
    if (!smilesInput) return;
    
    const smiles = smilesInput.value.trim();

    if (!smiles) {
        showError('Por favor, introduce una cadena SMILES válida.');
        return;
    }

    showLoading();

    try {
        const response = await fetch(HUGGING_FACE_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles: smiles })
        });

        if (!response.ok) {
            const errMsg = await response.text();
            throw new Error(errMsg || 'El servidor de Hugging Face devolvió un error.');
        }

        const data = await response.json();
        
        const toxicity = data.toxicity;
        const probability = data.probability;
        const topFeatures = data.top_shap_features; 

        // Ocultamos la animación justo antes de pintar los resultados renderizados
        hideLoading();
        displayResult(toxicity, probability, topFeatures, smiles);
        
        // Guardar la consulta exitosa automáticamente en el historial local
        saveToHistory(smiles, toxicity);

    } catch (err) {
        console.error('Prediction error:', err);
        hideLoading();
        showError(err.message || 'Ha ocurrido un error inesperado al procesar la molécula.');
    }
}

// Función para rellenar automáticamente el SMILES de ejemplo
function loadExample(smiles) {
    const smilesInput = document.getElementById('smiles-input');
    if (smilesInput) smilesInput.value = smiles;
    
    toggleVisibility('results', false);
    toggleVisibility('error', false);
}

// --- GESTIÓN PARA EL HISTORIAL (localStorage) ---

// Guarda la consulta en la caché del navegador evitando repeticiones
function saveToHistory(smiles, toxicity) {
    let history = JSON.parse(localStorage.getItem('tox21_history')) || [];
    
    // Evita duplicados idénticos en la lista filtrando el smiles actual si ya existía
    history = history.filter(item => item.smiles !== smiles);
    
    // Inserta el nuevo elemento al principio de la lista
    history.unshift({ smiles: smiles, toxicity: toxicity });
    
    // Mantiene un límite razonable (15 elementos) para no desbordar visualmente la tarjeta
    if (history.length > 15) {
        history.pop();
    }
    
    localStorage.setItem('tox21_history', JSON.stringify(history));
    renderHistory();
}

// Pinta dinámicamente los elementos en el contenedor del HTML
function renderHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;
    
    const history = JSON.parse(localStorage.getItem('tox21_history')) || [];
    historyList.innerHTML = '';
    
    if (history.length === 0) {
        historyList.innerHTML = '<li style="font-size: 0.85rem; color: #94a3b8; font-style: italic;">No hay consultas recientes.</li>';
        return;
    }
    
    history.forEach(item => {
        const li = document.createElement('li');
        const isToxic = item.toxicity.includes('Tóxico') && !item.toxicity.startsWith('No');
        const badgeClass = isToxic ? 'toxic-bg' : 'nontoxic-bg';
        const labelText = isToxic ? 'Tóxico' : 'No Tóxico';
        
        li.innerHTML = `
            <div class="history-item" onclick="loadHistoryItem('${item.smiles}')">
                <span class="history-smiles" title="${item.smiles}">${item.smiles}</span>
                <span class="history-status-badge ${badgeClass}">${labelText}</span>
            </div>
        `;
        historyList.appendChild(li);
    });
}

// Carga un compuesto guardado del historial y dispara inmediatamente su predicción
function loadHistoryItem(smiles) {
    loadExample(smiles);
    predictToxicity();
}

// Elimina permanentemente el historial de consultas del explorador
function clearHistory() {
    localStorage.removeItem('tox21_history');
    renderHistory();
}

// Renderiza el historial automáticamente al inicializar la aplicación web
window.addEventListener('DOMContentLoaded', renderHistory);