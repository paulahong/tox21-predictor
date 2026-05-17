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
    const labelEl = document.getElementById('toxicity-label');
    const barEl = document.getElementById('probability-bar');
    const probText = document.getElementById('probability-text');
    const resultBox = document.getElementById('prediction-result');
    const shapList = document.getElementById('shap-list');

    // --- 1. RENDERIZADO DE LA ESTRUCTURA QUÍMICA 2D (SmilesDrawer) ---
    try {
        const smilesDrawer = new SmilesDrawer.Drawer({
            width: 220, 
            height: 220, 
            bondThickness: 1.5,
            bondLength: 15,
            fontSizeLarge: 10,
            fontSizeSmall: 7,
            themes: {
                light: {
                    C: '#2c3e50',
                    O: '#e74c3c',
                    N: '#3498db',
                    F: '#27ae60',
                    Cl: '#27ae60',
                    Br: '#d35400',
                    I: '#8e44ad',
                    S: '#f1c40f',
                    P: '#e67e22'
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

    // --- 2. CONFIGURACIÓN DEL VEREDICTO DE TOXICIDAD ---
    const isToxic = toxicity.includes('Tóxico') && !toxicity.startsWith('No');

    if (labelEl) labelEl.textContent = toxicity;
    
    if (isToxic) {
        if (labelEl) labelEl.className = 'toxicity-label toxic';
        if (resultBox) resultBox.className = 'prediction-box toxic-border';
    } else {
        if (labelEl) labelEl.className = 'toxicity-label nontoxic';
        if (resultBox) resultBox.className = 'prediction-box nontoxic-border';
    }

    // --- BARRA DE PROBABILIDAD CON DEGRADADO DINÁMICO ---
    const percent = (probability * 100).toFixed(1);
    if (barEl) {
        barEl.style.width = `${percent}%`;
        const hue = ((1 - probability) * 120).toString(10); 
        barEl.style.background = `linear-gradient(to right, hsl(${hue}, 85%, 55%), hsl(${hue}, 85%, 40%))`;
    }
    if (probText) probText.textContent = `Probability of toxicity: ${percent}%`;

    // --- 3. ACTUALIZACIÓN DINÁMICA DE SHAP ---
    if (shapList) {
        shapList.innerHTML = '';

        if (topFeatures && topFeatures.length > 0) {
            topFeatures.forEach(feat => {
                const li = document.createElement('li');
                
                const isPositive = feat.shap_value > 0;
                const impactClass = isPositive ? 'shap-increase' : 'shap-decrease';
                const sign = isPositive ? '+' : '';

                // MODIFICACIÓN CRÍTICA: Llamamos al glosario externo pasándole el nombre del descriptor
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
        showError('Please enter a SMILES string.');
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
        showError(err.message || 'An unexpected error occurred.');
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