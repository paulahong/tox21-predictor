// app.js – Handles UI interactions and API communication

// Utility to show/hide elements
function toggleVisibility(id, show) {
    const el = document.getElementById(id);
    if (show) {
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

// Show loading spinner
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
    // Si el mensaje viene como JSON de error del backend, intentamos parsearlo
    let cleanMessage = message;
    try {
        const parsed = JSON.parse(message);
        if (parsed.error) cleanMessage = parsed.error;
    } catch (e) {
        // No era un JSON estructurado, mantenemos el texto original
    }

    const errEl = document.getElementById('error');
    errEl.textContent = cleanMessage;
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
        // Opciones estéticas adaptadas a 220px para encajar en el diseño de dos columnas
        const smilesDrawer = new SmilesDrawer.Drawer({
            width: 220, // Ajustado a 220 para que no desborde en el contenedor
            height: 220, // Ajustado a 220 para que cuadre con el alto de la tarjeta lateral
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

        // Parseamos la cadena SMILES y dibujamos sobre el canvas de index.html
        SmilesDrawer.parse(smiles, function(tree) {
            smilesDrawer.draw(tree, 'molecule-canvas', 'light', false);
        }, function(err) {
            console.error('Error al parsear el SMILES con SmilesDrawer:', err);
        });
    } catch (e) {
        console.error('Error al inicializar o dibujar con SmilesDrawer:', e);
    }

    // --- 2. CONFIGURACIÓN DEL VEREDICTO DE TOXICIDAD ---
    // Comprobamos de forma segura si es tóxico
    const isToxic = toxicity.includes('Tóxico') && !toxicity.startsWith('No');

    labelEl.textContent = toxicity;
    
    if (isToxic) {
        labelEl.className = 'toxicity-label toxic';
        resultBox.className = 'prediction-box toxic-border';
        barEl.style.backgroundColor = '#e74c3c'; // Rojo para toxicidad activa
    } else {
        labelEl.className = 'toxicity-label nontoxic';
        resultBox.className = 'prediction-box nontoxic-border';
        barEl.style.backgroundColor = '#27ae60'; // Verde para no tóxico
    }

    // Actualizar barra de probabilidad
    // --- NUEVO: BARRA DE PROBABILIDAD CON DEGRADADO DINÁMICO ---
    const percent = (probability * 100).toFixed(1);
    barEl.style.width = `${percent}%`;
    probText.textContent = `Probability of toxicity: ${percent}%`;

    // Calculamos el color dinámico (de verde a rojo pasando por amarillo/naranja)
    // Usamos interpolación HSL: 120 es Verde (seguro), 60 es Amarillo (alerta), 0 es Rojo (tóxico)
    const hue = ((1 - probability) * 120).toString(10); 
    
    // Aplicamos un degradado sutil que va desde un tono un poco más claro a uno más oscuro del color calculado
    barEl.style.background = `linear-gradient(to right, hsl(${hue}, 85%, 55%), hsl(${hue}, 85%, 40%))`;

    // --- 3. ACTUALIZACIÓN DINÁMICA DE SHAP ---
    // Limpiamos resultados anteriores de la lista
    shapList.innerHTML = '';

    if (topFeatures && topFeatures.length > 0) {
        topFeatures.forEach(feat => {
            const li = document.createElement('li');
            
            // Determinamos la clase CSS de impacto según el signo del valor de SHAP
            const isPositive = feat.shap_value > 0;
            const impactClass = isPositive ? 'shap-increase' : 'shap-decrease';
            const sign = isPositive ? '+' : '';

            li.innerHTML = `
                <div class="shap-item">
                    <div class="shap-header">
                        <span class="shap-name"><strong>${feat.descriptor}</strong></span>
                        <span class="shap-badge ${impactClass}">
                            ${feat.impact} (${sign}${feat.shap_value.toFixed(4)})
                        </span>
                    </div>
                    <p class="shap-desc">${feat.explanation}</p>
                </div>
            `;
            shapList.appendChild(li);
        });
    } else {
        // En caso de que no haya descriptores disponibles (fallback seguro)
        shapList.innerHTML = '<li>No SHAP descriptors available for this prediction.</li>';
    }
    
    toggleVisibility('results', true);
}

// Main prediction function
async function predictToxicity() {
    const smilesInput = document.getElementById('smiles-input');
    const smiles = smilesInput.value.trim();

    if (!smiles) {
        showError('Please enter a SMILES string.');
        return;
    }

    showLoading();

    try {
        const response = await fetch('/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ smiles })
        });

        if (!response.ok) {
            const errMsg = await response.text();
            throw new Error(errMsg || 'Server returned an error.');
        }

        const data = await response.json();
        
        // Extraemos las propiedades incluyendo la nueva lista de SHAP
        const toxicity = data.toxicity;
        const probability = data.probability;
        const topFeatures = data.top_shap_features; // Array devuelto por el backend

        // Pasamos todos los parámetros a la función de renderizado, INCLUYENDO 'smiles' al final
        displayResult(toxicity, probability, topFeatures, smiles);

    } catch (err) {
        console.error('Prediction error:', err);
        showError(err.message || 'An unexpected error occurred.');
    } finally {
        hideLoading();
    }
}

// NUEVO: Función para rellenar automáticamente el SMILES de ejemplo
function loadExample(smiles) {
    const smilesInput = document.getElementById('smiles-input');
    smilesInput.value = smiles;
    
    // Ocultamos resultados y errores anteriores para que la UI se limpie
    toggleVisibility('results', false);
    toggleVisibility('error', false);
}