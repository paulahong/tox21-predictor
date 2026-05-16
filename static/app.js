// app.js – Handles UI interactions and API communication

// Variable global para almacenar el motor químico WebAssembly
let RDKitEngine = null;

// --- INICIALIZACIÓN AUTOMÁTICA DE RDKIT.JS EN EL CLIENTE ---
window.initRDKitModule().then(function(instance) {
    RDKitEngine = instance;
    console.log("RDKit.js inicializado localmente en el navegador con éxito.");
    
    // Activamos de forma segura la interfaz de usuario una vez listo el motor químico
    document.getElementById("smiles-input").disabled = false;
    const btn = document.getElementById("predict-btn");
    btn.disabled = false;
    btn.innerText = "Predecir Toxicidad";
}).catch(function(err) {
    console.error("Fallo crítico al iniciar RDKit.js en el cliente:", err);
    showError("Error del sistema: No se pudo cargar el motor químico local en tu navegador.");
});

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
    let cleanMessage = message;
    try {
        const parsed = JSON.parse(message);
        if (parsed.error) cleanMessage = parsed.error;
    } catch (e) {
        // No era un JSON estructurado
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
        const smilesDrawer = new SmilesDrawer.Drawer({
            width: 220, 
            height: 220, 
            bondThickness: 1.5,
            bondLength: 15,
            fontSizeLarge: 10,
            fontSizeSmall: 7,
            themes: {
                light: {
                    C: '#2c3e50', O: '#e74c3c', N: '#3498db',
                    F: '#27ae60', Cl: '#27ae60', Br: '#d35400',
                    I: '#8e44ad', S: '#f1c40f', P: '#e67e22'
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

    labelEl.textContent = toxicity;
    
    if (isToxic) {
        labelEl.className = 'toxicity-label toxic';
        resultBox.className = 'prediction-box toxic-border';
        barEl.style.backgroundColor = '#e74c3c'; 
    } else {
        labelEl.className = 'toxicity-label nontoxic';
        resultBox.className = 'prediction-box nontoxic-border';
        barEl.style.backgroundColor = '#27ae60'; 
    }

    // Actualizar barra de probabilidad con degradado dinámico
    const percent = (probability * 100).toFixed(1);
    barEl.style.width = `${percent}%`;
    probText.textContent = `Probability of toxicity: ${percent}%`;

    const hue = ((1 - probability) * 120).toString(10); 
    barEl.style.background = `linear-gradient(to right, hsl(${hue}, 85%, 55%), hsl(${hue}, 85%, 40%))`;

    // --- 3. ACTUALIZACIÓN DINÁMICA DE SHAP ---
    shapList.innerHTML = '';

    if (topFeatures && topFeatures.length > 0) {
        topFeatures.forEach(feat => {
            const li = document.createElement('li');
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

    if (!RDKitEngine) {
        showError('El motor químico se está inicializando. Por favor, inténtalo de nuevo en unos segundos.');
        return;
    }

    showLoading();

    try {
        // Cargar la molécula en la memoria de WebAssembly
        let mol = RDKitEngine.get_mol(smiles);
        if (!mol) {
            throw new Error('Estructura SMILES molecular inválida o irreconocible.');
        }

        // ==========================================================
        // TIPO 1: GENERAR MORGAN FINGERPRINTS (2048 BITS)
        // ==========================================================
        const fpOptions = { radius: 2, nBits: 2048 };
        const binaryBitString = mol.get_morgan_fp_as_binary_text(fpOptions);

        let chemicalFeatures = [];
        for (let i = 0; i < binaryBitString.length; i++) {
            chemicalFeatures.push(parseFloat(binaryBitString[i]));
        }

        // ==========================================================
        // NUEVO: TIPOS 2 Y 3 (FRAGMENTOS Y DESCRIPTORES REALES)
        // Extraemos dinámicamente los valores calculados por RDKit.js
        // ==========================================================
        const computedDescriptors = JSON.parse(mol.get_descriptors());

        // Array maestro indexado para emparejar simétricamente con el backend (index.py)
        const targetDescriptors = [
            // --- Fragmentos estructurales ('fr_') ---
            'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
            'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 
            'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_R_CHO', 'fr_R_O_R', 'fr_Sulphonamam', 
            'fr_X', 'fr_amids', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
            // --- Parámetros físico-químicos ---
            'MaxAbsEStateIndex', 'FpDensityMorgan3', 'BCUT2D_MWLOW', 'HallKierAlpha', 'BCUT2D_CHGLO', 'SPS', 
            'BalabanJ', 'MolWt', 'PEOE_VSA7', 'BCUT2D_CHGHI', 'LogP', 'TPSA', 'LabuteASA', 'FractionCSP3', 'HeavyAtomCount'
        ];

        let extraFeatures = [];
        targetDescriptors.forEach(descName => {
            // Evaluamos si RDKit.js posee el cómputo del descriptor físico, de lo contrario se envía un marcador neutro (0.0)
            if (computedDescriptors[descName] !== undefined) {
                extraFeatures.push(parseFloat(computedDescriptors[descName]));
            } else {
                extraFeatures.push(0.0); 
            }
        });

        // ==========================================================
        // UNIFICACIÓN E IGUALACIÓN ESTRUCTURAL DEL VECTOR
        // ==========================================================
        // Concatenamos: 2048 Bits + los descriptores reales añadidos
        let unifiedVector = chemicalFeatures.concat(extraFeatures);

        // Si tu backend requiere un tamaño fijo total para no desalinear las dimensiones del Imputer (ej: 2348 columnas),
        // rellenamos con un margen dinámico de ceros al final únicamente si faltan posiciones.
        const exactExpectedLength = 2348; 
        if (unifiedVector.length < exactExpectedLength) {
            let shortfall = exactExpectedLength - unifiedVector.length;
            unifiedVector = unifiedVector.concat(new Array(shortfall).fill(0.0));
        } else if (unifiedVector.length > exactExpectedLength) {
            unifiedVector = unifiedVector.slice(0, exactExpectedLength);
        }

        // Liberar la estructura de la memoria del explorador
        mol.delete();

        // ==========================================================
        // COMUNICACIÓN CON LA API (Enviando la matriz numérica completa)
        // ==========================================================
        const response = await fetch('/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features: unifiedVector })
        });

        if (!response.ok) {
            const errMsg = await response.text();
            throw new Error(errMsg || 'Server returned an error.');
        }

        const data = await response.json();
        
        const toxicity = data.toxicity;
        const probability = data.probability;
        const topFeatures = data.top_shap_features;

        // Renderizado del resultado y canvas molecular
        displayResult(toxicity, probability, topFeatures, smiles);

    } catch (err) {
        console.error('Prediction error:', err);
        showError(err.message || 'An unexpected error occurred.');
    } finally {
        hideLoading();
    }
}

// Función para rellenar automáticamente el SMILES de ejemplo
function loadExample(smiles) {
    const smilesInput = document.getElementById('smiles-input');
    smilesInput.value = smiles;
    
    toggleVisibility('results', false);
    toggleVisibility('error', false);
}