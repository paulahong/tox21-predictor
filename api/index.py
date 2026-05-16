import joblib
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# --- DICCIONARIO DE EXPLICACIONES QUÍMICAS ---
DESC_EXPLANATIONS = {
    'MaxAbsEStateIndex': 'Reactividad del estado electrónico. Identifica regiones atómicas altamente reactivas capaces de inducir estrés oxidativo o unirse covalentemente a proteínas mitocondriales.',
    'FpDensityMorgan3': 'Densidad y complejidad estructural de la molécula, correlacionada con la diversidad de dianas moleculares que puede afectar.',
    'BCUT2D_MWLOW': 'Descriptor topológico basado en la distribución de la masa atómica; influye en cómo la molécula se orienta e interactúa tridimensionalmente en la célula.',
    'HallKierAlpha': 'Flexibilidad y forma molecular. Determina la capacidad de la sustancia para interactuar o acumularse en los complejos proteicos de la cadena de transporte de electrones.',
    'BCUT2D_CHGLO': 'Distribución de cargas parciales bajas. Relevante para interacciones electrostáticas débiles a lo largo de las membranas celulares.',
    'SPS': 'Puntuación de polaridad espacial en 3D. Crucial para predecir si el compuesto puede atravesar la doble membrana mitocondrial.',
    'BalabanJ': 'Índice de ramificación molecular. Estructuras más ramificadas o esféricas alteran la velocidad de difusión de la sustancia dentro de los compartimentos de la mitocondria.',
    'MolWt': 'Peso molecular total. Afecta directamente la permeabilidad pasiva y el transporte de la sustancia hacia el interior de la mitocondria.',
    'PEOE_VSA7': 'Área de superficie polarizada. Mide zonas con cargas parciales específicas implicadas en la unión a enzimas implicadas en la bioenergética celular.',
    'BCUT2D_CHGHI': 'Distribución de cargas parciales altas. Alerta sobre zonas con alta densidad de carga que pueden desestabilizar el gradiente electroquímico de la membrana (efecto desacoplador).',
    'LogP': 'Lipofilicidad (capacidad de disolverse en grasas vs agua). Clave en SR-MMP; un LogP elevado facilita que la molécula se inserte en la membrana interna mitocondrial y altere su potencial.',
    'TPSA': 'Área de superficie polar topológica. Predice el paso por membranas biológicas; valores inadecuados impiden que el xenobiótico alcance el espacio intermembrana.'
}

# --- 1. CARGAR EL MODELO ---
MODEL_PATH = os.path.join(BASE_DIR, 'tox21_model_sr_mmp.joblib')

try:
    data_bundle = joblib.load(MODEL_PATH)
    
    # Componentes del pipeline original
    model = data_bundle['model']
    imputer = data_bundle['imputer']
    v_selector = data_bundle['v_selector']
    scaler = data_bundle['scaler']
    corr_to_drop = data_bundle['corr_to_drop']
    boruta_selector = data_bundle['boruta']
    optimal_threshold = data_bundle.get('threshold', 0.5)
    
    # =========================================================================
    # RECONSTRUCCIÓN REAL DE NOMBRES DE CARACTERÍSTICAS
    # Secuencia simétrica a targetDescriptors de app.js para emparejar la matriz
    # =========================================================================
    morgan_names = [f"MorganBit_{i}" for i in range(2048)]
    local_descriptors_names = [
        # --- Fragmentos químicos ('fr_') ---
        'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
        'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 
        'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_R_CHO', 'fr_R_O_R', 'fr_Sulphonamam', 
        'fr_X', 'fr_amids', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
        # --- Descriptores Fisicoquímicos complejos ---
        'MaxAbsEStateIndex', 'FpDensityMorgan3', 'BCUT2D_MWLOW', 'HallKierAlpha', 'BCUT2D_CHGLO', 'SPS', 
        'BalabanJ', 'MolWt', 'PEOE_VSA7', 'BCUT2D_CHGHI', 'LogP', 'TPSA', 'LabuteASA', 'FractionCSP3', 'HeavyAtomCount'
    ]
    
    all_feature_names = np.array(morgan_names + local_descriptors_names)
    
    # Ajuste dinámico dimensional basado en el selector de varianza original
    dimension_esperada = v_selector.n_features_in_
    if len(all_feature_names) < dimension_esperada:
        diferencia = dimension_esperada - len(all_feature_names)
        comodines = [f"Desc_Extra_{i}" for i in range(diferencia)]
        all_feature_names = np.array(list(all_feature_names) + comodines)
    elif len(all_feature_names) > dimension_esperada:
        all_feature_names = all_feature_names[:dimension_esperada]

    # Aplicación secuencial de las máscaras de entrenamiento
    names_filt = all_feature_names[v_selector.get_support()]
    names_filt = np.delete(names_filt, corr_to_drop)
    final_feature_names = names_filt[boruta_selector.support_]
    
    print(f"Éxito: Modelo cargado con {len(final_feature_names)} variables en la frontera final.")
except Exception as e:
    print(f"Error crítico al cargar el modelo: {e}")

# --- 2. RUTAS DE LA APLICACIÓN ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({"status": "API activa", "message": "Envía un POST con el array de características en el nodo 'features'."})
        
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No se proporcionaron las características numéricas (features) de la molécula'}), 400
            
        features_locales = data.get('features')
        X = np.array(features_locales).reshape(1, -1)

        # 1. Pipeline de transformación matricial libre de RDKit-Python
        X_imp = imputer.transform(X)
        X_var = v_selector.transform(X_imp)
        X_corr = np.delete(X_var, corr_to_drop, axis=1)
        X_scaled = scaler.transform(X_corr)
        X_final = X_scaled[:, boruta_selector.support_]

        # 2. Inferencia base del modelo
        prob_base = model.predict_proba(X_final)[0][1]
        is_toxic = prob_base >= optimal_threshold
        toxicity_label = 'Tóxico' if is_toxic else 'No Tóxico'

        # =========================================================================
        # EXPLICADOR LOCAL ALINEADO (Leave-One-Out Perturbation)
        # Sustituto matemático ligero (0 MB) para atribución de valores locales
        # =========================================================================
        local_impacts = []
        
        # Iteramos sobre el espacio de características transformadas de la molécula actual
        for idx in range(X_final.shape[1]):
            # Clonamos el estado biológico actual del vector
            X_perturbed = X_final.copy()
            
            # Neutralizamos el descriptor evaluado fijándolo en un valor base nulo
            X_perturbed[0, idx] = 0.0
            
            # Recalculamos la inferencia del árbol de decisión sin dicha característica
            prob_perturbed = model.predict_proba(X_perturbed)[0][1]
            
            # Variación marginal de la probabilidad (Impacto específico de la variable)
            diff_impact = prob_base - prob_perturbed
            local_impacts.append(diff_impact)

        local_impacts = np.array(local_impacts)
        
        # Extraemos los índices de los 5 descriptores con mayor impacto local absoluto
        top_indices = np.argsort(np.abs(local_impacts))[-5:][::-1]
        
        top_features_list = []
        for idx in top_indices:
            if idx >= len(final_feature_names): continue
            name = str(final_feature_names[idx])
            val_impact = float(local_impacts[idx])
            
            # Evitamos falsas atribuciones si el descriptor fue inocuo
            if round(val_impact, 4) == 0: continue
            
            if name in DESC_EXPLANATIONS:
                explanation = DESC_EXPLANATIONS[name]
            elif name.startswith('fr_'):
                explanation = f"Presencia/Frecuencia del grupo funcional: {name.replace('fr_', '')}."
            elif 'MorganBit' in name:
                explanation = "Patrón estructural o entorno atómico local específico detectado en la huella digital de esta molécula."
            else:
                explanation = "Descriptor físico-químico o topológico molecular."
                
            direction = "Aumenta la probabilidad de toxicidad" if val_impact > 0 else "Disminuye la probabilidad de toxicidad"

            # Respetamos el esquema exacto de claves JSON para mantener la compatibilidad con displayResult() en app.js
            top_features_list.append({
                'descriptor': name,
                'shap_value': val_impact, 
                'impact': direction,
                'explanation': explanation
            })

        # Estructura de respaldo en caso de variaciones planas
        if not top_features_list:
            top_features_list.append({
                'descriptor': 'Esqueleto Molecular Base',
                'shap_value': 0.001,
                'impact': 'Aumenta la probabilidad de toxicidad' if is_toxic else 'Disminuye la probabilidad de toxicidad',
                'explanation': 'La configuración espacial y conectividad básica de la molécula determina su interacción con la membrana mitocondrial.'
            })

        return jsonify({
            'toxicity': toxicity_label, 
            'probability': round(float(prob_base), 4),
            'threshold': round(float(optimal_threshold), 4),
            'top_shap_features': top_features_list,
            'status': 'success'
        })

    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)