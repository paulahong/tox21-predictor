import joblib, os
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Fragments, Descriptors
from flask import Flask, request, jsonify, render_template

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# --- DICCIONARIO DE EXPLICACIONES QUÍMICAS ---
DESC_EXPLANATIONS = {
    'MaxAbsEStateIndex': 'Reactividad del estado electrónico de los átomos.',
    'FpDensityMorgan3': 'Densidad y complejidad estructural de la molécula.',
    'BCUT2D_MWLOW': 'Distribución del peso molecular en la topología 2D.',
    'HallKierAlpha': 'Flexibilidad y forma de la molécula para encajar en receptores.',
    'BCUT2D_CHGLO': 'Distribución de cargas parciales bajas (interacciones electrostáticas).',
    'SPS': 'Puntuación de polaridad espacial en 3D.',
    'BalabanJ': 'Índice de ramificación molecular de la estructura.',
    'MolWt': 'Peso molecular total de la sustancia.',
    'PEOE_VSA7': 'Área de superficie asociada a cargas parciales específicas.',
    'BCUT2D_CHGHI': 'Distribución de cargas parciales altas (sitios reactivos potenciales).',
    'LogP': 'Lipofilicidad (capacidad de disolverse en grasas vs agua).',
    'TPSA': 'Área de superficie polar topológica (predice paso por membranas).'
}

# --- 1. CARGAR EL MODELO Y PREPARAR SHAP ---
MODEL_PATH = os.path.join(BASE_DIR, 'tox21_model_ar.joblib')

try:
    data_bundle = joblib.load(MODEL_PATH)
    
    # Componentes del pipeline
    model = data_bundle['model']
    imputer = data_bundle['imputer']
    v_selector = data_bundle['v_selector']
    scaler = data_bundle['scaler']
    corr_to_drop = data_bundle['corr_to_drop']
    boruta_selector = data_bundle['boruta']
    optimal_threshold = data_bundle.get('threshold', 0.5)
    
    # RECONSTRUCCIÓN DE NOMBRES DE CARACTERÍSTICAS
    all_feature_names = np.array(
        [f"MorganBit_{i}" for i in range(2048)] + 
        [name for name in Fragments.__dict__.keys() if name.startswith('fr_')] + 
        [name for name, _ in Descriptors.descList]
    )
    
    # Aplicamos los mismos filtros de selección que en el entrenamiento
    names_filt = all_feature_names[v_selector.get_support()]
    names_filt = np.delete(names_filt, corr_to_drop)
    final_feature_names = names_filt[boruta_selector.support_]
    
    # INICIALIZAR EL EXPLICADOR SHAP (Una sola vez en el arranque)
    explainer = shap.TreeExplainer(model.get_booster())
    
    print(f"Éxito: Modelo y TreeExplainer cargados correctamente con {len(final_feature_names)} variables.")
except Exception as e:
    print(f"Error crítico al cargar el modelo o inicializar SHAP: {e}")

# --- 2. PROCESAMIENTO QUÍMICO ---
def extract_all_features_api(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    fp_array = AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol).astype(float)
    
    frag_counts = []
    for name, func in Fragments.__dict__.items():
        if name.startswith('fr_'):
            try: frag_counts.append(float(func(mol)))
            except: frag_counts.append(np.nan)
                
    physchem_desc = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            physchem_desc.append(np.nan if val is None or np.isnan(val) or np.isinf(val) else float(val))
        except: physchem_desc.append(np.nan)
            
    return np.concatenate([fp_array, np.array(frag_counts), np.array(physchem_desc)])

# --- 3. RUTAS DE LA APLICACIÓN ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({"status": "API activa", "message": "Envia un POST con tu SMILES."})
        
    try:
        data = request.get_json()
        if not data or 'smiles' not in data:
            return jsonify({'error': 'No se proporcionó la cadena SMILES'}), 400
            
        smiles = data.get('smiles').strip()

        # 1. Extracción e imputación
        features = extract_all_features_api(smiles)
        if features is None:
            return jsonify({'error': 'La estructura química (SMILES) es inválida'}), 400

        X = features.reshape(1, -1)
        X_imp = imputer.transform(X)
        X_var = v_selector.transform(X_imp)
        X_corr = np.delete(X_var, corr_to_drop, axis=1)
        X_scaled = scaler.transform(X_corr)
        X_final = X_scaled[:, boruta_selector.support_]

        # 2. Inferencia estándar
        prob = model.predict_proba(X_final)[0][1]
        is_toxic = prob >= optimal_threshold
        toxicity_label = 'Tóxico' if is_toxic else 'No Tóxico'

        # 3. CÁLCULO DINÁMICO DE SHAP (Explicación local)
        # Obtenemos los valores de SHAP para esta molécula en concreto
        shap_values = explainer.shap_values(X_final)
        
        # Resolver dimensiones de SHAP (XGBoost binario suele devolver array simple o de 2 dimensiones)
        if isinstance(shap_values, list):
            shap_local = shap_values[1][0]  # Clase positiva (Tóxico)
        elif len(shap_values.shape) == 3:
            shap_local = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            shap_local = shap_values[0]
        else:
            shap_local = shap_values

        # Obtener los índices de los 5 descriptores con mayor impacto ABSOLUTO
        top_indices = np.argsort(np.abs(shap_local))[-5:][::-1]
        
        top_features_list = []
        for idx in top_indices:
            name = str(final_feature_names[idx])
            val_shap = float(shap_local[idx])
            
            # Formatear la descripción/explicación química
            if name in DESC_EXPLANATIONS:
                explanation = DESC_EXPLANATIONS[name]
            elif name.startswith('fr_'):
                explanation = f"Presencia/Frecuencia del grupo funcional: {name.replace('fr_', '')}."
            elif 'MorganBit' in name:
                explanation = "Patrón estructural o entorno atómico local específico."
            else:
                explanation = "Descriptor físico-químico o topológico molecular."
                
            # ¿Aumenta o disminuye la predicción de toxicidad?
            direction = "Aumenta la probabilidad de toxicidad" if val_shap > 0 else "Disminuye la probabilidad de toxicidad"

            top_features_list.append({
                'descriptor': name,
                'shap_value': round(val_shap, 4),
                'impact': direction,
                'explanation': explanation
            })

        return jsonify({
            'smiles': smiles,
            'toxicity': toxicity_label, 
            'probability': round(float(prob), 4),
            'threshold': round(float(optimal_threshold), 4),
            'top_shap_features': top_features_list,
            'status': 'success'
        })

    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)