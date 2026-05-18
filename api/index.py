import joblib
import os
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import AllChem, Fragments, Descriptors
from rdkit.Chem.SaltRemover import SaltRemover
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 

# --- CONFIGURACIÓN DE RUTAS (CORREGIDA PARA HUGGING FACE) ---
# Al estar index.py en la raíz, un solo os.path.dirname apunta correctamente al directorio del Space
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# Permite que cualquier aplicación externa (como tu web de Vercel) consulte esta API sin bloqueos de seguridad
CORS(app)

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

# --- 1. CARGAR EL MODELO Y PREPARAR SHAP ---
MODEL_PATH = os.path.join(BASE_DIR, 'tox21_model_sr_mmp.joblib')

# Inicializar el limpiador de sales globalmente (igual que en el entrenamiento)
remover = SaltRemover()

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
    morgan_names = [f"MorganBit_{i}" for i in range(2048)]
    fragment_names = [name for name, _ in Fragments.__dict__.items() if name.startswith('fr_')]
    descriptor_names = [name for name, _ in Descriptors.descList]
    
    all_feature_names = np.array(morgan_names + fragment_names + descriptor_names)
    
    # Aplicamos los mismos filtros de selección que en el entrenamiento
    names_filt = all_feature_names[v_selector.get_support()]
    names_filt = np.delete(names_filt, corr_to_drop)
    final_feature_names = names_filt[boruta_selector.support_]
    
    # INICIALIZAR EL EXPLICADOR SHAP
    explainer = shap.TreeExplainer(model.get_booster())
    
    print(f"Éxito: Modelo y TreeExplainer cargados con {len(final_feature_names)} variables.")
except Exception as e:
    print(f"Error crítico al cargar el modelo o inicializar SHAP: {e}")

# --- 2. PROCESAMIENTO QUÍMICO ---
def extract_all_features_api(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        mol = remover.StripMol(mol)
        smiles_clean = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smiles_clean)
        if mol is None: return None
        
        # 1. Morgan Fingerprint (Radio 2, Tamaño 2048)
        generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        fp_array = generator.GetFingerprintAsNumPy(mol).astype(float)
        
        # 2. Fragmentos
        frag_counts = []
        for name, func in Fragments.__dict__.items():
            if name.startswith('fr_'):
                try: frag_counts.append(float(func(mol)))
                except: frag_counts.append(np.nan)
                    
        # 3. Descriptores fisicoquímicos
        physchem_desc = []
        for name, func in Descriptors.descList:
            try:
                val = func(mol)
                physchem_desc.append(np.nan if val is None or np.isnan(val) or np.isinf(val) else float(val))
            except: physchem_desc.append(np.nan)
                
        return np.concatenate([fp_array, np.array(frag_counts), np.array(physchem_desc)])
    except:
        return None

# --- 3. RUTAS DE LA APLICACIÓN ---

@app.route('/')
def index():
    return jsonify({"status": "API activa", "message": "Backend de Inteligencia Artificial funcionando correctamente."})

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
            return jsonify({'error': 'La estructura química (SMILES) es inválida o falló la desalinización'}), 400

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

        # 3. CÁLCULO DINÁMICO DE SHAP
        shap_values = explainer.shap_values(X_final)
        
        if isinstance(shap_values, list):
            shap_local = shap_values[1][0]  
        elif len(shap_values.shape) == 3:
            shap_local = shap_values[0, :, 1]
        elif len(shap_values.shape) == 2:
            shap_local = shap_values[0]
        else:
            shap_local = shap_values

        top_indices = np.argsort(np.abs(shap_local))[-5:][::-1]
        
        top_features_list = []
        for idx in top_indices:
            name = str(final_feature_names[idx])
            val_shap = float(shap_local[idx])
            
            if name in DESC_EXPLANATIONS:
                explanation = DESC_EXPLANATIONS[name]
            elif name.startswith('fr_'):
                explanation = f"Presencia/Frecuencia del grupo funcional: {name.replace('fr_', '')}."
            elif 'MorganBit' in name:
                explanation = "Patrón estructural o entorno atómico local específico."
            else:
                explanation = "Descriptor físico-químico o topológico molecular."
                
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
    # Forzamos la dirección global y el puerto 7860 obligatorio de Hugging Face
    app.run(host='0.0.0.0', port=7860, debug=False)