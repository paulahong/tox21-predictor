import joblib, os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Fragments, Descriptors
from flask import Flask, request, jsonify, render_template

# --- CONFIGURACIÓN DE RUTAS ---
# BASE_DIR apunta a la raíz 'tox21-predictor'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, 'templates'),
            static_folder=os.path.join(BASE_DIR, 'static'))

# --- 1. CARGAR EL MODELO ---
MODEL_PATH = os.path.join(BASE_DIR, 'tox21_model_ar.joblib')

try:
    # Intentamos cargar el bundle desde la raíz
    data_bundle = joblib.load(MODEL_PATH)
    
    # Extraemos componentes
    model = data_bundle['model']
    imputer = data_bundle['imputer']
    v_selector = data_bundle['v_selector']
    scaler = data_bundle['scaler']
    corr_to_drop = data_bundle['corr_to_drop']
    boruta_selector = data_bundle['boruta']
    optimal_threshold = data_bundle.get('threshold', 0.5)
    
    print(f"✅ Éxito: Modelo cargado correctamente desde {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error crítico al cargar el modelo: {e}")

# --- 2. PROCESAMIENTO QUÍMICO ---
def extract_all_features_api(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    
    # A. Morgan Fingerprints (Radio 2, 2048 bits)
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprintAsNumPy(mol).astype(float)
    
    # B. Fragmentos (Mismo orden que en el entrenamiento)
    frag_counts = []
    frag_names = sorted([name for name in Fragments.__dict__ if name.startswith('fr_')])
    for name in frag_names:
        func = getattr(Fragments, name)
        try: 
            frag_counts.append(float(func(mol)))
        except: 
            frag_counts.append(np.nan)
    
    # C. Descriptores Físico-químicos
    physchem_desc = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            physchem_desc.append(float(val) if val is not None else np.nan)
        except: 
            physchem_desc.append(np.nan)
            
    return np.concatenate([fp, np.array(frag_counts), np.array(physchem_desc)])

# --- 3. RUTAS DE LA APLICACIÓN ---

@app.route('/')
def index():
    # Renderiza index.html buscando en /templates (configurado arriba)
    return render_template('index.html')

@app.route('/api', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({"status": "API activa", "message": "Usa POST para predecir enviando un SMILES."})
        
    try:
        data = request.get_json()
        if not data or 'smiles' not in data:
            return jsonify({'error': 'No se proporcionó la cadena SMILES'}), 400
            
        smiles = data.get('smiles').strip()

        # 1. Extracción de características
        features = extract_all_features_api(smiles)
        if features is None:
            return jsonify({'error': 'La estructura química (SMILES) es inválida'}), 400

        X = features.reshape(1, -1)
        
        # 2. Pipeline de Preprocesamiento EXACTO al entrenamiento
        X = np.nan_to_num(X, nan=np.nan, posinf=1e6, neginf=-1e6) 
        
        X_imp = imputer.transform(X)
        X_var = v_selector.transform(X_imp)
        X_corr = np.delete(X_var, corr_to_drop, axis=1)
        X_scaled = scaler.transform(X_corr)
        X_final = X_scaled[:, boruta_selector.support_]

        # 3. Inferencia
        prob = model.predict_proba(X_final)[0][1]
        
        # Clasificación basada en el umbral optimizado
        is_toxic = prob >= optimal_threshold
        toxicity_label = 'Tóxico (NR-AR Active)' if is_toxic else 'No Tóxico (NR-AR Inactive)'

        return jsonify({
            'smiles': smiles,
            'toxicity': toxicity_label, 
            'probability': round(float(prob), 4),
            'threshold': optimal_threshold,
            'status': 'success'
        })

    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

if __name__ == '__main__':
    # Lanzamos la aplicación
    app.run(debug=True, port=5000)