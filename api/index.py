from flask import Flask, request, jsonify, send_from_directory
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Cargar el modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Cargar el scaler
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

DESC_LIST = Descriptors.descList
DESC_NAMES = [d[0] for d in DESC_LIST]

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [func(mol) for _, func in DESC_LIST]

# --- RUTAS PARA MOSTRAR TU DISEÑO WEB ---
# Definimos dónde está la carpeta principal (un nivel por encima de /api)
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')

@app.route('/')
def serve_index():
    # Entregar el archivo HTML principal
    return send_from_directory(ROOT_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    # Entregar el CSS y el JS
    return send_from_directory(ROOT_DIR, filename)
# ----------------------------------------

@app.route('/api', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return jsonify({"status": "API is running. Send POST request with SMILES."})
        
    try:
        data = request.get_json()
        smiles = data.get('smiles')
        if not smiles:
            return jsonify({'error': 'No SMILES provided'}), 400

        desc_vals = compute_descriptors(smiles)
        if desc_vals is None:
            return jsonify({'error': 'Invalid SMILES'}), 400

        X = pd.DataFrame([desc_vals], columns=DESC_NAMES)
        X = X.replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        X_clipped = np.clip(X_imputed, np.finfo(np.float32).min, np.finfo(np.float32).max)

        X_scaled = scaler.transform(X_clipped)

        prob = model.predict_proba(X_scaled)[0][1]
        toxicity = 'Toxic' if prob >= 0.5 else 'Non-Toxic'

        return jsonify({'toxicity': toxicity, 'probability': float(prob)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
