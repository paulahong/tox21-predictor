from flask import Flask, request, jsonify, send_from_directory
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

app = Flask(__name__)

# Cargar el modelo (pipeline completo con imputer, variance filter, scaler y classifier)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

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
        
        # Replace inf/-inf with nan (pipeline will handle imputation)
        X = X.replace([np.inf, -np.inf], np.nan)

        # Use the pipeline directly - it handles imputation, variance filtering, scaling, and prediction
        prob = model.predict_proba(X)[0][1]
        toxicity = 'Toxic' if prob >= 0.5 else 'Non-Toxic'

        return jsonify({'toxicity': toxicity, 'probability': float(prob)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500