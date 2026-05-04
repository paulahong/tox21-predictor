import json
import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer

# Load the trained model once at cold start
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Prepare descriptor metadata (same as training)
DESC_LIST = Descriptors.descList  # List of (name, function) tuples
DESC_NAMES = [d[0] for d in DESC_LIST]

def compute_descriptors(smiles):
    """Convert a SMILES string to a list of descriptor values."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [func(mol) for _, func in DESC_LIST]

def handler(request, context):
    """
    Vercel serverless function entry point.
    Expects a JSON payload: { "smiles": "C(C(=O)O)N" }
    Returns: { "toxicity": "Toxic" | "Non-Toxic", "probability": 0.73 }
    """
    try:
        # Parse request body
        if request.get('method') != 'POST':
            return {
                'statusCode': 405,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Method not allowed. Use POST.'})
            }

        body = request.get('body')
        if isinstance(body, str):
            body = json.loads(body)

        smiles = body.get('smiles')
        if not smiles:
            raise ValueError('Missing "smiles" in request payload.')

        # Compute descriptors
        desc_vals = compute_descriptors(smiles)
        if desc_vals is None:
            raise ValueError('Invalid SMILES string; could not be parsed by RDKit.')

        # Build DataFrame (single row)
        X = pd.DataFrame([desc_vals], columns=DESC_NAMES)

        # Replace infinities with NaN
        X = X.replace([np.inf, -np.inf], np.nan)

        # Impute missing values (mean strategy)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)

        # Clip to float32 limits to avoid overflow in tree‑based models
        X_clipped = np.clip(
            X_imputed,
            np.finfo(np.float32).min,
            np.finfo(np.float32).max
        )

        # Predict probability of the positive class (toxic)
        prob = model.predict_proba(X_clipped)[0][1]
        toxicity = 'Toxic' if prob >= 0.5 else 'Non‑Toxic'

        response_body = {
            'toxicity': toxicity,
            'probability': float(prob)  # ensure JSON serializable
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_body)
        }

    except Exception as e:
        # Return a 400 Bad Request with error details
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({'error': str(e)})
        }