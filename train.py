import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

def download_tox21():
    """Download Tox21 dataset from public S3 bucket"""
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    print("Downloading Tox21 dataset...")
    df = pd.read_csv(url)
    return df

def process_data(df, target_col='NR-AR'):
    """Extract SMILES, target labels, compute RDKit descriptors, handle missing data"""
    # Extract SMILES and target toxicity label
    smiles = df['smiles']
    y = df[target_col]
    
    # Drop rows with missing target values
    valid_idx = ~y.isna()
    smiles = smiles[valid_idx]
    y = y[valid_idx].astype(int)
    
    # Convert SMILES to RDKit Mol objects
    print("Converting SMILES to molecular objects...")
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    
    # Filter out invalid SMILES
    valid_mol_idx = [i for i, mol in enumerate(mols) if mol is not None]
    mols = [mols[i] for i in valid_mol_idx]
    y = y.iloc[valid_mol_idx]
    smiles = smiles.iloc[valid_mol_idx]
    print(f"Valid molecules: {len(mols)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Compute 1D/2D molecular descriptors using RDKit
    print("Computing molecular descriptors...")
    desc_list = Descriptors.descList  # List of (descriptor_name, function) tuples
    desc_names = [d[0] for d in desc_list]
    
    X = []
    for mol in mols:
        desc_vals = [d[1](mol) for d in desc_list]
        X.append(desc_vals)
    X = pd.DataFrame(X, columns=desc_names)
    
    # Replace inf/-inf with nan before imputation
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Impute missing descriptor values with mean
    print("Imputing missing descriptor values...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    # Clip values to float32 limits to prevent overflow in tree-based models
    X_imputed = np.clip(X_imputed, np.finfo(np.float32).min, np.finfo(np.float32).max)
    X = pd.DataFrame(X_imputed, columns=desc_names)
    
    return X, y, desc_names

def train_and_evaluate(X, y):
    """Train 3 ML classifiers, evaluate performance, return best model"""
    # Split into train/test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit StandardScaler on training data and save it for production use
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler to file for production use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to scaler.pkl")
    
    # Define 3 classifiers with class balancing for imbalanced data
    # All classifiers use scaled data
    classifiers = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', n_estimators=100, random_state=42
        ),
        'SVM': SVC(class_weight='balanced', probability=True, random_state=42)
    }
    
    best_score = -1
    best_model = None
    best_model_name = ''
    
    print("\nTraining and evaluating classifiers...")
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Track best model (using ROC-AUC as primary metric)
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = clf
            best_model_name = name
            best_X_train = X_train
    
    print(f"\nBest model: {best_model_name} with ROC-AUC: {best_score:.4f}")
    return best_model, best_model_name, best_X_train, y_train

def extract_top_descriptors(best_model, best_model_name, X_train, y_train):
    """Extract top 5 most important descriptors from best model"""
    print("\nExtracting top 5 important descriptors...")
    feature_names = X_train.columns.tolist()
    
    if best_model_name == 'Random Forest':
        importances = best_model.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
    elif best_model_name == 'Logistic Regression':
        clf = best_model.named_steps['clf']
        importances = np.abs(clf.coef_[0])
        top_indices = np.argsort(importances)[-5:][::-1]
    elif best_model_name == 'SVM':
        # SVM has no built-in feature importance, train auxiliary RF for explanation
        print("SVM has no native feature importance. Training auxiliary Random Forest for descriptor ranking...")
        rf = RandomForestClassifier(class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
    
    top_descriptors = [feature_names[i] for i in top_indices]
    return top_descriptors

def write_descriptor_explanations(top_descriptors, filename='descriptors_explanation.txt'):
    """Write chemical explanations for top descriptors"""
    # Predefined explanations for common RDKit descriptors
    desc_explanations = {
        'MolWt': 'Molecular Weight: Mass of the molecule in Daltons. Heavier molecules often have different ADME (absorption, distribution, metabolism, excretion) properties that can influence toxicity.',
        'LogP': 'Octanol-Water Partition Coefficient: Measures lipophilicity. High LogP indicates higher fat solubility, which can lead to bioaccumulation and increased toxicity.',
        'NumHDonors': 'Number of Hydrogen Bond Donors: Count of H atoms bonded to electronegative O/N atoms. More H donors increase interactions with biological targets, affecting toxicity.',
        'NumHAcceptors': 'Number of Hydrogen Bond Acceptors: Count of O/N atoms with lone pairs available for H-bonding. Influences molecular interactions with biological systems.',
        'TPSA': 'Topological Polar Surface Area: Estimates polar surface area, correlates with membrane permeability. High TPSA reduces cell permeability, affecting toxicity.',
        'NumRotatableBonds': 'Number of Rotatable Bonds: Indicates molecular flexibility. More rotatable bonds increase entropy, affecting target binding affinity.',
        'RingCount': 'Number of Rings: Count of ring structures. Rings increase molecular rigidity and can affect interactions with enzymes/receptors.',
        'FractionCSP3': 'Fraction of sp3 Hybridized Carbons: Indicates molecular saturation. Higher values mean more saturated, flexible structures that affect ADME.',
        'NumAromaticRings': 'Number of Aromatic Rings: Count of aromatic ring systems. Aromatic rings enable π-π stacking interactions with biological targets.',
        'MolLogP': 'Calculated LogP using the Wildman-Crippen method, similar to experimental LogP measurements.'
    }
    
    with open(filename, 'w') as f:
        f.write("Top 5 Most Important Molecular Descriptors\n")
        f.write("==========================================\n\n")
        for i, desc in enumerate(top_descriptors, 1):
            explanation = desc_explanations.get(desc, f"No detailed explanation available for {desc}.")
            f.write(f"{i}. {desc}: {explanation}\n")
    print(f"Descriptor explanations written to {filename}")

def save_best_model(best_model, filename='best_model.pkl'):
    """Save best model to disk"""
    with open(filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved as {filename}")

if __name__ == "__main__":
    # Download and process data
    df = download_tox21()
    X, y, desc_names = process_data(df, target_col='NR-AR')
    
    # Train models and get best performer
    best_model, best_model_name, X_train, y_train = train_and_evaluate(X, y)
    
    # Extract and explain top descriptors
    top_descriptors = extract_top_descriptors(best_model, best_model_name, X_train, y_train)
    print(f"Top 5 descriptors: {top_descriptors}")
    write_descriptor_explanations(top_descriptors)
    
    # Save best model
    save_best_model(best_model)