import matplotlib
matplotlib.use('Agg')
import joblib, shap
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from rdkit import Chem
from boruta import BorutaPy
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from rdkit.Chem.SaltRemover import SaltRemover
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from rdkit.Chem import AllChem, Descriptors, Fragments
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, precision_recall_curve, auc, matthews_corrcoef, make_scorer

def download_tox21():
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    print("Downloading Tox21 dataset...")
    df = pd.read_csv(url)
    return df

def preprocess_tox21(df, target_col='NR-AR'):
    print(f"Filas iniciales: {len(df)}")
    
    df = df.dropna(subset=[target_col]).copy()
    print(f"Filas tras eliminar nulos en {target_col}: {len(df)}")

    remover = SaltRemover() 
    
    def clean_molecule(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            
            mol = remover.StripMol(mol)
            
            return Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            return None

    print("Limpiando estructuras químicas (quitando sales)...")
    df['smiles_clean'] = df['smiles'].apply(clean_molecule)
    
    df = df.dropna(subset=['smiles_clean'])

    def handle_duplicates(group):
        if len(group[target_col].unique()) > 1:
            return None
        else:
            return group.iloc[0]

    print("Gestionando duplicados y casos contradictorios...")
    df = df.groupby('smiles_clean').apply(handle_duplicates).dropna(how='all').reset_index(drop=True)
    
    print(f"Filas finales tras limpieza profunda: {len(df)}")
    return df

def extract_all_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp = generator.GetFingerprintAsNumPy(mol) 
    fp_array = fp.astype(float) 
    
    frag_counts = []
    for name, func in Fragments.__dict__.items():
        if name.startswith('fr_'):
            try:
                frag_counts.append(float(func(mol)))
            except:
                frag_counts.append(np.nan)
    frag_array = np.array(frag_counts)
    
    physchem_desc = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            if val is None or np.isnan(val) or np.isinf(val):
                physchem_desc.append(np.nan)
            else:
                physchem_desc.append(float(val))
        except:
            physchem_desc.append(np.nan)
            
    phys_array = np.array(physchem_desc)
    
    return np.concatenate([fp_array, frag_array, phys_array])

def prepare_matrix(features_list):
    X = np.stack(features_list)
    print(f"Matriz de características creada con forma: {X.shape}")
    
    return X

def apply_downsampling(X_train, y_train, random_state=42):
    idx_toxic = np.where(y_train == 1)[0]
    idx_nontoxic = np.where(y_train == 0)[0]
    
    # Definimos el tamaño de la clase no tóxica como el doble de la tóxica
    target_nontoxic_size = len(idx_toxic) 
    
    # Guardamos un control por si acaso el dataset original no tuviera suficientes muestras
    # (aunque en Tox21 sobran no tóxicos, es una buena práctica de seguridad en código)
    nontoxic_size = min(len(idx_nontoxic), target_nontoxic_size)
    
    np.random.seed(random_state)
    idx_nontoxic_downsampled = np.random.choice(
        idx_nontoxic, size=nontoxic_size, replace=False
    )
    
    new_indices = np.concatenate([idx_toxic, idx_nontoxic_downsampled])
    np.random.shuffle(new_indices)
    
    return X_train[new_indices], y_train[new_indices]

def preprocess_pipeline(X_train, X_val, y_train, threshold=0.01, corr_threshold=0.95):
    imputer = SimpleImputer(strategy='median')
    X_train_prep = imputer.fit_transform(X_train)
    X_val_prep = imputer.transform(X_val)

    v_selector = VarianceThreshold(threshold=threshold)
    X_train_prep = v_selector.fit_transform(X_train_prep)
    X_val_prep = v_selector.transform(X_val_prep)

    corr_matrix = pd.DataFrame(X_train_prep).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    
    X_train_prep = np.delete(X_train_prep, to_drop, axis=1)
    X_val_prep = np.delete(X_val_prep, to_drop, axis=1)
    print(f"Columnas eliminadas por alta correlación: {len(to_drop)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_prep)
    X_val_scaled = scaler.transform(X_val_prep)

    print("Iniciando Boruta (esto puede tardar unos minutos)...")
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
    
    boruta_selector.fit(X_train_scaled, y_train)
    
    X_train_final = X_train_scaled[:, boruta_selector.support_]
    X_val_final = X_val_scaled[:, boruta_selector.support_]
    
    print(f"Boruta seleccionó {X_train_final.shape[1]} características de las {X_train_scaled.shape[1]} disponibles.")

    return X_train_final, X_val_final, imputer, v_selector, to_drop, scaler, boruta_selector

def find_optimal_threshold(y_true, y_probs):
    thresholds = np.linspace(0, 1, 100)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = [matthews_corrcoef(y_true, (y_probs >= t).astype(int)) for t in thresholds]
    
    best_threshold = thresholds[np.argmax(scores)]
    return best_threshold

def optimize_hyperparameters(X_train, y_train):
    print("\n--- PASO 2: Iniciando GridSearch (Optimización del Motor) ---")
    
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    mcc_scorer = make_scorer(matthews_corrcoef)
    
    pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    xgb_base = xgb.XGBClassifier(
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        scoring=mcc_scorer,
        cv=3,
        n_jobs=-1,
        verbose=1 
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nResultados del GridSearch:")
    print(f"  > Mejores parámetros: {grid_search.best_params_}")
    print(f"  > Mejor MCC obtenido: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_

def execute_final_purist_validation(X_train_full, y_train_full, best_params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    mcc_scores = []
    opt_thresholds = []

    print(f"\n--- Iniciando CV Purista con parámetros: {best_params} ---")
    
    for i, (train_index, val_index) in enumerate(skf.split(X_train_full, y_train_full)):
        X_tr_raw, X_val_raw = X_train_full[train_index], X_train_full[val_index]
        y_tr, y_val = y_train_full[train_index], y_train_full[val_index]

        X_tr_ready, X_val_ready, _, _, _, _, _ = preprocess_pipeline(X_tr_raw, X_val_raw, y_tr)

        pos_weight = np.sum(y_tr == 0) / np.sum(y_tr == 1)
        
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_tr_ready, y_tr)

        y_val_proba = model.predict_proba(X_val_ready)[:, 1]
        
        best_t = find_optimal_threshold(y_val, y_val_proba)
        opt_thresholds.append(best_t)
        
        y_val_pred = (y_val_proba >= best_t).astype(int)
        
        auc_fold = roc_auc_score(y_val, y_val_proba)
        mcc_fold = matthews_corrcoef(y_val, y_val_pred)
        
        auc_scores.append(auc_fold)
        mcc_scores.append(mcc_fold)
        
        print(f"Fold {i+1} | AUC: {auc_fold:.4f} | MCC: {mcc_fold:.4f} | Umbral: {best_t:.2f}")

    print("\n" + "="*30)
    print("RESUMEN VALIDACIÓN PURISTA")
    print(f"Promedio AUC-ROC: {np.mean(auc_scores):.4f}")
    print(f"Promedio MCC:     {np.mean(mcc_scores):.4f}")
    print(f"Umbral Medio:     {np.mean(opt_thresholds):.2f}")
    print("="*30)
    
    return auc_scores

def train_and_save_final_model(X_train_full, y_train_full, X_test, y_test, best_params, filename="tox21_model_ar.joblib"):
    print("\n--- PASO 4: Entrenamiento del Modelo de Producción Final ---")

    (X_train_final, X_test_final, imputer, v_selector, 
     to_drop, scaler, boruta_selector) = preprocess_pipeline(X_train_full, X_test, y_train_full)

    pos_weight = np.sum(y_train_full == 0) / np.sum(y_train_full == 1)
    
    final_model = xgb.XGBClassifier(
        **best_params,              
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss' 
    )

    final_model.fit(
        X_train_final, y_train_full,
        eval_set=[(X_train_final, y_train_full), (X_test_final, y_test)],
        verbose=False 
    )

    results = final_model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, results['validation_0']['logloss'], label='Entrenamiento (Train)', color='blue')
    plt.plot(x_axis, results['validation_1']['logloss'], label='Prueba (Test)', color='red')
    plt.legend()
    plt.ylabel('Log Loss')
    plt.xlabel('Número de Árboles (Iteraciones)')
    plt.title('Curva de Aprendizaje: Diagnóstico de Overfitting')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('grafica_4_learning_curve.png')
    plt.close()
    print("Gráfica de curva de aprendizaje guardada como 'grafica_4_learning_curve.png'")

    y_train_proba = final_model.predict_proba(X_train_final)[:, 1]
    optimal_t = find_optimal_threshold(y_train_full, y_train_proba)
    
    y_test_proba = final_model.predict_proba(X_test_final)[:, 1]
    y_test_pred = (y_test_proba >= optimal_t).astype(int) 

    test_auc = roc_auc_score(y_test, y_test_proba)
    test_mcc = matthews_corrcoef(y_test, y_test_pred) 
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
    test_pr_auc = auc(recall_vals, precision_vals)
    cm_perc = confusion_matrix(y_test, y_test_pred, normalize='true') * 100
    
    print(f"\n{'='*40}") 
    print(f"RESULTADOS FINALES EN TEST (MODELO OPTIMIZADO)")
    print(f"{'='*40}")    
    print(f"Umbral de decisión: {optimal_t:.3f}")
    print(f"AUC-ROC:           {test_auc:.4f}")
    print(f"PR-AUC:            {test_pr_auc:.4f}")
    print(f"MCC:               {test_mcc:.4f}")
    print(f"\nMatriz de Confusión (Normalizada %):")
    print(f"                  Pred. Neg | Pred. Pos")
    print(f"Real Negativo:    {cm_perc[0,0]:>8.1f}% | {cm_perc[0,1]:>8.1f}%")
    print(f"Real Positivo:    {cm_perc[1,0]:>8.1f}% | {cm_perc[1,1]:>8.1f}%")
    print(f"\nInforme de Clasificación:\n")
    print(classification_report(y_test, y_test_pred, target_names=['No Tóxico', 'Tóxico']))
    
    save_dict = {
        'model': final_model,
        'imputer': imputer,
        'v_selector': v_selector,
        'corr_to_drop': to_drop,
        'scaler': scaler,
        'boruta': boruta_selector,
        'threshold': optimal_t,
        'best_params': best_params 
    }
    joblib.dump(save_dict, filename)
    print(f"\nÉXITO: Modelo guardado en {filename}")
    
    return final_model, test_auc, X_test_final

def get_top_features_shap(model, X_data, feature_names, n=5):
    explainer = shap.TreeExplainer(model.get_booster())
    shap_values = explainer.shap_values(X_data)

    if isinstance(shap_values, list):
        shap_values_final = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_values_final = shap_values[:, :, 1]
    else:
        shap_values_final = shap_values

    global_importances = np.abs(shap_values_final).mean(axis=0)
    feature_names = np.array(feature_names)
    
    indices = np.argsort(global_importances)[-n:][::-1]
    return feature_names[indices].tolist()

def write_descriptor_explanations(top_descriptors, filename='explicacion_quimica.txt'):
    desc_explanations = {
        'MaxAbsEStateIndex': 'Maximum Absolute E-State Index: Measures the electronic state and reactivity of specific atoms in the molecule, often related to toxicophores.',
        'FpDensityMorgan3': 'Morgan Fingerprint Density (Radius 3): Indicates molecular complexity by comparing the number of structural patterns to the total number of atoms.',
        'BCUT2D_MWLOW': 'Lower Bound of 2D Molecular Weight: A topological descriptor that relates molecular weight to the atoms connectivity and distribution.',
        'HallKierAlpha': 'Hall-Kier Alpha: Describes molecular shape and flexibility, influencing how easily a molecule can fit into a biological receptor like NR-AR.',
        'BCUT2D_CHGLO': 'Lower Bound of 2D Partial Charge: Represents the distribution of low atomic charges across the molecule, relevant for electrostatic interactions.',
        'SPS': 'Spatial Polarity Score: Quantifies the distribution of polar atoms in 3D space, which is critical for membrane permeability and target binding.',
        'BalabanJ': 'Balaban J Index: A highly sophisticated topological index representing molecular branching. It helps distinguish between isomers with different toxicities.',
        'MolWt': 'Molecular Weight: Total mass of the molecule. It is a fundamental property that affects absorption and the ability to cross biological barriers.',
        'PEOE_VSA7': 'Partial Charge Surface Area (PEOE VSA): Measures the surface area associated with specific partial charge ranges, key for protein-ligand binding.',
        'BCUT2D_CHGHI': 'Upper Bound of 2D Partial Charge: Represents the distribution of high atomic charges, highlighting areas of the molecule likely to undergo chemical reactions.',
        'LogP': 'Lipophilicity (LogP): Measures how well a drug dissolves in fats vs. water, critical for predicting bioaccumulation.',
        'TPSA': 'Topological Polar Surface Area: Predicts how well a molecule will cross cell membranes.'
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Top 5 Most Important Molecular Descriptors (SHAP Analysis)\n")
        f.write("==========================================================\n\n")
        
        for i, desc in enumerate(top_descriptors, 1):
            if desc in desc_explanations:
                explanation = desc_explanations[desc]
            elif desc.startswith('fr_'):
                explanation = f"Functional Group Fragment ({desc.replace('fr_', '')}): The presence of this specific group is statistically linked to toxicity."
            elif 'MorganBit' in desc:
                explanation = "Structural Fingerprint Bit: Represents a specific local molecular environment or pattern."
            else:
                explanation = f"Physicochemical Property ({desc}): A molecular descriptor identified as relevant for the NR-AR toxicological endpoint."
            
            f.write(f"{i}. {desc}: {explanation}\n")
    
    print(f"Informe generado: {filename} con el análisis del Top {len(top_descriptors)}")

def generate_evaluation_plots(model, X_test, y_test, feature_names, threshold):
    print("\n--- Generando gráficas para la presentación ---")
    
    # Predecir probabilidades una sola vez para eficiencia
    probas = model.predict_proba(X_test)[:, 1]
    y_pred = (probas >= threshold).astype(int)

    # 1. Matriz de Confusión
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt=".2%", cmap='Blues', 
                xticklabels=['No Tóxico', 'Tóxico'], yticklabels=['No Tóxico', 'Tóxico'])
    plt.title('Matriz de Confusión (Normalizada)')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.savefig('grafica_1_confusion.png')
    plt.close()

    # 2. Curva Precision-Recall
    plt.figure(figsize=(6, 5))
    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR-AUC = {auc(recall, precision):.2f}')
    plt.xlabel('Recall (Sensibilidad)')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend()
    plt.savefig('grafica_2_precision_recall.png')
    plt.close()

    # 3. SHAP (Versión Limpia)
    try:
        # Usamos el modelo directamente. SHAP maneja la extracción del Booster.
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Ajuste de dimensiones: XGBoost en clasificación binaria a veces devuelve 
        # una lista [shap_neg, shap_pos] o una matriz 2D/3D.
        if isinstance(shap_values, list):
            shap_values_to_plot = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_to_plot = shap_values[:, :, 1]
        else:
            shap_values_to_plot = shap_values

        plt.figure(figsize=(10, 6))
        # Importante: X_test debe ser un array de NumPy si feature_names se pasa aparte
        shap.summary_plot(shap_values_to_plot, X_test, feature_names=feature_names, show=False)
        plt.title('Interpretación SHAP: Impacto de Descriptores')
        plt.tight_layout()
        plt.savefig('grafica_3_shap.png')
        plt.close()
    except Exception as e:
        print(f"Aviso: No se pudo generar la gráfica SHAP detallada. Error: {e}")
    
    print("Gráficas guardadas correctamente.")

if __name__ == "__main__":
    df_raw = download_tox21()
    df_clean = preprocess_tox21(df_raw, target_col='NR-AR')

    plt.figure(figsize=(6, 4))
    sns.countplot(x='NR-AR', data=df_clean, palette='viridis')
    plt.title('Distribución de la Clase (Target: NR-AR)')
    plt.xlabel('0: No Tóxico | 1: Tóxico')
    plt.ylabel('Número de Moléculas')
    plt.savefig('grafica_0_desbalance.png')
    plt.close()
    print("Gráfica de desbalance guardada.")

    features_list = [extract_all_features(s) for s in df_clean['smiles_clean']]
    X = prepare_matrix(features_list)
    y = df_clean['NR-AR'].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Antes del downsampling - Clase 0: {np.sum(y_train_full==0)}, Clase 1: {np.sum(y_train_full==1)}")
    X_train_full, y_train_full = apply_downsampling(X_train_full, y_train_full)
    print(f"Tras el downsampling - Clase 0: {np.sum(y_train_full==0)}, Clase 1: {np.sum(y_train_full==1)}")

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y_train_full, palette='viridis')    
    plt.title('Distribución de la Clase tras Downsampling (Train Set)')
    plt.xlabel('0: No Tóxico | 1: Tóxico')
    plt.ylabel('Número de Moléculas')
    plt.savefig('grafica_0_balanceado.png')
    plt.close()
    print("Gráfica de set balanceado guardada como 'grafica_0_balanceado.png'.")

    print("\n--- Ejecutando Selección de Características Global (Referencia) ---")
    (X_train_reduced, X_test_reduced, imputer, v_selector, 
     to_drop, scaler, boruta_selector) = preprocess_pipeline(X_train_full, X_test, y_train_full)

    best_params = optimize_hyperparameters(X_train_reduced, y_train_full)

    print("\n--- Validando metodología completa (Selección in-fold / Sin Leakage) ---")
    cv_scores = execute_final_purist_validation(X_train_full, y_train_full, best_params)
    avg_cv_auc = np.mean(cv_scores)

    final_model, test_auc, X_test_final = train_and_save_final_model(
        X_train_full, y_train_full, X_test, y_test, best_params
    )

    saved_data = joblib.load("tox21_model_ar.joblib")
    final_threshold = saved_data.get('threshold', 0.5) 

    all_feature_names = np.array(
        [f"MorganBit_{i}" for i in range(2048)] + 
        [name for name, _ in Fragments.__dict__.items() if name.startswith('fr_')] + 
        [name for name, _ in Descriptors.descList]
    )
    names_filt = all_feature_names[saved_data['v_selector'].get_support()]
    names_filt = np.delete(names_filt, saved_data['corr_to_drop'])
    final_feature_names = names_filt[saved_data['boruta'].support_]

    print("\n--- Calculando importancia de descriptores con SHAP ---")
    top_10 = get_top_features_shap(final_model, X_test_final, final_feature_names, n=10)
    print("\nTOP 10 DESCRIPTORES SEGÚN SHAP:")
    for i, name in enumerate(top_10, 1):
        print(f"{i}. {name}")

    write_descriptor_explanations(top_10[:5], filename='explicacion_quimica.txt')

    generate_evaluation_plots(final_model, X_test_final, y_test, final_feature_names, final_threshold)

    print("\n--- PROCESO COMPLETADO ---")
    print(f"1. AUC Medio de Validación (Purista): {avg_cv_auc:.4f}")
    print(f"2. AUC Final en Test (Sin Leakage): {test_auc:.4f}")
    print(f"3. Umbral de clasificación optimizado (MCC): {final_threshold:.3f}") 
    print(f"4. Informe de descriptores generado: explicacion_quimica.txt")