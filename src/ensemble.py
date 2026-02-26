import os
import argparse
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import log_loss, roc_auc_score
from scipy.optimize import minimize

def load_predictions(model_names, data_dir='data'):
    """
    Charge les OOF (Out-Of-Fold) pour l'ensemble et les test_preds pour la soumission.
    """
    oof_dfs = []
    test_dfs = []
    
    proc_dir = os.path.join(data_dir, 'processed')
    
    for m in model_names:
        oof_path = os.path.join(proc_dir, f"oof_{m}.csv")
        test_path = os.path.join(proc_dir, f"test_{m}.csv")
        
        if os.path.exists(oof_path) and os.path.exists(test_path):
            oof_dfs.append(pd.read_csv(oof_path).rename(columns={f'pred_{m}': m}).set_index('id'))
            test_dfs.append(pd.read_csv(test_path).rename(columns={f'pred_{m}': m}).set_index('id'))
        else:
            print(f"ATTENTION: Prédictions manquantes pour {m}. Avez-vous exécuté 'python src/train.py --model {m}' ?")
            
    # Concatenation selon l'ID
    OOF = pd.concat(oof_dfs, axis=1) if oof_dfs else pd.DataFrame()
    TEST = pd.concat(test_dfs, axis=1) if test_dfs else pd.DataFrame()
    
    return OOF, TEST

from sklearn.linear_model import LogisticRegression

def train_stacking_meta_model(OOF, y_true):
    """
    Entraîne un meta-modèle (Logistic Regression) sur les prédictions OOF.
    C'est plus puissant que le simple Blend car il apprend à corriger les biais de chaque modèle.
    """
    print("\n--- Entraînement du Meta-Modèle de Stacking (Logistic Regression) ---")
    
    # On utilise une régression logistique simple (souvent la meilleure en meta-modèle)
    # avec une régularisation L2 pour éviter d'overfitter sur les probabilités.
    meta_model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
    meta_model.fit(OOF, y_true)
    
    # Score OOF du meta-modèle
    meta_oof_preds = meta_model.predict_proba(OOF)[:, 1]
    score = roc_auc_score(y_true, meta_oof_preds)
    
    print(f"AUC OOF du Meta-Modèle : {score:.5f}")
    
    # Affichage des coefficients (importance relative des modèles)
    print("Coefficients du Meta-Modèle :")
    for m, coef in zip(OOF.columns, meta_model.coef_[0]):
        print(f" - {m.upper()} : {coef:.4f}")
        
    return meta_model

def create_submission_stacking(TEST, meta_model, output_path='../submission.csv'):
    """
    Utilise le meta-modèle pour prédire sur le Test Set.
    """
    final_preds = meta_model.predict_proba(TEST)[:, 1]
    
    sub_df = pd.DataFrame({
        'id': TEST.index,
        'Heart Disease': final_preds
    })
    
    sub_df.to_csv(output_path, index=False)
    print(f"\n[SUCCÈS] Soumission STACKING générée : {output_path}")

def create_submission(TEST, best_weights, output_path='../data/submission.csv'):
    """
    Applique les poids optimaux sur les prédictions du fichier test.csv
    et génère le format attendu par Kaggle.
    """
    final_preds = np.zeros(len(TEST))
    for m, w in best_weights.items():
        final_preds += w * TEST[m]
        
    # ATTENTION : Kaggle évalue sur l'AUC (Area Under the ROC Curve).
    # Ils s'attendent donc à recevoir des PROBABILITÉS (ex: 0.85, 0.12) et SURTOUT PAS des classes 0 ou 1.
    # On n'arrondit donc RIEN du tout ici !
    
    sub_df = pd.DataFrame({
        'id': TEST.index,
        'Heart Disease': final_preds
    })
    
    sub_df.to_csv(output_path, index=False)
    print(f"\n[SUCCÈS] Soumission générée : {output_path}")
    print("Prête à être uploadée sur Kaggle !")

if __name__ == "__main__":
    # Liste de nos cadors (On ajoute HistGrad pour la diversité !)
    model_list = ['lgbm', 'catboost', 'xgb', 'hist_grad']
    
    data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'
    
    # 1. Chargement OOF et Test
    OOF, TEST = load_predictions(model_list, data_dir)
    
    if OOF.empty or TEST.empty:
        print("Erreur : Impossible de builder l'ensemble, aucun fichier de prédiction trouvé.")
    else:
        # 2. Chargement de la Target
        # On détecte automatiquement si on a utilisé le pseudo-labeling
        train_filename = "train_pseudo.csv" if os.path.exists(os.path.join(data_dir, "train_pseudo.csv")) else "train.csv"
        train_path = os.path.join(data_dir, train_filename)
        
        print(f"Alignement automatique sur {train_path}...")
        train_df = pd.read_csv(train_path)
        
        # On mappe la target
        train_df['target'] = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
        
        # --- ALIGNEMENT ROBUSTE ---
        # On s'assure que OOF et y_true correspondent exactement via l'ID
        # (Indispensable si on mélange des modèles entraînés avec/sans pseudo-labels)
        common_ids = OOF.index.intersection(train_df['id'])
        
        if len(common_ids) == 0:
            print("ERREUR CRITIQUE : Aucun ID commun entre les prédictions et le fichier train.")
        else:
            if len(common_ids) < len(OOF):
                print(f"⚠️ Alignement : {len(common_ids)} ids trouvés sur {len(OOF)} prédictions.")
            
            # On filtre et on trie pour garantir l'ordre
            OOF_aligned = OOF.loc[common_ids]
            y_true = train_df.set_index('id').loc[common_ids]['target'].values
            
            # 3. Entraînement du Meta-Modèle (Stacking)
            meta_model = train_stacking_meta_model(OOF_aligned, y_true)
            
            # 4. Création finale de la soumission
            # On remonte d'un cran si on est dans src/
            output_p = '../submission.csv' if os.path.exists('../src') else 'submission.csv'
            create_submission_stacking(TEST, meta_model, output_path=output_p)
