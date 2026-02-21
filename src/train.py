import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import joblib

from features import HeartDiseaseFeatureEngineer
from models import LightGBMWrapper, XGBoostWrapper, CatBoostWrapper

# Dictionnaire pour appeler nos classes par leur nom court
MODEL_ZOO = {
    'lgbm': LightGBMWrapper,
    'xgb': XGBoostWrapper,
    'catboost': CatBoostWrapper
}

def train_and_eval(model_name, train_path, test_path, n_splits=5, seeds=[42, 43, 44]):
    """
    Entraîne un modèle spécifique en Validation Croisée (K-Fold Stratifié) sur plusieurs Seeds.
    Génère les prédictions "Out-Of-Fold" (OOF) pour l'ensembling futur 
    ainsi que les prédictions moyennes sur le Test Set.
    """
    print(f"--- Démarrage de l'entraînement avec {model_name.upper()} (Seed Averaging sur {len(seeds)} seeds) ---")
    
    # 1. Chargement des données
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 2. Pipeline de Feature Engineering
    print("Application du Feature Engineering...")
    # On force l'utilisation des types catégoriels pandas pour que CatBoost/LGBM s'en servent
    fe = HeartDiseaseFeatureEngineer(use_categories=True)
    
    # Apprentissage des features (pas crucial de séparer fit/transform ici car pas de state comme un StandardScaler)
    train_feat = fe.transform(train_df)
    test_feat = fe.transform(test_df)
    
    # Séparation Features (X) / Target (y)
    target_col = 'Heart Disease'
    # Encodage binaire 1/0
    train_feat[target_col] = train_feat[target_col].map({'Presence': 1, 'Absence': 0})
    
    y = train_feat[target_col]
    X = train_feat.drop(columns=['id', target_col])
    X_test = test_feat.drop(columns=['id'])
    
    # S'assurer que le test a exactement les mêmes colonnes
    X_test = X_test[X.columns]
    
    # Tableaux pour stocker les résultats OOF et Test finaux (moyennés sur toutes les seeds)
    oof_preds_final = np.zeros(len(X))
    test_preds_final = np.zeros(len(X_test))
    
    for seed in seeds:
        print(f"\n===== Entraînement avec la Seed : {seed} =====")
        oof_preds_seed = np.zeros(len(X))
        test_preds_seed = np.zeros(len(X_test))
        
        # Validation Croisée Stratifiée avec la seed spécifique
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  [Fold {fold+1}/{n_splits}]", end="")
            
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
            
            # Initialisation du modèle avec la seed spécifique (random_state pour LGBM/XGB, random_seed pour CatBoost)
            model_class = MODEL_ZOO[model_name]
            clf = model_class({'random_state': seed, 'random_seed': seed})
            
            # Entraînement avec early stopping sur le fold de validation
            clf.fit(X_tr, y_tr, X_va, y_va)
            
            # Prédiction sur la validation (OOF)
            val_preds = clf.predict_proba(X_va)
            oof_preds_seed[val_idx] = val_preds
            
            # Calcul de la métrique pour le suivi
            fold_auc = roc_auc_score(y_va, val_preds)
            print(f" -> AUC: {fold_auc:.4f}")
            metrics.append(fold_auc)
            
            # Prédiction sur le Test set (Averaging au sein de la seed)
            test_preds_seed += clf.predict_proba(X_test) / n_splits
            
        print(f"  Moyenne AUC CV (Seed {seed}) : {np.mean(metrics):.4f}")
        
        # Ajout aux ensembles finaux moyennés sur toutes les seeds
        oof_preds_final += oof_preds_seed / len(seeds)
        test_preds_final += test_preds_seed / len(seeds)

    print(f"\n--- Fin de {model_name.upper()} ---")
    print(f"AUC Out-Of-Fold Globale (Après Seed Averaging) : {roc_auc_score(y, oof_preds_final):.4f}")
    
    # 4. Sauvegarde des prédictions OOF et Test pour l'ensembling
    # On déduit le dossier de sauvegarde selon où on exécute
    base_dir = os.path.dirname(os.path.dirname(train_path)) if 'data' in train_path else '.'
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    # On sauve les prédictions d'entrainement pour le modèle "Stacking" / "Ensemble"
    pd.DataFrame({'id': train_df['id'], f'pred_{model_name}': oof_preds_final}).to_csv(os.path.join(proc_dir, f'oof_{model_name}.csv'), index=False)
    
    # On sauve les prédictions sur le test (moyennées)
    pd.DataFrame({'id': test_df['id'], f'pred_{model_name}': test_preds_final}).to_csv(os.path.join(proc_dir, f'test_{model_name}.csv'), index=False)
    
    print(f"Prédictions sauvegardées dans '{proc_dir}' pour l'ensembling.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['lgbm', 'xgb', 'catboost', 'all'], 
                        help="Modèle à entraîner (lgbm, xgb, catboost, ou all)")
    args = parser.parse_args()
    
    data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'
    train_p = f"{data_dir}/train.csv"
    test_p = f"{data_dir}/test.csv"
    
    if args.model == 'all':
        for m in ['lgbm', 'xgb', 'catboost']:
            train_and_eval(m, train_p, test_p)
    else:
        train_and_eval(args.model, train_p, test_p)
