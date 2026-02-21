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

def optimize_weights_optuna(OOF, y_true):
    """
    Utilise Optuna pour trouver les poids de "Blend" parfaits qui maximisent l'AUC Score global
    sur les prédictions Out-Of-Fold (OOF).
    """
    print("\n--- Début de l'optimisation Optuna des poids d'Ensemble ---")
    
    def objective(trial):
        # On définit un poids pour chaque modèle entre 0 et 1
        weights = [trial.suggest_float(f'w_{m}', 0, 1) for m in OOF.columns]
        
        # Normalisation pour que la somme fasse 1
        total_w = sum(weights)
        if total_w == 0:
            return 0.5 # Pénalité (AUC aléatoire)
        weights = [w / total_w for w in weights]
        
        # Prédiction finale 'blended' = somme pondérée
        blended_preds = np.zeros(len(OOF))
        for i, col in enumerate(OOF.columns):
            blended_preds += weights[i] * OOF[col]
            
        # Dans des compétitions, l'AUC ou le LogLoss est souvent la cible
        score = roc_auc_score(y_true, blended_preds)
        return score
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    
    best_weights_raw = [study.best_params[f'w_{m}'] for m in OOF.columns]
    
    # Normaliser les meilleurs poids
    total_w = sum(best_weights_raw)
    best_weights = {m: (w / total_w) for m, w in zip(OOF.columns, best_weights_raw)}
    
    print(f"\nMeilleure combinaison AUC OOF trouvée : {study.best_value:.5f}")
    print("Poids idéaux pour la Soumission :")
    for m, w in best_weights.items():
        print(f" - {m.upper()} : {w:.2%}")
        
    return best_weights

def create_submission(TEST, best_weights, output_path='../data/submission.csv'):
    """
    Applique les poids optimaux sur les prédictions du fichier test.csv
    et génère le format attendu par Kaggle.
    """
    final_preds = np.zeros(len(TEST))
    for m, w in best_weights.items():
        final_preds += w * TEST[m]
        
    # La compétition demande des entiers '0' ou '1'. 
    # Pour un AUC de 0.5 optimal, on peut chercher le meilleur seuil sur le Train,
    # mais 0.5 est le standard par défaut pour de la log_loss.
    final_class = (final_preds > 0.5).astype(int)
    
    sub_df = pd.DataFrame({
        'id': TEST.index,
        'Heart Disease': final_class
    })
    
    sub_df.to_csv(output_path, index=False)
    print(f"\n[SUCCÈS] Soumission générée : {output_path}")
    print("Prête à être uploadée sur Kaggle !")

if __name__ == "__main__":
    # Liste de nos cadors
    model_list = ['lgbm', 'catboost', 'xgb']
    
    data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'
    
    # 1. Chargement OOF et Test
    OOF, TEST = load_predictions(model_list, data_dir)
    
    if OOF.empty or TEST.empty:
        print("Erreur : Impossible de builder l'ensemble, aucun fichier de prédiction trouvé.")
    else:
        # 2. Chargement de la vraie Target d'entrainement pour optimiser par rapport aux OOF
        train_df = pd.read_csv(f"{data_dir}/train.csv")
        y_true = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0}).values
        
        # 3. Calcul des Poids Optimaux avec Optuna
        best_w = optimize_weights_optuna(OOF, y_true)
        
        # 4. Création finale
        create_submission(TEST, best_w, output_path=f"../submission.csv")
