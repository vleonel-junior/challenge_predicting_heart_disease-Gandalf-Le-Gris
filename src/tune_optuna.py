import os
import argparse
import json
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import HeartDiseaseFeatureEngineer

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def load_and_preprocess(train_path):
    train_df = pd.read_csv(train_path)
    # Important : Forcer le mode target binaire Presence=1 pour l'optimisation
    fe = HeartDiseaseFeatureEngineer(use_categories=True)
    train_feat = fe.transform(train_df)
    
    target_col = 'Heart Disease'
    train_feat[target_col] = train_feat[target_col].map({'Presence': 1, 'Absence': 0})
    
    y = train_feat[target_col].values
    X = train_feat.drop(columns=['id', target_col])
    
    return X, y

def objective_lgbm(trial, X, y):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_estimators': 1000 # On laisse beaucoup d'arbres, l'early stopping fera le reste
    }
    
    return evaluate_model(LGBMClassifier(**params), X, y, fit_params={'eval_metric': 'auc'})

def objective_xgb(trial, X, y):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'enable_categorical': True,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_estimators': 1000
    }
    
    # xgb bugge avec verbose parfois dans optuna s'il n'y a pas l'early stopping précis
    return evaluate_model(XGBClassifier(**params), X, y)

def objective_catboost(trial, X, y):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_seed': 42,
        'iterations': 1000,
        'verbose': False
    }
    
    cat_features = list(X.select_dtypes(include=['category', 'object']).columns)
    
    # Pas besoin de fit_params complexes pour Catboost, on l'initie et on fit direct
    return evaluate_model(CatBoostClassifier(**params), X, y, fit_params={'cat_features': cat_features, 'verbose': False})

def evaluate_model(clf, X, y, fit_params=None):
    """
    Entraîne le modèle sur 3 Folds stricts et renvoie l'AUC OOF Moyen (pour Optuna).
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    fit_p = fit_params if fit_params else {}
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        
        clf.fit(X_tr, y_tr, **fit_p)
        
        # Obtenir juste la probabilité de la classe 1
        oof_preds[val_idx] = clf.predict_proba(X_va)[:, 1]
        
    return roc_auc_score(y, oof_preds)


def tune_model(model_name, n_trials=50):
    print(f"\n--- Démarrage de l'Optimisation des Hyperparamètres : {model_name.upper()} ---")
    
    data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'
    train_p = f"{data_dir}/train.csv"
    
    print("Chargement et Feature Engineering...")
    X, y = load_and_preprocess(train_p)
    
    study = optuna.create_study(direction='maximize', study_name=f'tune_{model_name}')
    
    if model_name == 'lgbm':
        study.optimize(lambda trial: objective_lgbm(trial, X, y), n_trials=n_trials)
    elif model_name == 'xgb':
        study.optimize(lambda trial: objective_xgb(trial, X, y), n_trials=n_trials)
    elif model_name == 'catboost':
        study.optimize(lambda trial: objective_catboost(trial, X, y), n_trials=n_trials)
        
    print(f"\n[SUCCÈS] Meilleure combinaison trouvée pour {model_name.upper()} : AUC = {study.best_value:.5f}")
    
    # --- SAUVEGARDE DES PARAMÈTRES ---
    best_params = study.best_params
    print("Paramètres trouves :", best_params)
    
    models_dir = '../models' if os.path.exists('../data/train.csv') else 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    save_path = f"{models_dir}/best_params_{model_name}.json"
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print(f"\n✅ Les meilleurs hyperparamètres ont été sauvegardés automatiquement dans : {save_path}")
    print("models.py les chargera tout seul lors du prochain entraînement !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['lgbm', 'xgb', 'catboost'], 
                        help="Modèle à tuner (lgbm, xgb, ou catboost)")
    parser.add_argument("--trials", type=int, default=30, help="Nombre d'essais (plus = meilleur mais + long)")
    args = parser.parse_args()
    
    tune_model(args.model, args.trials)
