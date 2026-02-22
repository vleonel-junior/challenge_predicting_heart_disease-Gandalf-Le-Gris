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
from models import LightGBMWrapper, XGBoostWrapper, CatBoostWrapper

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

def objective_lgbm(trial, X, y, use_gpu=False):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 8, 256),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'verbose': -1
    }
    
    if use_gpu:
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
    
    return evaluate_model(LightGBMWrapper(params, load_best=False), X, y)

def objective_xgb(trial, X, y, use_gpu=False):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 20.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'random_state': 42
    }
    
    if use_gpu:
        params['tree_method'] = 'gpu_hist'
        # params['device'] = 'cuda' # For XGBoost 2.0+
    else:
        params['tree_method'] = 'hist'
    
    return evaluate_model(XGBoostWrapper(params, load_best=False), X, y)

def objective_catboost(trial, X, y, use_gpu=False):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False
    }
    
    if use_gpu:
        params['task_type'] = 'GPU'
        params['devices'] = '0'
    else:
        params['task_type'] = 'CPU'
    
    cat_features = list(X.select_dtypes(include=['category', 'object']).columns)
    
    return evaluate_model(CatBoostWrapper(params, cat_features=cat_features, load_best=False), X, y)

def evaluate_model(wrapper, X, y):
    """
    Entraîne le wrapper du modèle sur 3 Folds stricts et renvoie l'AUC OOF Moyen (pour Optuna).
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        
        # Le Wrapper gère son propre early stopping automatiquement avec le val_set
        wrapper.fit(X_tr, y_tr, X_va, y_va)
        
        # Le wrapper renvoie directement un array 1D de probabilités
        oof_preds[val_idx] = wrapper.predict_proba(X_va)
        
    return roc_auc_score(y, oof_preds)


def tune_model(model_name, n_trials=50, use_gpu=False):
    print(f"\n--- Démarrage de l'Optimisation des Hyperparamètres : {model_name.upper()} ---")
    if use_gpu:
        print("🚀 Accélération GPU ACTIVÉE")
    
    data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'
    train_p = f"{data_dir}/train.csv"
    
    print("Chargement et Feature Engineering...")
    X, y = load_and_preprocess(train_p)
    
    study = optuna.create_study(direction='maximize', study_name=f'tune_{model_name}')
    
    if model_name == 'lgbm':
        study.optimize(lambda trial: objective_lgbm(trial, X, y, use_gpu=use_gpu), n_trials=n_trials)
    elif model_name == 'xgb':
        study.optimize(lambda trial: objective_xgb(trial, X, y, use_gpu=use_gpu), n_trials=n_trials)
    elif model_name == 'catboost':
        study.optimize(lambda trial: objective_catboost(trial, X, y, use_gpu=use_gpu), n_trials=n_trials)
        
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
    parser.add_argument("--gpu", type=bool, default=True, help="Activer l'accélération GPU pour le tuning (True par défaut pour Kaggle)")
    args = parser.parse_args()
    
    tune_model(args.model, args.trials, use_gpu=args.gpu)
