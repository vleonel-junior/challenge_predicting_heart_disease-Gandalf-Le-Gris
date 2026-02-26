import os
import json
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class BaseModel(ABC):
    """
    Classe de base abstraite (Interface) pour tous nos modèles.
    Garantit que chaque modèle implémentera `fit` et `predict_proba`.
    """
    def __init__(self, params=None, load_best=True):
        self.params = params if params is not None else {}
        # Extraction du flag GPU si présent
        self.use_gpu = self.params.pop('use_gpu', False)
        self.model = None
        self.load_best = load_best

    def _load_best_params(self, model_name):
        """
        Cherche si un fichier JSON d'hyperparamètres optimisés par Optuna existe pour ce modèle.
        Si oui, il écrase les paramètres par défaut avec les paramètres optimisés.
        """
        if not self.load_best:
            return
        
        # On regarde dans le dossier d'exécution ou dans le dossier parent (selon si on lance depuis src/ ou racine)
        possible_paths = [
            f'models/best_params_{model_name}.json', 
            f'../models/best_params_{model_name}.json'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[INFO] Paramètres optimisés par Optuna trouvés pour {model_name} ! Chargement depuis {path}...")
                with open(path, 'r') as f:
                    best_params = json.load(f)
                    
                # On merge. IMPORTANT : On ne veut pas écraser les paramètres explicites 
                # (comme 'random_state' ou 'random_seed' fournis par le Seed Averaging)
                for k, v in best_params.items():
                    if k not in self.params:
                        self.params[k] = v
                break

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass


class LightGBMWrapper(BaseModel):
    """
    Wrapper pour LightGBM.
    """
    def __init__(self, params=None, load_best=True):
        super().__init__(params, load_best=load_best)
        self._load_best_params('lgbm')
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Paramètres par défaut robustes pour le tabulaire
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03, # Baisse du LR pour une meilleure convergence
            'num_leaves': 31,
            'colsample_bytree': 0.8, # Subsampling des colonnes (évite l'overfit)
            'subsample': 0.8, # Subsampling des lignes
            'min_child_samples': 20,
            'verbose': -1,
            'random_state': 42
        }
        
        if self.use_gpu:
            default_params.update({
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'gpu_use_dp': False
            })

        default_params.update(self.params)
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            self.model = lgb.train(
                default_params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dtrain, dval],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
            )
        else:
            self.model = lgb.train(default_params, dtrain, num_boost_round=100)
            
        return self

    def predict_proba(self, X):
        return self.model.predict(X)


class XGBoostWrapper(BaseModel):
    """
    Wrapper pour XGBoost.
    """
    def __init__(self, params=None, load_best=True):
        super().__init__(params, load_best=load_best)
        self._load_best_params('xgb')

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.03,
            'max_depth': 6,
            'colsample_bytree': 0.8,
            'subsample': 0.8,
            'min_child_weight': 1,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        if self.use_gpu:
            default_params.update({
                'tree_method': 'hist',
                'device': 'cuda'
            })
        else:
            default_params['tree_method'] = 'hist'

        default_params.update(self.params)
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
            self.model = xgb.train(
                default_params,
                dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
        else:
            self.model = xgb.train(default_params, dtrain, num_boost_round=100)
            
        return self

    def predict_proba(self, X):
        # On utilise gpu_predictor si on est en gpu_hist pour accélérer aussi l'inférence
        dtest = xgb.DMatrix(X, enable_categorical=True)
        return self.model.predict(dtest)


class CatBoostWrapper(BaseModel):
    """
    Wrapper pour CatBoost, le spécialiste des données tabulaires et catégorielles.
    """
    def __init__(self, params=None, cat_features=None, load_best=True):
        super().__init__(params, load_best=load_best)
        self.cat_features = cat_features
        self._load_best_params('catboost')

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        default_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'learning_rate': 0.03,
            'iterations': 1500, # Augmenté car LR plus faible
            'depth': 6,
            'l2_leaf_reg': 3, # Regularisation L2 robuste
            'random_seed': 42,
            'verbose': False
        }
        
        if self.use_gpu:
            default_params.update({
                'task_type': 'GPU',
                'devices': '0',
                'bootstrap_type': 'Bayesian'
            })
        else:
            default_params['task_type'] = 'CPU'

        default_params.update(self.params)
        
        # Identification automatique des colonnes catégorielles si non spécifiées
        if self.cat_features is None:
            self.cat_features = list(X_train.select_dtypes(include=['category', 'object']).columns)
            
        self.model = CatBoostClassifier(**default_params)
        
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                cat_features=self.cat_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train, cat_features=self.cat_features, verbose=False)
            
        return self

    def predict_proba(self, X):
        # Retourne uniquement la probabilité de la classe 1 (présence de maladie)
        return self.model.predict_proba(X)[:, 1]


from sklearn.ensemble import HistGradientBoostingClassifier

class HistGradWrapper(BaseModel):
    """
    Wrapper pour HistGradientBoostingClassifier (Scikit-learn).
    Apporte de la diversité à l'ensemble.
    """
    def __init__(self, params=None, load_best=True):
        super().__init__(params, load_best=load_best)
        self._load_best_params('hist_grad')

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        default_params = {
            'max_iter': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'random_state': 42
        }
        default_params.update(self.params)
        
        # HistGrad supporte les catégories nativement si on lui dit
        cat_indices = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype.name == 'category']
        if cat_indices:
            default_params['categorical_features'] = cat_indices
            
        self.model = HistGradientBoostingClassifier(**default_params)
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
