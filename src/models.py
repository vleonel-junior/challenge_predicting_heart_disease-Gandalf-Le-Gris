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
    def __init__(self, params=None):
        self.params = params if params is not None else {}
        self.model = None

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
    def __init__(self, params=None):
        super().__init__(params)
        
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
    def __init__(self, params=None):
        super().__init__(params)

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
        dtest = xgb.DMatrix(X, enable_categorical=True)
        return self.model.predict(dtest)


class CatBoostWrapper(BaseModel):
    """
    Wrapper pour CatBoost, le spécialiste des données tabulaires et catégorielles.
    """
    def __init__(self, params=None, cat_features=None):
        super().__init__(params)
        self.cat_features = cat_features

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        default_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'learning_rate': 0.03,
            'iterations': 1500, # Augmenté car LR plus faible
            'depth': 6,
            'l2_leaf_reg': 3, # Regularisation L2 robuste
            'subsample': 0.8,
            'random_seed': 42,
            'verbose': False
        }
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

class AutoGluonWrapper(BaseModel):
    """
    Wrapper pour AutoGluon. Fait l'AutoML (feature generation, stack ensembles, etc.) sur un temps donné.
    """
    def __init__(self, params=None):
        super().__init__(params)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from autogluon.tabular import TabularPredictor
        import os
        import numpy as np
        
        train_data = X_train.copy()
        train_data['target'] = y_train
        
        # Dossier unique par fit pour éviter que les Folds de CV de train.py ne s'écrasent
        save_path = f"AutogluonModels/fold_{np.random.randint(100000)}"
        # Pour AutoGluon, le secret c'est "best_quality" (qui fait du bagging et du stacking interne).
        # Attention : best_quality est TRES gourmand en temps
        presets = self.params.get('presets', 'best_quality')
        # On passe à 3600 secondes (1 heure) par fold par défaut pour lui laisser déployer ses ailes
        time_limit = self.params.get('time_limit', 3600)  
        
        # On ajoute verbosity=1 pour voir ce qu'il fait, et on force num_gpus 'auto' (ray gère) 
        # ou au moins d'autoriser les modèles à détecter le CUDA.
        self.model = TabularPredictor(label='target', eval_metric='roc_auc', path=save_path, verbosity=1).fit(
            train_data=train_data,
            presets=presets,
            time_limit=time_limit,
            num_gpus=1 # Autorise l'utilisation d'1 GPU par modèle si dispo
        )
        return self

    def predict_proba(self, X):
        # AutoGluon retourne un dataframe avec les probas pour chaque classe. On prend la classe '1'.
        proba = self.model.predict_proba(X)
        if 1 in proba.columns:
            return proba[1].values
        else:
            return proba.iloc[:, 1].values
