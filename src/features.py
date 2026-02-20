import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class HeartDiseaseFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to apply Feature Engineering steps
    identified during the EDA and Clinical Domain Study.
    """
    def __init__(self, use_categories=True):
        self.use_categories = use_categories
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        # Créer une copie pour ne pas modifier le dataframe original
        df = X.copy()
        
        # -------------------------------------------------------------------
        # 1. FEATURES CLINIQUES (Interactions Fortes)
        # -------------------------------------------------------------------
        
        # Déficit Chronotrope : La Max HR mesurée vs la Max HR théorique (220 - Age)
        # Un déficit négatif important indique que le coeur n'atteint pas son effort max théorique.
        df['Deficit_HR'] = df['Max HR'] - (220 - df['Age'])
        
        # Syndrome Ischémique Sévère (Interaction ST / Angine)
        # Croiser la dépression ST avec l'angine d'effort.
        df['Severe_Ischemia_Risk'] = df['ST depression'] * df['Exercise angina']
        
        # Ratio d'effort (plus il est bas, plus le risque est élevé)
        df['Effort_Ratio'] = df['Max HR'] / (df['ST depression'] + 1.0) # +1.0 pour éviter div by zero
        
        # -------------------------------------------------------------------
        # 2. BINNING / DISCRÉTISATION DU BRUIT
        # -------------------------------------------------------------------
        
        # Pression Artérielle (BP) -> Regroupement pour lisser les effets des arrondis médicaux
        df['BP_Group'] = pd.cut(df['BP'], 
                                bins=[0, 120, 130, 140, 150, 300], 
                                labels=['Optimum', 'Normal', 'High-Normal', 'Hypertension_S1', 'Hypertension_S2'])
        
        # Âge -> Regroupement par décennies
        df['Age_Group'] = pd.cut(df['Age'], 
                                 bins=[0, 40, 50, 60, 70, 100], 
                                 labels=['Under 40', '40s', '50s', '60s', 'Over 70'])
        
        # -------------------------------------------------------------------
        # 3. EXTRACTION CATÉGORIELLE ("Golden Features")
        # -------------------------------------------------------------------
        
        # Thallium : Extraire explicitement le défaut réversible (Type 7)
        df['Is_Reversible_Defect'] = (df['Thallium'] == 7).astype(int)
        
        # Vaisseaux : Binariser (0 vaisseaux atteints vs au moins 1)
        df['Has_Blocked_Vessels'] = (df['Number of vessels fluro'] > 0).astype(int)
        
        # Chest Pain : Isoler le type 4 (Asymptomatique / Haut risque)
        df['Is_Chest_Pain_Type4'] = (df['Chest pain type'] == 4).astype(int)
        
        # -------------------------------------------------------------------
        # 4. ENCODAGE / CASTING
        # -------------------------------------------------------------------
        
        # Variables à passer expressément en "category" pour LightGBM / CatBoost
        categorical_cols = [
            'Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 
            'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium',
            'BP_Group', 'Age_Group'
        ]
        
        if self.use_categories:
            for c in categorical_cols:
                if c in df.columns:
                    df[c] = df[c].astype('category')
                    
        return df

if __name__ == "__main__":
    # Test unitaire rapide du pipeline de features
    import os
    data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'
    
    try:
        train_df = pd.read_csv(f'{data_dir}/train.csv')
        fe = HeartDiseaseFeatureEngineer(use_categories=True)
        transformed_df = fe.transform(train_df)
        
        print("Feature Engineering appliqué avec succès.")
        print("Nouvelles colonnes créées :")
        for col in transformed_df.columns:
            if col not in train_df.columns:
                print(f" - {col}")
                
        print(f"\nDimensions originales : {train_df.shape}")
        print(f"Nouvelles dimensions : {transformed_df.shape}")
        
    except FileNotFoundError:
        print("Fichier test introuvable pour la démo autonome.")
