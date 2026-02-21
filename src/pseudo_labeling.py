import pandas as pd
import numpy as np
import os
import argparse

def create_pseudo_labels(train_path, test_path, sub_path, output_path, lower_threshold=0.01, upper_threshold=0.99):
    """
    Crée un nouveau dataset d'entraînement en ajoutant les prédictions très confiantes du test set.
    """
    print("--- Démarrage du Pseudo-Labeling ---")
    
    # Vérification des fichiers
    for path in [train_path, test_path, sub_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fichier introuvable : {path}")
            
    # Chargement
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sub_df = pd.read_csv(sub_path)
    
    # Sécuriser la fusion par l'id
    test_merged = test_df.merge(sub_df, on='id', how='left')
    
    # Filtrage des lignes très confiantes
    confident_mask = (test_merged['Heart Disease'] <= lower_threshold) | (test_merged['Heart Disease'] >= upper_threshold)
    confident_test = test_merged[confident_mask].copy()
    
    # Binariser la cible (0 ou 1) dous format texte si besoin selon train_df
    confident_test['Heart Disease'] = confident_test['Heart Disease'].apply(lambda x: 'Presence' if x >= upper_threshold else 'Absence')
    
    print(f"Lignes d'entraînement de base : {len(train_df)}")
    print(f"Total des lignes de test : {len(test_df)}")
    print(f"Lignes pseudo-labelisées retenues : {len(confident_test)} (soit {len(confident_test)/len(test_df):.2%})")
    print(f"  - Cas 'Presence' potentiels : {len(confident_test[confident_test['Heart Disease'] == 'Presence'])}")
    print(f"  - Cas 'Absence' potentiels : {len(confident_test[confident_test['Heart Disease'] == 'Absence'])}")
    
    # Concaténation
    augmented_train = pd.concat([train_df, confident_test], ignore_index=True)
    
    # Mélange (shuffle) et sauvegarde
    augmented_train = augmented_train.sample(frac=1, random_state=42).reset_index(drop=True)
    augmented_train.to_csv(output_path, index=False)
    
    print(f"\n[SUCCÈS] Nouveau dataset généré avec {len(augmented_train)} lignes.")
    print(f"Sauvegardé sous : {output_path}")
    print("\nPour l'utiliser, mettez à jour les chemins dans train.py ou renommez le fichier en train.csv !")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default="../submission.csv", help="Chemin vers le fichier de soumission source")
    parser.add_argument("--lower", type=float, default=0.01, help="Seuil inférieur (ex: 0.01)")
    parser.add_argument("--upper", type=float, default=0.99, help="Seuil supérieur (ex: 0.99)")
    args = parser.parse_args()
    
    data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'
    train_p = f"{data_dir}/train.csv"
    test_p = f"{data_dir}/test.csv"
    out_p = f"{data_dir}/train_pseudo.csv"
    
    create_pseudo_labels(train_p, test_p, args.sub, out_p, args.lower, args.upper)
