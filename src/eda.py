#%% [markdown]
# # Exploratory Data Analysis (EDA)
# 
# Notre objectif est d'explorer le jeu de données pour mieux comprendre les distributions et les 
# interactions éventuelles entre nos variables, afin d'optimiser notre Feature Engineering.

#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

#%% [markdown]
# ## 1. Chargement des données

#%%
print("Chargement des données...")
import os

# Determiner le chemin de base selon si on exécute depuis src/ ou la racine
data_dir = '../data' if os.path.exists('../data/train.csv') else 'data'

train_df = pd.read_csv(f'{data_dir}/train.csv')
test_df = pd.read_csv(f'{data_dir}/test.csv')

print(f"Dimensions de Train : {train_df.shape}")
print(f"Dimensions de Test  : {test_df.shape}")

# Encodage de la target directement pour faciliter l'analyse
if 'Heart Disease' in train_df.columns:
    train_df['Heart Disease'] = train_df['Heart Disease'].map({'Presence': 1, 'Absence': 0})
    print("Target 'Heart Disease' encodée (Presence=1, Absence=0).")

display(train_df.head())

#%% [markdown]
# ## 2. Qualité des données (Valeurs manquantes, doublons, types)
# On va regarder s'il y a des données manquantes et vérifier les types globaux.

#%%
def summary(df):
    summ = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summ['Missing'] = df.isnull().sum()
    summ['Missing%'] = (df.isnull().sum() / len(df)) * 100
    summ['Unique'] = df.nunique()
    summ['Duplicates'] = df.duplicated().sum()
    return summ

print("--- Résumé du set d'entrainement ---")
display(summary(train_df))

print("\n--- Résumé du test set ---")
display(summary(test_df))

#%%
# Visualisation de la Target
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Heart Disease')
plt.title('Distribution de la Target (1 = Presence, 0 = Absence)')
plt.show()

# Proportion
prop = train_df['Heart Disease'].value_counts(normalize=True) * 100
print(f"Proportion :\n{prop}")


#%% [markdown]
# ## 3. Analyse Univariée (Variables Continues)
# Regardons à quoi ressemblent nos variables continues.
# 
# 1. Analyse Univariée (Distribution des variables)
# 
# Cette première image nous permet de comprendre comment chaque variable est répartie individuellement, indépendamment des autres.
# 
#     Âge (Age) : La distribution est asymétrique avec plusieurs pics (multimodale), mais la majorité des patients se situe entre 40 et 65 ans. Le pic principal semble se trouver autour de 55-60 ans.
# 
#     Pression Artérielle (BP) : La distribution présente des pics très marqués et anormaux (ex: autour de 120, 130, 140). Cela indique très probablement que les mesures ont été arrondies lors de la collecte des données. La majorité des valeurs se concentrent entre 110 et 150.
# 
#     Cholestérol (Cholesterol) : La distribution se rapproche d'une courbe normale (en cloche), légèrement étalée vers la droite. La plupart des patients ont un taux compris entre 200 et 300.
# 
#     Fréquence Cardiaque Maximale (Max HR) : La distribution est asymétrique vers la gauche (les valeurs se concentrent sur la droite). La majorité des patients ont atteint une fréquence cardiaque maximale entre 140 et 180, avec un pic très prononcé autour de 160.
# 
#     Dépression ST (ST depression) : Cette distribution est extrêmement asymétrique vers la droite. Une écrasante majorité des patients ont une valeur de 0. Les autres valeurs s'étalent en diminuant jusqu'à environ 6.

#%%
continuous_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

fig, axes = plt.subplots(len(continuous_features), 1, figsize=(10, 5 * len(continuous_features)))
for i, col in enumerate(continuous_features):
    sns.histplot(train_df[col], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f'Distribution de {col}')
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 4. Analyse Bivariée (Features vs Target)
# Est-ce que certaines variables discriminent fortement la présence/absence de maladie ?
# 
# 2. Analyse Bivariée (Variables vs Target 'Heart Disease')
# 
# Les boîtes à moustaches (boxplots) nous montrent comment la distribution de chaque variable continue diffère selon que le patient est atteint d'une maladie cardiaque (Target = 1) ou non (Target = 0).
# 
#     Age vs Target : Les patients atteints de maladies cardiaques (1) ont tendance à être plus âgés. La médiane pour le groupe malade est proche de 58 ans, contre environ 52 ans pour les personnes saines.
# 
#     BP vs Target : Les deux boîtes sont quasiment identiques. La pression artérielle au repos, à elle seule, ne semble pas différencier clairement les personnes saines des personnes malades dans ce jeu de données. On note la présence de nombreuses valeurs aberrantes (points au-dessus des moustaches) dans les deux groupes.
# 
#     Cholesterol vs Target : Tout comme pour la pression artérielle, les médianes et les distributions sont très similaires entre les deux groupes. Le cholestérol ne semble pas être un fort facteur discriminant ici.
# 
#     Max HR vs Target : On observe une différence très nette. Les patients atteints de maladies cardiaques (1) ont une fréquence cardiaque maximale nettement plus basse (médiane vers 145) que les patients sains (médiane vers 162).
# 
#     ST depression vs Target : La différence est également très marquée. Les patients malades (1) ont une dépression ST beaucoup plus élevée (médiane > 1) par rapport aux personnes saines dont la médiane est à 0.

#%%
# Continues vs Target
fig, axes = plt.subplots(1, len(continuous_features), figsize=(20, 5))
for i, col in enumerate(continuous_features):
    sns.boxplot(data=train_df, x='Heart Disease', y=col, ax=axes[i])
    axes[i].set_title(f'{col} vs Target')
plt.tight_layout()
plt.show()

#%%
categorical_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

# Catégorielles vs Target
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
axes = axes.flatten()
for i, col in enumerate(categorical_features):
    sns.countplot(data=train_df, x=col, hue='Heart Disease', ax=axes[i])
    axes[i].set_title(f'{col} vs Target')
plt.tight_layout()
plt.show()

#%% [markdown]
# ## 5. Corrélations
# Y a-t-il des redondances (multicollinéarité) ou des relations fortes avec la target ?
# 
# 3. Matrice de Corrélation (Heatmap)
# 
# Cette matrice vient confirmer et quantifier les observations faites lors de l'analyse bivariée grâce au coefficient de corrélation (allant de -1 à 1).
# 
# Corrélations avec la variable cible (Heart Disease) :
# 
#     Fortes corrélations :
# 
#         Max HR (-0.44) : C'est la corrélation négative la plus forte. Plus la fréquence cardiaque maximale est faible, plus le risque de maladie cardiaque est élevé.
# 
#         ST depression (0.43) : C'est la corrélation positive la plus forte. Plus la dépression ST est élevée, plus le risque est grand.
# 
#     Corrélation modérée :
# 
#         Age (0.21) : Corrélation positive modérée, confirmant que le risque augmente avec l'âge.
# 
#     Corrélations très faibles ou nulles :
# 
#         Cholesterol (0.08) et BP (-0.01) : Ces variables ont très peu de lien linéaire avec la présence d'une maladie cardiaque dans ce dataset.
# 
# Corrélations entre les variables explicatives (multicolinéarité) :
# 
#     Il n'y a pas de corrélation forte entre les variables explicatives elles-mêmes (la plus forte étant de -0.23 entre Max HR et ST depression). C'est une excellente nouvelle pour la création d'un modèle prédictif, car cela signifie que les variables apportent des informations différentes sans se chevaucher de manière excessive.

#%%
corr = train_df[continuous_features + ['Heart Disease']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matrice de Corrélation (Variables continues + Target)')
plt.show()

#%% [markdown]
# ## 6. Analyse des Variables Catégorielles vs Target (Proportions)
# Le graphique précédent donnait une idée des volumes, mais regardons le **pourcentage exact de malades** 
# pour chaque modalité de nos variables catégorielles. Cela va nous dire lesquelles sont les plus discriminantes.
# 
# 1. Les indicateurs "Alertes Rouges" (Très discriminants)
# 
# Ces variables montrent une différence massive entre les modalités. Si un patient présente ces caractéristiques, la probabilité qu'il soit atteint d'une maladie cardiaque est très élevée.
# 
#     Angine de poitrine d'effort (Exercise angina) : C'est l'un des indicateurs les plus forts. Si un patient a une angine à l'effort (1), il a 80,63 % de chances d'être malade, contre seulement 31,34 % s'il n'en a pas.
# 
#     Nombre de vaisseaux colorés par fluoroscopie (Number of vessels fluro) : La corrélation est quasi linéaire et spectaculaire.
# 
#         0 vaisseau : 30,31 % de malades.
# 
#         1 vaisseau : 72,93 % de malades.
# 
#         2 ou 3 vaisseaux : ~90 % de malades. C'est un marqueur physique extrêmement fiable.
# 
#     Thallium : Un résultat de type 7 (défaut réversible) indique une probabilité de 81,54 % de maladie, contre seulement 19,80 % pour un résultat normal (type 3).
# 
#     Type de douleur thoracique (Chest pain type) : Le type 4 est particulièrement critique avec 69,75 % de cas positifs, alors que les types 1, 2 et 3 tournent entre 10 % et 19 %.
# 
# 2. Les facteurs démographiques et physiologiques
# 
#     Sexe (Sex) : Il y a une disparité énorme. Dans ce jeu de données, les hommes (si on suppose que 1 = Homme) sont beaucoup plus touchés (55,59 %) que les femmes (17,88 %). C'est une variable pivot pour votre futur modèle.
# 
#     Pente du segment ST (Slope of ST) : Les pentes de type 2 et 3 sont des indicateurs de risque majeur (environ 70 % de malades), tandis que la pente de type 1 est plutôt rassurante (26,23 %).
# 
#     Résultats EKG : Le type 2 (souvent lié à une hypertrophie ventriculaire gauche) montre un risque plus élevé (55,96 %) que les types 0 ou 1 (~35 %).
# 
# 3. Le facteur "Neutre" (Peu discriminant)
# 
#     Glycémie à jeun (FBS over 120) : C'est la variable la moins utile ici. Qu'on soit au-dessus ou en dessous de 120 mg/dl, la probabilité de maladie cardiaque reste assez proche (50,5 % contre 44,3 %). Cela confirme que le diabète seul (mesuré ainsi) n'est pas le prédicteur le plus fort dans ce contexte précis.

#%%
categorical_features = ['Sex', 'Chest pain type', 'FBS over 120', 'EKG results', 'Exercise angina', 'Slope of ST', 'Number of vessels fluro', 'Thallium']

for col in categorical_features:
    # Tableau croisé : Modalité de la feature vs Taux de la classe Target=1
    ct = pd.crosstab(train_df[col], train_df['Heart Disease'], margins=True, margins_name="Total")
    
    # On calcule le pourcentage de malades (Target 1) pour chaque sous-groupe :
    # = (Nombre de malades dans cette catégorie) / (Total de cette catégorie) * 100
    ct['% de Malades (Target=1)'] = (ct[1] / ct['Total']) * 100
    
    # On affiche
    print(f"\n--- {col.upper()} ---")
    display(ct.round(2))
    
#%% [markdown]
# ## 7. Analyse Multivariée (Interactions clés)
# Puisque on a identifié nos "Golden Features" (Age, Max HR, ST depression), regardons comment elles interagissent entre elles 
# par rapport à la Target. Les modèles par arbres raffolent de ce genre d'espace séparable.

#%%
plt.figure(figsize=(12, 5))

# Interaction 1 : Age vs Max HR
plt.subplot(1, 2, 1)
sns.scatterplot(data=train_df, x='Age', y='Max HR', hue='Heart Disease', alpha=0.5, edgecolor=None)
plt.title('Age vs Max HR (Coloré par Target)')
# Tracer la ligne théorique HR_Max = 220 - Age
x_age = np.linspace(train_df['Age'].min(), train_df['Age'].max(), 100)
plt.plot(x_age, 220 - x_age, color='red', linestyle='--', label='220 - Age')
plt.legend()

# Interaction 2 : ST depression vs Max HR
plt.subplot(1, 2, 2)
sns.scatterplot(data=train_df, x='Max HR', y='ST depression', hue='Heart Disease', alpha=0.5, edgecolor=None)
plt.title('Max HR vs ST depression (Coloré par Target)')

plt.tight_layout()
plt.show()

#%% [markdown]
# ## 8. Comparaison Train vs Test (Data Drift)
# Une étape cruciale sur Kaggle : s'assurer que les données sur lesquelles on va prédire (Test) 
# ressemblent bien aux données d'entraînement. S'il y a un décalage ("Shift" ou "Drift"), nos modèles vont échouer !

#%%
# On va superposer les distributions des variables continues pour Train vs Test
fig, axes = plt.subplots(len(continuous_features), 1, figsize=(10, 3 * len(continuous_features)))
for i, col in enumerate(continuous_features):
    sns.kdeplot(train_df[col], fill=True, label='Train', ax=axes[i], alpha=0.4, color='blue')
    sns.kdeplot(test_df[col], fill=True, label='Test', ax=axes[i], alpha=0.4, color='orange')
    axes[i].set_title(f'Train vs Test Drift : {col}')
    axes[i].legend()
plt.tight_layout()
plt.show()

# Pour les catégories, on va simplement comparer les proportions (Mean)
print("--- Proportions Catégorielles moyennes (Train vs Test) ---")
diff_df = pd.DataFrame({
    'Train_Mean': train_df.drop(columns=['id', 'Heart Disease', *continuous_features]).mean(),
    'Test_Mean' : test_df.drop(columns=['id', *continuous_features]).mean()
})
diff_df['Diff_Absolue'] = abs(diff_df['Train_Mean'] - diff_df['Test_Mean'])
display(diff_df.sort_values(by='Diff_Absolue', ascending=False))

#%% [markdown]
# ## Conclusion de l'Analyse Multivariée et Data Drift
# 
# 1. Analyse des interactions (Image : Scatter plots)
# 
# Ce graphique croise vos variables les plus importantes pour voir comment elles interagissent visuellement.
# 
#     Age vs Max HR (Le graphique de gauche) : * La ligne pointillée rouge 220 - Age est brillante ! C'est la formule physiologique classique pour estimer la fréquence cardiaque maximale théorique d'un être humain.
# 
#         Observation : On voit très clairement que la grande majorité des points orange (Malades, Target 1) se situent nettement en dessous de cette ligne théorique et en dessous du nuage de points bleus (Sains).
# 
#         Ce qu'on en retient pour le modèle : Les personnes atteintes de maladies cardiaques n'arrivent pas à atteindre leur fréquence cardiaque maximale théorique. Idée de Feature Engineering : Vous pourriez créer une nouvelle variable Deficit_HR = (220 - Age) - Max_HR. Cette variable capturera parfaitement cet écart et sera probablement un prédicteur redoutable !
# 
#     Max HR vs ST depression (Le graphique de droite) :
# 
#         Observation : En croisant nos deux variables les plus corrélées à la cible, nous obtenons une excellente séparation de l'espace.
# 
#         Le coin en bas à droite (Haute Fréquence Cardiaque, Basse Dépression ST) est un "territoire sain" (bleu massif).
#       
#         Le coin en haut à gauche (Basse Fréquence Cardiaque, Haute Dépression ST) est un "territoire malade" (orange massif).
# 
#         Ce qu'on en retient : Nos données sont très bien séparables. Un modèle capable de faire des combinaisons non linéaires (comme un arbre de décision, Random Forest ou XGBoost) va adorer cet espace de données.
# 
# 2. Évaluation du "Data Drift" (Image des distributions + Tableau)
# 
# Cette étape est cruciale et souvent oubliée. Elle permet de s'assurer que l'échantillon sur lequel nous allons entraîner notre modèle (Train) est similaire à celui sur lequel il sera évalué (Test).
# 
#     Variables Continues (Les courbes de densité) : * Les distributions bleues (Train) et orange (Test) se superposent de manière quasi parfaite. On retrouve exactement les mêmes pics atypiques sur la pression artérielle (BP) et les mêmes asymétries.
# 
#     Variables Catégorielles (Le tableau) : * Les différences absolues des moyennes (Diff_Absolue) sont microscopiques (la plus grande différence est de 0.003 pour Number of vessels). La proportion de chaque catégorie est identique entre les deux sets de données.
# 
# Ce qu'on en retient (La conclusion est excellente) : Il n'y a strictement aucun "Data Drift" (dérive des données). Notre jeu de test est un miroir parfait de notre jeu d'entraînement.
# 
#     Avantage : Nous n'aurons pas de mauvaises surprises. Si le modèle est performant sur le Train set (et validé par validation croisée), il aura des performances très similaires sur le Test set. Nous n'aurons pas besoin d'appliquer des techniques complexes de re-pondération des données.

# %%
