import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. CONFIGURATION ET CHARGEMENT DES DONNÉES ---
# ==============================================================================
# Remplace par le nom exact du fichier généré à la fin de ton script précédent
file_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Master_Database_Patient_Clustered_KMEANS_3_all_visit.pkl"

print("1. Chargement de la base de données...")
df = pd.read_pickle(file_path)

# On choisit la cible (le résultat du clustering que tu veux expliquer)
TARGET_COL = 'Profile_Max_Label'  # ou 'Profile_Max_Label' selon ton choix

# On nettoie pour ne garder que les patients qui ont bien été classés
df = df.dropna(subset=[TARGET_COL])

# ==============================================================================
# --- 2. GESTION DES VISITES (Éviter la fuite de données) ---
# ==============================================================================
print("2. Filtrage : Conservation de la première visite par patient (Indépendance stricte)")
# On trie par patient et par visite (pour avoir la plus ancienne en premier)
if 'ID_Visite' in df.columns:
    df = df.sort_values(by=['ID_Patient', 'ID_Visite'])
# On ne garde que la première ligne de chaque patient pour construire l'arbre
df_unique = df.drop_duplicates(subset=['ID_Patient'], keep='first').copy()

print(f"   -> Nombre de patients uniques pour l'entraînement : {len(df_unique)}")

# ==============================================================================
# --- 3. PRÉPARATION DES VARIABLES CLINIQUES (Fuzzification / Discrétisation) ---
# ==============================================================================
print("3. Discrétisation des variables cliniques (Faible / Moyen / Élevé)...")

# Sélection automatique de toutes tes variables cliniques
clinical_cols = [c for c in df_unique.columns if
                 c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_')) and not c.endswith('G')]

# Création d'un DataFrame vide pour stocker les variables discrétisées
X_cat = pd.DataFrame(index=df_unique.index)

# Méthode inspirée d'Armand et al. : on utilise les tertiles (33% et 66%) pour créer les zones
for col in clinical_cols:
    # On ignore les variables qui ont trop de valeurs manquantes (ex: > 30%)
    if df_unique[col].isna().mean() > 0.3:
        continue

    # Remplacer les petits NaN restants par la médiane
    val_median = df_unique[col].median()
    df_unique[col] = df_unique[col].fillna(val_median)

    # Calcul des seuils (Tertiles)
    seuil_bas = np.percentile(df_unique[col], 33)
    seuil_haut = np.percentile(df_unique[col], 66)

    # Si la variable a très peu de valeurs différentes (ex: un score de 0 à 4), les seuils peuvent se superposer.
    # On ajoute une petite sécurité
    if seuil_bas == seuil_haut:
        X_cat[col] = df_unique[col].apply(lambda x: 'Faible' if x <= seuil_bas else 'Elevé')
    else:
        # Discrétisation en 3 modalités
        conditions = [
            (df_unique[col] <= seuil_bas),
            (df_unique[col] > seuil_bas) & (df_unique[col] <= seuil_haut),
            (df_unique[col] > seuil_haut)
        ]
        choix = [f'{col}_Faible', f'{col}_Moyen', f'{col}_Elevé']
        X_cat[col] = np.select(conditions, choix, default='Inconnu')

# Pour Scikit-Learn, il faut transformer ces textes en variables binaires (One-Hot Encoding)
# Ex: La colonne "ROM_Cheville" devient "ROM_Cheville_Faible (0/1)", "ROM_Cheville_Moyen (0/1)"...
X = pd.get_dummies(X_cat)
y = df_unique[TARGET_COL]

# ==============================================================================
# --- 4. ENTRAÎNEMENT DE L'ARBRE DE DÉCISION ---
# ==============================================================================
print("4. Entraînement de l'Arbre de Décision...")

# Séparation des données : 80% pour créer l'arbre, 20% pour vérifier s'il est bon
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# On limite la profondeur à 4. Au-delà, les règles deviennent illisibles pour un humain !
tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, criterion='entropy', random_state=42)
tree_model.fit(X_train, y_train)

# Évaluation simple
y_pred = tree_model.predict(X_test)
print(f"\n✅ Précision du modèle sur de nouveaux patients (Test Set) : {accuracy_score(y_test, y_pred) * 100:.1f}%\n")

# ==============================================================================
# --- 5. EXTRACTION ET AFFICHAGE DES RÈGLES CLINIQUES ---
# ==============================================================================
print("======================================================")
print("             RÈGLES CLINIQUES DÉCOUVERTES             ")
print("======================================================\n")

# Scikit-Learn permet d'exporter l'arbre sous forme de texte (les fameux IF... THEN...)
tree_rules = export_text(tree_model, feature_names=list(X.columns))

# Petit nettoyage pour rendre le texte plus beau
tree_rules = tree_rules.replace("<= 0.50", "== NON").replace(">  0.50", "== OUI")
print(tree_rules)

# ==============================================================================
# --- 6. VISUALISATION GRAPHIQUE DE L'ARBRE ---
# ==============================================================================
plt.figure(figsize=(20, 10))
plot_tree(tree_model,
          feature_names=X.columns,
          class_names=tree_model.classes_,
          filled=True,
          rounded=True,
          fontsize=10,
          proportion=False)
plt.title("Arbre de Décision des Profils de Squat", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
# plt.savefig("Arbre_Decision_Clinique.png", dpi=300) # Décommente pour sauvegarder l'image