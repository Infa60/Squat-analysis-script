import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. CONFIGURATION ET CHARGEMENT DES DONNÉES ---
# ==============================================================================
file_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Master_Database_Patient_Clustered_GMM_2_average.pkl"

# CHOIX DE LA MÉTHODE PROM
METHODE_PROM = 'Soucie'  # Options : 'Soucie', 'zscore', 'Papageorgiou'

print("1. Chargement de la base de données...")
df = pd.read_pickle(file_path)

# FILTRAGE ET RENOMMAGE DES COLONNES PROM
prom_cols = [c for c in df.columns if 'pROM_' in c]
cols_to_drop = [c for c in prom_cols if METHODE_PROM not in c]
df = df.drop(columns=cols_to_drop)

rename_dict = {c: c.replace(f"_{METHODE_PROM}", "") for c in df.columns if 'pROM_' in c and METHODE_PROM in c}
df = df.rename(columns=rename_dict)

print(f"   -> Méthode pROM sélectionnée : {METHODE_PROM} ({len(cols_to_drop)} colonnes redondantes supprimées)")

TARGET_COL = 'Profile_Max_Label'
df = df.dropna(subset=[TARGET_COL])

# ==============================================================================
# --- 2. GESTION DES VISITES (Éviter la fuite de données) ---
# ==============================================================================
print("2. Filtrage : Conservation de la première visite par patient (Indépendance stricte)")
if 'ID_Visite' in df.columns:
    df = df.sort_values(by=['ID_Patient', 'ID_Visite'])
df_unique = df.drop_duplicates(subset=['ID_Patient'], keep='first').copy()

print(f"   -> Nombre de patients uniques pour l'entraînement : {len(df_unique)}")

# ==============================================================================
# --- 3. IDENTIFICATION DES DEUX GROUPES DE VARIABLES ---
# ==============================================================================
all_clinical_cols = [c for c in df_unique.columns if
                     c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_', 'pROM_')) and not c.endswith(
                         'G')]

cols_scores = [c for c in all_clinical_cols if 'Score' in c]
cols_autres = [c for c in all_clinical_cols if 'Score' not in c]

print(f"   -> Variables 'Scores' trouvées : {len(cols_scores)}")
print(f"   -> Autres variables cliniques trouvées : {len(cols_autres)}")


# ==============================================================================
# --- 4. FONCTION POUR GÉNÉRER UN ARBRE SUR MESURE (VALEURS CONTINUES) ---
# ==============================================================================
def build_and_evaluate_tree(features_list, title):
    print(f"\n{'=' * 70}")
    print(f"             ANALYSE : {title.upper()}             ")
    print(f"{'=' * 70}")

    if not features_list:
        print("⚠️ Aucune variable trouvée pour cette analyse. Vérifie le nom de tes colonnes.")
        return

    # --- A. Préparation des données réelles ---
    X = df_unique[features_list].copy()
    y = df_unique[TARGET_COL]

    # Traitement des valeurs manquantes avec la médiane (sans transformer en catégories)
    colonnes_a_garder = []
    for col in features_list:
        if X[col].isna().mean() > 0.3:
            # On ignore les colonnes avec plus de 30% de valeurs manquantes
            continue

        val_median = X[col].median()
        X[col] = X[col].fillna(val_median)
        colonnes_a_garder.append(col)

    # On filtre X pour ne garder que les colonnes valides
    X = X[colonnes_a_garder]

    if X.empty:
        print("⚠️ Toutes les variables ont été ignorées (trop de valeurs manquantes).")
        return

    # --- B. Entraînement ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # On augmente légèrement max_depth à 5 pour laisser l'arbre exploiter les valeurs continues
    tree_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, criterion='entropy', random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict(X_test)
    print(f"✅ Précision du modèle sur de nouveaux patients (Test Set) : {accuracy_score(y_test, y_pred) * 100:.1f}%\n")

    # --- C. Extraction des règles ---
    tree_rules = export_text(tree_model, feature_names=list(X.columns))
    print("RÈGLES DÉCOUVERTES (Seuils numériques réels) :")
    print("-" * 30)
    print(tree_rules)

    # --- D. Visualisation ---
    # Conversion explicite des classes en format string pour plot_tree
    class_names_str = [str(c) for c in tree_model.classes_]

    plt.figure(figsize=(25, 12))  # Taille légèrement augmentée pour un arbre potentiellement plus grand
    plot_tree(tree_model,
              feature_names=X.columns,
              class_names=class_names_str,
              filled=True,
              rounded=True,
              fontsize=9,
              proportion=False)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==============================================================================
# --- 5. LANCEMENT DES DEUX ANALYSES ---
# ==============================================================================

# 1er Arbre : Uniquement les SCORES
build_and_evaluate_tree(cols_scores, f"Arbre de Décision - Uniquement les Scores (pROM: {METHODE_PROM})")

# 2ème Arbre : TOUT LE RESTE (Force, ROM, Spasticité, etc.)
build_and_evaluate_tree(cols_autres, "Arbre de Décision - Clinique (Sans les Scores)")