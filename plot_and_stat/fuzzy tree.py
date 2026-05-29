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
file_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Master_Database_Patient_Clustered_KMEANS_3_all_visit.pkl"

# 🟢 NOUVEAU : CHOIX DE LA MÉTHODE PROM ICI
# Change simplement ce nom pour basculer toute ton analyse sur une autre méthode
METHODE_PROM = 'Soucie'  # Options : 'Soucie', 'zscore', 'Papageorgiou'

print("1. Chargement de la base de données...")
df = pd.read_pickle(file_path)

# 🟢 NOUVEAU : FILTRAGE ET RENOMMAGE DES COLONNES PROM
# On trouve toutes les colonnes qui concernent les pROM
prom_cols = [c for c in df.columns if 'pROM_' in c]

# On identifie les colonnes qui NE SONT PAS la méthode choisie et on les jette
cols_to_drop = [c for c in prom_cols if METHODE_PROM not in c]
df = df.drop(columns=cols_to_drop)

# On renomme les colonnes gardées pour enlever le suffixe (ex: pROM_Knee_Score_Soucie -> pROM_Knee_Score)
rename_dict = {c: c.replace(f"_{METHODE_PROM}", "") for c in df.columns if 'pROM_' in c and METHODE_PROM in c}
df = df.rename(columns=rename_dict)

print(f"   -> Méthode pROM sélectionnée : {METHODE_PROM} ({len(cols_to_drop)} colonnes redondantes supprimées)")

# On choisit la cible (le résultat du clustering que tu veux expliquer)
TARGET_COL = 'Profile_Max_Label'

# On nettoie pour ne garder que les patients qui ont bien été classés
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
# 🟢 MODIFIÉ : Ajout du préfixe 'pROM_' dans le startswith pour ne pas les rater
all_clinical_cols = [c for c in df_unique.columns if
                     c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_', 'pROM_')) and not c.endswith('G')]

# 🟢 MODIFIÉ : On cherche "Score" n'importe où dans le nom (pour capter "pROM_Knee_Score" par exemple)
cols_scores = [c for c in all_clinical_cols if 'Score' in c]
cols_autres = [c for c in all_clinical_cols if 'Score' not in c]

print(f"   -> Variables 'Scores' trouvées : {len(cols_scores)}")
print(f"   -> Autres variables cliniques trouvées : {len(cols_autres)}")

# ==============================================================================
# --- 4. FONCTION POUR GÉNÉRER UN ARBRE SUR MESURE ---
# ==============================================================================
def build_and_evaluate_tree(features_list, title):
    print(f"\n{'=' * 70}")
    print(f"             ANALYSE : {title.upper()}             ")
    print(f"{'=' * 70}")

    if not features_list:
        print("⚠️ Aucune variable trouvée pour cette analyse. Vérifie le nom de tes colonnes.")
        return

    # --- A. Discrétisation ---
    X_cat = pd.DataFrame(index=df_unique.index)

    for col in features_list:
        if df_unique[col].isna().mean() > 0.3:
            continue

        val_median = df_unique[col].median()
        df_unique[col] = df_unique[col].fillna(val_median)

        seuil_bas = np.percentile(df_unique[col], 33)
        seuil_haut = np.percentile(df_unique[col], 66)

        if seuil_bas == seuil_haut:
            X_cat[col] = df_unique[col].apply(lambda x: 'Faible' if x <= seuil_bas else 'Elevé')
        else:
            conditions = [
                (df_unique[col] <= seuil_bas),
                (df_unique[col] > seuil_bas) & (df_unique[col] <= seuil_haut),
                (df_unique[col] > seuil_haut)
            ]
            choix = [f'{col}_Faible', f'{col}_Moyen', f'{col}_Elevé']
            X_cat[col] = np.select(conditions, choix, default='Inconnu')

    # One-Hot Encoding
    X = pd.get_dummies(X_cat)
    y = df_unique[TARGET_COL]

    if X.empty:
        print("⚠️ Toutes les variables ont été ignorées (trop de valeurs manquantes).")
        return

    # --- B. Entraînement ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tree_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, criterion='entropy', random_state=42)
    tree_model.fit(X_train, y_train)

    y_pred = tree_model.predict(X_test)
    print(f"✅ Précision du modèle sur de nouveaux patients (Test Set) : {accuracy_score(y_test, y_pred) * 100:.1f}%\n")

    # --- C. Extraction des règles ---
    tree_rules = export_text(tree_model, feature_names=list(X.columns))
    tree_rules = tree_rules.replace("<= 0.50", "== NON").replace(">  0.50", "== OUI")
    print("RÈGLES DÉCOUVERTES :")
    print("-" * 30)
    print(tree_rules)

    # --- D. Visualisation ---
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model,
              feature_names=X.columns,
              class_names=tree_model.classes_,
              filled=True,
              rounded=True,
              fontsize=10,
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