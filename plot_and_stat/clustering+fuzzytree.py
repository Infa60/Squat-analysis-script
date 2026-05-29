import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations
import re
import warnings
from fonction import *

warnings.filterwarnings('ignore')

# --- PANDAS CONFIGURATION ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
AVERAGE_TRIALS_PER_VISIT = False  # Mode d'agrégation clinique
type_of_analysis = "average" if AVERAGE_TRIALS_PER_VISIT else "all_visit"

ALGO_CLUSTERING = 'GMM'
DEFAULT_FPS = 50
CUTOFF_FREQ = 3

# --- DÉFINITION DES VARIABLES DE CLUSTERING ---
FEATURES_BASE = ['Knee', 'Tibia', 'Trunk', 'Knee_Frontal']

FEATURE_MAPPING_MAX = {
    'Knee': 'Knee_Flexion_Max',
    'Trunk': 'Trunk_Lean_Max',
    'Tibia': 'Tibia_Lean_Max',
    'Knee_Frontal': 'Knee_Frontal_Max'
}

FEATURES_MAX = [FEATURE_MAPPING_MAX[f] for f in FEATURES_BASE]

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"
frontal_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_Frontal_all.pkl"
master_db_healthy_file = fr"{main_path}\Data\Master_Database_Healthy_all.pkl"

features_str = "_".join(FEATURES_BASE)
output_plot_folder = fr"{main_path}\Results_v3\Plot_{ALGO_CLUSTERING}_right_{type_of_analysis}"
os.makedirs(output_plot_folder, exist_ok=True)


# ==============================================================================
# --- 2. FONCTIONS UTILITAIRES ---
# ==============================================================================
def simplifier_diagnostic(diag):
    if pd.isna(diag): return 'Inconnu'
    d = str(diag).strip().lower().replace(' ', '').replace('é', 'e')
    return 'Hemi' if 'hemi' in d else ('Di' if 'di' in d else 'Autre')


def clean_side(cote):
    c = str(cote).lower()
    has_right = 'droit' in c or 'right' in c
    has_left = 'gauch' in c or 'left' in c
    if has_right and has_left:
        return 'Right > Left' if c.find('droit') < c.find('gauch') or c.find('right') < c.find(
            'left') else 'Left > Right'
    elif has_right:
        return 'Right'
    elif has_left:
        return 'Left'
    else:
        return 'Unknown'


def process_kinematics_max(curve, fps):
    c_array = np.array(curve)
    valid = ~np.isnan(c_array)
    if not valid.any(): return np.nan
    c_interp = np.interp(np.arange(len(c_array)), np.where(valid)[0], c_array[valid])
    c_filt = butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)
    return np.max(c_filt)


def calculate_valgus_varus_frontal(hip, knee, ankle, side):
    v1 = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    angle_deg = np.degrees(np.arctan2(cross_product, dot_product))
    if side == 'gauche': angle_deg = -angle_deg
    return angle_deg


# ==============================================================================
# --- 3. CHARGEMENT ET PRÉPARATION UNIFIÉE ---
# ==============================================================================
print(f"1. Chargement et extraction unifiée depuis les matrices 3D (Sagittal + Frontal)...")

df_master_pat = pd.read_pickle(master_db_patient_file)

if os.path.exists(frontal_db_patient_file):
    df_fro = pd.read_pickle(frontal_db_patient_file)[['File_Sagittal', 'Video_FPS', 'Raw_Keypoints']]
    df_fro.columns = ['File_Sagittal', 'FPS_frontal', 'Keypoints_frontal']
    df_master_pat = pd.merge(df_master_pat, df_fro, on='File_Sagittal', how='left')

mask_pat = (df_master_pat['Pose estimation'].astype(str).str.strip().str.upper() == 'Y') & \
           (df_master_pat['Caregiver assistance'].astype(str).str.strip() != '2') & \
           (df_master_pat['Hand-to-ground contact'].astype(str).str.strip() != '2') & \
           (df_master_pat['Heel_Rise_Binaire'] == 0) & \
           (df_master_pat['CoteDiagnostic'] != 'Gauche') & (df_master_pat['CoteDiagnostic'] != 'Gauche Droit')

df_base_pat = df_master_pat[mask_pat].copy()
df_base_pat['Diagnostic'] = df_base_pat['Diagnostic'].apply(simplifier_diagnostic)
df_base_pat['Clean_Side'] = df_base_pat.get('CoteDiagnostic', '').apply(clean_side)
df_base_pat['Diag_Lateralite'] = df_base_pat['Diagnostic'] + " (" + df_base_pat['Clean_Side'] + ")"

processed_pat = []
for idx, row in df_base_pat.iterrows():
    try:
        kpts = row['Raw_Keypoints']
        fps = row['Video_FPS'] if pd.notna(row['Video_FPS']) and row['Video_FPS'] > 0 else DEFAULT_FPS
        n_frames = kpts.shape[0]

        k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]
        tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
        tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

        k_max = process_kinematics_max(k_f, fps)
        tr_max = process_kinematics_max(tr_f, fps)
        tib_max = process_kinematics_max(tib_f, fps)

        kf_max = np.nan

        if 'Keypoints_frontal' in row and isinstance(row['Keypoints_frontal'], np.ndarray):
            kpts_fro = row['Keypoints_frontal']
            fps_fro = row['FPS_frontal'] if pd.notna(row['FPS_frontal']) and row['FPS_frontal'] > 0 else DEFAULT_FPS
            n_frames_fro = kpts_fro.shape[0]

            c_interp = np.interp(np.arange(len(k_f)), np.where(~np.isnan(k_f))[0], np.array(k_f)[~np.isnan(k_f)])
            c_filt = butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)
            idx_sagittal = np.argmax(c_filt)

            time_seconds = idx_sagittal / fps
            idx_frontal = int(round(time_seconds * fps_fro))
            idx_frontal = min(max(idx_frontal, 0), n_frames_fro - 1)

            kf_max = calculate_valgus_varus_frontal(kpts_fro[idx_frontal, 12], kpts_fro[idx_frontal, 14],
                                                    kpts_fro[idx_frontal, 16], 'droite')

        entry = {'ID_Patient': row['ID_Patient'], 'ID_Visite': row['ID_Visite'], 'File_Sagittal': row['File_Sagittal'],
                 'Diagnostic': row['Diagnostic'], 'Diag_Lateralite': row.get('Diag_Lateralite', 'Autre / Inconnu'),
                 'Scores_GMFCS': row.get('Scores_GMFCS', np.nan),
                 'Knee_Flexion_Max': k_max, 'Trunk_Lean_Max': tr_max, 'Tibia_Lean_Max': tib_max,
                 'Knee_Frontal_Max': kf_max}

        for col in df_base_pat.columns:
            if col.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_', 'pROM_')):
                entry[col] = pd.to_numeric(row[col], errors='coerce')
        processed_pat.append(entry)
    except:
        pass

df_pat_processed = pd.DataFrame(processed_pat)

# ==============================================================================
# --- 4. AGGRÉGATION & ALIGNEMENT ---
# ==============================================================================
if AVERAGE_TRIALS_PER_VISIT:
    clin_cols = [c for c in df_pat_processed.columns if
                 c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_', 'pROM_'))]
    agg_dict = {f: 'median' for f in FEATURES_MAX}
    agg_dict.update({c: 'mean' for c in clin_cols})
    agg_dict.update({'Scores_GMFCS': 'first', 'Diag_Lateralite': 'first'})
    df_pat_final = df_pat_processed.groupby(['ID_Patient', 'ID_Visite', 'Diagnostic']).agg(agg_dict).reset_index()
else:
    df_pat_final = df_pat_processed.copy()

# ==============================================================================
# --- 5. PIPELINE AUTOMATISÉ : CLUSTERING + FUZZY TREE ---
# ==============================================================================
print("\n--- DÉMARRAGE DU PIPELINE D'OPTIMISATION AUTOMATIQUE ---")

# 1. Définition de l'espace de recherche
toutes_variables_cinematiques = [FEATURE_MAPPING_MAX[f] for f in FEATURES_BASE]
k_range = range(2, 6)

combinaisons_cinematiques = []
for r in range(1, len(toutes_variables_cinematiques) + 1):
    combinaisons_cinematiques.extend(list(combinations(toutes_variables_cinematiques, r)))

# 2. Préparation des variables cliniques (Cibles de l'arbre)
all_clinical_cols = [c for c in df_pat_final.columns if
                     c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_', 'pROM_')) and not c.endswith(
                         'G')]
cols_scores = [c for c in all_clinical_cols if 'Score' in c]
cols_autres = [c for c in all_clinical_cols if 'Score' not in c]

resultats_pipeline = []
meilleur_score_global = 0
meilleure_configuration = {}


def evaluate_tree_silently(df_tree, variables_cliniques, target_col='Cluster_GMM'):
    X = df_tree[variables_cliniques].copy()
    y = df_tree[target_col]

    colonnes_a_garder = []
    for col in variables_cliniques:
        if X[col].isna().mean() <= 0.3:
            X[col] = X[col].fillna(X[col].median())
            colonnes_a_garder.append(col)

    X = X[colonnes_a_garder]
    if X.empty: return 0.0, 0.0, 0.0, None

    comptage = y.value_counts()
    min_class_count = comptage.min()

    for cluster_id, count in comptage.items():
        if count < 2:
            infos_isoles = df_tree[df_tree[target_col] == cluster_id][['ID_Patient', 'ID_Visite']].to_dict(
                orient='records')
            print(f"      🚨 CRITIQUE : Cluster {cluster_id} a 1 seul membre ! -> {infos_isoles}")
        elif count < 5:
            print(f"      ⚠️ WARNING : Cluster {cluster_id} a moins de 5 membres (n={count}).")

    tree_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, criterion='entropy', random_state=42)
    n_splits = 5

    if min_class_count >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores, sens_scores, spec_scores = [], [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        tree_model.fit(X_train, y_train)
        y_pred = tree_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        TN = np.sum(cm) - (FP + FN + TP)

        with np.errstate(divide='ignore', invalid='ignore'):
            sens_par_classe = np.nan_to_num(TP / (TP + FN), nan=0.0)
            spec_par_classe = np.nan_to_num(TN / (TN + FP), nan=0.0)

        acc_scores.append(acc)
        sens_scores.append(np.mean(sens_par_classe))
        spec_scores.append(np.mean(spec_par_classe))

    tree_model.fit(X, y)
    return np.mean(acc_scores), np.mean(sens_scores), np.mean(spec_scores), tree_model


# 3. La grande boucle de recherche
total_tests = len(combinaisons_cinematiques) * len(k_range)
test_actuel = 1

print(f"Lancement de {total_tests} tests de configurations...")

for kin_vars in combinaisons_cinematiques:
    kin_vars_list = list(kin_vars)
    df_temp = df_pat_final.dropna(subset=kin_vars_list).copy()

    if df_temp.empty: continue

    X_kin_scaled = StandardScaler().fit_transform(df_temp[kin_vars_list])

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
        df_temp['Cluster_GMM'] = gmm.fit_predict(X_kin_scaled)

        # --- CALCUL DE LA RÉPARTITION POUR L'EXCEL ---
        df_temp['Patient_Visit'] = df_temp['ID_Patient'].astype(str) + "_" + df_temp['ID_Visite'].astype(str)
        repartition_list = []
        for c_id in range(k):
            sub = df_temp[df_temp['Cluster_GMM'] == c_id]
            n_pat = sub['ID_Patient'].nunique()
            n_vis = sub['Patient_Visit'].nunique()
            n_ess = len(sub)
            repartition_list.append(f"C{c_id}: {n_pat}P/{n_vis}V/{n_ess}E")
        repartition_str = " | ".join(repartition_list)

        df_tree_base = df_temp.sort_values(by=['ID_Patient', 'ID_Visite']).drop_duplicates(subset=['ID_Patient'],
                                                                                           keep='first')

        acc_scores, sens_scores, spec_scores, _ = evaluate_tree_silently(df_tree_base, cols_scores)
        acc_autres, sens_autres, spec_autres, _ = evaluate_tree_silently(df_tree_base, cols_autres)

        max_acc = max(acc_scores, acc_autres)

        if acc_scores > acc_autres:
            type_gagnant = "Scores"
            best_sens = sens_scores
            best_spec = spec_scores
        else:
            type_gagnant = "Clinique (Autre)"
            best_sens = sens_autres
            best_spec = spec_autres

        resultats_pipeline.append({
            'Variables_Cinematiques': kin_vars_list,
            'Nb_Clusters': k,
            'Max_Accuracy': max_acc,
            'Sensibilite': best_sens,
            'Specificite': best_spec,
            'Meilleur_Predictif': type_gagnant,
            'Détails_Clusters (Pat/Vis/Essais)': repartition_str  # <-- AJOUT ICI
        })

        if max_acc > meilleur_score_global:
            meilleur_score_global = max_acc
            meilleure_configuration = {
                'Variables_Cinematiques': kin_vars_list,
                'Nb_Clusters': k,
                'Type_Clinique': 'cols_scores' if acc_scores > acc_autres else 'cols_autres',
                'Variables_X': cols_scores if acc_scores > acc_autres else cols_autres,
                'Dataframe': df_tree_base.copy()
            }

        print(
            f"Test {test_actuel}/{total_tests} | Cinématique: {kin_vars_list} | k={k} -> Max Acc: {max_acc * 100:.1f}%")
        test_actuel += 1

# ==============================================================================
# --- 6. AFFICHAGE DES RÉSULTATS GAGNANTS ET DU MEILLEUR ARBRE ---
# ==============================================================================
df_resultats = pd.DataFrame(resultats_pipeline).sort_values(by='Max_Accuracy', ascending=False)
print("\n" + "=" * 50)
print("🏆 RECHERCHE TERMINÉE - TOP 3 DES CONFIGURATIONS")
print("=" * 50)
print(df_resultats[['Variables_Cinematiques', 'Nb_Clusters', 'Max_Accuracy', 'Meilleur_Predictif']].head(3).to_string(
    index=False))

print("\nGénération de l'arbre final pour la meilleure configuration...")
df_best = meilleure_configuration['Dataframe']
best_vars_cliniques = meilleure_configuration['Variables_X']
_, _, _, best_tree_model = evaluate_tree_silently(df_best, best_vars_cliniques)

X_best = df_best[best_vars_cliniques].copy()
cols_valides = [c for c in best_vars_cliniques if X_best[c].isna().mean() <= 0.3]

plt.figure(figsize=(25, 12))
class_names_str = [f"Cluster {c}" for c in best_tree_model.classes_]

plot_tree(best_tree_model,
          feature_names=cols_valides,
          class_names=class_names_str,
          filled=True,
          rounded=True,
          fontsize=9)

plt.title(
    f"Meilleur Arbre (Acc: {meilleur_score_global * 100:.1f}%) | Clusters: {meilleure_configuration['Nb_Clusters']} | Variables: {meilleure_configuration['Variables_Cinematiques']}",
    fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "Best_Decision_Tree.png"), dpi=300)
plt.show()

df_resultats.to_excel(os.path.join(output_plot_folder, "Resultats_Pipeline_Complet.xlsx"), index=False)
print(f"\n✅ Pipeline terminé ! Les résultats ont été sauvegardés dans :\n{output_plot_folder}")