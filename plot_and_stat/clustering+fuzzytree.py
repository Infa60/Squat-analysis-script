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

# ---> NOUVEAU : CHOIX DE L'AGRÉGATION DES ESSAIS <---
# Options : 'mean' (moyenne), 'median' (médiane), 'max_flexion' (le meilleur essai), 'none' (tous les essais)
AGGREGATION_METHOD = 'mean'
type_of_analysis = AGGREGATION_METHOD
# ----------------------------------------------------

ALGOS_A_TESTER = ['GMM', 'KMEANS', 'HAC']

DEFAULT_FPS = 50
CUTOFF_FREQ = 3

# --- CHOIX DE LA MÉTHODE PROM (Z-score, Soucie ou Papageorgiou) ---
METHODE_PROM = 'Soucie'

# ---> CHOIX DE LA SYNCHRONISATION <---
# True = Tronc et Tibia pris au moment du maximum du Genou (At_Max)
# False = Tronc et Tibia pris à leur propre maximum absolu (Max)
SYNC_WITH_KNEE_MAX = False
# -----------------------------------------------

# --- DÉFINITION DES VARIABLES DE CLUSTERING ---
FEATURES_BASE = ['Knee', 'Tibia', 'Trunk', 'Knee_Frontal_Dynamic']

FEATURE_MAPPING_MAX = {
    'Knee': 'Knee_Flexion_Max',
    'Trunk': 'Trunk_Lean_At_Max' if SYNC_WITH_KNEE_MAX else 'Trunk_Lean_Max',
    'Tibia': 'Tibia_Lean_At_Max' if SYNC_WITH_KNEE_MAX else 'Tibia_Lean_Max',
    'Knee_Frontal_Dynamic': 'Knee_Frontal_Delta'
}

FEATURES_MAX = [FEATURE_MAPPING_MAX[f] for f in FEATURES_BASE]

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"
frontal_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_Frontal_all.pkl"
features_str = "_".join(FEATURES_BASE)


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
        return 'Right > Left' if c.find('droit') < c.find('gauch') or c.find('right') < c.find('left') else 'Left > Right'
    elif has_right:
        return 'Right'
    elif has_left:
        return 'Left'
    else:
        return 'Unknown'

def get_filtered_curve(curve, fps):
    c_array = np.array(curve)
    valid = ~np.isnan(c_array)
    if not valid.any(): return None
    c_interp = np.interp(np.arange(len(c_array)), np.where(valid)[0], c_array[valid])
    return butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)

def calculate_valgus_varus_frontal(hip, knee, ankle, side):
    v1 = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    angle_deg = np.degrees(np.arctan2(cross_product, dot_product))
    if side == 'gauche': angle_deg = -angle_deg
    return angle_deg

def get_clustering_model(algo_name, k):
    if algo_name == 'GMM': return GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=5)
    elif algo_name == 'KMEANS': return KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algo_name == 'HAC': return AgglomerativeClustering(n_clusters=k)
    else: raise ValueError(f"Algorithme {algo_name} non reconnu.")

def evaluate_tree_silently(df_tree, variables_cliniques, target_col='Cluster_ID', current_depth=5):
    X = df_tree[variables_cliniques].copy()
    y = df_tree[target_col]

    colonnes_a_garder = []
    for col in variables_cliniques:
        if X[col].isna().mean() <= 0.3:
            X[col] = X[col].fillna(X[col].median())
            colonnes_a_garder.append(col)

    X = X[colonnes_a_garder]
    if X.empty: return 0.0, 0.0, 0.0, None, [], []

    comptage = y.value_counts()
    min_class_count = comptage.min()

    tree_model = DecisionTreeClassifier(max_depth=current_depth, min_samples_leaf=3, criterion='entropy', random_state=42)
    n_splits = 5

    if min_class_count >= n_splits:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_scores, sens_scores, spec_scores = [], [], []
    sens_matrix, spec_matrix = [], []

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

        sens_matrix.append(sens_par_classe)
        spec_matrix.append(spec_par_classe)

    tree_model.fit(X, y)
    mean_sens_per_class = np.mean(sens_matrix, axis=0)
    mean_spec_per_class = np.mean(spec_matrix, axis=0)

    return np.mean(acc_scores), np.mean(sens_scores), np.mean(spec_scores), tree_model, mean_sens_per_class, mean_spec_per_class


def categorize_clinical_variable(series):
    """
    Reproduit la logique de l'article de S. Armand et al.
    Convertit une variable continue en 3 classes (0: Low, 1: Average, 2: High)
    basé sur la distribution de la cohorte.
    """
    p5 = series.quantile(0.05)
    median = series.median()
    p95 = series.quantile(0.95)

    # Pour simplifier la logique floue en logique discrète (Crisp) compatible avec scikit-learn :
    conditions = [
        (series <= (p5 + median) / 2),
        (series > (p5 + median) / 2) & (series <= (median + p95) / 2),
        (series > (median + p95) / 2)
    ]
    choices = [0, 1, 2]  # 0: Low, 1: Average, 2: High
    return pd.Series(np.select(conditions, choices, default=np.nan), index=series.index)


# ==============================================================================
# --- 3. CHARGEMENT ET PRÉPARATION UNIFIÉE ---
# ==============================================================================
print(f"1. Chargement et extraction unifiée depuis les matrices 3D (Sagittal + Frontal)...")

df_master_pat = pd.read_pickle(master_db_patient_file)

# --- FILTRAGE DES COLONNES pROM ---
prom_cols = [c for c in df_master_pat.columns if 'pROM_' in c]
cols_to_drop = [c for c in prom_cols if METHODE_PROM not in c]
df_master_pat = df_master_pat.drop(columns=cols_to_drop)

rename_dict = {c: c.replace(f"_{METHODE_PROM}", "") for c in df_master_pat.columns if 'pROM_' in c and METHODE_PROM in c}
df_master_pat = df_master_pat.rename(columns=rename_dict)
print(f"   -> Méthode pROM sélectionnée : {METHODE_PROM} ({len(cols_to_drop)} colonnes redondantes supprimées)")

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

        k_filt = get_filtered_curve(k_f, fps)
        tr_filt = get_filtered_curve(tr_f, fps)
        tib_filt = get_filtered_curve(tib_f, fps)

        k_max, tr_at_max, tib_at_max, tr_abs_max, tib_abs_max, kf_max, kf_delta = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        if k_filt is not None:
            idx_sagittal = np.argmax(k_filt)
            k_max = k_filt[idx_sagittal]

            if tr_filt is not None:
                tr_at_max = tr_filt[idx_sagittal]
                tr_abs_max = np.max(tr_filt)
            if tib_filt is not None:
                tib_at_max = tib_filt[idx_sagittal]
                tib_abs_max = np.max(tib_filt)

            if 'Keypoints_frontal' in row and isinstance(row['Keypoints_frontal'], np.ndarray):
                kpts_fro = row['Keypoints_frontal']
                fps_fro = row['FPS_frontal'] if pd.notna(row['FPS_frontal']) and row['FPS_frontal'] > 0 else DEFAULT_FPS
                n_frames_fro = kpts_fro.shape[0]

                time_seconds = idx_sagittal / fps
                idx_frontal = int(round(time_seconds * fps_fro))
                idx_frontal = min(max(idx_frontal, 0), n_frames_fro - 1)

                kf_start = calculate_valgus_varus_frontal(kpts_fro[0, 12], kpts_fro[0, 14], kpts_fro[0, 16], 'droite')
                kf_max = calculate_valgus_varus_frontal(kpts_fro[idx_frontal, 12], kpts_fro[idx_frontal, 14], kpts_fro[idx_frontal, 16], 'droite')
                kf_delta = kf_max - kf_start

        entry = {'ID_Patient': row['ID_Patient'], 'ID_Visite': row['ID_Visite'], 'File_Sagittal': row['File_Sagittal'],
                 'Diagnostic': row['Diagnostic'], 'Diag_Lateralite': row.get('Diag_Lateralite', 'Autre / Inconnu'),
                 'Scores_GMFCS': row.get('Scores_GMFCS', np.nan),
                 'Knee_Flexion_Max': k_max,
                 'Trunk_Lean_At_Max': tr_at_max,
                 'Tibia_Lean_At_Max': tib_at_max,
                 'Trunk_Lean_Max': tr_abs_max,
                 'Tibia_Lean_Max': tib_abs_max,
                 'Knee_Frontal_Max': kf_max,
                 'Knee_Frontal_Delta': kf_delta}

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
print(f"2. Application de la méthode d'agrégation : {AGGREGATION_METHOD}")

if AGGREGATION_METHOD == 'none':
    df_pat_final = df_pat_processed.copy()

elif AGGREGATION_METHOD == 'max_flexion':
    # --- NOUVEAU : On ne garde que l'essai avec le plus grand Knee_Flexion_Max pour chaque visite ---
    idx_max = df_pat_processed.groupby(['ID_Patient', 'ID_Visite'])['Knee_Flexion_Max'].idxmax()
    df_pat_final = df_pat_processed.loc[idx_max].reset_index(drop=True)

elif AGGREGATION_METHOD in ['mean', 'median']:
    clin_cols = [c for c in df_pat_processed.columns if c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_', 'pROM_'))]
    agg_dict = {f: AGGREGATION_METHOD for f in FEATURES_MAX}
    agg_dict.update({c: AGGREGATION_METHOD for c in clin_cols})
    agg_dict.update({'Scores_GMFCS': 'first', 'Diag_Lateralite': 'first'})
    df_pat_final = df_pat_processed.groupby(['ID_Patient', 'ID_Visite', 'Diagnostic']).agg(agg_dict).reset_index()

else:
    raise ValueError(f"Méthode d'agrégation '{AGGREGATION_METHOD}' non reconnue.")

# ==============================================================================
# --- 5 & 6. PIPELINE AUTOMATISÉ POUR CHAQUE ALGORITHME ---
# ==============================================================================
toutes_variables_cinematiques = [FEATURE_MAPPING_MAX[f] for f in FEATURES_BASE]
k_range = range(2, 6)
depth_range = [3, 4, 5, 6]

combinaisons_cinematiques = []
for r in range(1, len(toutes_variables_cinematiques) + 1):
    combinaisons_cinematiques.extend(list(combinations(toutes_variables_cinematiques, r)))

all_clinical_cols = [c for c in df_pat_final.columns if c.startswith(('Force_', 'ROM_', 'Spastic_', 'Score_')) and not c.endswith('G')]

cols_scores = [c for c in all_clinical_cols if 'Score' in c]
cols_autres = [c for c in all_clinical_cols if 'Score' not in c]

# Application à toutes vos colonnes cliniques
for col in cols_autres:
    df_pat_final[col + '_Fuzzy'] = categorize_clinical_variable(df_pat_final[col])

total_tests = len(combinaisons_cinematiques) * len(k_range) * len(depth_range)

for current_algo in ALGOS_A_TESTER:
    print(f"\n{'=' * 60}")
    print(f"🚀 LANCEMENT DU PIPELINE POUR L'ALGORITHME : {current_algo}")
    print(f"{'=' * 60}")

    sync_suffix = "SyncKnee" if SYNC_WITH_KNEE_MAX else "AbsMax"
    output_plot_folder = fr"{main_path}\Results_v2\Plot_{current_algo}_{type_of_analysis}_{sync_suffix}"
    os.makedirs(output_plot_folder, exist_ok=True)

    resultats_pipeline = []
    meilleur_score_global = 0
    meilleure_configuration = {}
    test_actuel = 1

    for kin_vars in combinaisons_cinematiques:
        kin_vars_list = list(kin_vars)
        df_temp = df_pat_final.dropna(subset=kin_vars_list).copy()

        if df_temp.empty: continue
        X_kin_scaled = StandardScaler().fit_transform(df_temp[kin_vars_list])

        for k in k_range:
            model = get_clustering_model(current_algo, k)
            df_temp['Cluster_ID'] = model.fit_predict(X_kin_scaled)

            df_temp['Patient_Visit'] = df_temp['ID_Patient'].astype(str) + "_" + df_temp['ID_Visite'].astype(str)
            repartition_list = []
            cluster_valide = True

            for c_id in range(k):
                sub = df_temp[df_temp['Cluster_ID'] == c_id]
                n_pat = sub['ID_Patient'].nunique()

                if n_pat < 8: cluster_valide = False

                n_vis = sub['Patient_Visit'].nunique()
                n_ess = len(sub)
                repartition_list.append(f"C{c_id}: {n_pat}P/{n_vis}V/{n_ess}E")

            repartition_str = " | ".join(repartition_list)

            if not cluster_valide:
                print(f"[{current_algo}] Rejeté (Cinématique: {kin_vars_list} | k={k}) -> Un cluster a moins de 8 patients.")
                test_actuel += len(depth_range)
                continue

            df_tree_base = df_temp.sort_values(by=['ID_Patient', 'ID_Visite']).drop_duplicates(subset=['ID_Patient'], keep='first')

            majority_baseline = df_tree_base['Cluster_ID'].value_counts().max() / len(df_tree_base)
            best_acc_for_current_setup = -1

            for depth in depth_range:
                acc_scores, sens_scores, spec_scores, _, sens_pc_scores, spec_pc_scores = evaluate_tree_silently(
                    df_tree_base, cols_scores, target_col='Cluster_ID', current_depth=depth)
                acc_autres, sens_autres, spec_autres, _, sens_pc_autres, spec_pc_autres = evaluate_tree_silently(
                    df_tree_base, cols_autres, target_col='Cluster_ID', current_depth=depth)

                max_acc = max(acc_scores, acc_autres)

                if acc_scores > acc_autres:
                    type_gagnant = "Scores"
                    best_sens, best_spec = sens_scores, spec_scores
                    best_sens_pc, best_spec_pc = sens_pc_scores, spec_pc_scores
                else:
                    type_gagnant = "Clinique (Autre)"
                    best_sens, best_spec = sens_autres, spec_autres
                    best_sens_pc, best_spec_pc = sens_pc_autres, spec_pc_autres

                if max_acc <= majority_baseline + 0.02:
                    print(f"[{current_algo}] Rejeté (Ciném: {kin_vars_list} | k={k} | depth={depth}) -> Acc ({max_acc * 100:.1f}%) <= Baseline Majoritaire ({majority_baseline * 100:.1f}%)")
                    test_actuel += 1
                    continue

                seuil_global = 0.55
                seuil_groupe = 0.50

                min_sens_cluster = np.min(best_sens_pc)
                min_spec_cluster = np.min(best_spec_pc)

                if best_sens < seuil_global or best_spec < seuil_global or min_sens_cluster < seuil_groupe or min_spec_cluster < seuil_groupe:
                    print(f"[{current_algo}] Rejeté (Ciném: {kin_vars_list} | k={k} | depth={depth}) -> Qualité insuffisante (Sens Moy: {best_sens * 100:.1f}%, Min groupe: {min_sens_cluster * 100:.1f}%)")
                    test_actuel += 1
                    continue

                if max_acc <= best_acc_for_current_setup + 0.001:
                    print(f"[{current_algo}] Rejeté (Ciném: {kin_vars_list} | k={k} | depth={depth}) -> N'améliore pas l'arbre plus petit")
                    test_actuel += 1
                    continue

                best_acc_for_current_setup = max_acc

                sens_str = " | ".join([f"C{i}: {val * 100:.1f}%" for i, val in enumerate(best_sens_pc)])
                spec_str = " | ".join([f"C{i}: {val * 100:.1f}%" for i, val in enumerate(best_spec_pc)])

                resultats_pipeline.append({
                    'Algorithme': current_algo,
                    'Variables_Cinematiques': kin_vars_list,
                    'Nb_Clusters': k,
                    'Profondeur_Arbre': depth,
                    'Max_Accuracy': max_acc,
                    'Sensibilite_Moyenne': best_sens,
                    'Specificite_Moyenne': best_spec,
                    'Détails_Sensibilité': sens_str,
                    'Détails_Spécificité': spec_str,
                    'Baseline_Majoritaire': round(majority_baseline, 3),
                    'Meilleur_Predictif': type_gagnant,
                    'Détails_Clusters (Pat/Vis/Essais)': repartition_str
                })

                if max_acc > meilleur_score_global:
                    meilleur_score_global = max_acc
                    meilleure_configuration = {
                        'Variables_Cinematiques': kin_vars_list,
                        'Nb_Clusters': k,
                        'Profondeur_Arbre': depth,
                        'Type_Clinique': 'cols_scores' if acc_scores > acc_autres else 'cols_autres',
                        'Variables_X': cols_scores if acc_scores > acc_autres else cols_autres,
                        'Dataframe': df_tree_base.copy()
                    }

                print(f"[{current_algo}] ✅ VALIDÉ Test {test_actuel}/{total_tests} | Cinématique: {kin_vars_list} | k={k} | depth={depth} -> Max Acc: {max_acc * 100:.1f}%")
                test_actuel += 1

    # --- AFFICHAGE ET SAUVEGARDE POUR L'ALGORITHME EN COURS ---
    if not resultats_pipeline:
        print(f"⚠️ Aucun résultat valide n'a survécu aux filtres cliniques stricts pour {current_algo}.")
        continue

    df_resultats = pd.DataFrame(resultats_pipeline).sort_values(by='Max_Accuracy', ascending=False)
    print("\n" + "-" * 50)
    print(f"🏆 TOP 3 DES RÉSULTATS ROBUSTES - {current_algo}")
    print("-" * 50)
    print(df_resultats[['Variables_Cinematiques', 'Nb_Clusters', 'Profondeur_Arbre', 'Max_Accuracy', 'Sensibilite_Moyenne']].head(3).to_string(index=False))

    df_best = meilleure_configuration['Dataframe']
    best_vars_cliniques = meilleure_configuration['Variables_X']
    best_depth = meilleure_configuration['Profondeur_Arbre']

    _, _, _, best_tree_model, _, _ = evaluate_tree_silently(df_best, best_vars_cliniques, target_col='Cluster_ID', current_depth=best_depth)

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

    plt.title(f"Meilleur Arbre Robuste ({current_algo}) - Acc: {meilleur_score_global * 100:.1f}% | Clusters: {meilleure_configuration['Nb_Clusters']} | Prof: {best_depth}\nVariables: {meilleure_configuration['Variables_Cinematiques']}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_folder, f"Best_Decision_Tree_Robust_{current_algo}.png"), dpi=300)
    plt.close()

    df_resultats.to_excel(os.path.join(output_plot_folder, f"Resultats_Pipeline_Robust_{current_algo}.xlsx"), index=False)
    print(f"✅ Résultats sauvegardés dans :\n{output_plot_folder}\n")

print("\n🎉 ANALYSE MULTI-ALGORITHMES TERMINÉE AVEC SUCCÈS ! 🎉")