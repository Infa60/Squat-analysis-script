"""
=========================================================================================
GENERATE MASTER TRIAL DATABASE (PICKLE EXPORT)
=========================================================================================
Objectif: Créer un fichier unique contenant TOUTES les données brutes par essai.
Structure: 1 ligne = 1 essai vidéo + Matrices 3D conservées grâce au format Pickle.
Amélioration: Tous les angles enregistrés considèrent 0° = Posture Droite/Tendue.
=========================================================================================
"""

from fonction import *
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import cv2

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"

# Création d'une liste contenant les deux dossiers à analyser (Qualisys et Vicon)
results_folders = [
    fr"{main_path}\Data\Processed\CP_qualisys\Sagittal_View\Results",
    fr"{main_path}\Data\Processed\CP_vicon\Results" # <-- Ajout du deuxième chemin ici
]

clinical_table_file = fr"{main_path}\Data\Processed\Results_Clinical_Table.xlsx"
examen_clinique_file = fr"{main_path}\Participant\examen_clinique_results_with_scores_data.xlsx"
main_file = fr"{main_path}\Participant\Main_file.xlsx"

# === Export ===
output_file_pkl = fr"{main_path}\Data\Master_Database_Patient_all.pkl"

# --- CONFIGURATION TECHNIQUE ---
modality = "ViTPose_Huge"
DEFAULT_FPS, CUTOFF_FREQ = 50, 3

# ==============================================================================
# 1. CHARGEMENT DE LA CINÉMATIQUE, FRÉQUENCE VIDÉO ET RAW DATA
# ==============================================================================
print("1. Extraction de la cinématique, des FPS et des données brutes (RAW)...")
ml_data = []

# Parcourir chaque dossier principal (Qualisys, puis Vicon)
for folder in results_folders:
    if not os.path.exists(folder):
        print(f"⚠️ Dossier introuvable ignoré : {folder}")
        continue

    print(f"-> Analyse du dossier : {folder}")

    for patient_name in os.listdir(folder):
        patient_path = os.path.join(folder, patient_name)

        # S'assurer que c'est bien un dossier (ignore les éventuels fichiers cachés à la racine)
        if not os.path.isdir(patient_path):
            continue

        mat_file = os.path.join(patient_path, f"{patient_name}_Results_Corrected.mat")
        video_file = os.path.join(patient_path, "ViTPose_Huge_Corrected.avi")

        if not os.path.exists(mat_file):
            continue

        try:
            video_fps = np.nan
            current_fps = DEFAULT_FPS

            # Extraction des FPS
            if os.path.exists(video_file):
                cap = cv2.VideoCapture(video_file)
                if cap.isOpened():
                    extracted_fps = cap.get(cv2.CAP_PROP_FPS)
                    if extracted_fps > 0:
                        video_fps = extracted_fps
                        current_fps = extracted_fps
                cap.release()

            # Chargement des Keypoints bruts
            data = sio.loadmat(mat_file)
            kpts = data[modality][0, 0]['Keypoints'][:, 0, :, :]  # kpts = [frames, marqueurs, coordonnees]

            # Calcul des angles bruts (ViTPose donne ~180° pour une posture droite)
            n_frames = kpts.shape[0]

            k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in
                   range(n_frames)]
            tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
            tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

            # Filtrage
            k_f_filt = butter_lowpass_filter(
                np.interp(np.arange(len(k_f)), np.where(~np.isnan(k_f))[0], np.array(k_f)[~np.isnan(k_f)]), CUTOFF_FREQ,
                current_fps)
            tr_f_filt = butter_lowpass_filter(
                np.interp(np.arange(len(tr_f)), np.where(~np.isnan(tr_f))[0], np.array(tr_f)[~np.isnan(tr_f)]), CUTOFF_FREQ,
                current_fps)
            tib_f_filt = butter_lowpass_filter(
                np.interp(np.arange(len(tib_f)), np.where(~np.isnan(tib_f))[0], np.array(tib_f)[~np.isnan(tib_f)]),
                CUTOFF_FREQ, current_fps)

            # TRANSFORMATION : 0° = Posture Droite / Tendue
            k_f_0 = k_f_filt
            tr_f_0 = tr_f_filt
            tib_f_0 = tib_f_filt

            # Le pic du squat est maintenant le MAXIMUM (puisque 0 = debout)
            idx = np.argmax(k_f_0)

            ml_data.append({
                'File_Sagittal': patient_name + ".avi",
                'Video_FPS': video_fps,

                'Knee_Flexion_At_Max': k_f_0[idx],
                'Trunk_Lean_At_Max': tr_f_0[idx],
                'Tibia_Lean_At_Max': tib_f_0[idx],

                'Knee_Flexion_Max': np.max(k_f_0),
                'Trunk_Lean_Max': np.max(tr_f_0),
                'Tibia_Lean_Max': np.max(tib_f_0),

                # Conservation de la matrice brute 3D dans le DataFrame
                'Raw_Keypoints': kpts
            })
        except Exception as e:
            # Utile pour débugger au cas où un fichier poserait problème
            # print(f"Erreur avec {patient_name}: {e}")
            pass

df_ml = pd.DataFrame(ml_data)

# ==============================================================================
# 2. CHARGEMENT TABLE CLINIQUE
# ==============================================================================
print("2. Chargement de la Table Clinique...")
df_trials = pd.read_excel(clinical_table_file)
df_trials['ID_Visite'] = df_trials['File_Sagittal'].apply(
    lambda x: int(str(x).split('_')[1]) if '_' in str(x) else np.nan)
df_trials['ID_Patient'] = df_trials['File_Sagittal'].apply(lambda x: str(x).split('_')[0] if '_' in str(x) else np.nan)

if all(col in df_trials.columns for col in ['Knee valgus_R', 'Knee valgus_L']):
    df_trials['Knee_Valgus_Binaire'] = df_trials.apply(
        lambda r: 1 if 'Y' in r[['Knee valgus_R', 'Knee valgus_L']].values else 0, axis=1)
if all(col in df_trials.columns for col in ['Heel rise_R', 'Heel rise_L']):
    df_trials['Heel_Rise_Binaire'] = df_trials.apply(
        lambda r: 1 if 'Y' in r[['Heel rise_R', 'Heel rise_L']].values else 0, axis=1)

master_df = pd.merge(df_trials, df_ml, on='File_Sagittal', how='inner')

# ==============================================================================
# 3. AJOUT DU MAIN FILE
# ==============================================================================
print("3. Ajout des métadonnées du Main File...")
df_main = pd.read_excel(main_file)
if 'Diagnostic' in df_main.columns:
    import numpy as np

    # Vérifier que la colonne existe bien
    if 'CoteDiagnostic' in df_main.columns:
        # 1. Convertir en texte (évite les erreurs si des cellules sont vides ou numériques)
        df_main['CoteDiagnostic'] = df_main['CoteDiagnostic'].astype(str)

        # 2. Supprimer les balises Excel/XML (_x000D_) et les sauts de ligne (\n)
        df_main['CoteDiagnostic'] = df_main['CoteDiagnostic'].str.replace('_x000D_', ' ', regex=False)
        df_main['CoteDiagnostic'] = df_main['CoteDiagnostic'].str.replace('\n', ' ', regex=False)

        # 3. Remplacer les espaces multiples par un seul espace (pour "1     Gauche")
        df_main['CoteDiagnostic'] = df_main['CoteDiagnostic'].str.replace(r'\s+', ' ', regex=True)

        # 4. Supprimer les espaces restants au tout début ou à la toute fin de la chaîne
        df_main['CoteDiagnostic'] = df_main['CoteDiagnostic'].str.strip()

        # 5. Remettre les vraies valeurs nulles (car astype(str) transforme les vrais vides en texte "nan")
        df_main['CoteDiagnostic'] = df_main['CoteDiagnostic'].replace('nan', np.nan)

cols_a_garder_main = df_main.columns.difference(master_df.columns).tolist() + ['ID_Visite']
master_df = pd.merge(master_df, df_main[cols_a_garder_main], on='ID_Visite', how='left')

# ==============================================================================
# 4. FUSION EXAMEN CLINIQUE DÉTAILLÉ
# ==============================================================================
print("4. Nettoyage et ajout des examens cliniques...")
df_clin_total = None
for s in ["Visite_ExamenClinique_Force", "Visite_ExamenClinique_Spastic", "Visite_ExamenClinique_ROM", "Visite_Scores"]:
    temp_df = pd.read_excel(examen_clinique_file, sheet_name=s)
    temp_df.columns = temp_df.columns.str.strip()
    for col in temp_df.columns:
        if col != 'ID_Visite':
            temp_df[col] = temp_df[col].apply(clean_clinical_value)

    prefix = s.split('_')[-1]
    rename_dict = {col: f"{prefix}_{col}" for col in temp_df.columns if col != 'ID_Visite'}
    temp_df.rename(columns=rename_dict, inplace=True)

    if df_clin_total is None:
        df_clin_total = temp_df
    else:
        cols_a_garder_temp = temp_df.columns.difference(df_clin_total.columns).tolist() + ['ID_Visite']
        df_clin_total = pd.merge(df_clin_total, temp_df[cols_a_garder_temp], on='ID_Visite', how='outer')

cols_a_garder_clin = df_clin_total.columns.difference(master_df.columns).tolist() + ['ID_Visite']

# Fonction pour retrouver une colonne même si son préfixe change (Force_, Spastic_, ROM_)
def get_col(col_name):
    match = [c for c in df_clin_total.columns if c.endswith(f"_{col_name}")]
    if match:
        return pd.to_numeric(df_clin_total[match[0]], errors='coerce')
    return pd.Series(np.nan, index=df_clin_total.index)

# 1. SPASTICITY
df_clin_total['Score_Spasticity_Hip'] = get_col('HancheFlexAD')
df_clin_total['Score_Spasticity_Knee'] = pd.DataFrame({'a': get_col('GenouFlexAD'), 'b': get_col('DuncanElyTestAD')}).sum(axis=1)
df_clin_total['Score_Spasticity_Ankle'] = pd.DataFrame({'a': get_col('TricepsAD'), 'b': get_col('SoleusAD')}).median(axis=1)
df_clin_total['Score_Spasticity_Composite'] = df_clin_total[['Score_Spasticity_Hip', 'Score_Spasticity_Knee', 'Score_Spasticity_Ankle']].sum(axis=1)

# 2. WEAKNESS
df_clin_total['Score_Weakness_Hip'] = pd.DataFrame({'a': get_col('FHancheFlexD'), 'b': get_col('FHancheExtD')}).sum(axis=1)
df_clin_total['Score_Weakness_Knee'] = pd.DataFrame({'a': get_col('FGenouFlexD'), 'b': get_col('FGenouExtD')}).sum(axis=1)
median_ankle_weakness = pd.DataFrame({'a': get_col('FTricepsD'), 'b': get_col('FSoleusD')}).median(axis=1)
df_clin_total['Score_Weakness_Ankle'] = median_ankle_weakness + get_col('FTibAntD')
df_clin_total['Score_Weakness_Composite'] = df_clin_total[['Score_Weakness_Hip', 'Score_Weakness_Knee', 'Score_Weakness_Ankle']].sum(axis=1)

# 3. SELECTIVITY
df_clin_total['Score_Selectivity_Hip'] = pd.DataFrame({'a': get_col('SHancheFlexD'), 'b': get_col('SHancheExtD')}).sum(axis=1)
df_clin_total['Score_Selectivity_Knee'] = pd.DataFrame({'a': get_col('SGenouFlexD'), 'b': get_col('SGenouExtD')}).sum(axis=1)
median_ankle_sel = pd.DataFrame({'a': get_col('STricepsD'), 'b': get_col('SSoleusD')}).median(axis=1)
df_clin_total['Score_Selectivity_Ankle'] = median_ankle_sel + get_col('STibAntD')
df_clin_total['Score_Selectivity_Composite'] = df_clin_total[['Score_Selectivity_Hip', 'Score_Selectivity_Knee', 'Score_Selectivity_Ankle']].sum(axis=1)


# 4.1 pROM (via Z-scores)
def calculate_zscore(series, invert_sign=False):
    """
    Calcule le Z-score.
    Si invert_sign=True, un Z-score négatif (valeur sous la moyenne) devient positif.
    Très utile si une petite valeur clinique indique une sévérité (ex: perte d'amplitude).
    """
    z_score = (series - series.mean()) / series.std(ddof=1)

    if invert_sign:
        return -z_score
    return z_score


# Récupération des données brutes
hip_zscore_prom_raw = get_col('ThomasTestD')
knee_zscore_prom_raw = pd.DataFrame({'a': get_col('AnglePopUniD'), 'b': get_col('AnglePopBiD')}).median(axis=1)
ankle_zscore_prom_raw = pd.DataFrame({'a': get_col('ChevilleFlexDor1D'), 'b': get_col('ChevilleFlexDor2D')}).median(axis=1)

# Application des Z-scores
df_clin_total['Score_zscore_pROM_Hip'] = calculate_zscore(hip_zscore_prom_raw, invert_sign=False)
df_clin_total['Score_zscore_pROM_Knee'] = calculate_zscore(knee_zscore_prom_raw, invert_sign=True)
df_clin_total['Score_zscore_pROM_Ankle'] = calculate_zscore(ankle_zscore_prom_raw, invert_sign=True)
df_clin_total['Score_zscore_pROM_Composite'] = df_clin_total[['Score_zscore_pROM_Hip', 'Score_zscore_pROM_Knee', 'Score_zscore_pROM_Ankle']].mean(
    axis=1)


# 4.2 pROM
def calculate_prom_score(series):
    p25, p75 = series.quantile(0.25), series.quantile(0.75)
    conditions = [(series < p25), (series >= p25) & (series <= p75), (series > p75)]
    choices = [2, 1, 0]
    return pd.Series(np.select(conditions, choices, default=np.nan), index=series.index)

hip_Papageorgiou_prom_raw = get_col('ThomasTestD')
knee_Papageorgiou_prom_raw = pd.DataFrame({'a': get_col('AnglePopUniD'), 'b': get_col('AnglePopBiD')}).median(axis=1)
ankle_Papageorgiou_prom_raw = pd.DataFrame({'a': get_col('ChevilleFlexDor1D'), 'b': get_col('ChevilleFlexDor2D')}).median(axis=1)

df_clin_total['Score_Papageorgiou_pROM_Hip'] = calculate_prom_score(hip_Papageorgiou_prom_raw)
df_clin_total['Score_Papageorgiou_pROM_Knee'] = calculate_prom_score(knee_Papageorgiou_prom_raw)
df_clin_total['Score_Papageorgiou_pROM_Ankle'] = calculate_prom_score(ankle_Papageorgiou_prom_raw)
df_clin_total['Score_Papageorgiou_pROM_Composite'] = df_clin_total[['Score_Papageorgiou_pROM_Hip', 'Score_Papageorgiou_pROM_Knee', 'Score_Papageorgiou_pROM_Ankle']].sum(axis=1)

# 4.3 pROM (Méthode de Soucie)

def categorize_soucie(series, age_series, cutoffs_young, cutoffs_old):
    """
    Transforme les angles bruts en scores (0, 1, 2, 3) selon les seuils de Soucie.
    """
    # 1. SÉCURITÉ : Forcer la conversion en nombres (les textes ou erreurs deviennent NaN)
    series = pd.to_numeric(series, errors='coerce')
    age_series = pd.to_numeric(age_series, errors='coerce')

    if cutoffs_young is None:
        # Pour le Thomas Test
        conditions = [series <= 10, series > 10]
        choices = [3, 0]
        return pd.Series(np.select(conditions, choices, default=np.nan), index=series.index)

    is_young = age_series < 8
    is_old = age_series >= 8

    conditions = [
        # Moins de 8 ans
        (is_young) & (series <= cutoffs_young[0]),
        (is_young) & (series > cutoffs_young[0]) & (series <= cutoffs_young[1]),
        (is_young) & (series > cutoffs_young[1]) & (series <= cutoffs_young[2]),
        (is_young) & (series > cutoffs_young[2]),
        # 8 ans et plus
        (is_old) & (series <= cutoffs_old[0]),
        (is_old) & (series > cutoffs_old[0]) & (series <= cutoffs_old[1]),
        (is_old) & (series > cutoffs_old[1]) & (series <= cutoffs_old[2]),
        (is_old) & (series > cutoffs_old[2])
    ]
    choices = [0, 1, 2, 3, 0, 1, 2, 3]

    return pd.Series(np.select(conditions, choices, default=np.nan), index=series.index)


# Récupération de l'âge des patients
age_patients = df_clin_total['ID_Visite'].map(master_df.drop_duplicates(subset=['ID_Visite']).set_index('ID_Visite')['Age'])
# Sécurité : on s'assure que c'est bien au format numérique
age_patients = pd.to_numeric(age_patients, errors='coerce')

# 1. ÉTAPE DE CATÉGORISATION (Application des seuils de Soucie)

# Hanche : Thomas Test
cat_thomas_D = categorize_soucie(get_col('ThomasTestD'), age_patients, None, None)

# Genou : Flexion (GenouFlexD) et Extension (GenouExtD)
cat_knee_flex_D = categorize_soucie(get_col('GenouFlexD'), age_patients, [148.90, 150.20, 151.50], [140.60, 142.25, 143.90])
cat_knee_ext_D = categorize_soucie(get_col('GenouExtD'), age_patients, [2.40, 3.50, 4.60], [1.20, 2.10, 3.00])

# Cheville : Dorsiflexion/Triceps (ChevilleFlexDor1D) et Flexion Plantaire (ChevilleFlexPlanD)
cat_ankle_df_D = categorize_soucie(get_col('ChevilleFlexDor1D'), age_patients, [21.90, 23.80, 25.70], [15.25, 16.80, 18.35])
cat_ankle_pf_D = categorize_soucie(get_col('ChevilleFlexPlanD'), age_patients, [59.60, 61.45, 63.30], [52.80, 55.05, 57.30])


# 2. ÉTAPE D'AGRÉGATION (Calcul de la médiane sur les catégories)

df_clin_total['Score_Soucie_pROM_Hip'] = cat_thomas_D
df_clin_total['Score_Soucie_pROM_Knee'] = pd.DataFrame({'flex': cat_knee_flex_D, 'ext': cat_knee_ext_D}).median(axis=1)
df_clin_total['Score_Soucie_pROM_Ankle'] = pd.DataFrame({'df': cat_ankle_df_D, 'pf': cat_ankle_pf_D}).median(axis=1)
df_clin_total['Score_Soucie_pROM_Composite'] = df_clin_total[['Score_Soucie_pROM_Hip', 'Score_Soucie_pROM_Knee', 'Score_Soucie_pROM_Ankle']].sum(axis=1)


# Fusion de l'examen clinique (et des scores) avec le master_df
cols_a_garder_clin = df_clin_total.columns.difference(master_df.columns).tolist() + ['ID_Visite']
master_df = pd.merge(master_df, df_clin_total[cols_a_garder_clin], on='ID_Visite', how='left')

# ==============================================================================
# 5. EXPORT PICKLE
# ==============================================================================
print("5. Organisation des colonnes et Export Pickle...")

cols_prioritaires = ['ID_Patient', 'ID_Visite', 'File_Sagittal', 'Diagnostic', 'Video_FPS', 'Knee_Valgus_Binaire',
                     'Heel_Rise_Binaire',
                     'Knee_Flexion_At_Max', 'Trunk_Lean_At_Max', 'Tibia_Lean_At_Max',
                     'Knee_Flexion_Max', 'Trunk_Lean_Max', 'Tibia_Lean_Max', 'Raw_Keypoints']

cols_existantes = [c for c in cols_prioritaires if c in master_df.columns]
remaining_cols = [c for c in master_df.columns if c not in cols_existantes]

master_df = master_df[cols_existantes + remaining_cols]

master_df.to_pickle(output_file_pkl)

print(f"\n✅ SUCCÈS : Fichier Master Pickle généré avec {len(master_df)} lignes et {len(master_df.columns)} colonnes.")
print(f"Chemin : {output_file_pkl}")