import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import re
import warnings
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from fonction import *  # Contient vos fonctions de filtrage/calcul d'angles

warnings.filterwarnings('ignore')

# --- PANDAS CONFIGURATION ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

# ==============================================================================
# --- 1. CONFIGURATION DE L'ANALYSE STRATIFIÉE ---
# ==============================================================================
AVERAGE_TRIALS_PER_VISIT = True
type_of_analysis = "average" if AVERAGE_TRIALS_PER_VISIT else "all_visit"

DEFAULT_FPS = 50
CUTOFF_FREQ = 3
POINTS_NORMALISATION = 101

FEATURES_BASE = ['Knee', 'Trunk', 'Tibia']
FEATURES_MAX = [f"{f}_Flexion_Max" if f == 'Knee' else f"{f}_Lean_Max" for f in FEATURES_BASE]
FEATURES_CURVE = [f"{f}_Curve" for f in FEATURES_BASE]

LABELS_MAX = {
    'Knee_Flexion_Max': 'Knee Flexion\n(0°=Straight)',
    'Tibia_Lean_Max': 'Tibia Lean\n(0°=Straight)',
    'Trunk_Lean_Max': 'Trunk Lean\n(0°=Straight)'
}

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"
master_db_healthy_file = fr"{main_path}\Data\Master_Database_Healthy_all.pkl"

# Nouveau dossier de sortie orienté Groupes Cliniques
output_plot_folder = fr"{main_path}\Results\Clinical_Stratification_RightOnly_{type_of_analysis}"
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
    return 'Right' if has_right else ('Left' if has_left else 'Unknown')


def process_kinematics(curve, fps):
    c_array = np.array(curve)
    valid = ~np.isnan(c_array)
    if not valid.any(): return np.zeros(POINTS_NORMALISATION), np.nan
    c_interp = np.interp(np.arange(len(c_array)), np.where(valid)[0], c_array[valid])
    c_filt = butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)
    c_norm = np.interp(np.linspace(0, 100, POINTS_NORMALISATION), np.linspace(0, 100, len(c_filt)), c_filt)
    return c_norm, np.max(c_filt)


# ---------------------------------------------------------
# BOXPLOTS AVEC BARRES DE SIGNIFICATIVITÉ
# ---------------------------------------------------------
def plot_stratified_boxplots(df, group_col, features, prefix_name, sig_dict=None):
    print(f"--- Génération des Boxplots pour {group_col} ({prefix_name}) ---")
    num_plots = len(features)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 7))
    if num_plots == 1: axes = [axes]

    unique_groups = sorted([str(g) for g in df[group_col].dropna().unique() if g != 'Sain'])
    order = unique_groups + ['Sain'] if 'Sain' in df[group_col].values else unique_groups
    index_map = {group: i for i, group in enumerate(order)}

    palette = sns.color_palette("Set2", len(unique_groups))
    if 'Sain' in order: palette.append("grey")

    for i, var in enumerate(features):
        ax = axes[i]
        sns.boxplot(x=group_col, y=var, data=df, ax=ax, palette=palette, order=order, showfliers=False)
        sns.stripplot(x=group_col, y=var, data=df, ax=ax, color='black', alpha=0.4, jitter=True, order=order)

        ax.set_title(LABELS_MAX.get(var, var).replace('\n', ' '), fontweight='bold', size=14)
        ax.set_xlabel("")
        ax.set_ylabel("Angle (°)", fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # --- AJOUT DES BARRES D'ÉTOILES ---
        if sig_dict and var in sig_dict and sig_dict[var]:
            y_max = df[var].max()
            y_range = df[var].max() - df[var].min()
            if y_range == 0 or pd.isna(y_range): y_range = 10
            step = y_range * 0.08
            current_y = y_max + step

            # Trier les paires pour afficher les barres les plus courtes en premier (évite les chevauchements)
            pairs_stats = sig_dict[var]
            pairs_stats.sort(key=lambda x: abs(index_map.get(x[0], 0) - index_map.get(x[1], 0)))

            for g1, g2, pval in pairs_stats:
                if g1 in index_map and g2 in index_map:
                    x1, x2 = index_map[g1], index_map[g2]
                    star = "***" if pval < 0.001 else "**" if pval < 0.01 else "*"

                    # Tracer la ligne et le texte
                    ax.plot([x1, x1, x2, x2], [current_y, current_y + step * 0.2, current_y + step * 0.2, current_y],
                            lw=1.5, color='black')
                    ax.text((x1 + x2) / 2, current_y + step * 0.2, star, ha='center', va='bottom', color='black',
                            fontsize=12, fontweight='bold')
                    current_y += step * 1.2

            # Ajuster la limite Y pour ne pas couper les barres
            ax.set_ylim(top=current_y + step)

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_folder, f"Boxplots_{group_col}_{prefix_name}.png"), dpi=300,
                bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# RADAR PLOT
# ---------------------------------------------------------
def plot_radar_stratified(df_plot, group_col, features, labels_map, output_path):
    df_radar_data = df_plot.groupby(group_col)[features].mean()
    df_radar_data = df_radar_data.drop(index=[i for i in ['Inconnu', 'Autre'] if i in df_radar_data.index],
                                       errors='ignore')

    mins_data, maxs_data = df_radar_data.min(), df_radar_data.max()
    ticks_plot, mins_plot, maxs_plot = {}, {}, {}

    for col in features:
        c_min = np.floor(mins_data[col] / 10) * 10
        c_max = np.ceil(maxs_data[col] / 10) * 10
        if c_max - c_min < 10: c_max += 10
        mins_plot[col], maxs_plot[col] = c_min, c_max
        ticks_plot[col] = np.linspace(c_min, c_max, 5)

    def normalize(val, col):
        return 0.1 + 0.9 * ((val - mins_plot[col]) / (maxs_plot[col] - mins_plot[col]))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist() + [0]

    groups = df_radar_data.index.tolist()
    colors = sns.color_palette("Set2", len(groups))
    pal_dict = {g: colors[i] if g != 'Sain' else 'grey' for i, g in enumerate(groups)}

    for group in groups:
        values = [normalize(df_radar_data.loc[group, c], c) for c in features]
        values += values[:1]
        is_healthy = (group == 'Sain')

        ax.plot(angles, values, color=pal_dict[group], linewidth=2.5 if is_healthy else 3.5,
                linestyle='--' if is_healthy else '-', label=group, zorder=3 if not is_healthy else 2)
        if not is_healthy: ax.fill(angles, values, color=pal_dict[group], alpha=0.15)

    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0.1, 1.0, 5))
    ax.set_yticklabels([])

    for i, angle in enumerate(angles[:-1]):
        for val_brute, val_norm in zip(ticks_plot[features[i]], np.linspace(0.1, 1.0, 5)):
            ax.text(angle, val_norm, f"{val_brute:.0f}°", ha='center', va='center', fontsize=9, color='dimgrey',
                    fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.7))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([labels_map.get(f, f) for f in features], fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', pad=40)
    plt.title(f"Radar Plot - Comparaison par {group_col}", size=18, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, frameon=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# FONCTION STATS GEE INTERNE ADAPTÉE (Retourne le dictionnaire des significativités)
# ---------------------------------------------------------
def run_gee_stratified(df, group_col, target_features, prefix_name):
    print(f"\n--- Analyse GEE pour la variable groupe : {group_col} ({prefix_name}) ---")

    df_clean = df.dropna(subset=[group_col]).copy()
    df_clean = df_clean[~df_clean[group_col].astype(str).str.contains('Inconnu|Autre', case=False)]

    results_gee = []
    sig_dict = {var: [] for var in target_features}  # Stocke les paires significatives

    # Regroupement robuste (Patients et Sujets Sains)
    # Utilise ID_Patient, et si c'est vide (sujets sains), utilise ID_Subject
    df_clean['Group_ID'] = df_clean.get('ID_Patient').fillna(
        df_clean.get('ID_Subject', pd.Series(df_clean.index))).astype(str)

    if AVERAGE_TRIALS_PER_VISIT:
        group_var = 'Group_ID'
    else:
        df_clean['Visit_ID'] = df_clean.get('ID_Visite', '1').astype(str)
        df_clean['Patient_Visit'] = df_clean['Group_ID'] + "_" + df_clean['Visit_ID']
        group_var = 'Patient_Visit'

    for var in target_features:
        try:
            valid_df = df_clean.dropna(subset=[var, group_var]).copy()
            n_patients_tot = valid_df['Group_ID'].nunique()
            n_trials_tot = len(valid_df)

            if valid_df[group_col].nunique() > 1 and n_patients_tot > 0:
                cov_struct = Exchangeable()
                gee_model = smf.gee(f"{var} ~ C({group_col})", groups=group_var, data=valid_df,
                                    family=sm.families.Gaussian(), cov_struct=cov_struct).fit()

                min_p = gee_model.pvalues.filter(like="C(").min()
                post_hoc_str = ""

                if min_p < 0.05:
                    groups_list = sorted(valid_df[group_col].unique())
                    pairs = list(itertools.combinations(groups_list, 2))
                    pvals, valid_pairs = [], []

                    for g1, g2 in pairs:
                        subset_df = valid_df[valid_df[group_col].isin([g1, g2])]
                        if subset_df[group_var].nunique() >= 3:
                            try:
                                pair_cov = Exchangeable()
                                pair_model = smf.gee(f"{var} ~ C({group_col})", groups=group_var, data=subset_df,
                                                     family=sm.families.Gaussian(), cov_struct=pair_cov).fit()
                                p = pair_model.pvalues.filter(like="C(").min()
                                pvals.append(p)
                                valid_pairs.append((g1, g2))
                            except:
                                pvals.append(1.0)
                                valid_pairs.append((g1, g2))
                        else:
                            pvals.append(1.0)
                            valid_pairs.append((g1, g2))

                    if pvals:
                        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                        sig_pairs_text = []
                        for i, (g1, g2) in enumerate(valid_pairs):
                            if reject[i]:
                                sig_pairs_text.append(f"{g1} vs {g2} (p={pvals_corrected[i]:.3f})")
                                sig_dict[var].append((g1, g2, pvals_corrected[i]))  # Sauvegarde pour boxplot
                        post_hoc_str = " | ".join(
                            sig_pairs_text) if sig_pairs_text else "Aucune différence après correction"

                results_gee.append({
                    'Variable': var,
                    'p-value_GEE': min_p,
                    'Différences_Significatives': post_hoc_str,
                    'Total_Patients': n_patients_tot,
                    'Total_Lignes': n_trials_tot
                })
        except Exception as e:
            pass

    df_stats = pd.DataFrame(results_gee)

    if not df_stats.empty:
        means_group = df_clean.groupby(group_col)[target_features].mean().T
        means_group.columns = [f"Moyenne_{c}" for c in means_group.columns]

        df_final_stats = pd.merge(df_stats, means_group, left_on='Variable', right_index=True, how='left')
        df_final_stats['Significatif'] = df_final_stats['p-value_GEE'].apply(
            lambda x: "*" if pd.notnull(x) and x < 0.05 else "")

        export_path = os.path.join(output_plot_folder, f"Tableau_GEE_{group_col}_{prefix_name}.xlsx")
        df_final_stats.to_excel(export_path, index=False)
        print(f"✅ Statistiques exportées : {export_path}")

    return sig_dict


# ==============================================================================
# --- 3. CHARGEMENT ET EXTRACTION AVEC FILTRE STRICT ET GMFCS ROBUSTE ---
# ==============================================================================
print(f"1. Chargement et extraction unifiée depuis les fichiers sources...")

# --- A. PATIENTS ---
df_master_pat = pd.read_pickle(master_db_patient_file)
mask_pat = (df_master_pat['Pose estimation'].astype(str).str.strip().str.upper() == 'Y') & \
           (df_master_pat['Caregiver assistance'].astype(str).str.strip() != '2') & \
           (df_master_pat['Hand-to-ground contact'].astype(str).str.strip() != '2') & \
           (df_master_pat['Heel_Rise_Binaire'] == 0) & \
           (df_master_pat['CoteDiagnostic'] != 'Gauche') & (df_master_pat['CoteDiagnostic'] != 'Gauche Droit')

df_base_pat = df_master_pat[mask_pat].copy()

# Création de la variable latérale
df_base_pat['Diag_Simple'] = df_base_pat['Diagnostic'].apply(simplifier_diagnostic)
df_base_pat['Clean_Side'] = df_base_pat.get('CoteDiagnostic', '').apply(clean_side)
df_base_pat['Diag_Lateralite'] = df_base_pat['Diag_Simple'] + " (" + df_base_pat['Clean_Side'] + ")"

# --- FILTRE STRICT : Uniquement Hemi (Right) et Di (Right > Left) ---
target_groups = ['Hemi (Right)', 'Di (Right > Left)']
df_base_pat = df_base_pat[df_base_pat['Diag_Lateralite'].isin(target_groups)].copy()

processed_pat = []
for idx, row in df_base_pat.iterrows():
    try:
        kpts = row['Raw_Keypoints']
        fps = row['Video_FPS'] if pd.notna(row['Video_FPS']) and row['Video_FPS'] > 0 else DEFAULT_FPS
        n_frames = kpts.shape[0]

        k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]
        tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
        tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

        k_curve, k_max = process_kinematics(k_f, fps)
        tr_curve, tr_max = process_kinematics(tr_f, fps)
        tib_curve, tib_max = process_kinematics(tib_f, fps)

        # Extraction ultra-robuste du GMFCS (gère les floats comme '1.0' ou ints)
        raw_gmfcs = row.get('Scores_GMFCS', np.nan)
        try:
            val = float(raw_gmfcs)
            gmfcs_clean = 'Inconnu' if np.isnan(val) else f"GMFCS_{int(val)}"
        except:
            gmfcs_clean = str(raw_gmfcs).strip()
            if gmfcs_clean == 'nan': gmfcs_clean = 'Inconnu'

        entry = {'ID_Patient': row['ID_Patient'], 'ID_Visite': row['ID_Visite'],
                 'Diagnostic': row['Diag_Lateralite'],
                 'Scores_GMFCS': gmfcs_clean,
                 'Knee_Curve': k_curve, 'Trunk_Curve': tr_curve, 'Tibia_Curve': tib_curve,
                 'Knee_Flexion_Max': k_max, 'Trunk_Lean_Max': tr_max, 'Tibia_Lean_Max': tib_max}

        processed_pat.append(entry)
    except:
        pass

df_pat_processed = pd.DataFrame(processed_pat)

# --- B. SAINS (POUR LA RÉFÉRENCE VISUELLE) ---
df_master_health = pd.read_pickle(master_db_healthy_file)
df_master_health = df_master_health[df_master_health['Source'] == 'YouTube_Healthy'].copy()

processed_health = []
for idx, row in df_master_health.iterrows():
    try:
        kpts = row['Raw_Keypoints']
        fps = row['Video_FPS'] if pd.notna(row['Video_FPS']) and row['Video_FPS'] > 0 else DEFAULT_FPS
        n_frames = kpts.shape[0]

        k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]
        tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
        tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

        k_curve, k_max = process_kinematics(k_f, fps)
        tr_curve, tr_max = process_kinematics(tr_f, fps)
        tib_curve, tib_max = process_kinematics(tib_f, fps)

        processed_health.append({
            'ID_Subject': row.get('ID_Subject', f'Subj_{idx}'), 'Diagnostic': 'Sain', 'Scores_GMFCS': 'Sain',
            'Knee_Curve': k_curve, 'Trunk_Curve': tr_curve, 'Tibia_Curve': tib_curve,
            'Knee_Flexion_Max': k_max, 'Trunk_Lean_Max': tr_max, 'Tibia_Lean_Max': tib_max
        })
    except:
        pass

df_healthy_processed = pd.DataFrame(processed_health)

# ==============================================================================
# --- 4. AGGRÉGATION / MOYENNAGE (PATIENTS & SAINS) ---
# ==============================================================================
if AVERAGE_TRIALS_PER_VISIT:
    agg_dict_pat = {f: lambda x: np.mean(np.vstack(x), axis=0) for f in FEATURES_CURVE}
    agg_dict_pat.update({f: 'mean' for f in FEATURES_MAX})
    agg_dict_pat.update({'Scores_GMFCS': 'first'})
    df_pat_final = df_pat_processed.groupby(['ID_Patient', 'ID_Visite', 'Diagnostic']).agg(agg_dict_pat).reset_index()

    agg_dict_health = {f: lambda x: np.mean(np.vstack(x), axis=0) for f in FEATURES_CURVE}
    agg_dict_health.update({f: 'mean' for f in FEATURES_MAX})
    agg_dict_health.update({'Diagnostic': 'first', 'Scores_GMFCS': 'first'})
    df_healthy_final = df_healthy_processed.groupby(['ID_Subject']).agg(agg_dict_health).reset_index()
else:
    df_pat_final = df_pat_processed.copy()
    df_healthy_final = df_healthy_processed.copy()

df_combined = pd.concat([df_pat_final, df_healthy_final], ignore_index=True)

# ==============================================================================
# --- 4.5 RÉSUMÉ DÉMOGRAPHIQUE DES INCLUSIONS ---
# ==============================================================================
print("\n" + "=" * 60)
print("--- RÉSUMÉ DES DONNÉES INCLUSES ---")
print("=" * 60)

# On utilise df_pat_processed car il contient la liste des essais/patients
# ayant passé tous les filtres d'inclusion, avant la fusion avec les sujets sains.
df_inclus = df_pat_processed.copy()

# 1. Total global (Patients + Sains + Essais)
total_patients = df_inclus['ID_Patient'].nunique()
total_visites = df_inclus['ID_Visite'].nunique()
total_essais_pat = len(df_inclus)  # Chaque ligne est un essai

total_sains = df_healthy_processed['ID_Subject'].nunique()
total_essais_sains = len(df_healthy_processed)

print(f"Total Enfants (Patients) uniques inclus : {total_patients}")
print(f"Total Visites uniques (Patients) incluses : {total_visites}")
print(f"Total Essais (Patients) inclus            : {total_essais_pat}\n")

print(f"Total Sujets Sains uniques inclus         : {total_sains}")
print(f"Total Essais (Sains) inclus               : {total_essais_sains}\n")

# 2. Répartition par Topologie (Diagnostic)
print("--- Répartition par Topologie (Hemi vs Di) ---")
resume_diag = df_inclus.groupby('Diagnostic').agg(
    Nombre_Enfants=('ID_Patient', 'nunique'),
    Nombre_Visites=('ID_Visite', 'nunique'),
    Nombre_Essais=('ID_Patient', 'count')  # Compte le nombre de lignes (essais)
).reset_index()
print(resume_diag.to_string(index=False))

# 3. Répartition par GMFCS
print("\n--- Répartition par GMFCS ---")
resume_gmfcs = df_inclus.groupby('Scores_GMFCS').agg(
    Nombre_Enfants=('ID_Patient', 'nunique'),
    Nombre_Visites=('ID_Visite', 'nunique'),
    Nombre_Essais=('ID_Patient', 'count')
).reset_index()

print(resume_gmfcs.to_string(index=False))

# 4. Exportation des données vers un fichier Excel
resume_path = os.path.join(output_plot_folder, "Demographie_Inclusions.xlsx")
with pd.ExcelWriter(resume_path) as writer:
    resume_diag.to_excel(writer, sheet_name="Par_Topologie", index=False)
    resume_gmfcs.to_excel(writer, sheet_name="Par_GMFCS", index=False)

print(f"\n✅ Résumé démographique calculé et exporté vers : {resume_path}")
print("=" * 60)

# Trouver les enfants qui ont plus d'un score GMFCS différent
gmfcs_par_patient = df_inclus.groupby('ID_Patient')['Scores_GMFCS'].nunique()
patients_multi_gmfcs = gmfcs_par_patient[gmfcs_par_patient > 1].index.tolist()

print("\n--- ATTENTION : Enfants avec plusieurs GMFCS différents ---")
if len(patients_multi_gmfcs) > 0:
    for pat in patients_multi_gmfcs:
        visites_pat = df_inclus[df_inclus['ID_Patient'] == pat][['ID_Visite', 'Scores_GMFCS']].drop_duplicates()
        print(f"Patient {pat} :")
        print(visites_pat.to_string(index=False))
else:
    print("Aucun enfant ne possède de GMFCS multiple. Tout est cohérent !")

# ==============================================================================
# --- 5. ANALYSE 1 : DIPLEGIC VS HEMIPLEGIC ---
# ==============================================================================
print("\n=== ANALYSE STRATIFIÉE 1 : DIPLEGIC (Right>Left) vs HEMIPLEGIC (Right) ===")

df_diag_analysis = df_combined[df_combined['Diagnostic'].isin(['Di (Right > Left)', 'Hemi (Right)', 'Sain'])].copy()

# A. Lancement Statistique (Retourne les paires significatives)
sig_dict_diag = run_gee_stratified(df_diag_analysis, 'Diagnostic', FEATURES_MAX, 'Kinematics')

# B. Boxplots (Trace les boxplots ET dessine les étoiles)
plot_stratified_boxplots(df_diag_analysis, 'Diagnostic', FEATURES_MAX, 'Max_Angles', sig_dict_diag)

# C. Radar Plot
plot_radar_stratified(df_diag_analysis, 'Diagnostic', FEATURES_MAX, LABELS_MAX,
                      os.path.join(output_plot_folder, "RadarPlot_Diagnostic.png"))

# D. Courbes
time_vec = np.linspace(0, 100, POINTS_NORMALISATION)
fig_c, axes_c = plt.subplots(1, len(FEATURES_CURVE), figsize=(18, 5))
colors_diag = {'Di (Right > Left)': '#1F77B4', 'Hemi (Right)': '#FF7F0E', 'Sain': 'grey'}

for diag_type, group in df_diag_analysis.groupby('Diagnostic'):
    if diag_type not in colors_diag: continue
    linestyle = '--' if diag_type == 'Sain' else '-'
    alpha_fill = 0.05 if diag_type == 'Sain' else 0.15

    for i, f in enumerate(FEATURES_CURVE):
        mat = np.vstack(group[f].values)
        m_c, s_c = np.mean(mat, axis=0), np.std(mat, axis=0)
        axes_c[i].plot(time_vec, m_c, color=colors_diag[diag_type], linestyle=linestyle, linewidth=2.5,
                       label=diag_type if i == 0 else "")
        axes_c[i].fill_between(time_vec, m_c - s_c, m_c + s_c, color=colors_diag[diag_type], alpha=alpha_fill)

for i, ax in enumerate(axes_c):
    ax.set_title(FEATURES_CURVE[i].replace('_Curve', ' Curve'), fontweight='bold')
    ax.set_xlabel('% du Cycle')
    ax.set_ylabel('Angle (°)')
fig_c.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "Curves_Comparison_Diagnostic.png"), dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# --- 6. ANALYSE 2 : COMPARAISON ENTRE NIVEAUX GMFCS ---
# ==============================================================================
print("\n=== ANALYSE STRATIFIÉE 2 : COMPARAISON SELON LE GMFCS ===")

df_gmfcs_analysis = df_combined[
    df_combined['Scores_GMFCS'].isin(['GMFCS_1', 'GMFCS_2', 'GMFCS_3', 'GMFCS_4', 'Sain'])].copy()

# A. Lancement Statistique
sig_dict_gmfcs = run_gee_stratified(df_gmfcs_analysis, 'Scores_GMFCS', FEATURES_MAX, 'Kinematics')

# B. Boxplots
plot_stratified_boxplots(df_gmfcs_analysis, 'Scores_GMFCS', FEATURES_MAX, 'Max_Angles', sig_dict_gmfcs)

# C. Radar Plot
plot_radar_stratified(df_gmfcs_analysis, 'Scores_GMFCS', FEATURES_MAX, LABELS_MAX,
                      os.path.join(output_plot_folder, "RadarPlot_GMFCS.png"))

# D. Courbes
fig_g, axes_g = plt.subplots(1, len(FEATURES_CURVE), figsize=(18, 5))
gmfcs_order = ['GMFCS_1', 'GMFCS_2', 'GMFCS_3', 'GMFCS_4', 'Sain']
colors_gmfcs = sns.color_palette("YlOrRd", 4)
colors_gmfcs_dict = {gmfcs_order[i]: colors_gmfcs[i] for i in range(4)}
colors_gmfcs_dict['Sain'] = 'grey'

for gmfcs_type, group in df_gmfcs_analysis.groupby('Scores_GMFCS'):
    if gmfcs_type not in colors_gmfcs_dict: continue
    linestyle = '--' if gmfcs_type == 'Sain' else '-'
    alpha_fill = 0.05 if gmfcs_type == 'Sain' else 0.12

    for i, f in enumerate(FEATURES_CURVE):
        mat = np.vstack(group[f].values)
        m_c, s_c = np.mean(mat, axis=0), np.std(mat, axis=0)
        axes_g[i].plot(time_vec, m_c, color=colors_gmfcs_dict[gmfcs_type], linestyle=linestyle, linewidth=2.5,
                       label=gmfcs_type if i == 0 else "")
        axes_g[i].fill_between(time_vec, m_c - s_c, m_c + s_c, color=colors_gmfcs_dict[gmfcs_type], alpha=alpha_fill)

for i, ax in enumerate(axes_g):
    ax.set_title(FEATURES_CURVE[i].replace('_Curve', ' Curve'), fontweight='bold')
    ax.set_xlabel('% du Cycle')
    ax.set_ylabel('Angle (°)')
fig_g.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "Curves_Comparison_GMFCS.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ Pipeline de stratification clinique terminé avec succès. Résultats sauvegardés dans : {output_plot_folder}")