import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
import itertools
import re
from statsmodels.genmod.cov_struct import Exchangeable
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
AVERAGE_TRIALS_PER_VISIT = False  # Mode d'agrégation clinique (Script 2)
type_of_analysis = "average" if AVERAGE_TRIALS_PER_VISIT else "all_visit"

CLUSTERING_MODE = 'MAX'  # 'MAX', 'CURVE', ou 'BOTH'
ALGO_CLUSTERING = 'KMEANS'  # Options : 'HAC', 'GMM', 'KMEANS'
NUM_CLUSTERS = 3
DEFAULT_FPS = 50
CUTOFF_FREQ = 3
POINTS_NORMALISATION = 101

# --- DÉFINITION DES VARIABLES DE CLUSTERING (Script 1 - Mapping dynamique) ---
# Vous pouvez ajouter 'Knee_Frontal' dans FEATURES_BASE si vous voulez l'inclure dans le clustering
FEATURES_BASE = ['Knee', 'Trunk']

FEATURE_MAPPING = {
    'Knee': ('Knee_Flexion_Max', 'Knee_Curve'),
    'Trunk': ('Trunk_Lean_Max', 'Trunk_Curve'),
    'Tibia': ('Tibia_Lean_Max', 'Tibia_Curve'),
    'Knee_Frontal': ('Knee_Frontal_Max', 'Knee_Frontal_Curve')
}

FEATURES_MAX = [FEATURE_MAPPING[f][0] for f in FEATURES_BASE]
FEATURES_CURVE = [FEATURE_MAPPING[f][1] for f in FEATURES_BASE]

LABELS_MAX = {
    'Knee_Flexion_Max': 'Knee Flexion\n(0°=Straight)',
    'Tibia_Lean_Max': 'Tibia Lean\n(0°=Straight)',
    'Trunk_Lean_Max': 'Trunk Lean\n(0°=Straight)',
    'Knee_Frontal_Max': 'Knee Valgus/Varus\n(+ Valgus, - Varus)'
}

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"
frontal_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_Frontal_all.pkl"  # Frontal (Script 1)
master_db_healthy_file = fr"{main_path}\Data\Master_Database_Healthy_all.pkl"

features_str = "_".join(FEATURES_BASE)
output_plot_folder = fr"{main_path}\Results\Plot_{ALGO_CLUSTERING}_{NUM_CLUSTERS}_{features_str}_right_{type_of_analysis}"
os.makedirs(output_plot_folder, exist_ok=True)


# ==============================================================================
# --- 2. FONCTIONS UTILITAIRES & GRAPHIQUES ---
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

def get_clustering_model(algo, n_clusters):
    if algo == 'HAC':
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif algo == 'GMM':
        return GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, n_init=10)
    elif algo == 'KMEANS':
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        raise ValueError("Algo inconnu")

def process_kinematics(curve, fps):
    c_array = np.array(curve)
    valid = ~np.isnan(c_array)
    if not valid.any(): return np.zeros(POINTS_NORMALISATION), np.nan
    c_interp = np.interp(np.arange(len(c_array)), np.where(valid)[0], c_array[valid])
    c_filt = butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)
    c_norm = np.interp(np.linspace(0, 100, POINTS_NORMALISATION), np.linspace(0, 100, len(c_filt)), c_filt)
    return c_norm, np.max(c_filt)

def calculate_valgus_varus_frontal(hip, knee, ankle, side):
    """Calcule l'angle frontal: Valgus (+) et Varus (-) (Script 1)"""
    v1 = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    angle_deg = np.degrees(np.arctan2(cross_product, dot_product))
    if side == 'gauche': angle_deg = -angle_deg
    return angle_deg

# ---------------------------------------------------------
# BOXPLOTS (Avec ligne Frontale du Script 1)
# ---------------------------------------------------------
def plot_kinematic_boxplots(df, cluster_col, features, prefix_name):
    print(f"--- Génération des Boxplots Cinématiques ({prefix_name}) ---")
    num_plots = len(features)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 6))
    if num_plots == 1: axes = [axes]

    unique_labels = df[cluster_col].dropna().unique().tolist()
    patient_profiles = sorted([l for l in unique_labels if "Prof" in l])
    healthy_profiles = sorted([l for l in unique_labels if "Healthy" in l])
    order = patient_profiles + healthy_profiles

    palette = sns.color_palette("Set2", len(patient_profiles))
    if any("Healthy" in h for h in healthy_profiles): palette.append("grey")

    for i, var in enumerate(features):
        sns.boxplot(x=cluster_col, y=var, data=df, ax=axes[i], palette=palette, order=order, showfliers=False)
        sns.stripplot(x=cluster_col, y=var, data=df, ax=axes[i], color='black', alpha=0.4, jitter=True, order=order)
        axes[i].set_title(LABELS_MAX.get(var, var).replace('\n', ' '), fontweight='bold', size=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Angle (°)", fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
        # Ajout de la ligne repère Valgus/Varus si variable frontale
        if 'Frontal' in var: axes[i].axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_folder, f"Boxplots_Cinematiques_{prefix_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_comparison(df_plot, features, labels_map, pal_dict, output_path):
    df_radar_data = df_plot.groupby('Profile_Max_Label')[features].mean()
    mins_data, maxs_data = df_radar_data.min(), df_radar_data.max()
    ticks_plot, mins_plot, maxs_plot = {}, {}, {}

    for col in features:
        c_min = np.floor(mins_data[col] / 10) * 10
        c_max = np.ceil(maxs_data[col] / 10) * 10
        if c_max - c_min < 10: c_max += 10
        mins_plot[col], maxs_plot[col] = c_min, c_max
        ticks_plot[col] = np.linspace(c_min, c_max, 5)

    def normalize(val, col):
        if pd.isna(val) or (maxs_plot[col] - mins_plot[col]) == 0: return 0.5
        return 0.1 + 0.9 * ((val - mins_plot[col]) / (maxs_plot[col] - mins_plot[col]))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist() + [0]

    for profil in df_radar_data.index:
        values = [normalize(df_radar_data.loc[profil, c], c) for c in features]
        values += values[:1]
        is_healthy = "Healthy" in profil
        color = pal_dict.get(profil, 'grey')

        ax.plot(angles, values, color=color, linewidth=2.5 if is_healthy else 3.5,
                linestyle='--' if is_healthy else '-', label=profil, zorder=3 if not is_healthy else 2)
        if not is_healthy: ax.fill(angles, values, color=color, alpha=0.15)

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
    plt.title("Normalized Radar Plot vs Healthy Ref", size=18, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11, frameon=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# STATS GEE & BARPLOTS CLINIQUES (Version enrichie du Script 2)
# ---------------------------------------------------------
def run_gee_and_plot(df, cluster_col, prefix_name):
    print(f"\n--- Analyse Clinique pour la méthode : {prefix_name} ---")
    clinical_cols = [c for c in df.columns if c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_')) and not c.endswith('G')]

    results_gee = []
    df = df.copy()

    if AVERAGE_TRIALS_PER_VISIT:
        group_var = 'ID_Patient'
        print("ℹ️ Mode Moyenne activé : Groupement GEE par Patient (ID_Patient)")
    else:
        if 'ID_Visite' in df.columns and 'ID_Patient' in df.columns:
            df['Patient_Visit'] = df['ID_Patient'].astype(str) + "_" + df['ID_Visite'].astype(str)
        else:
            df['Patient_Visit'] = df['ID_Patient'].astype(str)
        group_var = 'Patient_Visit'
        print("ℹ️ Mode Multi-essais activé : Groupement GEE par Visite (Patient_Visit)")

    for var in clinical_cols:
        try:
            valid_df = df.dropna(subset=[var, group_var, cluster_col]).copy()
            n_patients_tot = valid_df['ID_Patient'].nunique()
            n_trials_tot = len(valid_df)

            if 'ID_Visite' in valid_df.columns:
                valid_df['Temp_Visit'] = valid_df['ID_Patient'].astype(str) + "_" + valid_df['ID_Visite'].astype(str)
                n_visits_tot = valid_df['Temp_Visit'].nunique()
            else:
                n_visits_tot = "N/A"

            if n_patients_tot > 0 and valid_df[cluster_col].nunique() > 1:
                cov_struct = Exchangeable()
                gee_model = smf.gee(f"{var} ~ C({cluster_col})", groups=group_var, data=valid_df,
                                    family=sm.families.Gaussian(), cov_struct=cov_struct).fit()
                min_p = gee_model.pvalues.filter(like="C(").min()
                post_hoc_str = ""

                if min_p < 0.05:
                    clusters = sorted(valid_df[cluster_col].unique())
                    pairs = list(itertools.combinations(clusters, 2))
                    pvals, valid_pairs = [], []

                    for c1, c2 in pairs:
                        subset_df = valid_df[valid_df[cluster_col].isin([c1, c2])]
                        if subset_df[group_var].nunique() >= 3 and subset_df[cluster_col].nunique() == 2:
                            try:
                                pair_cov = Exchangeable()
                                pair_model = smf.gee(f"{var} ~ C({cluster_col})", groups=group_var, data=subset_df,
                                                     family=sm.families.Gaussian(), cov_struct=pair_cov).fit()
                                p = pair_model.pvalues.filter(like="C(").min()
                                pvals.append(p)
                                valid_pairs.append((c1, c2))
                            except:
                                pvals.append(1.0)
                                valid_pairs.append((c1, c2))
                        else:
                            pvals.append(1.0)
                            valid_pairs.append((c1, c2))

                    if pvals:
                        reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
                        sig_pairs = []
                        for i, (c1, c2) in enumerate(valid_pairs):
                            if reject[i]:
                                n1 = re.search(r'Prof_(\d+)', str(c1))
                                n2 = re.search(r'Prof_(\d+)', str(c2))
                                name1 = f"P{n1.group(1)}" if n1 else str(c1).split()[0]
                                name2 = f"P{n2.group(1)}" if n2 else str(c2).split()[0]
                                sig_pairs.append(f"{name1} vs {name2} (p={pvals_corrected[i]:.3f})")
                        post_hoc_str = " | ".join(sig_pairs) if sig_pairs else "Aucune après correction"

                results_gee.append({
                    'Variable': var, 'p-value_GEE': min_p, 'Différences_Significatives': post_hoc_str,
                    'Total_Patients': n_patients_tot, 'Total_Visites': n_visits_tot, 'Total_Lignes': n_trials_tot
                })
        except:
            pass

    df_stats = pd.DataFrame(results_gee)

    if not df_stats.empty:
        means_clin = df.groupby(cluster_col)[clinical_cols].mean().T
        means_clin.columns = [f"Moyenne_{c}" for c in means_clin.columns]
        counts_clin = df.groupby(cluster_col)[clinical_cols].count().T
        counts_clin.columns = [f"N_Lignes_{c}" for c in counts_clin.columns]

        df_final_stats = pd.merge(df_stats, counts_clin, left_on='Variable', right_index=True, how='left')
        df_final_stats = pd.merge(df_final_stats, means_clin, left_on='Variable', right_index=True, how='left')
        df_final_stats['Significatif'] = df_final_stats['p-value_GEE'].apply(lambda x: "*" if pd.notnull(x) and x < 0.05 else "")
        df_final_stats = df_final_stats.sort_values('p-value_GEE')

        cols = ['Variable', 'Significatif', 'p-value_GEE', 'Différences_Significatives', 'Total_Patients', 'Total_Visites', 'Total_Lignes']
        for c in df[cluster_col].unique():
            if f"N_Lignes_{c}" in df_final_stats.columns and f"Moyenne_{c}" in df_final_stats.columns:
                cols.extend([f"N_Lignes_{c}", f"Moyenne_{c}"])
        df_final_stats = df_final_stats[[c for c in cols if c in df_final_stats.columns]]

        df_final_stats.to_excel(os.path.join(output_plot_folder, f"Tableau_GEE_{prefix_name}.xlsx"), index=False)

    # BARPLOTS GMFCS ET LATÉRALITÉ (Script 2)
    if 'Scores_GMFCS' in df.columns:
        df_gmfcs = df.copy()
        df_gmfcs['Scores_GMFCS'] = df_gmfcs['Scores_GMFCS'].fillna('Inconnu')
        crosstab_gmfcs = pd.crosstab(df_gmfcs[cluster_col], df_gmfcs['Scores_GMFCS'], normalize='index') * 100
        fig_gmfcs, ax_gmfcs = plt.subplots(figsize=(10, 6))
        crosstab_gmfcs.plot(kind='bar', stacked=True, ax=ax_gmfcs, colormap='viridis', edgecolor='darkslategrey')
        ax_gmfcs.set_title(f"Répartition du GMFCS par {cluster_col}", fontweight='bold')
        ax_gmfcs.set_ylabel("Pourcentage des patients (%)", fontweight='bold')
        ax_gmfcs.set_xlabel("")
        plt.legend(title='GMFCS Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_folder, f"Barplot_GMFCS_{prefix_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    if 'Diag_Lateralite' in df.columns:
        df_diag = df.copy()
        crosstab_diag = pd.crosstab(df_diag[cluster_col], df_diag['Diag_Lateralite'], normalize='index') * 100
        color_map = {
            'Hemi (Left)': '#D62728', 'Di (Left > Right)': '#FF9896', 'Di (Left)': '#FF9896',
            'Hemi (Right)': '#1F77B4', 'Di (Right > Left)': '#AEC7E8', 'Di (Right)': '#AEC7E8',
            'Hemi (Unknown)': '#7F7F7F', 'Di (Unknown)': '#C7C7C7', 'Autre (Unknown)': '#E3E3E3'
        }
        plot_colors = [color_map.get(col, '#999999') for col in crosstab_diag.columns]

        fig_diag, ax_diag = plt.subplots(figsize=(10, 6))
        crosstab_diag.plot(kind='bar', stacked=True, ax=ax_diag, color=plot_colors, edgecolor='darkslategrey')
        ax_diag.set_title(f"Diagnostic & Côté plus atteint par {cluster_col}", fontweight='bold')
        ax_diag.set_ylabel("Pourcentage des patients (%)", fontweight='bold')
        ax_diag.set_xlabel("")
        plt.legend(title='Type & Côté', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_folder, f"Barplot_Diag_Cote_{prefix_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

# ==============================================================================
# --- 3. CHARGEMENT ET PRÉPARATION UNIFIÉE (Synchronisation Frontale - Script 1) ---
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

        k_curve, k_max = process_kinematics(k_f, fps)
        tr_curve, tr_max = process_kinematics(tr_f, fps)
        tib_curve, tib_max = process_kinematics(tib_f, fps)

        kf_max = np.nan
        kf_curve = np.full(POINTS_NORMALISATION, np.nan)

        if 'Keypoints_frontal' in row and isinstance(row['Keypoints_frontal'], np.ndarray):
            kpts_fro = row['Keypoints_frontal']
            fps_fro = row['FPS_frontal'] if pd.notna(row['FPS_frontal']) and row['FPS_frontal'] > 0 else DEFAULT_FPS
            n_frames_fro = kpts_fro.shape[0]

            kf_all = [calculate_valgus_varus_frontal(kpts_fro[f, 12], kpts_fro[f, 14], kpts_fro[f, 16], 'droite') for f in range(n_frames_fro)]
            kf_curve, _ = process_kinematics(kf_all, fps_fro)

            # Synchronisation sur le pic sagittal
            c_interp = np.interp(np.arange(len(k_f)), np.where(~np.isnan(k_f))[0], np.array(k_f)[~np.isnan(k_f)])
            c_filt = butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)
            idx_sagittal = np.argmax(c_filt)

            time_seconds = idx_sagittal / fps
            idx_frontal = int(round(time_seconds * fps_fro))
            idx_frontal = min(max(idx_frontal, 0), n_frames_fro - 1)

            kf_max = calculate_valgus_varus_frontal(kpts_fro[idx_frontal, 12], kpts_fro[idx_frontal, 14], kpts_fro[idx_frontal, 16], 'droite')

        entry = {'ID_Patient': row['ID_Patient'], 'ID_Visite': row['ID_Visite'], 'File_Sagittal': row['File_Sagittal'],
                 'Diagnostic': row['Diagnostic'], 'Diag_Lateralite': row.get('Diag_Lateralite', 'Autre / Inconnu'),
                 'Scores_GMFCS': row.get('Scores_GMFCS', np.nan),
                 'Knee_Curve': k_curve, 'Trunk_Curve': tr_curve, 'Tibia_Curve': tib_curve, 'Knee_Frontal_Curve': kf_curve,
                 'Knee_Flexion_Max': k_max, 'Trunk_Lean_Max': tr_max, 'Tibia_Lean_Max': tib_max, 'Knee_Frontal_Max': kf_max}

        for col in df_base_pat.columns:
            if col.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_')):
                entry[col] = pd.to_numeric(row[col], errors='coerce')
        processed_pat.append(entry)
    except:
        pass

df_pat_processed = pd.DataFrame(processed_pat)

# --- B. SAINS (YOUTUBE UNIQUEMENT) ---
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

        kf_curve, kf_max = np.full(POINTS_NORMALISATION, np.nan), np.nan

        processed_health.append({
            'ID_Subject': row.get('ID_Subject', f'Subj_{idx}'), 'Source': 'YouTube_Healthy',
            'Knee_Curve': k_curve, 'Trunk_Curve': tr_curve, 'Tibia_Curve': tib_curve, 'Knee_Frontal_Curve': kf_curve,
            'Knee_Flexion_Max': k_max, 'Trunk_Lean_Max': tr_max, 'Tibia_Lean_Max': tib_max, 'Knee_Frontal_Max': kf_max
        })
    except:
        pass

df_healthy_processed = pd.DataFrame(processed_health)
if not df_healthy_processed.empty:
    df_healthy_processed['Diagnostic'], df_healthy_processed['Profile_Max_Label'], df_healthy_processed['Profile_Curve_Label'] = 'Sain', 'Healthy Ref', 'Healthy Ref'

# ==============================================================================
# --- 4. AGGRÉGATION & ALIGNEMENT ---
# ==============================================================================
if AVERAGE_TRIALS_PER_VISIT:
    clin_cols = [c for c in df_pat_processed.columns if c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_'))]
    agg_dict = {f: lambda x: np.mean(np.vstack(x), axis=0) for f in FEATURES_CURVE}
    agg_dict.update({f: 'mean' for f in FEATURES_MAX})
    agg_dict.update({c: 'mean' for c in clin_cols})
    agg_dict.update({'Scores_GMFCS': 'first', 'Diag_Lateralite': 'first'})
    df_pat_final = df_pat_processed.groupby(['ID_Patient', 'ID_Visite', 'Diagnostic']).agg(agg_dict).reset_index()
else:
    df_pat_final = df_pat_processed.copy()

df_max_pat = df_pat_final.dropna(subset=FEATURES_MAX).copy()
df_curve_pat_agg = df_pat_final.copy()
df_healthy_max = df_healthy_processed.groupby(['ID_Subject', 'Source'])[FEATURES_MAX].mean().reset_index() if not df_healthy_processed.empty else pd.DataFrame()
if not df_healthy_max.empty: df_healthy_max['Diagnostic'], df_healthy_max['Profile_Max_Label'] = 'Sain', 'Healthy Ref'

if CLUSTERING_MODE == 'BOTH':
    common_patients = set(df_max_pat['ID_Patient']).intersection(set(df_curve_pat_agg['ID_Patient']))
    sort_cols = ['ID_Patient', 'ID_Visite']
    if not AVERAGE_TRIALS_PER_VISIT and 'File_Sagittal' in df_max_pat.columns: sort_cols.append('File_Sagittal')
    df_max_pat = df_max_pat[df_max_pat['ID_Patient'].isin(common_patients)].sort_values(sort_cols).reset_index(drop=True)
    df_curve_pat_agg = df_curve_pat_agg[df_curve_pat_agg['ID_Patient'].isin(common_patients)].sort_values(sort_cols).reset_index(drop=True)

# ==============================================================================
# --- 5. CLUSTERING MAX & VIZ ---
# ==============================================================================
if CLUSTERING_MODE in ['MAX', 'BOTH'] and not df_max_pat.empty:
    print("\n--- CLUSTERING SUR VALEURS MAXIMALES ---")
    X_max_scaled = StandardScaler().fit_transform(df_max_pat[FEATURES_MAX])
    df_max_pat['Cluster_Max'] = get_clustering_model(ALGO_CLUSTERING, NUM_CLUSTERS).fit_predict(X_max_scaled)
    df_max_pat['Profile_Max_Label'] = df_max_pat['Cluster_Max'].apply(lambda x: f"Max_Prof_{x + 1}")

    prof_counts = df_max_pat['Profile_Max_Label'].value_counts().to_dict()
    df_max_pat['Profile_Max_Label'] = df_max_pat['Profile_Max_Label'].apply(lambda x: f"{x} (n={prof_counts[x]})")
    pal_max = {f"Max_Prof_{i + 1} (n={prof_counts.get(f'Max_Prof_{i + 1}', 0)})": sns.color_palette("Set2", NUM_CLUSTERS)[i] for i in range(NUM_CLUSTERS)}
    pal_max.update({"Healthy Ref": "grey"})

    df_plot_max = pd.concat([df_max_pat, df_healthy_max], ignore_index=True)

    if len(FEATURES_MAX) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_plot_max, x=FEATURES_MAX[0], y=FEATURES_MAX[1], hue='Profile_Max_Label', style='Diagnostic', palette=pal_max, s=120, ax=ax)
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.savefig(os.path.join(output_plot_folder, "Max_ScatterPlot.png"), dpi=300, bbox_inches='tight')
        plt.close()
    elif len(FEATURES_MAX) >= 3:
        plot_radar_comparison(df_plot_max, FEATURES_MAX, LABELS_MAX, pal_max, os.path.join(output_plot_folder, "Max_RadarPlot.png"))

    plot_kinematic_boxplots(df_plot_max, 'Profile_Max_Label', FEATURES_MAX, 'Max_Values')
    run_gee_and_plot(df_max_pat, 'Profile_Max_Label', 'Max_Values')

# ==============================================================================
# --- 6. CLUSTERING CURVES & VIZ (Avec filtre PCA du script 1 et Viz du script 2) ---
# ==============================================================================
if CLUSTERING_MODE in ['CURVE', 'BOTH'] and not df_curve_pat_agg.empty:
    print("\n--- CLUSTERING SUR SÉRIES TEMPORELLES ---")

    # Sécurité PCA (Script 1) : Empêche la PCA de planter s'il y a des NaN
    valid_curve_idx = df_curve_pat_agg.apply(lambda row: not any(pd.isna(row[f]).any() for f in FEATURES_CURVE), axis=1)
    df_curve_pca_ready = df_curve_pat_agg[valid_curve_idx].copy()

    if not df_curve_pca_ready.empty:
        X_curve_pca = PCA(n_components=0.95, random_state=42).fit_transform(StandardScaler().fit_transform(
            np.array([list(itertools.chain(*[row[f] for f in FEATURES_CURVE])) for _, row in df_curve_pca_ready.iterrows()])))

        df_curve_pca_ready['Cluster_Curve'] = get_clustering_model(ALGO_CLUSTERING, NUM_CLUSTERS).fit_predict(X_curve_pca)
        df_curve_pca_ready['Profile_Curve_Label'] = df_curve_pca_ready['Cluster_Curve'].apply(lambda x: f"Curve_Prof_{x + 1}")

        prof_counts_c = df_curve_pca_ready['Profile_Curve_Label'].value_counts().to_dict()
        df_curve_pca_ready['Profile_Curve_Label'] = df_curve_pca_ready['Profile_Curve_Label'].apply(lambda x: f"{x} (n={prof_counts_c[x]})")
        prof_order_c = sorted(df_curve_pca_ready['Profile_Curve_Label'].unique())
        pal_curve = {f"Curve_Prof_{i + 1} (n={prof_counts_c.get(f'Curve_Prof_{i + 1}', 0)})": sns.color_palette("Set2", NUM_CLUSTERS)[i] for i in range(NUM_CLUSTERS)}

        time_vec = np.linspace(0, 100, POINTS_NORMALISATION)

        fig_c, axes_c = plt.subplots(1, len(FEATURES_CURVE), figsize=(6 * len(FEATURES_CURVE), 5), sharex=True)
        if len(FEATURES_CURVE) == 1: axes_c = [axes_c]

        for prof in prof_order_c:
            sub = df_curve_pca_ready[df_curve_pca_ready['Profile_Curve_Label'] == prof]
            for i, f in enumerate(FEATURES_CURVE):
                mat = np.vstack(sub[f].values)
                m_c, s_c = np.mean(mat, axis=0), np.std(mat, axis=0)
                axes_c[i].plot(time_vec, m_c, color=pal_curve[prof], linewidth=2.5, label=prof if i == 0 else "")
                axes_c[i].fill_between(time_vec, m_c - s_c, m_c + s_c, color=pal_curve[prof], alpha=0.15)

        # Affichage des courbes saines en référence (Script 2)
        if not df_healthy_processed.empty:
            for i, f in enumerate(FEATURES_CURVE):
                # On filtre les NaN potentiels pour les sains
                valid_h = df_healthy_processed[~df_healthy_processed[f].apply(lambda x: np.isnan(x).any() if isinstance(x, np.ndarray) else True)]
                if not valid_h.empty:
                    mat_h = np.vstack(valid_h[f].values)
                    m_h, s_h = np.mean(mat_h, axis=0), np.std(mat_h, axis=0)
                    axes_c[i].plot(time_vec, m_h, color='grey', linestyle='--', linewidth=2, label="Healthy Ref" if i == 0 else "")
                    axes_c[i].fill_between(time_vec, m_h - s_h, m_h + s_h, color='grey', alpha=0.1)

        for i, ax in enumerate(axes_c): ax.set_title(FEATURES_CURVE[i].replace('_Curve', ' Mean Curve'), fontweight='bold')
        fig_c.legend(loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=NUM_CLUSTERS, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_folder, "Curve_TimeSeries_MeanPlot.png"), dpi=300, bbox_inches='tight')
        plt.close()

        plot_kinematic_boxplots(pd.concat([df_curve_pca_ready, df_healthy_processed], ignore_index=True), 'Profile_Curve_Label', FEATURES_MAX, 'Time_Series_Curves')
        run_gee_and_plot(df_curve_pca_ready, 'Profile_Curve_Label', 'Time_Series_Curves')

# ==============================================================================
# --- 7. COMPARAISON DIRECTE : MAX vs CURVES ---
# ==============================================================================
if CLUSTERING_MODE == 'BOTH' and 'Profile_Max_Label' in df_max_pat.columns and 'Profile_Curve_Label' in df_curve_pca_ready.columns:
    df_compare = pd.DataFrame({'Profile_Max': df_max_pat['Profile_Max_Label'].reset_index(drop=True),
                               'Profile_Curve': df_curve_pca_ready['Profile_Curve_Label'].reset_index(drop=True)}).dropna()
    ari_score = adjusted_rand_score(df_compare['Profile_Max'], df_compare['Profile_Curve'])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pd.crosstab(df_compare['Profile_Max'], df_compare['Profile_Curve']), annot=True, fmt='d', cmap='Blues',
                cbar=False, ax=ax, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Chevauchement des Clusters (ARI = {ari_score:.2f})", fontweight='bold')
    plt.savefig(os.path.join(output_plot_folder, "Heatmap_Comparison.png"), dpi=300)
    plt.close()

print("\n✅ Pipeline de Clustering complet terminé avec succès ! Le fichier Excel est enrichi.")


# ==============================================================================
# --- 8. SAUVEGARDE DES LABELS DE CLUSTERING POUR LE FUZZY TREE ---
# ==============================================================================
print("\n--- SAUVEGARDE DES RÉSULTATS POUR LE FUZZY TREE ---")

# 1. Définition des clés de jointure selon la méthode de calcul
# Si on a moyenné par visite, on fusionne sur le Patient et la Visite.
# Si on a gardé tous les essais, on fusionne de manière unique jusqu'au nom du fichier (essai).
if AVERAGE_TRIALS_PER_VISIT:
    merge_keys = ['ID_Patient', 'ID_Visite']
else:
    merge_keys = ['ID_Patient', 'ID_Visite', 'File_Sagittal']

# 2. Récupération des labels de clustering MAX (si calculés)
df_labels_max = pd.DataFrame()
if CLUSTERING_MODE in ['MAX', 'BOTH'] and 'Profile_Max_Label' in df_max_pat.columns:
    df_labels_max = df_max_pat[merge_keys + ['Profile_Max_Label']].copy()
    # Suppression des doublons potentiels pour la fusion sécurisée
    df_labels_max = df_labels_max.drop_duplicates(subset=merge_keys)

# 3. Récupération des labels de clustering CURVE (si calculés)
df_labels_curve = pd.DataFrame()
if CLUSTERING_MODE in ['CURVE', 'BOTH'] and 'Profile_Curve_Label' in df_curve_pca_ready.columns:
    df_labels_curve = df_curve_pca_ready[merge_keys + ['Profile_Curve_Label']].copy()
    # Suppression des doublons potentiels pour la fusion sécurisée
    df_labels_curve = df_labels_curve.drop_duplicates(subset=merge_keys)

# 4. Fusion avec la base de données traitée (df_pat_processed)
# On utilise df_pat_processed plutôt que df_master_pat car les variables
# cliniques y sont déjà converties en numérique (prêt pour l'IA).
df_export = df_pat_processed.copy()

if not df_labels_max.empty:
    df_export = pd.merge(df_export, df_labels_max, on=merge_keys, how='left')

if not df_labels_curve.empty:
    df_export = pd.merge(df_export, df_labels_curve, on=merge_keys, how='left')

# 5. Sauvegarde du nouveau fichier pickle
export_file_name = f"Master_Database_Patient_Clustered_{ALGO_CLUSTERING}_{NUM_CLUSTERS}_{type_of_analysis}.pkl"
export_path = os.path.join(main_path, "Data", export_file_name)

df_export.to_pickle(export_path)

# Petit résumé dans la console pour vérifier que tout s'est bien passé
lignes_exportees = len(df_export)
patients_exportes = df_export['ID_Patient'].nunique()
print(f"✅ Sauvegarde réussie : {export_file_name}")
print(f"   -> {lignes_exportees} lignes (essais) sauvegardées pour {patients_exportes} patients uniques.")
if 'Profile_Max_Label' in df_export.columns:
    print(f"   -> Répartition Max   : {df_export['Profile_Max_Label'].value_counts().to_dict()}")
if 'Profile_Curve_Label' in df_export.columns:
    print(f"   -> Répartition Curve : {df_export['Profile_Curve_Label'].value_counts().to_dict()}")