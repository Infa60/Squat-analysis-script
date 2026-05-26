"""
=========================================================================================
Kinematic Clustering (Multi-Algorithms & Dynamic Features)
Valeurs Maximales | Genou tendu = 0° | Auto Scatter / Normalized Radar Plot
Légendes dynamiques avec comptage (n=X)
=========================================================================================
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- PANDAS CONFIGURATION ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')

# ==============================================================================
# --- ANALYSIS CONFIGURATION ---
# ==============================================================================
NUM_CLUSTERS = 6

# 🟢 CONFIGURATIONS PRINCIPALES 🟢
INCLUDE_HEALTHY_IN_CLUSTERING = True
ALGO_CLUSTERING = 'GMM'  # Options : 'HAC', 'GMM', 'KMEANS'
STRAIGHT_KNEE_IS_ZERO = True  # True = Genou tendu à 0° (Flexion positive)

# 🔵 CHOIX DES VARIABLES (Scatter si 2, Normalized Radar si 3+) 🔵
FEATURES_TO_USE = ['Knee_Flexion_Max', 'Trunk_Lean_Max', 'Tibia_Lean_Max']
# FEATURES_TO_USE = ['Knee_Flexion_Max', 'Trunk_Lean_Max'] # Décommenter pour Scatter Plot

LABELS = {
    'Knee_Flexion_Max': 'Knee Flexion Max (°)' if not STRAIGHT_KNEE_IS_ZERO else 'Knee Flexion Max\n(0°=Straight)',
    'Tibia_Lean_Max': 'Tibia Lean\nMax (°)',
    'Trunk_Lean_Max': 'Trunk Lean\nMax (°)'
}

# --- PATH CONFIGURATION ---
master_db_patient_file = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Master_Database_Patient.pkl"
master_db_healthy_file = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Master_Database_Healthy.pkl"

features_str = "_".join([f.split('_')[0] for f in FEATURES_TO_USE])
output_plot_folder = rf"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Plot_Max_{ALGO_CLUSTERING}_{features_str}"
os.makedirs(output_plot_folder, exist_ok=True)

# ==============================================================================
# 1. LOAD DATA (MASTER DB PATIENTS & HEALTHY)
# ==============================================================================
print("1. Chargement des Master Databases (Pickle)...")


def apply_straight_knee(df):
    if STRAIGHT_KNEE_IS_ZERO and 'Knee_Flexion_Max' in df.columns:
        df['Knee_Flexion_Max'] = 180 - df['Knee_Flexion_Max']
    return df


# --- A. PATIENTS ---
df_master_pat = pd.read_pickle(master_db_patient_file)


def simplifier_diagnostic(diag):
    if pd.isna(diag): return 'Inconnu'
    diag_str = str(diag).strip().lower().replace(' ', '').replace('é', 'e')
    if 'hemi' in diag_str:
        return 'Hemi'
    elif 'di' in diag_str:
        return 'Di'
    else:
        return 'Autre'


df_master_pat['Diagnostic'] = df_master_pat['Diagnostic'].apply(simplifier_diagnostic)
mask = (df_master_pat['Pose estimation'].astype(str).str.strip().str.upper() == 'Y') & \
       (df_master_pat['Caregiver assistance'].astype(str).str.strip() != '2') & \
       (df_master_pat['Hand-to-ground contact'].astype(str).str.strip() != '2') & \
       (df_master_pat['Heel_Rise_Binaire'] == 0)

df_patients_trials = df_master_pat[mask].copy()
df_patients_trials = apply_straight_knee(df_patients_trials)

clinical_cols = [c for c in df_patients_trials.columns if c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_'))]
cols_to_mean_pat = list(set(FEATURES_TO_USE + clinical_cols))

for col in cols_to_mean_pat:
    df_patients_trials[col] = pd.to_numeric(df_patients_trials[col], errors='coerce')

df_patients = df_patients_trials.groupby(['ID_Patient', 'ID_Visite', 'Diagnostic'])[
    cols_to_mean_pat].mean().reset_index()
df_patients = df_patients.dropna(subset=FEATURES_TO_USE)

# --- B. HEALTHY (VICON & YOUTUBE) ---
df_master_health = pd.read_pickle(master_db_healthy_file)
df_master_health = apply_straight_knee(df_master_health)

for col in FEATURES_TO_USE:
    if col in df_master_health.columns:
        df_master_health[col] = pd.to_numeric(df_master_health[col], errors='coerce')

df_healthy = df_master_health.groupby(['ID_Subject', 'Source'])[FEATURES_TO_USE].mean().reset_index()
df_healthy['Diagnostic'] = 'Sain'
df_healthy = df_healthy.dropna(subset=FEATURES_TO_USE)

print(f"   -> Patients conservés : {len(df_patients)}")
print(f"   -> Sains conservés : {len(df_healthy)}")

# ==============================================================================
# 2. CLUSTERING DYNAMIQUE
# ==============================================================================
scaler = StandardScaler()


def get_clustering_model(algo, n_clusters):
    if algo == 'HAC':
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    elif algo == 'GMM':
        return GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, n_init=10)
    elif algo == 'KMEANS':
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    else:
        raise ValueError(f"Algorithme inconnu : {algo}")


model = get_clustering_model(ALGO_CLUSTERING, NUM_CLUSTERS)

if INCLUDE_HEALTHY_IN_CLUSTERING:
    print(f"\n2. Clustering ({ALGO_CLUSTERING} sur PATIENTS + SAINS combinés)...")
    df_combined = pd.concat([df_patients, df_healthy], ignore_index=True)
    X_scaled = scaler.fit_transform(df_combined[FEATURES_TO_USE])

    df_combined['Cluster_ID'] = model.fit_predict(X_scaled)
    df_combined['Profil_Cinematique'] = df_combined['Cluster_ID'].apply(lambda x: f"Profile {x + 1}")

    df_patients = df_combined[df_combined['Diagnostic'] != 'Sain'].copy()
    df_healthy = df_combined[df_combined['Diagnostic'] == 'Sain'].copy()

else:
    print(f"\n2. Clustering ({ALGO_CLUSTERING} sur PATIENTS UNIQUEMENT)...")
    X_patients_scaled = scaler.fit_transform(df_patients[FEATURES_TO_USE])

    df_patients['Cluster_ID'] = model.fit_predict(X_patients_scaled)
    df_patients['Profil_Cinematique'] = df_patients['Cluster_ID'].apply(lambda x: f"Profile {x + 1}")

    df_healthy['Profil_Cinematique'] = df_healthy.apply(
        lambda r: 'Healthy Ref (Vicon)' if r['Source'] == 'Vicon_Healthy' else 'Healthy Ref (YT)', axis=1
    )

df_patients = df_patients.sort_values(by='Profil_Cinematique')

# ==============================================================================
# 3. GEE ANALYSIS ON PATIENTS ONLY
# ==============================================================================
print("3. Running GEE models on Patients (Grouped by ID_Patient)...")
results_gee = []
for var in clinical_cols:
    try:
        valid_df = df_patients.dropna(subset=[var, 'ID_Patient'])
        if valid_df['ID_Patient'].nunique() > 0 and valid_df['Profil_Cinematique'].nunique() > 1:
            gee_model = smf.gee(f"{var} ~ C(Profil_Cinematique)", groups="ID_Patient", data=valid_df,
                                family=sm.families.Gaussian()).fit()
            results_gee.append({'Variable': var, 'p-value_GEE': gee_model.pvalues.filter(like="Profile").min()})
    except:
        pass
df_stats = pd.DataFrame(results_gee)

# ==============================================================================
# 4. VISUALIZATION: GRAPHICS WITH (n=X) LABELS
# ==============================================================================
print("4. Génération des graphiques...")
sns.set_theme(style="white")
colors = sns.color_palette("Set2", NUM_CLUSTERS)
palette_dict = {f"Profile {i + 1}": colors[i] for i in range(NUM_CLUSTERS)}
palette_dict["Healthy Ref (Vicon)"] = "black"
palette_dict["Healthy Ref (YT)"] = "grey"

# Rassemblement des données pour le tracé
df_plot_combined = pd.concat([df_patients, df_healthy], ignore_index=True)

# ---> AJOUT DU COMPTAGE (n=X) DANS LES NOMS DE PROFILS <---
profile_counts = df_plot_combined['Profil_Cinematique'].value_counts().to_dict()

new_palette_dict = {}
for old_name, color in palette_dict.items():
    if old_name in profile_counts:
        new_name = f"{old_name} (n={profile_counts[old_name]})"
        new_palette_dict[new_name] = color
    else:
        new_palette_dict[old_name] = color

palette_dict = new_palette_dict
df_plot_combined['Profil_Cinematique'] = df_plot_combined['Profil_Cinematique'].apply(
    lambda x: f"{x} (n={profile_counts[x]})")
profile_order = sorted(df_plot_combined['Profil_Cinematique'].unique())

# ---> CONDITION GRAPHIQUE <---
if len(FEATURES_TO_USE) == 2:
    print("   -> 2 Variables détectées : Création du Scatter Plot...")
    markers_dict = {'Di': 's', 'Hemi': 'o', 'Autre': 'v', 'Inconnu': 'D', 'Sain': '^'}
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot_combined, x=FEATURES_TO_USE[0], y=FEATURES_TO_USE[1], hue='Profil_Cinematique', style='Diagnostic',
        palette=palette_dict, markers=markers_dict, s=120, alpha=0.85, edgecolor='k', ax=ax_scatter,
        hue_order=profile_order
    )
    ax_scatter.set_title(
        f"2D {ALGO_CLUSTERING} Clustering: {LABELS.get(FEATURES_TO_USE[0], FEATURES_TO_USE[0])} vs {LABELS.get(FEATURES_TO_USE[1], FEATURES_TO_USE[1])}",
        fontsize=14, fontweight='bold')
    ax_scatter.set_xlabel(LABELS.get(FEATURES_TO_USE[0], FEATURES_TO_USE[0]).replace('\n', ' '), fontsize=14)
    ax_scatter.set_ylabel(LABELS.get(FEATURES_TO_USE[1], FEATURES_TO_USE[1]).replace('\n', ' '), fontsize=14)
    ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Groupes & Diagnostics")
    plt.tight_layout()
    fig_scatter.savefig(os.path.join(output_plot_folder, f"ScatterPlot_{ALGO_CLUSTERING}.png"), dpi=300)
    plt.close(fig_scatter)

elif len(FEATURES_TO_USE) >= 3:
    print(f"   -> {len(FEATURES_TO_USE)} Variables détectées : Création du Radar Plot Normalisé...")

    # --- LOGIQUE DE NORMALISATION ---
    df_radar_plot = np.round(df_plot_combined.groupby('Profil_Cinematique')[FEATURES_TO_USE].mean())
    mins_data, maxs_data = df_radar_plot.min(), df_radar_plot.max()

    ticks_plot, mins_plot = {}, pd.Series(index=FEATURES_TO_USE, dtype=float)
    for col in FEATURES_TO_USE:
        range_val = maxs_data[col] - mins_data[col]
        target_step = range_val / 4.0 if range_val > 0 else 1.0
        magnitude = 10 ** np.floor(np.log10(target_step)) if target_step > 0 else 1.0
        norm_target = target_step / magnitude
        step = (
                   1 if norm_target <= 1 else 2 if norm_target <= 2 else 2.5 if norm_target <= 2.5 else 5 if norm_target <= 5 else 10) * magnitude
        ticks = [maxs_data[col] - i * step for i in range(5)][::-1]
        ticks_plot[col] = ticks
        mins_plot[col] = ticks[0]


    def normalize_radar(val, col):
        r_val = maxs_data[col] - mins_plot[col]
        return 1.0 if r_val == 0 else np.clip(0.1 + 0.9 * ((val - mins_plot[col]) / r_val), 0.1, 1.0)


    # --- SETUP PLOT ---
    fig_radar, ax_radar = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(FEATURES_TO_USE), endpoint=False).tolist()
    angles += angles[:1]

    # --- TRACÉ DES PROFILS ---
    for profil in profile_order:
        if profil not in df_radar_plot.index: continue
        values_norm = [normalize_radar(df_radar_plot.loc[profil, c], c) for c in FEATURES_TO_USE]
        values_norm += values_norm[:1]

        is_healthy = "Healthy" in profil or "Vicon" in profil or "YT" in profil
        color = palette_dict.get(profil, 'grey')
        ls = '--' if is_healthy else '-'
        lw = 2.5 if is_healthy else 3.0

        ax_radar.plot(angles, values_norm, color=color, linewidth=lw, linestyle=ls, label=profil)
        if not is_healthy:
            ax_radar.fill(angles, values_norm, color=color, alpha=0.2)

    # --- FORMATTAGE AXES ET TEXTES ---
    ax_radar.set_ylim(0, 1.0)
    ax_radar.set_yticks(np.linspace(0.1, 1.0, 5))
    ax_radar.set_yticklabels([])

    for i, angle in enumerate(angles[:-1]):
        for tr, tn in zip(ticks_plot[FEATURES_TO_USE[i]], np.linspace(0.1, 1.0, 5)):
            ax_radar.text(angle, tn, f"{tr:g}°", ha='center', va='center', fontsize=9, color='dimgrey',
                          bbox=dict(facecolor='white', edgecolor='none', pad=1, alpha=0.8))

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels([LABELS.get(f, f) for f in FEATURES_TO_USE], fontsize=13, fontweight='bold')
    ax_radar.tick_params(axis='x', pad=30)

    plt.title(f"Normalized Kinematic Signature vs Normative Data ({ALGO_CLUSTERING})", size=16, fontweight='bold',
              y=1.1)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    fig_radar.savefig(os.path.join(output_plot_folder, f"RadarPlot_Normalized_{ALGO_CLUSTERING}.png"), dpi=300,
                      bbox_inches='tight')
    plt.close(fig_radar)

# --- Dot plots (s'adapte au nombre de features) ---
sns.set_theme(style="whitegrid")
fig2, axes = plt.subplots(1, len(FEATURES_TO_USE), figsize=(5 * len(FEATURES_TO_USE), 6))
if len(FEATURES_TO_USE) == 1: axes = [axes]

for i, col in enumerate(FEATURES_TO_USE):
    sns.boxplot(x='Profil_Cinematique', y=col, data=df_plot_combined, ax=axes[i], color='white', showfliers=False,
                width=0.5, order=profile_order)
    sns.stripplot(x='Profil_Cinematique', y=col, data=df_plot_combined, ax=axes[i], palette=palette_dict, size=6,
                  alpha=0.7, jitter=True, hue='Profil_Cinematique', legend=False, order=profile_order)
    axes[i].set_title(LABELS.get(col, col).replace('\n', ' '), fontsize=14, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Degrees (°)', fontsize=12)
    axes[i].tick_params(axis='x', rotation=45)

plt.suptitle(f"Kinematic Distribution ({ALGO_CLUSTERING})", size=16, fontweight='bold', y=1.05)
plt.tight_layout()
fig2.savefig(os.path.join(output_plot_folder, f"Dot_Plots_{ALGO_CLUSTERING}.png"), dpi=300, bbox_inches='tight')
plt.close(fig2)

# ==============================================================================
# 5. SUMMARY TABLES & EXPORT
# ==============================================================================
print("\n" + "=" * 90)
print("DISTRIBUTION DES DIAGNOSTICS PAR PROFIL")
print("=" * 90)

diag_counts = pd.crosstab(df_patients['Profil_Cinematique'], df_patients['Diagnostic'], margins=True,
                          margins_name="Total")
print(diag_counts)

summary_mean = df_patients.groupby('Profil_Cinematique')[FEATURES_TO_USE + clinical_cols].mean().T
summary_mean.columns = [f"{c} (Mean)" for c in summary_mean.columns]

summary_n = df_patients.groupby('Profil_Cinematique')[FEATURES_TO_USE + clinical_cols].count().T
summary_n.columns = [f"{c} (n)" for c in summary_n.columns]

subgroup_mean = df_patients.groupby(['Profil_Cinematique', 'Diagnostic'])[FEATURES_TO_USE + clinical_cols].mean().T
subgroup_mean.columns = [f"{prof} - {diag} (Mean)" for prof, diag in subgroup_mean.columns]

summary = pd.concat([summary_mean, summary_n, subgroup_mean], axis=1)

ordered_cols = []
for profil in sorted(df_patients['Profil_Cinematique'].unique()):
    ordered_cols.append(f"{profil} (Mean)")
    for diag in ['Hemi', 'Di', 'Autre', 'Inconnu']:
        col_name = f"{profil} - {diag} (Mean)"
        if col_name in summary.columns: ordered_cols.append(col_name)
    ordered_cols.append(f"{profil} (n)")

summary = summary[ordered_cols]

if not df_stats.empty:
    tableau_final = pd.merge(summary, df_stats.set_index('Variable'), left_index=True, right_index=True, how='left')
else:
    tableau_final = summary
    tableau_final['p-value_GEE'], tableau_final['p-value_Hemi'], tableau_final['p-value_Di'] = np.nan, np.nan, np.nan

tableau_final['Sig_Global'] = tableau_final['p-value_GEE'].apply(lambda x: "*" if pd.notnull(x) and x < 0.05 else "")
for pcol in ['p-value_GEE', 'p-value_Hemi', 'p-value_Di']:
    if pcol in tableau_final.columns:
        tableau_final[pcol] = tableau_final[pcol].round(4)

tableau_final = tableau_final.sort_values(by='p-value_GEE', ascending=True, na_position='last')
excel_export_path = os.path.join(output_plot_folder, f"Tableau_Resumes_{ALGO_CLUSTERING}.xlsx")
tableau_final.to_excel(excel_export_path)

print(f"\n✅ Le tableau complet a été sauvegardé en Excel ici : {excel_export_path}")

# ==============================================================================
# 6. TRANSFERT DU MODÈLE OU RÉPARTITION DES SAINS
# ==============================================================================
print("\n" + "=" * 90)
if not INCLUDE_HEALTHY_IN_CLUSTERING:
    print(f"TRANSFERT DU MODÈLE {ALGO_CLUSTERING} SUR LES SUJETS SAINS")
    print("=" * 90)

    for source in df_healthy['Source'].unique():
        df_source = df_healthy[df_healthy['Source'] == source].copy()
        X_source_scaled = scaler.transform(df_source[FEATURES_TO_USE])

        if ALGO_CLUSTERING == 'HAC':
            knn_projector = KNeighborsClassifier(n_neighbors=1)
            knn_projector.fit(X_patients_scaled, df_patients['Cluster_ID'])
            predicted_clusters = knn_projector.predict(X_source_scaled)
        else:
            predicted_clusters = model.predict(X_source_scaled)

        counts = pd.Series([f"Profile {c + 1}" for c in predicted_clusters]).value_counts()
        print(f"\n--- Predictions pour {source} ---")
        print(counts.to_string())
else:
    print(f"RÉPARTITION DES SUJETS SAINS DANS LES CLUSTERS (Entraînés ensemble avec {ALGO_CLUSTERING})")
    print("=" * 90)

    for source in df_healthy['Source'].unique():
        counts = df_healthy[df_healthy['Source'] == source]['Profil_Cinematique'].value_counts()
        print(f"\n--- Répartition pour {source} ---")
        print(counts.to_string())