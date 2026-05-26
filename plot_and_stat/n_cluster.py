import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings
from fonction import *

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
AVERAGE_TRIALS_PER_VISIT = True
DEFAULT_FPS = 50
CUTOFF_FREQ = 3
POINTS_NORMALISATION = 101
MAX_CLUSTERS_TO_TEST = 20  # Nombre maximum de clusters à évaluer

# Variables de base
FEATURES_BASE = ['Knee', 'Trunk', 'Tibia']
FEATURES_MAX = [f"{f}_Flexion_Max" if f == 'Knee' else f"{f}_Lean_Max" for f in FEATURES_BASE]
FEATURES_CURVE = [f"{f}_Curve" for f in FEATURES_BASE]

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"

output_plot_folder = fr"{main_path}\Results\Optimal_Clusters_Evaluation"
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


def process_kinematics(curve, fps):
    c_array = np.array(curve)
    valid = ~np.isnan(c_array)
    if not valid.any(): return np.zeros(POINTS_NORMALISATION), np.nan
    c_interp = np.interp(np.arange(len(c_array)), np.where(valid)[0], c_array[valid])
    c_filt = butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)
    c_norm = np.interp(np.linspace(0, 100, POINTS_NORMALISATION), np.linspace(0, 100, len(c_filt)), c_filt)
    return c_norm, np.max(c_filt)


# ==============================================================================
# --- 3. CHARGEMENT ET PRÉPARATION DES DONNÉES PATIENTS ---
# ==============================================================================
print("1. Chargement et extraction unifiée depuis les matrices 3D...")

df_master_pat = pd.read_pickle(master_db_patient_file)
mask_pat = (df_master_pat['Pose estimation'].astype(str).str.strip().str.upper() == 'Y') & \
           (df_master_pat['Caregiver assistance'].astype(str).str.strip() != '2') & \
           (df_master_pat['Hand-to-ground contact'].astype(str).str.strip() != '2') & \
           (df_master_pat['Heel_Rise_Binaire'] == 0) & \
           (df_master_pat['CoteDiagnostic'] != 'Gauche') & (df_master_pat['CoteDiagnostic'] != 'Gauche Droit')

df_base_pat = df_master_pat[mask_pat].copy()
df_base_pat['Diagnostic'] = df_base_pat['Diagnostic'].apply(simplifier_diagnostic)

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

        processed_pat.append({
            'ID_Patient': row['ID_Patient'], 'ID_Visite': row['ID_Visite'], 'Diagnostic': row['Diagnostic'],
            'Knee_Curve': k_curve, 'Trunk_Curve': tr_curve, 'Tibia_Curve': tib_curve,
            'Knee_Flexion_Max': k_max, 'Trunk_Lean_Max': tr_max, 'Tibia_Lean_Max': tib_max
        })
    except:
        pass

df_pat_processed = pd.DataFrame(processed_pat)

if AVERAGE_TRIALS_PER_VISIT:
    agg_dict = {f: lambda x: np.mean(np.vstack(x), axis=0) for f in FEATURES_CURVE}
    agg_dict.update({f: 'mean' for f in FEATURES_MAX})
    df_pat_final = df_pat_processed.groupby(['ID_Patient', 'ID_Visite', 'Diagnostic']).agg(agg_dict).reset_index()
else:
    df_pat_final = df_pat_processed.copy()


# ==============================================================================
# --- 4. FONCTION D'ÉVALUATION DES CLUSTERS ---
# ==============================================================================
def evaluate_clusters(X, title_prefix, max_k=MAX_CLUSTERS_TO_TEST):
    print(f"\n--- Évaluation pour : {title_prefix} ---")

    inertias = []
    silhouettes = []
    davies_bouldins = []

    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        davies_bouldins.append(davies_bouldin_score(X, labels))

    # Tracé des graphiques
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Métriques d\'évaluation pour {title_prefix}', fontsize=16, fontweight='bold')

    # 1. Elbow Method (Inertia)
    axes[0].plot(K_range, inertias, marker='o', linestyle='-', color='b')
    axes[0].set_title('Méthode du Coude (Inertie)\n[Chercher la cassure]', fontsize=12)
    axes[0].set_xlabel('Nombre de clusters (k)')
    axes[0].set_ylabel('Inertie')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xticks(K_range)

    # 2. Silhouette Score
    axes[1].plot(K_range, silhouettes, marker='s', linestyle='-', color='g')
    axes[1].set_title('Score de Silhouette\n[Chercher la valeur maximale]', fontsize=12)
    axes[1].set_xlabel('Nombre de clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xticks(K_range)

    # 3. Davies-Bouldin Index
    axes[2].plot(K_range, davies_bouldins, marker='^', linestyle='-', color='r')
    axes[2].set_title('Indice de Davies-Bouldin\n[Chercher la valeur minimale]', fontsize=12)
    axes[2].set_xlabel('Nombre de clusters (k)')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_xticks(K_range)

    plt.tight_layout()
    output_path = os.path.join(output_plot_folder, f"Optimal_K_{title_prefix.replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Graphique sauvegardé : {output_path}")

    # Identifier les meilleurs K suggérés (simplifié)
    best_k_silhouette = K_range[np.argmax(silhouettes)]
    best_k_db = K_range[np.argmin(davies_bouldins)]

    print(f"-> Meilleur k selon Silhouette : {best_k_silhouette}")
    print(f"-> Meilleur k selon Davies-Bouldin : {best_k_db}")
    print("-> Note : Vérifiez le graphique d'Inertie pour confirmer le 'coude'.")


# ==============================================================================
# --- 5. LANCEMENT DES ÉVALUATIONS ---
# ==============================================================================
# Préparation des données MAX
df_max_pat = df_pat_final.dropna(subset=FEATURES_MAX).copy()
X_max_scaled = StandardScaler().fit_transform(df_max_pat[FEATURES_MAX])
evaluate_clusters(X_max_scaled, "Valeurs Maximales (MAX)")

# Préparation des données CURVE
df_curve_pat = df_pat_final.copy()
# Aplatissement des courbes pour l'ACP
X_curve_raw = np.array([list(itertools.chain(*[row[f] for f in FEATURES_CURVE])) for _, row in df_curve_pat.iterrows()])
X_curve_scaled = StandardScaler().fit_transform(X_curve_raw)
X_curve_pca = PCA(n_components=0.95, random_state=42).fit_transform(X_curve_scaled)
evaluate_clusters(X_curve_pca, "Series Temporelles (CURVE - PCA)")

print("\n✅ Analyse terminée. Consultez les graphiques générés pour choisir le 'NUM_CLUSTERS' idéal.")