import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from fonction import butter_lowpass_filter, calculate_angle_0_is_straight, calculate_lean_0_is_straight

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. CONFIGURATION DES CHEMINS ET PARAMÈTRES ---
# ==============================================================================
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"
frontal_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_Frontal_all.pkl"

DEFAULT_FPS = 50
CUTOFF_FREQ = 3


# ==============================================================================
# --- 2. FONCTIONS DE CALCUL (Identiques au pipeline) ---
# ==============================================================================
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


# ==============================================================================
# --- 3. CHARGEMENT ET RECALCUL COMPLET DES DONNÉES ---
# ==============================================================================
print("1. Chargement des bases de données brutes...")
df_master_pat = pd.read_pickle(master_db_patient_file)

if os.path.exists(frontal_db_patient_file):
    df_fro = pd.read_pickle(frontal_db_patient_file)[['File_Sagittal', 'Video_FPS', 'Raw_Keypoints']]
    df_fro.columns = ['File_Sagittal', 'FPS_frontal', 'Keypoints_frontal']
    df_master_pat = pd.merge(df_master_pat, df_fro, on='File_Sagittal', how='left')

# Filtre de base (identique au clustering)
mask_pat = (df_master_pat['Pose estimation'].astype(str).str.strip().str.upper() == 'Y') & \
           (df_master_pat['Heel_Rise_Binaire'] == 0) & \
           (df_master_pat['CoteDiagnostic'] != 'Gauche') & (df_master_pat['CoteDiagnostic'] != 'Gauche Droit')
df_base_pat = df_master_pat[mask_pat].copy()

print(f"2. Recalcul des courbes et extraction des variables synchronisées pour {len(df_base_pat)} essais...")
processed_data = []

for idx, row in df_base_pat.iterrows():
    try:
        kpts = row['Raw_Keypoints']
        fps = row['Video_FPS'] if pd.notna(row['Video_FPS']) and row['Video_FPS'] > 0 else DEFAULT_FPS
        n_frames = kpts.shape[0]

        # Calcul des courbes brutes
        k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]
        tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
        tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

        # Filtrage lissé
        k_filt = get_filtered_curve(k_f, fps)
        tr_filt = get_filtered_curve(tr_f, fps)
        tib_filt = get_filtered_curve(tib_f, fps)

        k_max, tr_at_max, tib_at_max, kf_max, kf_delta = np.nan, np.nan, np.nan, np.nan, np.nan

        if k_filt is not None:
            # 1. Repérage du pic du genou
            idx_sagittal = np.argmax(k_filt)
            k_max = k_filt[idx_sagittal]

            # 2. Synchronisation du Tronc et du Tibia
            if tr_filt is not None: tr_at_max = tr_filt[idx_sagittal]
            if tib_filt is not None: tib_at_max = tib_filt[idx_sagittal]

            # 3. Synchronisation et calcul du plan frontal
            if 'Keypoints_frontal' in row and isinstance(row['Keypoints_frontal'], np.ndarray):
                kpts_fro = row['Keypoints_frontal']
                fps_fro = row['FPS_frontal'] if pd.notna(row['FPS_frontal']) and row['FPS_frontal'] > 0 else DEFAULT_FPS
                n_frames_fro = kpts_fro.shape[0]

                time_seconds = idx_sagittal / fps
                idx_frontal = int(round(time_seconds * fps_fro))
                idx_frontal = min(max(idx_frontal, 0), n_frames_fro - 1)

                kf_start = calculate_valgus_varus_frontal(kpts_fro[0, 12], kpts_fro[0, 14], kpts_fro[0, 16], 'droite')
                kf_max = calculate_valgus_varus_frontal(kpts_fro[idx_frontal, 12], kpts_fro[idx_frontal, 14],
                                                        kpts_fro[idx_frontal, 16], 'droite')
                kf_delta = kf_max - kf_start

        # On ne stocke que les données cinématiques nécessaires pour la Heatmap
        processed_data.append({
            'Knee_Flexion_Max': k_max,
            'Trunk_Lean_At_Max': tr_at_max,
            'Tibia_Lean_At_Max': tib_at_max,
            'Knee_Frontal_Max': kf_max,
            'Knee_Frontal_Delta': kf_delta
        })
    except Exception as e:
        pass

df_final = pd.DataFrame(processed_data)

# ==============================================================================
# --- 4. CALCUL ET AFFICHAGE DE LA HEATMAP ---
# ==============================================================================
print("3. Génération de la Matrice de Corrélation...")

# On supprime les lignes où il manque des données (ex: patients Vicon sans vue frontale)
# pour que la corrélation soit calculée sur la même population
df_corr = df_final.dropna()

if df_corr.empty:
    print("❌ Erreur : Il n'y a pas assez de données communes pour calculer la corrélation.")
else:
    # Renommage propre pour l'affichage graphique
    df_corr.columns = [
        'Genou Flexion (Max)',
        'Tronc (Sync)',
        'Tibia (Sync)',
        'Valgus Absolu (Sync)',
        'Valgus Delta (Dynamique)'
    ]

    matrice_correlation = df_corr.corr()

    # Création de la figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_correlation,
                annot=True,
                cmap='vlag',  # Palette esthétique de bleu (négatif) à rouge (positif)
                vmin=-1, vmax=1,
                fmt=".2f",
                linewidths=1,
                annot_kws={"size": 12, "weight": "bold"})

    plt.title("Matrice de Corrélation des Variables Cinématiques", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    output_path = os.path.join(main_path, "Results_v4", "Correlation_Heatmap_Synchronisee.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)

    print(f"✅ Terminé ! Matrice générée avec succès : {output_path}")
    plt.show()