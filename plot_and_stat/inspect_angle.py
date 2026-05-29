import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from fonction import butter_lowpass_filter, calculate_angle_0_is_straight, calculate_lean_0_is_straight

warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
# 🔴 METTEZ ICI LES ID DES PATIENTS ISOLÉS PAR L'ALGORITHME
PATIENTS_A_INSPECTER = ['3071']

DEFAULT_FPS = 50
CUTOFF_FREQ = 3
POINTS_NORMALISATION = 101

main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"
frontal_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_Frontal_all.pkl"
master_db_healthy_file = fr"{main_path}\Data\Master_Database_Healthy_all.pkl"
output_folder = fr"{main_path}\Results_v3\Inspection_Outliers"
os.makedirs(output_folder, exist_ok=True)


# ==============================================================================
# --- 2. FONCTIONS MATHÉMATIQUES ---
# ==============================================================================
def process_kinematics_curve(curve, fps):
    """Extrait la courbe complète de 0 à 100% du mouvement."""
    c_array = np.array(curve)
    valid = ~np.isnan(c_array)
    if not valid.any(): return np.full(POINTS_NORMALISATION, np.nan)
    c_interp = np.interp(np.arange(len(c_array)), np.where(valid)[0], c_array[valid])
    c_filt = butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)
    c_norm = np.interp(np.linspace(0, 100, POINTS_NORMALISATION), np.linspace(0, 100, len(c_filt)), c_filt)
    return c_norm


def calculate_valgus_varus_frontal(hip, knee, ankle, side):
    v1 = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    angle_deg = np.degrees(np.arctan2(cross_product, dot_product))
    if side == 'gauche': angle_deg = -angle_deg
    return angle_deg


# ==============================================================================
# --- 3. CHARGEMENT DES RÉFÉRENCES SAINES (Corridor visuel) ---
# ==============================================================================
print("Chargement des données saines pour référence...")
df_health = pd.read_pickle(master_db_healthy_file)
df_health = df_health[df_health['Source'] == 'YouTube_Healthy']

healthy_curves = {'Knee': [], 'Trunk': [], 'Tibia': [], 'Knee_Frontal': []}

for _, row in df_health.iterrows():
    try:
        kpts = row['Raw_Keypoints']
        fps = row['Video_FPS'] if pd.notna(row['Video_FPS']) and row['Video_FPS'] > 0 else DEFAULT_FPS
        n_frames = kpts.shape[0]

        k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]
        tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
        tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

        healthy_curves['Knee'].append(process_kinematics_curve(k_f, fps))
        healthy_curves['Trunk'].append(process_kinematics_curve(tr_f, fps))
        healthy_curves['Tibia'].append(process_kinematics_curve(tib_f, fps))
        # Les sains n'ont pas de vue frontale, on remplit de NaN
        healthy_curves['Knee_Frontal'].append(np.full(POINTS_NORMALISATION, np.nan))
    except:
        pass

# Calcul de la moyenne et l'écart-type sains
ref_stats = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    for joint, curves in healthy_curves.items():
        mat = np.vstack([c for c in curves])
        ref_stats[joint] = {'mean': np.nanmean(mat, axis=0), 'std': np.nanstd(mat, axis=0)}

# ==============================================================================
# --- 4. CHARGEMENT ET TRACÉ DES PATIENTS CIBLÉS ---
# ==============================================================================
print(f"Recherche et traitement des patients : {PATIENTS_A_INSPECTER}")

# Chargement de la base patient + fusion frontale
df_pat = pd.read_pickle(master_db_patient_file)
if os.path.exists(frontal_db_patient_file):
    df_fro = pd.read_pickle(frontal_db_patient_file)[['File_Sagittal', 'Video_FPS', 'Raw_Keypoints']]
    df_fro.columns = ['File_Sagittal', 'FPS_frontal', 'Keypoints_frontal']
    df_pat = pd.merge(df_pat, df_fro, on='File_Sagittal', how='left')

df_cible = df_pat[df_pat['ID_Patient'].isin(PATIENTS_A_INSPECTER)].copy()

if df_cible.empty:
    print("❌ Aucun patient trouvé. Vérifiez l'orthographe des ID_Patient.")
    exit()

time_vec = np.linspace(0, 100, POINTS_NORMALISATION)

# On génère un graphique par patient
for patient_id in PATIENTS_A_INSPECTER:
    df_indiv = df_cible[df_cible['ID_Patient'] == patient_id]
    if df_indiv.empty: continue

    print(f"Génération du graphique pour {patient_id} ({len(df_indiv)} essais trouvés)...")

    # Grille 2x2 pour inclure les 4 courbes proprement
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()  # Permet de boucler dessus facilement
    fig.suptitle(f"Inspection Cinématique : {patient_id}", fontsize=16, fontweight='bold', y=0.98)

    joints = ['Knee', 'Trunk', 'Tibia', 'Knee_Frontal']
    titles = ['Flexion Genou (0°=Droit)', 'Inclinaison Tronc', 'Inclinaison Tibia', 'Genou Frontal (+Valgus / -Varus)']

    for ax, joint, title in zip(axes, joints, titles):
        # 1. Dessiner le corridor sain (Gris) si la donnée existe
        if not np.isnan(ref_stats[joint]['mean']).all():
            ax.plot(time_vec, ref_stats[joint]['mean'], color='black', linestyle='--', label='Sain (Moy)')
            ax.fill_between(time_vec,
                            ref_stats[joint]['mean'] - ref_stats[joint]['std'],
                            ref_stats[joint]['mean'] + ref_stats[joint]['std'],
                            color='grey', alpha=0.2, label='Sain (±1 SD)')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('% du Mouvement')
        ax.set_ylabel('Angle (°)')
        ax.grid(True, linestyle=':', alpha=0.7)
        if joint == 'Knee_Frontal':
            ax.axhline(0, color='red', linestyle='--', alpha=0.3)  # Ligne de référence neutre

    # 2. Dessiner chaque essai du patient ciblé
    colors = sns.color_palette("husl", len(df_indiv))

    for idx, (i, row) in enumerate(df_indiv.iterrows()):
        try:
            # Courbes sagittales
            kpts = row['Raw_Keypoints']
            fps = row['Video_FPS'] if pd.notna(row['Video_FPS']) and row['Video_FPS'] > 0 else DEFAULT_FPS
            n_frames = kpts.shape[0]

            k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]
            tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
            tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

            # Courbe Frontale
            kf_curve = np.full(POINTS_NORMALISATION, np.nan)
            if 'Keypoints_frontal' in row and isinstance(row['Keypoints_frontal'], np.ndarray):
                kpts_fro = row['Keypoints_frontal']
                fps_fro = row['FPS_frontal'] if pd.notna(row['FPS_frontal']) and row['FPS_frontal'] > 0 else DEFAULT_FPS
                n_frames_fro = kpts_fro.shape[0]
                kf_all = [calculate_valgus_varus_frontal(kpts_fro[f, 12], kpts_fro[f, 14], kpts_fro[f, 16], 'droite')
                          for f in range(n_frames_fro)]
                kf_curve = process_kinematics_curve(kf_all, fps_fro)

            # Assemblage des 4 courbes
            curves = [
                process_kinematics_curve(k_f, fps),
                process_kinematics_curve(tr_f, fps),
                process_kinematics_curve(tib_f, fps),
                kf_curve
            ]

            essai_nom = row.get('ID_Visite', f'Essai_{idx + 1}')

            for ax, curve in zip(axes, curves):
                # On ne trace pas si la courbe est entièrement vide (ex: tracking frontal raté)
                if not np.isnan(curve).all():
                    ax.plot(time_vec, curve, color=colors[idx], alpha=0.8, linewidth=2, label=f'{essai_nom}')
        except:
            pass

    # Ne mettre la légende que sur le premier graphique pour éviter de surcharger
    axes[0].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"Inspection_{patient_id}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ Sauvegardé : {plot_path}")
    plt.close()

print("\nTerminé ! Allez voir le dossier d'inspection.")