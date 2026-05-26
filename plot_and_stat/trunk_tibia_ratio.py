import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from fonction import *  # Contient vos fonctions de filtrage/calcul d'angles

warnings.filterwarnings('ignore')

# --- PANDAS CONFIGURATION ---
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ==============================================================================
# --- 1. CONFIGURATION ---
# ==============================================================================
AVERAGE_TRIALS_PER_VISIT = False  # Mettre sur True pour moyenner par patient/sujet
type_of_analysis = "average" if AVERAGE_TRIALS_PER_VISIT else "all_visit"

DEFAULT_FPS = 50
CUTOFF_FREQ = 3

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
master_db_patient_file = fr"{main_path}\Data\Master_Database_Patient_all.pkl"
master_db_healthy_file = fr"{main_path}\Data\Master_Database_Healthy.pkl"

output_plot_folder = fr"{main_path}\Results\Scatter_Ratio_RightOnly_{type_of_analysis}"
os.makedirs(output_plot_folder, exist_ok=True)


# ==============================================================================
# --- 2. FONCTIONS DE TRAITEMENT ---
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
    return 'Right' if has_right else ('Left' if has_left else 'Unknown')

def clean_and_filter(curve, fps):
    c_array = np.array(curve)
    valid = ~np.isnan(c_array)
    if not valid.any(): return None
    c_interp = np.interp(np.arange(len(c_array)), np.where(valid)[0], c_array[valid])
    return butter_lowpass_filter(c_interp, CUTOFF_FREQ, fps)


# ==============================================================================
# --- 3. CHARGEMENT ET EXTRACTION DU RATIO ---
# ==============================================================================
print(f"1. Extraction synchronisée du ratio Tronc-Tibia à la flexion max du genou...")

def process_dataframe(df_base, is_healthy=False):
    processed_list = []
    for idx, row in df_base.iterrows():
        try:
            kpts = row['Raw_Keypoints']
            fps = row['Video_FPS'] if pd.notna(row['Video_FPS']) and row['Video_FPS'] > 0 else DEFAULT_FPS
            n_frames = kpts.shape[0]

            # Calcul des angles bruts
            k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]
            tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
            tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

            # Filtrage
            k_filt = clean_and_filter(k_f, fps)
            tr_filt = clean_and_filter(tr_f, fps)
            tib_filt = clean_and_filter(tib_f, fps)

            # EXTRACTION SYNCHRONISÉE
            if k_filt is not None and tr_filt is not None and tib_filt is not None:
                idx_max_knee = np.argmax(k_filt)
                k_max = k_filt[idx_max_knee]
                diff_trunk_tibia = tr_filt[idx_max_knee] - tib_filt[idx_max_knee]
            else:
                k_max = diff_trunk_tibia = np.nan

            if is_healthy:
                entry = {
                    'ID_Subject': row.get('ID_Subject', f'Subj_{idx}'), 'Diagnostic': 'Sain', 'Scores_GMFCS': 'Sain',
                    'Knee_Flexion_Max': k_max, 'Ratio_Trunk_Tibia': diff_trunk_tibia
                }
            else:
                raw_gmfcs = row.get('Scores_GMFCS', np.nan)
                try:
                    val = float(raw_gmfcs)
                    gmfcs_clean = 'Inconnu' if np.isnan(val) else f"GMFCS_{int(val)}"
                except:
                    gmfcs_clean = str(raw_gmfcs).strip()
                    if gmfcs_clean == 'nan': gmfcs_clean = 'Inconnu'

                entry = {
                    'ID_Patient': row['ID_Patient'], 'ID_Visite': row['ID_Visite'],
                    'Diagnostic': row['Diag_Lateralite'], 'Scores_GMFCS': gmfcs_clean,
                    'Knee_Flexion_Max': k_max, 'Ratio_Trunk_Tibia': diff_trunk_tibia
                }
            processed_list.append(entry)
        except:
            pass
    return pd.DataFrame(processed_list)

# --- A. PATIENTS ---
df_master_pat = pd.read_pickle(master_db_patient_file)
mask_pat = (df_master_pat['Pose estimation'].astype(str).str.strip().str.upper() == 'Y') & \
           (df_master_pat['Caregiver assistance'].astype(str).str.strip() != '2') & \
           (df_master_pat['Hand-to-ground contact'].astype(str).str.strip() != '2') & \
           (df_master_pat['Heel_Rise_Binaire'] == 0)

df_base_pat = df_master_pat[mask_pat].copy()
df_base_pat['Diag_Simple'] = df_base_pat['Diagnostic'].apply(simplifier_diagnostic)
df_base_pat['Clean_Side'] = df_base_pat.get('CoteDiagnostic', '').apply(clean_side)
df_base_pat['Diag_Lateralite'] = df_base_pat['Diag_Simple'] + " (" + df_base_pat['Clean_Side'] + ")"
df_base_pat = df_base_pat[df_base_pat['Diag_Lateralite'].isin(['Hemi (Right)', 'Di (Right > Left)'])].copy()

df_pat_processed = process_dataframe(df_base_pat, is_healthy=False)

# --- B. SAINS ---
df_master_health = pd.read_pickle(master_db_healthy_file)
df_master_health = df_master_health[df_master_health['Source'] == 'YouTube_Healthy'].copy()
df_healthy_processed = process_dataframe(df_master_health, is_healthy=True)


# ==============================================================================
# --- 4. AGGRÉGATION / MOYENNAGE ---
# ==============================================================================
if AVERAGE_TRIALS_PER_VISIT:
    df_pat_final = df_pat_processed.groupby(['ID_Patient', 'ID_Visite', 'Diagnostic', 'Scores_GMFCS'])[['Knee_Flexion_Max', 'Ratio_Trunk_Tibia']].mean().reset_index()
    df_healthy_final = df_healthy_processed.groupby(['ID_Subject', 'Diagnostic', 'Scores_GMFCS'])[['Knee_Flexion_Max', 'Ratio_Trunk_Tibia']].mean().reset_index()
else:
    df_pat_final = df_pat_processed.copy()
    df_healthy_final = df_healthy_processed.copy()

df_combined = pd.concat([df_pat_final, df_healthy_final], ignore_index=True)


# ==============================================================================
# --- 5. VISUALISATIONS (SCATTER PLOTS) ---
# ==============================================================================
print("\n2. Génération des graphiques...")

sns.set_theme(style="whitegrid")

# --- GRAPHIQUE 1 : PAR DIAGNOSTIC ---
df_diag = df_combined[df_combined['Diagnostic'].isin(['Di (Right > Left)', 'Hemi (Right)', 'Sain'])].copy()
colors_diag = {'Di (Right > Left)': '#1F77B4', 'Hemi (Right)': '#FF7F0E', 'Sain': 'grey'}

plt.figure(figsize=(10, 8))
ax1 = sns.scatterplot(
    data=df_diag,
    x='Ratio_Trunk_Tibia',
    y='Knee_Flexion_Max',
    hue='Diagnostic',
    palette=colors_diag,
    s=120,
    alpha=0.7,
    edgecolor="black"
)

plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=0)
plt.title("Stratégie Posturale à la Flexion Maximale du Genou (par Diagnostic)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Ratio Tronc - Tibia (°)\n< 0 : Tibia plus penché  |  > 0 : Tronc plus penché", fontsize=12, fontweight='bold')
plt.ylabel("Flexion Maximale du Genou (°)", fontsize=12, fontweight='bold')
plt.legend(title='Groupe Clinique', title_fontsize='13', fontsize='11', loc='best', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "Scatter_Ratio_vs_KneeMax_Diagnostic.png"), dpi=300)
plt.close()


# --- GRAPHIQUE 2 : PAR GMFCS ---
df_gmfcs = df_combined[df_combined['Scores_GMFCS'].isin(['GMFCS_1', 'GMFCS_2', 'GMFCS_3', 'GMFCS_4', 'Sain'])].copy()

gmfcs_order = ['GMFCS_1', 'GMFCS_2', 'GMFCS_3', 'GMFCS_4', 'Sain']
colors_gmfcs = sns.color_palette("YlOrRd", 4)
colors_gmfcs_dict = {gmfcs_order[i]: colors_gmfcs[i] for i in range(4)}
colors_gmfcs_dict['Sain'] = 'grey'

plt.figure(figsize=(10, 8))
ax2 = sns.scatterplot(
    data=df_gmfcs,
    x='Ratio_Trunk_Tibia',
    y='Knee_Flexion_Max',
    hue='Scores_GMFCS',
    hue_order=gmfcs_order,
    palette=colors_gmfcs_dict,
    s=120,
    alpha=0.7,
    edgecolor="black"
)

plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, zorder=0)
plt.title("Stratégie Posturale à la Flexion Maximale du Genou (par GMFCS)", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Ratio Tronc - Tibia (°)\n< 0 : Tibia plus penché  |  > 0 : Tronc plus penché", fontsize=12, fontweight='bold')
plt.ylabel("Flexion Maximale du Genou (°)", fontsize=12, fontweight='bold')
plt.legend(title='Niveau GMFCS', title_fontsize='13', fontsize='11', loc='best', frameon=True, shadow=True)
plt.tight_layout()
plt.savefig(os.path.join(output_plot_folder, "Scatter_Ratio_vs_KneeMax_GMFCS.png"), dpi=300)
plt.close()

print(f"✅ Nuages de points générés avec succès. Dossier : {output_plot_folder}")