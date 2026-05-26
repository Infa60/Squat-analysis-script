"""
=========================================================================================
GENERATE MASTER DATABASE HEALTHY (PICKLE EXPORT) - VITPOSE ONLY
=========================================================================================
Objectif: Créer un fichier unique contenant les essais des sujets sains (YouTube / ViTPose).
Structure: 1 ligne = 1 essai vidéo.
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

yt_healthy_folder = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Processed\Control\Results"

# === Export ===
output_file_healthy_pkl = fr"{main_path}\Data\Master_Database_Healthy_all.pkl"

# --- CONFIGURATION TECHNIQUE ---
modality = "ViTPose_Huge"
DEFAULT_FPS, CUTOFF_FREQ = 50, 3

healthy_data = []

# ==============================================================================
# 1. TRAITEMENT DES SUJETS SAINS (ViTPose 2D)
# ==============================================================================
print("1. Extraction des données des sujets sains (ViTPose)...")

if os.path.exists(yt_healthy_folder):
    for root, dirs, files in os.walk(yt_healthy_folder):
        for file in files:
            if file.endswith("_Results_Filtered.mat") or file.endswith("_Results_Corrected.mat"):
                mat_file = os.path.join(root, file)

                video_file_avi = os.path.join(root, "ViTPose_Huge.avi")
                video_file_mp4 = os.path.join(root, "ViTPose_Huge.mp4")
                # Vérifie d'abord l'existence de l'AVI, sinon se rabat sur le MP4
                video_file = video_file_avi if os.path.exists(video_file_avi) else video_file_mp4

                try:
                    nom_brut = file.replace("_Results_Filtered.mat", "").replace("_Results_Corrected.mat", "")
                    parts = nom_brut.rsplit('_', 1)
                    base_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else nom_brut

                    # --- Extraction des FPS via OpenCV ---
                    video_fps = np.nan
                    current_fps = DEFAULT_FPS
                    if os.path.exists(video_file):
                        cap = cv2.VideoCapture(video_file)
                        if cap.isOpened():
                            extracted_fps = cap.get(cv2.CAP_PROP_FPS)
                            if extracted_fps > 0:
                                video_fps = extracted_fps
                                current_fps = extracted_fps
                        cap.release()

                    # --- Chargement et calculs ---
                    data = sio.loadmat(mat_file)
                    kpts = data[modality][0, 0]['Keypoints'][:, 0, :, :]
                    n_frames = kpts.shape[0]

                    k_f = [calculate_angle_0_is_straight(kpts[f, 12], kpts[f, 14], kpts[f, 16]) for f in
                           range(n_frames)]
                    tr_f = [calculate_lean_0_is_straight(kpts[f, 6], kpts[f, 12]) for f in range(n_frames)]
                    tib_f = [calculate_lean_0_is_straight(kpts[f, 14], kpts[f, 16]) for f in range(n_frames)]

                    # Filtrage passe-bas
                    k_f_filt = butter_lowpass_filter(np.interp(np.arange(len(k_f)), np.where(~np.isnan(k_f))[0], np.array(k_f)[~np.isnan(k_f)]), CUTOFF_FREQ, current_fps)
                    tr_f_filt = butter_lowpass_filter(np.interp(np.arange(len(tr_f)), np.where(~np.isnan(tr_f))[0], np.array(tr_f)[~np.isnan(tr_f)]), CUTOFF_FREQ, current_fps)
                    tib_f_filt = butter_lowpass_filter(np.interp(np.arange(len(tib_f)), np.where(~np.isnan(tib_f))[0], np.array(tib_f)[~np.isnan(tib_f)]), CUTOFF_FREQ, current_fps)

                    # TRANSFORMATION STRICTE : 0° = Posture Droite / Tendue
                    k_f_0 = k_f_filt
                    tr_f_0 = tr_f_filt
                    tib_f_0 = tib_f_filt

                    # Identification de la flexion maximale du genou (pic du mouvement)
                    idx = np.argmax(k_f_0)

                    healthy_data.append({
                        'ID_Subject': base_name,
                        'Source': 'YouTube_Healthy',
                        'Trial_Name': file,
                        'Video_FPS': video_fps,

                        'Knee_Flexion_At_Max': k_f_0[idx],
                        'Trunk_Lean_At_Max': tr_f_0[idx],
                        'Tibia_Lean_At_Max': tib_f_0[idx],

                        'Knee_Flexion_Max': np.max(k_f_0),
                        'Trunk_Lean_Max': np.max(tr_f_0),
                        'Tibia_Lean_Max': np.max(tib_f_0),

                        'Raw_Keypoints': kpts
                    })
                except Exception as e:
                    print(f"Erreur sur {file} : {e}")
else:
    print(f"⚠️ Le dossier {yt_healthy_folder} est introuvable.")

# ==============================================================================
# 2. EXPORT PICKLE
# ==============================================================================
print("2. Export vers le fichier Pickle...")
df_healthy = pd.DataFrame(healthy_data)

if not df_healthy.empty:
    cols_order = ['ID_Subject', 'Source', 'Trial_Name', 'Video_FPS',
                  'Knee_Flexion_At_Max', 'Trunk_Lean_At_Max', 'Tibia_Lean_At_Max',
                  'Knee_Flexion_Max', 'Trunk_Lean_Max', 'Tibia_Lean_Max', 'Raw_Keypoints']

    df_healthy = df_healthy[cols_order]
    df_healthy.to_pickle(output_file_healthy_pkl)

    print(f"\n✅ SUCCÈS : Base de données Healthy (ViTPose) générée avec {len(df_healthy)} essais.")
    print(f"Chemin : {output_file_healthy_pkl}")
else:
    print("\n❌ ERREUR : Aucune donnée n'a été extraite. Vérifie que tes fichiers .mat sont bien présents.")