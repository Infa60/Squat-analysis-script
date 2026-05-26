import os
import glob
import pandas as pd
import ezc3d
import pickle
import numpy as np

# ---------------------------------------------------------
# 1. Configuration des chemins
# ---------------------------------------------------------
c3d_dir = r"S:\KLab\#SHARE\RESEARCH\ENABLE\squat\squat_qualisys"
excel_annotations_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Results_annotations_squat_qualisys.xlsx"
excel_fps_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\CP_qualisys\Videos_frequency.xlsx"
output_excel_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_c3d\Converted_Squat_Timestamps.xlsx"

# NOUVEAU : Chemin pour sauvegarder les C3D découpés
output_c3d_folder = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_c3d"

# Création du dossier s'il n'existe pas
os.makedirs(output_c3d_folder, exist_ok=True)

# ---------------------------------------------------------
# 2. Chargement et NETTOYAGE des données Excel
# ---------------------------------------------------------
print("Chargement des fichiers Excel...")
df_annotations = pd.read_excel(excel_annotations_path)
df_fps = pd.read_excel(excel_fps_path)


def clean_id_column(col):
    return col.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


df_annotations['ID_Patient'] = clean_id_column(df_annotations['ID_Patient'])
df_annotations['ID_Visite'] = clean_id_column(df_annotations['ID_Visite'])
df_fps['ID_Patient'] = clean_id_column(df_fps['ID_Patient'])
df_fps['ID_Visite'] = clean_id_column(df_fps['ID_Visite'])

# ---------------------------------------------------------
# 3. Parcours des fichiers C3D
# ---------------------------------------------------------
c3d_files = glob.glob(os.path.join(c3d_dir, "*.c3d"))
results = []

print(f"{len(c3d_files)} fichiers .c3d trouvés. Début du traitement...\n")

for c3d_path in c3d_files:
    filename = os.path.basename(c3d_path)
    name_without_ext = filename.replace('.c3d', '')

    parts = name_without_ext.split('_')
    if len(parts) < 2:
        print(f"⚠️ Nom de fichier non conforme ignoré : {filename}")
        continue

    id_patient = clean_id_column(pd.Series([parts[0]]))[0]
    id_visite = clean_id_column(pd.Series([parts[1]]))[0]

    # ---------------------------------------------------------
    # 4. Recherche dans les fichiers Excel
    # ---------------------------------------------------------
    match_anno = df_annotations[(df_annotations['ID_Patient'] == id_patient) &
                                (df_annotations['ID_Visite'] == id_visite)]

    match_fps = df_fps[(df_fps['ID_Patient'] == id_patient) &
                       (df_fps['ID_Visite'] == id_visite)]

    if match_anno.empty:
        print(f"❌ MANQUANT dans 'Results_annotations' : {id_patient}_{id_visite}")
        continue

    if match_fps.empty:
        print(f"❌ MANQUANT dans 'Videos_frequency' : {id_patient}_{id_visite}")
        continue

    row_anno = match_anno.iloc[0]
    fps_video = match_fps.iloc[0]['FPS_Final_Sync']

    # ---------------------------------------------------------
    # 5. Récupération de la fréquence du C3D original
    # ---------------------------------------------------------
    try:
        c3d_orig = ezc3d.c3d(c3d_path)
        fps_c3d = c3d_orig['header']['points']['frame_rate']
        fps_analog = c3d_orig['header']['analogs']['frame_rate']

        # Le ratio permet de savoir combien de frames analogiques il y a pour 1 frame 3D
        analog_ratio = int(fps_analog / fps_c3d) if fps_c3d > 0 else 1
    except Exception as e:
        print(f"❌ Erreur lecture C3D {filename}: {e}")
        continue

    ratio_video = fps_c3d / fps_video

    file_data = {
        'Fichier_C3D': filename, 'ID_Patient': id_patient, 'ID_Visite': id_visite,
        'FPS_Video': fps_video, 'FPS_C3D': fps_c3d, 'Ratio_Conversion': ratio_video
    }

    print(f"Traitement de {filename}...")

    # ---------------------------------------------------------
    # 6. Conversion et DÉCOUPAGE pour les 5 squats
    # ---------------------------------------------------------
    for i in range(1, 6):
        col_debut = f"Debut_squat_{i}"
        col_fin = f"Fin_squat_{i}"

        if col_debut in row_anno and col_fin in row_anno:
            val_debut = row_anno[col_debut]
            val_fin = row_anno[col_fin]

            if pd.notna(val_debut) and pd.notna(val_fin):
                frame_c3d_debut = int(round(val_debut * ratio_video))
                frame_c3d_fin = int(round(val_fin * ratio_video))

                file_data[f'{col_debut}_video'] = val_debut
                file_data[f'{col_fin}_video'] = val_fin
                file_data[f'{col_debut}_C3D'] = frame_c3d_debut
                file_data[f'{col_fin}_C3D'] = frame_c3d_fin

                # --- DÉBUT DU DÉCOUPAGE ---
                try:
                    c3d_crop = ezc3d.c3d(c3d_path)

                    start_idx = max(0, frame_c3d_debut)
                    end_idx = frame_c3d_fin

                    # 1. Découpage Points 3D
                    if c3d_crop['data']['points'].shape[2] > 0:
                        c3d_crop['data']['points'] = c3d_crop['data']['points'][:, :, start_idx:end_idx + 1]

                    # 2. Découpage Analogique
                    if len(c3d_crop['data']['analogs']) > 0 and c3d_crop['data']['analogs'].shape[2] > 0:
                        start_ana = start_idx * analog_ratio
                        end_ana = (end_idx + 1) * analog_ratio
                        c3d_crop['data']['analogs'] = c3d_crop['data']['analogs'][:, :, start_ana:end_ana]

                    # Suppression des métadonnées conflictuelles
                    if 'meta_points' in c3d_crop['data']:
                        del c3d_crop['data']['meta_points']
                    if 'meta_analogs' in c3d_crop['data']:
                        del c3d_crop['data']['meta_analogs']

                    # 3. Mise à jour des paramètres
                    nb_frames = (end_idx - start_idx) + 1
                    c3d_crop['parameters']['POINT']['FRAMES']['value'] = [nb_frames]

                    # ---------------------------------------------------------
                    # 4. Sauvegarde en fichier Pickle (.pkl) "Propre"
                    # ---------------------------------------------------------
                    out_name = f"{name_without_ext}_{i}.pkl"
                    out_path = os.path.join(output_c3d_folder, out_name)

                    # Récupération des noms des marqueurs
                    labels_points = [str(label) for label in c3d_crop['parameters']['POINT']['LABELS']['value']]

                    # Récupération des noms des canaux analogiques (s'ils existent)
                    labels_analogs = []
                    if 'ANALOG' in c3d_crop['parameters'] and 'LABELS' in c3d_crop['parameters']['ANALOG']:
                        labels_analogs = [str(label) for label in c3d_crop['parameters']['ANALOG']['LABELS']['value']]

                    # Création d'un dictionnaire 100% Python/Numpy (AUCUN objet C++ SWIG)
                    clean_dict = {
                        'points_3d': np.array(c3d_crop['data']['points']),
                        'analogs': np.array(c3d_crop['data']['analogs']) if len(
                            c3d_crop['data']['analogs']) > 0 else np.array([]),
                        'labels_points': labels_points,
                        'labels_analogs': labels_analogs,
                        'fps_3d': fps_c3d,
                        'fps_analog': fps_analog
                        # (assurez-vous que fps_analog est bien défini en étape 5 du script principal)
                    }

                    # Sauvegarde
                    with open(out_path, 'wb') as f:
                        pickle.dump(clean_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print(f"  -> Squat {i} exporté en PKL : {out_name}")
                    # ---------------------------------------------------------
                    # ---------------------------------------------------------

                except Exception as e:
                    print(f"  ❌ Erreur de découpage sur {filename} (Squat {i}) : {e}")
                # --- FIN DU DÉCOUPAGE ---

            else:
                file_data[f'{col_debut}_video'] = None
                file_data[f'{col_fin}_video'] = None
                file_data[f'{col_debut}_C3D'] = None
                file_data[f'{col_fin}_C3D'] = None

    results.append(file_data)

# ---------------------------------------------------------
# 7. Sauvegarde des résultats Excel
# ---------------------------------------------------------
if results:
    df_results = pd.DataFrame(results)
    df_results.to_excel(output_excel_path, index=False)
    print(f"\n✅ Terminé ! Excel généré ici : {output_excel_path}")
    print(f"📂 Tous vos C3D découpés sont disponibles dans : {output_c3d_folder}")
else:
    print("\n❌ Aucun fichier n'a pu être traité.")