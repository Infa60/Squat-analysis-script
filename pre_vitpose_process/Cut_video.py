import pandas as pd
import cv2
import os

# 1. Définition des chemins
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw"
excel_path = fr"{main_path}\Results_annotations_squat_vicon.xlsx"

src_dir_frontal = fr"{main_path}\Full_video_vicon\Frontal_View"
src_dir_sagittal = fr"{main_path}\Full_video_vicon\Sagittal_View"

dst_dir_frontal = fr"{main_path}\Squat_video\CP_vicon\Frontal_View"
dst_dir_sagittal = fr"{main_path}\Squat_video\CP_vicon\Sagittal_View"

os.makedirs(dst_dir_frontal, exist_ok=True)
os.makedirs(dst_dir_sagittal, exist_ok=True)

# 2. Lecture du fichier Excel
print("Lecture du fichier Excel...")
df = pd.read_excel(excel_path)
df_analyse = df[df['Analyse'].astype(str).str.strip().str.upper() == 'Y']
print(f"{len(df_analyse)} ligne(s) à analyser trouvée(s).")


def create_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# 3. Traitement
for index, row in df_analyse.iterrows():
    file_frontal = str(row['Fichier_1_Frontal']).strip()
    file_sagittal = str(row['Fichier_2_Sagittal']).strip()

    path_f = os.path.join(src_dir_frontal, file_frontal)
    path_s = os.path.join(src_dir_sagittal, file_sagittal)

    if not os.path.exists(path_s):
        print(f"ERREUR CRITIQUE: Fichier Sagittal introuvable: {path_s}. Ligne ignorée.")
        continue

    has_frontal = os.path.exists(path_f)
    print(f"\nTraitement en cours: {file_sagittal} (Frontal: {'OK' if has_frontal else 'ABSENT'})")

    # --- RÉCUPÉRATION DES FPS ---
    cap_s = cv2.VideoCapture(path_s)
    fps_s = cap_s.get(cv2.CAP_PROP_FPS)
    w_s, h_s = int(cap_s.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_s.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_f = 0
    cap_f = None
    if has_frontal:
        cap_f = cv2.VideoCapture(path_f)
        fps_f = cap_f.get(cv2.CAP_PROP_FPS)
        w_f, h_f = int(cap_f.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- ÉTAPE A: Adapter les index selon la référence Sagittale ---
    squats_s = []  # Intervalles spécifiques à la vidéo sagittale
    squats_f = []  # Intervalles spécifiques à la vidéo frontale
    max_frame_s = 0
    max_frame_f = 0

    for i in range(1, 6):
        debut_col, fin_col = f'Debut_squat_{i}', f'Fin_squat_{i}'
        if debut_col in row and fin_col in row:
            debut_excel, fin_excel = row[debut_col], row[fin_col]
            if pd.notna(debut_excel) and pd.notna(fin_excel):

                # 1. Les frames de l'Excel (base 1)
                deb_s, fin_s = int(debut_excel), int(fin_excel)
                squats_s.append((i, deb_s, fin_s))
                if fin_s > max_frame_s: max_frame_s = fin_s

                # 2. Calcul des vraies frames pour le Frontal avec correction de l'index 0
                if has_frontal:
                    ratio_fps = fps_f / fps_s

                    # On soustrait 1 pour le calcul, on applique le ratio, on rajoute 1
                    deb_f = int((deb_s - 1) * ratio_fps) + 1
                    fin_f = int((fin_s - 1) * ratio_fps) + 1

                    squats_f.append((i, deb_f, fin_f))
                    if fin_f > max_frame_f: max_frame_f = fin_f

    if not squats_s:
        print("  -> Aucun intervalle trouvé. Vidéo ignorée.")
        cap_s.release()
        if cap_f: cap_f.release()
        continue

    # --- Étape B: Traitement SAGITTAL (Indépendant) ---
    name_s, ext_s = os.path.splitext(file_sagittal)
    writers_s = {}
    for (i, debut, fin) in squats_s:
        out_s = os.path.join(dst_dir_sagittal, f"{name_s}_{i}{ext_s}")
        writers_s[i] = create_writer(out_s, fps_s, w_s, h_s)

    frame_idx = 0
    while cap_s.isOpened():
        ret, frame = cap_s.read()
        if not ret or frame_idx > max_frame_s: break

        for (i, debut, fin) in squats_s:
            if debut <= frame_idx <= fin:
                writers_s[i].write(frame)
        frame_idx += 1

    cap_s.release()
    for w in writers_s.values(): w.release()

    # --- Étape C: Traitement FRONTAL (Indépendant) ---
    if has_frontal:
        name_f, ext_f = os.path.splitext(file_frontal)
        writers_f = {}
        for (i, debut, fin) in squats_f:
            out_f = os.path.join(dst_dir_frontal, f"{name_f}_{i}{ext_f}")
            writers_f[i] = create_writer(out_f, fps_f, w_f, h_f)

        frame_idx = 0
        while cap_f.isOpened():
            ret, frame = cap_f.read()
            if not ret or frame_idx > max_frame_f: break

            for (i, debut, fin) in squats_f:
                if debut <= frame_idx <= fin:
                    writers_f[i].write(frame)
            frame_idx += 1

        cap_f.release()
        for w in writers_f.values(): w.release()

    print(f"  -> Sauvegarde terminée pour {len(squats_s)} squat(s).")

print("\nTraitement terminé avec succès !")