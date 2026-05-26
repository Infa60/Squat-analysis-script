import cv2
import pandas as pd
import os
import math


def extraire_clips_squats(fichier_excel, dossier_source, dossier_destination):
    """
    Lit l'Excel, retrouve les vidéos, découpe les périodes de squats,
    applique le crop spatial, et sauvegarde les nouvelles vidéos.
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(dossier_destination, exist_ok=True)
    print(f"📁 Dossier de destination prêt : {dossier_destination}")

    # 1. Lire le fichier Excel
    try:
        df = pd.read_excel(fichier_excel)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier Excel : {e}")
        return

    # 2. Filtrer uniquement les lignes qui contiennent une annotation de squat
    # (On ignore les lignes où on a juste mis 'Y' sans annoter de squat)
    df_squats = df.dropna(subset=['Debut_Squat_sec', 'Fin_Squat_sec'])

    if df_squats.empty:
        print("⚠️ Aucun squat annoté n'a été trouvé dans le fichier Excel.")
        return

    print(f"▶️ {len(df_squats)} clip(s) de squat à générer...")

    # Compteur pour nommer les fichiers de manière unique s'il y a plusieurs squats dans une même vidéo
    compteur_clips = 1

    # 3. Parcourir les annotations
    for index, row in df_squats.iterrows():
        nom_video_source = str(row['Fichier_Video'])
        chemin_source = os.path.join(dossier_source, nom_video_source)

        if not os.path.exists(chemin_source):
            print(f"⚠️ Vidéo introuvable, ignorée : {chemin_source}")
            continue

        # Récupération des données temporelles
        t_debut = float(row['Debut_Squat_sec'])
        t_fin = float(row['Fin_Squat_sec'])
        fps_attendu = float(row['Frequence_FPS'])

        # Récupération du crop
        crop_x = int(row.get('Crop_X', 0))
        crop_y = int(row.get('Crop_Y', 0))
        crop_w = int(row.get('Crop_Largeur', 0))
        crop_h = int(row.get('Crop_Hauteur', 0))

        # Ouverture de la vidéo
        cap = cv2.VideoCapture(chemin_source)
        fps_reel = cap.get(cv2.CAP_PROP_FPS)

        # Calcul des frames de début et de fin
        frame_debut = int(math.floor(t_debut * fps_reel))
        frame_fin = int(math.ceil(t_fin * fps_reel))

        # Configuration de la vidéo de sortie
        nom_base, _ = os.path.splitext(nom_video_source)
        # Nom du fichier : ex -> "video1_squat_001.mp4"
        nom_fichier_sortie = f"{nom_base}_squat_{compteur_clips:03d}.mp4"
        chemin_sortie = os.path.join(dossier_destination, nom_fichier_sortie)

        # Définir la taille finale de la vidéo (taille croppée, ou taille d'origine si pas de crop)
        largeur_finale = crop_w if crop_w > 0 else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hauteur_finale = crop_h if crop_h > 0 else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Codec standard MP4 pour OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(chemin_sortie, fourcc, fps_reel, (largeur_finale, hauteur_finale))

        print(f"⏳ Extraction : {nom_fichier_sortie} ({t_debut}s à {t_fin}s)...", end=" ")

        # Se placer à la bonne frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_debut)

        # Lire et écrire les frames une par une
        for current_frame in range(frame_debut, frame_fin + 1):
            ret, frame = cap.read()
            if not ret:
                break  # Fin de la vidéo atteinte prématurément

            # Appliquer le crop si défini
            if crop_w > 0 and crop_h > 0:
                frame_finale = frame[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
            else:
                frame_finale = frame

            out.write(frame_finale)

        # Libérer la mémoire pour cette vidéo
        cap.release()
        out.release()
        print("✅ Terminé")

        compteur_clips += 1

    print(f"\n🎉 Extraction complète ! Vos {len(df_squats)} vidéos sont prêtes dans :")
    print(dossier_destination)


if __name__ == "__main__":
    # --- CONFIGURATION DES CHEMINS ---

    # 1. Le fichier Excel contenant vos annotations
    FICHIER_EXCEL = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_control\dataset_squats.xlsx"

    # 2. Le dossier où se trouvent les vidéos longues originales
    DOSSIER_SOURCE = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_control\Video_squat_v2"

    # 3. Le dossier de destination (Le "r" devant les guillemets est obligatoire pour les chemins Windows)
    DOSSIER_DESTINATION = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\Control2"

    extraire_clips_squats(FICHIER_EXCEL, DOSSIER_SOURCE, DOSSIER_DESTINATION)