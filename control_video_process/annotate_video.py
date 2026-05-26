import cv2
import pandas as pd
import numpy as np
import os


def lancer_annotateur(dossier_videos, fichier_excel_sortie="annotations_squats.xlsx"):
    extensions = ('.mp4', '.avi', '.mov', '.mkv')
    toutes_videos = [f for f in os.listdir(dossier_videos) if f.lower().endswith(extensions)]

    if not toutes_videos:
        print(f"❌ Aucune vidéo trouvée dans le dossier '{dossier_videos}'.")
        return

    donnees_annotations = []
    videos_deja_vues = set()

    # --- 1. LECTURE DE L'EXCEL EXISTANT (POUR REPRENDRE OÙ ON S'EST ARRÊTÉ) ---
    if os.path.exists(fichier_excel_sortie):
        try:
            df_existant = pd.read_excel(fichier_excel_sortie)
            donnees_annotations = df_existant.to_dict('records')

            # Trouver toutes les vidéos qui ont déjà le tag "Y"
            if "Video_Vue" in df_existant.columns:
                videos_deja_vues = set(df_existant[df_existant["Video_Vue"] == "Y"]["Fichier_Video"])
            print(f"📂 Fichier Excel détecté ! {len(videos_deja_vues)} vidéo(s) déjà traitée(s) ignorée(s).")
        except Exception as e:
            print(f"⚠️ Erreur de lecture Excel : {e}. Démarrage à zéro.")

    # Filtrer pour ne garder que les vidéos non vues
    videos_a_traiter = [v for v in toutes_videos if v not in videos_deja_vues]

    if not videos_a_traiter:
        print("✅ Super ! Toutes les vidéos de ce dossier ont déjà été traitées (marquées d'un 'Y').")
        return

    print(f"▶️ Il reste {len(videos_a_traiter)} vidéo(s) à annoter.")

    for nom_video in videos_a_traiter:
        chemin_video = os.path.join(dossier_videos, nom_video)
        cap = cv2.VideoCapture(chemin_video)

        if not cap.isOpened():
            print(f"⚠️ Impossible d'ouvrir {nom_video}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- ÉTAPE 1 : CROP ---
        ret, frame_initiale = cap.read()
        if not ret:
            continue

        print(f"\n" + "=" * 50)
        print(f"🎬 VIDÉO ACTUELLE : {nom_video}")
        print("=" * 50)

        nom_fenetre_crop = f"CROP - {nom_video}"
        cv2.namedWindow(nom_fenetre_crop, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        bbox = cv2.selectROI(nom_fenetre_crop, frame_initiale, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(nom_fenetre_crop)
        x, y, w, h = bbox

        # --- ÉTAPE 2 : LECTURE ET ANNOTATION ---
        nom_fenetre_player = f"PLAYER - {nom_video}"
        cv2.namedWindow(nom_fenetre_player, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        en_lecture = True
        squat_en_cours = False
        debut_squat_sec = 0.0
        quitter_programme = False
        update_frame_requise = False
        mise_a_jour_auto = False
        current_frame = frame_initiale.copy()

        squats_video_actuelle = []

        def on_trackbar(val):
            nonlocal update_frame_requise
            if not mise_a_jour_auto:
                cap.set(cv2.CAP_PROP_POS_FRAMES, val)
                update_frame_requise = True

        cv2.createTrackbar("Prog", nom_fenetre_player, 0, total_frames, on_trackbar)
        delai_attente = int(1000 / fps) if fps > 0 else 30

        while True:
            if en_lecture or update_frame_requise:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    en_lecture = False
                    continue

                current_frame = frame.copy()
                update_frame_requise = False

                mise_a_jour_auto = True
                cv2.setTrackbarPos("Prog", nom_fenetre_player, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                mise_a_jour_auto = False

            frame_affichage = current_frame.copy()
            temps_actuel_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Affichage du Crop
            if w > 0 and h > 0:
                cv2.rectangle(frame_affichage, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- AFFICHAGE DES TEXTES ET RÉCAPITULATIF DES TOUCHES ---
            statut = "LECTURE" if en_lecture else "PAUSE"
            cv2.putText(frame_affichage, f"Temps: {temps_actuel_sec:.2f}s | {statut}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if squat_en_cours:
                cv2.putText(frame_affichage, f"🔴 SQUAT EN COURS",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Bandeau noir en bas de l'image pour afficher les touches de façon lisible
            hauteur_video, largeur_video = frame_affichage.shape[:2]
            cv2.rectangle(frame_affichage, (10, hauteur_video - 70), (1000, hauteur_video - 10), (0, 0, 0), -1)

            texte_touches_1 = "TOUCHES : [Espace] Pause  |  [a] Reculer  |  [d] Avancer  |  [n] Suivant  |  [q] Quitter"
            texte_touches_2 = "SQUATS  : [s] Debut Squat  |  [e] Fin Squat"
            cv2.putText(frame_affichage, texte_touches_1, (20, hauteur_video - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.putText(frame_affichage, texte_touches_2, (20, hauteur_video - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

            # --- TIMELINE ---
            hauteur_timeline = max(40, int(hauteur_video * 0.08))
            timeline = np.zeros((hauteur_timeline, largeur_video, 3), dtype=np.uint8)
            timeline[:] = (40, 40, 40)

            for (debut_sec, fin_sec) in squats_video_actuelle:
                x_debut = int((debut_sec * fps / total_frames) * largeur_video)
                x_fin = int((fin_sec * fps / total_frames) * largeur_video)
                cv2.rectangle(timeline, (x_debut, 0), (x_fin, hauteur_timeline), (255, 100, 0), -1)

            if squat_en_cours:
                x_debut = int((debut_squat_sec * fps / total_frames) * largeur_video)
                x_actuel = int((temps_actuel_sec * fps / total_frames) * largeur_video)
                cv2.rectangle(timeline, (x_debut, 0), (x_actuel, hauteur_timeline), (255, 255, 0), -1)

            x_curseur = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * largeur_video)
            cv2.line(timeline, (x_curseur, 0), (x_curseur, hauteur_timeline), (0, 0, 255),
                     max(2, int(largeur_video * 0.004)))

            frame_finale = cv2.vconcat([frame_affichage, timeline])
            cv2.imshow(nom_fenetre_player, frame_finale)

            # --- GESTION DES TOUCHES MODIFIÉE ---
            touche = cv2.waitKeyEx(delai_attente if en_lecture else 30)

            if touche != -1:
                touche_ascii = touche & 0xFF

                if touche_ascii == ord(' '):
                    en_lecture = not en_lecture

                # [a] pour Reculer d'une frame
                elif touche in (2424832, 65361) or touche_ascii == ord('a'):
                    en_lecture = False
                    pos_actuelle = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos_actuelle - 2))
                    update_frame_requise = True

                # [d] pour Avancer d'une frame
                elif touche in (2555904, 65363) or touche_ascii == ord('d'):
                    en_lecture = False
                    update_frame_requise = True

                elif touche_ascii == ord('s') and not squat_en_cours:
                    debut_squat_sec = temps_actuel_sec
                    squat_en_cours = True

                elif touche_ascii == ord('e') and squat_en_cours:
                    fin_squat_sec = temps_actuel_sec
                    squat_en_cours = False
                    squats_video_actuelle.append((debut_squat_sec, fin_squat_sec))

                    donnees_annotations.append({
                        "Fichier_Video": nom_video, "Frequence_FPS": round(fps, 2),
                        "Crop_X": x, "Crop_Y": y, "Crop_Largeur": w, "Crop_Hauteur": h,
                        "Debut_Squat_sec": round(debut_squat_sec, 2),
                        "Fin_Squat_sec": round(fin_squat_sec, 2),
                        "Video_Vue": "Y"  # ✅ Ajout automatique du Y
                    })

                elif touche_ascii == ord('n'):
                    break

                elif touche_ascii == ord('q'):
                    quitter_programme = True
                    break

        cap.release()
        cv2.destroyAllWindows()

        # --- ENREGISTREMENT AUTOMATIQUE DU STATUT "VUE" ---
        # Vérifier si la vidéo avait des squats. Si non, on ajoute une ligne vide juste pour dire qu'on l'a regardée ("Y")
        a_des_squats = any(d.get("Fichier_Video") == nom_video for d in donnees_annotations)
        if not a_des_squats:
            donnees_annotations.append({
                "Fichier_Video": nom_video,
                "Frequence_FPS": round(fps, 2),
                "Video_Vue": "Y"
            })

        # Sauvegarde de l'Excel APRÈS CHAQUE VIDÉO
        pd.DataFrame(donnees_annotations).to_excel(fichier_excel_sortie, index=False)
        print(f"💾 Fichier Excel mis à jour. '{nom_video}' est enregistrée comme VUE.")

        if quitter_programme:
            break

    print(f"\n✅ Terminée ! Toutes les annotations sont sauvegardées dans '{fichier_excel_sortie}'.")


if __name__ == "__main__":
    # Renseignez ici le nom du dossier qui contient vos vidéos
    DOSSIER_VIDEOS = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_control\Video_squat_v2"
    FICHIER_SORTIE = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_control\dataset_squats.xlsx"

    lancer_annotateur(DOSSIER_VIDEOS, FICHIER_SORTIE)