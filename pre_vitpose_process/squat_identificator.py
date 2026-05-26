# ==============================================================================
# 🎮 CONTRÔLES CLAVIER DE L'ANNOTATEUR (AVEC PRÉCHARGEMENT VISUEL)
# ==============================================================================
#
# --- NAVIGATION ET LECTURE ---
# [ESPACE]               : Lecture / Pause
# [D] ou [Flèche Droite] : Avancer d'une image (frame par frame en pause)
# [A] ou [Flèche Gauche] : Reculer d'une image (frame par frame en pause)
#
# --- ANNOTATION DES SQUATS (Max 5 par vidéo) ---
# [S]                    : Définir le début d'un squat (Start)
# [E]                    : Définir la fin d'un squat (End) et sauvegarder
# [C]                    : Effacer TOUTES les annotations de la vidéo en cours (Clear)
#
# --- GESTION DU PROGRAMME ---
# [Q]                    : Valider la vidéo (Analyse = 'Y') et passer à la suivante
# [W]                    : Signaler un problème sur la vidéo (Analyse = 'W') et passer à la suivante
# [ECHAP]                : Quitter immédiatement le programme complet
# ==============================================================================

import cv2

print(f"Version OpenCV : {cv2.__version__}")
import os
import pandas as pd
import time
import sys
import warnings
import numpy as np
import threading
import queue
import gc

# --- SUPPRESSION DE L'AVERTISSEMENT PANDAS ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION DES DOSSIERS ---
dossier_frontal = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_vicon\Frontal_View"
dossier_sagittal = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_vicon\Sagittal_View"
outcome_excel = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw"
fichier_patients_main = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Participant\Main_file.xlsx"

ECHELLE_MEMOIRE = 0.5
LARGEUR_CIBLE_FENETRE = 800
default_fps = 30
nom_fichier_excel = "Results_annotations_squat_vicon.xlsx"
fichier_sortie = os.path.join(outcome_excel, nom_fichier_excel)
extensions_video = ('.mp4', '.avi', '.mov', '.mkv')

# ==============================================================================
# 🛠️ MENUS DE DÉMARRAGE (VUE & FILTRE DIAGNOSTIC)
# ==============================================================================
print("\n==============================================================================")
print("🎥 CONFIGURATION DE L'AFFICHAGE")
print("==============================================================================")
choix_vue = ""
while choix_vue not in ['S', 'B']:
    choix_vue = input(
        "Voulez-vous afficher uniquement la vue Sagittale ou les Deux vues ? (S = Sagittale uniquement / B = Les deux) : ").strip().upper()

print("\n==============================================================================")
print("🏥 FILTRE DES PATIENTS (PC Droit / Droit>Gauche)")
print("==============================================================================")
choix_filtre = ""
while choix_filtre not in ['O', 'N']:
    choix_filtre = input(
        "Voulez-vous filtrer UNIQUEMENT les enfants avec PC Droit ou Droit>Gauche ? (O = Oui / N = Non) : ").strip().upper()

patients_autorises = set()
if choix_filtre == 'O':
    print("⏳ Chargement du fichier Main_file.xlsx...")
    try:
        df_main = pd.read_excel(fichier_patients_main)
        df_main['Diag_clean'] = df_main['CoteDiagnostic'].astype(str).str.lower().str.replace(' ', '').str.replace('\n',
                                                                                                                   '')
        valeurs_cibles = ['droit', 'droit>gauche', 'droitgauche']
        mask = df_main['Diag_clean'].isin(valeurs_cibles)
        patients_autorises = set(df_main[mask]['ID_Patient'].astype(str).values)
        print(f"✅ Filtre activé : {len(patients_autorises)} patients trouvés correspondant au critère.")
    except Exception as e:
        print(f"⚠️ ERREUR lors de la lecture de Main_file.xlsx : {e}")
        print("Le filtre patient sera ignoré (tous les patients seront chargés).")
        choix_filtre = 'N'
print("==============================================================================\n")

# --- INITIALISATION DES VARIABLES GLOBALES ---
action_souris = None
nouvelle_frame_souris = None

etat_chargement = {
    'id_session': '',
    'frames_chargees': 0,
    'total_frames': 0,
    'statut': 'Inactif'
}

# --- PRÉPARATION DES COLONNES EXCEL (Avec les nouveaux FPS) ---
colonnes = ["ID_Patient", "ID_Visite", "Fichier_1_Frontal", "Fichier_2_Sagittal", "Analyse", "FPS_Frontal",
            "FPS_Sagittal"]
for i in range(1, 6):
    colonnes.append(f"Debut_squat_{i}")
    colonnes.append(f"Fin_squat_{i}")

if os.path.exists(fichier_sortie):
    df_global = pd.read_excel(fichier_sortie)
    for col in colonnes:
        if col not in df_global.columns:
            df_global[col] = None
else:
    df_global = pd.DataFrame(columns=colonnes)


def get_infos_fichier(nom_fichier):
    nom_base = os.path.splitext(nom_fichier)[0]
    parts = nom_base.split('_')
    id_patient = parts[0] if len(parts) > 0 else "Inconnu"
    id_visite = parts[1] if len(parts) > 1 else "Inconnu"
    cle_session = f"{id_patient}_{id_visite}"
    return cle_session, id_patient, id_visite


# --- MISE A JOUR DE LA FONCTION DE SAUVEGARDE ---
def sauvegarder_donnees_video(id_patient, id_visite, nom_f, nom_s, annotations, fps_f=None, fps_s=None,
                              analyse_faite='N', quiet=False):
    global df_global
    nouvelle_donnee = {
        "ID_Patient": id_patient, "ID_Visite": id_visite,
        "Fichier_1_Frontal": nom_f, "Fichier_2_Sagittal": nom_s,
        "Analyse": analyse_faite,
        "FPS_Frontal": fps_f,
        "FPS_Sagittal": fps_s
    }
    for i in range(1, 6):
        if i <= len(annotations):
            nouvelle_donnee[f"Debut_squat_{i}"] = annotations[i - 1][0]
            nouvelle_donnee[f"Fin_squat_{i}"] = annotations[i - 1][1]
        else:
            nouvelle_donnee[f"Debut_squat_{i}"] = None
            nouvelle_donnee[f"Fin_squat_{i}"] = None

    mask = (df_global['ID_Patient'].astype(str) == str(id_patient)) & (
                df_global['ID_Visite'].astype(str) == str(id_visite))
    if mask.any():
        idx = df_global[mask].index[0]
        if analyse_faite == 'Wrong side' and df_global.loc[idx, 'Analyse'] == 'Wrong side':
            return
        for k, v in nouvelle_donnee.items():
            df_global.loc[idx, k] = v
    else:
        df_global.loc[len(df_global)] = nouvelle_donnee

    try:
        df_global.to_excel(fichier_sortie, index=False)
        if not quiet:
            print(
                f"\n -> Sauvegarde Excel effectuée pour Patient: {id_patient} | Visite: {id_visite} (Statut: {analyse_faite})")
    except Exception as e:
        print(f"\nERREUR SAUVEGARDE : {e}")


def gestion_souris(event, x, y, flags, param):
    global action_souris, nouvelle_frame_souris
    h, w, total_f = param['hauteur'], param['largeur'], param['total_frames']
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < 20:
            nouvelle_frame_souris = int((x / w) * total_f)
            action_souris = "seek"
        elif h - 50 < y < h - 10:
            if 20 < x < 120:
                action_souris = "prev"
            elif 130 < x < 230:
                action_souris = "pause"
            elif 240 < x < 340:
                action_souris = "next"


def dessiner_interface(img, en_pause, frame_idx, total_f, annotations_existantes):
    global etat_chargement
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 20), (30, 30, 30), -1)
    for (deb, fin) in annotations_existantes:
        x_deb, x_fin = int((deb / total_f) * w), int((fin / total_f) * w)
        cv2.rectangle(img, (x_deb, 0), (x_fin, 20), (255, 200, 0), -1)
    pos_x = int((frame_idx / total_f) * w)
    cv2.line(img, (pos_x, 0), (pos_x, 20), (0, 0, 255), 2)
    cv2.rectangle(img, (130, h - 50), (230, h - 10), (0, 100, 0) if not en_pause else (0, 0, 150), -1)
    txt_pause = "PAUSE ||" if not en_pause else "PLAY >"
    cv2.putText(img, txt_pause, (145, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if etat_chargement['statut'] == 'Chargement...':
        pct = 0
        if etat_chargement['total_frames'] > 0:
            pct = int((etat_chargement['frames_chargees'] / etat_chargement['total_frames']) * 100)
            pct = min(100, pct)
        txt_load = f"Chargement suivant ({etat_chargement['id_session']}) : {pct}%"
        cv2.putText(img, txt_load, (w - 400, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    elif etat_chargement['statut'] == 'Prêt':
        txt_load = f"Video suivante ({etat_chargement['id_session']}) : PRETE"
        cv2.putText(img, txt_load, (w - 350, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


buffer_video = queue.Queue(maxsize=1)


def worker_prechargement(cle_session, data):
    global etat_chargement
    id_patient = data['id_patient']
    id_visite = data['id_visite']
    chemin_f = data['chemin_f']
    chemin_s = data['chemin_s']

    etat_chargement['id_session'] = cle_session

    mask = (df_global['ID_Patient'].astype(str) == str(id_patient)) & (
                df_global['ID_Visite'].astype(str) == str(id_visite))
    if mask.any():
        status_analyse = df_global.loc[mask, 'Analyse'].values[0]
        if pd.notna(status_analyse) and str(status_analyse).strip() in ['Y', 'W', 'Wrong side']:
            etat_chargement['statut'] = 'Ignoré'
            buffer_video.put((cle_session, "SKIPPED", [], {}))
            return

    etat_chargement['statut'] = 'Chargement...'
    frames_cache = []

    cap_f = cv2.VideoCapture(chemin_f) if chemin_f else None
    cap_s = cv2.VideoCapture(chemin_s) if chemin_s else None

    # Extraction brute des FPS pour l'Excel
    raw_fps_f = cap_f.get(cv2.CAP_PROP_FPS) if cap_f and cap_f.isOpened() else None
    raw_fps_s = cap_s.get(cv2.CAP_PROP_FPS) if cap_s and cap_s.isOpened() else None

    # Sécurité mathématique minimale UNIQUEMENT pour la vitesse de lecture
    # (Si la vidéo est corrompue et renvoie 0, on met 30 juste pour éviter que le logiciel ne plante,
    #  mais la valeur enregistrée dans Excel restera bien "0" ou vide pour vous alerter de l'anomalie).
    calc_fps_f = raw_fps_f if raw_fps_f and raw_fps_f > 0 else default_fps
    calc_fps_s = raw_fps_s if raw_fps_s and raw_fps_s > 0 else default_fps

    meta = {
        'fps_f': raw_fps_f,
        'fps_s': raw_fps_s,
        'target_fps': 0
    }

    # ==============================================================================
    # 🔄 CAS 1 : UNE SEULE VUE (SAGITTALE)
    # ==============================================================================
    if not cap_f and cap_s:
        meta['target_fps'] = calc_fps_s
        etat_chargement['total_frames'] = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
        target_idx = 0

        while True:
            ret, frame_s = cap_s.read()
            if not ret: break

            if target_idx % 10 == 0: etat_chargement['frames_chargees'] = target_idx

            img_s = cv2.resize(frame_s, (0, 0), fx=ECHELLE_MEMOIRE, fy=ECHELLE_MEMOIRE)
            h_s, w_s = img_s.shape[:2]

            if w_s != LARGEUR_CIBLE_FENETRE:
                ratio = LARGEUR_CIBLE_FENETRE / w_s
                new_height = int(h_s * ratio)
                img_s = cv2.resize(img_s, (LARGEUR_CIBLE_FENETRE, new_height))

            frames_cache.append(img_s)
            target_idx += 1

        cap_s.release()
        etat_chargement['frames_chargees'] = target_idx
        etat_chargement['statut'] = 'Prêt'
        buffer_video.put((cle_session, "READY", frames_cache, meta))
        return

    # ==============================================================================
    # 🔄 CAS 2 : DEUX VUES (FRONTAL + SAGITTAL)
    # ==============================================================================
    if cap_f and cap_s:
        target_fps = max(calc_fps_f, calc_fps_s)
        meta['target_fps'] = target_fps

        total_f_f = int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT))
        total_f_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))

        etat_chargement['total_frames'] = int(
            max(total_f_f * (target_fps / calc_fps_f), total_f_s * (target_fps / calc_fps_s)))

        shape_f, shape_s = None, None
        ret_f, frame_f = cap_f.read()
        ret_s, frame_s = cap_s.read()

        idx_f = 0 if ret_f else -1
        idx_s = 0 if ret_s else -1
        target_idx = 0

        while True:
            if idx_f == -1 and idx_s == -1: break
            if target_idx % 10 == 0: etat_chargement['frames_chargees'] = target_idx

            t_cible = target_idx / target_fps
            cible_idx_f = int(t_cible * calc_fps_f)
            cible_idx_s = int(t_cible * calc_fps_s)

            while cap_f and idx_f != -1 and idx_f < cible_idx_f:
                ret, next_frame = cap_f.read()
                if ret:
                    frame_f = next_frame;
                    idx_f += 1
                else:
                    idx_f = -1;
                    frame_f = None

            while cap_s and idx_s != -1 and idx_s < cible_idx_s:
                ret, next_frame = cap_s.read()
                if ret:
                    frame_s = next_frame;
                    idx_s += 1
                else:
                    idx_s = -1;
                    frame_s = None

            img_f, img_s = None, None
            if frame_f is not None:
                img_f = cv2.resize(frame_f, (0, 0), fx=ECHELLE_MEMOIRE, fy=ECHELLE_MEMOIRE)
                shape_f = img_f.shape
            elif shape_f is not None:
                img_f = np.zeros(shape_f, dtype=np.uint8)

            if frame_s is not None:
                img_s = cv2.resize(frame_s, (0, 0), fx=ECHELLE_MEMOIRE, fy=ECHELLE_MEMOIRE)
                shape_s = img_s.shape
            elif shape_s is not None:
                img_s = np.zeros(shape_s, dtype=np.uint8)

            if img_f is not None and img_s is not None:
                h_f, w_f = img_f.shape[:2]
                h_s, w_s = img_s.shape[:2]
                if h_f != h_s:
                    new_w_s = int(w_s * (h_f / h_s))
                    img_s = cv2.resize(img_s, (new_w_s, h_f))
                combined_frame = cv2.hconcat([img_f, img_s])
            else:
                break

            h_comb, w_comb = combined_frame.shape[:2]
            if w_comb != LARGEUR_CIBLE_FENETRE:
                ratio = LARGEUR_CIBLE_FENETRE / w_comb
                new_height = int(h_comb * ratio)
                interpolation = cv2.INTER_AREA if w_comb > LARGEUR_CIBLE_FENETRE else cv2.INTER_CUBIC
                combined_frame = cv2.resize(combined_frame, (LARGEUR_CIBLE_FENETRE, new_height),
                                            interpolation=interpolation)

            frames_cache.append(combined_frame)
            target_idx += 1

        cap_f.release()
        cap_s.release()
        etat_chargement['frames_chargees'] = target_idx
        etat_chargement['statut'] = 'Prêt'
        buffer_video.put((cle_session, "READY", frames_cache, meta))


# ==============================================================================
# 🚀 INITIALISATION ET BOUCLE PRINCIPALE
# ==============================================================================
dict_sessions_temp = {}

# 1. On parcourt les dossiers
for dossier, type_vue in [(dossier_frontal, 'frontal'), (dossier_sagittal, 'sagittal')]:
    if not os.path.exists(dossier): continue
    if choix_vue == 'S' and type_vue == 'frontal': continue

    for f in os.listdir(dossier):
        if f.lower().endswith(extensions_video):
            cle_session, id_patient, id_visite = get_infos_fichier(f)
            if cle_session not in dict_sessions_temp:
                dict_sessions_temp[cle_session] = {'id_patient': id_patient, 'id_visite': id_visite, 'chemin_f': None,
                                                   'chemin_s': None, 'nom_f': "", 'nom_s': ""}
            if type_vue == 'frontal':
                dict_sessions_temp[cle_session]['chemin_f'] = os.path.join(dossier, f)
                dict_sessions_temp[cle_session]['nom_f'] = f
            else:
                dict_sessions_temp[cle_session]['chemin_s'] = os.path.join(dossier, f)
                dict_sessions_temp[cle_session]['nom_s'] = f

# 2. Application du filtre
sessions_list = []
nb_wrong_side = 0

for cle_session, data in dict_sessions_temp.items():
    id_patient = data['id_patient']
    if choix_filtre == 'O' and str(id_patient) not in patients_autorises:
        sauvegarder_donnees_video(id_patient, data['id_visite'], data['nom_f'], data['nom_s'], [],
                                  analyse_faite='Wrong side', quiet=True)
        nb_wrong_side += 1
        continue
    sessions_list.append((cle_session, data))

if nb_wrong_side > 0:
    print(
        f"🚫 {nb_wrong_side} patients ont été enregistrés avec l'analyse 'Wrong side' (ne correspondant pas au filtre) et seront ignorés de l'interface.")

if not sessions_list:
    print("\n❌ Aucune vidéo trouvée correspondant à vos critères de filtrage. Fin du programme.")
    sys.exit()

print(
    "\nCommandes : [ESPACE]=Pause | [A/D]=Naviguer | [S]=Start Squat | [E]=End Squat | [C]=Clear | [Q]=Valider | [W]=Problème | [ESC]=Quitter")

if sessions_list:
    cle_0, data_0 = sessions_list[0]
    threading.Thread(target=worker_prechargement, args=(cle_0, data_0), daemon=True).start()

for i in range(len(sessions_list)):
    cle_session, data = sessions_list[i]
    id_patient = data['id_patient']
    id_visite = data['id_visite']
    nom_f = data['nom_f']
    nom_s = data['nom_s']
    infos_console = f"Patient: {id_patient} | Visite: {id_visite}"

    print(f"\n⏳ Initialisation pour {infos_console}...")
    while True:
        try:
            cle_recue, statut, frames_cache, meta = buffer_video.get(timeout=0.2)
            break
        except queue.Empty:
            if etat_chargement['statut'] == 'Chargement...':
                pct = 0
                if etat_chargement['total_frames'] > 0:
                    pct = int((etat_chargement['frames_chargees'] / etat_chargement['total_frames']) * 100)
                    pct = min(100, pct)
                sys.stdout.write(
                    f"\r[CHARGEMENT EN COURS] {pct}% ({etat_chargement['frames_chargees']}/{etat_chargement['total_frames']} frames)")
                sys.stdout.flush()

    print("\n✅ Vidéo prête !")

    if i + 1 < len(sessions_list):
        cle_suiv, data_suiv = sessions_list[i + 1]
        threading.Thread(target=worker_prechargement, args=(cle_suiv, data_suiv), daemon=True).start()

    if statut == "SKIPPED":
        print(f"⏭️ {infos_console} déjà traité. Ignoré.")
        continue

    total_frames = len(frames_cache)
    if total_frames == 0:
        print(f"⚠️ Impossible de lire les vidéos pour {infos_console}. Passage à la suivante.")
        continue

    fps_f_natif = meta.get('fps_f')
    fps_s_natif = meta.get('fps_s')
    target_fps = meta.get('target_fps', default_fps)

    mask = (df_global['ID_Patient'].astype(str) == str(id_patient)) & (
                df_global['ID_Visite'].astype(str) == str(id_visite))
    if not mask.any():
        sauvegarder_donnees_video(id_patient, id_visite, nom_f, nom_s, [], fps_f=fps_f_natif, fps_s=fps_s_natif,
                                  analyse_faite='N')

    target_frame_time = 1.0 / target_fps
    h_display, w_display = frames_cache[0].shape[:2]

    nom_fenetre = f"Annotateur : {infos_console}"
    cv2.namedWindow(nom_fenetre, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nom_fenetre, w_display, h_display)
    cv2.setMouseCallback(nom_fenetre, gestion_souris,
                         {'hauteur': h_display, 'largeur': w_display, 'total_frames': total_frames})

    liste_visuelle_annotations = []
    video_row = df_global[(df_global['ID_Patient'].astype(str) == str(id_patient)) & (
                df_global['ID_Visite'].astype(str) == str(id_visite))].iloc[0]
    for k in range(1, 6):
        if pd.notnull(video_row[f"Debut_squat_{k}"]):
            liste_visuelle_annotations.append((int(video_row[f"Debut_squat_{k}"]), int(video_row[f"Fin_squat_{k}"])))

    frame_index, en_pause, en_enregistrement = 0, True, False
    frame_debut_squat = 0
    update_image_necessaire = True
    dernier_pourcentage_affiche = -1

    while True:
        start_time = time.time()

        if etat_chargement['statut'] == 'Chargement...' and etat_chargement['total_frames'] > 0:
            pct_actuel = int((etat_chargement['frames_chargees'] / etat_chargement['total_frames']) * 100)
            if pct_actuel != dernier_pourcentage_affiche:
                update_image_necessaire = True
                dernier_pourcentage_affiche = pct_actuel
        elif etat_chargement['statut'] == 'Prêt' and dernier_pourcentage_affiche != 100:
            update_image_necessaire = True
            dernier_pourcentage_affiche = 100

        if action_souris:
            if action_souris == "seek":
                frame_index = nouvelle_frame_souris
            elif action_souris == "pause":
                en_pause = not en_pause
            elif action_souris == "next":
                frame_index = min(total_frames - 1, frame_index + 1)
            elif action_souris == "prev":
                frame_index = max(0, frame_index - 1)
            action_souris = None
            update_image_necessaire = True

        if not en_pause and not update_image_necessaire:
            frame_index += 1
            if frame_index >= total_frames:
                frame_index = total_frames - 1
                en_pause = True
            update_image_necessaire = True

        if update_image_necessaire:
            frame_display = frames_cache[frame_index].copy()
            dessiner_interface(frame_display, en_pause, frame_index, total_frames, liste_visuelle_annotations)

            if en_enregistrement and frame_index >= frame_debut_squat:
                cv2.rectangle(frame_display, (w_display - 160, 20), (w_display - 10, 60), (0, 0, 255), -1)
                cv2.putText(frame_display, "SQUAT EN COURS", (w_display - 155, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

            for (deb, fin) in liste_visuelle_annotations:
                if deb <= frame_index <= fin:
                    cv2.putText(frame_display, "SQUAT VALIDE", (w_display // 2 - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

            cv2.putText(frame_display, f"Squats: {len(liste_visuelle_annotations)}/5", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.imshow(nom_fenetre, frame_display)
            update_image_necessaire = False

        processing_time = time.time() - start_time
        wait_time = 10 if en_pause else int((target_frame_time - processing_time) * 1000)
        if wait_time < 1: wait_time = 1
        key = cv2.waitKey(wait_time) & 0xFF

        if key == 32:
            en_pause = not en_pause
        elif key == 83 or key == ord('d'):
            frame_index = min(total_frames - 1, frame_index + 1);
            update_image_necessaire = True
        elif key == 81 or key == ord('a'):
            frame_index = max(0, frame_index - 1);
            update_image_necessaire = True
        elif key == ord('s'):
            frame_debut_squat = frame_index;
            en_enregistrement = True;
            update_image_necessaire = True
        elif key == ord('e'):
            if en_enregistrement:
                if len(liste_visuelle_annotations) < 5:
                    liste_visuelle_annotations.append((frame_debut_squat, frame_index))
                    liste_visuelle_annotations.sort()
                    sauvegarder_donnees_video(id_patient, id_visite, nom_f, nom_s, liste_visuelle_annotations,
                                              fps_f=fps_f_natif, fps_s=fps_s_natif, analyse_faite='N')
                else:
                    print("Maximum de 5 squats atteint !")
                en_enregistrement = False;
                update_image_necessaire = True
        elif key == ord('c'):
            liste_visuelle_annotations = [];
            en_enregistrement = False
            sauvegarder_donnees_video(id_patient, id_visite, nom_f, nom_s, liste_visuelle_annotations,
                                      fps_f=fps_f_natif, fps_s=fps_s_natif, analyse_faite='N')
            update_image_necessaire = True
        elif key == ord('q'):
            sauvegarder_donnees_video(id_patient, id_visite, nom_f, nom_s, liste_visuelle_annotations,
                                      fps_f=fps_f_natif, fps_s=fps_s_natif, analyse_faite='Y');
            break
        elif key == ord('w'):
            sauvegarder_donnees_video(id_patient, id_visite, nom_f, nom_s, liste_visuelle_annotations,
                                      fps_f=fps_f_natif, fps_s=fps_s_natif, analyse_faite='W');
            break
        elif key == 27:
            sys.exit()

    cv2.destroyAllWindows()
    frames_cache.clear()
    del frames_cache
    gc.collect()

print("\n--- Toutes les sessions ont été traitées ---")