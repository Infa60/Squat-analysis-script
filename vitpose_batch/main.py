import os
import cv2
import numpy as np
import scipy.io as sio
import sys
import pandas as pd
import mmdet
# Importation allégée
from model_loading import load_all_models
from vitpose_module import get_vitpose_full_image

# Paires COCO (17 points)
COCO_PAIRS = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# Paires HALPE (26 points)
HALPE_PAIRS = COCO_PAIRS + [
    (15, 17), (15, 18), (18, 19), (19, 17),
    (16, 20), (16, 21), (21, 22), (22, 20)
]


# ==========================================================
# FONCTIONS DE TRACKING SPATIO-TEMPOREL
# ==========================================================

def estimate_kinematic_projection(trajectory_history):
    if not trajectory_history or len(trajectory_history) < 2:
        return trajectory_history[-1]['kpts'] if trajectory_history else None

    kpts_t1 = trajectory_history[-1]['kpts']
    kpts_t2 = trajectory_history[-2]['kpts']
    pred_kpts = np.copy(kpts_t1)

    for pt in range(len(kpts_t1)):
        if not np.isnan(kpts_t1[pt][0]) and not np.isnan(kpts_t2[pt][0]):
            vx = kpts_t1[pt][0] - kpts_t2[pt][0]
            vy = kpts_t1[pt][1] - kpts_t2[pt][1]
            pred_kpts[pt][0] = kpts_t1[pt][0] + vx
            pred_kpts[pt][1] = kpts_t1[pt][1] + vy
    return pred_kpts


def compute_spatiotemporal_matching_cost(projected_kpts, candidate_pose):
    if projected_kpts is None:
        return float('inf')

    new_kpts = candidate_pose['kpts']
    mask = ~np.isnan(projected_kpts[:, 0]) & ~np.isnan(new_kpts[:, 0])

    if np.sum(mask) < 3:
        return float('inf')

    dists = np.sqrt(np.sum((projected_kpts[mask] - new_kpts[mask]) ** 2, axis=1))
    return np.mean(dists)


def perform_spatiotemporal_tracking(candidates, history_registry, img_w, img_h):
    """
    Algorithme de suivi par association de données (Data Association).
    Attribue l'identité du Patient (Index 0) et du Soignant (Index 1).
    """
    if not candidates:
        return [None, None], history_registry

    # Seuil de tolérance cinématique (12% de la dimension max de l'image)
    dynamic_threshold = max(img_w, img_h) * 0.12
    assigned_identities = [None, None]

    # --- ÉTAPE 1 : INITIALISATION (Frame 1) ---
    if len(history_registry[0]) == 0:
        # Priorité au sujet le plus bas dans l'image (Y max)
        candidates = sorted(candidates, key=lambda c: c['box'][3], reverse=True)
        assigned_identities[0] = candidates[0]
        if len(candidates) > 1:
            assigned_identities[1] = candidates[1]

        # On retourne directement ici pour la première frame
        for i in [0, 1]:
            if assigned_identities[i] is not None:
                history_registry[i].append(assigned_identities[i])
        return assigned_identities, history_registry

    # --- ÉTAPE 2 : ASSOCIATION PAR PROJECTION ---
    available_candidates = candidates.copy()

    # Génération des squelettes "fantômes" (projections)
    projection_0 = estimate_kinematic_projection(history_registry[0])
    projection_1 = estimate_kinematic_projection(history_registry[1])

    if len(available_candidates) >= 2 and projection_0 is not None and projection_1 is not None:
        # Matrice de coût simplifiée pour les deux scénarios d'attribution
        # Scénario A : Patient=Cand0, Soignant=Cand1
        cost_A = compute_spatiotemporal_matching_cost(projection_0, available_candidates[0]) + \
                 compute_spatiotemporal_matching_cost(projection_1, available_candidates[1])

        # Scénario B : Patient=Cand1, Soignant=Cand0
        cost_B = compute_spatiotemporal_matching_cost(projection_0, available_candidates[1]) + \
                 compute_spatiotemporal_matching_cost(projection_1, available_candidates[0])

        if cost_A < cost_B:
            assigned_identities[0] = available_candidates[0]
            assigned_identities[1] = available_candidates[1]
        else:
            assigned_identities[0] = available_candidates[1]
            assigned_identities[1] = available_candidates[0]

        if assigned_identities[0] in available_candidates: available_candidates.remove(assigned_identities[0])
        if assigned_identities[1] in available_candidates: available_candidates.remove(assigned_identities[1])

    # --- ÉTAPE 3 : PROCÉDURE DE RATTRAPAGE (Occlusion partielle) ---
    else:
        for id_idx in [0, 1]:
            proj = projection_0 if id_idx == 0 else projection_1
            if proj is not None and available_candidates:
                # Recherche du candidat minimisant le coût par rapport à la projection
                best_match = min(available_candidates,
                                 key=lambda c: compute_spatiotemporal_matching_cost(proj, c))

                if compute_spatiotemporal_matching_cost(proj, best_match) < dynamic_threshold:
                    assigned_identities[id_idx] = best_match
                    available_candidates.remove(best_match)

    # --- ÉTAPE 4 : L'ARBITRE SPATIAL (Correction par la boîte) ---
    # Si les deux sujets sont clairement identifiés par la cinématique, on vérifie

    # if assigned_identities[0] is not None and assigned_identities[1] is not None:
        # y_bottom_0 = assigned_identities[0]['box'][3]
        # y_bottom_1 = assigned_identities[1]['box'][3]

        # if y_bottom_1 > y_bottom_0:
            # Le soignant a une BBox plus basse que le patient. On inverse pour corriger !
            # assigned_identities[0], assigned_identities[1] = assigned_identities[1], assigned_identities[0]

    # Mise à jour sélective de l'historique temporel (fenêtre glissante)
    for i in [0, 1]:
        if assigned_identities[i] is not None:
            history_registry[i].append(assigned_identities[i])
            if len(history_registry[i]) > 5:
                history_registry[i].pop(0)

    return assigned_identities, history_registry


def process_modality(video_path, out_video_path, model_key, models, config):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    temp_frame = cap.read()[1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, test_preds = get_vitpose_full_image(temp_frame, models[model_key])

    n_kpts = len(test_preds[0]['keypoints']) if test_preds else 17
    pairs_to_draw = HALPE_PAIRS if n_kpts > 17 else COCO_PAIRS

    max_ppl = 2
    all_frames = np.arange(total_frames)

    # --- DONNÉES FILTRÉES ---
    all_boxes = np.full((total_frames, max_ppl, 4), np.nan)
    all_keypoints = np.full((total_frames, max_ppl, n_kpts, 2), np.nan)
    history_registry = {0: [], 1: []}

    # VOTRE BOUCLIER ANTI-ABERRATION (Préservé)
    prev_keypoints = {0: np.full((n_kpts, 2), np.nan), 1: np.full((n_kpts, 2), np.nan)}

    # --- DONNÉES BRUTES (NOUVEAU) ---
    raw_boxes = np.empty((total_frames,), dtype=object)
    raw_keypoints = np.empty((total_frames,), dtype=object)
    raw_scores = np.empty((total_frames,), dtype=object)

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        percent = (frame_idx + 1) / total_frames * 100
        sys.stdout.write(
            f"\r      Progress: [{'#' * int(percent // 5):<20}] {percent:.1f}% ({frame_idx + 1}/{total_frames})")
        sys.stdout.flush()

        # NOUVEAU : Amélioration du contraste (CLAHE)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        viz_frame = frame.copy()
        candidates = []
        frame_raw_boxes, frame_raw_kpts, frame_raw_scores = [], [], []

        # Envoi de l'image améliorée à l'IA
        _, predictions = get_vitpose_full_image(frame, models[model_key])

        for p in predictions:
            # (f"DEBUG - Taille de l'image: {width}x{height} | Coordonnées Box: {p['bbox'][0]}")
            # Remplissage RAW
            frame_raw_boxes.append(p['bbox'][0])
            frame_raw_kpts.append(np.array(p['keypoints'])[:, :2])
            frame_raw_scores.append(p['keypoint_scores'])

            # Remplissage FILTERED (avec votre seuil de confiance)
            kpts = np.full((n_kpts, 2), np.nan)
            for i in range(min(n_kpts, len(p['keypoints']))):
                if p['keypoint_scores'][i] > config.get('min_conf', 0.4):
                    kpts[i] = [p['keypoints'][i][0], p['keypoints'][i][1]]
            candidates.append({'box': p['bbox'][0], 'kpts': kpts})

        # Sauvegarde RAW
        raw_boxes[frame_idx] = np.array(frame_raw_boxes) if frame_raw_boxes else np.empty((0, 4))
        raw_keypoints[frame_idx] = np.array(frame_raw_kpts) if frame_raw_kpts else np.empty((0, n_kpts, 2))
        raw_scores[frame_idx] = np.array(frame_raw_scores) if frame_raw_scores else np.empty((0, n_kpts))

        assigned_candidates, history_registry = perform_spatiotemporal_tracking(candidates, history_registry, width,
                                                                                height)

        for p_idx, c in enumerate(assigned_candidates[:max_ppl]):
            if c is not None:
                # ==========================================================
                # VOTRE BOUCLIER ANTI-ABERRATION (Rejet des téléportations)
                # ==========================================================
                bbox_w = c['box'][2] - c['box'][0]
                bbox_h = c['box'][3] - c['box'][1]
                max_joint_jump = max(bbox_w, bbox_h) * 0.25

                for i in range(n_kpts):
                    curr_pt = c['kpts'][i]
                    if not np.isnan(curr_pt[0]):
                        prev_pt = prev_keypoints[p_idx][i]
                        if not np.isnan(prev_pt[0]):
                            dist = np.sqrt((curr_pt[0] - prev_pt[0]) ** 2 + (curr_pt[1] - prev_pt[1]) ** 2)

                            if dist > max_joint_jump:
                                c['kpts'][i] = [np.nan, np.nan]  # Rejet
                            else:
                                prev_keypoints[p_idx][i] = curr_pt
                        else:
                            prev_keypoints[p_idx][i] = curr_pt

                all_boxes[frame_idx, p_idx] = c['box']
                all_keypoints[frame_idx, p_idx] = c['kpts']

                box = all_boxes[frame_idx, p_idx]
                color = (0, 255, 0) if p_idx == 0 else (0, 165, 255)

                cv2.rectangle(viz_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)

                for i in range(n_kpts):
                    kp = all_keypoints[frame_idx, p_idx, i]
                    if not np.isnan(kp[0]):
                        cv2.circle(viz_frame, (int(kp[0]), int(kp[1])), 4, color, -1)

                for pt1, pt2 in pairs_to_draw:
                    if pt1 < n_kpts and pt2 < n_kpts:
                        k1, k2 = all_keypoints[frame_idx, p_idx, pt1], all_keypoints[frame_idx, p_idx, pt2]
                        if not np.isnan(k1[0]) and not np.isnan(k2[0]):
                            cv2.line(viz_frame, (int(k1[0]), int(k1[1])), (int(k2[0]), int(k2[1])), color, 2)

        out_video.write(viz_frame)
        frame_idx += 1

    cap.release()
    out_video.release()
    sys.stdout.write(f"\r      Progress: [{'#' * 20}] 100.0% - Terminé.          \n")
    sys.stdout.flush()

    # NOUVEAU : On retourne les deux dicos
    return (
        {'Frames': all_frames, 'BoundingBoxes': all_boxes, 'Keypoints': all_keypoints},
        {'Frames': all_frames, 'BoundingBoxes_Raw': raw_boxes, 'Keypoints_Raw': raw_keypoints, 'Scores_Raw': raw_scores}
    )


if __name__ == '__main__':
    print("====================================================")
    print("STEP 1: INITIALISATION DES MODÈLES D'IA")
    print("====================================================")
    models = load_all_models()

    modalities = {
        "ViTPose_Huge": {"model_key": "vitpose_base", "min_conf": 0.4},
    }

    # in_folder = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\CP_vicon\Sagittal_View"
    # in_folder = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\Control"
    in_folder = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\CP_qualisys\Frontal_View"

    out_folder = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Processed\CP_qualisys\Frontal_View\Results"

    out_excel = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Processed\CP_qualisys\Frontal_View"


    os.makedirs(out_folder, exist_ok=True)

    videos = [f for f in os.listdir(in_folder) if f.endswith(('.avi', '.mp4'))]
    num_videos = len(videos)

    print(f"\n[INFO] {num_videos} vidéos trouvées.")

    # VOTRE GESTION EXCEL (Préservée)
    tracker_path = os.path.join(out_excel, "Tracker_Analyses.xlsx")
    video_basenames = [os.path.splitext(v)[0] for v in videos]

    if os.path.exists(tracker_path):
        df_tracker = pd.read_excel(tracker_path)
    else:
        df_tracker = pd.DataFrame(columns=['Video_Name'] + list(modalities.keys()))
        df_tracker['Video_Name'] = video_basenames

    existing_videos = df_tracker['Video_Name'].tolist()
    missing_videos = [v for v in video_basenames if v not in existing_videos]
    if missing_videos:
        new_df = pd.DataFrame({'Video_Name': missing_videos})
        df_tracker = pd.concat([df_tracker, new_df], ignore_index=True)

    for mod in modalities.keys():
        if mod not in df_tracker.columns:
            df_tracker[mod] = None

    df_tracker.to_excel(tracker_path, index=False)
    print(f"[INFO] Tracker Excel initialisé/chargé : {tracker_path}")

    # BOUCLE D'ANALYSE
    for v_idx, vid in enumerate(videos):
        nom_base = os.path.splitext(vid)[0]
        print(f"\n[{v_idx + 1}/{num_videos}] >>> ANALYSE DU SUJET : {nom_base}")

        participant_folder = os.path.join(out_folder, nom_base)
        os.makedirs(participant_folder, exist_ok=True)

        # VOTRE LOGIQUE DE MISE A JOUR DES .MAT (Préservée et adaptée pour Raw/Filtered)
        mat_path_filtered = os.path.join(participant_folder, f"{nom_base}_Results_Filtered.mat")
        mat_path_raw = os.path.join(participant_folder, f"{nom_base}_Results_Raw.mat")

        if os.path.exists(mat_path_filtered):
            try:
                loaded_mat_f = sio.loadmat(mat_path_filtered)
                video_mat_filtered = {k: v for k, v in loaded_mat_f.items() if not k.startswith('__')}
            except Exception:
                video_mat_filtered = {}
        else:
            video_mat_filtered = {}

        if os.path.exists(mat_path_raw):
            try:
                loaded_mat_r = sio.loadmat(mat_path_raw)
                video_mat_raw = {k: v for k, v in loaded_mat_r.items() if not k.startswith('__')}
            except Exception:
                video_mat_raw = {}
        else:
            video_mat_raw = {}

        for mod_name, config in modalities.items():
            row_idx = df_tracker.index[df_tracker['Video_Name'] == nom_base].tolist()[0]
            if df_tracker.at[row_idx, mod_name] == 'x':
                print(f"   [SKIPPED] Modalité '{mod_name}' déjà traitée ! (x trouvé dans Excel)")
                continue

            print(f"   Modalité : {mod_name}")
            out_v_path = os.path.join(participant_folder, f"{mod_name}.avi")

            filtered_data, raw_data = process_modality(os.path.join(in_folder, vid), out_v_path, config['model_key'],
                                                       models, config)

            video_mat_filtered[mod_name] = filtered_data
            video_mat_raw[mod_name] = raw_data

            sio.savemat(mat_path_filtered, video_mat_filtered)
            sio.savemat(mat_path_raw, video_mat_raw)

            df_tracker.at[row_idx, mod_name] = 'x'
            df_tracker.to_excel(tracker_path, index=False)

    print("\n================ TOUTES LES ANALYSES SONT TERMINÉES ================")