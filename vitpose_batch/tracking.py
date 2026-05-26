import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class RobustPoseTracker:
    def __init__(self, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h

        # Historique des cibles: 0 = Patient (devant), 1 = Soignant (derrière)
        self.tracks = {0: None, 1: None}

        # ANCRES VISUELLES : On garde en mémoire la couleur de la toute première frame
        self.anchor_features = {0: None, 1: None}

        # Poids de la matrice de coût
        self.W_POSE = 0.5    # 50%
        self.W_VISUAL = 0.3  # 30%
        self.W_SPATIAL = 0.2 # 20%

    def extract_visual_features(self, frame, kpts):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        valid_pts = 0

        for pt in kpts:
            x, y = pt[0], pt[1]
            if not np.isnan(x) and not np.isnan(y):
                if 0 <= int(x) < self.img_w and 0 <= int(y) < self.img_h:
                    cv2.circle(mask, (int(x), int(y)), 15, 255, -1)
                    valid_pts += 1

        if valid_pts == 0:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten()

    def compute_pose_cost(self, kpts_track, kpts_cand):
        mask = ~np.isnan(kpts_track[:, 0]) & ~np.isnan(kpts_cand[:, 0])
        if np.sum(mask) < 3:
            return 1.0

        dists = np.sqrt(np.sum((kpts_track[mask] - kpts_cand[mask]) ** 2, axis=1))
        mean_dist = np.mean(dists)

        max_dist_allowed = max(self.img_w, self.img_h) * 0.10
        return min(mean_dist / max_dist_allowed, 1.0)

    def compute_cost_matrix(self, candidates, frame):
        num_tracks = 2
        num_candidates = len(candidates)
        cost_matrix = np.full((num_tracks, num_candidates), 1e5)

        for t_id in [0, 1]:
            track = self.tracks[t_id]
            if track is None:
                continue

            anchor_self = self.anchor_features[t_id]
            other_t_id = 1 if t_id == 0 else 0
            anchor_other = self.anchor_features[other_t_id]

            for c_idx, cand in enumerate(candidates):
                # 1. Spatial
                dist_spatial = np.sqrt((track['center'][0] - cand['center'][0]) ** 2 +
                                       (track['center'][1] - cand['center'][1]) ** 2)
                cost_spatial = min(dist_spatial / (max(self.img_w, self.img_h) * 0.3), 1.0)

                # 2. Pose
                cost_pose = self.compute_pose_cost(track['kpts'], cand['kpts'])

                # 3. Visuel
                if anchor_self is not None and cand['features'] is not None:
                    cost_visual_self = cv2.compareHist(anchor_self, cand['features'], cv2.HISTCMP_BHATTACHARYYA)
                else:
                    cost_visual_self = 1.0

                total_cost = (self.W_SPATIAL * cost_spatial) + (self.W_POSE * cost_pose) + (self.W_VISUAL * cost_visual_self)

                # PROTECTION CROISÉE
                if anchor_other is not None and cand['features'] is not None:
                    cost_visual_other = cv2.compareHist(anchor_other, cand['features'], cv2.HISTCMP_BHATTACHARYYA)
                    if cost_visual_other < (cost_visual_self - 0.15):
                        total_cost = 1e5

                if cost_visual_self > 0.75:
                    total_cost = 1e5

                cost_matrix[t_id, c_idx] = total_cost

        return cost_matrix

    def update(self, candidates, frame):
        if not candidates:
            return [None, None]

        # ==========================================================
        # 1. SÉCURISATION DES TYPES
        # ==========================================================
        for c in candidates:
            box = c['box']
            cx = float(box[0] + box[2]) / 2.0
            cy = float(box[1] + box[3]) / 2.0
            c['center'] = (cx, cy)
            c['area'] = float(box[2] - box[0]) * float(box[3] - box[1])
            c['features'] = self.extract_visual_features(frame, c['kpts'])

        # --- INITIALISATION (Frame 1) ---
        if self.tracks[0] is None and self.tracks[1] is None:
            img_center_x = float(self.img_w) / 2.0
            img_center_y = float(self.img_h) / 2.0
            max_dist = float(np.sqrt(img_center_x ** 2 + img_center_y ** 2))

            for c in candidates:
                dist_to_center = float(
                    np.sqrt((c['center'][0] - img_center_x) ** 2 + (c['center'][1] - img_center_y) ** 2))
                score_center = 1.0 - (dist_to_center / max_dist)
                score_area = c['area'] / float(self.img_w * self.img_h)
                c['patient_score'] = (0.6 * score_center) + (0.4 * score_area)

            candidates = sorted(candidates, key=lambda x: x['patient_score'], reverse=True)

            self.tracks[0] = candidates[0]
            self.anchor_features[0] = candidates[0]['features']

            output = [candidates[0], None]

            if len(candidates) > 1:
                self.tracks[1] = candidates[1]
                self.anchor_features[1] = candidates[1]['features']
                output[1] = candidates[1]

            return output

        # --- ASSOCIATION ---
        cost_matrix = self.compute_cost_matrix(candidates, frame)
        track_indices, cand_indices = linear_sum_assignment(cost_matrix)

        current_outputs = {0: None, 1: None}
        matched_candidates = set()  # NOUVEAU: Garde la trace de qui a été assigné

        for t_idx, c_idx in zip(track_indices, cand_indices):
            if cost_matrix[t_idx, c_idx] < 0.9:
                self.tracks[t_idx] = candidates[c_idx]
                current_outputs[t_idx] = candidates[c_idx]
                matched_candidates.add(c_idx)

        # ==========================================================
        # NOUVEAU : RÉCUPÉRATION DES PISTES NON INITIALISÉES
        # Permet de découvrir le Soignant à la Frame 2, 3... ou 100
        # ==========================================================
        unmatched_candidates = [i for i, _ in enumerate(candidates) if i not in matched_candidates]

        for t_id in [0, 1]:
            # Si on a une place libre et quelqu'un qui n'a pas de boîte
            if self.tracks[t_id] is None and len(unmatched_candidates) > 0:
                c_idx = unmatched_candidates.pop(0)  # On prend le premier disponible

                # On l'initialise comme si c'était la première frame pour lui
                self.tracks[t_id] = candidates[c_idx]
                self.anchor_features[t_id] = candidates[c_idx]['features']
                current_outputs[t_id] = candidates[c_idx]

        return [current_outputs[0], current_outputs[1]]