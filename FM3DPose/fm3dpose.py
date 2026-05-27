import cv2
import numpy as np
from ultralytics import YOLO

# --- PATCH ANTI-CRASH CPU ---
import torch

original_load = torch.load


def safe_cpu_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return original_load(*args, **kwargs)


torch.load = safe_cpu_load
# ----------------------------

from fmpose3d import FMPose3DInference

# --- CONFIGURATION ---
VIDEO_IN = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\CP_qualisys\Sagittal_View\4256_6448_SView_1.avi"
POSE_OUT = "output_poses_3d.npy"


# --- TRADUCTEUR YOLO (COCO) VERS FMPOSE3D (Human3.6M) ---
def convertir_yolo_vers_h36m(points_yolo):
    """
    Réorganise les points 2D de YOLO dans l'ordre attendu par le modèle 3D.
    """
    h36m = np.zeros((17, 2), dtype=np.float32)

    # 0: Bassin (Calculé : Milieu des hanches 11 et 12 de YOLO)
    h36m[0] = (points_yolo[11] + points_yolo[12]) / 2.0

    # 1, 2, 3: Jambe droite (Hanche, Genou, Cheville)
    h36m[1] = points_yolo[12]
    h36m[2] = points_yolo[14]
    h36m[3] = points_yolo[16]

    # 4, 5, 6: Jambe gauche
    h36m[4] = points_yolo[11]
    h36m[5] = points_yolo[13]
    h36m[6] = points_yolo[15]

    # 8: Thorax (Calculé : Milieu des épaules 5 et 6)
    h36m[8] = (points_yolo[5] + points_yolo[6]) / 2.0

    # 7: Colonne vertébrale (Calculé : Milieu entre le bassin et le thorax)
    h36m[7] = (h36m[0] + h36m[8]) / 2.0

    # 9: Cou / Menton (On utilise le nez de YOLO pour approximer la base du visage)
    h36m[9] = points_yolo[0]

    # 10: Sommet de la tête (Calculé : Milieu des yeux 1 et 2)
    h36m[10] = (points_yolo[1] + points_yolo[2]) / 2.0

    # 11, 12, 13: Bras gauche (Épaule, Coude, Poignet)
    h36m[11] = points_yolo[5]
    h36m[12] = points_yolo[7]
    h36m[13] = points_yolo[9]

    # 14, 15, 16: Bras droit
    h36m[14] = points_yolo[6]
    h36m[15] = points_yolo[8]
    h36m[16] = points_yolo[10]

    return h36m


# --------------------------------------------------------

def main():
    print("Initialisation du pipeline Hybride...")

    yolo_model = YOLO("yolov8x-pose.pt")
    fm3d_model = FMPose3DInference()

    print(f"Traitement de la vidéo '{VIDEO_IN}'...")
    cap = cv2.VideoCapture(VIDEO_IN)

    if not cap.isOpened():
        print("❌ ERREUR : Impossible d'ouvrir la vidéo. Vérifiez le chemin.")
        return

    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    taille_image = (largeur, hauteur)

    poses_3d_list = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Extraction : {frame_count} frames traitées...")

        results = yolo_model(frame, verbose=False)

        if len(results[0].keypoints) > 0 and results[0].keypoints.has_visible:
            # On extrait les points de l'unique personne (format 17, 2)
            keypoints_yolo = results[0].keypoints.xy.cpu().numpy()[0]

            # 🪄 TRADUCTION ANATOMIQUE ICI 🪄
            keypoints_h36m = convertir_yolo_vers_h36m(keypoints_yolo)

            # On rajoute la dimension "batch" requise par l'API (forme finale : 1, 17, 2)
            keypoints_2d_traduits = np.expand_dims(keypoints_h36m, axis=0)

            try:
                resultat_3d = fm3d_model.pose_3d(
                    keypoints_2d=keypoints_2d_traduits,
                    image_size=taille_image
                )

                donnees_3d = resultat_3d.poses_3d

                if isinstance(donnees_3d, torch.Tensor):
                    donnees_3d = donnees_3d.cpu().numpy()
                elif not isinstance(donnees_3d, np.ndarray):
                    donnees_3d = np.array(donnees_3d)

                pose_3d_array = np.squeeze(donnees_3d)

                if pose_3d_array.shape == (17, 3):
                    poses_3d_list.append(pose_3d_array)
                else:
                    poses_3d_list.append(np.zeros((17, 3)))

            except Exception as e:
                print(f"Erreur 3D à la frame {frame_count} : {e}")
                poses_3d_list.append(np.zeros((17, 3)))
        else:
            poses_3d_list.append(np.zeros((17, 3)))

    cap.release()

    if len(poses_3d_list) == 0:
        print("⚠️ Aucune frame n'a été traitée.")
        return

    final_3d_poses = np.array(poses_3d_list, dtype=np.float32)

    if final_3d_poses.ndim > 3:
        final_3d_poses = np.squeeze(final_3d_poses)

    np.save(POSE_OUT, final_3d_poses)

    print(f"✅ Succès ! {frame_count} frames analysées.")
    print(f"Poses 3D sauvegardées dans {POSE_OUT}.")


if __name__ == "__main__":
    main()