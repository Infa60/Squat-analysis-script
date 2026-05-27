import numpy as np
import scipy.io as sio
import torch

# --- PATCH ANTI-CRASH CPU ---
original_load = torch.load


def safe_cpu_load(*args, **kwargs):
    kwargs['map_location'] = torch.device('cpu')
    return original_load(*args, **kwargs)


torch.load = safe_cpu_load
# ----------------------------

from fmpose3d import FMPose3DInference

# --- CONFIGURATION ---
FICHIER_MAT_IN = "vos_poses_vitpose.mat"
POSE_OUT = "output_poses_3d_depuis_mat.npy"

# Dimension de la vidéo d'origine (Nécessaire pour les calculs de focale de la caméra 3D)
TAILLE_IMAGE = (1920, 1080)  # <-- À MODIFIER SELON VOTRE VIDÉO ORIGINALE


# --- TRADUCTEUR ViTPose (COCO) VERS FMPOSE3D (Human3.6M) ---
def convertir_coco_vers_h36m(points_coco):
    h36m = np.zeros((17, 2), dtype=np.float32)
    h36m[0] = (points_coco[11] + points_coco[12]) / 2.0  # Bassin
    h36m[1] = points_coco[12]  # Hanche D
    h36m[2] = points_coco[14]  # Genou D
    h36m[3] = points_coco[16]  # Cheville D
    h36m[4] = points_coco[11]  # Hanche G
    h36m[5] = points_coco[13]  # Genou G
    h36m[6] = points_coco[15]  # Cheville G
    h36m[8] = (points_coco[5] + points_coco[6]) / 2.0  # Thorax
    h36m[7] = (h36m[0] + h36m[8]) / 2.0  # Colonne
    h36m[9] = points_coco[0]  # Tête/Cou
    h36m[10] = (points_coco[1] + points_coco[2]) / 2.0  # Sommet Tête
    h36m[11] = points_coco[5]  # Épaule G
    h36m[12] = points_coco[7]  # Coude G
    h36m[13] = points_coco[9]  # Poignet G
    h36m[14] = points_coco[6]  # Épaule D
    h36m[15] = points_coco[8]  # Coude D
    h36m[16] = points_coco[10]  # Poignet D
    return h36m


def main():
    print(f"Ouverture du fichier {FICHIER_MAT_IN}...")

    try:
        donnees_mat = sio.loadmat(FICHIER_MAT_IN)
    except FileNotFoundError:
        print("❌ Fichier .mat introuvable.")
        return

    # ⚠️ ATTENTION ICI : Le nom de la variable dépend de comment le fichier a été sauvegardé
    # Par convention, on essaie 'keypoints', 'poses', ou 'data'
    cles_possibles = ['keypoints', 'poses', 'data', 'pred_instances']
    cle_trouvee = None

    for cle in cles_possibles:
        if cle in donnees_mat:
            cle_trouvee = cle
            break

    if not cle_trouvee:
        print(f"❌ Impossible de trouver les coordonnées. Clés disponibles : {donnees_mat.keys()}")
        return

    # On extrait le tableau (Format attendu : [Frames, 17, 2] ou [Frames, 17, 3] avec la confiance)
    keypoints_2d_bruts = donnees_mat[cle_trouvee]
    nb_frames = keypoints_2d_bruts.shape[0]
    print(f"Données 2D chargées : {nb_frames} frames trouvées.")

    print("Initialisation de FMPose3D...")
    fm3d_model = FMPose3DInference()
    poses_3d_list = []

    print("Calcul de la 3D en cours...")
    for frame_idx in range(nb_frames):
        if frame_idx % 30 == 0:
            print(f"Progression : {frame_idx}/{nb_frames}...")

        # Extraction des points de la frame (on ne garde que X et Y, on ignore le score de confiance s'il y en a un)
        points_frame = keypoints_2d_bruts[frame_idx, :, :2]

        # Si les points sont des zéros (pas de détection)
        if np.all(points_frame == 0):
            poses_3d_list.append(np.zeros((17, 3)))
            continue

        # Traduction anatomique
        keypoints_h36m = convertir_coco_vers_h36m(points_frame)
        keypoints_2d_traduits = np.expand_dims(keypoints_h36m, axis=0)

        try:
            resultat_3d = fm3d_model.pose_3d(
                keypoints_2d=keypoints_2d_traduits,
                image_size=TAILLE_IMAGE
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
            print(f"Erreur à la frame {frame_idx}: {e}")
            poses_3d_list.append(np.zeros((17, 3)))

    # Sauvegarde
    final_3d_poses = np.array(poses_3d_list, dtype=np.float32)
    np.save(POSE_OUT, final_3d_poses)
    print(f"✅ Succès ! {nb_frames} frames converties en 3D dans {POSE_OUT}.")


if __name__ == "__main__":
    main()