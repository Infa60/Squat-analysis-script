import os
import glob
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
output_c3d_folder = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_c3d"
results_excel_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_Angles_Max_rotation_matrice.xlsx"
anthrop_excel_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Participant\examen_clinique_results_with_scores_data.xlsx"

# NOUVEAU : Choix de la méthode de calcul des angles
# 'GLOBAL_2D'            = Plan sagittal du labo (Y, Z purs, ancienne méthode sujette à parallaxe)
# 'LOCAL_PROJECTED'      = Plan sagittal du patient (tourne avec le patient, méthode robuste 1 dDL)
# '3D_ROTATION_MATRICES' = Matrices de rotation 3D (Gold standard ISB, 3 dDL)
CALCULATION_METHOD = '3D_ROTATION_MATRICES'

max_gap_size = 5
interpolation_method = 'spline'
spline_order = 3

# Noms exacts de vos marqueurs
M_RANK = "RANK"
M_RTIB = "RTIB"
M_RSHO = "RSHO"
M_RKNE = "RKNE"
M_RASI = "RASI"
M_LASI = "LASI"
M_RPSI = "RPSI"
M_LPSI = "LPSI"
M_RTHI = "RTHI"


# --- FONCTIONS UTILITAIRES ---
def calc_angle_vectors(v1, v2):
    """Calcule l'angle en degrés entre deux séries de vecteurs (shape: 2 ou 3, N_frames)"""
    norm1 = np.linalg.norm(v1, axis=0)
    norm2 = np.linalg.norm(v2, axis=0)

    if v2.ndim == 1:
        dot_product = np.sum(v1 * v2[:, np.newaxis], axis=0)
    else:
        dot_product = np.sum(v1 * v2, axis=0)

    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta) * (180.0 / np.pi)


def project_vector_onto_plane(vec, normal):
    """Projette un vecteur 3D sur un plan défini par son vecteur normal."""
    dot_v_n = np.sum(vec * normal, axis=0)
    dot_n_n = np.sum(normal * normal, axis=0)
    dot_n_n[dot_n_n == 0] = 1e-10
    k = dot_v_n / dot_n_n
    vec_ortho = k * normal
    return vec - vec_ortho


def clean_id(val):
    v = str(val).strip()
    try:
        return str(int(float(v)))
    except ValueError:
        return v


# --- CHARGEMENT DES DONNÉES CLINIQUES ---
print("Chargement des données anthropométriques...")
try:
    df_anthrop = pd.read_excel(anthrop_excel_path, sheet_name="Visite_ExamenClinique_Anthrop")
    df_anthrop['ID_Match'] = df_anthrop['ID_Visite'].apply(clean_id)
    print("✅ Données cliniques chargées.\n")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    exit()

# --- PIPELINE ---
pkl_files = glob.glob(os.path.join(output_c3d_folder, "*.pkl"))
print(f"{len(pkl_files)} fichiers .pkl trouvés.\n")

all_results = []

for pkl_path in pkl_files:
    filename = os.path.basename(pkl_path)
    parts = filename.split('_')
    if len(parts) < 3:
        continue

    id_visite_brut = parts[1]
    id_recherche = clean_id(id_visite_brut)

    row = df_anthrop[df_anthrop['ID_Match'] == id_recherche]
    if row.empty or pd.isna(row.iloc[0]['LJambeD']):
        continue

    LL = float(row.iloc[0]['LJambeD'])

    with open(pkl_path, 'rb') as f:
        data_dict = pickle.load(f)

    points_3d = data_dict['points_3d']
    labels = data_dict['labels_points']

    if points_3d.size == 0:
        continue

    xyz = points_3d[0:3, :, :].astype(float)
    coords, n_markers, frames = xyz.shape
    is_missing = (xyz[0, :, :] == 0.0) & (xyz[1, :, :] == 0.0) & (xyz[2, :, :] == 0.0)
    xyz[:, is_missing] = np.nan

    xyz_transposed = np.transpose(xyz, (2, 1, 0))
    data_for_gapfill = xyz_transposed.reshape(frames, n_markers * 3)

    if np.isnan(data_for_gapfill).any():
        try:
            df = pd.DataFrame(data_for_gapfill)
            df_filled = df.interpolate(method=interpolation_method, order=spline_order, limit=max_gap_size,
                                       limit_area='inside')
            data_filled = df_filled.to_numpy()
            if np.isnan(data_filled).sum() == np.isnan(data_for_gapfill).sum():
                df_filled = df.interpolate(method='linear', limit=max_gap_size, limit_area='inside')
                data_filled = df_filled.to_numpy()
        except Exception:
            data_filled = data_for_gapfill
    else:
        data_filled = data_for_gapfill

    xyz_filled = np.transpose(data_filled.reshape(frames, n_markers, 3), (2, 1, 0))

    try:
        idx_rank = labels.index(M_RANK)
        idx_rtib = labels.index(M_RTIB)
        idx_rsho = labels.index(M_RSHO)
        idx_rkne = labels.index(M_RKNE)
        idx_rasi = labels.index(M_RASI)
        idx_lasi = labels.index(M_LASI)
        idx_rpsi = labels.index(M_RPSI)
        idx_lpsi = labels.index(M_LPSI)
        idx_rthi = labels.index(M_RTHI)

        # TODO pour les matrices 3D : Ajouter les index des nouveaux marqueurs ici
    except ValueError:
        continue

    pos_rank = xyz_filled[:, idx_rank, :]
    pos_rtib = xyz_filled[:, idx_rtib, :]
    pos_rsho = xyz_filled[:, idx_rsho, :]
    pos_rkne = xyz_filled[:, idx_rkne, :]
    pos_rasi = xyz_filled[:, idx_rasi, :]
    pos_lasi = xyz_filled[:, idx_lasi, :]
    pos_rpsi = xyz_filled[:, idx_rpsi, :]
    pos_lpsi = xyz_filled[:, idx_lpsi, :]
    pos_rthi = xyz_filled[:, idx_rthi, :]
    pos_ij = xyz_filled[:, labels.index("CLAV"), :]
    pos_c7 = xyz_filled[:, labels.index("C7"), :]
    pos_px = xyz_filled[:, labels.index("STRN"), :]
    pos_t8 = xyz_filled[:, labels.index("T10"), :]

    # CALCUL DU HJC (Valable pour toutes les méthodes)
    mid_asis = (pos_rasi + pos_lasi) / 2.0
    pos_hjc = np.zeros_like(mid_asis)
    pos_hjc[0, :] = mid_asis[0, :] + (8.0 + 0.086 * LL)
    pos_hjc[1, :] = mid_asis[1, :] + (11.0 + 0.063 * LL)
    pos_hjc[2, :] = mid_asis[2, :] - (9.0 + 0.078 * LL)

    # =========================================================================
    # --- ROUTAGE SELON LA MÉTHODE CHOISIE ---
    # =========================================================================

    if CALCULATION_METHOD in ['GLOBAL_2D', 'LOCAL_PROJECTED']:

        # 1. Création des vecteurs bruts
        vec_tibia_up = pos_rkne - pos_rank
        vec_trunk = pos_rsho - pos_hjc
        vec_thigh_up = pos_hjc - pos_rkne

        if CALCULATION_METHOD == 'LOCAL_PROJECTED':
            vec_pelvis_normal = pos_lasi - pos_rasi
            GLOBAL_VERTICAL_3D = np.array([[0], [0], [1]])

            vec_tibia_proj = project_vector_onto_plane(vec_tibia_up, vec_pelvis_normal)
            vec_trunk_proj = project_vector_onto_plane(vec_trunk, vec_pelvis_normal)
            vec_thigh_proj = project_vector_onto_plane(vec_thigh_up, vec_pelvis_normal)
            vec_vertical_proj = project_vector_onto_plane(GLOBAL_VERTICAL_3D, vec_pelvis_normal)

        elif CALCULATION_METHOD == 'GLOBAL_2D':
            vec_tibia_proj = vec_tibia_up[1:3, :]
            vec_trunk_proj = vec_trunk[1:3, :]
            vec_thigh_proj = vec_thigh_up[1:3, :]
            vec_vertical_proj = np.array([0, 1])

        # 2. Calcul des angles par produit scalaire
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            angles_tibia_vert = calc_angle_vectors(vec_tibia_proj, vec_vertical_proj)
            max_tibia_vert = np.nanmax(angles_tibia_vert)

            angles_trunk_vert = calc_angle_vectors(vec_trunk_proj, vec_vertical_proj)
            max_trunk_vert = np.nanmax(angles_trunk_vert)

            angles_knee_flex = calc_angle_vectors(vec_thigh_proj, vec_tibia_proj)
            max_knee_flex = np.nanmax(angles_knee_flex)


    elif CALCULATION_METHOD == '3D_ROTATION_MATRICES':

        # ---------------------------------------------------------
        # ÉTAPE 1 : REPÈRE DU BASSIN
        # ---------------------------------------------------------

        vec_z_pelvis = pos_rasi - pos_lasi
        norm_z = np.linalg.norm(vec_z_pelvis, axis=0)
        norm_z[norm_z == 0] = 1e-10
        Z_pelvis = vec_z_pelvis / norm_z

        mid_asis = (pos_rasi + pos_lasi) / 2.0
        mid_psis = (pos_rpsi + pos_lpsi) / 2.0
        temp_x_pelvis = mid_asis - mid_psis

        vec_y_pelvis = np.cross(Z_pelvis, temp_x_pelvis, axis=0)
        norm_y = np.linalg.norm(vec_y_pelvis, axis=0)
        norm_y[norm_y == 0] = 1e-10
        Y_pelvis = vec_y_pelvis / norm_y

        X_pelvis = np.cross(Y_pelvis, Z_pelvis, axis=0)

        R_pelvis = np.stack((X_pelvis.T, Y_pelvis.T, Z_pelvis.T), axis=-1)

        # ---------------------------------------------------------
        # ÉTAPE 2 : REPÈRE DE LA CUISSE
        # ---------------------------------------------------------

        vec_y_femur = pos_hjc - pos_rkne
        norm_y = np.linalg.norm(vec_y_femur, axis=0)
        norm_y[norm_y == 0] = 1e-10
        Y_femur = vec_y_femur / norm_y

        vec_rthi = pos_rthi - pos_rkne

        vec_x_femur = np.cross(vec_rthi, Y_femur, axis=0)
        norm_x = np.linalg.norm(vec_x_femur, axis=0)
        norm_x[norm_x == 0] = 1e-10
        X_femur = vec_x_femur / norm_x

        Z_femur = np.cross(X_femur, Y_femur, axis=0)

        R_femur = np.stack((X_femur.T, Y_femur.T, Z_femur.T), axis=-1)

        # =========================================================
        # ÉTAPE 3 : TIBIA (Avec RTIB comme marqueur LATÉRAL)
        # =========================================================

        vec_y_tibia = pos_rkne - pos_rank
        norm_y = np.linalg.norm(vec_y_tibia, axis=0)
        norm_y[norm_y == 0] = 1e-10
        Y_tibia = vec_y_tibia / norm_y

        temp_lat_tibia = pos_rtib - pos_rank

        vec_x_tibia = np.cross(Y_tibia, temp_lat_tibia, axis=0)
        norm_x = np.linalg.norm(vec_x_tibia, axis=0)
        norm_x[norm_x == 0] = 1e-10
        X_tibia = vec_x_tibia / norm_x

        vec_z_tibia = np.cross(X_tibia, Y_tibia, axis=0)
        norm_z = np.linalg.norm(vec_z_tibia, axis=0)
        norm_z[norm_z == 0] = 1e-10
        Z_tibia = vec_z_tibia / norm_z

        R_tibia = np.stack((X_tibia.T, Y_tibia.T, Z_tibia.T), axis=-1)

        # ---------------------------------------------------------
        # ÉTAPE 4 : REPÈRE DU THORAX / TRONC (Standard ISB)
        # ---------------------------------------------------------

        mid_px_t8 = (pos_px + pos_t8) / 2.0
        mid_ij_c7 = (pos_ij + pos_c7) / 2.0

        vec_y_thorax = mid_ij_c7 - mid_px_t8
        norm_y = np.linalg.norm(vec_y_thorax, axis=0)
        norm_y[norm_y == 0] = 1e-10
        Y_thorax = vec_y_thorax / norm_y

        vec_backwards = pos_c7 - pos_ij

        vec_z_thorax = np.cross(Y_thorax, vec_backwards, axis=0)
        norm_z = np.linalg.norm(vec_z_thorax, axis=0)
        norm_z[norm_z == 0] = 1e-10
        Z_thorax = vec_z_thorax / norm_z

        X_thorax = np.cross(Y_thorax, Z_thorax, axis=0)

        R_thorax = np.stack((X_thorax.T, Y_thorax.T, Z_thorax.T), axis=-1)

        # =========================================================
        # ÉTAPE 5 : CALCUL DES ANGLES FINAUX
        # =========================================================

        GLOBAL_VERTICAL_3D = np.array([[0], [0], [1]])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            Y_thorax_proj = project_vector_onto_plane(Y_thorax, Z_pelvis)
            vertical_proj = project_vector_onto_plane(GLOBAL_VERTICAL_3D, Z_pelvis)

            inclinaison_tronc = calc_angle_vectors(Y_thorax_proj, vertical_proj)
            max_trunk_vert = np.nanmax(inclinaison_tronc)

            Y_tibia_proj = project_vector_onto_plane(Y_tibia, Z_pelvis)

            inclinaison_tibia = calc_angle_vectors(Y_tibia_proj, vertical_proj)
            max_tibia_vert = np.nanmax(inclinaison_tibia)

            R_genou = np.matmul(np.transpose(R_femur, axes=(0, 2, 1)), R_tibia)
            flexion_genou = np.arctan2(-R_genou[:, 0, 1], R_genou[:, 1, 1]) * (180.0 / np.pi)
            #plt.plot(flexion_genou)
            #plt.show()
            max_knee_flex = np.nanmax(flexion_genou)

    else:
        print(f"⚠️ Méthode '{CALCULATION_METHOD}' non reconnue.")
        continue

    # --- ENREGISTREMENT DES RÉSULTATS ---
    all_results.append({
        "Fichier": filename,
        "ID_Visite": id_visite_brut,
        "Méthode_Calcul": CALCULATION_METHOD,
        "Longueur_Jambe (LL)": LL,
        "Max_Angle_Tibia_Vert (deg)": max_tibia_vert,
        "Max_Angle_Tronc_Vert (deg)": max_trunk_vert,
        "Max_Flexion_Genou (deg)": max_knee_flex
    })
    print(
        f"  -> {filename} ({CALCULATION_METHOD}) : Tibia={max_tibia_vert:.1f}°, Tronc={max_trunk_vert:.1f}°, Genou_MaxFlex={max_knee_flex:.1f}°")

# --- EXPORTATION ---
if all_results:
    df_results = pd.DataFrame(all_results)
    df_results.to_excel(results_excel_path, index=False)
    print(f"\n✅ Terminé ! Tous les résultats ont été sauvegardés dans : {results_excel_path}")
else:
    print("\n⚠️ Aucun résultat calculé.")