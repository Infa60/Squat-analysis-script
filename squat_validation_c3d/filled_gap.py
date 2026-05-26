import numpy as np
from scipy.spatial.distance import cdist
import warnings
import pandas as pd

def pca_econ(data):
    """Équivalent de [U, S, PC] = svd(Y, 'econ') en MATLAB."""
    N, M = data.shape
    Y = data / np.sqrt(N - 1)
    # Dans numpy SVD: data = U @ np.diag(S) @ Vh
    # Le V de MATLAB (PC vectors) correspond à la transposée de Vh en Python
    U, S, Vh = np.linalg.svd(Y, full_matrices=False)
    PC = Vh.T
    sqrtEV = S
    return PC, sqrtEV


def distance2marker(marker_data, cols_with_gaps):
    """Calcule la distance moyenne entre les marqueurs troués et les autres."""
    frames, cols = marker_data.shape
    n_markers = cols // 3

    # On trouve les indices des marqueurs (0, 1, 2...) correspondant aux colonnes
    markers_with_gaps = np.unique(cols_with_gaps // 3)
    n_gapped = len(markers_with_gaps)

    # Reshape en (frames, n_markers, 3)
    reshaped_data = marker_data.reshape((frames, n_markers, 3))

    dist_array = np.full((n_gapped, n_markers, frames), np.nan)

    for i in range(frames):
        frame_data = reshaped_data[i, :, :]  # Forme: (n_markers, 3)
        gapped_coords = frame_data[markers_with_gaps, :]  # Forme: (n_gapped, 3)

        # cdist calcule la distance euclidienne entre deux ensembles de points
        dist_array[:, :, i] = cdist(gapped_coords, frame_data, metric='euclidean')

    # Moyenne sur la dimension du temps (en ignorant les NaNs)
    return np.nanmean(dist_array, axis=2)


def predict_missing_markers(data_gaps, algorithm=2, weight_scale=200.0,
                            mm_weight=0.02, distal_threshold=0.5, min_cum_sv=0.99):
    data = np.copy(data_gaps)
    frames, columns = data.shape
    n_markers = columns // 3

    # Détection des trous
    cols_with_gaps = np.where(np.isnan(data).any(axis=0))[0]
    markers_with_gaps = np.unique(cols_with_gaps // 3)
    frames_with_gaps = np.where(np.isnan(data).any(axis=1))[0]

    if len(frames_with_gaps) == 0:
        warnings.warn("Les données ne semblent pas avoir de trous (assurez-vous qu'ils sont représentés par np.nan).")
        return data

    # Séparation des données saines
    cols_without_gaps = np.setdiff1d(np.arange(columns), cols_with_gaps)

    # Calcul et soustraction de la trajectoire moyenne (X, Y, Z séparément)
    mean_traj_x = np.nanmean(data[:, [c for c in cols_without_gaps if c % 3 == 0]], axis=1, keepdims=True)
    mean_traj_y = np.nanmean(data[:, [c for c in cols_without_gaps if c % 3 == 1]], axis=1, keepdims=True)
    mean_traj_z = np.nanmean(data[:, [c for c in cols_without_gaps if c % 3 == 2]], axis=1, keepdims=True)

    data[:, 0::3] -= mean_traj_x
    data[:, 1::3] -= mean_traj_y
    data[:, 2::3] -= mean_traj_z

    gapfilled_data = np.copy(data)

    # Fonction interne de reconstruction
    def reconstruct(data2reconstruct):
        c_gaps = np.where(np.isnan(data2reconstruct).any(axis=0))[0]

        # Vecteur de poids spatial
        weight_matrix = distance2marker(data, c_gaps)
        if weight_matrix.shape[0] > 1:
            weight_vector = np.min(weight_matrix, axis=0)
        else:
            weight_vector = weight_matrix.flatten()

        weight_vector = np.exp(-(weight_vector ** 2) / (2 * weight_scale ** 2))

        # Application du poids minimum au marqueur manquant lui-même
        m_gaps_idx = np.unique(c_gaps // 3)
        weight_vector[m_gaps_idx] = mm_weight

        # Expansion du vecteur de poids pour couvrir [X, Y, Z]
        weight_vector_expanded = np.repeat(weight_vector, 3)

        M = np.copy(data2reconstruct)
        M_zeros = np.copy(M)
        M_zeros[:, c_gaps] = 0

        # Trames sans aucun trou
        N_no_gaps = data2reconstruct[~np.isnan(M).any(axis=1), :]
        N_zeros = np.copy(N_no_gaps)
        N_zeros[:, c_gaps] = 0

        # Normalisation
        mean_N_no_gaps = np.mean(N_no_gaps, axis=0)
        mean_N_zeros = np.mean(N_zeros, axis=0)
        stdev_N_no_gaps = np.std(N_no_gaps, axis=0, ddof=0)
        stdev_N_no_gaps[stdev_N_no_gaps == 0] = 1.0  # Éviter la division par zéro

        # Application de la normalisation et des poids
        M_zeros = (M_zeros - mean_N_zeros) / stdev_N_no_gaps * weight_vector_expanded
        N_no_gaps = (N_no_gaps - mean_N_no_gaps) / stdev_N_no_gaps * weight_vector_expanded
        N_zeros = (N_zeros - mean_N_zeros) / stdev_N_no_gaps * weight_vector_expanded

        # PCA
        PC_no_gaps, sqrtEV_no_gaps = pca_econ(N_no_gaps)
        PC_zeros, sqrtEV_zeros = pca_econ(N_zeros)

        # Sélection du nombre de vecteurs propres (MinCumSV)
        cum_sv_no_gaps = np.cumsum(sqrtEV_no_gaps) / np.sum(sqrtEV_no_gaps)
        cum_sv_zeros = np.cumsum(sqrtEV_zeros) / np.sum(sqrtEV_zeros)

        n_eig_no_gaps = np.argmax(cum_sv_no_gaps >= min_cum_sv) + 1
        n_eig_zeros = np.argmax(cum_sv_zeros >= min_cum_sv) + 1
        n_eig = max(n_eig_no_gaps, n_eig_zeros)

        PC_no_gaps = PC_no_gaps[:, :n_eig]
        PC_zeros = PC_zeros[:, :n_eig]

        # Matrice de transformation (Equation 1 Federolf)
        T = PC_no_gaps.T @ PC_zeros

        # Reconstruction
        reconstructed = M_zeros @ PC_zeros @ T @ PC_no_gaps.T

        # Inverser la normalisation
        reconstructed = mean_N_no_gaps + (reconstructed * stdev_N_no_gaps) / weight_vector_expanded

        res = np.copy(data2reconstruct)
        for j in c_gaps:
            res[:, j] = reconstructed[:, j]

        return res

    # Choix de la stratégie
    if algorithm == 1:
        reconstructed_full = reconstruct(data)
        gaps_idx = np.isnan(data)
        gapfilled_data[gaps_idx] = reconstructed_full[gaps_idx]

    elif algorithm == 2:
        for i in markers_with_gaps:
            # Indices des colonnes XYZ pour le marqueur i
            idx_i = slice(i * 3, i * 3 + 3)

            # Distance par rapport au marqueur i
            dist2markers = distance2marker(data, np.array([i * 3]))[0]
            thresh = distal_threshold * np.mean(dist2markers)

            # Identifier les colonnes à mettre à zéro (trop lointaines)
            cols2zero = []
            for m in range(n_markers):
                if dist2markers[m] > thresh and np.isnan(data[:, m * 3:m * 3 + 3]).any():
                    cols2zero.extend([m * 3, m * 3 + 1, m * 3 + 2])

            data_removed_cols = np.copy(data)
            data_removed_cols[:, cols2zero] = 0
            data_removed_cols[:, idx_i] = data[:, idx_i]

            # Retirer les trous superposés temporels
            gapped_frames_i = np.where(np.isnan(data[:, i * 3]))[0]
            for ii in markers_with_gaps:
                if ii != i:
                    if np.isnan(data_removed_cols[gapped_frames_i, ii * 3]).any():
                        data_removed_cols[:, ii * 3:ii * 3 + 3] = 0

            # Identifier les trames pour la reconstruction
            frames_no_gaps = np.where(~np.isnan(data_removed_cols).any(axis=1))[0]
            frames2rec = np.where(np.isnan(data[:, i * 3]))[0]

            complete_and_gapped = np.concatenate((frames_no_gaps, frames2rec))
            fill_frames = np.arange(len(frames_no_gaps), len(complete_and_gapped))

            temp_reconstructed = reconstruct(data_removed_cols[complete_and_gapped, :])
            gapfilled_data[frames2rec, idx_i] = temp_reconstructed[fill_frames, idx_i]

    else:
        raise ValueError("L'algorithme doit être 1 ou 2.")

    # Restituer la trajectoire moyenne globale
    gapfilled_data[:, 0::3] += mean_traj_x
    gapfilled_data[:, 1::3] += mean_traj_y
    gapfilled_data[:, 2::3] += mean_traj_z

    return gapfilled_data


def interpolate_small_gaps(data_for_gapfill, max_gap_size=5):
    """
    Remplit uniquement les petits trous par interpolation linéaire temporelle.
    - data_for_gapfill : tableau 2D (frames, marqueurs*3) contenant des np.nan
    - max_gap_size : nombre de frames consécutives maximum à boucher.
                     (Ex: 5 frames à 100Hz = 0.05 secondes).
    """
    # Conversion en DataFrame Pandas car sa fonction d'interpolation est très puissante
    df = pd.DataFrame(data_for_gapfill)

    # Interpolation linéaire colonne par colonne.
    # limit=max_gap_size : ne bouche pas les trous plus grands que cette taille.
    # limit_area='inside' : n'extrapole pas si le trou est au tout début ou à la toute fin du fichier.
    df_interpolated = df.interpolate(method='linear', limit=max_gap_size, limit_area='inside')

    # Retour au format numpy array
    return df_interpolated.to_numpy()