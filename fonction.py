import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt


def clean_clinical_value(val):
    if pd.isna(val): return np.nan

    # Nettoyer les espaces inutiles autour et mettre en majuscule
    s = str(val).strip().upper()

    # --- NOUVELLE LOGIQUE POUR + ET - ISOLÉS ---
    if s == '+':
        return 1.0

    if s == '-':
        return 0.0
    # -------------------------------------------

    # Si la cellule est l'une de ces valeurs textuelles (sans le + et -), on met NaN
    if s in ['NT', 'NA', 'N/A', 'ND', '', 'SY', 'VA']:
        return np.nan

    # Gérer les scores avec un '+' (ex: "3+")
    if s.endswith('+'):
        try:
            return float(s[:-1]) + 0.33
        except ValueError:
            return np.nan

    # Gérer les scores avec un '-' (ex: "3-")
    if s.endswith('-'):
        try:
            return float(s[:-1]) - 0.33
        except ValueError:
            return np.nan

    # Gérer les nombres normaux
    try:
        return float(s)
    except ValueError:
        return np.nan

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any(): return np.nan
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def angle_with_vertical(p_sup, p_inf):
    vector = np.array(p_inf) - np.array(p_sup)
    vertical = np.array([0, 1])
    mag_v = np.linalg.norm(vector)
    if mag_v == 0 or np.isnan(mag_v): return np.nan
    cosine_angle = np.dot(vector, vertical) / mag_v
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def fill_nans_1d(arr):
    arr = np.array(arr)
    mask = ~np.isnan(arr)
    if not mask.any(): return arr
    indices = np.arange(len(arr))
    return np.interp(indices, indices[mask], arr[mask])


def butter_lowpass_filter(data, cutoff, fs, order=4):
    if len(data) <= 15: return data
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def calculate_lean(point_sup, point_inf):
    point_sup, point_inf = np.array(point_sup)[:2], np.array(point_inf)[:2]
    if np.isnan(point_sup).any() or np.isnan(point_inf).any(): return np.nan
    trunk_vec, vertical_vec = point_sup - point_inf, np.array([0, -1])
    norm_trunk, norm_vert = np.linalg.norm(trunk_vec), np.linalg.norm(vertical_vec)
    if norm_trunk == 0 or norm_vert == 0: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(trunk_vec, vertical_vec) / (norm_trunk * norm_vert), -1.0, 1.0)))


def get_dynamic_col(df, col_name):
    """
    Extrait une colonne du DataFrame de manière sécurisée.
    Convertit automatiquement les valeurs en nombres et gère les colonnes manquantes.
    """
    # 1. Vérifie si la colonne existe bien dans la base de données
    if col_name in df.columns:
        # On force la conversion en numérique (transforme les erreurs de frappe ou espaces en NaN)
        return pd.to_numeric(df[col_name], errors='coerce')
    else:
        # Si la colonne n'existe pas, on alerte sans faire planter le script
        print(f"⚠️ Attention : La colonne '{col_name}' est introuvable. Remplacée par des NaN.")
        return pd.Series(np.nan, index=df.index)


def calculate_zscore(series, invert=False):
    """
    Standardise une variable continue (calcule le Z-score).
    Un score positif (>0) indique toujours que le patient est plus atteint que la moyenne de la cohorte.

    Paramètres :
    - series : La colonne Pandas contenant les données (ex: angles).
    - invert (bool) :
        * False = Un grand chiffre est pathologique (ex: Test de Thomas positif).
        * True = Un petit chiffre est pathologique (ex: Manque de dorsiflexion de cheville).
    """
    # Calcule l'écart à la moyenne, divisé par l'écart-type
    z_score = (series - series.mean()) / series.std()

    # Inverse le signe si le sens clinique de la mesure l'exige
    if invert:
        z_score = -z_score

    return z_score


def calculate_angle_0_is_straight(p_hanche, p_genou, p_cheville):
    """
    Calcule l'angle de flexion (0° = jambe parfaitement tendue).
    p1 = Hanche, p2 = Genou (sommet), p3 = Cheville
    """
    h = np.array(p_hanche)
    g = np.array(p_genou)
    c = np.array(p_cheville)

    # Vecteur Hanche -> Genou (au lieu de Genou -> Hanche)
    hg = g - h
    # Vecteur Genou -> Cheville
    gc = c - g

    # Produit scalaire
    cosine_angle = np.dot(hg, gc) / (np.linalg.norm(hg) * np.linalg.norm(gc))

    # Clip pour éviter les erreurs d'arrondi de numpy et calcul de l'angle
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle


def calculate_lean_0_is_straight(p_top, p_bot):
    """
    Calcule l'angle d'un segment (0° = segment parfaitement vertical).
    En coordonnées vidéo (Y vers le bas), un segment vertical va de p_bot vers p_top (vers le haut),
    donc son vecteur Y est négatif. On le compare au vecteur purement vertical [0, -1].
    """
    # Vecteur allant du bas vers le haut (ex: Hanche -> Epaule, ou Cheville -> Genou)
    v = np.array(p_top) - np.array(p_bot)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.nan

    # On compare avec le vecteur vertical absolu [0, -1] (pointant vers le haut de l'image)
    vertical_ref = np.array([0, -1])

    cosine_angle = np.dot(v, vertical_ref) / norm
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle


def calculate_valgus_varus_frontal(hip, knee, ankle, side):
    """
    Calcule l'angle frontal du genou.
    0° = Jambe parfaitement alignée
    Valgus = Valeur Positive (+)
    Varus = Valeur Négative (-)
    """
    # Vecteur Fémur (Hanche -> Genou)
    v1 = np.array([knee[0] - hip[0], knee[1] - hip[1]])

    # Vecteur Tibia (Genou -> Cheville)
    v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])

    # Produit scalaire et produit vectoriel (déterminant 2D)
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    # Calcul de l'angle signé
    angle_rad = np.arctan2(cross_product, dot_product)
    angle_deg = np.degrees(angle_rad)

    # Ajustement anatomique : Valgus toujours positif
    if side == 'gauche':
        angle_deg = -angle_deg

    return angle_deg