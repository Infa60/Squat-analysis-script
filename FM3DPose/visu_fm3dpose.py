import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

# --- CONFIGURATION ---
POSE_FILE = r"output_poses_3d.npy"

# --- COULEURS ---
BONES_COLORED = [
    ((0, 1), 'darkblue'), ((1, 2), 'darkblue'), ((2, 3), 'darkblue'),
    ((0, 4), 'darkred'), ((4, 5), 'darkred'), ((5, 6), 'darkred'),
    ((0, 7), 'green'), ((7, 8), 'green'), ((8, 9), 'green'), ((9, 10), 'green'),
    ((8, 11), 'lightcoral'), ((11, 12), 'lightcoral'), ((12, 13), 'lightcoral'),
    ((8, 14), 'skyblue'), ((14, 15), 'skyblue'), ((15, 16), 'skyblue')
]

JOINT_COLORS = [
    'green',
    'darkblue', 'darkblue', 'darkblue',
    'darkred', 'darkred', 'darkred',
    'green', 'green', 'green', 'green',
    'lightcoral', 'lightcoral', 'lightcoral',
    'skyblue', 'skyblue', 'skyblue'
]


# --- NOUVEAU : LE FILTRE BIOMÉCANIQUE ---
def lisser_trajectoires_3d(poses_3d, window_length=11, polyorder=3):
    """
    Applique un filtre de Savitzky-Golay sur l'axe du temps (axis=0)
    pour éliminer les tremblements (jittering) sans écraser le mouvement.
    """
    nb_frames = poses_3d.shape[0]

    # Sécurité : la fenêtre temporelle doit être impaire et plus petite que la vidéo
    if nb_frames < window_length:
        window_length = nb_frames if nb_frames % 2 != 0 else nb_frames - 1

    if window_length <= polyorder:
        print("⚠️ Vidéo trop courte pour le filtre, affichage brut.")
        return poses_3d

    print(f"Application du filtre Savitzky-Golay (fenêtre: {window_length}, ordre: {polyorder})...")
    # On applique le filtre le long de l'axe 0 (le temps) pour toutes les coordonnées
    poses_filtrees = savgol_filter(poses_3d, window_length, polyorder, axis=0)
    return poses_filtrees


# ----------------------------------------

def main():
    print(f"Chargement des données depuis {POSE_FILE}...")

    try:
        poses_3d = np.load(POSE_FILE, allow_pickle=True)
    except FileNotFoundError:
        print("Erreur : Fichier introuvable.")
        return

    # --- APPLICATION DU FILTRE ---
    poses_3d = lisser_trajectoires_3d(poses_3d)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- SYSTÈME DE PAUSE ---
    anim_running = True

    def on_press(event):
        nonlocal anim_running
        if event.key == ' ':
            if anim_running:
                ani.pause()
                ax.set_title("⏸️ EN PAUSE - Tournez avec la souris")
                fig.canvas.draw()
            else:
                ani.resume()
            anim_running = not anim_running

    fig.canvas.mpl_connect('key_press_event', on_press)

    def update(frame_idx):
        ax.clear()

        pose = poses_3d[frame_idx]

        radius = 1.5
        ax.set_xlim3d([-radius, radius])
        ax.set_ylim3d([-radius, radius])
        ax.set_zlim3d([-radius, radius])

        ax.set_xlabel('Largeur (X)')
        ax.set_ylabel('Profondeur (Z)')
        ax.set_zlabel('Hauteur (Y)')

        if anim_running:
            ax.set_title(f"▶️ Visualisation 3D LISSÉE - Frame {frame_idx} (Espace pour pause)")

        ax.view_init(elev=0, azim=0)

        if np.all(pose == 0):
            return

        # Redressement anatomique (IA -> Matplotlib)
        x_pts = pose[:, 0]
        y_pts = pose[:, 2]
        z_pts = -pose[:, 1]

        ax.scatter(x_pts, y_pts, z_pts, c=JOINT_COLORS, s=30, depthshade=True)

        for bone, color in BONES_COLORED:
            p1, p2 = bone
            x_line = [pose[p1, 0], pose[p2, 0]]
            y_line = [pose[p1, 2], pose[p2, 2]]
            z_line = [-pose[p1, 1], -pose[p2, 1]]
            ax.plot(x_line, y_line, z_line, c=color, linewidth=2.5)

    print("Génération de l'animation... Appuyez sur ESPACE pour faire pause.")
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=poses_3d.shape[0],
        interval=33,
        repeat=True
    )

    plt.show()


if __name__ == "__main__":
    main()