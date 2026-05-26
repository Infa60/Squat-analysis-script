import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pingouin as pg

# --- CONFIGURATION DES CHEMINS ---
excel_calc_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_Angles_Max_rotation_matrice.xlsx"
master_pkl_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Master_Database_Patient.pkl"
output_comparison_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Comparison_Results.xlsx"
output_plots_dir = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Results\Comparison_markerless_c3d"

# Sécurité : Crée le dossier s'il n'existe pas encore
os.makedirs(output_plots_dir, exist_ok=True)

# --- 1. CHARGEMENT DES DONNÉES ---
print("Chargement des bases de données...")
df_calc = pd.read_excel(excel_calc_path)
df_master = pd.read_pickle(master_pkl_path)

# Ajustement de l'angle du genou
df_master['Knee_Flexion_Max'] = 180 - df_master['Knee_Flexion_Max']


# --- 2. EXTRACTION DE L'ID COMMUN ---
def format_excel_id(filename):
    if pd.isna(filename):
        return None
    return os.path.splitext(str(filename))[0]


def format_pkl_id(filename):
    if pd.isna(filename):
        return None
    base_name = os.path.splitext(str(filename))[0]
    return base_name.replace('_SView', '')


df_calc['ID_Match'] = df_calc['Fichier'].apply(format_excel_id)
if 'File_Sagittal' in df_master.columns:
    df_master['ID_Match'] = df_master['File_Sagittal'].apply(format_pkl_id)
else:
    print("⚠️ La colonne 'File_Sagittal' est introuvable. Vérifiez votre fichier PKL.")

# --- 3. FUSION DES DONNÉES ---
cols_master = ['ID_Match', 'Knee_Flexion_Max', 'Trunk_Lean_Max', 'Tibia_Lean_Max']
if 'File_Sagittal' in df_master.columns:
    cols_master.append('File_Sagittal')

df_master_subset = df_master[cols_master]
df_merged = pd.merge(df_calc, df_master_subset, on='ID_Match', how='inner')

# --- DIAGNOSTIC ET SÉCURITÉ ---
df_merged['ID_Match'] = df_merged['ID_Match'].astype(str)

doublons = df_merged[df_merged.duplicated(subset=['ID_Match'], keep=False)]
if not doublons.empty:
    print(f"\n⚠️ ANOMALIE DÉTECTÉE : {len(doublons)} lignes se chevauchent sur les mêmes ID_Match.")
    print("-> Le script va conserver uniquement la première occurrence de chaque ID pour continuer le calcul.")

df_merged = df_merged.drop_duplicates(subset=['ID_Match'])

cols_to_check = [
    'Max_Flexion_Genou (deg)', 'Knee_Flexion_Max',
    'Max_Angle_Tronc_Vert (deg)', 'Trunk_Lean_Max',
    'Max_Angle_Tibia_Vert (deg)', 'Tibia_Lean_Max'
]
df_merged = df_merged.dropna(subset=cols_to_check)

# --- 4. CALCUL DES DIFFÉRENCES ---
df_merged['Diff_Genou (deg)'] = np.abs(df_merged['Max_Flexion_Genou (deg)'] - df_merged['Knee_Flexion_Max'])
df_merged['Diff_Tronc (deg)'] = np.abs(df_merged['Max_Angle_Tronc_Vert (deg)'] - df_merged['Trunk_Lean_Max'])
df_merged['Diff_Tibia (deg)'] = np.abs(df_merged['Max_Angle_Tibia_Vert (deg)'] - df_merged['Tibia_Lean_Max'])

# --- 5. RÉSULTATS ET EXPORTATION ---
print(f"\n✅ {len(df_merged)} patients uniques et complets validés pour la comparaison.")

if len(df_merged) > 0:
    print("\n--- Aperçu des erreurs moyennes absolues ---")
    print(f"Différence Genou : {df_merged['Diff_Genou (deg)'].mean():.2f}°")
    print(f"Différence Tronc : {df_merged['Diff_Tronc (deg)'].mean():.2f}°")
    print(f"Différence Tibia : {df_merged['Diff_Tibia (deg)'].mean():.2f}°")

    df_merged.to_excel(output_comparison_path, index=False)

    # --- 6. CALCUL DE L'ICC ET STOCKAGE ---
    print("\n--- Analyse de Concordance (ICC) ---")

    comparisons = {
        'Genou': ('Max_Flexion_Genou (deg)', 'Knee_Flexion_Max'),
        'Tronc': ('Max_Angle_Tronc_Vert (deg)', 'Trunk_Lean_Max'),
        'Tibia': ('Max_Angle_Tibia_Vert (deg)', 'Tibia_Lean_Max')
    }

    # NOUVEAU : Un dictionnaire pour mémoriser les ICC pour les graphiques
    icc_dict = {}

    for name, (col1, col2) in comparisons.items():
        df_merged[col1] = pd.to_numeric(df_merged[col1], errors='coerce')
        df_merged[col2] = pd.to_numeric(df_merged[col2], errors='coerce')

        df_long = pd.melt(df_merged, id_vars=['ID_Match'], value_vars=[col1, col2],
                          var_name='Outil', value_name='Angle').dropna(subset=['Angle'])

        try:
            icc_results = pg.intraclass_corr(data=df_long, targets='ID_Match', raters='Outil', ratings='Angle')
            icc_row = icc_results.loc[icc_results['Type'] == 'ICC(A,1)']

            if not icc_row.empty:
                icc2_val = icc_row['ICC'].values[0]
                ci_95 = icc_row['CI95'].values[0]
                # On stocke la valeur en mémoire
                icc_dict[name] = icc2_val
                print(f"ICC2 ({name}) : {icc2_val:.3f} (IC 95%: {ci_95})")
            else:
                icc_dict[name] = None
                print(f"⚠️ ICC ({name}) : Introuvable.")

        except Exception as e:
            icc_dict[name] = None
            print(f"⚠️ Impossible de calculer l'ICC pour {name}. Erreur : {e}")

    # --- 7. GRAPHIQUE COMBINÉ : CORRÉLATION & BLAND-ALTMAN ---
    print("\nGénération et sauvegarde du graphique combiné avec les ICC...")

    plots_data = [
        ('Genou', df_merged['Max_Flexion_Genou (deg)'], df_merged['Knee_Flexion_Max']),
        ('Tronc', df_merged['Max_Angle_Tronc_Vert (deg)'], df_merged['Trunk_Lean_Max']),
        ('Tibia', df_merged['Max_Angle_Tibia_Vert (deg)'], df_merged['Tibia_Lean_Max'])
    ]


    # NOUVEAU : On ajoute icc_val comme paramètre à la fonction
    def draw_correlation(ax, name, data1, data2, icc_val):
        d1, d2 = np.asarray(data1), np.asarray(data2)
        r_val = np.corrcoef(d2, d1)[0, 1]

        ax.scatter(d2, d1, alpha=0.6, color='green', edgecolors='k')

        min_val = min(np.min(d1), np.min(d2)) - 5
        max_val = max(np.max(d1), np.max(d2)) + 5
        ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Identité parfaite (x=y)')

        m, b = np.polyfit(d2, d1, 1)
        ax.plot(d2, m * d2 + b, color='red', label=f'Régression linéaire (r={r_val:.2f})')

        # NOUVEAU : Intégration de l'ICC dans le titre si le calcul a réussi
        if icc_val is not None:
            ax.set_title(f'Corrélation : {name} (ICC = {icc_val:.2f})', fontweight='bold')
        else:
            ax.set_title(f'Corrélation : {name}')

        ax.set_xlabel('Qualisys(°)')
        ax.set_ylabel('Markerless VitPose (°)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)


    def draw_bland_altman(ax, name, data1, data2):
        d1, d2 = np.asarray(data1), np.asarray(data2)
        mean = np.mean([d1, d2], axis=0)
        diff = d1 - d2
        md = np.mean(diff)
        sd = np.std(diff, axis=0)

        ax.scatter(mean, diff, alpha=0.6, color='blue', edgecolors='k')
        ax.axhline(md, color='red', linestyle='-', linewidth=2, label=f'Biais ({md:.1f}°)')
        ax.axhline(md + 1.96 * sd, color='gray', linestyle='--', label=f'LOA sup ({md + 1.96 * sd:.1f}°)')
        ax.axhline(md - 1.96 * sd, color='gray', linestyle='--', label=f'LOA inf ({md - 1.96 * sd:.1f}°)')

        ax.set_title(f'Bland-Altman : {name}')
        ax.set_xlabel('Moyenne des deux méthodes (°)')
        ax.set_ylabel('Différence (Markerless - C3D) (°)')
        ax.legend(loc='upper right')


    fig_comb, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (name, data1, data2) in enumerate(plots_data):
        # On récupère l'ICC depuis le dictionnaire créé à l'étape 6
        icc_actuel = icc_dict.get(name)

        draw_correlation(axes[0, i], name, data1, data2, icc_actuel)  # Ligne du haut
        draw_bland_altman(axes[1, i], name, data1, data2)  # Ligne du bas

    plt.tight_layout()
    combined_file_path = os.path.join(output_plots_dir, 'Analyse_Complete_Combined_rotation_matrice.png')
    plt.savefig(combined_file_path, dpi=300, bbox_inches='tight')
    print(f" 💾 Sauvegardé (Combiné) : {combined_file_path}")

    # plt.show()

else:
    print("\n⚠️ Aucune correspondance valide trouvée après le nettoyage. Le script s'arrête.")