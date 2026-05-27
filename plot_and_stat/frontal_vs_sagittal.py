"""
=========================================================================================
SYNCHRONIZE VIDS & PLOT FRONTAL KNEE ANGLE - G/D & CORRÉLATION CLINIQUE (LMM / GEE)
=========================================================================================
Objectif: Fusionner les données, extraire les angles frontaux au moment du pic sagittal.
          1. Graphes Boxplots Gauche vs Droite avec statistiques intra-groupes (Modèles Mixtes).
          2. Analyse de corrélation (Angle Affecté Droit vs Clinique Droit).
          3. Comparaison Hémiplégiques vs Diplégiques (LMM).
          4. Export Excel + Génération de Graphiques.
=========================================================================================
"""

# --- CONFIGURATION POUR FORCER L'OUVERTURE DE NOUVELLES FENÊTRES ---
import matplotlib
matplotlib.use('TkAgg') # Ouvre les graphiques dans des fenêtres interactives

from fonction import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
import warnings

# Ignorer les petits avertissements de convergence pour garder la console propre
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# --- CONFIGURATION DES CHEMINS ---
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1"
pkl_sagittal_path = os.path.join(main_path, "Data", "Master_Database_Patient_all.pkl")
pkl_frontal_path = os.path.join(main_path, "Data", "Master_Database_Patient_Frontal_all.pkl")

# Création des dossiers de résultats
output_results_folder = os.path.join(main_path, "Results")
plots_correlations_folder = os.path.join(output_results_folder, "Significant_Correlations_Plots")
os.makedirs(output_results_folder, exist_ok=True)
os.makedirs(plots_correlations_folder, exist_ok=True)

CUTOFF_FREQ = 3

# ==============================================================================
# 1. CHARGEMENT ET FUSION
# ==============================================================================
print("1. Chargement des fichiers bases de données...")
if not os.path.exists(pkl_sagittal_path) or not os.path.exists(pkl_frontal_path):
    raise FileNotFoundError("❌ L'un des fichiers pkl est introuvable.")

df_sag = pd.read_pickle(pkl_sagittal_path).rename(columns={'Video_FPS': 'FPS_sagittal', 'Raw_Keypoints': 'Keypoints_sagittal'})
df_fro = pd.read_pickle(pkl_frontal_path)[['File_Sagittal', 'Video_FPS', 'Raw_Keypoints']].rename(columns={'Video_FPS': 'FPS_frontal', 'Raw_Keypoints': 'Keypoints_frontal'})

master_combined = pd.merge(df_sag, df_fro, on='File_Sagittal', how='inner')

if all(col in master_combined.columns for col in ['Pose estimation', 'Caregiver assistance', 'Hand-to-ground contact', 'Heel_Rise_Binaire']):
    mask_pat = (
        (master_combined['Pose estimation'].astype(str).str.strip().str.upper() == 'Y') &
        (master_combined['Caregiver assistance'].astype(str).str.strip() != '2') &
        (master_combined['Hand-to-ground contact'].astype(str).str.strip() != '2') &
        (master_combined['Heel_Rise_Binaire'] == 0) &
        (master_combined['CoteDiagnostic'] != 'Gauche') & (master_combined['CoteDiagnostic'] != 'Gauche Droit')
    )
    master_combined = master_combined[mask_pat].copy()
    print(f"-> Après filtrage (Uniquement patients Droits), il reste {len(master_combined)} essais cliniquement valides.")

# ==============================================================================
# 2. FONCTIONS MATHÉMATIQUES ET CLINIQUES
# ==============================================================================
def calculate_valgus_varus_frontal(hip, knee, ankle, side):
    v1 = np.array([knee[0] - hip[0], knee[1] - hip[1]])
    v2 = np.array([ankle[0] - knee[0], ankle[1] - knee[1]])
    dot_product = np.dot(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    angle_deg = np.degrees(np.arctan2(cross_product, dot_product))
    if side == 'gauche':
        angle_deg = -angle_deg
    return angle_deg

def get_detailed_diagnosis(diag, cote):
    d = str(diag).lower()
    c = str(cote).lower()
    is_hemi = 'hémi' in d or 'hemi' in d
    is_di = 'di' in d
    has_right = 'droit' in c or 'right' in c
    has_left = 'gauch' in c or 'left' in c

    if is_hemi:
        if has_right: return 'Hémi Droit'
        if has_left: return 'Hémi Gauche'
        return 'Hémi (Inconnu)'
    elif is_di:
        if has_right and has_left:
            if c.find('droit') < c.find('gauch') or c.find('right') < c.find('left'):
                return 'Di Droit > Gauche'
            else: return 'Di Gauche > Droit'
        elif has_right: return 'Di Droit'
        elif has_left: return 'Di Gauche'
        return 'Di (Inconnu)'
    return 'Autre'

# ==============================================================================
# 3. CALCUL ET SYNCHRONISATION
# ==============================================================================
print("2. Recherche du pic sagittal droit et extraction des angles frontaux...")
trials_data = []

for idx, row in master_combined.iterrows():
    try:
        kpts_sag, kpts_fro = row['Keypoints_sagittal'], row['Keypoints_frontal']
        fps_sag, fps_fro = row['FPS_sagittal'], row['FPS_frontal']

        if pd.isna(fps_sag) or pd.isna(fps_fro) or fps_sag <= 0 or fps_fro <= 0: continue
        n_frames_sag, n_frames_fro = kpts_sag.shape[0], kpts_fro.shape[0]

        diag_detail = get_detailed_diagnosis(row.get('Diagnostic'), row.get('CoteDiagnostic'))
        diag_global = 'Hémiplégique' if 'Hémi' in diag_detail else ('Diplégique' if 'Di' in diag_detail else 'Autre')

        # Pic sagittal (Jambe Droite)
        k_f_right = [calculate_angle_0_is_straight(kpts_sag[f, 12], kpts_sag[f, 14], kpts_sag[f, 16]) for f in range(n_frames_sag)]
        k_f_interp = np.interp(np.arange(len(k_f_right)), np.where(~np.isnan(k_f_right))[0], np.array(k_f_right)[~np.isnan(k_f_right)])
        k_f_filt = butter_lowpass_filter(k_f_interp, CUTOFF_FREQ, fps_sag)

        idx_sagittal = np.argmax(k_f_filt)
        time_seconds = idx_sagittal / fps_sag
        idx_frontal = int(round(time_seconds * fps_fro))
        idx_frontal = min(max(idx_frontal, 0), n_frames_fro - 1)

        # Extraction Valgus/Varus Frontal
        hip_front_R, knee_front_R, ankle_front_R = kpts_fro[idx_frontal, 12], kpts_fro[idx_frontal, 14], kpts_fro[idx_frontal, 16]
        angle_droit = calculate_valgus_varus_frontal(hip_front_R, knee_front_R, ankle_front_R, 'droite')

        hip_front_L, knee_front_L, ankle_front_L = kpts_fro[idx_frontal, 11], kpts_fro[idx_frontal, 13], kpts_fro[idx_frontal, 15]
        angle_gauche = calculate_valgus_varus_frontal(hip_front_L, knee_front_L, ankle_front_L, 'gauche')

        # Puisqu'on a filtré pour n'avoir que les patients droits, Angle_Affecte = Angle_Droit
        if 'Droit' in diag_detail:
            angle_aff = angle_droit
        else:
            angle_aff = np.nan

        trials_data.append({
            'Patient_Trial': row['File_Sagittal'],
            'ID_Patient': str(row['File_Sagittal']).split('_')[0],
            'ID_Visite': str(row['File_Sagittal']).split('_')[1],
            'Diagnostic_Detail': diag_detail,
            'Diagnostic_Global': diag_global,
            'Angle_Gauche': angle_gauche,
            'Angle_Droit': angle_droit,
            'Angle_Affecte': angle_aff
        })

    except Exception as e: pass

df_wide = pd.DataFrame(trials_data).dropna(subset=['Angle_Gauche', 'Angle_Droit'])
df_wide = df_wide[~df_wide['Diagnostic_Detail'].str.contains('Inconnu|Autre')]

# --- Récupération des colonnes cliniques pour la corrélation ---
clinical_cols = [c for c in master_combined.columns if c.startswith(('Force_', 'ROM_', 'Spastic_', 'Selectivite_', 'Score_')) and not c.endswith('G')]
df_wide = pd.merge(df_wide, master_combined[['File_Sagittal'] + clinical_cols], left_on='Patient_Trial', right_on='File_Sagittal', how='left')

# Transformation en format long pour le Graphique Gauche/Droite
# AJOUT DE 'ID_Visite' dans les id_vars
df_long = df_wide.melt(
    id_vars=['Patient_Trial', 'ID_Patient', 'ID_Visite', 'Diagnostic_Detail', 'Diagnostic_Global'],
    value_vars=['Angle_Gauche', 'Angle_Droit'],
    var_name='Jambe', value_name='Angle_Frontal'
)
df_long['Jambe'] = df_long['Jambe'].replace({'Angle_Gauche': 'Gauche', 'Angle_Droit': 'Droite'})

ordre_diag = ['Hémi Gauche', 'Hémi Droit', 'Di Gauche > Droit', 'Di Droit > Gauche']
ordre_diag = [d for d in ordre_diag if d in df_wide['Diagnostic_Detail'].unique()]


# ==============================================================================
# 4. ANALYSE DE CORRÉLATION CLINIQUE (LMM) ET EXPORT DES GRAPHIQUES
# ==============================================================================
print("\n" + "="*75)
print(" 📊 CORRÉLATIONS : ANGLE AFFECTÉ vs EXAMEN CLINIQUE (LMM)")
print("="*75)

results_corr = []
sns.set_theme(style="ticks")

for var in clinical_cols:
    valid_df = df_wide.dropna(subset=[var, 'Angle_Affecte', 'ID_Patient', 'ID_Visite']).copy()
    n_patients = valid_df['ID_Patient'].nunique()

    if n_patients >= 3:
        try:
            # Essai avec le Modèle Linéaire Mixte (Patient -> Visite)
            md = smf.mixedlm(f"Angle_Affecte ~ Q('{var}')", data=valid_df, groups=valid_df["ID_Patient"], vc_formula={"Visite": "0 + C(ID_Visite)"})
            mdf = md.fit()
            pval = mdf.pvalues[f"Q('{var}')"]
            coef = mdf.params[f"Q('{var}')"]
            model_used = "LMM"

        except Exception:
            try:
                # Fallback sur GEE classique si LMM ne converge pas
                md = smf.gee(f"Angle_Affecte ~ Q('{var}')", groups="ID_Patient", data=valid_df, family=sm.families.Gaussian(), cov_struct=Exchangeable())
                mdf = md.fit()
                pval = mdf.pvalues[f"Q('{var}')"]
                coef = mdf.params[f"Q('{var}')"]
                model_used = "GEE (Fallback)"
            except:
                continue

        results_corr.append({
            'Variable_Clinique': var,
            'Coefficient_Relation': coef,
            'p-value': pval,
            'Significatif': '*' if pval < 0.05 else '',
            'Modele': model_used,
            'N_Patients': n_patients,
            'N_Essais': len(valid_df)
        })

        # --- GÉNÉRATION DU PLOT SI SIGNIFICATIF ---
        if pval < 0.05:
            plt.figure(figsize=(8, 6))
            sns.regplot(data=valid_df, x=var, y='Angle_Affecte', scatter_kws={'alpha':0.6, 'color': '#1f77b4', 'edgecolor': 'k'}, line_kws={'color':'darkred', 'linewidth': 2})
            plt.title(f"Corrélation: Angle Valgus/Varus vs {var}\n{model_used} p-value = {pval:.4f} | Coef = {coef:.3f}", fontsize=12, fontweight='bold')
            plt.xlabel(f"Score Clinique : {var}", fontsize=11, fontweight='bold')
            plt.ylabel("Angle Frontal (Degrés)\n[Varus < 0 < Valgus]", fontsize=11, fontweight='bold')
            plt.axhline(0, color='gray', linestyle='--', alpha=0.5)

            safe_var_name = str(var).replace('/', '_').replace('\\', '_').replace(' ', '_')
            plot_path = os.path.join(plots_correlations_folder, f"Corr_{safe_var_name}.png")
            # .savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

df_corr = pd.DataFrame(results_corr)
if not df_corr.empty:
    df_corr = df_corr.sort_values('p-value')
    excel_export_path = os.path.join(output_results_folder, "Correlations_Angle_Frontal_vs_Clinique.xlsx")
    df_corr.to_excel(excel_export_path, index=False)
    print(f"✅ Tableau des corrélations généré avec succès.")
    print(f"🖼️ Graphiques significatifs enregistrés dans : \n   -> {plots_correlations_folder}")
else:
    print("⚠️ Aucune corrélation n'a pu être calculée.")


# ==============================================================================
# 5. GRAPHIQUE 1 : BOXPLOTS GAUCHE vs DROITE
# ==============================================================================
print("\n3. Génération de la Fenêtre (Comparaison Gauche/Droite)...")
sns.set_theme(style="whitegrid")
fig1, ax1 = plt.subplots(figsize=(12, 7))
fig1.canvas.manager.set_window_title('Comparaison : Jambe Gauche vs Jambe Droite')

sns.boxplot(data=df_long, x='Diagnostic_Detail', y='Angle_Frontal', hue='Jambe', hue_order=['Gauche', 'Droite'], palette={'Gauche': '#1f77b4', 'Droite': '#ff7f0e'}, order=ordre_diag, width=0.6, boxprops={'alpha': 0.5}, ax=ax1)
sns.stripplot(data=df_long, x='Diagnostic_Detail', y='Angle_Frontal', hue='Jambe', hue_order=['Gauche', 'Droite'], palette={'Gauche': '#1f77b4', 'Droite': '#ff7f0e'}, order=ordre_diag, dodge=True, linewidth=1, edgecolor='gray', alpha=0.8, ax=ax1)

ymin, ymax = ax1.get_ylim()
y_range = ymax - ymin

for i, diag in enumerate(ordre_diag):
    df_diag = df_long[df_long['Diagnostic_Detail'] == diag].dropna(subset=['Angle_Frontal', 'ID_Patient', 'ID_Visite']).copy()
    if df_diag['Jambe'].nunique() == 2 and df_diag['ID_Patient'].nunique() > 2:
        try:
            md = smf.mixedlm("Angle_Frontal ~ C(Jambe)", data=df_diag, groups=df_diag["ID_Patient"], vc_formula={"Visite": "0 + C(ID_Visite)"})
            mdf = md.fit()
            p_keys = [k for k in mdf.pvalues.index if 'Jambe' in k]

            if p_keys:
                p_val = mdf.pvalues[p_keys[0]]
                sig = '***' if p_val <= 0.001 else '**' if p_val <= 0.01 else '*' if p_val <= 0.05 else 'ns'
                x1, x2 = i - 0.2, i + 0.2
                y_bar = df_diag['Angle_Frontal'].max() + (y_range * 0.03)
                h = y_range * 0.02
                ax1.plot([x1, x1, x2, x2], [y_bar, y_bar+h, y_bar+h, y_bar], lw=1.5, c='black')
                ax1.text((x1+x2)*.5, y_bar+h, sig, ha='center', va='bottom', color='black', fontweight='bold', fontsize=12)
        except:
            pass # Si le modèle mixte ne converge pas pour ce sous-groupe, on saute l'affichage de l'étoile

ax1.set_ylim(ymin, ymax + y_range * 0.15)
ymin, ymax = ax1.get_ylim()

ax1.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
if ymax > 0: ax1.text(1.01, ymax / 2, 'VALGUS (+)', transform=ax1.get_yaxis_transform(), color='black', fontsize=12, fontweight='bold', ha='left', va='center', rotation=90, alpha=0.6)
if ymin < 0: ax1.text(1.01, ymin / 2, 'VARUS (-)', transform=ax1.get_yaxis_transform(), color='black', fontsize=12, fontweight='bold', ha='left', va='center', rotation=90, alpha=0.6)

ax1.set_ylabel("Varus (-)   <---   Angle Frontal (Degrés)   --->   Valgus (+)", fontsize=13, fontweight='bold')
ax1.set_title("Valgus/Varus au moment du pic de flexion sagittale DROIT", fontsize=15, fontweight='bold')
ax1.set_xlabel("Profil Clinique de l'Enfant", fontsize=13)
handles, labels = ax1.get_legend_handles_labels()
if len(handles) >= 2: ax1.legend(handles[0:2], labels[0:2], title="Jambe Évaluée", loc='upper left', bbox_to_anchor=(1.05, 1))
fig1.tight_layout(rect=[0, 0, 0.95, 1])

plot_box_path = os.path.join(plots_correlations_folder, "Comparaison_Gauche_Droite_Angle_Frontal.png")
fig1.savefig(plot_box_path, dpi=300, bbox_inches='tight')


# ==============================================================================
# 5.5 GRAPHIQUE 2 : COMPARAISON HÉMIPLÉGIQUE vs DIPLÉGIQUE (LMM)
# ==============================================================================
print("\n" + "=" * 75)
print(" 📊 COMPARAISON : HÉMIPLÉGIQUE vs DIPLÉGIQUE (LMM Patient -> Visite)")
print("=" * 75)

if df_long['Diagnostic_Global'].nunique() > 1:

    # 1. Comparaison Jambe Droite : Hémi vs Di
    df_droite = df_long[df_long['Jambe'] == 'Droite'].dropna(subset=['Angle_Frontal', 'ID_Patient', 'ID_Visite']).copy()
    try:
        md_d = smf.mixedlm("Angle_Frontal ~ C(Diagnostic_Global)", data=df_droite, groups=df_droite["ID_Patient"], vc_formula={"Visite": "0 + C(ID_Visite)"})
        res_d = md_d.fit()
        pval_d = res_d.pvalues.filter(like='Diagnostic_Global').iloc[0]
        print(f"-> Jambe DROITE (Hémi vs Di) : p-value = {pval_d:.4f} {'***' if pval_d <= 0.001 else '**' if pval_d <= 0.01 else '*' if pval_d <= 0.05 else '(ns)'}")
    except Exception as e:
        print("Impossible de calculer LMM pour Jambe Droite.")

    # 2. Comparaison Jambe Gauche : Hémi vs Di
    df_gauche = df_long[df_long['Jambe'] == 'Gauche'].dropna(subset=['Angle_Frontal', 'ID_Patient', 'ID_Visite']).copy()
    try:
        md_g = smf.mixedlm("Angle_Frontal ~ C(Diagnostic_Global)", data=df_gauche, groups=df_gauche["ID_Patient"], vc_formula={"Visite": "0 + C(ID_Visite)"})
        res_g = md_g.fit()
        pval_g = res_g.pvalues.filter(like='Diagnostic_Global').iloc[0]
        print(f"-> Jambe GAUCHE (Hémi vs Di) : p-value = {pval_g:.4f} {'***' if pval_g <= 0.001 else '**' if pval_g <= 0.01 else '*' if pval_g <= 0.05 else '(ns)'}")
    except Exception as e:
        print("Impossible de calculer LMM pour Jambe Gauche.")

    # 3. Comparaison Générale : Hémi vs Di
    df_clean = df_long.dropna(subset=['Angle_Frontal', 'ID_Patient', 'ID_Visite']).copy()
    try:
        md_gen = smf.mixedlm("Angle_Frontal ~ C(Diagnostic_Global)", data=df_clean, groups=df_clean["ID_Patient"], vc_formula={"Visite": "0 + C(ID_Visite)"})
        res_gen = md_gen.fit()
        pval_gen = res_gen.pvalues.filter(like='Diagnostic_Global').iloc[0]
        print(f"-> GÉNÉRAL (Hémi vs Di)      : p-value = {pval_gen:.4f} {'***' if pval_gen <= 0.001 else '**' if pval_gen <= 0.01 else '*' if pval_gen <= 0.05 else '(ns)'}")
    except Exception as e:
        print("Impossible de calculer LMM Général.")

    # --- GRAPHIQUE ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.canvas.manager.set_window_title('Comparaison : Hémiplégique vs Diplégique')

    ordre_jambes = ['Gauche', 'Droite']
    ordre_diag = ['Hémiplégique', 'Diplégique']

    sns.boxplot(data=df_long, x='Jambe', y='Angle_Frontal', hue='Diagnostic_Global', hue_order=ordre_diag, order=ordre_jambes, palette={'Hémiplégique': '#2ca02c', 'Diplégique': '#d62728'}, width=0.6, boxprops={'alpha': 0.5}, ax=ax2)
    sns.stripplot(data=df_long, x='Jambe', y='Angle_Frontal', hue='Diagnostic_Global', hue_order=ordre_diag, order=ordre_jambes, palette={'Hémiplégique': '#2ca02c', 'Diplégique': '#d62728'}, dodge=True, linewidth=1, edgecolor='gray', alpha=0.8, ax=ax2)

    ymin, ymax = ax2.get_ylim()
    y_range = ymax - ymin
    y_bar = df_long['Angle_Frontal'].max() + (y_range * 0.05)
    h = y_range * 0.02

    def draw_significance(ax, x1, x2, y, h, pval):
        sig = '***' if pval <= 0.001 else '**' if pval <= 0.01 else '*' if pval <= 0.05 else 'ns'
        color = 'black' if pval <= 0.05 else 'gray'
        lw = 1.5 if pval <= 0.05 else 1.0
        alpha = 1.0 if pval <= 0.05 else 0.5
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c=color, alpha=alpha)
        ax.text((x1 + x2) * .5, y + h, sig, ha='center', va='bottom', color=color, fontweight='bold', fontsize=12, alpha=alpha)

    try:
        if 'pval_g' in locals(): draw_significance(ax2, -0.2, 0.2, y_bar, h, pval_g)
        if 'pval_d' in locals(): draw_significance(ax2, 0.8, 1.2, y_bar, h, pval_d)
    except NameError: pass

    ax2.set_ylim(ymin, y_bar + (y_range * 0.15))
    ymin, ymax = ax2.get_ylim()

    ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    if ymax > 0: ax2.text(1.02, ymax / 2, 'VALGUS (+)', transform=ax2.get_yaxis_transform(), color='black', fontsize=12, fontweight='bold', ha='left', va='center', rotation=90, alpha=0.6)
    if ymin < 0: ax2.text(1.02, ymin / 2, 'VARUS (-)', transform=ax2.get_yaxis_transform(), color='black', fontsize=12, fontweight='bold', ha='left', va='center', rotation=90, alpha=0.6)

    ax2.set_ylabel("Varus (-)   <---   Angle Frontal (Degrés)   --->   Valgus (+)", fontsize=13, fontweight='bold')
    ax2.set_title("Comparaison des angles frontaux : Hémiplégiques vs Diplégiques", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Jambe Évaluée", fontsize=13)

    handles, labels = ax2.get_legend_handles_labels()
    if len(handles) >= 2: ax2.legend(handles[0:2], labels[0:2], title="Diagnostic", loc='upper left', bbox_to_anchor=(1.05, 1))

    fig2.tight_layout(rect=[0, 0, 0.92, 1])
    plot_hemi_di_path = os.path.join(plots_correlations_folder, "Comparaison_Hemi_vs_Di_Angle_Frontal.png")
    fig2.savefig(plot_hemi_di_path, dpi=300, bbox_inches='tight')

else:
    print("⚠️ Impossible de comparer Hémi vs Di : un seul groupe diagnostic est présent.")

# ==============================================================================
# 6. AFFICHAGE DES FENÊTRES
# ==============================================================================
print("✅ Affichage des graphiques (fermez les fenêtres pour terminer le script)...")
plt.show()