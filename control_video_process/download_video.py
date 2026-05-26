import pandas as pd
import yt_dlp
import os


def telecharger_videos(fichier_excel, dossier_sortie):
    """
    Lit un fichier Excel, extrait les liens YouTube/TikTok de la colonne 'Link'
    et télécharge les vidéos dans le dossier spécifié.
    """
    # 1. Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)
        print(f"📁 Dossier créé : {dossier_sortie}")

    # 2. Lire le fichier Excel
    try:
        df = pd.read_excel(fichier_excel)
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier Excel : {e}")
        return

    # Vérifier que la colonne 'Link' existe
    if "Link" not in df.columns:
        print("❌ Erreur : La colonne 'Link' est introuvable dans le fichier Excel.")
        print("Colonnes disponibles :", ", ".join(df.columns))
        return

    # 3. Configurer les options de téléchargement de yt-dlp
    ydl_opts = {
        # Modèle de nom de fichier : Titre_de_la_vidéo.extension
        'outtmpl': os.path.join(dossier_sortie, '%(title)s.%(ext)s'),
        'format': 'best',  # Télécharge la meilleure qualité vidéo/audio combinée
        'ignoreerrors': True,  # Passe à la vidéo suivante si l'une d'elles échoue
        'quiet': False,  # Affiche la progression dans la console
        'no_warnings': True
    }

    # 4. Parcourir les liens et télécharger
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        total_lignes = len(df)

        for index, row in df.iterrows():
            lien = str(row["Link"]).strip()

            # Ignorer les cases vides
            if lien.lower() == "nan" or not lien:
                continue

            # Vérifier si c'est un lien YouTube ou TikTok
            if "youtube.com" in lien or "youtu.be" in lien or "tiktok.com" in lien:
                print(f"\n⏳ [{index + 1}/{total_lignes}] Tentative de téléchargement : {lien}")
                try:
                    ydl.download([lien])
                except Exception as e:
                    print(f"⚠️ Échec du téléchargement pour {lien} : {e}")
            else:
                print(f"⏭️ Lien ignoré (ni YouTube ni TikTok) : {lien}")

    print(f"\n✅ Terminé ! Vérifiez le dossier '{dossier_sortie}'.")


if __name__ == "__main__":
    # --- Configuration ---
    # Remplacez par le nom exact de votre fichier Excel
    NOM_FICHIER_EXCEL = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_control\Link_video_squat.xlsx"

    # Nom du dossier où les vidéos seront sauvegardées
    NOM_DOSSIER_SORTIE = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Full_video_control\Video_squat_v2"

    telecharger_videos(NOM_FICHIER_EXCEL, NOM_DOSSIER_SORTIE)