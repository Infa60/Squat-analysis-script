import pickle
import pandas as pd

df_test = pd.read_pickle(r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Master_Database_Patient_all.pkl")

# Affichons toutes les colonnes qui contiennent le mot "Hanche" :
colonnes_hanche = [col for col in df_test.columns if 'Hanche' in col]
print(colonnes_hanche)

# Chemin vers votre base de données maître
master_pkl_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Master_Database_Healthy.pkl"

print("Chargement du fichier en cours...\n")

try:
    # 1. Ouverture du fichier
    with open(master_pkl_path, 'rb') as f:
        master_data = pickle.load(f)

    print(f"✅ Fichier chargé avec succès !")
    print(f"Type de l'objet principal : {type(master_data)}\n")

    # 2. Exploration selon le type d'objet

    # --- CAS A : C'est un tableau Pandas ---
    if isinstance(master_data, pd.DataFrame):
        print("📊 FORMAT DETECTÉ : Pandas DataFrame")
        print("-" * 40)
        print(f"Dimensions : {master_data.shape[0]} lignes et {master_data.shape[1]} colonnes.\n")

        print("🏷️ LISTE DES COLONNES :")
        # On affiche toutes les colonnes pour que vous sachiez comment faire vos jointures
        for col in master_data.columns:
            print(f"  - {col}")

        print("\n🔍 APERÇU (3 premières lignes) :")
        print(master_data.head(3))

    # --- CAS B : C'est un Dictionnaire ---
    elif isinstance(master_data, dict):
        print("📚 FORMAT DETECTÉ : Dictionnaire (Dict)")
        print("-" * 40)
        print(f"Nombre de clés à la racine : {len(master_data.keys())}\n")

        print("🔑 LISTE DES CLÉS ET TYPE DE DONNÉES :")
        # On n'affiche que les 20 premières clés pour ne pas polluer l'écran
        for i, (key, value) in enumerate(master_data.items()):
            if i >= 20:
                print("  ... et d'autres clés.")
                break

            # Si la valeur est un array ou un sous-dictionnaire, on donne sa taille
            if hasattr(value, 'shape'):
                info = f"shape {value.shape}"
            elif isinstance(value, dict) or isinstance(value, list):
                info = f"longueur {len(value)}"
            else:
                info = f"valeur simple"

            print(f"  - '{key}' : {type(value).__name__} ({info})")

    # --- CAS C : C'est une Liste ---
    elif isinstance(master_data, list):
        print("📑 FORMAT DETECTÉ : Liste")
        print("-" * 40)
        print(f"Nombre d'éléments : {len(master_data)}\n")
        if len(master_data) > 0:
            print(f"Type du premier élément : {type(master_data[0])}")
            print(f"Aperçu du premier élément : \n{master_data[0]}")

    # --- CAS D : Autre chose ---
    else:
        print("❓ FORMAT INCONNU OU SIMPLE")
        print("-" * 40)
        print(master_data)

except FileNotFoundError:
    print(f"❌ Erreur : Le fichier n'a pas été trouvé à ce chemin :\n{master_pkl_path}")
except Exception as e:
    print(f"❌ Une erreur s'est produite lors de l'ouverture : {e}")