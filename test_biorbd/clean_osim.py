import xml.etree.ElementTree as ET

# Vos chemins de fichiers
osim_filepath = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Code\DynamicJumperModel.osim"
clean_osim_filepath = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Code\DynamicJumperModel_NoMuscles.osim"

print("Lecture du fichier OpenSim...")
tree = ET.parse(osim_filepath)
root = tree.getroot()

# Chercher le modèle et supprimer le bloc 'ForceSet' (qui contient les muscles)
for model in root.findall('Model'):
    forceset = model.find('ForceSet')
    if forceset is not None:
        model.remove(forceset)
        print("Opération réussie : Bloc ForceSet (muscles) supprimé ! ✂️")

# Sauvegarder le nouveau fichier propre
tree.write(clean_osim_filepath)
print(f"Nouveau fichier sauvegardé ici : {clean_osim_filepath}")