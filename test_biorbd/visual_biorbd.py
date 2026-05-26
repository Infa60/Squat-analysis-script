import bioviz

# Remplacez par le chemin exact de votre fichier .bioMod final
biomod_filepath = r"C:\Users\bourgema\Documents\Hamner_NoMuscles.bioMod"

# Lancer le visualiseur 3D
b = bioviz.Viz(biomod_filepath)
b.exec()