from biobuddy import BiomechanicalModelReal

osim_filepath = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Code\DynamicJumperModel.osim"
biomod_filepath = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Code\DynamicJumperModel.bioMod"

# Read an .osim file
model = BiomechanicalModelReal().from_osim(
    filepath=osim_filepath,
    # Other optional parameters here
)

# Translate it into a .bioMod file
model.to_biomod(biomod_filepath)