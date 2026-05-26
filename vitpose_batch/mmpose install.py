# 1. Création et activation de l'environnement "laboratoire" (Python 3.9 est le plus stable ici)
conda create -n MMPose_PhD python=3.10 -y
conda activate MMPose_PhD

# 2. Outils de compilation Windows et rustine pour le module 3D
pip install setuptools wheel
conda install -c conda-forge chumpy -y

# 3. Verrouillage des versions pour éviter les crashs (Crise NumPy 2.0)
pip install "numpy<2" "opencv-python<4.10" "opencv-contrib-python<4.10"

# 4. Installation du moteur PyTorch (Version CPU 2.1.0 ultra-stable)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 5. Outil d'installation de la fondation OpenMMLab
pip install -U openmim

# 6. Installation des briques de l'écosystème dans l'ordre strict
mim install mmengine
mim install "mmcv==2.1.0"
mim install "mmdet>=3.1.0"
mim install "mmpretrain>=1.0.0"

# 7. Installation finale de MMPose
pip install "mmpose>=1.0.0"