from mmpose.apis import MMPoseInferencer
import mmdet  # Test pour voir si le détecteur est vraiment dispo

def load_all_models(load_vitpose=True, load_hrnet=False, load_rtmpose=False, load_depth=False):
    print("Chargement des modèles en VRAM...")
    models = {}

    if load_vitpose:
        print(" -> Chargement ViTPose Huge + Détecteur RTMDet-X")
        # On utilise le nom EXACT trouvé par votre script de vérification
        models['vitpose_base'] = MMPoseInferencer(
            pose2d='vitpose-h',
            # det_model='rtmdet_x_8xb32-300e_coco'
            # det_model = 'rtmdet-x'

        )

    print("Modèles chargés !")
    return models