import mmdet  # Test pour voir si le détecteur est vraiment dispo
def get_vitpose_full_image(frame, vitpose_model, bbox_thr=0.2, nms_thr=0.85):
    """
    Runs ViTPose's internal detector on the whole frame.
    Modifié pour forcer la détection lors d'occlusions sévères (ex: vue sagittale).

    - bbox_thr (0.2) : Accepte de détecter une personne même si l'IA n'est sûre qu'à 20%
                       (utile quand on ne voit qu'un bout de dos ou une jambe).
    - nms_thr (0.85) : Autorise deux boîtes englobantes à se chevaucher à 85% sans être fusionnées
                       (crucial quand le soignant est collé derrière le patient).
    """
    result_generator = vitpose_model(
        frame,
        return_vis=True,
        bbox_thr=bbox_thr,  # Baisse la garde du détecteur
        nms_thr=nms_thr  # Désactive la fusion des boîtes superposées
    )
    result = next(result_generator)
    return result['visualization'][0], result['predictions'][0]
