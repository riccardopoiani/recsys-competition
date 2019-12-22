from src.model.Interface import ICollaborativeModel, IContentModel


# ----- ABBREVIATIONS ------ #
# K1: keep-1-out
# CV: cross validation

class ItemCBF_CF(IContentModel):
    """
    Item CF
     - MAP@10 K1-CV5 (only warm and target): 0.054315 ± 0.0007
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender

    best_parameters = {'topK': 17, 'shrink': 1463, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.07899555402911075, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"


class P3Alpha(ICollaborativeModel):
    """
    P3Alpha
     - MAP@10 K1-CV (only warm): 0.0463
     - MAP@10 K1-CV (only warm and target): TODO
    """
    from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender

    best_parameters = {'topK': 122, 'alpha': 0.38923114168898876, 'normalize_similarity': True}
    recommender_class = P3alphaRecommender
    recommender_name = "P3alpha"
