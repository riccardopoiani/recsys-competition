from src.model.Interface import ICollaborativeModel


class ItemCF(ICollaborativeModel):
    """
    Item CF
     - MAP@10 K1-CV (only warm): 0.04777
     - MAP@10 K1-CV (only warm and target): TODO
    """
    from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    best_parameters = {'topK': 5, 'shrink': 1510, 'normalize': True, 'similarity': 'asymmetric',
                       'asymmetric_alpha': 0.0, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCFRecommender
    recommender_name = "ItemCF"


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
