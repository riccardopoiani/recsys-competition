from src.model.Interface import IContentModel


class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM_all  (containing sub_class, price, asset, sub_class_count, item_pop with
    discretization bins 200, 200, 50, 50)

     - MAP TUNING CV5 (ut 22): 0.060109±0.0010
     - MAP CV10 (ut 22): 0.0576804±0.0010

     Further comments:
     - Worse than new_best_models.ItemCBF_CF (this has 0.0583160±0.0011)
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
    best_parameters = param = {'topK': 21, 'shrink': 330, 'normalize': False, 'interactions_feature_weighting': 'BM25',
                               'similarity': 'asymmetric', 'asymmetric_alpha': 1.3212918035757932,
                               'feature_weighting': 'TF-IDF'}

    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"

