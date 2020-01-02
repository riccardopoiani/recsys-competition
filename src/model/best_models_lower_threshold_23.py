from src.model.Interface import IContentModel


class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM_all  (containing sub_class, price, asset, sub_class_count, item_pop with
    discretization bins 200, 200, 50, 50)

     - MAP TUNING X-VAL (lt 23): 0.036662+-0.0021


     Further comments:
     - Twersky similarity is also very good, since it reaches 0.03654
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
    best_parameters = param = {'topK': 2, 'shrink': 81, 'normalize': False, 'interactions_feature_weighting': 'BM25',
                               'similarity': 'cosine',
                               'feature_weighting': 'TF-IDF'}

    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"
