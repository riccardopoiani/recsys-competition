from skopt.space import Categorical, Integer, Real

from src.model import best_models
from src.model.Interface import IContentModel, IBestModel


# ---------------- CONTENT BASED FILTERING -----------------

class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM_all  (containing sub_class, price, asset, sub_class_count, item_pop with
    discretization bins 200, 200, 50, 50)
     - MAP (only warm): 0.03560
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
    best_parameters = {'topK': 17, 'shrink': 1463, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.07899555402911075, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"


class ItemCBF_all(IContentModel):
    """
    Item CBF tuned with URM_train and ICM_all (containing sub_class, price, asset, sub_class_count, item_pop with
    discretization bins 200, 200, 50, 50)
     - MAP (only warm): 0.01260
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'topK': 5, 'shrink': 1500, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.0, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCBFRecommender
    recommender_name = "ItemCBF_all"


# ---------------- ENSEMBLES -----------------

class FusionMergeItem_CBF_CF(IBestModel):
    """
    Fusion, i.e. bagging w/o bootstrap, merge of best models: Item_CBF_CF using ICM_all (sub_class, price, asset,
    sub_class_count and item_pop; all discretized with bins 200, 200, 50, 50)
     - MAP (warm users): range  of [0.0359, 0.0363]
    """
    best_parameters = {'num_models': 100}

    @classmethod
    def get_hyperparameters(cls):
        hyper_parameters_range = {}
        for par, value in ItemCBF_CF.get_best_parameters().items():
            hyper_parameters_range[par] = Categorical([value])
        hyper_parameters_range['topK'] = Integer(low=3, high=50)
        hyper_parameters_range['shrink'] = Integer(low=0, high=2000)
        hyper_parameters_range['asymmetric_alpha'] = Real(low=1e-2, high=1e-1, prior="log-uniform")
        return hyper_parameters_range

    @classmethod
    def get_model(cls, URM_train, ICM_all):
        from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
        from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
        model = BaggingMergeItemSimilarityRecommender(URM_train, ItemKNNCBFCFRecommender, do_bootstrap=False,
                                                      ICM_train=ICM_all)
        model.fit(num_models=100, hyper_parameters_range=cls.get_hyperparameters())
        return model


class FusionMergeP3Alpha(IBestModel):
    """
    Fusion, i.e. bagging w/o bootstrap, merge of best models: P3Alpha
     - MAP (warm users): range  of [0.0331] (not explored so well)
    """
    best_parameters = {'num_models': 100}

    @classmethod
    def get_hyperparameters(cls):
        hyper_parameters_range = {}
        for par, value in best_models.P3Alpha.get_best_parameters().items():
            hyper_parameters_range[par] = Categorical([value])
        hyper_parameters_range['topK'] = Integer(low=20, high=150)
        hyper_parameters_range['alpha'] = Real(low=1e-2, high=1e0, prior="log-uniform")
        return hyper_parameters_range

    @classmethod
    def get_model(cls, URM_train, ICM_all):
        from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
        from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
        model = BaggingMergeItemSimilarityRecommender(URM_train, ItemKNNCBFCFRecommender, do_bootstrap=False,
                                                      ICM_train=ICM_all)
        model.fit(num_models=100, hyper_parameters_range=cls.get_hyperparameters())
        return model