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
        hyper_parameters_range['topK'] = Integer(low=3, high=30)
        hyper_parameters_range['shrink'] = Integer(low=1000, high=2000)
        return hyper_parameters_range

    @classmethod
    def get_model(cls, URM_train, ICM_train):
        from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
        from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
        model = BaggingMergeItemSimilarityRecommender(URM_train, ItemKNNCBFCFRecommender, do_bootstrap=False,
                                                      ICM_train=ICM_train)
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


# ---------------- HYBRIDS -----------------

class MixedItem(IBestModel):
    """
    Improvement from FusionMergeItemCBF_CF from 0.0362 to 0.0364 (not so good)

    """

    best_parameters = {'topK': 1951, 'alpha1': 0.0321100284685163,
                       'alpha2': 0.13471921002086043, 'alpha3': 0.0360985576372509}

    @classmethod
    def get_model(cls, URM_train, ICM_all, load_model=False):
        from src.model.best_models import ItemCF
        from src.model.HybridRecommender.HybridMixedSimilarityRecommender import ItemHybridModelRecommender

        item_cf = ItemCF.get_model(URM_train, load_model=load_model)
        item_cbf_cf = FusionMergeItem_CBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all)
        item_cbf_all = ItemCBF_all.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=load_model)
        hybrid = ItemHybridModelRecommender(URM_train)
        hybrid.add_similarity_matrix(item_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_all.W_sparse)

        hybrid.fit(**cls.get_best_parameters())

        return hybrid


class UserItemKNNCBFCFDemographic(IBestModel):
    """
    User Item KNN CBF CF Demographic Recommender trained with URM_train, ICM_all, UCM_all with preprocessing of
        data_reader = DataPreprocessingFeatureEngineering(data_reader,
                                                          ICM_names_to_count=["ICM_sub_class"],
                                                          ICM_names_to_UCM=["ICM_sub_class", "ICM_price", "ICM_asset"],
                                                          UCM_names_to_ICM=[])
        data_reader = DataPreprocessingImputation(data_reader,
                                                  ICM_name_to_agg_mapper={"ICM_asset": np.median,
                                                                          "ICM_price": np.median})
        data_reader = DataPreprocessingTransform(data_reader,
                                                 ICM_name_to_transform_mapper={"ICM_asset": lambda x: np.log1p(1 / x),
                                                                               "ICM_price": lambda x: np.log1p(1 / x),
                                                                               "ICM_item_pop": np.log1p,
                                                                               "ICM_sub_class_count": np.log1p},
                                                 UCM_name_to_transform_mapper={"UCM_price": lambda x: np.log1p(1 / x),
                                                                               "UCM_asset": lambda x: np.log1p(1 / x)})
        data_reader = DataPreprocessingDiscretization(data_reader,
                                                      ICM_name_to_bins_mapper={"ICM_asset": 200,
                                                                               "ICM_price": 200,
                                                                               "ICM_item_pop": 50,
                                                                               "ICM_sub_class_count": 50},
                                                      UCM_name_to_bins_mapper={"UCM_price": 200,
                                                                               "UCM_asset": 200,
                                                                               "UCM_user_act": 50})
     - MAP (only warm): 0.03679
     - X3-MAP (only warm): 0.03656
     - X1-MAP (only warm): 0.0493
    """
    best_parameters = {'user_similarity_type': 'cosine', 'item_similarity_type': 'asymmetric',
                       'user_feature_weighting': 'BM25',
                       'item_feature_weighting': 'TF-IDF', 'user_normalize': True,
                       'item_normalize': True, 'item_asymmetric_alpha': 0.1539884061705812,
                       'user_topK': 16, 'user_shrink': 1000, 'item_topK': 12, 'item_shrink': 1374}

    @classmethod
    def get_model(cls, URM_train, ICM_train, UCM_train):
        from src.model.KNN.UserItemCBFCFDemographicRecommender import UserItemCBFCFDemographicRecommender
        model = UserItemCBFCFDemographicRecommender(URM_train=URM_train, UCM_train=UCM_train, ICM_train=ICM_train)
        model.fit(**cls.get_best_parameters())
        return model