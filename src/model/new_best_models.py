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


# ---------------- DEMOGRAPHICS -----------------
class UserCBF_CF_Cold(IBestModel):
    """
    User CBF_CF tuned with URM_train and UCM_all
     - MAP on tuning (only cold users): 0.0107
    """
    best_parameters = {'topK': 2973, 'shrink': 117, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.007315425738737337, 'feature_weighting': 'BM25',
                       'interactions_feature_weighting': 'TF-IDF'}

    @classmethod
    def get_model(cls, URM_train, UCM_train):
        from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
        model = UserKNNCBFCFRecommender(URM_train=URM_train, UCM_train=UCM_train)
        model.fit(**cls.get_best_parameters())
        return model


class UserCBF_CF_Warm(IBestModel):
    """
    User CBF tuned with URM_train and UCM (containing age, region and URM_train)
     - MAP (only warm): 0.0305
    """
    best_parameters = {'topK': 998, 'shrink': 968, 'similarity': 'cosine', 'normalize': False,
                       'feature_weighting': 'BM25', 'interactions_feature_weighting': "TF-IDF"}

    @classmethod
    def get_model(cls, URM_train, UCM_train):
        from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
        model = UserKNNCBFCFRecommender(URM_train=URM_train, UCM_train=UCM_train)
        model.fit(**cls.get_best_parameters())
        return model


class UserSimilarity(IBestModel):
    """
    User Similarity Recommender tuned with URM_train and UCM_all using new best models ItemCBF_CF as recommender helper
     - MAP (only cold): 0.0098
    """
    best_parameters = {'topK': 1000, 'shrink': 0, 'similarity': 'cosine', 'normalize': True,
                       'feature_weighting': 'BM25'}

    @classmethod
    def get_model(cls, URM_train, UCM_train, recommender):
        from src.model.KNN.UserSimilarityRecommender import UserSimilarityRecommender
        model = UserSimilarityRecommender(URM_train=URM_train, UCM_train=UCM_train, recommender=recommender)
        model.fit(**cls.get_best_parameters())
        return model


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

