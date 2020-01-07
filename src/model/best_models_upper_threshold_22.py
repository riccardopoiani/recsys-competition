from skopt.space import Categorical, Integer

from src.model import new_best_models
from src.model.Interface import IContentModel, IBestModel, ICollaborativeModel
import scipy.sparse as sps


class Item_CF(ICollaborativeModel):
    """
    Item CF tuned with URM_train

     - MAP-K1 TUNING CV5 (ut 22): 0.058194±0.0006
     - MAP-K1 CV10 (ut 22): 0.0558135±0.0012
    """
    from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
    best_parameters = {'topK': 4, 'shrink': 1981, 'normalize': True, 'similarity': 'asymmetric',
                       'asymmetric_alpha': 0.023846382470785306, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCFRecommender
    recommender_name = "ItemCF"


class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM_all  (containing sub_class, price, asset, sub_class_count, item_pop with
    discretization bins 200, 200, 50, 50)

     - MAP-K1 TUNING CV5 (ut 22): 0.060109±0.0010
     - MAP-K1 CV10 (ut 22): 0.0576804±0.0010

     Further comments:
     - Worse than new_best_models.ItemCBF_CF (this has 0.0583160±0.0011)
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
    best_parameters = {'topK': 21, 'shrink': 330, 'normalize': False, 'interactions_feature_weighting': 'BM25',
                       'similarity': 'asymmetric', 'asymmetric_alpha': 1.3212918035757932,
                       'feature_weighting': 'TF-IDF'}

    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"


class RP3Beta_side_info(IBestModel):
    """
    RP3 beta with side info by using TF_IDF([URM_train, ICM_all.T])

    - MAP-K1 CV5 (ut 22): 0.059398±0.0006
    - MAP-K1 CV10 (ut 22): 0.0575176±0.0012

    The old RP3beta_side_info has MAP CV10: 0.0574857±0.0012
    """

    best_parameters = {'topK': 19, 'alpha': 0.47158012440334407,
                       'beta': 0.008497496928514538, 'normalize_similarity': False}

    @classmethod
    def get_model(cls, URM_train, ICM_train, apply_tf_idf=True):
        from course_lib.Base.IR_feature_weighting import TF_IDF

        if apply_tf_idf:
            URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        else:
            URM_train_side_info = sps.vstack([URM_train, ICM_train.T]).tocsr()

        from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
        model = RP3betaRecommender(URM_train_side_info)
        model.fit(**cls.get_best_parameters())
        return model


class NewPureSVD_side_info(IBestModel):
    """
    MAP 5CV: 0.052407±0.0006
    MAP 10CV TF IDF True: 0.0502948±0.0015
    MAP 10CV TF IDF FALSE: 0.0455457±0.0011
    """
    best_parameters = {'num_factors': 840, 'n_oversamples': 26, 'n_iter': 17, 'feature_weighting': 'TF-IDF'}

    @classmethod
    def get_model(cls, URM_train, ICM_train, apply_tf_idf=True):
        from src.model.MatrixFactorization.NewPureSVDRecommender import NewPureSVDRecommender
        from course_lib.Base.IR_feature_weighting import TF_IDF
        if apply_tf_idf:
            URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        else:
            URM_train_side_info = sps.vstack([URM_train, ICM_train.T]).tocsr()
        model = NewPureSVDRecommender(URM_train_side_info)
        model.fit(**cls.get_best_parameters())
        return model


class User_Dot_CF(ICollaborativeModel):
    """
    User Dot CF tuned with URM_train

     - MAP-K1 CV5 (ut 22): 0.053763±0.0012
     - MAP-K1 CV10 (ut 22): 0.0522300±0.0016
    """
    from src.model.KNN.UserKNNDotCFRecommender import UserKNNDotCFRecommender
    best_parameters = {'topK': 745, 'shrink': 1969, 'normalize': False, 'feature_weighting': 'BM25'}
    recommender_class = UserKNNDotCFRecommender
    recommender_name = "UserDotCF"


class FusionMergeItem_CBF_CF(IBestModel):
    """
    Fusion, i.e. bagging w/o bootstrap, merge of best models: Item_CBF_CF using ICM_all (sub_class, price, asset,
    sub_class_count and item_pop; all discretized with bins 200, 200, 50, 50)
     - MAP-K1 CV10 (lt 22): 0.0592046±0.0011

    The old fusion is: MAP-K1 CV10 (ut 22): 0.0592439±0.0011
    """
    best_parameters = {'num_models': 10, 'topK': 2946}
    recommender_name = "FusionMergeItem_CBF_CF"

    @classmethod
    def get_hyperparameters(cls):
        hyper_parameters_range = {}
        for par, value in new_best_models.ItemCBF_CF.get_best_parameters().items():
            hyper_parameters_range[par] = Categorical([value])
        hyper_parameters_range['topK'] = Integer(low=3, high=30)
        hyper_parameters_range['shrink'] = Integer(low=1000, high=2000)
        return hyper_parameters_range

    @classmethod
    def get_model(cls, URM_train, ICM_train, load_model=False, save_model=False):
        from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
        from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
        model = BaggingMergeItemSimilarityRecommender(URM_train, ItemKNNCBFCFRecommender, do_bootstrap=False,
                                                      ICM_train=ICM_train)

        try:
            if load_model:
                model = cls._load_model(model)
                return model
        except FileNotFoundError:
            print("WARNING: Cannot find model to be loaded")

        model.fit(**cls.get_best_parameters(), hyper_parameters_range=cls.get_hyperparameters())
        if save_model:
            cls._save_model(model)
        return model
