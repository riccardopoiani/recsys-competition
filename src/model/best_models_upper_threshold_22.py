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


class ItemDotCF(ICollaborativeModel):
    """
    Item Dot CF tuned with URM_train

     - MAP-K1 TUNING CV5 (ut 22): 0.056246±0.0004
     - MAP-K1 CV10 (ut 22): TODO
    """
    from src.model.KNN.ItemKNNDotCFRecommender import ItemKNNDotCFRecommender
    best_parameters = {'topK': 14, 'shrink': 389, 'normalize': True, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNDotCFRecommender
    recommender_name = "ItemDotCF"


class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM_all  (containing sub_class, price, asset, sub_class_count, item_pop with
    discretization bins 200, 200, 50, 50)

     - MAP-K1 TUNING CV5 (ut 22): 0.060109±0.0010
     - MAP-K1 CV10 (ut 22): 0.0576804±0.0010

     Further comments:
     - Worse than new_best_models.ItemCBF_CF (this has 0.0583160±0.0011)   !!!!!

    Thus, this will have the best parameters of new_best_models
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
    ut_22_best_parameters = {'topK': 21, 'shrink': 330, 'normalize': False, 'interactions_feature_weighting': 'BM25',
                             'similarity': 'asymmetric', 'asymmetric_alpha': 1.3212918035757932,
                             'feature_weighting': 'TF-IDF'}
    best_parameters = {'topK': 17, 'shrink': 1463, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.07899555402911075, 'feature_weighting': 'TF-IDF'}

    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"


class ItemCBF_all_FW(IBestModel):
    """
    Item CBF non tuned with FW, but only took the best Item CBF

     - MAP-K1 CV10 (ut 22): TODO
    """

    best_parameters = {'topK': 5, 'shrink': 1500, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.0}
    CFW_parameters = {'topK': 1959, 'add_zeros_quota': 0.04991675690486688, 'normalize_similarity': False}

    @classmethod
    def get_model(cls, URM_train, ICM_train):
        from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
        from course_lib.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
        from src.model import best_models

        item_cf = best_models.ItemCF.get_model(URM_train=URM_train)
        cfw = CFW_D_Similarity_Linalg(URM_train=URM_train, ICM=ICM_train, S_matrix_target=item_cf.W_sparse)
        cfw.fit(**cls.CFW_parameters)

        model = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_train)
        model.fit(row_weights=cfw.D_best, **cls.get_best_parameters())
        return model


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


class User_CBF_CF(IBestModel):
    """
    User CBF CF using URM_train and UCM_train_new function
     - MAP-K1 CV5 (lt 22): 0.051373±0.0011
     - MAP-K1 CV10 (lt 22): 0.0502215±0.0013
    """
    best_parameters = {'topK': 1050, 'shrink': 151, 'normalize': True, 'interactions_feature_weighting': 'TF-IDF',
                       'similarity': 'cosine', 'feature_weighting': 'TF-IDF'}
    recommender_name = "UserCBF_CF"

    @classmethod
    def get_model(cls, URM_train, UCM_train, load_model=False, save_model=False):
        from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
        model = UserKNNCBFCFRecommender(URM_train=URM_train, UCM_train=UCM_train)

        try:
            if load_model:
                model = cls._load_model(model)
                return model
        except FileNotFoundError:
            print("WARNING: Cannot find model to be loaded")

        model.fit(**cls.get_best_parameters())
        if save_model:
            cls._save_model(model)
        return model


class FusionMergeItem_CBF_CF(IBestModel):
    """
    Fusion, i.e. bagging w/o bootstrap, merge of best models: Item_CBF_CF using ICM_all (sub_class, price, asset,
    sub_class_count and item_pop; all discretized with bins 200, 200, 50, 50)
     - MAP-K1 CV10 (lt 22): 0.0592468±0.0011

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


class WeightedAverageItemBased(IBestModel):
    """
     - MAP-K1 CV5: 0.062276±0.0005
     - MAP-K1 CV10: 0.0599453±0.0011
    """
    best_parameters = {'Fusion': 0.88, 'ItemDotCF': 1, 'ItemCBF_CF': 0.1, 'ItemCF': 0.13, 'RP3betaSide': 0.13}

    @classmethod
    def get_model(cls, URM_train, ICM_all):
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender

        fusion = FusionMergeItem_CBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)
        item_cbf_cf = ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)
        item_cf = Item_CF.get_model(URM_train=URM_train, load_model=False)
        rp3beta = RP3Beta_side_info.get_model(URM_train=URM_train, ICM_train=ICM_all, apply_tf_idf=False)
        item_dot = ItemDotCF.get_model(URM_train=URM_train, load_model=False)

        hybrid = HybridWeightedAverageRecommender(URM_train, normalize=True)
        hybrid.add_fitted_model("ItemDotCF", item_dot)
        hybrid.add_fitted_model("Fusion", fusion)
        hybrid.add_fitted_model("ItemCBF_CF", item_cbf_cf)
        hybrid.add_fitted_model("ItemCF", item_cf)
        hybrid.add_fitted_model("RP3betaSide", rp3beta)

        hybrid.fit(**cls.get_best_parameters())
        return hybrid

class WeightedAverageAll(IBestModel):
    """
         - MAP-K1 CV5: 0.062508±0.0004
         - MAP-K1 CV10: 0.0601679±0.0013
        """
    best_parameters = {'HybridAvg': 1, 'UserDotCF': 0.8, 'UserCBF_CF': 0.1, 'S_PureSVD': 0.43}

    @classmethod
    def get_model(cls, URM_train, ICM_all, UCM_all):
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender

        weighted_avg = WeightedAverageItemBased.get_model(URM_train=URM_train, ICM_all=ICM_all)
        user_dot = User_Dot_CF.get_model(URM_train=URM_train, load_model=False)
        user_cbf_cf = User_CBF_CF.get_model(URM_train=URM_train, UCM_train=UCM_all, load_model=False)
        s_pure_svd = NewPureSVD_side_info.get_model(URM_train=URM_train, ICM_train=ICM_all, apply_tf_idf=False)

        hybrid = HybridWeightedAverageRecommender(URM_train, normalize=True)
        hybrid.add_fitted_model("HybridAvg", weighted_avg)
        hybrid.add_fitted_model("UserDotCF", user_dot)
        hybrid.add_fitted_model("UserCBF_CF", user_cbf_cf)
        hybrid.add_fitted_model("S_PureSVD", s_pure_svd)

        hybrid.fit(**cls.get_best_parameters())
        return hybrid
