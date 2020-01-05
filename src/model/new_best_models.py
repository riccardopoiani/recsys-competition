from skopt.space import Categorical, Integer, Real

from src.model import best_models
from src.model.Interface import IContentModel, IBestModel, ICollaborativeModel

import scipy.sparse as sps


# ---------------- CONTENT BASED FILTERING -----------------
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


class ItemCBF_all_FW(IBestModel):
    """
    ItemCBF_all boosted with feature weighting from CFW using best_models.ItemCF similarity matrix
     - MAP (only warm): 0.0167
    """

    best_parameters = {'topK': 5, 'shrink': 1500, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.0}
    CFW_parameters = {'topK': 1959, 'add_zeros_quota': 0.04991675690486688, 'normalize_similarity': False}
    recommender_name = "ItemCBF_all_FW"

    @classmethod
    def get_model(cls, URM_train, ICM_train, load_model=False, save_model=False):
        from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
        from course_lib.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
        model = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_train)

        try:
            if load_model:
                model = cls._load_model(model)
                return model
        except FileNotFoundError as e:
            print("WARNING: Cannot find model to be loaded")

        item_cf = best_models.ItemCF.get_model(URM_train=URM_train)
        cfw = CFW_D_Similarity_Linalg(URM_train=URM_train, ICM=ICM_train, S_matrix_target=item_cf.W_sparse)
        cfw.fit(**cls.CFW_parameters)

        model.fit(row_weights=cfw.D_best, **cls.get_best_parameters())
        if save_model:
            cls._save_model(model)
        return model


# ---------------- COLLABORATIVE -----------------
class UserCF(ICollaborativeModel):
    """
    User CF tuned with URM_train but the feature weighting is applied on URM_train (not on URM_train.T like
    in UserKNNCFRecommender of the course_lib

     - MAP (only warm): 0.2688
    """
    from src.model.KNN.NewUserKNNCFRecommender import NewUserKNNCFRecommender

    best_parameters = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                       'feature_weighting': 'TF-IDF'}
    recommender_class = NewUserKNNCFRecommender
    recommender_name = "NewUserCF"


class SSLIM_BPR(ICollaborativeModel):
    """
    SSLIM BPR tuned with URM_train stacked with ICM.T

    - MAP (only warm): 0.0295
    """
    from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    best_parameters = {'topK': 13, 'epochs': 600, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 1.0491923839859983e-07, 'lambda_j': 2.8474498323850406,
                       'learning_rate': 1e-06}
    recommender_class = SLIM_BPR_Cython
    recommender_name = "SSLIM_BPR"


# ---------------- DEMOGRAPHICS -----------------
class UserCBF(IBestModel):
    """
    User CBF with UCM_all
     - MAP@10 K3 (only warm): 0.0239197
    """
    best_parameters = {'topK': 2941, 'shrink': 36, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.1808758516271842, 'feature_weighting': 'BM25',
                       'interactions_feature_weighting': 'TF-IDF'}
    @classmethod
    def get_model(cls, URM_train, UCM_train):
        from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
        model = UserKNNCBFRecommender(URM_train=URM_train, UCM_train=UCM_train)
        model.fit(**cls.get_best_parameters())
        return model


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
    User CBF tuned with URM_train and UCM_all
    - MAP (only warm): 0.0305
    - MAP LT 23: 0.0269084�0.0022
    """
    best_parameters = {'topK': 998, 'shrink': 968, 'similarity': 'cosine', 'normalize': False,
                       'feature_weighting': 'BM25', 'interactions_feature_weighting': "TF-IDF"}
    recommender_name = "UserCBF_CF_Warm"

    @classmethod
    def get_model(cls, URM_train, UCM_train, load_model=False, save_model=False):
        from src.model.KNN.UserKNNCBFCFRecommender import UserKNNCBFCFRecommender
        model = UserKNNCBFCFRecommender(URM_train=URM_train, UCM_train=UCM_train)

        try:
            if load_model:
                model = cls._load_model(model)
                return model
        except FileNotFoundError as e:
            print("WARNING: Cannot find model to be loaded")

        model.fit(**cls.get_best_parameters())
        if save_model:
            cls._save_model(model)
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
     - MAP (warm users) with weighted item features: 0.03658
    """
    best_parameters = {'num_models': 100}
    recommender_name = "FusionMergeItem_CBF_CF"

    @classmethod
    def get_hyperparameters(cls):
        hyper_parameters_range = {}
        for par, value in ItemCBF_CF.get_best_parameters().items():
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
        except FileNotFoundError as e:
            print("WARNING: Cannot find model to be loaded")

        model.fit(num_models=100, hyper_parameters_range=cls.get_hyperparameters())
        if save_model:
            cls._save_model(model)
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
    def get_model(cls, URM_train):
        from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
        from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
        model = BaggingMergeItemSimilarityRecommender(URM_train, P3alphaRecommender, do_bootstrap=False)
        model.fit(num_models=100, hyper_parameters_range=cls.get_hyperparameters())
        return model


# ---------------- HYBRIDS -----------------
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


class RP3BetaSideInfo(IBestModel):
    """
    RP3 beta with side info by using TF_IDF([URM_train, ICM_all.T])
     - MAP (only warm): 0.0344
    """
    best_parameters = {'topK': 8, 'alpha': 0.47878856384101826, 'beta': 9.816192345071759e-11,
                       'normalize_similarity': False}
    recommender_name = "RP3BetaSideInfo"

    @classmethod
    def get_model(cls, URM_train, ICM_train, load_model=False, save_model=False):
        from course_lib.Base.IR_feature_weighting import TF_IDF
        from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
        URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        model = RP3betaRecommender(URM_train_side_info)

        try:
            if load_model:
                model = cls._load_model(model)
                return model
        except FileNotFoundError as e:
            print("WARNING: Cannot find model to be loaded")

        model.fit(**cls.get_best_parameters())
        if save_model:
            cls._save_model(model)
        return model


class PureSVDSideInfo(IBestModel):
    """
    PureSVD recommender with side info by using TF_IDF([URM_train, ICM_all.T])
     - MAP (only warm): 0.0253
    """
    best_parameters = {'num_factors': 376}

    @classmethod
    def get_model(cls, URM_train, ICM_train):
        from course_lib.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
        from course_lib.Base.IR_feature_weighting import TF_IDF
        URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        model = PureSVDRecommender(URM_train_side_info)
        model.fit(**cls.get_best_parameters())
        return model


class IALSSideInfo(IBestModel):
    """
    IALS recommender with side info by using [URM_train, ICM_all.T]
     - MAP (only warm): 0.0310
    """
    best_parameters = {'num_factors': 704, 'regularization': 53.45866759523391, 'epochs': 50,
                       'confidence_scaling': 'linear', 'alpha': 10.031975039845253}

    @classmethod
    def get_model(cls, URM_train, ICM_train):
        from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
        URM_train_side_info = sps.vstack([URM_train, ICM_train.T], format="csr")
        model = ImplicitALSRecommender(URM_train_side_info)
        model.fit(**cls.get_best_parameters())
        return model


class MixedItem(IBestModel):
    """
    Improvement from FusionMergeItemCBF_CF from 0.0362 to 0.03654 (not so good)

    """

    best_parameters = {'topK': 40, 'alpha1': 0.05753775407896912, 'alpha2': 0.8806597865751026,
                       'alpha3': 0.0006963455386220786, 'alpha4': 0.018211744418817236}

    @classmethod
    def get_model(cls, URM_train, ICM_all, load_model=False, save_model=False):
        from src.model.best_models import ItemCF
        from src.model.HybridRecommender.HybridMixedSimilarityRecommender import ItemHybridModelRecommender

        item_cf = ItemCF.get_model(URM_train, load_model=load_model, save_model=save_model)
        item_cbf_cf = FusionMergeItem_CBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=load_model,
                                                       save_model=save_model)
        item_cbf_all = ItemCBF_all_FW.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=load_model,
                                                save_model=save_model)
        rp3beta = RP3BetaSideInfo.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=load_model,
                                            save_model=save_model)

        hybrid = ItemHybridModelRecommender(URM_train)
        hybrid.add_similarity_matrix(item_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_all.W_sparse)
        hybrid.add_similarity_matrix(rp3beta.W_sparse)

        hybrid.fit(**cls.get_best_parameters())

        return hybrid


class WeightedAverageItemBased(IBestModel):
    """
    Hybrid of Normalized weighted average of item based models.

     - MAP (only warm) with item feature weighted: 0.3680
     - MAP@10 K1-10CV (only warm) with ICM_all_weighted: 0.0545078±0.0009
    """
    best_parameters = {'FUSION_ITEM_CBF_CF': 0.9710063605002539, 'RP3BETA': 0.06174619037883712,
                       'ITEM_CF': 0.03242008947930531, 'ITEM_CBF': 0.05542859287050307}

    @classmethod
    def get_model(cls, URM_train, ICM_all):
        from src.model.best_models import ItemCF
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender

        item_cf = ItemCF.get_model(URM_train)
        item_cbf_cf = FusionMergeItem_CBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all)
        item_cbf_all = ItemCBF_all_FW.get_model(URM_train=URM_train, ICM_train=ICM_all)
        rp3beta = RP3BetaSideInfo.get_model(URM_train=URM_train, ICM_train=ICM_all)

        hybrid = HybridWeightedAverageRecommender(URM_train, normalize=True)
        hybrid.add_fitted_model("ITEM_CF", item_cf)
        hybrid.add_fitted_model("FUSION_ITEM_CBF_CF", item_cbf_cf)
        hybrid.add_fitted_model("ITEM_CBF", item_cbf_all)
        hybrid.add_fitted_model("RP3BETA", rp3beta)

        hybrid.fit(**cls.get_best_parameters())
        return hybrid


class BoostingFoh5(IBestModel):
    """
    MAP foh < 5: 0.0404
    """
    best_parameters = {'learning_rate': 0.1, 'gamma': 0.1, 'max_depth': 4, 'max_delta_step': 1, 'subsample': 0.5,
                       'colsample_bytree': 0.6,
                       'scale_pos_weight': 3.3298688818925513, 'objective': 'binary:logistic'}

    @classmethod
    def get_model(cls, model_path, URM_train, train_df, y_train, valid_df):
        from src.model.Ensemble.Boosting.Boosting import BoostingFixedData

        boosting = BoostingFixedData(URM_train=URM_train, X=train_df, y=y_train, df_test=valid_df,
                                     cutoff=20)
        boosting.load_model_from_file(file_path=model_path, params=cls.get_best_parameters())

        return boosting


class Boosting(IBestModel):
    """
    Map warm 0.0348
    """

    best_parameters = {'learning_rate': 0.01, 'gamma': 0.001, 'max_depth': 4, 'max_delta_step': 10, 'subsample': 0.7,
                       'colsample_bytree': 0.6, 'scale_pos_weight': 3.6583305754205218,
                       'objective': 'binary:logistic'}

    @classmethod
    def get_model(cls, model_path, URM_train, train_df, y_train, valid_df):
        from src.model.Ensemble.Boosting.Boosting import BoostingFixedData

        boosting = BoostingFixedData(URM_train=URM_train, X=train_df, y=y_train, df_test=valid_df,
                                     cutoff=20)
        boosting.load_model_from_file(file_path=model_path, params=cls.get_best_parameters())

        return boosting
