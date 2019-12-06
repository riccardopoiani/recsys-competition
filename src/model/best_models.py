from abc import ABC

from src.model.HybridRecommender.HybridMixedSimilarityRecommender import ItemHybridModelRecommender, \
    UserHybridModelRecommender
from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
from src.model.Interface import IBestModel, ICollaborativeModel, IContentModel


# ---------------- CONTENT BASED FILTERING -----------------

class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM (containing sub_class and URM_train)
     - MAP (all users): 0.0273
     - MAP (only warm): 0.03498
     - Tuned only on the users with more than 30 interactions, we obtain a best-model
     {'topK': 13, 'shrink': 1045, 'similarity': 'asymmetric', 'normalize': True,
     'asymmetric_alpha': 0.11207989960736334, 'feature_weighting': 'TF-IDF'}
     with a Map of 0.0279: it has seems to have improved a bit the score.
     Indeed, in that case, we have a MAP of 0.0273
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
    best_parameters = {'topK': 10, 'shrink': 1056, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.026039165670822324, 'feature_weighting': 'TF-IDF'}

    best_parameters_foh_30 = {'topK': 13, 'shrink': 1045, 'similarity': 'asymmetric', 'normalize': True,
                              'asymmetric_alpha': 0.11207989960736334, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCBFRecommender
    recommender_name = "ItemCBF_CF"


class ItemCBF_numerical(IContentModel):
    """
    Item CBF tuned with URM_train and ICM_numerical (containing price, asset and item popularity)
     - MAP (all users): 0.002
     - MAP (only warm): 0.00258
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'feature_weighting': 'none', 'normalize': False, 'normalize_avg_row': True,
                       'shrink': 0, 'similarity': 'euclidean', 'similarity_from_distance_mode': 'exp',
                       'topK': 1000}
    recommender_class = ItemKNNCBFRecommender
    recommender_name = "ItemCBF_Numerical"


class ItemCBF_categorical(IContentModel):
    """
    Item CBF tuned with URM_train and ICM_categorical (containing sub_class)
     - MAP (all users): 0.0041
     - MAP (only warm): 0.00528
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    recommender_class = ItemKNNCBFRecommender
    recommender_name = "ItemCBF_Categorical"


class ItemCBF_all(IContentModel):
    """
    Item CBF tuned with URM_train and ICM_all (containing sub_class, price and asset) done with euclidean similarity
     - MAP (only warm): 0.0095
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'topK': 499, 'shrink': 1018, 'similarity': 'euclidean', 'normalize': False,
                       'normalize_avg_row': True, 'similarity_from_distance_mode': 'log', 'feature_weighting': 'none'}
    recommender_class = ItemKNNCBFRecommender
    recommender_name = "ItemCBF_all"


# ---------------- COLLABORATIVE FILTERING -----------------
class ItemCF(ICollaborativeModel):
    """
    Item CF tuned with URM_train
     - MAP (all users): about 0.0262
     - MAP (only warm): 0.03357
    """
    from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    best_parameters = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                       'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCFRecommender
    recommender_name = "ItemCF"


class UserCF(ICollaborativeModel):
    """
    User CF tuned with URM_train
     - MAP (all users): about 0.019
     - MAP (only warm): 0.02531
    """
    from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender

    best_parameters = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                       'feature_weighting': 'TF-IDF'}
    recommender_class = UserKNNCFRecommender
    recommender_name = "UserCF"


class P3Alpha(ICollaborativeModel):
    """
    P3Alpha recommender tuned with URM_train
     - MAP (all users): 0.0247
     - MAP (only warm): 0.031678
    """
    from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender

    best_parameters = {'topK': 84, 'alpha': 0.6033770403001427, 'normalize_similarity': True}
    recommender_class = P3alphaRecommender
    recommender_name = "P3alpha"


class RP3Beta(ICollaborativeModel):
    """
    RP3Beta recommender tuned with URM_train
     - MAP (all users): 0.0241
     - MAP (only warm): 0.03093
    """
    from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender

    best_parameters = {'topK': 5, 'alpha': 0.37829128706576887, 'beta': 0.0, 'normalize_similarity': False}
    recommender_class = RP3betaRecommender
    recommender_name = "RP3beta"


class SLIM_BPR(ICollaborativeModel):
    """
    SLIM_BPR recommender tuned with URM_train
     - There is still need to be tuned better
     - MAP (only warm): 0.02861
    """
    from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

    best_parameters = {'topK': 10, 'epochs': 1500, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 0.6893323214774385, 'lambda_j': 3.7453749998335963e-06,
                       'learning_rate': 1e-06}
    recommender_class = SLIM_BPR_Cython
    recommender_name = "SLIM_BPR"


class PureSVD(ICollaborativeModel):
    """
    PureSVD recommender tuned with URM_train
     - MAP (all users): 0.0147
     - MAP (only warm): 0.0187
    """
    from course_lib.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
    best_parameters = {'num_factors': 376}
    recommender_class = PureSVDRecommender
    recommender_name = "PureSVD"


class IALS(ICollaborativeModel):
    """
    IALS recommender
     - MAP (only warm): 0.0297
    """

    from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
    best_parameters = {'num_factors': 500, 'regularization': 100.0, 'epochs': 50,
                       'confidence_scaling': 'linear', 'alpha': 18.879765001014476}
    recommender_class = ImplicitALSRecommender
    recommender_name = "IALS"


class MF_BPR(ICollaborativeModel):
    """
    Matrix Factorization with BPR recommender
     - MAP (only warm): 0.0202
    """
    from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
    best_parameters = {'epochs': 300, 'num_factors': 600, 'regularization': 0.01220345289273659, 'learning_rate': 0.1}
    recommender_class = MF_BPR_Recommender
    recommender_name = "MF_BPR"


# ---------------- DEMOGRAPHIC FILTERING -----------------
class CFW(IBestModel):
    """
    CFW model tuned with URM train and ICM+URM
    - Map on tuning (all users): 0.0218353
    """
    best_parameters = {'topK': 1271, 'add_zeros_quota': 0.2980393873513228, 'normalize_similarity': False}

    @classmethod
    def get_model(cls, URM_train, ICM):
        from course_lib.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
        item_cf = ItemCF.get_model(URM_train, load_model=False)
        W_sparse_CF = item_cf.W_sparse

        model = CFW_D_Similarity_Linalg(URM_train=URM_train, ICM=ICM, S_matrix_target=W_sparse_CF)
        model.fit(**cls.best_parameters)
        return model


class AdvancedTopPop(IBestModel):
    """
    Advanced Top Popular tuned with URM_train and UCM+user_profile_len
     - MAP on tuning (only cold users): 0.0087
    """
    best_parameters = {'clustering_method': 'kmodes', 'n_clusters': 45, 'init_method': 'random'}

    @classmethod
    def get_model(cls, URM_train, demographic_df, original_to_train_user_id_mapper):
        from src.model.FallbackRecommender.AdvancedTopPopular import AdvancedTopPopular
        model = AdvancedTopPopular(URM_train=URM_train, data=demographic_df,
                                   mapper_dict=original_to_train_user_id_mapper)
        model.fit(**cls.best_parameters)
        return model


class UserCBF_CF(IBestModel, ABC):
    """
    User CBF tuned with URM_train and UCM (containing age, region and URM_train)
     - MAP on tuning (only cold users): 0.0109
    """
    best_parameters = {'topK': 3285, 'shrink': 1189, 'similarity': 'cosine',
                       'normalize': False, 'feature_weighting': 'BM25'}

    @classmethod
    def get_model_warm(cls, URM_train, UCM_train):
        best_parameters_warm = {'topK': 998, 'shrink': 968, 'similarity': 'cosine', 'normalize': False,
                                'feature_weighting': 'BM25'}
        from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
        model = UserKNNCBFRecommender(URM_train=URM_train, UCM_train=UCM_train)
        model.fit(**best_parameters_warm)
        return model

    @classmethod
    def get_model_cold(cls, URM_train, UCM_train):
        from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
        model = UserKNNCBFRecommender(URM_train=URM_train, UCM_train=UCM_train)
        model.fit(**cls.get_best_parameters())
        return model


# ---------------- HYBRIDS -----------------
class WeightedAverageMixed(IBestModel):
    """
    X-VAL MAP: 035516 (improvement on the two mixed hybrid)
    """
    best_parameters = {'MIXED_ITEM': 0.014667586445465623, 'MIXED_USER': 0.0013235051989859417}

    @classmethod
    def get_model(cls, URM_train, ICM_subclass_all, ICM_all, UCM_all):
        all_models = {}
        all_models['MIXED_ITEM'] = MixedItem.get_model(URM_train=URM_train,
                                                       ICM_subclass_all=ICM_subclass_all,
                                                       ICM_all=ICM_all,
                                                       load_model=False)
        all_models['MIXED_USER'] = MixedUser.get_model(URM_train=URM_train,
                                                       UCM_all=UCM_all,
                                                       load_model=False)

        model = HybridWeightedAverageRecommender(URM_train, normalize=False)

        for model_name, model_object in all_models.items():
            model.add_fitted_model(model_name, model_object)

        model.fit(**cls.get_best_parameters())

        return all_models


class MixedUser(IBestModel):
    """
    MAP: 0.0247 (improvement on MAP 0.0243 of weighted avg. on the same models)
    X-VAL MAP: 0.0247 (vs 0.0244)
    """

    best_parameters = {'topK': 645, 'alpha1': 0.49939044012800426, 'alpha2': 0.08560351971043635}

    @classmethod
    def get_model(cls, URM_train, UCM_all, load_model=False, save_model=False):
        user_cf = UserCF.get_model(URM_train=URM_train, load_model=load_model, save_model=False)
        user_cbf = UserCBF_CF.get_model_warm(URM_train=URM_train, UCM_train=UCM_all)

        hybrid = UserHybridModelRecommender(URM_train)
        hybrid.add_similarity_matrix(user_cf.W_sparse)
        hybrid.add_similarity_matrix(user_cbf.W_sparse)

        hybrid.fit(**cls.get_best_parameters())

        return hybrid


class MixedItem(IBestModel):
    """
    Improvement from item cbf cf on warm users:
    from 0.0347 to 0.0349262

    X-valid map: 0.035246  (n_folds = 10) (beating weighted average on the same models 0.035213)
    """

    best_parameters = {'topK': 1145, 'alpha1': 0.5046913488531505, 'alpha2': 0.9370816093542697,
                       'alpha3': 0.44518202509454385}

    @classmethod
    def get_model(cls, URM_train, ICM_subclass_all, ICM_all, load_model=False):
        item_cf = ItemCF.get_model(URM_train, load_model=load_model)
        item_cbf_cf = ItemCBF_CF.get_model(URM_train=URM_train, load_model=load_model,
                                           ICM_train=ICM_subclass_all)
        item_cbf_all = ItemCBF_all.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=load_model)

        hybrid = ItemHybridModelRecommender(URM_train)
        hybrid.add_similarity_matrix(item_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_cf.W_sparse)
        hybrid.add_similarity_matrix(item_cbf_all.W_sparse)

        hybrid.fit(**cls.get_best_parameters())

        return hybrid


class UserItemKNNCBFCFDemographic(IBestModel):
    best_parameters = {'user_similarity_type': 'jaccard', 'item_similarity_type': 'asymmetric',
                       'user_feature_weighting': 'none', 'item_feature_weighting': 'TF-IDF', 'user_normalize': False,
                       'item_normalize': True, 'item_asymmetric_alpha': 0.1878964519144738, 'user_topK': 1000,
                       'user_shrink': 1702, 'item_topK': 5, 'item_shrink': 1163}

    @classmethod
    def get_model(cls, URM_train, ICM_train, UCM_train):
        from src.model.KNN.UserItemCBFCFDemographicRecommender import UserItemCBFCFDemographicRecommender
        model = UserItemCBFCFDemographicRecommender(URM_train=URM_train, UCM_train=UCM_train, ICM_train=ICM_train)
        model.fit(**cls.get_best_parameters())
        return model


class HybridWeightedAvgSubmission1(IBestModel):
    """
    Hybrid Weighted Average without TopPop Fallback
    (done for the first submission of Kaggle competition with TopPop Fallback)
     - MAP (all users): 0.0272
     - MAP (only warm): 0.03481
    """

    best_parameters = {'ITEM_CF': 0.969586046573504, 'USER_CF': 0.943330450168123,
                       'ITEM_CBF_NUM': 0.03250599212747674, 'ITEM_CBF_CAT': 0.018678076600871066,
                       'SLIM_BPR': 0.03591603993769955, 'P3ALPHA': 0.7474845972085382, 'RP3BETA': 0.1234024366177027}

    @classmethod
    def _get_all_models(cls, URM_train, ICM_numerical, ICM_categorical, load_model, save_model):
        all_models = {'ITEM_CF': ItemCF.get_model(URM_train=URM_train, load_model=load_model, save_model=save_model),
                      'USER_CF': UserCF.get_model(URM_train=URM_train, load_model=load_model, save_model=save_model),
                      'ITEM_CBF_NUM': ItemCBF_numerical.get_model(URM_train=URM_train, ICM_train=ICM_numerical,
                                                                  load_model=load_model, save_model=save_model),
                      'ITEM_CBF_CAT': ItemCBF_categorical.get_model(URM_train=URM_train, ICM_train=ICM_categorical,
                                                                    load_model=load_model, save_model=save_model),
                      'SLIM_BPR': SLIM_BPR.get_model(URM_train=URM_train, load_model=load_model, save_model=save_model),
                      'P3ALPHA': P3Alpha.get_model(URM_train=URM_train, load_model=load_model, save_model=save_model),
                      'RP3BETA': RP3Beta.get_model(URM_train=URM_train, load_model=load_model, save_model=save_model)}
        return all_models

    @classmethod
    def get_model(cls, URM_train, ICM_numerical, ICM_categorical, load_model=False, save_model=False):
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
        model = HybridWeightedAverageRecommender(URM_train=URM_train, normalize=True)
        all_models = cls._get_all_models(URM_train=URM_train, ICM_numerical=ICM_numerical,
                                         ICM_categorical=ICM_categorical, load_model=load_model, save_model=save_model)
        for model_name, model_object in all_models.items():
            model.add_fitted_model(model_name, model_object)
        model.fit(**cls.get_best_parameters())
        return model


class HybridWeightedAvgSubmission2(IBestModel):
    """
    Hybrid Weighted Average (only warm, submission on Kaggle with UserCBF as Fallback)
     - MAP:
    """
    best_parameters = {'ITEM_CBF_CF': 1.0, 'P3ALPHA': 1.0, 'IALS': 1.0, 'USER_ITEM_ALL': 1.0}

    @classmethod
    def _get_all_models(cls, URM_train, ICM_train, UCM_train, load_model, save_model):
        all_models = {'ITEM_CBF_CF': ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_train,
                                                          load_model=load_model, save_model=save_model),
                      'IALS': IALS.get_model(URM_train=URM_train, load_model=load_model, save_model=save_model),
                      'P3ALPHA': P3Alpha.get_model(URM_train=URM_train, load_model=load_model, save_model=save_model),
                      'USER_ITEM_ALL': UserItemKNNCBFCFDemographic.get_model(URM_train, ICM_train, UCM_train)}
        return all_models

    @classmethod
    def get_model(cls, URM_train, ICM_train, UCM_train, load_model=False, save_model=False):
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
        model = HybridWeightedAverageRecommender(URM_train=URM_train, normalize=True)
        all_models = cls._get_all_models(URM_train=URM_train, ICM_train=ICM_train,
                                         UCM_train=UCM_train, load_model=load_model, save_model=save_model)
        for model_name, model_object in all_models.items():
            model.add_fitted_model(model_name, model_object)
        model.fit(**cls.get_best_parameters())
        return model
